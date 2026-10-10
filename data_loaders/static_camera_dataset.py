"""Dataset adapter for the static-camera SkelMo v2 record schema."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils import data

from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    permute_non_root_joints,
    prepend_null_video_token,
    prepend_rest_video_token,
    prune_leaf_joints,
)
from data_loaders.truebones.truebones_utils.motion_process import (
    create_topology_edge_relations,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_PATH_LEN


def _root_aligned_rest(rest, first_pose, parents):
    """Translate and uniformly scale numeric REST geometry to motion frame zero."""
    rest = np.asarray(rest, dtype=np.float64)
    first_pose = np.asarray(first_pose, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise ValueError(f"Expected one skeleton root, got {roots.tolist()}")
    root = int(roots[0])
    joints = np.flatnonzero(parents >= 0)
    pose_lengths = np.linalg.norm(
        first_pose[joints] - first_pose[parents[joints]], axis=-1
    )
    rest_lengths = np.linalg.norm(rest[joints] - rest[parents[joints]], axis=-1)
    valid = (pose_lengths > 1e-8) & (rest_lengths > 1e-8)
    scale = float(np.median(pose_lengths[valid] / rest_lengths[valid])) if valid.any() else 1.0
    aligned = (rest - rest[root]) * scale + first_pose[root]
    return aligned.astype(np.float32), scale


def _structural_joint_features(rest, parents, width=768):
    """Encode rest geometry/topology without relying on a rendered REST image."""
    rest = np.asarray(rest, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    root = int(np.flatnonzero(parents < 0)[0])
    relative = rest - rest[root]
    extent = max(float(np.linalg.norm(np.ptp(rest, axis=0))), 1e-6)
    relative /= extent
    directions = np.zeros_like(rest)
    lengths = np.zeros((len(rest), 1), dtype=np.float32)
    valid = parents >= 0
    bone = rest[valid] - rest[parents[valid]]
    bone_length = np.linalg.norm(bone, axis=-1, keepdims=True)
    directions[valid] = bone / np.maximum(bone_length, 1e-8)
    lengths[valid] = bone_length / extent
    depth = np.zeros((len(rest), 1), dtype=np.float32)
    for joint in range(len(rest)):
        parent = int(parents[joint])
        depth[joint] = 0.0 if parent < 0 else depth[parent] + 1.0
    depth /= max(float(depth.max()), 1.0)
    children = np.bincount(parents[valid], minlength=len(rest)).astype(np.float32)[:, None]
    children /= max(float(children.max()), 1.0)
    base = np.concatenate(
        [relative, directions, lengths, depth, children, (~valid)[:, None]], axis=-1
    )
    frequencies = np.arange(1, width // (2 * base.shape[1]) + 1, dtype=np.float32)
    phase = base[..., None] * frequencies[None, None] * np.pi
    features = np.concatenate([np.sin(phase), np.cos(phase)], axis=-1).reshape(len(rest), -1)
    if features.shape[1] < width:
        features = np.pad(features, ((0, 0), (0, width - features.shape[1])))
    return features[:, :width].astype(np.float32)


def _appearance_joint_features(root, record, joint_names):
    """Load 14-view DINO descriptors in the rig's canonical joint order."""
    asset = Path(record["asset_path"]).stem.split("_", 1)[0]
    feature_root = root / "rest_joint_features" / asset
    averaged_path = feature_root / "joint_averaged_features.npy"
    dominant_path = feature_root / "joint_max_weight_features.npy"
    if not averaged_path.is_file() or not dominant_path.is_file():
        raise FileNotFoundError(f"Missing rest joint features for {asset}: {feature_root}")
    averaged = np.load(averaged_path, allow_pickle=True).item()
    dominant = np.load(dominant_path, allow_pickle=True).item()
    features = []
    for name in np.asarray(joint_names).astype(str):
        base = name.replace("_end_site", "")
        feature = dominant.get(name, dominant.get(base))
        if feature is None:
            feature = averaged.get(name, averaged.get(base))
        if feature is None:
            raise KeyError(f"No 14-view DINO feature for joint {name!r} in {asset}")
        features.append(feature)
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (len(joint_names), 768):
        raise ValueError(f"Invalid rest feature shape for {asset}: {features.shape}")
    return features


def _conditioning_pose(motion, rest, mode):
    """Choose the numeric reference token without changing motion targets."""
    if mode == "rest":
        return np.asarray(rest).copy()
    if mode == "first_frame":
        return np.asarray(motion[0]).copy()
    raise ValueError(f"Unsupported reference condition mode: {mode}")


def _prepend_conditioning_video_token(video_features, mode):
    """Align the visual reference token with the selected numeric reference."""
    if mode == "rest":
        return prepend_null_video_token(video_features)
    if mode == "first_frame":
        return prepend_rest_video_token(video_features, video_features[0])
    raise ValueError(f"Unsupported reference condition mode: {mode}")


class StaticCameraDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        num_frames,
        temporal_window,
        video_cache_filename,
        video_feature_mode="dense_dino",
        motion_representation="positions_rot6d",
        reference_condition_mode="rest",
        joint_permutation_prob=0.0,
        joint_feature_dropout_prob=0.0,
        video_joint_dropout_prob=0.0,
        leaf_prune_prob=0.0,
        joint_subdivision_prob=0.0,
        **_,
    ):
        self.root = Path(root)
        self.max_motion_length = int(num_frames)
        self.video_cache_filename = video_cache_filename
        self.video_feature_mode = video_feature_mode
        self.motion_representation = motion_representation
        self.reference_condition_mode = reference_condition_mode
        self.joint_permutation_prob = float(joint_permutation_prob)
        self.joint_feature_dropout_prob = float(joint_feature_dropout_prob)
        self.video_joint_dropout_prob = float(video_joint_dropout_prob)
        self.leaf_prune_prob = float(leaf_prune_prob)
        if joint_subdivision_prob:
            raise ValueError(
                "Joint subdivision is disabled for global 6D rotation targets; "
                "use --joint_subdivision_prob 0."
            )
        if motion_representation not in {"positions", "positions_rot6d"}:
            raise ValueError(f"Unsupported motion representation: {motion_representation}")
        if reference_condition_mode not in {"rest", "first_frame"}:
            raise ValueError(
                f"Unsupported reference condition mode: {reference_condition_mode}"
            )
        self.feature_len = 9 if motion_representation == "positions_rot6d" else 3
        pca_path = self.root / "dense_dino" / "pca_128.npz"
        if not pca_path.is_file():
            raise FileNotFoundError(f"Missing shared video/rest DINO PCA basis: {pca_path}")
        with np.load(pca_path, allow_pickle=False) as pca:
            self.dino_pca_mean = pca["mean"].astype(np.float32)
            self.dino_pca_components = pca["components"].astype(np.float32)
        stats_path = self.root / "training_stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Missing {stats_path}; run prepare_training_stats first."
            )
        stats = json.loads(stats_path.read_text())
        self.mean = np.zeros(self.feature_len, dtype=np.float32)
        self.std = np.ones(self.feature_len, dtype=np.float32)
        self.mean[:3] = stats["position_mean"]
        self.std[:3] = np.maximum(stats["position_std"], 1e-5)
        records = []
        for path in sorted((self.root / "records").glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("split") == split:
                records.append(record)
        if not records:
            raise ValueError(f"No {split!r} records under {self.root}")
        self.records = records
        self.temporal_mask_template = create_temporal_mask_for_window(
            temporal_window, self.max_motion_length
        )
        self.opt = SimpleNamespace(max_joints=143)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        clip_id = record["clip_id"]
        with np.load(record["motion_path"], allow_pickle=False) as source:
            values = {key: source[key] for key in source.files}
        with np.load(record["rig_path"], allow_pickle=False) as source:
            rig = {key: source[key] for key in source.files}
        parents = rig["parents"].astype(np.int64)
        positions = values["joint_position_gravity_camera"].astype(np.float32)
        if self.motion_representation == "positions_rot6d":
            motion = np.concatenate(
                [positions, values["skin_rotation_gravity_camera_6d"].astype(np.float32)],
                axis=-1,
            )
        else:
            motion = positions
        length = min(len(motion), self.max_motion_length)
        start = random.randint(0, len(motion) - length) if len(motion) > length else 0
        motion = motion[start:start + length]
        rest, _ = _root_aligned_rest(
            values["rest_global_position_gravity_camera"], positions[start], parents
        )
        # Preserve camera-space geometry for topology descriptors.  The per-axis
        # training normalization below is useful for the model target, but would
        # otherwise distort bone directions and relative lengths.
        rest_geometry = rest.copy()
        if self.motion_representation == "positions_rot6d":
            identity6 = np.tile(
                np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
                (len(parents), 1),
            )
            rest = np.concatenate([rest, identity6], axis=-1)
        motion = (motion - self.mean) / self.std
        rest = (rest - self.mean) / self.std
        reference_pose = _conditioning_pose(motion, rest, self.reference_condition_mode)
        padded_motion = np.zeros(
            (self.max_motion_length, len(parents), self.feature_len), dtype=np.float32
        )
        padded_motion[:length] = motion
        motion = padded_motion

        if self.video_feature_mode == "joint_tracks":
            cache_root = self.root / "tracks"
        elif self.video_feature_mode == "dense_dino":
            cache_root = self.root / "dense_dino"
        else:
            raise ValueError(
                "StaticCameraDataset expects cached dense_dino or joint_tracks, "
                f"got {self.video_feature_mode!r}"
            )
        track_path = cache_root / f"{clip_id}.{self.video_cache_filename}"
        if not track_path.is_file():
            raise FileNotFoundError(f"Missing video feature cache: {track_path}")
        tracks = np.asarray(np.load(track_path, mmap_mode="r")[start:start + length]).copy()
        expected_middle = len(parents) if self.video_feature_mode == "joint_tracks" else 1369
        if tracks.shape[:2] != (length, expected_middle):
            raise ValueError(
                f"Video cache shape {tracks.shape} does not match "
                f"T={length}, tokens={expected_middle}"
            )
        tracks = _prepend_conditioning_video_token(
            tracks, self.reference_condition_mode
        )

        relations, graph_dist = create_topology_edge_relations(
            parents.tolist(), max_path_len=MAX_PATH_LEN
        )
        offsets = rest_geometry.copy()
        valid = parents >= 0
        offsets[valid] -= rest_geometry[parents[valid]]
        structure = _structural_joint_features(
            rest_geometry, parents, width=self.dino_pca_components.shape[1]
        )
        structure /= np.maximum(np.linalg.norm(structure, axis=-1, keepdims=True), 1e-8)
        appearance = _appearance_joint_features(
            self.root, record, rig["joint_names"]
        )
        appearance = (
            appearance - self.dino_pca_mean
        ) @ self.dino_pca_components
        appearance /= np.maximum(
            np.linalg.norm(appearance, axis=-1, keepdims=True), 1e-8
        )
        joint_features = np.concatenate([appearance, structure], axis=-1)

        if random.random() < self.leaf_prune_prob:
            joint_arrays = [
                (motion, 1), (reference_pose, 0), (offsets, 0), (joint_features, 0)
            ]
            if self.video_feature_mode == "joint_tracks":
                joint_arrays.append((tracks, 1))
            arrays, parents, relations, graph_dist = prune_leaf_joints(
                joint_arrays,
                parents,
            )
            motion, reference_pose, offsets, joint_features = arrays[:4]
            if self.video_feature_mode == "joint_tracks":
                tracks = arrays[4]
        if random.random() < self.joint_permutation_prob:
            joint_arrays = [
                (motion, 1), (reference_pose, 0), (offsets, 0), (joint_features, 0)
            ]
            if self.video_feature_mode == "joint_tracks":
                joint_arrays.append((tracks, 1))
            arrays, parents, relations, graph_dist = permute_non_root_joints(
                joint_arrays,
                parents,
                relations,
                graph_dist,
            )
            motion, reference_pose, offsets, joint_features = arrays[:4]
            if self.video_feature_mode == "joint_tracks":
                tracks = arrays[4]
        if self.joint_feature_dropout_prob:
            drop = np.random.random(len(parents)) < self.joint_feature_dropout_prob
            drop[0] = False
            joint_features[drop] = 0
        if self.video_joint_dropout_prob:
            if self.video_feature_mode == "joint_tracks":
                drop = np.random.random(len(parents)) < self.video_joint_dropout_prob
                drop[0] = False
                tracks[:, drop] = 0
            else:
                # Drop fixed image regions through the whole clip to make
                # joint-to-image correspondence robust to occlusion.
                drop = np.random.random(tracks.shape[1]) < self.video_joint_dropout_prob
                tracks[:, drop] = 0

        padded_tracks = np.zeros(
            (self.max_motion_length + 1, expected_middle, tracks.shape[-1]),
            dtype=tracks.dtype,
        )
        padded_tracks[:len(tracks)] = tracks
        mean = np.broadcast_to(self.mean, (len(parents), self.feature_len)).copy()
        std = np.broadcast_to(self.std, (len(parents), self.feature_len)).copy()
        return (
            motion,
            length,
            parents,
            reference_pose,
            offsets,
            self.temporal_mask_template,
            graph_dist,
            relations,
            joint_features,
            start,
            mean,
            std,
            padded_tracks,
            self.opt.max_joints,
            self.video_feature_mode,
        )
