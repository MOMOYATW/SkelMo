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


class StaticCameraDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        num_frames,
        temporal_window,
        video_cache_filename,
        motion_representation="positions_rot6d",
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
        self.motion_representation = motion_representation
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
        self.feature_len = 9 if motion_representation == "positions_rot6d" else 3
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
        padded_motion = np.zeros(
            (self.max_motion_length, len(parents), self.feature_len), dtype=np.float32
        )
        padded_motion[:length] = motion
        motion = padded_motion

        track_path = self.root / "tracks" / f"{clip_id}.{self.video_cache_filename}"
        if not track_path.is_file():
            raise FileNotFoundError(f"Missing video track cache: {track_path}")
        tracks = np.asarray(np.load(track_path, mmap_mode="r")[start:start + length]).copy()
        if tracks.shape[:2] != (length, len(parents)):
            raise ValueError(
                f"Track shape {tracks.shape} does not match T={length}, J={len(parents)}"
            )
        tracks = prepend_null_video_token(tracks)

        relations, graph_dist = create_topology_edge_relations(
            parents.tolist(), max_path_len=MAX_PATH_LEN
        )
        offsets = rest_geometry.copy()
        valid = parents >= 0
        offsets[valid] -= rest_geometry[parents[valid]]
        joint_features = _structural_joint_features(rest_geometry, parents)

        if random.random() < self.leaf_prune_prob:
            arrays, parents, relations, graph_dist = prune_leaf_joints(
                [(motion, 1), (rest, 0), (offsets, 0), (joint_features, 0), (tracks, 1)],
                parents,
            )
            motion, rest, offsets, joint_features, tracks = arrays
        if random.random() < self.joint_permutation_prob:
            arrays, parents, relations, graph_dist = permute_non_root_joints(
                [(motion, 1), (rest, 0), (offsets, 0), (joint_features, 0), (tracks, 1)],
                parents,
                relations,
                graph_dist,
            )
            motion, rest, offsets, joint_features, tracks = arrays
        if self.joint_feature_dropout_prob:
            drop = np.random.random(len(parents)) < self.joint_feature_dropout_prob
            drop[0] = False
            joint_features[drop] = 0
        if self.video_joint_dropout_prob:
            drop = np.random.random(len(parents)) < self.video_joint_dropout_prob
            drop[0] = False
            tracks[:, drop] = 0

        padded_tracks = np.zeros(
            (self.max_motion_length + 1, len(parents), tracks.shape[-1]),
            dtype=tracks.dtype,
        )
        padded_tracks[:len(tracks)] = tracks
        mean = np.broadcast_to(self.mean, (len(parents), self.feature_len)).copy()
        std = np.broadcast_to(self.std, (len(parents), self.feature_len)).copy()
        return (
            motion,
            length,
            parents,
            rest,
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
            "joint_tracks",
        )
