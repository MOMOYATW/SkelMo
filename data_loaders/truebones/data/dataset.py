import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from os.path import join as pjoin
import random
import math
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    remove_joints_augmentation,
    add_joint_augmentation,
    create_topology_edge_relations,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_PATH_LEN
from pathlib import Path
import re
from tqdm import tqdm
from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from scipy.fftpack import dct, idct
import json
from PIL import Image
from torchvision import transforms


GLOBAL_POSITION_MEAN = np.array([-0.00501097, 0.00403511, 0.08854984])
GLOBAL_POSITION_STD = np.array([0.15595314, 0.25545279, 0.25357881])


def absolute_joint_positions(motion):
    """Convert root-relative non-root positions without erasing root placement."""
    result = np.asarray(motion).copy()
    if result.shape[-1] < 3:
        raise ValueError(f"Motion needs XYZ channels, got shape {result.shape}")
    result[:, 1:, :3] += result[:, 0:1, :3]
    return result


def true_rest_pose(cond_dict, motion_shape):
    """Read the rig rest token and reject the old first-motion-frame shortcut."""
    if 'tpos_first_frame' not in cond_dict:
        raise KeyError("cond.npy is missing the true rest-pose field 'tpos_first_frame'")
    rest = np.asarray(cond_dict['tpos_first_frame']).copy()
    expected = tuple(motion_shape[1:])
    if rest.shape != expected:
        if rest.ndim == 2 and rest.shape[0] == expected[0] and rest.shape[1] >= expected[1]:
            rest = rest[:, :expected[1]]
        else:
            raise ValueError(
                f"True rest pose has shape {rest.shape}; expected {expected} "
                "to match the motion representation"
            )
    return rest


def prepend_rest_video_token(video_features, rest_features=None):
    """Prepend aligned REST features; use a zero only for legacy datasets."""
    video_features = np.asarray(video_features)
    if video_features.ndim < 2 or len(video_features) == 0:
        raise ValueError(f"Video features must be a non-empty sequence, got {video_features.shape}")
    if rest_features is None:
        rest = np.zeros_like(video_features[0])
    else:
        rest = np.asarray(rest_features)
        if rest.shape == (1,) + video_features.shape[1:]:
            rest = rest[0]
        if rest.shape != video_features.shape[1:]:
            raise ValueError(
                f"REST features have shape {rest.shape}; expected {video_features.shape[1:]}"
            )
        rest = rest.astype(video_features.dtype, copy=False)
    return np.concatenate([rest[None], video_features], axis=0)


def rest_cache_path(video_cache_path):
    """Return the explicit sidecar path used for an aligned REST feature."""
    path = Path(video_cache_path)
    return path.with_name(f"rest_{path.name}")


def find_rest_image(image_dir):
    """Find an aligned REST render without mixing it into motion frame sorting."""
    image_dir = Path(image_dir)
    for directory in (image_dir, image_dir.parent):
        for name in ("rest.webp", "rest.png", "rest.jpg", "rest.jpeg"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None

def get_mean_std(data):
    if len(data) > 0:
        Mean = data.mean(axis=0)
        Std = data.std(axis=0)
        Std[0, :3] = Std[0, :3].mean()

        Std[1:, :3] = Std[1:, :3].mean()

        return Mean, Std

def farthest_point_sampling_with_seeds(points, n_samples, seed_indices=None):
    n_pts = points.shape[0]
    centroids = np.zeros(n_samples, dtype=np.int32)
    distance = np.ones(n_pts) * 1e10

    count = 0
    if seed_indices is not None and len(seed_indices) > 0:
        for idx in seed_indices:
            if count >= n_samples: break
            centroids[count] = idx
            centroid = points[idx, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            distance = np.minimum(distance, dist)
            count += 1
        farthest = np.argmax(distance)
    else:
        farthest = np.random.randint(0, n_pts)

    for i in range(count, n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return centroids

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def get_motion_parents(motion):
    """Extract the parent index of each joint from the first frame."""
    joints_num = motion.shape[1]
    parents_map = np.sum(motion[0]**2, axis=2)
    parents = [-1]
    for j in range(1, joints_num):
        j_parent = np.where(parents_map[j] != 0)[0][0]
        parents.append(j_parent)
    return parents

def create_temporal_mask_for_window(window, max_len):
    """Create local motion attention with a global true-rest-pose token."""
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask

def permute_non_root_joints(arrays, parents, relations, graph_dist, rng=np.random):
    """Apply one consistent slot permutation while keeping the root at index 0."""
    joint_count = len(parents)
    if joint_count <= 2:
        return arrays, np.asarray(parents), relations, graph_dist
    order = np.concatenate(([0], rng.permutation(np.arange(1, joint_count))))
    inverse = np.empty(joint_count, dtype=np.int64)
    inverse[order] = np.arange(joint_count)
    old_parents = np.asarray(parents)
    new_parents = old_parents[order].copy()
    valid = new_parents >= 0
    new_parents[valid] = inverse[new_parents[valid]]
    permuted = []
    for value, axis in arrays:
        permuted.append(np.take(value, order, axis=axis))
    return (
        permuted,
        new_parents,
        np.asarray(relations)[np.ix_(order, order)],
        np.asarray(graph_dist)[np.ix_(order, order)],
    )


def prune_leaf_joints(arrays, parents, rng=np.random, fraction=0.15):
    """Remove a small random set of leaves and rebuild graph relations."""
    parents = np.asarray(parents)
    children = np.bincount(parents[parents >= 0], minlength=len(parents))
    leaves = np.flatnonzero((children == 0) & (np.arange(len(parents)) != 0))
    if len(leaves) == 0:
        return arrays, parents, *create_topology_edge_relations(
            parents.tolist(), max_path_len=MAX_PATH_LEN
        )
    count = min(len(leaves), max(1, int(math.ceil(len(leaves) * fraction))))
    removed = set(np.asarray(rng.choice(leaves, size=count, replace=False)).tolist())
    order = np.asarray([joint for joint in range(len(parents)) if joint not in removed])
    inverse = np.full(len(parents), -1, dtype=np.int64)
    inverse[order] = np.arange(len(order))
    new_parents = parents[order].copy()
    valid = new_parents >= 0
    new_parents[valid] = inverse[new_parents[valid]]
    values = [np.take(value, order, axis=axis) for value, axis in arrays]
    relations, graph_dist = create_topology_edge_relations(
        new_parents.tolist(), max_path_len=MAX_PATH_LEN
    )
    return values, new_parents, relations, graph_dist


def subdivide_joint_edge(motion, tpose, offsets, joint_features, video_tracks,
                         parents, rng=np.random):
    """Insert one degree-2 joint while preserving endpoint trajectories."""
    parents = np.asarray(parents).copy()
    candidates = np.arange(1, len(parents))
    if len(candidates) == 0:
        relations, graph_dist = create_topology_edge_relations(
            parents.tolist(), max_path_len=MAX_PATH_LEN
        )
        return (motion, tpose, offsets, joint_features, video_tracks,
                parents, relations, graph_dist)
    child = int(rng.choice(candidates))
    parent = int(parents[child])
    new_index = len(parents)
    new_motion = 0.5 * (motion[:, parent] + motion[:, child])
    new_tpose = 0.5 * (tpose[parent] + tpose[child])
    new_joint_feature = 0.5 * (joint_features[parent] + joint_features[child])
    new_video = 0.5 * (video_tracks[:, parent] + video_tracks[:, child])
    # The last channel records whether a joint has direct skin support.
    new_video[..., -1] = np.maximum(
        video_tracks[:, parent, -1], video_tracks[:, child, -1]
    )
    old_offset = offsets[child].copy()
    offsets = offsets.copy()
    offsets[child] = old_offset * 0.5
    offsets = np.concatenate([offsets, (old_offset * 0.5)[None]], axis=0)
    motion = np.concatenate([motion, new_motion[:, None]], axis=1)
    tpose = np.concatenate([tpose, new_tpose[None]], axis=0)
    joint_features = np.concatenate([joint_features, new_joint_feature[None]], axis=0)
    video_tracks = np.concatenate([video_tracks, new_video[:, None]], axis=1)
    parents[child] = new_index
    parents = np.concatenate([parents, np.asarray([parent])])
    relations, graph_dist = create_topology_edge_relations(
        parents.tolist(), max_path_len=MAX_PATH_LEN
    )
    return (motion, tpose, offsets, joint_features, video_tracks,
            parents, relations, graph_dist)


class MotionDataset(data.Dataset):
    def __init__(self, opt, temporal_window, balanced, video_feature_mode="rgb",
                 video_cache_filename="joint_tracks_dinov2_vitb14_fp16.npy",
                 joint_permutation_prob=0.0, joint_feature_dropout_prob=0.0,
                 video_joint_dropout_prob=0.0, leaf_prune_prob=0.0,
                 joint_subdivision_prob=0.0):
        print("in MotionDataset constructor")
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.balanced = balanced
        self.video_feature_mode = video_feature_mode
        self.video_cache_filename = video_cache_filename
        self.joint_permutation_prob = float(joint_permutation_prob)
        self.joint_feature_dropout_prob = float(joint_feature_dropout_prob)
        self.video_joint_dropout_prob = float(video_joint_dropout_prob)
        self.leaf_prune_prob = float(leaf_prune_prob)
        self.joint_subdivision_prob = float(joint_subdivision_prob)
        for name, probability in (
            ("joint_permutation_prob", self.joint_permutation_prob),
            ("joint_feature_dropout_prob", self.joint_feature_dropout_prob),
            ("video_joint_dropout_prob", self.video_joint_dropout_prob),
            ("leaf_prune_prob", self.leaf_prune_prob),
            ("joint_subdivision_prob", self.joint_subdivision_prob),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")
        if self.video_feature_mode not in {"rgb", "dense_dino", "joint_tracks"}:
            raise ValueError(f"Unsupported video feature mode: {self.video_feature_mode}")

        self.motions = []
        self.dino_features = []
        self.rest_dino_features = []
        self.cond_dict = []
        self.points_features = []
        self.points_xyzs = []
        self.points_masks_dict = []

        root_path = Path(opt.data_root)
        for npy_file in tqdm(root_path.rglob('*/processed_skeleton/global_motions/*.npy')):
            if not os.path.exists(npy_file.parent.parent.parent / f'joint_max_weight_features.npy'):
                print(f"Warning: joint_max_weight_features.npy not found in {npy_file.parent.parent.parent}")
                continue
            gt_motion = np.load(str(npy_file))

            self.motions.append(gt_motion)
            features_dir = npy_file.parent.parent.parent / f'imgs_{str(npy_file).split("/")[-1][19:-4]}'
            dino_files = [
                file for file in os.listdir(features_dir)
                if re.match(r'timestep_\d+_view_\d+\.png$', file)
            ]

            def extract_numbers(filename):
                match = re.match(r'timestep_(\d+)_view_(\d+)\.png', filename)
                if match:
                    timestep = int(match.group(1))
                    return timestep
                return 0

            dino_files.sort(key=extract_numbers)

            if self.video_feature_mode == "rgb":
                video_source = [features_dir / file for file in dino_files]
                video_length = len(video_source)
                rest_source = find_rest_image(features_dir)
            else:
                video_source = features_dir / self.video_cache_filename
                if not video_source.exists():
                    raise FileNotFoundError(
                        f"Missing {self.video_feature_mode} cache for {npy_file}: {video_source}"
                    )
                cached = np.load(video_source, mmap_mode="r")
                video_length = len(cached)
                if self.video_feature_mode == "joint_tracks" and cached.ndim != 3:
                    raise ValueError(f"Expected [T,J,D] joint tracks in {video_source}")
                if self.video_feature_mode == "dense_dino" and cached.ndim != 3:
                    raise ValueError(f"Expected [T,N,D] dense DINO tokens in {video_source}")
                rest_candidate = rest_cache_path(video_source)
                rest_source = rest_candidate if rest_candidate.is_file() else None
            self.dino_features.append(video_source)
            self.rest_dino_features.append(rest_source)

            assert video_length == len(self.motions[-1]), f"Length mismatch between video features and motion for {npy_file}"

            cond_dict = np.load(npy_file.parent.parent / 'cond.npy', allow_pickle=True).item()
            cond_dict = cond_dict["cond"]
            self.cond_dict.append(cond_dict)

            points_mask_dict = np.load(npy_file.parent.parent.parent / f'joint_max_weight_features.npy', allow_pickle=True).item()
            self.points_masks_dict.append(points_mask_dict)


        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window, self.max_motion_length)

        self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def inv_transform(self, x, y):
        mean = self.cond_dict[y['object_type']]['mean']
        std = self.cond_dict[y['object_type']]['std']
        return x * std + mean

    def augment(self, data):
        object_type = data['object_type']
        if object_type != "Dragon":
            aug_type = random.choice([0, 1, 2])
        else:
            aug_type = random.choice([0, 1])
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        if aug_type == 0: #no augmentation
            return data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains'], mean, std
        elif aug_type == 1: # remove_joints
            removal_rate = random.choice([0.1, 0.2, 0.3])
            return  remove_joints_augmentation(data, removal_rate, mean, std)
        else: #add joint
            return add_joint_augmentation(data, mean, std)

    def __len__(self):
        return len(self.motions) - self.pointer

    def fit_b_spline_control_points(self, trajectories, num_control_points=16, degree=3):
        """Fit ``num_control_points`` B-spline controls to ``(T, J, 3)`` trajectories."""
        T, J, C = trajectories.shape
        K = num_control_points

        knots = np.concatenate([
            np.zeros(degree),
            np.linspace(0, 1, K - degree + 1),
            np.ones(degree)
        ])

        t = np.linspace(0, 1, T)
        B = np.zeros((T, K))
        for k in range(K):
            c = np.zeros(K)
            c[k] = 1
            spline = BSpline(knots, c, degree)
            B[:, k] = spline(t)

        # Solve B * P = Y for all joints in one least-squares operation.
        Y = trajectories.reshape(T, -1)
        control_points_flat, _, _, _ = lstsq(B, Y)
        control_points = control_points_flat.reshape(K, J, 3)
        return control_points, knots

    def fit_dct_coefficients(self, trajectories, num_coeffs=16):
        """Return the first ``num_coeffs`` temporal DCT coefficients."""
        T, J, C = trajectories.shape
        K = num_coeffs

        full_coeffs = dct(trajectories, type=2, axis=0, norm='ortho')
        compressed_coeffs = full_coeffs[:K, :, :]

        return compressed_coeffs

    def __getitem__(self, item):
        if self.balanced:
            idx = item #self.pointer + item (handled in weighted sampler)
        else:
            idx = self.pointer + item

        motion = self.motions[idx].copy()

        cond_dict = self.cond_dict[idx]
        parents = cond_dict['parents']
        joints_graph_dist = cond_dict['joints_graph_dist']
        joint_relations = cond_dict['joint_relations']
        offsets = cond_dict['offsets']
        kinematic_chains = cond_dict['kinematic_chains']
        points_masks_dict = self.points_masks_dict[idx]

        joint_seeds = []
        for joint, mask in points_masks_dict.items():
            joint_indices = np.where(mask == 1)[0]
            if len(joint_indices) > 0:
                seed = np.random.choice(joint_indices)
                joint_seeds.append(seed)
        joint_seeds = list(set(joint_seeds))

        joints_names_embs = np.array([
            points_masks_dict[joint_name]
            if joint_name in points_masks_dict
            else points_masks_dict[joint_name.replace('_end_site', '')]
            for joint_name in cond_dict['joints_names']
        ])
        data = {
            'motion': motion,
            'length': len(motion),
            'parents': parents,
            'joints_graph_dist': joints_graph_dist,
            'joints_relations': joint_relations,
            'offsets': offsets,
            'joints_names_embs': joints_names_embs,
            'kinematic_chains': kinematic_chains,
        }

        mean = cond_dict['mean']
        std = cond_dict['std']
        motion, m_length, parents, joints_graph_dist, joints_relations, offsets, joints_names_embs, kinematic_chains = data["motion"], data["length"], data["parents"], data["joints_graph_dist"], data["joints_relations"], data["offsets"], data["joints_names_embs"], data["kinematic_chains"]
        ind = 0
        global_mean = GLOBAL_POSITION_MEAN
        global_std = GLOBAL_POSITION_STD
        motion = absolute_joint_positions(motion)
        # REST is rig geometry, not the first observed motion frame. Keeping
        # the root translation also preserves absolute camera-space position.
        rest_pose = true_rest_pose(cond_dict, motion.shape)
        if m_length < self.max_motion_length:
            tpos_first_frame = (rest_pose - global_mean) / (global_std + 1e-5)
            motion = (motion - global_mean) / (global_std + 1e-5)
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        elif m_length >= self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length].copy()
            tpos_first_frame = (rest_pose - global_mean) / (global_std + 1e-5)
            motion = (motion - global_mean) / (global_std + 1e-5)
            m_length = self.max_motion_length

        if motion.max() > 100 or motion.min() < -100:
            print(f"Warning: motion values are too large after normalization for index {idx}. Max: {motion.max()}, Min: {motion.min()}")

        video_source = self.dino_features[idx]
        rest_source = self.rest_dino_features[idx]
        if self.video_feature_mode == "rgb":
            frames = []
            for image_path in video_source[ind: ind + m_length]:
                with Image.open(image_path) as image:
                    frames.append(self.transform(image.convert('RGB')).numpy())
            video_dino = np.stack(frames, axis=0)
            if rest_source is not None:
                with Image.open(rest_source) as image:
                    rest_video_dino = self.transform(image.convert('RGB')).numpy()
            else:
                rest_video_dino = None
        else:
            cached = np.load(video_source, mmap_mode="r")
            video_dino = np.asarray(cached[ind: ind + m_length]).copy()
            rest_video_dino = (
                np.asarray(np.load(rest_source, mmap_mode="r")).copy()
                if rest_source is not None
                else None
            )

        # Visual tokens now align exactly with skeletal [REST, F0, F1, ...].
        # Legacy datasets remain readable through the explicit zero fallback.
        video_dino = prepend_rest_video_token(video_dino, rest_video_dino)

        if self.video_feature_mode == "joint_tracks":
            if video_dino.shape[1] != len(parents):
                raise ValueError(
                    f"Joint-track count {video_dino.shape[1]} does not match skeleton "
                    f"count {len(parents)} for {video_source}"
                )
            if random.random() < self.leaf_prune_prob:
                values, parents, joints_relations, joints_graph_dist = prune_leaf_joints(
                    [
                        (motion, 1),
                        (tpos_first_frame, 0),
                        (offsets, 0),
                        (joints_names_embs, 0),
                        (video_dino, 1),
                    ],
                    parents,
                )
                motion, tpos_first_frame, offsets, joints_names_embs, video_dino = values
            if (
                random.random() < self.joint_subdivision_prob
                and len(parents) < self.opt.max_joints
            ):
                (
                    motion,
                    tpos_first_frame,
                    offsets,
                    joints_names_embs,
                    video_dino,
                    parents,
                    joints_relations,
                    joints_graph_dist,
                ) = subdivide_joint_edge(
                    motion,
                    tpos_first_frame,
                    offsets,
                    joints_names_embs,
                    video_dino,
                    parents,
                )
            if random.random() < self.joint_permutation_prob:
                values, parents, joints_relations, joints_graph_dist = permute_non_root_joints(
                    [
                        (motion, 1),
                        (tpos_first_frame, 0),
                        (offsets, 0),
                        (joints_names_embs, 0),
                        (video_dino, 1),
                    ],
                    parents,
                    joints_relations,
                    joints_graph_dist,
                )
                motion, tpos_first_frame, offsets, joints_names_embs, video_dino = values
            if self.joint_feature_dropout_prob:
                drop = np.random.random(len(parents)) < self.joint_feature_dropout_prob
                drop[0] = False
                joints_names_embs = joints_names_embs.copy()
                joints_names_embs[drop] = 0
            if self.video_joint_dropout_prob:
                drop = np.random.random(len(parents)) < self.video_joint_dropout_prob
                drop[0] = False
                video_dino[:, drop] = 0

        pad_frames = self.max_motion_length + 1 - len(video_dino)
        if pad_frames > 0:
            video_dino = np.concatenate(
                [video_dino, np.zeros((pad_frames,) + video_dino.shape[1:], dtype=video_dino.dtype)],
                axis=0,
            )

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, joints_names_embs, ind, mean, std, video_dino, self.opt.max_joints, self.video_feature_mode

class TruebonesSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        num_samples = len(data_source)
        object_types = data_source.motion_dataset.cond_dict.keys()
        name_list = data_source.motion_dataset.name_list
        total_samples = len(name_list)
        weights = np.zeros(total_samples)
        object_share = 1.0/len(object_types)
        pointer = data_source.motion_dataset.pointer
        for object_type in object_types:
            object_indices = [i for i in range(num_samples) if i>=pointer and name_list[i].startswith(f'{object_type}_')]
            object_prob = object_share / len(object_indices)
            weights[object_indices] = object_prob
        super().__init__(num_samples=num_samples, weights=weights)

class Truebones(data.Dataset):
    def __init__(self, split="train", temporal_window=31, **kwargs):
        print("in TruebonesMixedJoints constructor")
        abs_base_path = f'.'
        device = None
        opt = get_opt(device, data_root=kwargs.get('data_dir') or None)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        # num_frames is a crop/window size, not a model-wide 40-frame ceiling.
        opt.max_motion_length = kwargs['num_frames']
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        print('Loading Truebones dataset')

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.motion_dataset = MotionDataset(
            self.opt,
            temporal_window,
            self.balanced,
            video_feature_mode=kwargs.get("video_feature_mode", "rgb"),
            video_cache_filename=kwargs.get(
                "video_cache_filename", "joint_tracks_dinov2_vitb14_fp16.npy"
            ),
            joint_permutation_prob=kwargs.get("joint_permutation_prob", 0.0),
            joint_feature_dropout_prob=kwargs.get("joint_feature_dropout_prob", 0.0),
            video_joint_dropout_prob=kwargs.get("video_joint_dropout_prob", 0.0),
            leaf_prune_prob=kwargs.get("leaf_prune_prob", 0.0),
            joint_subdivision_prob=kwargs.get("joint_subdivision_prob", 0.0),
        )
        assert len(self.motion_dataset) >= 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()
