import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from os.path import join as pjoin
import random
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import remove_joints_augmentation, add_joint_augmentation
from pathlib import Path
import re
from tqdm import tqdm
from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from scipy.fftpack import dct, idct
import json
from PIL import Image
from torchvision import transforms

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
    """Create a local temporal-attention mask with a global rest-pose token."""
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask

class MotionDataset(data.Dataset):
    def __init__(self, opt, temporal_window, balanced):
        print("in MotionDataset constructor")
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.balanced = balanced

        self.motions = []
        self.dino_features = []
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
            dino_files = [file for file in os.listdir(features_dir) if file.endswith('.png')]

            def extract_numbers(filename):
                match = re.match(r'timestep_(\d+)_view_(\d+)\.png', filename)
                if match:
                    timestep = int(match.group(1))
                    return timestep
                return 0

            dino_files.sort(key=extract_numbers)

            dino_features = []

            for file in dino_files:
                dino_features.append(features_dir / file)
            self.dino_features.append(dino_features)

            assert len(dino_features) == len(self.motions[-1]), f"Length mismatch between dino features and motion for {npy_file}"

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
            points_masks_dict.get(
                joint_name,
                points_masks_dict[joint_name.replace('_end_site', '')]
            )
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
        global_mean = np.array([-0.00501097,  0.00403511,  0.08854984])
        global_std = np.array([0.15595314, 0.25545279, 0.25357881])
        motion[:, 1:] += motion[:, 0:1]
        motion = motion - motion[0:1, 0:1]
        if m_length < self.max_motion_length:
            tpos_first_frame = motion[0].copy()
            tpos_first_frame = (tpos_first_frame - global_mean) / (global_std + 1e-5)
            motion = (motion - global_mean) / (global_std + 1e-5)
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        elif m_length >= self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length].copy()
            tpos_first_frame = motion[0].copy()
            tpos_first_frame = (tpos_first_frame - global_mean) / (global_std + 1e-5)
            motion = (motion - global_mean) / (global_std + 1e-5)
            m_length = self.max_motion_length

        if motion.max() > 100 or motion.min() < -100:
            print(f"Warning: motion values are too large after normalization for index {idx}. Max: {motion.max()}, Min: {motion.min()}")

        video_dino_length = len(self.dino_features[idx])
        dino_features = []
        for dino_path in self.dino_features[idx][ind: ind + m_length]:
            img = Image.open(dino_path).convert('RGB')
            img = self.transform(img)
            dino_features.append(img.numpy())
        dino_features = np.stack(dino_features, axis=0)

        if video_dino_length <= self.max_motion_length:
            video_dino = np.concatenate([dino_features[0:1], dino_features], axis=0)
            video_dino = np.concatenate([
                video_dino,
                np.zeros((self.max_motion_length - video_dino_length,
                        video_dino.shape[1],
                        video_dino.shape[2],
                        video_dino.shape[3]))
            ], axis=0)
        elif video_dino_length > self.max_motion_length:
            video_dino = np.concatenate([dino_features[0:1], dino_features], axis=0)

        ind = 0

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, joints_names_embs, ind, mean, std, video_dino, self.opt.max_joints

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
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        print('Loading Truebones dataset')

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.motion_dataset = MotionDataset(self.opt, temporal_window, self.balanced)
        assert len(self.motion_dataset) >= 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()
