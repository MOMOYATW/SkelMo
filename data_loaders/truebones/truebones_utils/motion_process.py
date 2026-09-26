import BVH
from Animation import *
from InverseKinematics import animation_from_positions
import numpy as np
import os
from os.path import join as pjoin
from Quaternions import Quaternions
import re
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
import random
import math
import statistics
import torch
import bisect
import re
from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN, FOOT_CONTACT_HEIGHT_THRESH, FACE_JOINTS, DATASET_DIR, MAX_PATH_LEN, ANIMATIONS_DIR, MOTION_DIR, NO_BVHS, FOOT_CONTACT_VEL_THRESH, RAW_DATA_DIR, BVHS_DIR
from utils.rotation_conversions import rotation_6d_to_matrix_np

def get_root_quat(joints, object_type, face_joint_indx=None):
    """Return the root rotation that makes the selected face joints point along +Z."""
    if face_joint_indx is None:
        face_joint_indx = FACE_JOINTS[object_type]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
    forward = np.array([[0, 1, 0]]).repeat(len(across), axis=0)
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)
    if object_type == "Anaconda":
        root_quat = Quaternions.from_euler(np.array([0, -np.pi/2, 0]), "xyz") * root_quat
    return root_quat

def put_on_ground(anim, ground_height=None):
    """Place the skeleton on the XZ ground plane."""
    if ground_height is None:
        t_pos_global_positions = positions_global(anim)
        ground_height = t_pos_global_positions.min(axis=0).min(axis=0)[1]
    new_positions = anim.positions.copy()
    new_positions[:, 0, 1] -= ground_height
    new_offsets = anim.offsets.copy()
    new_offsets[0, 1] -= ground_height
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, ground_height

def move_xz_to_origin(anim, root_pose_init_xz=None):
    """Move the first-frame root to the XZ origin."""
    if root_pose_init_xz is None:
        root_pos_init = anim.positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_pose_init_xz
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_pose_init_xz
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_pose_init_xz

def rotate_to_hml_orientation(anim, object_type, face_joints=None):
    """Rotate the initial pose to face +Z."""
    global_pos = positions_global(anim)
    qs_rot = get_root_quat(global_pos, object_type, face_joint_indx=face_joints)[0]
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim

def rotate_90_x(anim):
    angle = -np.pi / 2
    qs_rot = Quaternions.from_angle_axis(angle, np.array([1, 0, 0]))

    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot * new_rots[:, 0]

    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot * new_pos[:, 0]

    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim

def scale(anim, scale_factor=None):
    """Scale a skeleton to the reference average bone length."""
    if scale_factor is None:
        lengths = offset_lengths(anim)
        mean_len = statistics.mean(lengths)
        scale_factor = HML_AVG_BONELEN/mean_len
    new_anim = Animation(anim.rotations.copy(), anim.positions * scale_factor ,anim.orients.copy(), anim.offsets * scale_factor,
                         anim.parents.copy())
    return new_anim, scale_factor

def get_foot_contact(positions, foot_joints_indices, vel_thresh):
    """Detect contacts from foot height and velocity."""
    frames_num, joints_num = positions.shape[:2]
    foot_vel_x = (positions[1:,foot_joints_indices ,0] - positions[:-1,foot_joints_indices ,0]) ** 2
    foot_vel_y = (positions[1:, foot_joints_indices, 1] - positions[:-1, foot_joints_indices, 1]) **2
    foot_vel_z = (positions[1:, foot_joints_indices, 2] - positions[:-1, foot_joints_indices, 2]) **2
    total_vel = foot_vel_x + foot_vel_y + foot_vel_z
    foot_contact_vel_map = np.where(np.logical_and(total_vel <= vel_thresh, np.abs(positions[1:, foot_joints_indices,1]) <= FOOT_CONTACT_HEIGHT_THRESH), 1, 0)
    foot_cont = np.zeros((frames_num-1, joints_num))
    foot_cont[:, foot_joints_indices] = foot_contact_vel_map.astype(int)

    return foot_cont

def get_6d_rep(qs):
    """Convert quaternions to the continuous 6D rotation representation."""
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)

def process_anim(anim, root_pose_init_xz=None, scale_factor=None, ground_height=None):
    """Normalize orientation, origin, scale, and ground height."""
    rotated = rotate_90_x(anim)
    centered, root_pose_init_xz_ = move_xz_to_origin(rotated, root_pose_init_xz)
    scaled, scale_factor_ = scale(centered, scale_factor)
    grounded, ground_height_ = put_on_ground(scaled, ground_height)
    return grounded, root_pose_init_xz_, ground_height_, scale_factor_

def get_common_features_from_T_pose(t_pose_bvh):
    """Extract skeleton-wide properties from a T-pose BVH file."""
    t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pose_bvh)
    t_pose_positions = positions_global(t_pose_anim)
    t_pose_anim, _1, _2 = animation_from_positions(positions=t_pose_positions, parents=t_pose_anim.parents, offsets=t_pose_anim.offsets, iterations=150)
    ground_height=None
    scaled, root_pose_init_xz, ground_height, scale_factor = process_anim(t_pose_anim, ground_height=ground_height)
    offsets = offsets_from_positions(positions_global(scaled), scaled.parents)[0]
    suspected_foot_indices = []
    new_pose_positions = positions_global(scaled)
    for i in range(len(t_pos_names)):
        if new_pose_positions[0, i, 1] < FOOT_CONTACT_HEIGHT_THRESH:
            suspected_foot_indices.append(i)
    return root_pose_init_xz, scale_factor, ground_height, offsets, suspected_foot_indices, scaled.rotations, t_pos_names, scaled

def get_motion_features(ric_positions, rotations, foot_contact, velocity, max_joints):
    """Combine position, rotation, velocity, and contact into 13D features."""

    frames, joints = ric_positions.shape[0:2]
    if joints > max_joints:
        max_joints = joints
    pos = ric_positions[:-1]
    rot = rotations[:-1]
    vel = velocity
    foot = foot_contact.reshape(frames - 1, joints , 1)
    features= np.concatenate([pos, rot, vel, foot], axis=-1)
    return features, max_joints

def get_rifke(global_positions, root_rot):
    """Express positions in root coordinates with each frame facing +Z."""
    positions = global_positions.copy()
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = np.repeat(root_rot[:, None], positions.shape[1], axis=1) * positions
    return positions

def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    """Express animation rotations relative to a natural T-pose."""
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots

def object_policy(obj):
    """Choose the kinematic-chain traversal order for an object type."""
    if obj in ["Mousey_m", "MouseyNoFingers", "Scorpion", "Raptor2"]:
        return "l_first"
    else:
        return "h_first"

def get_bvh_cont6d_params(anim):
    """Extract per-joint 6D rotations and root velocities from an animation."""
    positions = positions_global(anim)
    quat_params = anim.rotations
    r_rot = Quaternions.id(positions.shape[0])
    cont_6d_params = get_6d_rep(quat_params)
    cont_6d_params_reordered = np.zeros_like(cont_6d_params)
    for j, p in enumerate(anim.parents[1:], 1):
        cont_6d_params_reordered[:, j] = cont_6d_params[:, p]
    cont_6d_params_reordered[:, 0] = get_6d_rep(r_rot)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = r_rot[1:] * velocity
    r_velocity = r_rot[1:] * -r_rot[:-1]
    return cont_6d_params_reordered, r_velocity, velocity, r_rot, positions

def get_hml_aligned_anim(bvh_path, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, slice_inds=None):
    """Align an animation with the dataset orientation and scale."""
    if not isinstance(bvh_path, Animation):
        raw_anim, names, frame_time = BVH.load(bvh_path)
        if slice_inds:
            raw_anim = raw_anim[slice_inds[0]:slice_inds[1]]
        print('frame time', frame_time )
        frames_num, joints_num = raw_anim.positions.shape[:2]
        squared_positions_error[bvh_path] = 0
        print("positions mismatch error for file: " + bvh_path + " is " + str(squared_positions_error[bvh_path]))

        processed_anim, _xz, _gh, _sf = process_anim(raw_anim, root_pose_init_xz, scale_factor, ground_height)
    else:
        names = list()
        processed_anim = bvh_path
        frames_num = len(processed_anim)

    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    new_anim = Animation(rots, anim_positions  , processed_anim.orients, offsets, processed_anim.parents)

    return new_anim, names

def get_motion(bvh_path, foot_contact_vel_thresh, max_joints,root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, slice_inds=None):
    """Convert a BVH animation into the model's motion representation."""
    try:
        new_anim, names = get_hml_aligned_anim(bvh_path, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, slice_inds)
        cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(new_anim)
        foot_contact = get_foot_contact(global_positions, foot_indices, foot_contact_vel_thresh)
        positions = get_rifke(global_positions, r_rot)
        local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
        features, max_joints = get_motion_features(positions, cont_6d_params, foot_contact, local_vel, max_joints)
        return features, new_anim.parents, max_joints, new_anim
    except Exception as err:
        print(err)
        return None, None, max_joints, None

def get_mean_std(data):
    """Compute feature statistics with shared scales for each feature group."""
    if len(data) > 0:
        Mean = data.mean(axis=0)
        Std = data.std(axis=0)
        Std[0, :3] = Std[0, :3].mean()
        Std[0, 3:9] = Std[0, 3:9].mean()
        Std[0, 9:12] = Std[0, 9:12].mean()

        Std[1:, :3] = Std[1:, :3].mean()
        Std[1:, 3:9] = Std[1:, 3:9].mean()
        Std[1:, 9:12] = Std[1:, 9:12].mean()
        if len(Std[:, 12][Std[:, 12]!=0]) > 0:
            Std[:, 12][Std[:, 12]!=0] = Std[:, 12][Std[:, 12]!=0].mean()
        Std[:, 12][Std[:, 12]==0] = 1.0

        return Mean, Std

def create_topology_edge_relations(parents, max_path_len = 5):
    """Compute pairwise edge types and capped graph distances."""
    edge_types = {'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5, 'ts_token_conn': 6}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation']
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            """Update edge type"""
            edge_type = edge_types['no_relation']
            if i == j: #self
                edge_type = edge_types['self']
            elif parent_j == i: #child
                ee=False
                edge_type = edge_types['child']
            elif j == parent: #parent
                edge_type = edge_types['parent']
            elif parent_j == parent: #sibling
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            """Update path length type"""

            if i == j:
                topo_rel[i, j] = 0
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i:
                topo_rel[i, j] = 1
            else: #any other
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']

    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel

""" find tpos bvh"""
def find_tpos_path(bvh_files):
    t_pos_path = None
    for f in bvh_files:
        if "tpos" in f.lower():
            t_pos_path = f
            break
    if t_pos_path is not None:
        bvh_files.remove(t_pos_path)
    else:
        for f in bvh_files:
            fnam = os.path.basename(f)
            if fnam.lower().startswith('idle') or fnam.lower().startswith('__idle'):
                t_pos_path = f
                break
    if t_pos_path is None:
        t_pos_path = bvh_files[0]
    return t_pos_path

def process_object(files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DATASET_DIR, bvhs_dir=None, t_pos_path=None):
    """Process all BVH files and return counters plus skeleton conditions."""
    object_cond = dict()
    bvh_files = [pjoin(bvhs_dir, f) for f in os.listdir(bvhs_dir) if f.lower().endswith('.bvh')]
    if len(bvh_files) == 0:
        return files_counter, frames_counter, max_joints
    if t_pos_path is None or t_pos_path == '':
        t_pos_path = find_tpos_path(bvh_files)
    else:
        # Exclude the static rest pose from motion samples.
        bvh_files.remove(t_pos_path)

    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(t_pos_path)
    t_pos_motion, parents, max_joints, new_anim = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error)
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    kinematic_chains = parents2kinchains(parents)
    object_cond['kinematic_chains'] = kinematic_chains
    all_tensors = list()

    for f in bvh_files:
        print("processing file: " + f)
        raw_anim, names, frame_time = BVH.load(f)
        anim_len = len(raw_anim)
        begin = 0
        slice_ind = anim_len
        while begin < anim_len:
            if anim_len - begin > 240:
                slice_ind = begin + 200
            else:
                slice_ind = anim_len
            motion, parents, max_joints, new_anim = get_motion(f, FOOT_CONTACT_VEL_THRESH, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, slice_inds=[begin, slice_ind])
            begin = slice_ind
            if motion is not None:
                _, file_name = os.path.split(f)
                action = file_name.split('.')[0]
                all_tensors.append(motion)
                files_counter += 1
                frames_counter += motion.shape[0]
                name = action + "_" + str(files_counter)
                np.save(pjoin(save_dir, MOTION_DIR, name + '.npy'), motion)
                BVH.save(pjoin(save_dir, BVHS_DIR, name+".bvh"), new_anim, names)
                positions = recover_from_bvh_ric_np(motion)
                fc = [[j for j in range(len(parents)) if motion[f, j , 12] != 0] for f in range(motion.shape[0])]
                plot_general_skeleton_3d_motion(pjoin(save_dir, ANIMATIONS_DIR, name+"_from_ric.mp4"), parents, positions, dataset="truebones", title="", fps=20, fc = fc)

            else:
                print(f'failed to process file: {f}, slice {begin}:{slice_ind}')
    all_tensors = np.concatenate(all_tensors, axis=0)
    mean, std = get_mean_std(all_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std

    return files_counter, frames_counter, max_joints, object_cond

def create_data_samples():
    """Create the processed motion dataset."""
    os.makedirs(pjoin(DATASET_DIR, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, BVHS_DIR), exist_ok=True)

    objects = [obj for obj in FACE_JOINTS.keys() if FACE_JOINTS[obj] != []]
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()

    for object_type in objects:
        if object_type in NO_BVHS:
            continue
        cur_counter = files_counter
        files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error)
        cond[object_type] = object_cond
        objects_counter[object_type] = files_counter - cur_counter

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(DATASET_DIR, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(DATASET_DIR, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()

    np.save(pjoin(DATASET_DIR, "cond.npy"), cond)
def recover_root_quat_and_pos_np(data):
    """Recover root rotations and positions from NumPy motion features."""
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
    r_pos[...,1] = data[..., 1]
    return r_rot_quat, r_pos

def recover_root_quat_and_pos(data):
    """Recover root rotations and positions from motion features."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_rot_quat = Quaternions(r_rot_quat)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = -r_rot_quat * r_pos

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_bvh_ric_np(data):
    """Recover global positions from root-relative coordinates."""
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
    positions = data[..., 1:, :3]
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions

def recover_from_bvh_rot_np(data, parents, offsets):
    """Recover an animation and its global positions from joint rotations."""
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:, 0])
    r_rot_cont6d = get_6d_rep(r_rot_quat)

    start_indx = 3
    end_indx = 9
    cont6d_params = data[..., 1:, start_indx:end_indx]
    cont6d_params = np.concatenate([r_rot_cont6d[:, None, :], cont6d_params], axis=-2)
    cont6d_params_hml_order = rotation_6d_to_matrix_np(cont6d_params)

    # Rotations already follow joint order; remapping through parents corrupts them.
    final_rot_matrices = cont6d_params_hml_order
    rotations = Quaternions.from_transforms(final_rot_matrices)
    rotations[:, 0] = -r_rot_quat * rotations[:, 0]

    positions = offsets[None].repeat(data.shape[0], axis=0)
    positions[:, 0] = r_pos
    anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))
    return positions_global(anim), anim
def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

def parents2kinchains(parents, policy = 'h_first'):
    chains = list()
    children_dict = {i:[] for i in range(len(parents))}
    for j,p in enumerate(parents[1: ], start=1):
        if policy == 'h_first':
            reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)
    recursion_kinchains([], 0, children_dict, chains, policy)
    return chains

def recursion_kinchains(chain, j, children_dict, chains, policy):
    children = children_dict[j]
    if len(children) == 0:
        chain.append(j)
        chains.append(chain)
    elif len(children) == 1:
        chain.append(j)
        recursion_kinchains(chain, children[0], children_dict, chains, policy)
    else:
        chain.append(j)
        if policy == 'h_first':
            main_child = max(children)
        else:
            main_child = min(children)
        for child in children:
            if child == main_child:
                recursion_kinchains(chain, child, children_dict, chains, policy)
            else:
                recursion_kinchains([j], child, children_dict, chains, policy)

def remove_joints_augmentation(data, removal_rate, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    if object_type in ['KingCobra', 'Anaconda']:
        possible_feet=[]
    removal_options = [j for j in ee if j not in possible_feet]
    remove_joints = sorted(random.sample(removal_options, math.floor(len(removal_options) * removal_rate)), reverse=True)
    motion = np.delete(motion, remove_joints, axis=1)
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)

    for rj in remove_joints:
        parents[parents > rj] -= 1
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std

def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    # Split a non-root joint that has exactly one child.
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
    add_j = random.choice(possible_joints_to_add)
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy()
    new_feats[..., 12] = j_feats[..., 12].copy()
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)

    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy()
    tpos_new_feats[12] = tpos_j_feats[12]
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy()
    mean_new_feats[12] = mean_j_feats[12]
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    std_new_feats = std[add_j].copy()

    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2

    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)

    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'
    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains, mean_augmented, std_augmented
def process_single_object_type(object_type, save_dir):
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)

    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    if object_type in NO_BVHS:
        print(f"No bvh files exist for object_type {object_type}")
        exit(1)
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir)
    cond[object_type] = object_cond
    objects_counter[object_type] = files_counter - cur_counter

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()

    np.save(pjoin(save_dir, "cond.npy"), cond)


def process_skeleton(bvh_dir, save_dir, tpos_bvh=None):
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)

    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir, bvhs_dir=bvh_dir, t_pos_path=tpos_bvh)
    cond["cond"] = object_cond

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()

    np.save(pjoin(save_dir, "cond.npy"), cond)
