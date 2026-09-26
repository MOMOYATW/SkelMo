"""Generate skeletal motion from a processed rig and driving-video features.

The sampling loop is based on OpenAI's guided-diffusion implementation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from pathlib import Path
import re


# Statistics used by the released large-scale checkpoint for global XYZ
# positions.  The checkpoint predicts positions directly, rather than the
# 13-channel root-relative representation used by the original AnyTop model.
GLOBAL_POSITION_MEAN = np.array([0.00612526, -0.00540305, 0.09123921])
GLOBAL_POSITION_STD = np.array([0.1696524, 0.25047082, 0.19756686])

def main(args = None, cond_dict = None):
    if args is None:
        args = generate_args()
    fixseed(args.seed)
    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict=np.load(args.cond_path, allow_pickle=True).item()
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    features_dir = os.path.join(args.video_dir, 'features')
    if not os.path.isdir(features_dir):
        raise FileNotFoundError(f"DINOv2 feature directory not found: {features_dir}")
    n_frames = min(
        len([file for file in os.listdir(features_dir) if file.endswith('.npz')]),
        40,
    )
    if n_frames == 0:
        raise FileNotFoundError(f"No DINOv2 NPZ features found in: {features_dir}")
    max_joints = opt.max_joints
    dist_util.setup_dist(args.device)
    object_types = args.object_type
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = len(object_types)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    try:
        state_dict = torch.load(args.model_path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()
    joints_feature_path = Path(args.cond_path).parent.parent / 'joint_averaged_features.npy'
    _, model_kwargs = create_condition(object_types, cond_dict, n_frames, args.temporal_window, joints_feature_path, max_joints=opt.max_joints, feature_len=opt.feature_len, video_feature_path=args.video_dir)

    cond_dict = cond_dict["cond"]
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, max_joints, model.feature_len, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            guidance_scale=2.0
        )


        bs, max_joints, n_feats, n_frames = sample.shape
        for i, motion in enumerate(sample):
            n_joints = model_kwargs['y']["n_joints"][i].item()
            motion = motion[:n_joints]
            parents = model_kwargs['y']["parents"][i]
            motion = motion.cpu().permute(2, 0, 1).numpy()
            motion = motion * GLOBAL_POSITION_STD + GLOBAL_POSITION_MEAN
            offsets = cond_dict['offsets']
            global_positions = motion
            out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
            name_pref = '%s_rep_%d'%(object_types[i], rep_i)
            existing_npy_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.npy')]
            existing_mp4_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.mp4')]
            npy_name = name_pref+'_#%d.npy'%(len(existing_npy_files))
            mp4_name = name_pref+'_#%d.mp4'%(len(existing_mp4_files))
            bvh_name = name_pref+'_#%d.bvh'%(len(existing_mp4_files))
            plot_general_skeleton_3d_motion(pjoin(out_path, mp4_name), parents, global_positions, title=name_pref, fps=fps)
            np.save(pjoin(out_path, npy_name), motion)
            if out_anim is not None:
                BVH.save(
                    pjoin(out_path, bvh_name),
                    out_anim,
                    cond_dict['joints_names'],
                    frametime=1.0 / fps,
                )
            print("repetition #" + str(rep_i) + " ,created motion: "+ npy_name)

def create_condition(object_types, cond_dict, n_frames, temporal_window, joints_feature_path, max_joints, feature_len, video_feature_path):
    batches = list()
    skeleton_cond = cond_dict["cond"]
    for object_type in object_types:
        batch=list()
        parents = skeleton_cond['parents']
        n_joints = len(parents)
        mean = skeleton_cond['mean']
        std = skeleton_cond['std']
        tpos_first_frame = np.asarray(skeleton_cond['tpos_first_frame'])
        if tpos_first_frame.shape != (n_joints, feature_len):
            if tpos_first_frame.ndim != 2 or tpos_first_frame.shape[0] != n_joints or tpos_first_frame.shape[1] < feature_len:
                raise ValueError(
                    "cond.npy has an incompatible tpos_first_frame shape: "
                    f"{tpos_first_frame.shape}; expected ({n_joints}, {feature_len}) "
                    "or a wider legacy feature array."
                )
            tpos_first_frame = tpos_first_frame[:, :feature_len]
        tpos_first_frame = (
            (tpos_first_frame - GLOBAL_POSITION_MEAN)
            / (GLOBAL_POSITION_STD + 1e-5)
        )
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = skeleton_cond['joint_relations']
        joints_graph_dist = skeleton_cond['joints_graph_dist']
        offsets = skeleton_cond['offsets']
        joints_features_dict = np.load(joints_feature_path, allow_pickle=True).item()
        joints_names_embs = np.array([
            joints_features_dict[joint_name]
            if joint_name in joints_features_dict
            else joints_features_dict[joint_name.replace('_end_site', '')]
            for joint_name in skeleton_cond['joints_names']
        ])
        dino_dir = video_feature_path
        features_dir = os.path.join(dino_dir, 'features')
        npy_files = [file for file in os.listdir(features_dir) if file.endswith('.npz')]

        def extract_numbers(filename):
            match = re.match(r'timestep_(\d+)_view_(\d+)_features\.npz', filename)
            if match:
                timestep = int(match.group(1))
                return timestep
            return 0

        npy_files.sort(key=extract_numbers)

        video_dino = []
        for file in npy_files[:n_frames]:
            video_dino.append(np.load(os.path.join(features_dir, file))["features"][0])
        video_dino = np.stack(video_dino, axis=0)
        # Match the rest-pose token prepended to the skeletal sequence.
        video_dino = np.concatenate([video_dino[:1], video_dino], axis=0)
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(tpos_first_frame)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batch.append(video_dino[:n_frames + 1])
        batch.append(max_joints)
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == "__main__":
    main()
