from argparse import ArgumentParser
import argparse
import os
import json
import copy

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]

    # load args from model
    assert not isinstance(args, list) and not isinstance(args.model_path, list), 'Deprecated feature..'
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=8, type=int, help="Batch size during training.")
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc', choices=['trans_enc'], type=str,
                       help="SkelMo transformer architecture.")
    group.add_argument("--layers", default=4, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_fs", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--temporal_window", default=31, type=int,
                       help="temporal window size")
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--objects_subset", default='all', choices=['all', 'quadropeds' , 'flying', 'bipeds', 'millipeds', 'millipeds_snakes', 'quadropeds_clean', 'millipeds_clean', 'flying_clean', 'bipeds_clean', 'all_clean'], type=str,
                       help="Object subset.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--ml_platform_type", default='TensorboardPlatform', choices=['NoPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    group.add_argument("--weight_decay", default=1e-4, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=50, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=2_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=120, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--balanced", action='store_true',
                       help="Use balancing sampler for fairness between topologies")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--cond_path", required=True, type=str,
                       help="Path to the target skeleton's processed cond.npy file.")
    group.add_argument("--video_dir", required=True, type=str,
                       help="Directory containing a features/ subdirectory with DINOv2 NPZ files.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--object_type", default=['Target'], type=str, nargs='+',
                       help="Name used for the generated target asset.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--bvh_dir", required=True, type=str,
                       help="Path to a directory containing BVH files of the skeleton. More files improve statistical accuracy for \
                           motion denormalization.")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--tpos_bvh", default='', type=str,
                       help="A BVH file of the character's natural rest pose for meaningful rotation learning. \
                            If missing, the code selects a pose from the provided BVH files.")
    args = parser.parse_args()
    return args
