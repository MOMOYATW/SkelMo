from model.anytop import SkelMo
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model(model, state_dict):
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    assert all([k.startswith('clip_model.') or k.endswith('inv_freq') or k.startswith("visual_model.") for k in missing_keys]), f"Missing keys that do not start with 'clip_model.': {missing_keys}"

def create_model_and_diffusion_general_skeleton(args):
    model = SkelMo(**get_gmdm_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    cond_mode = 'video_frames'
    feature_len=3

    return {'njoints': njoints, 'nfeats': nfeats,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints,
            'feature_len':feature_len,  'value_emb': args.value_emb, 'root_input_feats': 3}

def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = 100
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
    )
