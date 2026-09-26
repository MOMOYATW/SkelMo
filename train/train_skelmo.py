"""Train SkelMo on a processed video-motion dataset."""

import json
import os

from data_loaders.get_data import get_dataset_loader
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.ml_platforms import NoPlatform, TensorboardPlatform
from utils.model_util import create_model_and_diffusion_general_skeleton
from utils.parser_util import train_args


PLATFORMS = {
    cls.__name__: cls
    for cls in (NoPlatform, TensorboardPlatform)
}


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        prefix = args.model_prefix or "SkelMo"
        model_name = f"{prefix}_bs_{args.batch_size}_latentdim_{args.latent_dim}"
        save_root = os.path.join(os.getcwd(), "save")
        os.makedirs(save_root, exist_ok=True)
        existing = [name for name in os.listdir(save_root) if name.startswith(model_name)]
        if existing and not args.overwrite:
            model_name = f"{model_name}_{len(existing)}"
        args.save_dir = os.path.join(save_root, model_name)

    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError(f"Save directory already exists: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    platform = PLATFORMS[args.ml_platform_type](save_dir=args.save_dir)
    platform.report_args(args, name="Args")
    with open(os.path.join(args.save_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("Creating data loader...")
    data = get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        balanced=args.balanced,
        objects_subset=args.objects_subset,
        data_dir=args.data_dir,
    )

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())
    platform.watch_model(model)

    print("Training...")
    TrainLoop(args, platform, model, diffusion, data).run_loop()
    platform.close()


if __name__ == "__main__":
    main()
