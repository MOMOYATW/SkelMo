"""Fit a compact DINO PCA basis and cache dense per-frame patch features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


BACKBONE = "vit_base_patch14_dinov2.lvd142m"
DEFAULT_CACHE_NAME = "dense_dinov2_vitb14_pca128_fp16.npy"


def transform_images(paths, transform, device):
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors).to(device, non_blocking=True)


def patch_tokens(model, images):
    output = model.forward_features(images)
    if isinstance(output, dict):
        if "x_norm_patchtokens" in output:
            tokens = output["x_norm_patchtokens"]
        elif "x_prenorm" in output:
            tokens = output["x_prenorm"][:, 1:]
        else:
            raise ValueError(f"Unsupported DINO output: {output.keys()}")
    else:
        tokens = output[:, 1:]
    if tokens.shape[1] != 37 * 37 or tokens.shape[2] != 768:
        raise ValueError(f"Expected [B,1369,768], got {tuple(tokens.shape)}")
    return tokens


def load_records(root, split=None):
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "records").glob("*.json"))
    ]
    if split is not None:
        records = [record for record in records if record.get("split") == split]
    return records


def frame_paths(record):
    return sorted((Path(record["render_dir"]) / "v000").glob("*.webp"))


def fit_pca(model, transform, args):
    rng = random.Random(args.seed)
    candidates = []
    for record in load_records(args.root, "train"):
        for path in frame_paths(record):
            candidates.append(path)
    rng.shuffle(candidates)
    selected = candidates[: min(args.pca_frames, len(candidates))]
    samples = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        for start in tqdm(range(0, len(selected), args.batch_size), desc="PCA frames"):
            paths = selected[start : start + args.batch_size]
            images = transform_images(paths, transform, args.device)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.float16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                tokens = patch_tokens(model, images).float().cpu()
            for frame in tokens:
                indices = torch.randperm(len(frame), generator=generator)[: args.patches_per_frame]
                samples.append(frame[indices])
    matrix = torch.cat(samples).to(args.device)
    mean = matrix.mean(0)
    centered = matrix - mean
    _, singular, components = torch.pca_lowrank(
        centered, q=args.components, center=False, niter=args.pca_iterations
    )
    total_variance = centered.square().sum()
    explained = singular.square().sum() / total_variance.clamp_min(1e-12)
    args.pca_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.pca_path,
        mean=mean.cpu().numpy().astype(np.float32),
        components=components.cpu().numpy().astype(np.float32),
        singular_values=singular.cpu().numpy().astype(np.float32),
        explained_variance_ratio=np.asarray(float(explained.cpu()), dtype=np.float32),
        backbone=np.asarray(BACKBONE),
        sample_frames=np.asarray(len(selected), dtype=np.int32),
        sample_patches=np.asarray(len(matrix), dtype=np.int32),
    )
    print(
        json.dumps(
            {
                "pca_path": str(args.pca_path),
                "frames": len(selected),
                "patches": len(matrix),
                "components": args.components,
                "explained_variance_ratio": float(explained.cpu()),
            },
            sort_keys=True,
        )
    )


def extract(model, transform, args):
    pca = np.load(args.pca_path, allow_pickle=False)
    mean = torch.from_numpy(pca["mean"]).to(args.device)
    components = torch.from_numpy(pca["components"]).to(args.device)
    if components.shape != (768, args.components):
        raise ValueError(
            f"PCA shape {components.shape} does not match (768, {args.components})"
        )
    checksum = hashlib.sha256(args.pca_path.read_bytes()).hexdigest()
    records = load_records(args.root)[args.shard_index :: args.num_shards]
    output_root = args.root / "dense_dino"
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    for record in tqdm(records, desc=f"dense shard {args.shard_index}"):
        clip_id = record["clip_id"]
        output = output_root / f"{clip_id}.{args.cache_filename}"
        expected = (int(record["frame_count"]), 1369, args.components)
        if output.is_file() and not args.overwrite:
            cached = np.load(output, mmap_mode="r")
            if cached.shape == expected and cached.dtype == np.float16:
                counts["skipped"] += 1
                continue
            raise ValueError(f"Invalid existing dense cache: {output} {cached.shape}")
        paths = frame_paths(record)
        if len(paths) != expected[0]:
            counts["failed"] += 1
            print(f"failed {clip_id}: {len(paths)} frames, expected {expected[0]}", flush=True)
            continue
        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}.npy")
        result = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=expected
        )
        try:
            with torch.inference_mode():
                for start in range(0, len(paths), args.batch_size):
                    batch_paths = paths[start : start + args.batch_size]
                    images = transform_images(batch_paths, transform, args.device)
                    with torch.autocast(
                        device_type=torch.device(args.device).type,
                        dtype=torch.float16,
                        enabled=torch.device(args.device).type == "cuda",
                    ):
                        tokens = patch_tokens(model, images)
                    compact = (tokens.float() - mean) @ components
                    if not torch.isfinite(compact).all():
                        raise ValueError("non-finite projected features")
                    result[start : start + len(batch_paths)] = (
                        compact.cpu().numpy().astype(np.float16)
                    )
            result.flush()
            del result
            os.replace(temporary, output)
            counts["written"] += 1
        except Exception as error:
            counts["failed"] += 1
            del result
            temporary.unlink(missing_ok=True)
            print(f"failed {clip_id}: {error}", flush=True)
    metadata = {
        "format": "skelmo_dense_dino_pca_v1",
        "backbone": BACKBONE,
        "patch_grid": [37, 37],
        "feature_dim": args.components,
        "dtype": "float16",
        "pca_sha256": checksum,
        "visual_sequence": "null_then_motion_frames",
        "shard": args.shard_index,
        "records": len(records),
        **counts,
    }
    (output_root / f"metadata_shard_{args.shard_index}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit-pca", "extract"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pca-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--components", default=128, type=int)
    parser.add_argument("--pca-frames", default=256, type=int)
    parser.add_argument("--patches-per-frame", default=128, type=int)
    parser.add_argument("--pca-iterations", default=4, type=int)
    parser.add_argument("--seed", default=20261010, type=int)
    parser.add_argument("--cache-filename", default=DEFAULT_CACHE_NAME)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.pca_path is None:
        args.pca_path = args.root / "dense_dino" / f"pca_{args.components}.npz"
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    import timm

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval().to(args.device)
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    if args.mode == "fit-pca":
        fit_pca(model, transform, args)
    else:
        extract(model, transform, args)


if __name__ == "__main__":
    main()
