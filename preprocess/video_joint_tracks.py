"""Precompute compact DINOv2 video-to-joint correspondence tracks.

The training loader consumes one FP16 ``[T, J, 776]`` array per rendered
sequence. Dense 37x37 patch grids never reach the training loop or disk cache.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DEFAULT_CACHE_NAME = "joint_tracks_dinov2_vitb14_fp16.npy"


def rest_cache_path(video_cache_path):
    path = Path(video_cache_path)
    return path.with_name(f"rest_{path.name}")


def find_rest_image(image_dir):
    image_dir = Path(image_dir)
    for directory in (image_dir, image_dir.parent):
        for name in ("rest.webp", "rest.png", "rest.jpg", "rest.jpeg"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def match_dino_patches(patches, joint_features, support, temperature=0.07):
    """Match ``[T,D,H,W]`` patches to ``[J,D]`` queries -> ``[T,J,D+8]``."""
    if patches.ndim != 4 or joint_features.ndim != 2:
        raise ValueError("Expected patches [T,D,H,W] and joint features [J,D].")
    if patches.shape[1] != joint_features.shape[1] or temperature <= 0:
        raise ValueError("DINO dimensions/temperature are incompatible.")
    with torch.autocast(device_type=patches.device.type, enabled=False):
        patch_tokens = F.normalize(patches.float().flatten(2).transpose(1, 2), dim=-1)
        queries = F.normalize(joint_features.float(), dim=-1)
        scores = torch.einsum("jd,tpd->tjp", queries, patch_tokens)
        weights = (scores / temperature).softmax(-1)
        appearance = F.normalize(
            torch.einsum("tjp,tpd->tjd", weights, patch_tokens), dim=-1
        )
        height, width = patches.shape[-2:]
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=patches.device) + 0.5) * 2 / height - 1,
            (torch.arange(width, device=patches.device) + 0.5) * 2 / width - 1,
            indexing="ij",
        )
        xy = torch.stack((xx, yy), -1).reshape(-1, 2)
        mean = weights @ xy
        variance = (weights @ xy.square() - mean.square()).clamp_min(0)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        entropy = entropy / max(math.log(height * width), 1.0)
        supported = support.to(patches).view(1, -1, 1).expand(len(patches), -1, -1)
        return torch.cat(
            (
                appearance,
                mean,
                variance,
                scores.max(-1).values[..., None],
                entropy,
                weights.max(-1).values[..., None],
                supported,
            ),
            -1,
        )


def frame_number(path):
    match = re.match(r"timestep_(\d+)_view_(\d+)\.png$", path.name)
    return (int(match.group(1)), int(match.group(2))) if match else (10**12, path.name)


def load_joint_queries(asset_root):
    cond_path = asset_root / "processed_skeleton" / "cond.npy"
    dominant_path = asset_root / "joint_max_weight_features.npy"
    averaged_path = asset_root / "joint_averaged_features.npy"
    cond = np.load(cond_path, allow_pickle=True).item()["cond"]
    dominant = np.load(dominant_path, allow_pickle=True).item()
    averaged = (
        np.load(averaged_path, allow_pickle=True).item()
        if averaged_path.exists()
        else {}
    )
    queries, support = [], []
    for name in cond["joints_names"]:
        base = name.replace("_end_site", "")
        if name in dominant or base in dominant:
            queries.append(dominant.get(name, dominant.get(base)))
            support.append(True)
        elif name in averaged or base in averaged:
            queries.append(averaged.get(name, averaged.get(base)))
            support.append(False)
        else:
            raise KeyError(f"No visual descriptor for joint {name!r} in {asset_root}")
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 768:
        raise ValueError(f"Expected [J,768] DINOv2 joint queries, got {queries.shape}")
    return queries, np.asarray(support, dtype=np.float32), cond["joints_names"]


def extract_patch_maps(model, images):
    output = model.forward_features(images)
    if isinstance(output, dict):
        if "x_norm_patchtokens" in output:
            patches = output["x_norm_patchtokens"]
        elif "x_prenorm" in output:
            patches = output["x_prenorm"][:, 1:]
        else:
            raise ValueError(f"Unsupported DINO feature dictionary: {output.keys()}")
    else:
        patches = output[:, 1:]
    side = int(patches.shape[1] ** 0.5)
    if side * side != patches.shape[1]:
        raise ValueError(f"Patch count is not square: {patches.shape[1]}")
    return patches.transpose(1, 2).reshape(len(images), patches.shape[-1], side, side)


def sequence_dirs(dataset_root):
    roots = []
    for feature_path in dataset_root.rglob("joint_max_weight_features.npy"):
        asset_root = feature_path.parent
        for image_dir in sorted(asset_root.glob("imgs_*")):
            if image_dir.is_dir():
                roots.append((asset_root, image_dir))
    return roots


def process_sequence(model, transform, asset_root, image_dir, args):
    output = image_dir / args.cache_filename
    rest_image = find_rest_image(image_dir)
    rest_output = rest_cache_path(output)
    write_video = True
    if output.exists() and not args.overwrite:
        cached = np.load(output, mmap_mode="r")
        if cached.ndim == 3 and cached.shape[-1] == 776:
            write_video = False
        else:
            raise ValueError(f"Existing cache has the wrong schema: {output}")
    if rest_image is not None and rest_output.exists() and not args.overwrite:
        cached_rest = np.load(rest_output, mmap_mode="r")
        if cached_rest.ndim != 2 or cached_rest.shape[-1] != 776:
            raise ValueError(f"Existing REST cache has the wrong schema: {rest_output}")
        write_rest = False
    else:
        write_rest = rest_image is not None
    if not write_video and not write_rest:
        return "skipped"
    frames = sorted(
        (path for path in image_dir.glob("*.png") if frame_number(path)[0] != 10**12),
        key=frame_number,
    )
    if not frames:
        raise FileNotFoundError(f"No rendered frames in {image_dir}")
    query_np, support_np, joint_names = load_joint_queries(asset_root)
    query = torch.from_numpy(query_np).to(args.device)
    support = torch.from_numpy(support_np).to(args.device)
    with torch.inference_mode():
        if write_video:
            temporary = output.with_suffix(".tmp.npy")
            result = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float16,
                shape=(len(frames), len(query_np), 776),
            )
            for start in range(0, len(frames), args.batch_size):
                batch_paths = frames[start:start + args.batch_size]
                tensors = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                images = torch.stack(tensors).to(args.device, non_blocking=True)
                with torch.autocast(
                    device_type=torch.device(args.device).type,
                    dtype=torch.float16,
                    enabled=torch.device(args.device).type == "cuda",
                ):
                    patches = extract_patch_maps(model, images)
                tracks = match_dino_patches(patches, query, support, args.temperature)
                if not torch.isfinite(tracks).all():
                    raise ValueError(f"Non-finite correspondence track in {image_dir}")
                result[start:start + len(batch_paths)] = tracks.cpu().numpy().astype(np.float16)
            result.flush()
            del result
            os.replace(temporary, output)
        if write_rest:
            with Image.open(rest_image) as image:
                rest_tensor = transform(image.convert("RGB"))[None].to(
                    args.device, non_blocking=True
                )
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.float16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                rest_patches = extract_patch_maps(model, rest_tensor)
            rest_tracks = match_dino_patches(
                rest_patches, query, support, args.temperature
            )[0]
            if not torch.isfinite(rest_tracks).all():
                raise ValueError(f"Non-finite REST correspondence track in {image_dir}")
            temporary_rest = rest_output.with_suffix(".tmp.npy")
            np.save(temporary_rest, rest_tracks.cpu().numpy().astype(np.float16))
            os.replace(temporary_rest, rest_output)
    metadata = {
        "format": "skelmo_joint_tracks_v1",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "frames": len(frames),
        "joints": len(joint_names),
        "feature_dim": 776,
        "dtype": "float16",
        "temperature": args.temperature,
        "joint_names": list(joint_names),
        "rest_conditioning": "aligned_render" if rest_image is not None else "legacy_zero",
        "rest_cache": rest_output.name if rest_image is not None else None,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return "written"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--cache-filename", default=DEFAULT_CACHE_NAME)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    import timm

    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    ).eval().to(args.device)
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    sequences = sequence_dirs(args.data_dir)
    sequences = sequences[args.shard_index::args.num_shards]
    counts = {"written": 0, "skipped": 0}
    for asset_root, image_dir in tqdm(sequences, desc=f"shard {args.shard_index}"):
        status = process_sequence(model, transform, asset_root, image_dir, args)
        counts[status] += 1
    print(json.dumps({"sequences": len(sequences), **counts}, indent=2))


if __name__ == "__main__":
    main()
