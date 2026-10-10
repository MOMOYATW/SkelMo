"""Extract compact DINO features at ground-truth projected joint locations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def _patch_maps(model, images):
    output = model.forward_features(images)
    if isinstance(output, dict):
        patches = output.get("x_norm_patchtokens", output.get("x_prenorm"))
        if patches is None:
            raise ValueError(f"Unsupported DINO output keys: {output.keys()}")
        if patches.shape[1] == 1370:
            patches = patches[:, 1:]
    else:
        patches = output[:, 1:]
    side = int(patches.shape[1] ** 0.5)
    if side * side != patches.shape[1]:
        raise ValueError(f"Patch count is not square: {patches.shape[1]}")
    return patches.transpose(1, 2).reshape(len(images), patches.shape[-1], side, side)


def _extract_record(model, transform, root, record, args):
    clip_id = record["clip_id"]
    output = root / "tracks" / f"{clip_id}.{args.cache_filename}"
    if output.is_file() and not args.overwrite:
        cached = np.load(output, mmap_mode="r")
        if cached.ndim == 3 and cached.shape[-1] == 776:
            return "skipped"
        raise ValueError(f"Invalid existing cache: {output} {cached.shape}")
    with np.load(record["motion_path"], allow_pickle=False) as motion:
        uv = motion["joint_position_2d_px"].astype(np.float32)
        visible = motion["joint_in_frame"].astype(np.float32)
    frame_paths = sorted((Path(record["render_dir"]) / "v000").glob("*.webp"))
    if len(frame_paths) != len(uv):
        raise ValueError(f"Frame/label mismatch for {clip_id}: {len(frame_paths)} vs {len(uv)}")
    resolution = float(json.loads((Path(record["render_dir"]) / "v000.json").read_text())["resolution"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}.npy")
    result = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(uv), uv.shape[1], 776)
    )
    with torch.inference_mode():
        for start in range(0, len(frame_paths), args.batch_size):
            paths = frame_paths[start:start + args.batch_size]
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            images = torch.stack(tensors).to(args.device, non_blocking=True)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.float16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                patches = F.normalize(_patch_maps(model, images).float(), dim=1)
                xy = torch.from_numpy(uv[start:start + len(paths)]).to(args.device)
                xy_norm = xy / max(resolution - 1.0, 1.0) * 2.0 - 1.0
                grid = xy_norm.clamp(-1.0, 1.0).unsqueeze(2)
                appearance = F.grid_sample(
                    patches, grid, mode="bilinear", align_corners=True
                ).squeeze(-1).transpose(1, 2)
                support = torch.from_numpy(
                    visible[start:start + len(paths)]
                ).to(args.device).unsqueeze(-1)
                appearance = appearance * support
                zeros2 = torch.zeros_like(xy_norm)
                zeros1 = support * 0.0
                tracks = torch.cat(
                    [
                        appearance,
                        xy_norm.clamp(-2.0, 2.0),
                        zeros2,
                        support,
                        zeros1,
                        support,
                        support,
                    ],
                    dim=-1,
                )
            if tracks.shape[-1] != 776 or not torch.isfinite(tracks).all():
                raise ValueError(f"Invalid track tensor for {clip_id}: {tracks.shape}")
            result[start:start + len(paths)] = tracks.cpu().numpy().astype(np.float16)
    result.flush()
    del result
    os.replace(temporary, output)
    return "written"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--cache-filename", default="joint_tracks_dinov2_vitb14_fp16.npy")
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--limit", type=int)
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
    records = [
        json.loads(path.read_text())
        for path in sorted((args.root / "records").glob("*.json"))
    ][args.shard_index::args.num_shards]
    if args.limit is not None:
        records = records[:args.limit]
    counts = {"written": 0, "skipped": 0, "failed": 0}
    for record in tqdm(records, desc=f"shard {args.shard_index}"):
        try:
            status = _extract_record(model, transform, args.root, record, args)
            counts[status] += 1
        except Exception as error:
            counts["failed"] += 1
            print(f"failed {record['clip_id']}: {error}", flush=True)
    print(json.dumps({"records": len(records), **counts}, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
