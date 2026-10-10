"""Sample held-out static-camera clips and render prediction/GT overlays."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_loaders.static_camera_dataset import StaticCameraDataset
from data_loaders.tensors import truebones_batch_collate
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils.rotation_conversions import rotation_6d_to_matrix_safe


COLORS = {
    "gt": (35, 225, 245, 255),
    "pred": (255, 82, 45, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_training_args(checkpoint):
    payload = json.loads((checkpoint.parent / "args.json").read_text())
    # Sampling is a single-process CUDA job; the model/diffusion fields come
    # directly from the checkpoint's training configuration.
    payload["device"] = 0
    return SimpleNamespace(**payload)


def select_indices(dataset, clip_ids):
    by_id = {record["clip_id"]: i for i, record in enumerate(dataset.records)}
    if clip_ids:
        missing = [clip_id for clip_id in clip_ids if clip_id not in by_id]
        if missing:
            raise KeyError(f"Validation clip ids not found: {missing}")
        return [by_id[clip_id] for clip_id in clip_ids]
    # Deterministic spread across topology sizes, preferring full-length clips.
    candidates = [
        (i, int(record["joint_count"]))
        for i, record in enumerate(dataset.records)
        if int(record["frame_count"]) >= dataset.max_motion_length
    ]
    candidates.sort(key=lambda item: item[1])
    if len(candidates) < 3:
        candidates = sorted(
            [(i, int(record["joint_count"])) for i, record in enumerate(dataset.records)],
            key=lambda item: item[1],
        )
    locations = [0, len(candidates) // 2, len(candidates) - 1]
    return [candidates[location][0] for location in locations]


def to_device(condition, device):
    return {
        "y": {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in condition["y"].items()
        }
    }


def denormalize(array, mean, std):
    return array * std[None] + mean[None]


def rotation_error_degrees(predicted, target):
    pred = rotation_6d_to_matrix_safe(torch.from_numpy(predicted[..., 3:9]).float())
    truth = rotation_6d_to_matrix_safe(torch.from_numpy(target[..., 3:9]).float())
    relative = pred.transpose(-1, -2) @ truth
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).numpy()


def compute_metrics(predicted, target, parents):
    position_error = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1)
    root_error = position_error[:, 0]
    valid = np.asarray(parents) >= 0
    child = np.flatnonzero(valid)
    parent = np.asarray(parents)[child]
    pred_lengths = np.linalg.norm(
        predicted[:, child, :3] - predicted[:, parent, :3], axis=-1
    )
    true_lengths = np.linalg.norm(
        target[:, child, :3] - target[:, parent, :3], axis=-1
    )
    bone_error = np.abs(pred_lengths - true_lengths)
    metrics = {
        "mpjpe_cm": float(position_error.mean() * 100.0),
        "root_error_cm": float(root_error.mean() * 100.0),
        "bone_length_error_cm": float(bone_error.mean() * 100.0),
    }
    if predicted.shape[-1] >= 9 and target.shape[-1] >= 9:
        angle_error = rotation_error_degrees(predicted, target)
        metrics.update(
            rotation_error_deg=float(angle_error.mean()),
            rotation_error_median_deg=float(np.median(angle_error)),
        )
    return metrics


def transform_points(matrix, points):
    homogeneous = np.concatenate(
        [points, np.ones(points.shape[:-1] + (1,), dtype=points.dtype)], axis=-1
    )
    return homogeneous @ np.asarray(matrix).T


def project_gravity_points(points, motion):
    gravity_to_world = np.linalg.inv(motion["world_to_camera_gravity"])
    world = transform_points(gravity_to_world, points)[..., :3]
    camera = transform_points(motion["world_to_camera_cv"], world)[..., :3]
    depth = camera[..., 2]
    projected = camera @ np.asarray(motion["intrinsics"]).T
    uv = projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)
    return uv, depth


def draw_skeleton(draw, uv, depth, parents, color, width, radius):
    valid = np.isfinite(uv).all(axis=-1) & (depth > 1e-6)
    for joint, parent in enumerate(parents):
        if parent >= 0 and valid[joint] and valid[parent]:
            draw.line(
                [tuple(uv[parent]), tuple(uv[joint])],
                fill=color,
                width=width,
            )
    for joint, point in enumerate(uv):
        if valid[joint]:
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(0, 0, 0, 220),
                width=1,
            )


def render_overlay(output, record, predicted, target, parents, metrics):
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with np.load(record["motion_path"], allow_pickle=False) as source:
        motion = {key: source[key] for key in source.files}
    fps = int(motion["fps"])
    start = int(record["frame_start"])
    render_dir = Path(record["render_dir"]) / "v000"
    for frame_index in range(len(predicted)):
        frame_path = render_dir / f"{start + frame_index:04d}.webp"
        image = Image.open(frame_path).convert("RGBA")
        original = image.copy()
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        gt_uv, gt_depth = project_gravity_points(target[frame_index, :, :3], motion)
        pred_uv, pred_depth = project_gravity_points(predicted[frame_index, :, :3], motion)
        width = max(2, image.width // 170)
        radius = max(2, image.width // 220)
        draw_skeleton(draw, gt_uv, gt_depth, parents, COLORS["gt"], width + 1, radius)
        draw_skeleton(draw, pred_uv, pred_depth, parents, COLORS["pred"], width, radius)
        canvas = Image.new("RGBA", (image.width * 2, image.height + 42), (14, 18, 23, 255))
        canvas.paste(original, (0, 42))
        canvas.paste(overlay, (image.width, 42))
        header = ImageDraw.Draw(canvas)
        header.text((12, 12), "INPUT", fill=(235, 240, 245, 255))
        header.text(
            (image.width + 12, 12),
            "GT (cyan)  PRED (orange)",
            fill=(235, 240, 245, 255),
        )
        canvas.convert("RGB").save(frames_dir / f"{frame_index:04d}.jpg", quality=92)
    video = output / "comparison.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frames_dir / "%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video),
        ],
        check=True,
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return video


def build_index(output, checkpoint, split, rows):
    cards = []
    for row in rows:
        metrics = row["metrics"]
        rotation_metric = ""
        if "rotation_error_deg" in metrics:
            rotation_metric = (
                f"<dt>Rotation error</dt><dd>{metrics['rotation_error_deg']:.1f}° mean / "
                f"{metrics['rotation_error_median_deg']:.1f}° median</dd>"
            )
        cards.append(
            f"""<article><h2>{html.escape(row['clip_id'])}</h2>
            <p>{row['joints']} joints · {row['frames']} frames · yaw {row['yaw']:+.1f}° · seed {row['seed']}</p>
            <video src="{html.escape(row['video'])}" controls loop muted playsinline preload="metadata"></video>
            <dl><dt>MPJPE</dt><dd>{metrics['mpjpe_cm']:.2f} cm</dd>
            <dt>Root error</dt><dd>{metrics['root_error_cm']:.2f} cm</dd>
            <dt>Bone-length error</dt><dd>{metrics['bone_length_error_cm']:.2f} cm</dd>
            {rotation_metric}</dl></article>"""
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>SkelMo {html.escape(split)} evaluation</title><style>
    body{{margin:0;background:#101419;color:#eef3f7;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}
    .note{{color:#aebbc7}}article{{background:#192028;border:1px solid #303b46;border-radius:14px;padding:18px;margin:18px 0}}
    h2{{font-size:17px;margin:0}}article p{{color:#9eb0bf}}video{{display:block;width:100%;border-radius:10px;background:#000}}
    dl{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}dt{{color:#93a5b5}}dd{{font-size:19px;margin:2px 0}}
    @media(max-width:720px){{dl{{grid-template-columns:repeat(2,1fr)}}}}
    </style></head><body><main><h1>SkelMo {html.escape(split)} split evaluation</h1>
    <p class="note">Checkpoint {html.escape(checkpoint.name)} · split {html.escape(split)}. Left: input RGB. Right: ground truth skeleton in cyan and sampled prediction in orange. Metrics are diagnostic only; this is a single stochastic sample per clip.</p>
    {''.join(cards)}</main></body></html>"""
    (output / "index.html").write_text(document)


def main():
    cli = parse_args()
    args = load_training_args(cli.checkpoint)
    cli.output.mkdir(parents=True, exist_ok=True)
    dataset = StaticCameraDataset(
        root=cli.root,
        split=cli.split,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        video_cache_filename=args.video_cache_filename,
        video_feature_mode=args.video_feature_mode,
        motion_representation=args.motion_representation,
        reference_condition_mode=getattr(args, "reference_condition_mode", "rest"),
        joint_permutation_prob=0.0,
        joint_feature_dropout_prob=0.0,
        video_joint_dropout_prob=0.0,
        leaf_prune_prob=0.0,
        joint_subdivision_prob=0.0,
    )
    selected = select_indices(dataset, cli.clip_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    try:
        state = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(cli.checkpoint, map_location="cpu")
    load_model(model, state)
    model.to(device).eval()
    rows = []
    for order, dataset_index in enumerate(selected):
        seed = cli.seed + order
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        item = dataset[dataset_index]
        record = dataset.records[dataset_index]
        motion, condition = truebones_batch_collate([item])
        length = int(condition["y"]["lengths"][0])
        joints = int(condition["y"]["n_joints"][0])
        device_condition = to_device(condition, device)
        print(f"Sampling {record['clip_id']} ({joints} joints, {length} frames)", flush=True)
        with torch.inference_mode():
            sampled = diffusion.p_sample_loop(
                model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=device_condition,
                device=device,
                progress=True,
                guidance_scale=cli.guidance_scale,
            )
        pred_norm = sampled[0, :joints, :, :length].permute(2, 0, 1).cpu().numpy()
        true_norm = motion[0, :joints, :, :length].permute(2, 0, 1).numpy()
        mean = condition["y"]["mean"][0, :joints].numpy()
        std = condition["y"]["std"][0, :joints].numpy()
        predicted = denormalize(pred_norm, mean, std)
        target = denormalize(true_norm, mean, std)
        parents = np.asarray(item[2], dtype=np.int64)
        metrics = compute_metrics(predicted, target, parents)
        clip_output = cli.output / record["clip_id"]
        video = render_overlay(clip_output, record, predicted, target, parents, metrics)
        np.savez_compressed(
            clip_output / "prediction.npz",
            predicted=predicted,
            target=target,
            parents=parents,
        )
        rows.append(
            {
                "clip_id": record["clip_id"],
                "joints": joints,
                "frames": length,
                "yaw": float(record.get("camera_azimuth_deg", 0.0)),
                "seed": seed,
                "metrics": metrics,
                "video": f"{record['clip_id']}/{video.name}",
            }
        )
        print(json.dumps(rows[-1], indent=2), flush=True)
    build_index(cli.output, cli.checkpoint, cli.split, rows)
    (cli.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(cli.output / "index.html")


if __name__ == "__main__":
    main()
