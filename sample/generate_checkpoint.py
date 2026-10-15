"""Infer with the SkelMo checkpoint.

The prepared NPZ describes a skeleton in the training gravity-camera frame:
parents [J], first_pose [J,3], joint_uv [J,2], joint_visible [J],
resolution (scalar), and optionally dense_dino [T,1369,128]. If dense_dino
is absent, --frames-dir supplies T ordered RGB frames for DINO extraction.
The optional rest-pose appearance checkpoint also needs rest_pose [J,3] and
rest_appearance [J,768] in the prepared NPZ.
Coordinates use the training scene-normalized gravity-camera frame (the
scene's longest bbox dimension is one unit) and must match the rendered frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_loaders.static_camera_dataset import (
    _root_aligned_rest,
    _sample_dense_joint_features,
    _structural_joint_features,
)
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    prepend_rest_video_token,
)
from data_loaders.truebones.truebones_utils.motion_process import create_topology_edge_relations
from data_loaders.truebones.truebones_utils.param_utils import MAX_PATH_LEN
from sample.evaluate_static_camera import decode_positions, to_device, zero_first_frame
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


EXPECTED_SHA256 = {
    "first_frame_dino": {
        "model000099999.pt": "d662b18663c1d6ae5ba547adbec927e0d9854291b3e4615b146b1cbc628a4547",
        "args.json": "e6e5139536e57d08b76a56c479fc97608585de1fc7fd5f02b911c71777fa7ee6",
        "training_stats.json": "c62f631c9cd66b210a955219031a0e00155d9eee9e12f6fa2fc69dd7c289c640",
        "pca_128.npz": "ea6b3e778f1a08333525e4edefba882b4bf1c9f41f7ed4fed61687ab2c2bae6a",
    },
    "rest_merged": {
        "model000099999.pt": "a72f90152390b72b3521f0d88d5d316448170c663da68aad984613fc481ede3c",
        "args.json": "327a5b997d1c86856d342cd47a155e97ed41a34aa4ec9b17fc050ee390332a5a",
        "training_stats.json": "c62f631c9cd66b210a955219031a0e00155d9eee9e12f6fa2fc69dd7c289c640",
        "pca_128.npz": "ea6b3e778f1a08333525e4edefba882b4bf1c9f41f7ed4fed61687ab2c2bae6a",
    },
}


def verify_bundle(checkpoint, stats, pca, joint_feature_source):
    files = {
        "model000099999.pt": checkpoint,
        "args.json": checkpoint.parent / "args.json",
        "training_stats.json": stats,
        "pca_128.npz": pca,
    }
    for name, path in files.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != EXPECTED_SHA256[joint_feature_source][name]:
            raise ValueError(f"{name} checksum mismatch: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Prepared skeleton NPZ")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path, help="Matching training_stats.json")
    parser.add_argument("--pca", required=True, type=Path, help="Matching pca_128.npz")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--features", type=Path, help="Cached [T,1369,128] PCA-DINO NPY")
    group.add_argument("--frames-dir", type=Path, help="Ordered input PNG/JPG/WebP frames")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ik-bvh", action="store_true", help="Also fit fixed-length IK and export BVH")
    parser.add_argument("--ik-iterations", type=int, default=50)
    return parser.parse_args()


def validate_checkpoint(args):
    joint_feature_source = getattr(args, "joint_feature_source", None)
    if joint_feature_source not in EXPECTED_SHA256:
        raise ValueError(f"Unsupported checkpoint joint feature source: {joint_feature_source}")
    required = {
        "num_frames": 32,
        "motion_representation": "positions",
        "reference_condition_mode": "first_frame",
        "position_target_mode": "first_frame_delta",
        "video_feature_mode": "dense_dino",
        "video_dense_dim": 128,
        "joint_feature_dim": 256,
        "video_spatiotemporal_attention": True,
        "video_attention_fusion": "fixed_residual",
        "video_attention_residual_scale": 0.1,
        "reference_fusion_layers": 0,
    }
    mismatches = {key: (getattr(args, key, None), value)
                  for key, value in required.items() if getattr(args, key, None) != value}
    if mismatches:
        raise ValueError(f"Checkpoint configuration does not match this release: {mismatches}")
    if getattr(args, "cond_mask_prob", None) != 0.0:
        raise ValueError("This inference route requires the pure-conditional checkpoint")
    return joint_feature_source


def load_dense_frames(frames_dir, pca_path, device, resolution):
    from PIL import Image
    import timm
    from torchvision import transforms
    from preprocess.static_camera_pilot.extract_dense_dino import BACKBONE, patch_tokens

    paths = sorted(path for path in frames_dir.iterdir()
                   if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise ValueError(f"No image frames in {frames_dir}")
    if len(paths) > 32:
        raise ValueError("This checkpoint supports at most 32 ordered input frames")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (resolution, resolution):
                raise ValueError(
                    f"Frame {path} has size {image.size}; prepared projection "
                    f"requires the same uncropped {resolution}x{resolution} render"
                )
    with np.load(pca_path, allow_pickle=False) as pca:
        mean = torch.from_numpy(pca["mean"].astype(np.float32)).to(device)
        components = torch.from_numpy(pca["components"].astype(np.float32)).to(device)
    if mean.shape != (768,) or components.shape != (768, 128):
        raise ValueError("The checkpoint requires a 768-to-128 DINO PCA basis")
    dino = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((518, 518)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(paths), 8):
            batch = torch.stack([transform(Image.open(path).convert("RGB"))
                                 for path in paths[start:start + 8]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                tokens = patch_tokens(dino, batch)
            outputs.append(((tokens.float() - mean) @ components).cpu().numpy())
    return np.concatenate(outputs).astype(np.float16)


def load_prepared(path, features_path, frames_dir, pca_path, device, joint_feature_source):
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    resolution = int(values["resolution"])
    if features_path is not None:
        dense = np.load(features_path, allow_pickle=False)
    elif frames_dir is not None:
        dense = load_dense_frames(frames_dir, pca_path, device, resolution)
    elif "dense_dino" in values:
        dense = values["dense_dino"]
    else:
        raise ValueError("Provide --features, --frames-dir, or dense_dino in the input NPZ")
    parents = np.asarray(values["parents"], dtype=np.int64)
    first_pose = np.asarray(values["first_pose"], dtype=np.float32)
    uv = np.asarray(values["joint_uv"], dtype=np.float32)
    visible = np.asarray(values["joint_visible"], dtype=bool)
    if parents.ndim != 1 or len(parents) < 2 or len(parents) > 143:
        raise ValueError("parents must describe 2-143 joints")
    if np.flatnonzero(parents < 0).tolist() != [0] or np.any(parents[1:] >= np.arange(1, len(parents))):
        raise ValueError("parents must be parent-first with a single root at index 0")
    if first_pose.shape != (len(parents), 3) or uv.shape != (len(parents), 2) or visible.shape != (len(parents),):
        raise ValueError("first_pose, joint_uv, and joint_visible must match parents")
    if not np.isfinite(first_pose).all() or not np.isfinite(uv).all() or resolution < 32:
        raise ValueError("Invalid first pose, projection, or image resolution")
    if not visible.any() or np.any(
        visible & ((uv[:, 0] < 0) | (uv[:, 0] >= resolution)
                   | (uv[:, 1] < 0) | (uv[:, 1] >= resolution))
    ):
        raise ValueError("joint_visible disagrees with joint_uv and image resolution")
    if dense.ndim != 3 or dense.shape[1:] != (1369, 128) or not 1 <= len(dense) <= 32:
        raise ValueError(f"Expected 1-32 frames of [1369,128] DINO, got {dense.shape}")
    if not np.isfinite(dense).all():
        raise ValueError("DINO features contain NaN/Inf")
    if joint_feature_source == "rest_merged":
        if "rest_pose" not in values or "rest_appearance" not in values:
            raise ValueError("Rest checkpoint requires rest_pose and rest_appearance")
        rest_pose = np.asarray(values["rest_pose"], dtype=np.float32)
        rest_appearance = np.asarray(values["rest_appearance"], dtype=np.float32)
        if rest_pose.shape != (len(parents), 3) or rest_appearance.shape != (len(parents), 768):
            raise ValueError("Rest checkpoint requires rest_pose [J,3] and rest_appearance [J,768]")
        if not np.isfinite(rest_pose).all() or not np.isfinite(rest_appearance).all():
            raise ValueError("Rest pose or appearance contains NaN/Inf")
    return values, dense.astype(np.float16), parents, first_pose, uv, visible, resolution


def save_projection_preview(frames_dir, output, uv, visible, parents):
    from PIL import Image, ImageDraw

    first = sorted(path for path in frames_dir.iterdir()
                   if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})[0]
    image = Image.open(first).convert("RGB")
    draw = ImageDraw.Draw(image)
    for joint, parent in enumerate(parents):
        if parent >= 0 and visible[joint] and visible[parent]:
            draw.line([tuple(uv[parent]), tuple(uv[joint])], fill=(0, 240, 255), width=3)
    for joint, point in enumerate(uv):
        if visible[joint]:
            x, y = map(float, point)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 80, 30))
    image.save(output / "first_frame_projection.png")


def build_item(dense, parents, first_pose, uv, visible, resolution, stats,
               joint_feature_source="first_frame_dino", values=None, pca_path=None):
    length = len(dense)
    mean_1 = np.asarray(stats["position_mean"], dtype=np.float32)
    std_1 = np.asarray(stats["position_std"], dtype=np.float32)
    if mean_1.shape != (3,) or std_1.shape != (3,) or np.any(std_1 <= 0):
        raise ValueError("Invalid training normalization statistics")
    mean = np.broadcast_to(mean_1, first_pose.shape).copy()
    std = np.broadcast_to(std_1, first_pose.shape).copy()
    reference = (first_pose - mean) / std
    if joint_feature_source == "rest_merged":
        rest_geometry, _ = _root_aligned_rest(values["rest_pose"], first_pose, parents)
        structure = _structural_joint_features(rest_geometry, parents, width=128)
        structure /= np.maximum(np.linalg.norm(structure, axis=-1, keepdims=True), 1e-8)
        with np.load(pca_path, allow_pickle=False) as pca:
            appearance = (values["rest_appearance"].astype(np.float32) - pca["mean"]) @ pca["components"]
        appearance /= np.maximum(np.linalg.norm(appearance, axis=-1, keepdims=True), 1e-8)
        joint_features = np.concatenate([appearance, structure], axis=-1)
        offsets = rest_geometry.copy()
    else:
        aligned = _sample_dense_joint_features(dense[0], uv, visible, resolution)
        aligned /= np.maximum(np.linalg.norm(aligned, axis=-1, keepdims=True), 1e-8)
        joint_features = np.concatenate([aligned, np.zeros_like(aligned)], axis=-1)
        offsets = first_pose.copy()
    relations, graph_dist = create_topology_edge_relations(parents.tolist(), max_path_len=MAX_PATH_LEN)
    offsets[1:] -= offsets[parents[1:]]
    padded_tracks = np.zeros((33, 1369, 128), dtype=np.float16)
    padded_tracks[:length + 1] = prepend_rest_video_token(dense, dense[0])
    return (
        np.zeros((32, len(parents), 3), dtype=np.float32), length, parents,
        reference, offsets, create_temporal_mask_for_window(31, 32),
        graph_dist, relations, joint_features, 0, mean, std,
        padded_tracks, 143, "dense_dino",
    )


def main():
    cli = parse_args()
    if cli.output.exists():
        raise FileExistsError(f"Output already exists: {cli.output}")
    args = SimpleNamespace(**json.loads((cli.checkpoint.parent / "args.json").read_text()))
    joint_feature_source = validate_checkpoint(args)
    verify_bundle(cli.checkpoint, cli.stats, cli.pca, joint_feature_source)
    if cli.guidance_scale != 1.0:
        raise ValueError("This model was trained without classifier-free dropout; use guidance-scale=1")
    device = torch.device(cli.device)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cli.seed)
    values, dense, parents, first_pose, uv, visible, resolution = load_prepared(
        cli.input, cli.features, cli.frames_dir, cli.pca, device, joint_feature_source
    )
    stats = json.loads(cli.stats.read_text())
    item = build_item(dense, parents, first_pose, uv, visible, resolution, stats,
                      joint_feature_source, values, cli.pca)
    motion, condition = truebones_batch_collate([item])
    args.device = 0
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_avg", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_model(model, state)
    model.to(device).eval()
    with torch.inference_mode():
        sampled = diffusion.p_sample_loop(
            model, motion.shape, clip_denoised=False,
            model_kwargs=to_device(condition, device), device=device,
            progress=False, guidance_scale=1.0, denoised_fn=zero_first_frame,
        )
    joints = len(parents)
    pred_norm = sampled[0, :joints, :, :len(dense)].permute(2, 0, 1).cpu().numpy()
    predicted = decode_positions(pred_norm, item[10], item[11], item[3], "first_frame_delta")
    cli.output.mkdir(parents=True)
    if cli.frames_dir is not None:
        save_projection_preview(cli.frames_dir, cli.output, uv, visible, parents)
    np.savez_compressed(cli.output / "prediction.npz", predicted=predicted,
                        parents=parents, first_pose=first_pose,
                        joint_uv=uv, joint_visible=visible)
    summary = {"checkpoint": str(cli.checkpoint), "input": str(cli.input),
               "frames": len(dense), "joints": joints, "seed": cli.seed,
               "guidance_scale": 1.0, "visible_first_frame_joints": int(visible.sum()),
               "coordinate_frame": "scene-normalized gravity camera: X-left, Y-up, Z-forward; longest bbox dimension = 1",
               "output": str(cli.output / "prediction.npz")}
    if cli.ik_bvh:
        from sample.ik_export import fit_animation, safe_joint_names, save_and_verify
        offsets = first_pose.copy()
        offsets[1:] -= first_pose[parents[1:]]
        offsets[0] = 0
        animation, order, fitted = fit_animation(predicted, parents, offsets, cli.ik_iterations)
        names = values.get("joint_names", np.asarray([f"joint_{i}" for i in range(joints)]))
        fps = float(values.get("fps", 30.0))
        bvh_path = cli.output / "prediction_ik.bvh"
        summary["bvh"] = save_and_verify(bvh_path, animation, safe_joint_names(names)[order], fps)
        fitted_in_input_order = np.empty_like(fitted)
        fitted_in_input_order[:, order] = fitted
        np.save(cli.output / "prediction_ik.npy", fitted_in_input_order)
    (cli.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
