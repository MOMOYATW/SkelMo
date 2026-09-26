"""Extract DINOv2/DINOv3 descriptors and pool them to vertices and joints.

Inputs are rendered multi-view PNGs and the ``geometry.npz`` files written by
``export_geometry.py``. The consolidated archive remains object-free; the two
SkelMo compatibility dictionaries intentionally use NumPy's dictionary format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDER_ROOT = REPO_ROOT / "data/joint_features/renders"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/joint_features/features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=None,
        help="Root containing <animal>/geometry.npz (default: --output-root).",
    )
    parser.add_argument(
        "--asset", "--animal", dest="animal", metavar="NAME", action="append", default=[]
    )
    parser.add_argument("--model", default="dinov2_vitb14")
    parser.add_argument("--family", choices=["dinov2", "dinov3"], default="dinov2")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main",
        help="Local facebookresearch/dinov3 checkout (only used with --family dinov3).",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--background", type=float, default=0.5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--expected-feature-dim",
        type=int,
        default=768,
        help="Reject descriptors incompatible with the released SkelMo model.",
    )
    parser.add_argument("--no-save-view-features", action="store_true")
    parser.add_argument(
        "--save-view-features-for",
        "--view-feature-animal",
        dest="view_feature_animal",
        metavar="NAME",
        action="append",
        default=[],
        help="Save patch maps for these IDs even with --no-save-view-features.",
    )
    parser.add_argument("--no-fill-uncovered", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str,
    device: str,
    family: str = "dinov2",
    checkpoint: Path | None = None,
    repo_dir: Path | None = None,
):
    import torch

    if family == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    else:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("--family dinov3 requires a valid --checkpoint")
        if repo_dir is None or not (repo_dir / "dinov3/hub/backbones.py").is_file():
            raise FileNotFoundError(
                f"DINOv3 source not found at {repo_dir}; clone facebookresearch/dinov3 there first"
            )
        sys.path.insert(0, str(repo_dir))
        from dinov3.hub import backbones

        constructor = getattr(backbones, model_name)
        model = constructor(weights=str(checkpoint.resolve()), pretrained=True)
    model.eval().to(device)
    return model


def sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_images(paths: list[Path], image_size: int, background: float):
    import torch
    import torch.nn.functional as F

    images = []
    alphas = []
    for path in paths:
        rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3] * alpha + float(background) * (1.0 - alpha)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        alphas.append(torch.from_numpy(alpha).permute(2, 0, 1))
    batch = torch.stack(images, dim=0)
    alpha_batch = torch.stack(alphas, dim=0)
    batch = F.interpolate(
        batch, size=(image_size, image_size), mode="bicubic", align_corners=False, antialias=True
    )
    alpha_batch = F.interpolate(
        alpha_batch, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=batch.dtype)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=batch.dtype)[None, :, None, None]
    return (batch - mean) / std, alpha_batch


def extract_patch_maps(model, images, device: str):
    import torch

    with torch.inference_mode():
        output = model.forward_features(images.to(device))
        tokens = output["x_norm_patchtokens"]
    token_count = int(tokens.shape[1])
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise RuntimeError(f"Expected a square DINO patch grid, got {token_count} tokens")
    return tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1]).float().cpu().numpy()


def sample_vertex_features(
    patch_maps: np.ndarray,
    vertex_uv: np.ndarray,
    vertex_visible: np.ndarray,
    vertex_view_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    feature_map = torch.from_numpy(patch_maps).permute(0, 3, 1, 2)
    grid = torch.from_numpy(vertex_uv).unsqueeze(2) * 2.0 - 1.0
    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(-1).permute(0, 2, 1)
    weights = torch.from_numpy(vertex_view_weight * vertex_visible.astype(np.float32))
    weight_sum = weights.sum(dim=0)
    pooled = (sampled * weights[..., None]).sum(dim=0)
    pooled /= weight_sum.clamp_min(1.0e-12)[..., None]
    return pooled.numpy(), weight_sum.numpy()


def sample_vertex_alpha(alpha_maps, vertex_uv: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    grid = torch.from_numpy(vertex_uv).unsqueeze(2) * 2.0 - 1.0
    return (
        F.grid_sample(alpha_maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        .squeeze(1)
        .squeeze(-1)
        .numpy()
    )


def nearest_visible_sources(vertices: np.ndarray, mesh_ids: np.ndarray, covered: np.ndarray) -> np.ndarray:
    sources = np.arange(len(vertices), dtype=np.int32)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    for mesh_id in np.unique(mesh_ids):
        mesh_mask = mesh_ids == mesh_id
        known = np.flatnonzero(mesh_mask & covered)
        missing = np.flatnonzero(mesh_mask & ~covered)
        if not len(missing) or not len(known):
            continue
        if cKDTree is not None:
            tree = cKDTree(vertices[known])
            _, nearest = tree.query(vertices[missing], k=1)
            sources[missing] = known[np.asarray(nearest)]
            continue
        for start in range(0, len(missing), 256):
            batch = missing[start : start + 256]
            squared = ((vertices[batch, None] - vertices[known][None]) ** 2).sum(axis=-1)
            sources[batch] = known[squared.argmin(axis=1)]
    return sources


def pool_joint_features(
    vertex_features: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
    joint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros((joint_count, vertex_features.shape[1]), dtype=np.float32)
    weight_sum = np.zeros(joint_count, dtype=np.float32)
    for slot in range(skin_indices.shape[1]):
        joint_index = skin_indices[:, slot]
        weight = skin_weights[:, slot]
        valid = (joint_index >= 0) & (weight > 0)
        if not valid.any():
            continue
        np.add.at(feature_sum, joint_index[valid], vertex_features[valid] * weight[valid, None])
        np.add.at(weight_sum, joint_index[valid], weight[valid])
    feature_sum /= np.maximum(weight_sum[:, None], 1.0e-12)
    return feature_sum, weight_sum


def pool_dominant_joint_features(
    vertex_features: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
    joint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool vertices by their strongest skinning influence."""
    dominant_slot = skin_weights.argmax(axis=1)
    row_indices = np.arange(len(skin_indices))
    joint_indices = skin_indices[row_indices, dominant_slot]
    dominant_weights = skin_weights[row_indices, dominant_slot]
    valid = (joint_indices >= 0) & (dominant_weights > 0)

    feature_sum = np.zeros((joint_count, vertex_features.shape[1]), dtype=np.float32)
    vertex_count = np.zeros(joint_count, dtype=np.int32)
    np.add.at(feature_sum, joint_indices[valid], vertex_features[valid])
    np.add.at(vertex_count, joint_indices[valid], 1)
    feature_sum /= np.maximum(vertex_count[:, None], 1)
    return feature_sum, vertex_count


def save_skelmo_joint_features(
    output_dir: Path,
    joint_names: np.ndarray,
    averaged_features: np.ndarray,
    dominant_features: np.ndarray,
    dominant_counts: np.ndarray,
) -> None:
    """Write the dictionaries consumed by SkelMo inference and training."""
    names = joint_names.astype(str).tolist()
    averaged = {
        name: averaged_features[index].astype(np.float32)
        for index, name in enumerate(names)
    }
    dominant = {
        name: dominant_features[index].astype(np.float32)
        for index, name in enumerate(names)
        if dominant_counts[index] > 0
    }
    np.save(output_dir / "joint_averaged_features.npy", averaged, allow_pickle=True)
    np.save(output_dir / "joint_max_weight_features.npy", dominant, allow_pickle=True)


def fill_unsupported_joints(
    joint_features: np.ndarray,
    joint_weight_sum: np.ndarray,
    joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill non-deforming helper joints while retaining raw skin support masks."""
    filled = joint_features.copy()
    sources = np.arange(len(joint_features), dtype=np.int32)
    supported = joint_weight_sum > 0
    known = np.flatnonzero(supported)
    missing = np.flatnonzero(~supported)
    if not len(known) or not len(missing):
        return filled, sources
    try:
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(joint_positions[known]).query(joint_positions[missing], k=1)
        sources[missing] = known[np.asarray(nearest)]
    except ImportError:
        squared = ((joint_positions[missing, None] - joint_positions[known][None]) ** 2).sum(axis=-1)
        sources[missing] = known[squared.argmin(axis=1)]
    filled[missing] = joint_features[sources[missing]]
    return filled, sources


def process_animal(
    animal: str,
    render_root: Path,
    geometry_root: Path,
    output_root: Path,
    model,
    device: str,
    model_name: str,
    image_size: int,
    background: float,
    output_dtype,
    save_view_features: bool,
    fill_uncovered: bool,
    expected_feature_dim: int,
) -> dict:
    render_dir = render_root / animal
    output_dir = output_root / animal
    geometry_path = geometry_root / animal / "geometry.npz"
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Missing {geometry_path}; run the Blender projection exporter first")
    geometry = np.load(geometry_path, allow_pickle=False)
    view_names = geometry["view_names"].astype(str).tolist()
    image_paths = [render_dir / f"{view}.png" for view in view_names]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing rendered views: {missing_images}")

    images, alpha_maps = load_images(image_paths, image_size, background)
    patch_maps = extract_patch_maps(model, images, device)
    if expected_feature_dim and patch_maps.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"Model {model_name} produced {patch_maps.shape[-1]} features; "
            f"SkelMo expects {expected_feature_dim}."
        )
    vertex_features, direct_weight = sample_vertex_features(
        patch_maps,
        geometry["vertex_uv"],
        geometry["vertex_visible"],
        geometry["vertex_view_weight"],
    )
    covered = direct_weight > 0
    alpha_at_vertex = sample_vertex_alpha(alpha_maps, geometry["vertex_uv"])
    projection_mask = geometry["vertex_in_frame"] & (alpha_at_vertex >= 0.25)
    fallback_features, fallback_weight = sample_vertex_features(
        patch_maps,
        geometry["vertex_uv"],
        projection_mask,
        geometry["vertex_view_weight"],
    )
    projection_fallback = (~covered) & (fallback_weight > 0)
    vertex_features[projection_fallback] = fallback_features[projection_fallback]
    resolved = covered | projection_fallback
    fill_sources = np.arange(len(covered), dtype=np.int32)
    if fill_uncovered and not resolved.all():
        fill_sources = nearest_visible_sources(geometry["vertices"], geometry["mesh_ids"], resolved)
        can_fill = resolved[fill_sources]
        vertex_features[~resolved & can_fill] = vertex_features[fill_sources[~resolved & can_fill]]

    joint_features, joint_weight_sum = pool_joint_features(
        vertex_features,
        geometry["skin_joint_indices"],
        geometry["skin_weights"],
        len(geometry["joint_names"]),
    )
    joint_features_filled, joint_fill_source = fill_unsupported_joints(
        joint_features, joint_weight_sum, geometry["joint_positions"]
    )
    dominant_features, dominant_counts = pool_dominant_joint_features(
        vertex_features,
        geometry["skin_joint_indices"],
        geometry["skin_weights"],
        len(geometry["joint_names"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_skelmo_joint_features(
        output_dir,
        geometry["joint_names"],
        joint_features_filled,
        dominant_features,
        dominant_counts,
    )
    if save_view_features:
        np.savez_compressed(
            output_dir / "view_features.npz",
            features=patch_maps.astype(output_dtype),
            view_names=geometry["view_names"],
            image_size=np.asarray(image_size, dtype=np.int32),
            model=np.asarray(model_name),
        )

    np.savez_compressed(
        output_dir / "features.npz",
        vertex_features=vertex_features.astype(output_dtype),
        joint_features=joint_features.astype(output_dtype),
        joint_features_filled=joint_features_filled.astype(output_dtype),
        vertex_view_count=geometry["vertex_visible"].sum(axis=0).astype(np.uint8),
        vertex_direct_weight=direct_weight.astype(np.float32),
        vertex_direct_mask=covered,
        vertex_projection_fallback_mask=projection_fallback,
        vertex_fill_source=fill_sources,
        joint_skin_weight_sum=joint_weight_sum,
        joint_has_skin_support=joint_weight_sum > 0,
        joint_fill_source=joint_fill_source,
        joint_dominant_vertex_count=dominant_counts,
        vertices=geometry["vertices"],
        mesh_ids=geometry["mesh_ids"],
        mesh_names=geometry["mesh_names"],
        joint_names=geometry["joint_names"],
        joint_parents=geometry["joint_parents"],
        joint_positions=geometry["joint_positions"],
        skin_joint_indices=geometry["skin_joint_indices"],
        skin_weights=geometry["skin_weights"],
        model=np.asarray(model_name),
        feature_dim=np.asarray(vertex_features.shape[1], dtype=np.int32),
    )

    unresolved = (~resolved) & (~resolved[fill_sources])
    result = {
        "animal": animal,
        "model": model_name,
        "feature_dim": int(vertex_features.shape[1]),
        "patch_grid": [int(patch_maps.shape[1]), int(patch_maps.shape[2])],
        "vertices": int(len(vertex_features)),
        "directly_covered_vertices": int(covered.sum()),
        "direct_coverage": float(covered.mean()),
        "projection_fallback_vertices": int(projection_fallback.sum()),
        "nearest_filled_vertices": int(((~resolved) & (~unresolved)).sum()),
        "unresolved_vertices": int(unresolved.sum()),
        "joints": int(len(joint_features)),
        "joints_with_skin_support": int((joint_weight_sum > 0).sum()),
        "joints_filled_from_nearest_supported": int((joint_weight_sum <= 0).sum()),
        "joint_averaged_features": str(output_dir / "joint_averaged_features.npy"),
        "joint_max_weight_features": str(output_dir / "joint_max_weight_features.npy"),
        "features": str(output_dir / "features.npz"),
        "view_features": str(output_dir / "view_features.npz") if save_view_features else None,
    }
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    import torch

    render_root = args.render_root.resolve()
    output_root = args.output_root.resolve()
    geometry_root = (args.geometry_root or args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.animal)
    view_feature_animals = set(args.view_feature_animal)
    animals = sorted(
        path.name
        for path in render_root.iterdir()
        if path.is_dir() and (not selected or path.name in selected)
    )
    device = choose_device(torch, args.device)
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    checkpoint_hash = sha256(args.checkpoint)
    model_label = args.model
    if checkpoint_hash:
        model_label = f"{args.model}_{checkpoint_hash[:8]}"
    print(f"Loading {model_label} on {device}")
    model = load_model(args.model, device, args.family, args.checkpoint, args.repo_dir)
    report = {
        "render_root": str(render_root),
        "geometry_root": str(geometry_root),
        "output_root": str(output_root),
        "family": args.family,
        "model": model_label,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "device": device,
        "results": [],
        "errors": [],
    }
    for index, animal in enumerate(animals, 1):
        final_path = output_root / animal / "features.npz"
        if final_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(animals)}] skip {animal}")
            metadata_path = output_root / animal / "feature_metadata.json"
            if metadata_path.is_file():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                existing["status"] = "skipped_existing"
                report["results"].append(existing)
            continue
        print(f"[{index}/{len(animals)}] DINO {animal}")
        try:
            result = process_animal(
                animal,
                render_root,
                geometry_root,
                output_root,
                model,
                device,
                model_label,
                args.image_size,
                args.background,
                output_dtype,
                (not args.no_save_view_features) or animal in view_feature_animals,
                not args.no_fill_uncovered,
                args.expected_feature_dim,
            )
            report["results"].append(result)
            print(
                f"  vertices={result['vertices']} coverage={result['direct_coverage']:.2%} "
                f"fallback={result['projection_fallback_vertices']} "
                f"nearest={result['nearest_filled_vertices']} "
                f"joints={result['joints_with_skin_support']}/{result['joints']}"
            )
        except Exception as exc:
            report["errors"].append(
                {"animal": animal, "error": str(exc), "traceback": traceback.format_exc()}
            )
            print(f"ERROR {animal}: {exc}")
        (output_root / "_feature_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    successful = [item for item in report["results"] if "vertices" in item]
    report["summary"] = {
        "animals": len(successful),
        "vertices": sum(item["vertices"] for item in successful),
        "directly_covered_vertices": sum(item["directly_covered_vertices"] for item in successful),
        "projection_fallback_vertices": sum(item["projection_fallback_vertices"] for item in successful),
        "nearest_filled_vertices": sum(item["nearest_filled_vertices"] for item in successful),
        "unresolved_vertices": sum(item["unresolved_vertices"] for item in successful),
        "joints": sum(item["joints"] for item in successful),
        "joints_with_skin_support": sum(item["joints_with_skin_support"] for item in successful),
        "joints_filled_from_nearest_supported": sum(
            item["joints_filled_from_nearest_supported"] for item in successful
        ),
    }
    if report["summary"]["vertices"]:
        report["summary"]["direct_coverage"] = (
            report["summary"]["directly_covered_vertices"] / report["summary"]["vertices"]
        )
    (output_root / "_feature_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Done: results={len(report['results'])}, errors={len(report['errors'])}")


if __name__ == "__main__":
    main()
