"""Build multi-view rest-pose DINO joint descriptors for static-camera records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = REPO_ROOT / "preprocess" / "joint_features"
VIEWS = (
    "front,back,left,right,top,bottom,diag_nnn,diag_nnp,diag_npn,diag_npp,"
    "diag_pnn,diag_pnp,diag_ppn,diag_ppp"
)


def _run(command: list[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--samples", default=16, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    records = [
        json.loads(path.read_text())
        for path in sorted((args.root / "records").glob("*.json"))
    ]
    by_asset = {}
    for record in records:
        source = Path(record["asset_path"])
        asset = source.stem.split("_", 1)[0]
        by_asset.setdefault(asset, source)
    selected = sorted(by_asset.items())[args.shard_index :: args.num_shards]
    if not selected:
        raise ValueError("The selected shard is empty")

    render_root = args.root / "rest_renders"
    feature_root = args.root / "rest_joint_features"
    log_root = args.root / "rest_feature_logs"
    render_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    asset_args = [value for asset, _ in selected for value in ("--asset", asset)]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for _, source in selected:
            handle.write(str(source) + "\n")
        input_list = Path(handle.name)
    try:
        render_command = [
            str(args.blender), "--background", "--python-exit-code", "1",
            "--python", str(MODULE_ROOT / "render_rest_pose.py"), "--",
            "--one-per-species", "--input-dir", str(selected[0][1].parent),
            "--input-list", str(input_list), "--output-root", str(render_root),
            "--views", VIEWS, "--resolution", "512",
            "--engine", "BLENDER_EEVEE_NEXT", "--samples", str(args.samples),
            "--margin", "1.35",
            "--report", str(log_root / f"render_{args.shard_index}.json"),
        ]
        if args.overwrite:
            render_command.append("--overwrite")
        _run(render_command, REPO_ROOT)

        project_command = [
            str(args.blender), "--background", "--python-exit-code", "1",
            "--python", str(MODULE_ROOT / "export_geometry.py"), "--",
            "--render-root", str(render_root), "--output-root", str(feature_root),
            "--max-influences", "8", *asset_args,
        ]
        if args.overwrite:
            project_command.append("--overwrite")
        _run(project_command, REPO_ROOT)

        extract_command = [
            sys.executable, str(MODULE_ROOT / "extract_features.py"),
            "--render-root", str(render_root), "--geometry-root", str(feature_root),
            "--output-root", str(feature_root), "--family", "dinov2",
            "--model", "dinov2_vitb14", "--image-size", "518",
            "--device", args.device, "--background", "0.5", "--dtype", "float16",
            "--expected-feature-dim", "768", "--no-save-view-features", *asset_args,
        ]
        if args.overwrite:
            extract_command.append("--overwrite")
        _run(extract_command, REPO_ROOT)
    finally:
        input_list.unlink(missing_ok=True)

    missing = [
        asset for asset, _ in selected
        if not (feature_root / asset / "joint_averaged_features.npy").is_file()
    ]
    summary = {
        "shard": args.shard_index,
        "assets": len(selected),
        "complete": len(selected) - len(missing),
        "missing": missing,
    }
    (log_root / f"summary_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
