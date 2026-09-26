"""Run and validate the complete rest-pose render-to-DINO pipeline.

The heavy stages remain independently runnable.  This driver only standardizes
their paths, view set, model configuration, restart behavior, and QA report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from .views import DEFAULT_VIEW_NAMES, parse_view_names
except ImportError:
    from views import DEFAULT_VIEW_NAMES, parse_view_names


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "joint_features"
RENDER_SCRIPT = MODULE_DIR / "render_rest_pose.py"
PROJECTION_SCRIPT = MODULE_DIR / "export_geometry.py"
FEATURE_SCRIPT = MODULE_DIR / "extract_features.py"
STAGE_ORDER = ("render", "project", "extract", "validate")


def default_blender() -> Path:
    configured = os.environ.get("BLENDER")
    if configured:
        return Path(configured).expanduser()
    macos = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if macos.is_file():
        return macos
    executable = shutil.which("blender")
    return Path(executable) if executable else Path("blender")


def parse_stages(value: str) -> list[str]:
    if value == "all":
        return list(STAGE_ORDER)
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if not stages or unknown:
        raise ValueError(
            f"Invalid stages {unknown or stages}; choose from {', '.join(STAGE_ORDER)}"
        )
    return [stage for stage in STAGE_ORDER if stage in stages]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--render-root", type=Path, default=DEFAULT_DATA_ROOT / "renders"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "features",
        help="Feature output root; geometry is stored here unless --geometry-root is set.",
    )
    parser.add_argument("--geometry-root", type=Path, default=None)
    parser.add_argument("--input-list", type=Path, default=None)
    parser.add_argument(
        "--asset", "--animal", dest="animal", metavar="NAME", action="append", default=[]
    )
    parser.add_argument("--stages", default="all")
    parser.add_argument("--blender", type=Path, default=default_blender())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--views", default=",".join(DEFAULT_VIEW_NAMES))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--engine",
        choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
        default="CYCLES",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument(
        "--hdri-path", type=Path, default=DEFAULT_DATA_ROOT / "lighting.exr"
    )
    parser.add_argument("--hdri-strength", type=float, default=1.0)
    parser.add_argument("--max-influences", type=int, default=8)
    parser.add_argument("--family", choices=["dinov2", "dinov3"], default="dinov2")
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--background", type=float, default=0.5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--expected-feature-dim", type=int, default=768)
    parser.add_argument("--no-save-view-features", action="store_true")
    parser.add_argument(
        "--save-view-features-for",
        "--view-feature-animal",
        dest="view_feature_animal",
        metavar="NAME",
        action="append",
        default=[],
    )
    parser.add_argument("--no-fill-uncovered", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    try:
        args.stages = parse_stages(args.stages)
        args.views = parse_view_names(args.views)
    except ValueError as exc:
        parser.error(str(exc))
    if args.family == "dinov3" and args.checkpoint is None:
        parser.error("--family dinov3 requires --checkpoint")
    if args.model is None:
        args.model = "dinov2_vitb14" if args.family == "dinov2" else "dinov3_vitb16"
    return args


def append_repeated(command: list[str], option: str, values: list[str]) -> None:
    for value in values:
        command.extend((option, value))


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    render_root = args.render_root.resolve()
    output_root = args.output_root.resolve()
    geometry_root = (args.geometry_root or args.output_root).resolve()
    common_animals: list[str] = args.animal

    render = [
        str(args.blender.expanduser()),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        "--one-per-species",
        "--input-dir",
        str(args.input_dir.resolve()),
        "--output-root",
        str(render_root),
        "--views",
        ",".join(args.views),
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
        "--samples",
        str(args.samples),
        "--margin",
        str(args.margin),
        "--hdri-path",
        str(args.hdri_path.resolve()),
        "--hdri-strength",
        str(args.hdri_strength),
    ]
    if args.input_list:
        render.extend(("--input-list", str(args.input_list.resolve())))
    append_repeated(render, "--animal", common_animals)

    project = [
        str(args.blender.expanduser()),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(PROJECTION_SCRIPT),
        "--",
        "--render-root",
        str(render_root),
        "--output-root",
        str(geometry_root),
        "--max-influences",
        str(args.max_influences),
    ]
    append_repeated(project, "--animal", common_animals)

    extract = [
        str(args.python.expanduser()),
        str(FEATURE_SCRIPT),
        "--render-root",
        str(render_root),
        "--geometry-root",
        str(geometry_root),
        "--output-root",
        str(output_root),
        "--family",
        args.family,
        "--model",
        args.model,
        "--image-size",
        str(args.image_size),
        "--device",
        args.device,
        "--background",
        str(args.background),
        "--dtype",
        args.dtype,
        "--expected-feature-dim",
        str(args.expected_feature_dim),
    ]
    append_repeated(extract, "--animal", common_animals)
    append_repeated(extract, "--view-feature-animal", args.view_feature_animal)
    if args.family == "dinov3":
        extract.extend(("--checkpoint", str(args.checkpoint.resolve())))
        extract.extend(("--repo-dir", str(args.repo_dir.expanduser().resolve())))
    if args.no_save_view_features:
        extract.append("--no-save-view-features")
    if args.no_fill_uncovered:
        extract.append("--no-fill-uncovered")
    if args.overwrite:
        render.append("--overwrite")
        project.append("--overwrite")
        extract.append("--overwrite")
    return {"render": render, "project": project, "extract": extract}


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return value if isinstance(value, dict) else {"_read_error": "expected JSON object"}


def selected_animals(render_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return sorted(set(requested))
    if not render_root.is_dir():
        return []
    return sorted(path.name for path in render_root.iterdir() if path.is_dir())


def validate_outputs(
    render_root: Path,
    geometry_root: Path,
    output_root: Path,
    requested_animals: list[str],
    required_views: list[str],
    require_view_features: bool,
) -> dict:
    animals = selected_animals(render_root, requested_animals)
    results = []
    error_count = 0
    warning_count = 0
    for animal in animals:
        errors = []
        warnings = []
        render_dir = render_root / animal
        metadata_path = render_dir / "metadata.json"
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        views = metadata.get("views", required_views)
        try:
            views = parse_view_names(views)
        except ValueError as exc:
            errors.append(str(exc))
            views = required_views
        if list(views) != list(required_views):
            errors.append(f"view set mismatch: metadata={views}, requested={required_views}")
        required = [metadata_path, render_dir / "rest_pose.glb"]
        required.extend(render_dir / f"{view}.png" for view in required_views)
        required.extend(
            [
                geometry_root / animal / "geometry.npz",
                geometry_root / animal / "geometry_metadata.json",
                output_root / animal / "features.npz",
                output_root / animal / "feature_metadata.json",
                output_root / animal / "joint_averaged_features.npy",
                output_root / animal / "joint_max_weight_features.npy",
            ]
        )
        if require_view_features:
            required.append(output_root / animal / "view_features.npz")
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty: {path}")
        camera_data = metadata.get("view_cameras", {})
        missing_camera_matrices = []
        for view in required_views:
            if "matrix_world" not in camera_data.get(view, {}):
                missing_camera_matrices.append(view)
        if missing_camera_matrices:
            camera_distance = metadata.get("camera_distance")
            if isinstance(camera_distance, (int, float)) and camera_distance > 0:
                warnings.append(
                    "legacy camera metadata: using view directions and camera_distance "
                    f"for {len(missing_camera_matrices)} views"
                )
            else:
                errors.append(
                    "camera metadata has neither per-view matrix_world values nor a "
                    f"positive camera_distance: {metadata_path}"
                )
        error_count += len(errors)
        warning_count += len(warnings)
        results.append(
            {
                "animal": animal,
                "status": "ok" if not errors else "error",
                "errors": errors,
                "warnings": warnings,
            }
        )
    if requested_animals:
        missing_animals = sorted(set(requested_animals) - set(animals))
    else:
        missing_animals = []
    for animal in missing_animals:
        error_count += 1
        results.append(
            {
                "animal": animal,
                "status": "error",
                "errors": ["render directory missing"],
                "warnings": [],
            }
        )
    if not animals and not requested_animals:
        error_count += 1
        results.append(
            {
                "animal": None,
                "status": "error",
                "errors": ["no rendered animals found"],
                "warnings": [],
            }
        )
    return {
        "animals": len([item for item in results if item["animal"] is not None]),
        "ok": sum(item["status"] == "ok" for item in results),
        "error_count": error_count,
        "warning_count": warning_count,
        "results": results,
    }


def run_command(stage: str, command: list[str], dry_run: bool) -> dict:
    print(f"\n[{stage}] {shlex.join(command)}", flush=True)
    if dry_run:
        return {"stage": stage, "status": "dry_run", "command": command, "seconds": 0.0}
    started = time.monotonic()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode:
        return {
            "stage": stage,
            "status": "error",
            "command": command,
            "exit_code": completed.returncode,
            "seconds": round(elapsed, 3),
        }
    return {
        "stage": stage,
        "status": "complete",
        "command": command,
        "exit_code": completed.returncode,
        "seconds": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    args.render_root = args.render_root.resolve()
    args.output_root = args.output_root.resolve()
    geometry_root = (args.geometry_root or args.output_root).resolve()
    report_path = (args.report or args.output_root / "_pipeline_report.json").resolve()
    commands = build_commands(args)
    report = {
        "input_dir": str(args.input_dir.resolve()),
        "render_root": str(args.render_root),
        "geometry_root": str(geometry_root),
        "output_root": str(args.output_root),
        "views": args.views,
        "animals": args.animal,
        "family": args.family,
        "model": args.model,
        "stages": args.stages,
        "runs": [],
    }
    if not args.dry_run:
        args.render_root.mkdir(parents=True, exist_ok=True)
        geometry_root.mkdir(parents=True, exist_ok=True)
        args.output_root.mkdir(parents=True, exist_ok=True)
    failed_stage = None
    for stage in args.stages:
        if stage == "validate":
            if args.dry_run:
                report["validation"] = {"status": "dry_run"}
            else:
                report["validation"] = validate_outputs(
                    args.render_root,
                    geometry_root,
                    args.output_root,
                    args.animal,
                    args.views,
                    not args.no_save_view_features,
                )
            continue
        run = run_command(stage, commands[stage], args.dry_run)
        report["runs"].append(run)
        if run["status"] == "error":
            failed_stage = stage
            break
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nPipeline report: {report_path}")
        if failed_stage:
            raise SystemExit(f"{failed_stage} failed; see {report_path}")
        validation = report.get("validation")
        if validation and validation.get("error_count"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
