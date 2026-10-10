"""Validate frame alignment, camera geometry and rotation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import matrix_to_rotation_6d_columns, project_world_points, transform_points


def validate_record(record_path):
    record = json.loads(record_path.read_text())
    motion = np.load(record["motion_path"])
    rig = np.load(record["rig_path"])
    frames = int(record["frame_count"])
    joints = int(record["joint_count"])
    errors = []

    expected_shapes = {
        "joint_position_world": (frames, joints, 3),
        "joint_position_gravity_camera": (frames, joints, 3),
        "joint_position_2d_px": (frames, joints, 2),
        "skin_rotation_gravity_camera_matrix": (frames, joints, 3, 3),
        "skin_rotation_gravity_camera_6d": (frames, joints, 6),
        "contact_heuristic": (frames, joints),
        "rest_global_position_gravity_camera": (joints, 3),
        "rest_global_position_2d_px": (joints, 2),
    }
    for key, shape in expected_shapes.items():
        if key not in motion or motion[key].shape != shape:
            errors.append(f"{key}: expected {shape}, got {getattr(motion.get(key), 'shape', None)}")
    if rig["parents"].shape != (joints,):
        errors.append(f"parents: expected {(joints,)}, got {rig['parents'].shape}")

    rendered = len(list((Path(record["render_dir"]) / "v000").glob("*.webp")))
    if rendered != frames:
        errors.append(f"rendered frames: expected {frames}, got {rendered}")

    numeric = [motion[key] for key in motion.files if motion[key].dtype.kind in "fci"]
    if not all(np.isfinite(value).all() for value in numeric):
        errors.append("motion contains NaN or Inf")

    rotation = motion["skin_rotation_gravity_camera_matrix"].astype(np.float64)
    orth_error = float(np.max(np.abs(rotation @ np.swapaxes(rotation, -1, -2) - np.eye(3))))
    det_error = float(np.max(np.abs(np.linalg.det(rotation) - 1.0)))
    if orth_error > 1e-4 or det_error > 1e-4:
        errors.append(f"invalid SO(3): orth={orth_error:.3g}, det={det_error:.3g}")
    six_error = float(np.max(np.abs(
        matrix_to_rotation_6d_columns(rotation)
        - motion["skin_rotation_gravity_camera_6d"]
    )))
    if six_error > 1e-5:
        errors.append(f"6D layout mismatch: {six_error:.3g}")

    camera_inverse_error = float(np.max(np.abs(
        motion["world_to_camera_gravity"] @ motion["camera_to_world_gravity"]
        - np.eye(4)
    )))
    if camera_inverse_error > 1e-5:
        errors.append(f"camera inverse mismatch: {camera_inverse_error:.3g}")
    uv, depth, in_frame = project_world_points(
        motion["joint_position_world"],
        motion["world_to_camera_cv"],
        motion["intrinsics"],
        json.loads((Path(record["render_dir"]) / "v000.json").read_text())["resolution"],
    )
    projection_error = float(np.max(np.abs(uv - motion["joint_position_2d_px"])))
    if projection_error > 1e-3:
        errors.append(f"projection mismatch: {projection_error:.3g}px")
    if not np.array_equal(in_frame, motion["joint_in_frame"]):
        errors.append("in-frame mask mismatch")
    if not np.allclose(depth, motion["joint_depth_camera_cv"], atol=1e-5):
        errors.append("camera depth mismatch")

    rest_gravity = transform_points(
        motion["world_to_camera_gravity"], motion["rest_global_position_world"]
    )
    if not np.allclose(
        rest_gravity, motion["rest_global_position_gravity_camera"], atol=1e-5
    ):
        errors.append("rest pose is not aligned to the gravity-camera frame")
    rest_uv, rest_depth, rest_in_frame = project_world_points(
        motion["rest_global_position_world"],
        motion["world_to_camera_cv"],
        motion["intrinsics"],
        json.loads((Path(record["render_dir"]) / "v000.json").read_text())["resolution"],
    )
    if not np.allclose(rest_uv, motion["rest_global_position_2d_px"], atol=1e-3):
        errors.append("rest-pose projection mismatch")
    if not np.allclose(rest_depth, motion["rest_depth_camera_cv"], atol=1e-5):
        errors.append("rest-pose depth mismatch")
    if not np.array_equal(rest_in_frame, motion["rest_in_frame"]):
        errors.append("rest-pose in-frame mask mismatch")

    return record, errors, {
        "orth_error": orth_error,
        "det_error": det_error,
        "projection_error_px": projection_error,
        "camera_inverse_error": camera_inverse_error,
        "in_frame_fraction": float(in_frame.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    record_paths = sorted((args.root / "records").glob("*.json"))
    failures = {}
    metrics = []
    rigs = set()
    for path in record_paths:
        record, errors, values = validate_record(path)
        rigs.add(record["rig_id"])
        metrics.append(values)
        if errors:
            failures[record["clip_id"]] = errors
    if args.expected is not None and len(record_paths) != args.expected:
        failures["__dataset__"] = [
            f"expected {args.expected} records, found {len(record_paths)}"
        ]
    summary = {
        "format": "skelmo_static_camera_validation_v1",
        "records": len(record_paths),
        "unique_rigs": len(rigs),
        "failures": failures,
        "max_orth_error": max((m["orth_error"] for m in metrics), default=0.0),
        "max_det_error": max((m["det_error"] for m in metrics), default=0.0),
        "max_projection_error_px": max(
            (m["projection_error_px"] for m in metrics), default=0.0
        ),
        "min_in_frame_fraction": min(
            (m["in_frame_fraction"] for m in metrics), default=0.0
        ),
    }
    (args.root / "validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
