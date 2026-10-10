"""Build aligned static-camera renders and deformation-space motion labels.

Run this with UniMate's Python environment because rendering uses ``bpy``.
The UniMate checkout is imported but never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

from .geometry import (
    contact_heuristic,
    matrix_to_rotation_6d_columns,
    normalized_yup_camera,
    pinhole_intrinsics,
    primary_bone_axes,
    project_world_points,
    skinning_rotations_from_matrices,
    transform_points,
    zup_to_yup_matrix,
)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _read_selection(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _select_action(row, action_names):
    index = int(row["clip_index"])
    expected = _normalized_name(row.get("clip_name", ""))
    name_matches = [
        name for name in action_names
        if expected and (
            expected in _normalized_name(name)
            or _normalized_name(name) in expected
        )
    ]
    # glTF import can split one source animation into separate object actions.
    # In that case metadata clip_index addresses the unsplit source list and
    # no longer indexes discover_pose_actions(); the source clip name remains
    # the reliable key.
    if len(name_matches) == 1:
        return name_matches[0]
    if not 0 <= index < len(action_names):
        raise IndexError(
            f"clip_index={index}, pose_actions={len(action_names)}, "
            f"name_matches={name_matches}"
        )
    selected = action_names[index]
    actual = _normalized_name(selected)
    if expected and expected not in actual and actual not in expected:
        raise ValueError(
            f"Animation metadata mismatch at index {index}: "
            f"CSV={row.get('clip_name')!r}, Blender={selected!r}, "
            f"name_matches={name_matches}"
        )
    return selected


def deterministic_camera_azimuth(clip_id, seed, minimum=-180.0, maximum=180.0):
    """Map a clip id to a reproducible uniform static-camera yaw in degrees."""
    if not minimum < maximum:
        raise ValueError("camera azimuth minimum must be smaller than maximum")
    payload = f"{int(seed)}|{clip_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    unit = integer / float(2**64)
    return float(minimum + unit * (maximum - minimum))


def _sample_anchors(mesh, skin_matrix, count):
    vertices = mesh.data.vertices
    if not vertices or skin_matrix.shape[0] != len(vertices):
        return {
            "anchor_vertex_indices": np.zeros(0, dtype=np.int32),
            "anchor_positions_source_yup": np.zeros((0, 3), dtype=np.float32),
            "anchor_joint_indices": np.zeros((0, 4), dtype=np.int32),
            "anchor_joint_weights": np.zeros((0, 4), dtype=np.float32),
        }
    count = min(int(count), len(vertices))
    indices = np.unique(np.linspace(0, len(vertices) - 1, count).round().astype(np.int32))
    local = np.asarray([vertices[int(i)].co[:] for i in indices], dtype=np.float64)
    world_zup = local @ np.asarray(mesh.matrix_world)[:3, :3].T + np.asarray(mesh.matrix_world)[:3, 3]
    source_yup = world_zup @ zup_to_yup_matrix().T
    weights = skin_matrix[indices]
    width = min(4, weights.shape[1])
    top = np.argsort(weights, axis=1)[:, -width:][:, ::-1]
    top_weights = np.take_along_axis(weights, top, axis=1)
    if width < 4:
        top = np.pad(top, ((0, 0), (0, 4 - width)))
        top_weights = np.pad(top_weights, ((0, 0), (0, 4 - width)))
    denom = top_weights.sum(axis=1, keepdims=True)
    top_weights = np.divide(top_weights, denom, out=np.zeros_like(top_weights), where=denom > 1e-8)
    return {
        "anchor_vertex_indices": indices,
        "anchor_positions_source_yup": source_yup.astype(np.float32),
        "anchor_joint_indices": top.astype(np.int32),
        "anchor_joint_weights": top_weights.astype(np.float32),
    }


def _extract_evaluated_globals(armature, joint_names, start_frame, end_frame):
    """Read bone heads/orientations in the exact Blender world used to render.

    UniMate's portable motion exporter intentionally removes the armature
    object's scale. That is useful for retargeting, but it makes its FK joint
    positions disagree with pixels for scaled GLBs. Dataset labels therefore
    come from evaluated pose matrices, including the full object transform.
    """
    import bpy

    axis = zup_to_yup_matrix()
    frame_count = end_frame - start_frame + 1
    joint_count = len(joint_names)
    positions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    rotations = np.zeros((frame_count, joint_count, 3, 3), dtype=np.float64)
    for frame_index, frame in enumerate(range(start_frame, end_frame + 1)):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for joint_index, name in enumerate(joint_names):
            matrix_zup = armature.matrix_world @ armature.pose.bones[str(name)].matrix
            positions[frame_index, joint_index] = axis @ np.asarray(matrix_zup.to_translation())
            rotation_zup = np.asarray(matrix_zup.to_quaternion().to_matrix())
            rotations[frame_index, joint_index] = axis @ rotation_zup @ axis.T

    rest_positions = np.zeros((joint_count, 3), dtype=np.float64)
    rest_rotations = np.zeros((joint_count, 3, 3), dtype=np.float64)
    for joint_index, name in enumerate(joint_names):
        matrix_zup = armature.matrix_world @ armature.data.bones[str(name)].matrix_local
        rest_positions[joint_index] = axis @ np.asarray(matrix_zup.to_translation())
        rotation_zup = np.asarray(matrix_zup.to_quaternion().to_matrix())
        rest_rotations[joint_index] = axis @ rotation_zup @ axis.T
    return positions, rotations, rest_positions, rest_rotations


def _render_aligned_rest_pose(render_dir, args, api):
    """Render true REST using the action render's existing camera/normalization."""
    import bpy

    armatures = api["get_scene_armatures"]()
    previous_modes = [armature.data.pose_position for armature in armatures]
    try:
        for armature in armatures:
            armature.data.pose_position = "REST"
            armature.data.update_tag()
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get()
        api["enable_color_output"](
            args.resolution,
            str(render_dir),
            mode="IMAGE",
            image_format="WEBP",
            image_quality=args.image_quality,
        )
        rest_path = render_dir / "rest.webp"
        api["render_still"](str(rest_path))
    finally:
        for armature, mode in zip(armatures, previous_modes):
            armature.data.pose_position = mode
            armature.data.update_tag()
        bpy.context.view_layer.update()
    return rest_path


def _build_one(row, args, api):
    clip_id = row["clip_id"]
    rig_id = row["rig_id"]
    render_dir = args.output / "renders" / clip_id
    motion_path = args.output / "motions" / f"{clip_id}.npz"
    rig_path = args.output / "rigs" / f"{rig_id}.npz"
    record_path = args.output / "records" / f"{clip_id}.json"
    rest_render_path = render_dir / "rest.webp"
    if (
        record_path.is_file()
        and motion_path.is_file()
        and rig_path.is_file()
        and rest_render_path.is_file()
        and not args.overwrite
    ):
        return "skipped"

    asset_path = args.dataset_root / row["file"]
    armature = api["load_and_prep_asset"](str(asset_path), fps=args.fps)
    if armature is None:
        raise ValueError("Asset has no renderable armature/mesh")
    action_names, frame_ranges = api["discover_pose_actions"]()
    action_name = _select_action(row, action_names)
    start_frame, raw_end_frame = frame_ranges[action_name]
    end_frame = min(raw_end_frame, start_frame + args.max_frames - 1)
    camera_azimuth_deg = deterministic_camera_azimuth(
        clip_id,
        args.camera_seed,
        args.camera_azimuth_min,
        args.camera_azimuth_max,
    )
    view_name = "random_yaw"
    # UniMate exposes named views; registering one in-memory keeps that repo
    # untouched while reusing its tested normalization/render implementation.
    api["blender_render"].VIEW_AZIMUTH_BY_NAME[view_name] = camera_azimuth_deg

    meshes = api["find_meshes"]()
    mesh = api["choose_main_mesh"](meshes)
    if mesh is None:
        raise ValueError("Asset has no mesh after render preparation")
    skel = api["prepare_skeleton"](armature, mesh, dtype=np.float64, apply_world=True)
    extracted = api["extract_all_actions"](
        armature,
        {action_name: (start_frame, end_frame)},
        skel["rest_local_pos"],
        skel["parents_array"],
        skel["rest_anim"],
        skel["nbones"],
        dtype=np.float64,
    )
    if len(extracted) != 1:
        raise RuntimeError(f"Expected one extracted action, got {len(extracted)}")
    _, anim = extracted[0]
    rest_anim = skel["rest_anim_shared"]
    names = np.asarray(skel["bone_names"])
    parents = np.asarray(anim.parents, dtype=np.int32)
    anchors = _sample_anchors(mesh, skel["skin_matrix"], args.anchor_count)
    (source_positions, pose_global_rotmat, rest_source_positions,
     rest_global_rotmat) = _extract_evaluated_globals(
        armature, names, start_frame, end_frame
    )

    # Rendering uses the identical inclusive source frame range. We do not
    # remove rest-looking frames, so frame k in the labels is frame k onscreen.
    api["render_action_multiview"](
        armature=armature,
        action_name=action_name,
        start_frame=start_frame,
        end_frame=end_frame,
        save_root=str(render_dir),
        camera_dist=args.camera_dist,
        scene_scale=args.scene_scale,
        resolution=args.resolution,
        views=[view_name],
        render_video=False,
        render_frames=True,
        compose_video=args.video,
        fps=args.fps,
        max_render_frames=args.max_frames,
        save_camera_metadata=True,
        image_format="WEBP",
        image_quality=args.image_quality,
    )
    # Do not append REST to the action frames or inspection MP4. It is a
    # separate conditioning image so motion frame counts remain unambiguous.
    rest_render_path = _render_aligned_rest_pose(render_dir, args, api)
    camera_meta = json.loads((render_dir / "v000.json").read_text())
    transforms = normalized_yup_camera(
        camera_meta["transform_matrix"],
        camera_meta["scene_scale"],
        camera_meta["scene_offset"],
    )
    intrinsics = pinhole_intrinsics(args.resolution, camera_meta["camera_angle_x"])

    world_positions = transform_points(transforms["source_yup_to_world"], source_positions)
    gravity_positions = transform_points(transforms["world_to_camera_gravity"], world_positions)
    uv, depth, in_frame = project_world_points(
        world_positions, transforms["world_to_camera_cv"], intrinsics, args.resolution
    )
    skin_rotmat, skin_rot6d = skinning_rotations_from_matrices(
        pose_global_rotmat,
        rest_global_rotmat,
        transforms["world_to_camera_gravity"][:3, :3],
    )
    rest_world_positions = transform_points(
        transforms["source_yup_to_world"], rest_source_positions
    )
    rest_gravity_positions = transform_points(
        transforms["world_to_camera_gravity"], rest_world_positions
    )
    rest_uv, rest_depth, rest_in_frame = project_world_points(
        rest_world_positions,
        transforms["world_to_camera_cv"],
        intrinsics,
        args.resolution,
    )
    axes_source = primary_bone_axes(rest_source_positions, parents)
    axes_gravity = axes_source @ transforms["world_to_camera_gravity"][:3, :3].T
    posed_axes_gravity = np.einsum("tjik,jk->tji", skin_rotmat, axes_gravity)
    contacts, leaf_mask = contact_heuristic(
        world_positions, parents, ground_y=-0.5 * float(args.scene_scale)
    )

    anchor_world = transform_points(
        transforms["source_yup_to_world"], anchors["anchor_positions_source_yup"]
    )
    _atomic_npz(
        rig_path,
        format=np.asarray("skelmo_static_rig_v1"),
        joint_names=names,
        parents=parents,
        rest_local_position_export_yup=rest_anim.positions[0].astype(np.float32),
        rest_local_rotation_export_yup_wxyz=rest_anim.rotations.qs[0].astype(np.float32),
        rest_global_position_source_yup=rest_source_positions.astype(np.float32),
        rest_global_rotation_source_yup_matrix=rest_global_rotmat.astype(np.float32),
        primary_bone_axis_source_yup=axes_source.astype(np.float32),
        leaf_mask=leaf_mask,
        **anchors,
    )
    _atomic_npz(
        motion_path,
        format=np.asarray("skelmo_static_motion_v2"),
        fps=np.asarray(args.fps, dtype=np.int32),
        camera_azimuth_deg=np.asarray(camera_azimuth_deg, dtype=np.float32),
        frame_start=np.asarray(start_frame, dtype=np.int32),
        frame_end=np.asarray(end_frame, dtype=np.int32),
        source_yup_to_world=transforms["source_yup_to_world"].astype(np.float32),
        camera_to_world_yup=transforms["camera_to_world_yup"].astype(np.float32),
        world_to_camera_cv=transforms["world_to_camera_cv"].astype(np.float32),
        camera_to_world_gravity=transforms["camera_to_world_gravity"].astype(np.float32),
        world_to_camera_gravity=transforms["world_to_camera_gravity"].astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        joint_position_world=world_positions.astype(np.float32),
        joint_position_gravity_camera=gravity_positions.astype(np.float32),
        joint_position_2d_px=uv.astype(np.float32),
        joint_depth_camera_cv=depth.astype(np.float32),
        joint_in_frame=in_frame,
        local_rotation_export_yup_wxyz=anim.rotations.qs.astype(np.float32),
        global_rotation_source_yup_matrix=pose_global_rotmat.astype(np.float32),
        skin_rotation_gravity_camera_matrix=skin_rotmat.astype(np.float32),
        skin_rotation_gravity_camera_6d=skin_rot6d.astype(np.float32),
        primary_bone_axis_gravity_camera=axes_gravity.astype(np.float32),
        posed_bone_axis_gravity_camera=posed_axes_gravity.astype(np.float32),
        contact_heuristic=contacts,
        rest_global_position_world=rest_world_positions.astype(np.float32),
        rest_global_position_gravity_camera=rest_gravity_positions.astype(np.float32),
        rest_global_position_2d_px=rest_uv.astype(np.float32),
        rest_depth_camera_cv=rest_depth.astype(np.float32),
        rest_in_frame=rest_in_frame,
        anchor_position_world=anchor_world.astype(np.float32),
    )
    frame_count = end_frame - start_frame + 1
    rendered_frames = len(list((render_dir / "v000").glob("*.webp")))
    if rendered_frames != frame_count or len(anim.positions) != frame_count:
        raise RuntimeError(
            f"Frame mismatch: labels={len(anim.positions)}, render={rendered_frames}, expected={frame_count}"
        )
    record = {
        **row,
        "format": "skelmo_static_camera_record_v2",
        "asset_path": str(asset_path),
        "action_name_blender": action_name,
        "frame_start": start_frame,
        "frame_end": end_frame,
        "frame_count": frame_count,
        "joint_count": len(names),
        "render_dir": str(render_dir),
        "rest_render_path": str(rest_render_path),
        "motion_path": str(motion_path),
        "rig_path": str(rig_path),
        "rotation_target": "global_skinning_deformation_in_gravity_camera",
        "rotation_6d_layout": "first_column_xyz_then_second_column_xyz",
        "visibility_target": "in_frame_only_no_occlusion_test",
        "camera_motion": "none",
        "camera": "static_random_yaw",
        "camera_seed": args.camera_seed,
        "camera_azimuth_deg": camera_azimuth_deg,
        "camera_elevation_deg": 0.0,
    }
    _atomic_json(record_path, record)
    return "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--unimate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker-id", default=0, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--max-frames", default=128, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--samples", default=32, type=int)
    parser.add_argument("--image-quality", default=90, type=int)
    parser.add_argument("--camera-dist", default=1.5, type=float)
    parser.add_argument("--camera-seed", default=20261010, type=int)
    parser.add_argument("--camera-azimuth-min", default=-180.0, type=float)
    parser.add_argument("--camera-azimuth-max", default=180.0, type=float)
    parser.add_argument("--scene-scale", default=1.0, type=float)
    parser.add_argument("--anchor-count", default=512, type=int)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        parser.error("worker-id must satisfy 0 <= worker-id < num-workers")
    if args.camera_azimuth_min >= args.camera_azimuth_max:
        parser.error("camera-azimuth-min must be smaller than camera-azimuth-max")

    sys.path.insert(0, str(args.unimate_root.resolve()))
    from data_process.utils.blender_export import (
        choose_main_mesh,
        discover_pose_actions,
        extract_all_actions,
        prepare_skeleton,
    )
    from data_process.utils import blender_render
    from data_process.utils.blender_render import (
        enable_color_output,
        find_meshes,
        get_scene_armatures,
        init_render_engine,
        load_and_prep_asset,
        render_action_multiview,
        render_still,
    )

    api = locals()
    init_render_engine(render_samples=args.samples)
    rows = _read_selection(args.selection)[args.worker_id :: args.num_workers]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output / "build_config.json",
        {
            "format": "skelmo_static_camera_build_config_v2",
            "selection": str(args.selection.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "unimate_root": str(args.unimate_root.resolve()),
            "fps": args.fps,
            "max_frames": args.max_frames,
            "resolution": args.resolution,
            "samples": args.samples,
            "camera": "static_random_yaw",
            "camera_seed": args.camera_seed,
            "camera_azimuth_range_deg": [
                args.camera_azimuth_min,
                args.camera_azimuth_max,
            ],
            "camera_elevation_deg": 0.0,
            "background": "transparent",
            "rest_conditioning": "separate_same_camera_same_normalization",
        },
    )
    counts = {"complete": 0, "skipped": 0, "failed": 0}
    for index, row in enumerate(rows, 1):
        failure_path = args.output / "failures" / f"{row.get('clip_id', index)}.json"
        try:
            status = _build_one(row, args, api)
            counts[status] += 1
            failure_path.unlink(missing_ok=True)
            print(f"[{index}/{len(rows)}] {status}: {row['clip_id']}", flush=True)
        except Exception as error:
            counts["failed"] += 1
            failure = {
                "clip_id": row.get("clip_id"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(failure_path, failure)
            print(f"[{index}/{len(rows)}] failed: {row.get('clip_id')}: {error}", flush=True)
            if args.fail_fast:
                raise
    print(json.dumps(counts, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
