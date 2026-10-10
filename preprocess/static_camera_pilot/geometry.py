"""Coordinate and rotation helpers for the static-camera dataset."""

from __future__ import annotations

import numpy as np


def zup_to_yup_matrix(dtype=np.float64):
    """Return the active -90 degree X rotation used by UniMate's exporter."""
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=dtype,
    )


def transform_points(transform, points):
    """Apply a homogeneous column-vector transform to ``[..., 3]`` points."""
    transform = np.asarray(transform)
    points = np.asarray(points)
    return points @ transform[:3, :3].T + transform[:3, 3]


def normalized_yup_camera(camera_to_world_zup, scene_scale, scene_offset_zup):
    """Build camera transforms for the normalized Y-up render world.

    Blender camera coordinates are x-right, y-up, z-back. ``camera_to_world_yup``
    preserves those camera axes and only changes the world basis. The OpenCV
    transform additionally flips camera y and z to y-down, z-forward.
    """
    axis = zup_to_yup_matrix()
    axis4 = np.eye(4, dtype=np.float64)
    axis4[:3, :3] = axis
    c2w_yup = axis4 @ np.asarray(camera_to_world_zup, dtype=np.float64)
    blender_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_cv = blender_to_cv @ np.linalg.inv(c2w_yup)

    forward = -c2w_yup[:3, 2]
    forward[1] = 0.0
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("Camera forward direction is parallel to gravity")
    z_axis = forward / norm
    y_axis = np.asarray([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    c2w_gravity = np.eye(4, dtype=np.float64)
    c2w_gravity[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    c2w_gravity[:3, 3] = c2w_yup[:3, 3]
    world_to_gravity = np.linalg.inv(c2w_gravity)

    offset_yup = axis @ np.asarray(scene_offset_zup, dtype=np.float64)
    source_yup_to_world = np.eye(4, dtype=np.float64)
    source_yup_to_world[:3, :3] *= float(scene_scale)
    source_yup_to_world[:3, 3] = float(scene_scale) * offset_yup
    return {
        "source_yup_to_world": source_yup_to_world,
        "camera_to_world_yup": c2w_yup,
        "world_to_camera_cv": world_to_cv,
        "camera_to_world_gravity": c2w_gravity,
        "world_to_camera_gravity": world_to_gravity,
    }


def pinhole_intrinsics(resolution, camera_angle_x):
    focal = 0.5 * float(resolution) / np.tan(0.5 * float(camera_angle_x))
    center = 0.5 * float(resolution)
    return np.asarray(
        [[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_world_points(points_world, world_to_camera_cv, intrinsics, resolution):
    points_cv = transform_points(world_to_camera_cv, points_world)
    depth = points_cv[..., 2]
    safe_depth = np.where(np.abs(depth) > 1e-8, depth, 1e-8)
    uv = np.empty(points_cv.shape[:-1] + (2,), dtype=np.float64)
    uv[..., 0] = intrinsics[0, 0] * points_cv[..., 0] / safe_depth + intrinsics[0, 2]
    uv[..., 1] = intrinsics[1, 1] * points_cv[..., 1] / safe_depth + intrinsics[1, 2]
    in_frame = (
        (depth > 0.0)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] < resolution)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] < resolution)
    )
    return uv, depth, in_frame


def quaternion_wxyz_to_matrix(quaternions):
    """Convert normalized-or-unnormalized wxyz quaternions to matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - z * w)
    result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w)
    result[..., 2, 1] = 2 * (y * z + x * w)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def matrix_to_rotation_6d_columns(matrices):
    """Store the first two matrix columns as ``[x_axis, y_axis]``."""
    matrices = np.asarray(matrices)
    return np.concatenate([matrices[..., :, 0], matrices[..., :, 1]], axis=-1)


def skinning_rotations(global_pose_wxyz, global_rest_wxyz, world_to_target_rotation):
    """Return target-frame deformation rotations and column-major 6D form."""
    pose = quaternion_wxyz_to_matrix(global_pose_wxyz)
    rest = quaternion_wxyz_to_matrix(global_rest_wxyz)
    deformation_world = pose @ np.swapaxes(rest, -1, -2)
    basis = np.asarray(world_to_target_rotation, dtype=np.float64)
    deformation_target = basis @ deformation_world @ basis.T
    return deformation_target, matrix_to_rotation_6d_columns(deformation_target)


def skinning_rotations_from_matrices(global_pose, global_rest, world_to_target_rotation):
    """Matrix-input variant for evaluated Blender world-space transforms."""
    pose = np.asarray(global_pose, dtype=np.float64)
    rest = np.asarray(global_rest, dtype=np.float64)
    deformation_world = pose @ np.swapaxes(rest, -1, -2)
    basis = np.asarray(world_to_target_rotation, dtype=np.float64)
    deformation_target = basis @ deformation_world @ basis.T
    return deformation_target, matrix_to_rotation_6d_columns(deformation_target)


def primary_bone_axes(rest_positions, parents):
    """Choose a stable rest-space axis per joint for swing/twist supervision."""
    positions = np.asarray(rest_positions, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    axes = np.zeros_like(positions)
    for joint, child_ids in enumerate(children):
        if child_ids:
            candidates = positions[child_ids] - positions[joint]
            axis = candidates[np.argmax(np.linalg.norm(candidates, axis=-1))]
        elif parents[joint] >= 0:
            axis = positions[joint] - positions[parents[joint]]
        else:
            axis = np.asarray([1.0, 0.0, 0.0])
        length = np.linalg.norm(axis)
        axes[joint] = axis / length if length > 1e-8 else (1.0, 0.0, 0.0)
    return axes


def contact_heuristic(positions_world, parents, ground_y=-0.5):
    """Conservative leaf-contact labels; intended as a weak auxiliary target."""
    positions = np.asarray(positions_world, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    is_leaf = np.ones(len(parents), dtype=bool)
    is_leaf[parents[parents >= 0]] = False
    leaf_ids = np.flatnonzero(is_leaf)
    contacts = np.zeros(positions.shape[:2], dtype=bool)
    if len(leaf_ids) == 0 or len(positions) < 2:
        return contacts, is_leaf

    leaf_y = positions[:, leaf_ids, 1]
    leaf_min = leaf_y.min(axis=0)
    eligible = leaf_min <= max(float(ground_y) + 0.15, float(leaf_min.min()) + 0.12)
    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    velocity = np.concatenate([velocity[:1], velocity], axis=0)
    near_floor = leaf_y <= np.maximum(leaf_min[None] + 0.025, float(ground_y) + 0.04)
    contacts[:, leaf_ids] = near_floor & (velocity[:, leaf_ids] <= 0.025) & eligible[None]
    return contacts, is_leaf
