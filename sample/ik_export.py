"""Fixed-length IK and BVH export for generated skeleton motion."""

from __future__ import annotations

import re

import numpy as np

import Animation
import BVH
from InverseKinematics import BasicInverseKinematics


def safe_joint_names(names):
    """Return unique BVH-safe names without changing the joint order."""
    result = []
    counts = {}
    for index, value in enumerate(np.asarray(names).astype(str)):
        name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
        if not name:
            name = f"joint_{index}"
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        result.append(name if occurrence == 0 else f"{name}__{occurrence}")
    return np.asarray(result)


def fit_animation(positions, parents, offsets, iterations):
    """Project predicted XYZ onto the fixed offsets of the supplied skeleton."""
    if np.flatnonzero(parents < 0).tolist() != [0]:
        raise ValueError("BVH export requires exactly one root at joint 0")
    animation, order, sorted_parents = Animation.animation_from_offsets(
        offsets, parents, positions.shape
    )
    order = np.asarray(order)
    sorted_positions = positions[:, order]
    sorted_root = int(np.flatnonzero(sorted_parents < 0)[0])
    animation.positions[:, sorted_root] = sorted_positions[:, sorted_root]

    best_animation = animation.copy()
    best_error = float("inf")
    for _ in range(max(int(iterations), 1)):
        animation = BasicInverseKinematics(
            animation, sorted_positions, iterations=1, silent=True
        )()
        fitted = Animation.positions_global(animation)
        error = float(np.linalg.norm(fitted - sorted_positions, axis=-1).mean())
        if error < best_error:
            best_error = error
            best_animation = animation.copy()
    fitted = Animation.positions_global(best_animation)
    return best_animation, order, fitted


def save_and_verify(path, animation, names, fps):
    """Write BVH and measure the joint-position error after reloading it."""
    BVH.save(str(path), animation, names, frametime=1.0 / float(fps))
    loaded, loaded_names, frametime = BVH.load(str(path))
    loaded_positions = Animation.positions_global(loaded)
    expected_positions = Animation.positions_global(animation)
    expected_by_name = {name: i for i, name in enumerate(names)}
    expected_indices = [expected_by_name[name] for name in loaded_names]
    if loaded_positions.shape[0] != expected_positions.shape[0]:
        raise ValueError(f"BVH frame count changed after reload: {path}")
    roundtrip = np.linalg.norm(
        loaded_positions - expected_positions[:, expected_indices], axis=-1
    )
    return {
        "frames": int(loaded_positions.shape[0]),
        "exported_joints": int(len(names)),
        "reloaded_joints": int(len(loaded_names)),
        "frametime": float(frametime),
        "roundtrip_mpjpe_cm": float(roundtrip.mean() * 100.0),
        "roundtrip_max_cm": float(roundtrip.max() * 100.0),
    }
