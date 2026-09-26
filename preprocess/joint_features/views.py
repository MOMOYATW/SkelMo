"""Shared camera definitions for the rest-pose DINO pipeline.

The module deliberately has no Blender, PyTorch, or NumPy dependency, so the
renderer, projection exporter, feature extractor, and host-side driver can all
use exactly the same ordered view set.
"""

from __future__ import annotations

from collections.abc import Iterable


VIEW_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "diag_nnn": (-1.0, -1.0, -1.0),
    "diag_nnp": (-1.0, -1.0, 1.0),
    "diag_npn": (-1.0, 1.0, -1.0),
    "diag_npp": (-1.0, 1.0, 1.0),
    "diag_pnn": (1.0, -1.0, -1.0),
    "diag_pnp": (1.0, -1.0, 1.0),
    "diag_ppn": (1.0, 1.0, -1.0),
    "diag_ppp": (1.0, 1.0, 1.0),
}

DEFAULT_VIEW_NAMES: tuple[str, ...] = tuple(VIEW_DIRECTIONS)


def parse_view_names(value: str | Iterable[str]) -> list[str]:
    """Return a validated, ordered list of camera names."""
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        raise ValueError("At least one rest-pose view is required")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate rest-pose views: {', '.join(duplicates)}")
    unknown = [name for name in names if name not in VIEW_DIRECTIONS]
    if unknown:
        raise ValueError(f"Unknown rest-pose views: {', '.join(unknown)}")
    return names
