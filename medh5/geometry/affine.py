"""Index <-> world geometry (spec §3.3).

Continuous index coordinates put voxel centres at integers, so a voxel spans
``[i - 0.5, i + 0.5]`` and a box ``[a, b]`` covers the slice ``a+0.5 : b+0.5``.
Every function here is the engine's (``medh5._core``).
"""

from __future__ import annotations

from medh5._core import (
    ORTHONORMAL_TOL,
    affine_summary,
    apply_affine_to_box,
    box_corners,
    box_to_slices,
    build_affine,
    check_orthonormal,
    decompose_affine,
    index_to_world,
    is_orthonormal,
    is_proper_rotation,
    slices_to_box,
    voxel_volume,
    world_to_index,
)

__all__ = [
    "ORTHONORMAL_TOL",
    "affine_summary",
    "apply_affine_to_box",
    "box_corners",
    "box_to_slices",
    "build_affine",
    "check_orthonormal",
    "decompose_affine",
    "index_to_world",
    "is_orthonormal",
    "is_proper_rotation",
    "slices_to_box",
    "voxel_volume",
    "world_to_index",
]
