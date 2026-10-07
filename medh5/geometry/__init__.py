"""Geometry: grids, the index-world affine, and multiscale pyramids (spec §3, §4.3)."""

from __future__ import annotations

from medh5.geometry.affine import (
    box_corners,
    box_to_slices,
    build_affine,
    check_orthonormal,
    decompose_affine,
    index_to_world,
    is_orthonormal,
    is_proper_rotation,
    slices_to_box,
    world_to_index,
)
from medh5.geometry.grid import AXIS_KINDS, Grid
from medh5.geometry.multiscale import (
    Pyramid,
    check_pyramid,
    derive_level_grid,
)

__all__ = [
    "AXIS_KINDS",
    "Grid",
    "Pyramid",
    "box_corners",
    "box_to_slices",
    "build_affine",
    "check_orthonormal",
    "check_pyramid",
    "decompose_affine",
    "derive_level_grid",
    "index_to_world",
    "is_orthonormal",
    "is_proper_rotation",
    "slices_to_box",
    "world_to_index",
]
