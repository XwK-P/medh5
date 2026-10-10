"""Geometry: grids, the index-world affine, and multiscale pyramids (spec §3, §4.3).

A grid is an **empty HDF5 group carrying only attributes**.  That makes grids
essentially free, which is what lets the format say "two acquisitions with the
same lattice in different timepoints are two grids, not one shared grid" ---
keeping ``timepoint`` and ``frame_uid`` single-valued per grid and removing any
need for per-image geometry overrides.  Grids are written with
``SampleWriter.add_grid`` and read from ``Sample.grids``, or from a stored group
with :func:`read_grid`.

Continuous index coordinates put voxel centres at integers, so a voxel spans
``[i - 0.5, i + 0.5]`` and a box ``[a, b]`` covers the slice ``a+0.5 : b+0.5``
(§3.3).  A pyramid level's spacing is level 0's times its factor and its origin
carries the half-voxel shift (§4.3): ``derive_level_grid`` applies the rule and
``check_pyramid`` reports every level that breaks it.

Everything here is the format engine's (``medh5._core``): :class:`Grid`'s checks
--- orthonormal direction, positive spacing, axis kinds --- and every index/world
mapping, so every frontend computes the same geometry.
"""

from __future__ import annotations

from medh5._core import (
    AXIS_KINDS,
    DOWNSAMPLE_METHODS,
    GEOMETRY_RTOL,
    KNOWN_UNITS,
    LABEL_SAFE_METHODS,
    ORTHONORMAL_TOL,
    SPEC_GRID_ATTRS,
    TIME_UNITS,
    Grid,
    Pyramid,
    affine_summary,
    apply_affine_to_box,
    box_corners,
    box_to_slices,
    build_affine,
    check_orthonormal,
    check_pyramid,
    decompose_affine,
    derive_level_grid,
    index_to_world,
    is_orthonormal,
    is_proper_rotation,
    pyramid_factors,
    read_grid,
    read_grids,
    slices_to_box,
    voxel_volume,
    world_to_index,
)

MIN_SPATIAL, MAX_SPATIAL = 2, 3

__all__ = [
    "AXIS_KINDS",
    "DOWNSAMPLE_METHODS",
    "GEOMETRY_RTOL",
    "KNOWN_UNITS",
    "LABEL_SAFE_METHODS",
    "MAX_SPATIAL",
    "MIN_SPATIAL",
    "ORTHONORMAL_TOL",
    "SPEC_GRID_ATTRS",
    "TIME_UNITS",
    "Grid",
    "Pyramid",
    "affine_summary",
    "apply_affine_to_box",
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
    "pyramid_factors",
    "read_grid",
    "read_grids",
    "slices_to_box",
    "voxel_volume",
    "world_to_index",
]
