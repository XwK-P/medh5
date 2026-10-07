"""Multiscale pyramids (spec §4.3): one image, several levels, one geometry rule.

A level's spacing is level 0's times its factor and its origin carries the
half-voxel shift; ``derive_level_grid`` applies the rule and ``check_pyramid``
reports every level that breaks it.  Both are the engine's (``medh5._core``).
"""

from __future__ import annotations

from medh5._core import (
    DOWNSAMPLE_METHODS,
    GEOMETRY_RTOL,
    LABEL_SAFE_METHODS,
    Pyramid,
    check_pyramid,
    derive_level_grid,
    pyramid_factors,
)

__all__ = [
    "DOWNSAMPLE_METHODS",
    "GEOMETRY_RTOL",
    "LABEL_SAFE_METHODS",
    "Pyramid",
    "check_pyramid",
    "derive_level_grid",
    "pyramid_factors",
]
