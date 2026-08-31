"""Multiscale pyramids (spec §4.3).

Every level has its **own grid**, and the geometry of level *l* is derived from
level 0 by a rule the validator checks to 1e-3 relative tolerance::

    spacing'_k = spacing_k * f_k
    origin'    = origin + direction @ (spacing * ((f - 1) / 2))

The half-voxel term is the whole point.  Downsampling by *f* merges *f* voxels
into one whose centre sits ``(f-1)/2`` voxels further along the axis; a writer
that copies the origin unchanged shifts every level by up to half a low-res
voxel relative to level 0, which is invisible in a viewer and fatal when a
model trained at level 2 has its predictions mapped back to level 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import ORTHONORMAL_TOL
from medh5.geometry.grid import Grid

DOWNSAMPLE_METHODS = ("mean", "nearest", "gaussian", "max")
LABEL_SAFE_METHODS = ("nearest", "max")
"""Methods that preserve class identity; ``mean`` on a labelmap invents classes."""

GEOMETRY_RTOL = 1e-3


@dataclass(frozen=True, slots=True)
class Pyramid:
    """The declaration on an ``images/<id>`` group that holds multiple levels."""

    levels: int
    downsample_factors: npt.NDArray[np.float64]
    downsample_method: str
    grid_levels: tuple[str, ...]

    def __post_init__(self) -> None:
        factors = np.asarray(self.downsample_factors, dtype=np.float64)
        if factors.ndim != 2 or factors.shape[0] != self.levels:
            raise MEDH5ValidationError(
                f"downsample_factors must be (levels, S); got {factors.shape} "
                f"for levels={self.levels}",
                code="E105",
            )
        if len(self.grid_levels) != self.levels:
            raise MEDH5ValidationError(
                f"grid_levels has {len(self.grid_levels)} entries for "
                f"levels={self.levels}",
                code="E105",
            )
        if self.downsample_method not in DOWNSAMPLE_METHODS:
            raise MEDH5ValidationError(
                f"unknown downsample_method {self.downsample_method!r}", code="E105"
            )
        if np.any(factors <= 0):
            raise MEDH5ValidationError("downsample factors must be > 0", code="E105")
        if not np.allclose(factors[0], 1.0):
            raise MEDH5ValidationError(
                "level 0 must have downsample factor 1 on every axis", code="E105"
            )
        object.__setattr__(self, "downsample_factors", factors)

    def attrs(self) -> dict[str, Any]:
        return {
            "levels": int(self.levels),
            "downsample_factors": self.downsample_factors,
            "downsample_method": self.downsample_method,
            "grid_levels": list(self.grid_levels),
        }


def derive_level_grid(
    base: Grid,
    factors: Sequence[float],
    grid_id: str,
    *,
    shape: Sequence[int] | None = None,
) -> Grid:
    """Derive the grid of one pyramid level from level 0 (spec §4.3).

    *shape* defaults to ``ceil(shape / f)``, which keeps the full field of view
    covered.  A writer that downsamples differently passes its own shape; only
    the spacing/origin relation is normative.
    """
    f = np.asarray(factors, dtype=np.float64)
    if f.shape != (base.n_spatial,):
        raise MEDH5ValidationError(
            f"factors must have {base.n_spatial} entries, got {f.shape}", code="E105"
        )
    spacing = np.asarray(base.spacing, dtype=np.float64) * f
    shift = base.direction @ (
        np.asarray(base.spacing, dtype=np.float64) * (f - 1.0) / 2.0
    )
    origin = np.asarray(base.origin, dtype=np.float64) + shift
    if shape is None:
        spatial = tuple(
            max(1, math.ceil(n / factor))
            for n, factor in zip(base.spatial_shape, f, strict=True)
        )
        lead = base.shape[: base.ndim - base.n_spatial]
        full_shape: tuple[int, ...] = tuple(lead) + spatial
    else:
        full_shape = tuple(int(s) for s in shape)
    return Grid(
        grid_id=grid_id,
        shape=full_shape,
        axis_names=base.axis_names,
        axis_kinds=base.axis_kinds,
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in origin),
        direction=base.direction,
        coord_system=base.coord_system,
        units=base.units,
        timepoint=base.timepoint,
        frame_uid=base.frame_uid,
        time_values=base.time_values,
        time_units=base.time_units,
    )


def check_pyramid(
    base: Grid,
    levels: Sequence[Grid],
    factors: npt.ArrayLike,
    *,
    rtol: float = GEOMETRY_RTOL,
) -> list[str]:
    """Check every level's geometry against the derivation rule.

    Returns human-readable messages; an empty list means the pyramid conforms.
    Used by the validator to raise E105 with a message that names the axis.
    """
    problems: list[str] = []
    f_all = np.asarray(factors, dtype=np.float64)
    for level, (grid, f) in enumerate(zip(levels, f_all, strict=True)):
        expected = derive_level_grid(base, f, grid.grid_id, shape=grid.shape)
        for axis in range(base.n_spatial):
            got_s, want_s = grid.spacing[axis], expected.spacing[axis]
            if not math.isclose(got_s, want_s, rel_tol=rtol, abs_tol=0.0):
                problems.append(
                    f"level {level} axis {axis}: spacing {got_s:g} != "
                    f"{want_s:g} (= spacing0 * {f[axis]:g})"
                )
            got_o, want_o = grid.origin[axis], expected.origin[axis]
            scale = max(abs(want_o), float(base.spacing[axis]))
            if abs(got_o - want_o) > rtol * scale:
                problems.append(
                    f"level {level} axis {axis}: origin {got_o:g} != {want_o:g} "
                    "(half-voxel shift missing?)"
                )
        # `direction` is checked against the base, not against `expected`:
        # `derive_level_grid` builds the level *from* the base's direction, so
        # comparing the two would compare a value with itself and pass for any
        # level whatsoever.  A level whose axes are permuted relative to level 0
        # is the failure this module's docstring describes -- predictions made
        # at level 2 mapped back to level 0 land transposed -- and it passed
        # both existing checks, because the expected origin was computed from
        # the base's direction rather than the level's.
        if not np.allclose(grid.direction, base.direction, atol=ORTHONORMAL_TOL):
            problems.append(
                f"level {level}: direction differs from level 0; a pyramid "
                "level shares its parent's orientation, only its sampling changes"
            )
        if tuple(grid.axis_kinds) != tuple(base.axis_kinds):
            problems.append(f"level {level}: axis_kinds differ from level 0")
        # Bounded, not equal.  `derive_level_grid` was handed `shape=grid.shape`
        # above, so `expected.shape` is the level's own shape and comparing the
        # two compares a value with itself.  The rule cannot be exact equality
        # either: only the spacing/origin relation is normative, and a writer
        # may round a level's extent down where this module rounds up.  So the
        # bound is one voxel either side of n/f, which admits both conventions
        # and still rejects a level that is not a resampling of its parent at
        # all -- (3, 3, 3) at factor 2 of a 16-voxel axis.
        for axis in range(base.n_spatial):
            n = base.spatial_shape[axis]
            got_n = grid.spatial_shape[axis]
            low = max(1, math.floor(n / f[axis]))
            high = max(1, math.ceil(n / f[axis]))
            if not low <= got_n <= high:
                problems.append(
                    f"level {level} axis {axis}: extent {got_n} is not a "
                    f"factor-{f[axis]:g} resampling of {n} (expected "
                    f"{low} or {high})"
                )
        if grid.coord_system != base.coord_system or grid.units != base.units:
            problems.append(f"level {level}: coord_system/units differ from level 0")
        if grid.frame_uid != base.frame_uid:
            problems.append(f"level {level}: frame_uid differs from level 0")
    return problems


def pyramid_factors(base: Grid, levels: Sequence[Grid]) -> npt.NDArray[np.float64]:
    """Recover per-level downsample factors from the levels' spacings."""
    base_spacing = np.asarray(base.spacing, dtype=np.float64)
    return np.stack(
        [np.asarray(g.spacing, dtype=np.float64) / base_spacing for g in levels]
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
