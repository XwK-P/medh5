"""``displacement`` transforms: a dense vector field on a grid (spec §10.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry.grid import Grid
from medh5.transforms.base import Transform

FLOAT16_SAFE_VOXELS: float = _core.FLOAT16_SAFE_VOXELS


def encode_displacement(
    field: npt.ArrayLike,
    *,
    field_grid: str,
    vector_space: str = "world",
    interpolation: str = "linear",
    extrapolation: str = "zero",
    dtype: npt.DTypeLike = np.float32,
) -> Any:
    """Pack a ``(S, *spatial)`` displacement field (§10.4), cast to *dtype*."""
    return _core.encode_displacement(
        field,
        field_grid=field_grid,
        vector_space=vector_space,
        interpolation=interpolation,
        extrapolation=extrapolation,
        dtype=dtype,
    )


class DisplacementTransform(Transform):
    """``T(x) = x + u(x)`` with ``u`` sampled on ``field_grid``."""

    __slots__ = ()

    @property
    def field(self) -> Any:
        """The stored field dataset (read regions with :meth:`read_field`)."""
        return self._handle.field

    @property
    def field_grid_id(self) -> str:
        return str(self._handle.field_grid_id)

    @property
    def field_grid(self) -> Grid:
        found: Grid = self._handle.field_grid
        return found

    @property
    def vector_space(self) -> str:
        return str(self._handle.vector_space)

    @property
    def interpolation(self) -> str:
        return str(self._handle.interpolation)

    @property
    def extrapolation(self) -> str:
        return str(self._handle.extrapolation)

    def read_field(
        self, roi: Sequence[slice] | None = None, component: int | None = None
    ) -> npt.NDArray[Any]:
        found: npt.NDArray[Any] = self._handle.read_field(roi, component)
        return found

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """World displacement vectors at ``(..., S)`` world points."""
        found: npt.NDArray[np.float64] = self._handle.displacement_at(points)
        return found

    def sample_indices(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The stored field interpolated at ``(N, S)`` continuous field indices.

        A paired dataset asks for one displacement per training item, and
        reading the *entire* field for each turned a deformable registration
        into a full-volume decompress per item.  Linear interpolation needs the
        two lattice points either side of each query along each axis, so the
        read is the bounding window of the points, padded by one --- kilobytes
        of a 512³ field.  The result equals sampling the whole field
        (:func:`~medh5.transforms.apply.sample_field`).  Cubic interpolation
        reads the whole field: its spline coefficients are global.
        """
        found: npt.NDArray[np.float64] = self._handle.sample_indices(indices)
        return found

    def jacobian_determinant(
        self, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.jacobian_determinant(roi)
        return found

    def folding_fraction(self, roi: Sequence[slice] | None = None) -> float:
        """The fraction of voxels where the map folds (``det J <= 0``)."""
        return float(self._handle.folding_fraction(roi))

    @property
    def max_magnitude(self) -> float:
        return float(self._handle.max_magnitude)


__all__ = ["FLOAT16_SAFE_VOXELS", "DisplacementTransform", "encode_displacement"]
