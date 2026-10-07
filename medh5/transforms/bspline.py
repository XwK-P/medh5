"""``bspline`` transforms: control-point coefficients on a grid (spec §10.5)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry.grid import Grid
from medh5.transforms.base import Transform

SUPPORTED_ORDERS: tuple[int, ...] = _core.SUPPORTED_ORDERS
DEFAULT_ORDER: int = _core.DEFAULT_ORDER

basis = _core.basis
encode_bspline = _core.encode_bspline


class BSplineTransform(Transform):
    """A free-form deformation evaluated with a uniform B-spline basis."""

    __slots__ = ()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.control_points
        return found

    @property
    def cp_grid_id(self) -> str:
        return str(self._handle.cp_grid_id)

    @property
    def cp_grid(self) -> Grid:
        found: Grid = self._handle.cp_grid
        return found

    @property
    def order(self) -> int:
        return int(self._handle.order)

    @property
    def vector_space(self) -> str:
        return str(self._handle.vector_space)

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.displacement_at(points)
        return found

    def to_displacement_field(self, grid: Grid) -> npt.NDArray[np.float32]:
        """The equivalent dense ``(S, *spatial)`` field on *grid*."""
        found: npt.NDArray[np.float32] = self._handle.to_displacement_field(grid)
        return found


__all__ = [
    "DEFAULT_ORDER",
    "SUPPORTED_ORDERS",
    "BSplineTransform",
    "basis",
    "encode_bspline",
]
