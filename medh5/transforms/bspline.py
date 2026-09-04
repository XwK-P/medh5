"""``bspline`` --- a free-form deformation on a control-point lattice (§10.5).

The displacement at a point is a tensor-product B-spline of the surrounding
control points.  A cubic FFD is the usual case; the basis is written out rather
than pulled from a library so that a reader without SciPy still evaluates the
same transform the writer intended.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_int, as_str
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.transforms.apply import to_world_vectors
from medh5.transforms.base import VECTOR_SPACES, Transform

SUPPORTED_ORDERS = (1, 3)
DEFAULT_ORDER = 3


def basis(order: int, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """B-spline basis weights at fractional position *t* ∈ [0, 1).

    Returns ``(order + 1, N)``: weight of each control point in the support.
    """
    if order == 1:
        return np.stack([1.0 - t, t])
    if order == 3:
        t2, t3 = t * t, t * t * t
        return np.stack(
            [
                (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
                (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
                (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
                t3 / 6.0,
            ]
        )
    raise MEDH5ValidationError(
        f"B-spline order {order} is not supported; expected one of "
        f"{list(SUPPORTED_ORDERS)}",
        code="E502",
    )


def encode_bspline(
    control_points: npt.ArrayLike,
    *,
    cp_grid: str,
    order: int = DEFAULT_ORDER,
    vector_space: str = "world",
) -> AnnotationPayload:
    """Pack ``(S, *cp_shape)`` control-point coefficients."""
    if order not in SUPPORTED_ORDERS:
        raise MEDH5ValidationError(
            f"B-spline order {order} is not supported; expected one of "
            f"{list(SUPPORTED_ORDERS)}",
            code="E502",
        )
    if vector_space not in VECTOR_SPACES:
        raise MEDH5ValidationError(
            f"vector_space {vector_space!r} must be one of {list(VECTOR_SPACES)}",
            code="E502",
        )
    array = np.asarray(control_points, dtype=np.float64)
    if array.ndim < 3 or array.shape[0] != array.ndim - 1:
        raise MEDH5ValidationError(
            f"control_points must be (S, *cp_shape) with S components, got "
            f"{array.shape}",
            code="E503",
        )
    if any(extent < order + 1 for extent in array.shape[1:]):
        raise MEDH5ValidationError(
            f"an order-{order} B-spline needs at least {order + 1} control points per "
            f"axis, got {array.shape[1:]}",
            code="E503",
        )
    return AnnotationPayload(
        kind="bspline",
        datasets={"control_points": array},
        attrs={"cp_grid": cp_grid, "order": int(order), "vector_space": vector_space},
        stacked_axes=1,
    )


class BSplineTransform(Transform):
    """Reader for ``kind = "bspline"``."""

    __slots__ = ()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        if "control_points" not in self.group:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: `bspline` requires `control_points`",
                code="E502",
            )
        return np.asarray(self.group["control_points"][...], dtype=np.float64)

    @property
    def cp_grid_id(self) -> str:
        value = self.group.attrs.get("cp_grid")
        if value is None:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: `bspline` requires `cp_grid`",
                code="E503",
            )
        return as_str(value)

    @property
    def cp_grid(self) -> Grid:
        gid = self.cp_grid_id
        try:
            return self._grids[gid]
        except KeyError:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: control-point grid {gid!r} does "
                "not exist",
                code="E101",
            ) from None

    @property
    def order(self) -> int:
        return as_int(self.group.attrs.get("order", DEFAULT_ORDER))

    @property
    def vector_space(self) -> str:
        return as_str(self.group.attrs.get("vector_space", "world"))

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the tensor-product B-spline at world points."""
        grid = self.cp_grid
        coefficients = self.control_points
        order = self.order
        dim = coefficients.shape[0]
        extent = np.asarray(coefficients.shape[1:], dtype=np.int64)

        world = np.asarray(points, dtype=np.float64)
        cp_index = grid.world_to_index(world.reshape(-1, dim))
        # The support of an order-k basis starts (k-1)//2 cells before the cell
        # containing the point, so a cubic spline reaches one cell either side.
        first = np.floor(cp_index).astype(np.int64) - (order - 1) // 2
        frac = cp_index - np.floor(cp_index)
        weights = np.stack([basis(order, frac[:, axis]) for axis in range(dim)])

        out = np.zeros((cp_index.shape[0], dim), dtype=np.float64)
        for offsets in itertools.product(range(order + 1), repeat=dim):
            weight = np.ones(cp_index.shape[0], dtype=np.float64)
            for axis, offset in enumerate(offsets):
                weight = weight * weights[axis, offset]
            index = np.clip(first + np.asarray(offsets), 0, extent - 1)
            out += weight[:, None] * coefficients[(slice(None), *index.T)].T
        return to_world_vectors(out, grid, self.vector_space).reshape(world.shape)

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        return values + self.displacement_at(values)

    def to_displacement_field(self, grid: Grid) -> npt.NDArray[np.float32]:
        """Sample the spline onto a grid --- the bridge to a dense field."""
        coords = np.stack(
            np.meshgrid(*[np.arange(n) for n in grid.spatial_shape], indexing="ij"),
            axis=-1,
        ).reshape(-1, grid.n_spatial)
        world = grid.index_to_world(coords.astype(np.float64))
        displacement = self.displacement_at(world)
        return np.ascontiguousarray(
            displacement.reshape(*grid.spatial_shape, grid.n_spatial).transpose(
                grid.n_spatial, *range(grid.n_spatial)
            ),
            dtype=np.float32,
        )

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update(
            {
                "cp_grid": self.cp_grid_id,
                "order": self.order,
                "vector_space": self.vector_space,
                "cp_shape": [int(v) for v in self.control_points.shape],
            }
        )
        return out


__all__ = [
    "DEFAULT_ORDER",
    "SUPPORTED_ORDERS",
    "BSplineTransform",
    "basis",
    "encode_bspline",
]
