"""``displacement`` --- a dense deformation field (spec §10.4).

``T(x) = x + u(x)``, with ``u`` interpolated from ``field`` at ``x`` expressed in
the field grid.  The field **must** live in ``from_frame``: it is sampled at the
points being mapped, and those are source-frame points.

Components sit on the *leading* axis, ``(S, Z, Y, X)`` chunked ``(1, …)``, rather
than trailing ``(Z, Y, X, S)``.  That lets a reader fetch one component, or one
ROI of one component, without decompressing the rest --- the same reason
``layers`` and ``bitmask`` stack the way they do (§14.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_str
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.transforms.apply import (
    folding_fraction,
    inside_extent,
    jacobian_determinant,
    linear_sample,
    refuse_outside,
    sample_field,
    to_world_vectors,
)
from medh5.transforms.base import (
    EXTRAPOLATIONS,
    INTERPOLATIONS,
    VECTOR_SPACES,
    Transform,
)

FLOAT16_SAFE_VOXELS = 64.0
"""Displacement magnitude below which ``float16`` costs ~5e-4 relative precision.

Far below any registration's accuracy, and it halves the field on disk, which is
why the spec recommends it rather than merely permitting it.
"""


def encode_displacement(
    field: npt.ArrayLike,
    *,
    field_grid: str,
    vector_space: str = "world",
    interpolation: str = "linear",
    extrapolation: str = "zero",
    dtype: npt.DTypeLike = np.float32,
) -> AnnotationPayload:
    """Pack a ``(S, *spatial)`` displacement field."""
    if vector_space not in VECTOR_SPACES:
        raise MEDH5ValidationError(
            f"vector_space {vector_space!r} must be one of {list(VECTOR_SPACES)}",
            code="E502",
        )
    if interpolation not in INTERPOLATIONS:
        raise MEDH5ValidationError(
            f"interpolation {interpolation!r} must be one of {list(INTERPOLATIONS)}",
            code="E502",
        )
    if extrapolation not in EXTRAPOLATIONS:
        raise MEDH5ValidationError(
            f"extrapolation {extrapolation!r} must be one of {list(EXTRAPOLATIONS)}",
            code="E502",
        )
    array = np.asarray(field, dtype=dtype)
    if array.ndim < 3:
        raise MEDH5ValidationError(
            f"displacement field must be (S, *spatial), got {array.shape}", code="E503"
        )
    if array.shape[0] != array.ndim - 1:
        raise MEDH5ValidationError(
            f"displacement field has {array.shape[0]} components on a "
            f"{array.ndim - 1}-D lattice; they must match",
            code="E503",
        )
    return AnnotationPayload(
        kind="displacement",
        datasets={"field": array},
        attrs={
            "field_grid": field_grid,
            "vector_space": vector_space,
            "interpolation": interpolation,
            "extrapolation": extrapolation,
        },
        stacked_axes=1,
    )


class DisplacementTransform(Transform):
    """Reader for ``kind = "displacement"``."""

    __slots__ = ()

    @property
    def field(self) -> Any:
        if "field" not in self.group:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: `displacement` requires a `field` "
                "dataset",
                code="E502",
            )
        return self.group["field"]

    @property
    def field_grid_id(self) -> str:
        value = self.group.attrs.get("field_grid")
        if value is None:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: `displacement` requires "
                f"`field_grid`",
                code="E503",
            )
        return as_str(value)

    @property
    def field_grid(self) -> Grid:
        gid = self.field_grid_id
        try:
            return self._grids[gid]
        except KeyError:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: field grid {gid!r} does not exist",
                code="E101",
            ) from None

    @property
    def vector_space(self) -> str:
        return as_str(self.group.attrs.get("vector_space", "world"))

    @property
    def interpolation(self) -> str:
        return as_str(self.group.attrs.get("interpolation", "linear"))

    @property
    def extrapolation(self) -> str:
        return as_str(self.group.attrs.get("extrapolation", "zero"))

    def read_field(
        self, roi: Sequence[slice] | None = None, component: int | None = None
    ) -> npt.NDArray[Any]:
        """Read the field, or one component, or one ROI --- in a single call."""
        window = (
            tuple(roi) if roi is not None else (slice(None),) * (self.field.ndim - 1)
        )
        if component is not None:
            return np.asarray(self.field[(component, *window)])
        return np.asarray(self.field[(slice(None), *window)])

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """World-space displacement ``u(x)`` at world points ``x``."""
        grid = self.field_grid
        world = np.asarray(points, dtype=np.float64)
        indices = grid.world_to_index(world.reshape(-1, grid.n_spatial))
        raw = self._sample(indices)
        return to_world_vectors(raw, grid, self.vector_space).reshape(world.shape)

    def _sample(self, indices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Interpolate at continuous field indices, reading what the points touch.

        A paired dataset asks for one displacement per training item, and this
        used to answer by reading the *entire* field --- every chunk, every
        time --- so a deformable registration turned each item into a
        full-volume decompress.  Linear interpolation needs the two lattice
        points either side of each query along each axis, so the read is the
        bounding window of the points, padded by one; on a 512³ field that is
        kilobytes rather than gigabytes.  The result is identical to sampling
        the whole field: the window is clamped inside the array exactly where
        ``nearest`` would clamp, and the ``zero``/``error`` cases are decided
        against the full extent before the window is cut.

        Cubic interpolation reads the whole field as before.  Its spline
        coefficients are global, so a window would change the answer near its
        own edges, and an answer that depends on which other points were asked
        about at the same time is not one.
        """
        field = self.field
        spatial = np.asarray(field.shape[1:], dtype=np.int64)
        if self.interpolation != "linear" or indices.shape[0] == 0:
            return sample_field(
                np.asarray(field[...]),
                indices,
                interpolation=self.interpolation,
                extrapolation=self.extrapolation,
            )
        inside = inside_extent(spatial, indices)
        if self.extrapolation == "error":
            refuse_outside(inside)
        lo = np.clip(np.floor(indices.min(axis=0)).astype(np.int64) - 1, 0, spatial - 1)
        hi = np.clip(np.ceil(indices.max(axis=0)).astype(np.int64) + 2, lo + 1, spatial)
        # Two lattice points per axis where the array has them, so the clamp
        # at the far edge lands on the same voxel it would in the whole field.
        lo = np.minimum(lo, np.maximum(hi - 2, 0))
        window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        block = np.asarray(field[(slice(None), *window)])
        raw = linear_sample(block, indices - lo, extrapolation="nearest")
        if self.extrapolation == "zero":
            raw[~inside] = 0.0
        return raw

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        return values + self.displacement_at(values)

    def jacobian_determinant(
        self, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.float64]:
        """``det(I + du/dx)`` per voxel --- values ≤ 0 mark folding."""
        grid = self.field_grid
        field = self.read_field(roi)
        if roi is not None:
            grid = _cropped_grid(grid, tuple(roi))
        return jacobian_determinant(field, grid, vector_space=self.vector_space)

    def folding_fraction(self, roi: Sequence[slice] | None = None) -> float:
        return folding_fraction(self.jacobian_determinant(roi))

    @property
    def max_magnitude(self) -> float:
        """Largest displacement magnitude, in the field's own component units."""
        field = np.asarray(self.field[...], dtype=np.float64)
        return float(np.sqrt((field**2).sum(axis=0)).max()) if field.size else 0.0

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update(
            {
                "field_grid": self.field_grid_id,
                "vector_space": self.vector_space,
                "interpolation": self.interpolation,
                "field_shape": [int(v) for v in self.field.shape],
                "field_dtype": self.field.dtype.str,
            }
        )
        return out


def _cropped_grid(grid: Grid, roi: tuple[slice, ...]) -> Grid:
    """The grid of an ROI: same lattice, origin moved to the ROI's first voxel."""
    starts = [0 if s.start is None else int(s.start) for s in roi]
    shape = tuple(
        (int(s.stop) if s.stop is not None else n) - start
        for s, n, start in zip(roi, grid.spatial_shape, starts, strict=True)
    )
    origin = grid.index_to_world(np.asarray(starts, dtype=np.float64))
    lead = grid.shape[: grid.ndim - grid.n_spatial]
    return Grid(
        grid_id=grid.grid_id,
        shape=(*lead, *shape),
        axis_names=grid.axis_names,
        axis_kinds=grid.axis_kinds,
        spacing=grid.spacing,
        origin=tuple(float(v) for v in origin),
        direction=grid.direction,
        coord_system=grid.coord_system,
        units=grid.units,
        timepoint=grid.timepoint,
        frame_uid=grid.frame_uid,
    )


__all__ = [
    "FLOAT16_SAFE_VOXELS",
    "DisplacementTransform",
    "encode_displacement",
]
