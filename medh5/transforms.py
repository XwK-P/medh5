"""Registration transforms (spec §10).

A transform maps points from ``from_frame`` to ``to_frame``: ``x_M = T(x_F)``,
the ITK convention, with no attribute to switch it.  The classes here are views
over the format engine's transform model, which evaluates every kind:
``identity`` and ``affine`` (§10.3), ``displacement`` (§10.4), ``bspline``
(§10.5) and ``composite`` (§10.2).

Resolution walks the frame graph, not transform names; it inverts a step only
where the inverse can be evaluated, refuses an ambiguous pair, and returns
``None`` --- never an invented transform --- when no path exists.

The field numerics are the engine's too --- linear and cubic interpolation match
SciPy's ``map_coordinates`` between a field's outermost samples, and take the
outermost sample's value in the half-voxel margin beyond them (§10.4), and the
Jacobian matches NumPy's gradient --- so every frontend evaluates a stored field
identically (§10.4--§10.6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry import Grid

TRANSFORM_KINDS: tuple[str, ...] = _core.TRANSFORM_KINDS
VECTOR_SPACES: tuple[str, ...] = _core.VECTOR_SPACES
INTERPOLATIONS: tuple[str, ...] = _core.INTERPOLATIONS
EXTRAPOLATIONS: tuple[str, ...] = _core.EXTRAPOLATIONS
SPEC_TRANSFORM_ATTRS: tuple[str, ...] = _core.SPEC_TRANSFORM_ATTRS
LAST_ROW_TOL: float = _core.LAST_ROW_TOL
FLOAT16_SAFE_VOXELS: float = _core.FLOAT16_SAFE_VOXELS
SUPPORTED_ORDERS: tuple[int, ...] = _core.SUPPORTED_ORDERS
DEFAULT_ORDER: int = _core.DEFAULT_ORDER

TransformHeader = _core.TransformHeader
"""The attribute header every transform carries (spec §10.1)."""

check_transform_id = _core.check_transform_id

# -- encoders: what ``SampleWriter.add_transform`` stores -----------------------

encode_identity = _core.encode_identity
encode_affine = _core.encode_affine
encode_bspline = _core.encode_bspline
encode_composite = _core.encode_composite


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


# -- field numerics (§10.4--§10.6) ----------------------------------------------

basis = _core.basis
inside_extent = _core.inside_extent
refuse_outside = _core.refuse_outside
linear_sample = _core.linear_sample
cubic_sample = _core.cubic_sample
sample_field = _core.sample_field
linear_part = _core.linear_part
to_world_vectors = _core.to_world_vectors
jacobian_determinant = _core.jacobian_determinant
folding_fraction = _core.folding_fraction
target_registration_error = _core.target_registration_error

# -- the read contract every kind shares (§10.1) ---------------------------------


class Transform:
    """What every transform answers, whatever its kind."""

    __slots__ = ("_handle", "transform_id")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.transform_id: str = handle.transform_id

    @property
    def header(self) -> Any:
        return self._handle.header

    @property
    def group(self) -> Any:
        """The stored group (a read-only :class:`~medh5.nodes.Group`)."""
        return self._handle.group

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    @property
    def from_frame(self) -> str:
        return str(self._handle.from_frame)

    @property
    def to_frame(self) -> str:
        return str(self._handle.to_frame)

    @property
    def units(self) -> str:
        return str(self._handle.units)

    @property
    def prov(self) -> str | None:
        found: str | None = self._handle.prov
        return found

    @property
    def metrics_key(self) -> str | None:
        found: str | None = self._handle.metrics_key
        return found

    @property
    def is_invertible(self) -> bool:
        return bool(self._handle.is_invertible)

    @property
    def timepoints(self) -> tuple[str, ...]:
        return tuple(self._handle.timepoints)

    def grid_in(self, frame: str) -> Grid | None:
        """A grid of this transform's file declared in *frame*, if any."""
        found: Grid | None = self._handle.grid_in(frame)
        return found

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map ``(..., S)`` world points from ``from_frame`` to ``to_frame``."""
        found: npt.NDArray[np.float64] = self._handle.transform_points(points)
        return found

    def inverse(self) -> Transform | None:
        """The stored inverse (``inverse_id``), or ``None``."""
        found = self._handle.inverse()
        return None if found is None else wrap_transform(found)

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def __repr__(self) -> str:
        return str(self._handle.__repr__())


# -- the kinds --------------------------------------------------------------------


class IdentityTransform(Transform):
    """Two frames that coincide, said explicitly."""

    __slots__ = ()


class AffineTransform(Transform):
    """A homogeneous ``(S+1, S+1)`` matrix in world coordinates."""

    __slots__ = ()

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.matrix
        return found

    @property
    def n_spatial(self) -> int:
        return int(self._handle.n_spatial)

    def inverse_matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_matrix()
        return found

    def inverse_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_points(points)
        return found

    @property
    def jacobian_determinant_value(self) -> float:
        return float(self._handle.jacobian_determinant_value)


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
        (:func:`sample_field`).  Cubic interpolation reads the whole field: its
        spline coefficients are global.
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


class CompositeTransform(Transform):
    """Components applied in order; each one's ``to_frame`` is the next's
    ``from_frame``."""

    __slots__ = ()

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self._handle.component_ids)

    def components(self) -> tuple[Transform, ...]:
        return tuple(wrap_transform(h) for h in self._handle.components())

    def check_chain(self) -> list[str]:
        """Every break in the frame chain (empty when it is whole)."""
        return list(self._handle.check_chain())


class InverseTransform(Transform):
    """The inverse of another transform, evaluated (not stored)."""

    __slots__ = ()

    def __init__(self, inner: Transform) -> None:
        super().__init__(_core.TransformHandle.inverse_of(inner._handle))

    @staticmethod
    def can_invert(inner: Transform) -> bool:
        """Whether *inner*'s inverse can be evaluated, not merely declared."""
        return bool(_core.TransformHandle.can_invert(inner._handle))


class ChainTransform(Transform):
    """Several transforms applied in order --- a resolved path."""

    __slots__ = ()

    def __init__(self, chain: Sequence[Transform]) -> None:
        super().__init__(_core.TransformHandle.chain([t._handle for t in chain]))

    @property
    def steps(self) -> tuple[Transform, ...]:
        return tuple(wrap_transform(h) for h in self._handle.steps or ())


_READERS: dict[str, type[Transform]] = {
    cls.__name__: cls
    for cls in (
        IdentityTransform,
        AffineTransform,
        DisplacementTransform,
        BSplineTransform,
        CompositeTransform,
        InverseTransform,
        ChainTransform,
    )
}
"""Engine class name -> facade class."""


def wrap_transform(handle: Any) -> Transform:
    """The facade class for an engine transform handle."""
    cls = _READERS.get(handle.class_name, Transform)
    made: Transform = cls.__new__(cls)
    Transform.__init__(made, handle)
    return made


# -- resolution (§10.2) -------------------------------------------------------------


def frame_graph(transforms: Mapping[str, Transform]) -> dict[str, list[str]]:
    """Frame -> frames one hop away, as :func:`resolve_between` would walk
    them: a reverse edge exists only where the inverse can be evaluated."""
    found: dict[str, list[str]] = _core.frame_graph(transforms)
    return found


def resolve_between(
    transforms: Mapping[str, Transform], from_frame: str, to_frame: str
) -> Transform | None:
    """The transform relating two frame uids, or ``None`` when none does."""
    found = _core.resolve_between(transforms, from_frame, to_frame)
    return None if found is None else wrap_transform(found)


def frames_of_timepoint(grids: Mapping[str, Grid], timepoint: str) -> tuple[str, ...]:
    """The frames of every grid of one timepoint, in grid order."""
    return tuple(_core.frames_of_timepoint(grids, timepoint))


__all__ = [
    "DEFAULT_ORDER",
    "EXTRAPOLATIONS",
    "FLOAT16_SAFE_VOXELS",
    "INTERPOLATIONS",
    "LAST_ROW_TOL",
    "SPEC_TRANSFORM_ATTRS",
    "SUPPORTED_ORDERS",
    "TRANSFORM_KINDS",
    "VECTOR_SPACES",
    "AffineTransform",
    "BSplineTransform",
    "ChainTransform",
    "CompositeTransform",
    "DisplacementTransform",
    "IdentityTransform",
    "InverseTransform",
    "Transform",
    "TransformHeader",
    "basis",
    "check_transform_id",
    "cubic_sample",
    "encode_affine",
    "encode_bspline",
    "encode_composite",
    "encode_displacement",
    "encode_identity",
    "folding_fraction",
    "frame_graph",
    "frames_of_timepoint",
    "inside_extent",
    "jacobian_determinant",
    "linear_part",
    "linear_sample",
    "refuse_outside",
    "resolve_between",
    "sample_field",
    "target_registration_error",
    "to_world_vectors",
    "wrap_transform",
]
