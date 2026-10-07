"""The transform header and the read contract every kind shares (spec §10.1).

A transform maps points from ``from_frame`` to ``to_frame``: ``x_M = T(x_F)``,
the ITK convention, with no attribute to switch it.  The classes here are
views over the format engine's transform model, which evaluates every kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry.grid import Grid

TRANSFORM_KINDS: tuple[str, ...] = _core.TRANSFORM_KINDS
VECTOR_SPACES: tuple[str, ...] = _core.VECTOR_SPACES
INTERPOLATIONS: tuple[str, ...] = _core.INTERPOLATIONS
EXTRAPOLATIONS: tuple[str, ...] = _core.EXTRAPOLATIONS
SPEC_TRANSFORM_ATTRS: tuple[str, ...] = _core.SPEC_TRANSFORM_ATTRS

TransformHeader = _core.TransformHeader
"""The attribute header every transform carries (spec §10.1)."""

check_transform_id = _core.check_transform_id


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


def transform_readers() -> dict[str, Any]:
    """Engine class name -> facade class."""
    from medh5.transforms.affine import AffineTransform, IdentityTransform
    from medh5.transforms.bspline import BSplineTransform
    from medh5.transforms.composite import CompositeTransform
    from medh5.transforms.displacement import DisplacementTransform
    from medh5.transforms.resolve import ChainTransform, InverseTransform

    return {
        "IdentityTransform": IdentityTransform,
        "AffineTransform": AffineTransform,
        "DisplacementTransform": DisplacementTransform,
        "BSplineTransform": BSplineTransform,
        "CompositeTransform": CompositeTransform,
        "InverseTransform": InverseTransform,
        "ChainTransform": ChainTransform,
    }


_READERS: dict[str, Any] = {}


def wrap_transform(handle: Any) -> Transform:
    """The facade class for an engine transform handle."""
    if not _READERS:
        _READERS.update(transform_readers())
    cls = _READERS.get(handle.class_name, Transform)
    made: Transform = cls.__new__(cls)
    Transform.__init__(made, handle)
    return made


def frame_graph(transforms: Mapping[str, Transform]) -> dict[str, list[str]]:
    """Frame -> frames one hop away, as :func:`resolve_between` would walk
    them: a reverse edge exists only where the inverse can be evaluated."""
    found: dict[str, list[str]] = _core.frame_graph(transforms)
    return found


__all__ = [
    "EXTRAPOLATIONS",
    "INTERPOLATIONS",
    "SPEC_TRANSFORM_ATTRS",
    "TRANSFORM_KINDS",
    "VECTOR_SPACES",
    "Transform",
    "TransformHeader",
    "check_transform_id",
    "frame_graph",
    "transform_readers",
    "wrap_transform",
]
