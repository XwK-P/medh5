"""Resolving the transform between two frames (spec §10.2).

Resolution walks the frame graph, not transform names; it inverts a step only
where the inverse can be evaluated, refuses an ambiguous pair, and returns
``None`` --- never an invented transform --- when no path exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from medh5 import _core
from medh5.geometry.grid import Grid
from medh5.transforms.base import Transform, wrap_transform


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


def resolve_between(
    transforms: Mapping[str, Transform], from_frame: str, to_frame: str
) -> Transform | None:
    """The transform relating two frame uids, or ``None`` when none does."""
    found = _core.resolve_between(transforms, from_frame, to_frame)
    return None if found is None else wrap_transform(found)


def frames_of_timepoint(grids: Mapping[str, Grid], timepoint: str) -> tuple[str, ...]:
    """The frames of every grid of one timepoint, in grid order."""
    return tuple(_core.frames_of_timepoint(grids, timepoint))


def stored_inverse(transform: Transform) -> Transform | None:
    """The stored inverse a transform names (``inverse_id``), if any."""
    return transform.inverse()


__all__ = [
    "ChainTransform",
    "InverseTransform",
    "frames_of_timepoint",
    "resolve_between",
    "stored_inverse",
]
