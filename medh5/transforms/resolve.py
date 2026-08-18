"""Finding the transform between two frames, or two timepoints (spec §10).

Callers ask "how do I get from here to there", not "which transform id do I
need".  Resolution therefore walks the **frame graph** rather than matching
names: a file may relate baseline to follow-up through one affine, through a
composite, or through an affine followed by a deformable refinement, and a
consumer should not have to know which.

Inverses are used where they are available and never invented: an affine is
inverted analytically, anything carrying ``inverse_id`` delegates to the stored
inverse, and a deformable transform with neither is simply not traversable
backwards --- because approximating the inverse of a dense field silently is how
a registration pipeline reports errors it never measured.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.transforms.base import Transform, TransformHeader


class InverseTransform(Transform):
    """``T⁻¹`` for a transform that can actually be inverted."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Transform) -> None:
        self._inner = inner
        header = TransformHeader(
            kind=inner.kind,
            from_frame=inner.to_frame,
            to_frame=inner.from_frame,
            units=inner.units,
            invertible=True,
        )
        super().__init__(
            f"{inner.transform_id}⁻¹",
            inner.group,
            header,
            inner._grids,  # noqa: SLF001 - same-package internal
            inner._siblings,  # noqa: SLF001
        )

    @staticmethod
    def can_invert(inner: Transform) -> bool:
        """Whether an inverse can be *evaluated*, not merely declared.

        ``is_invertible`` is the file's claim; this is what ``transform_points``
        below can actually carry out, and the two are not the same set.  A
        composite of invertible affines reports ``True`` and carries no
        ``inverse_id``, as does a displacement field written
        ``invertible=True``, and neither has an analytic inverse here.
        Resolution that trusted the claim handed back paths that raised the
        moment they were used --- from a call documented to answer ``None`` when
        no path exists.  The predicate lives next to the dispatch it mirrors so
        the two cannot drift apart again.
        """
        from medh5.transforms.affine import AffineTransform, IdentityTransform

        if not inner.is_invertible:
            return False
        if isinstance(inner, (IdentityTransform, AffineTransform)):
            return True
        return inner.inverse() is not None

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        from medh5.transforms.affine import AffineTransform, IdentityTransform

        if isinstance(self._inner, IdentityTransform):
            return np.asarray(points, dtype=np.float64)
        stored = self._inner.inverse()
        if stored is not None:
            return stored.transform_points(points)
        if isinstance(self._inner, AffineTransform):
            return self._inner.inverse_points(points)
        raise MEDH5ValidationError(
            f"transform {self._inner.transform_id!r} of kind {self._inner.kind!r} has "
            "no analytic inverse and declares no `inverse_id`; approximating one "
            "would report an accuracy nobody measured"
        )

    @property
    def is_invertible(self) -> bool:
        return True


class ChainTransform(Transform):
    """An in-memory composition, the result of resolving a multi-hop path."""

    __slots__ = ("_chain",)

    def __init__(self, chain: Sequence[Transform]) -> None:
        if not chain:
            raise MEDH5ValidationError("a transform chain needs at least one step")
        self._chain = tuple(chain)
        header = TransformHeader(
            kind="composite",
            from_frame=self._chain[0].from_frame,
            to_frame=self._chain[-1].to_frame,
            units=self._chain[0].units,
        )
        super().__init__(
            " -> ".join(t.transform_id for t in self._chain),
            self._chain[0].group,
            header,
            self._chain[0]._grids,  # noqa: SLF001 - same-package internal
            self._chain[0]._siblings,  # noqa: SLF001
        )

    @property
    def steps(self) -> tuple[Transform, ...]:
        return self._chain

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        for step in self._chain:
            values = step.transform_points(values)
        return values

    @property
    def is_invertible(self) -> bool:
        return all(step.is_invertible for step in self._chain)

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["steps"] = [step.transform_id for step in self._chain]
        return out


def _edges(
    transforms: Mapping[str, Transform],
) -> dict[str, list[tuple[str, Transform]]]:
    """Frame -> [(neighbour, step)], including inverses where they are usable."""
    out: dict[str, list[tuple[str, Transform]]] = {}
    for transform in transforms.values():
        out.setdefault(transform.from_frame, []).append((transform.to_frame, transform))
        out.setdefault(transform.to_frame, [])
        # `can_invert`, not `is_invertible`: an edge the walker cannot evaluate
        # is not an edge, and adding it turns "no path exists" into a path that
        # raises when the caller uses it.
        if InverseTransform.can_invert(transform):
            out[transform.to_frame].append(
                (transform.from_frame, InverseTransform(transform))
            )
    return out


def resolve_between(
    transforms: Mapping[str, Transform], from_frame: str, to_frame: str
) -> Transform | None:
    """The shortest transform path between two frames, or ``None``.

    Returns the transform itself for a single hop and a :class:`ChainTransform`
    for several, so the caller's code is the same either way.
    """
    if from_frame == to_frame:
        return None
    graph = _edges(transforms)
    if from_frame not in graph:
        return None
    queue: deque[tuple[str, list[Transform]]] = deque([(from_frame, [])])
    seen = {from_frame}
    while queue:
        frame, path = queue.popleft()
        for neighbour, step in graph.get(frame, ()):
            if neighbour in seen:
                continue
            extended = [*path, step]
            if neighbour == to_frame:
                return extended[0] if len(extended) == 1 else ChainTransform(extended)
            seen.add(neighbour)
            queue.append((neighbour, extended))
    return None


def frames_of_timepoint(grids: Mapping[str, Any], timepoint: str) -> tuple[str, ...]:
    """Every frame of reference used by a timepoint's grids."""
    out: list[str] = []
    for grid in grids.values():
        if grid.timepoint == timepoint and grid.frame_uid and grid.frame_uid not in out:
            out.append(grid.frame_uid)
    return tuple(out)


__all__ = [
    "ChainTransform",
    "InverseTransform",
    "frames_of_timepoint",
    "resolve_between",
]
