"""``composite`` --- an ordered chain of transforms (spec §10.5).

Components apply **left to right**: ``T(x) = T_n(… T_1(x))``, and their frames
must chain, with the first ``from_frame`` and the last ``to_frame`` equal to the
composite's own.  A validator checks the chain (E501), because a composite whose
middle link is reversed produces plausible-looking coordinates that are wrong
everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.transforms.base import Transform, component_ids, open_transform


def encode_composite(components: Sequence[str]) -> AnnotationPayload:
    """Declare an ordered composition by component id."""
    if len(components) < 2:
        raise MEDH5ValidationError(
            "a composite transform needs at least two components", code="E501"
        )
    return AnnotationPayload(kind="composite", attrs={"components": list(components)})


class CompositeTransform(Transform):
    """Reader for ``kind = "composite"``."""

    __slots__ = ()

    @property
    def component_ids(self) -> tuple[str, ...]:
        return component_ids(self.group)

    def components(self) -> tuple[Transform, ...]:
        """The chain, resolved and in application order."""
        out: list[Transform] = []
        for name in self.component_ids:
            if name not in self._siblings:
                raise MEDH5ValidationError(
                    f"composite {self.transform_id!r} names component {name!r}, "
                    "which does not exist",
                    code="E501",
                )
            out.append(
                open_transform(name, self._siblings[name], self._grids, self._siblings)
            )
        return tuple(out)

    def check_chain(self) -> list[str]:
        """Frame-chaining problems, as messages; empty means the chain is sound."""
        problems: list[str] = []
        try:
            chain = self.components()
        except MEDH5ValidationError as exc:
            return [exc.message]
        if not chain:
            return ["composite declares no components"]
        if chain[0].from_frame != self.from_frame:
            problems.append(
                f"first component starts in {chain[0].from_frame!r}, but the composite "
                f"declares {self.from_frame!r}"
            )
        if chain[-1].to_frame != self.to_frame:
            problems.append(
                f"last component ends in {chain[-1].to_frame!r}, but the composite "
                f"declares {self.to_frame!r}"
            )
        for left, right in zip(chain, chain[1:], strict=False):
            if left.to_frame != right.from_frame:
                problems.append(
                    f"{left.transform_id!r} ends in {left.to_frame!r} but "
                    f"{right.transform_id!r} starts in {right.from_frame!r}"
                )
        # Units, not just frames.  §10.1 makes `units` a MUST --- "coordinate
        # units, matching the frames' grids" --- and only the frames were
        # checked, so a composite of an `mm` leg and a `um` leg chained cleanly
        # and applied a 1000x error to the second half of the transform. The
        # frames agreeing says the legs meet; the units agreeing says they meet
        # in the same space.
        mismatched = [t.transform_id for t in chain if t.units != self.units]
        if mismatched:
            listed = ", ".join(
                f"{t.transform_id!r} in {t.units!r}"
                for t in chain
                if t.units != self.units
            )
            problems.append(
                f"composite declares units {self.units!r} but {listed} --- a "
                "chain whose legs are in different units does not compose"
            )
        return problems

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        problems = self.check_chain()
        if problems:
            raise MEDH5ValidationError(
                f"composite {self.transform_id!r} has a broken frame chain: "
                + "; ".join(problems),
                code="E501",
            )
        values = np.asarray(points, dtype=np.float64)
        for component in self.components():
            values = component.transform_points(values)
        return values

    @property
    def is_invertible(self) -> bool:
        if self.header.invertible is not None:
            return bool(self.header.invertible)
        if self.header.inverse_id is not None:
            return True
        try:
            return all(c.is_invertible for c in self.components())
        except MEDH5ValidationError:
            return False

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["components"] = list(self.component_ids)
        out["chain_ok"] = not self.check_chain()
        return out


__all__ = ["CompositeTransform", "encode_composite"]
