"""``composite`` transforms: an ordered chain of sibling transforms (§10.2)."""

from __future__ import annotations

from medh5 import _core
from medh5.transforms.base import Transform, wrap_transform

encode_composite = _core.encode_composite


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


__all__ = ["CompositeTransform", "encode_composite"]
