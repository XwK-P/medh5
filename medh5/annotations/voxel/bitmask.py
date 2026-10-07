"""``bitmask``: one bit per class per voxel, any overlap (spec §7.3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import VoxelAnnotation

BITS_PER_PLANE: int = _core.BITS_PER_PLANE

encode_bitmask = _core.encode_bitmask


class BitmaskAnnotation(VoxelAnnotation):
    """Classes as bits of ``uint64`` planes; position from ``bit_class_ids``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def bit_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.bit_class_ids
        return found

    @property
    def n_planes(self) -> int:
        return int(self._handle.n_planes)

    @property
    def position_of(self) -> dict[int, int]:
        """class id -> its bit position (plane * 64 + bit)."""
        found: dict[int, int] = self._handle.position_of
        return found

    def classes_at(self, voxel: Sequence[int]) -> tuple[int, ...]:
        """Every class present at one voxel."""
        return tuple(self._handle.classes_at([int(v) for v in voxel]))


__all__ = ["BITS_PER_PLANE", "BitmaskAnnotation", "encode_bitmask"]
