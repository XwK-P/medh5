"""``mask``: one boolean volume with no classes --- FOV, ignore regions (§7.7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import VoxelAnnotation

encode_mask = _core.encode_mask


class MaskAnnotation(VoxelAnnotation):
    """A region, not a class: what a field of view or an ignore mask is."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    def read(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        found: npt.NDArray[np.bool_] = self._handle.read_mask(roi)
        return found


__all__ = ["MaskAnnotation", "encode_mask"]
