"""``labelmap``: one integer volume, classes mutually exclusive (spec §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import VoxelAnnotation

encode_labelmap = _core.encode_labelmap


class LabelmapAnnotation(VoxelAnnotation):
    """Mutually exclusive classes in one volume; ``65535`` marks ignore."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        """The in-band ignore region (the reserved id), ``False`` elsewhere."""
        found: npt.NDArray[np.bool_] = self._handle.ignore_mask(roi)
        return found


__all__ = ["LabelmapAnnotation", "encode_labelmap"]
