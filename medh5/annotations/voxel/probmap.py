"""``probmap``: per-class probabilities, soft labels (spec §7.5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import VoxelAnnotation

DEFAULT_THRESHOLD: float = _core.DEFAULT_THRESHOLD
STORAGE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))

contains_at = _core.contains_at


def storage_dtype(
    planes: Any, threshold: float, requested: npt.DTypeLike = np.float16
) -> np.dtype[Any]:
    """The narrowest of ``float16``/``float32`` that keeps every voxel's
    containment decision (§7.5)."""
    found: np.dtype[Any] = _core.storage_dtype(planes, threshold, requested)
    return found


def encode_probmap(
    probabilities: Any,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    dtype: npt.DTypeLike = np.float16,
    normalized: bool = False,
    threshold: float | None = None,
) -> Any:
    """Pack per-class probability planes (spec §7.5)."""
    return _core.encode_probmap(
        probabilities,
        spatial_shape,
        dtype=dtype,
        normalized=normalized,
        threshold=threshold,
    )


class ProbmapAnnotation(VoxelAnnotation):
    """Soft labels; a voxel contains a class at or above ``threshold``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def normalized(self) -> bool:
        return bool(self._handle.normalized)

    @property
    def threshold(self) -> float:
        return float(self._handle.threshold)

    def probabilities(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.float32]:
        found: npt.NDArray[np.float32] = self._handle.probabilities(classes, roi)
        return found


__all__ = [
    "DEFAULT_THRESHOLD",
    "STORAGE_DTYPES",
    "ProbmapAnnotation",
    "contains_at",
    "encode_probmap",
    "storage_dtype",
]
