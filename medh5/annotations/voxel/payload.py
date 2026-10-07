"""Mask helpers shared by the voxel encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.payload import AnnotationPayload

Masks = Mapping[int, npt.NDArray[np.bool_]]
"""class id -> boolean occupancy over the grid's spatial shape."""

SLAB_BYTES: int = _core.SLAB_BYTES
"""Read budget for a scan that only needs a yes/no or a count."""


def checked_class_id(class_id: Any) -> int:
    """A class id in the writable range ``1..65534``, or a coded refusal (E303)."""
    return int(_core.check_class_id(class_id))


def normalize_masks(
    masks: Masks, spatial_shape: tuple[int, ...] | None = None
) -> tuple[dict[int, npt.NDArray[np.bool_]], tuple[int, ...]]:
    """Coerce a mask mapping to ``bool`` arrays of one agreed shape."""
    resolved, shape = _core.normalize_masks(masks, spatial_shape)
    return dict(resolved), tuple(shape)


__all__ = [
    "SLAB_BYTES",
    "AnnotationPayload",
    "Masks",
    "checked_class_id",
    "normalize_masks",
]
