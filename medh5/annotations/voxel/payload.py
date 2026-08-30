"""Mask helpers shared by the voxel encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import BACKGROUND_ID, IGNORE_ID, MAX_CLASS_ID

Masks = Mapping[int, npt.NDArray[np.bool_]]

"""class id -> boolean occupancy over the grid's spatial shape."""


def _checked_class_id(class_id: Any) -> int:
    """Reject reserved and out-of-range ids here, not only at the writer.

    `AnnotationHeader` and the writer already enforce §5.3, but these encoders
    are exported from `medh5.annotations` and are what a third-party converter
    calls directly.  Unchecked, the id was cast into the labelmap dtype and
    wrapped: 0 became background, -1 became 255, 65535 became the ignore value,
    70000 became 4464 -- each one decoding as a different class than was asked
    for, with nothing raised.
    """
    value = int(class_id)
    if not BACKGROUND_ID < value <= MAX_CLASS_ID:
        raise MEDH5ValidationError(
            f"class id {value} is outside the writable range "
            f"[{BACKGROUND_ID + 1}, {MAX_CLASS_ID}]: {BACKGROUND_ID} is background and "
            f"{IGNORE_ID} is ignore (spec §5.3)",
            code="E303",
        )
    return value


def normalize_masks(
    masks: Masks, spatial_shape: tuple[int, ...] | None = None
) -> tuple[dict[int, npt.NDArray[np.bool_]], tuple[int, ...]]:
    """Coerce a mask mapping to ``bool`` arrays of one agreed shape."""
    from medh5.errors import MEDH5ValidationError

    out: dict[int, npt.NDArray[np.bool_]] = {}
    shape = spatial_shape
    for class_id, mask in masks.items():
        arr = np.asarray(mask)
        if arr.dtype != np.bool_:
            arr = arr.astype(bool)
        if shape is None:
            shape = arr.shape
        elif arr.shape != tuple(shape):
            raise MEDH5ValidationError(
                f"mask for class {class_id} has shape {arr.shape}, expected "
                f"{tuple(shape)}",
                code="E405",
            )
        out[_checked_class_id(class_id)] = arr
    if shape is None:
        raise MEDH5ValidationError("no masks were supplied", code="E410")
    return out, tuple(shape)


__all__ = ["Masks", "AnnotationPayload", "normalize_masks"]
