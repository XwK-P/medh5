"""``mask``: one boolean volume with no classes (spec §4.4, §7.7).

Used for a validity/FOV mask, and for the ignore region of a ``bitmask`` or
``probmap`` annotation, which cannot carry an in-band ignore value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError


def encode_mask(mask: npt.NDArray[Any]) -> AnnotationPayload:
    """Wrap a boolean volume as a ``mask`` payload."""
    data = np.asarray(mask, dtype=bool)
    return AnnotationPayload(
        kind="mask", datasets={"data": data}, attrs={}, stacked_axes=0, class_ids=()
    )


class MaskAnnotation(VoxelAnnotation):
    """Reader for ``kind = "mask"``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        try:
            return self.group["data"]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `mask` requires a `data` dataset",
                code="E410",
            ) from None

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        del class_id  # a mask has no classes; the volume itself is the answer
        return np.asarray(self.data[roi]).astype(bool)

    def read(self, roi: Any = None) -> npt.NDArray[np.bool_]:
        window = self._roi(roi)
        return np.asarray(self.data[window]).astype(bool)

    def dense(self, classes: Any = None, roi: Any = None) -> npt.NDArray[np.bool_]:
        """The volume itself, as a single plane.

        A `mask` has no classes (§4.4), so naming one selects nothing --- but
        the argument is *validated* rather than dropped.  Discarding it meant
        ``dense([65535])`` handed back the whole mask for the reserved ignore
        id, which is the one answer the guard in
        :meth:`~medh5.annotations.base.VoxelAnnotation.resolve_classes` exists
        to prevent, and this was the only encoding that gave it: the other five
        route through that guard on the way to their planes.  So a caller who
        asked all six the same question got a refusal from five and a full
        mask --- read as "ignored everywhere" --- from one.
        """
        if classes is not None:
            self.resolve_classes(classes)
        return self.read(roi)[None, ...]

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["true_voxels"] = int(np.count_nonzero(self.data[...]))
        return out


__all__ = ["MaskAnnotation", "encode_mask"]
