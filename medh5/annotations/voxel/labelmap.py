"""``labelmap``: one dense integer volume, classes mutually exclusive (spec §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.voxel.payload import Masks, VoxelPayload, normalize_masks
from medh5.annotations.voxel.select import label_dtype_size
from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import IGNORE_ID


def encode_labelmap(
    masks: Masks,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    ignore: npt.NDArray[np.bool_] | None = None,
    ignore_id: int = IGNORE_ID,
) -> VoxelPayload:
    """Pack mutually exclusive class masks into one integer volume.

    Raises E404 when two classes claim the same voxel: silently letting the last
    writer win would turn a curation error into a training set nobody can audit.
    """
    resolved, shape = normalize_masks(masks, spatial_shape)
    class_ids = tuple(sorted(resolved))
    itemsize = label_dtype_size(class_ids, ignore=ignore is not None)
    dtype = np.uint8 if itemsize == 1 else np.uint16
    data = np.zeros(shape, dtype=dtype)
    claimed = np.zeros(shape, dtype=bool)
    for class_id in class_ids:
        mask = resolved[class_id]
        clash = mask & claimed
        if clash.any():
            first = tuple(int(v) for v in np.argwhere(clash)[0])
            raise MEDH5ValidationError(
                f"labelmap requires mutually exclusive classes; class {class_id} "
                f"overlaps an earlier class at voxel {first}. Use `layers` or "
                "`bitmask` for overlapping structures.",
                code="E404",
            )
        data[mask] = class_id
        claimed |= mask
    if ignore is not None:
        ignore_arr = np.asarray(ignore, dtype=bool)
        data[ignore_arr & ~claimed] = dtype(ignore_id)
    return VoxelPayload(
        kind="labelmap",
        datasets={"data": data},
        attrs={},
        stacked_axes=0,
        class_ids=class_ids,
    )


class LabelmapAnnotation(VoxelAnnotation):
    """Reader for ``kind = "labelmap"``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        try:
            return self.group["data"]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `labelmap` requires a `data` dataset",
                code="E410",
            ) from None

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        block: npt.NDArray[np.bool_] = np.asarray(self.data[roi]) == class_id
        return block

    def _encodes_ignore(self) -> bool:
        return bool(np.any(np.asarray(self.data[...]) == self.ignore_id))

    def labelmap(
        self,
        roi: Sequence[slice] | None = None,
        priority: Sequence[int | str] | None = None,
        *,
        dtype: npt.DTypeLike = np.uint16,
    ) -> npt.NDArray[Any]:
        """The stored volume itself --- no flattening is needed or possible."""
        window = self._roi(roi)
        return np.asarray(self.data[window]).astype(dtype)

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        window = self._roi(roi)
        block: npt.NDArray[np.bool_] = np.asarray(self.data[window]) == self.ignore_id
        return block


__all__ = ["LabelmapAnnotation", "encode_labelmap"]
