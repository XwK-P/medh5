"""``probmap``: per-class probability volumes (spec §7.5).

For soft ground truth, inter-rater probability maps, distillation targets and
predicted logits after sigmoid/softmax.  It is the one encoding for which
transcoding is lossless only under a declared threshold, which is why the
threshold is an explicit argument everywhere it appears rather than a constant
buried in a decoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel.payload import checked_class_id
from medh5.errors import MEDH5ValidationError

DEFAULT_THRESHOLD = 0.5

STORAGE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))
"""What §7.5 lets ``data`` be, narrowest first."""


def contains_at(values: npt.NDArray[Any], threshold: float) -> npt.NDArray[np.bool_]:
    """``values >= threshold``, decided in the precision *values* are stored in.

    The threshold is a float64 attribute and the data is usually float16.
    Comparing the two directly moved every voxel stored exactly at a threshold
    float16 cannot represent to the wrong side of it: a three-rater map
    thresholded at 1/3 stores 1/3 as 0.33325, below the 0.33333 it is compared
    with, so "one rater of three" read as absent (§7.5).  Rounding the threshold
    the way the data was rounded makes equal values compare equal.
    """
    array = np.asarray(values)
    if array.dtype.kind == "f":
        out: npt.NDArray[np.bool_] = array >= array.dtype.type(threshold)
        return out
    return np.asarray(array, dtype=np.float64) >= float(threshold)


def storage_dtype(
    planes: Sequence[npt.NDArray[np.float64]],
    threshold: float,
    requested: npt.DTypeLike = np.float16,
) -> np.dtype[Any]:
    """The narrowest allowed dtype, no narrower than *requested*, under which
    every voxel lands on the same side of *threshold* as it was given.

    Storage is a cost decision and must not change an answer: a value just
    under the threshold that float16 rounds up to it, or one on it that float16
    rounds down, would change which voxels contain the class.  Where float16
    would, the map is stored as float32 instead.
    """
    wanted = np.dtype(requested)
    candidates: list[np.dtype[Any]] = [
        d for d in STORAGE_DTYPES if d.itemsize >= wanted.itemsize
    ] or [wanted]
    for candidate in candidates:
        if all(
            np.array_equal(
                arr >= threshold, contains_at(arr.astype(candidate), threshold)
            )
            for arr in planes
        ):
            return candidate
    # Within float32 rounding of the threshold the given value and the stored
    # one are the same number for every purpose §7.5 serves; the stored value
    # then defines containment, as it does for any reader of the file.
    return candidates[-1]


def encode_probmap(
    probabilities: Mapping[int, npt.NDArray[Any]],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    dtype: npt.DTypeLike = np.float16,
    normalized: bool = False,
    threshold: float | None = None,
) -> AnnotationPayload:
    """Stack per-class probability volumes on a leading class axis.

    ``threshold`` is the probability at or above which a voxel *contains* a
    class (§7.5).  It is written only when given; the reader's default is 0.5.

    *dtype* is the narrowest storage wanted.  It is widened to ``float32``
    when storing at *dtype* would move a voxel across the threshold (see
    :func:`storage_dtype`), so ``contains`` answers for the stored file what it
    answered for the arrays given.
    """
    attrs: dict[str, Any] = {"normalized": bool(normalized)}
    if threshold is not None:
        value = float(threshold)
        if not 0.0 <= value <= 1.0 or value != value:
            raise MEDH5ValidationError(
                f"threshold {threshold!r} must lie in [0, 1]", code="E404"
            )
        attrs["threshold"] = value
    class_ids = tuple(sorted(checked_class_id(c) for c in probabilities))
    decide = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    shape = spatial_shape
    planes = []
    by_id = {int(c): v for c, v in probabilities.items()}
    for class_id in class_ids:
        arr = np.asarray(by_id[class_id], dtype=np.float64)
        if shape is None:
            shape = arr.shape
        elif arr.shape != tuple(shape):
            raise MEDH5ValidationError(
                f"probability map for class {class_id} has shape {arr.shape}, "
                f"expected {tuple(shape)}",
                code="E405",
            )
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise MEDH5ValidationError(
                f"probability map for class {class_id} has values outside [0, 1]",
                code="E411",
            )
        planes.append(arr)
    if shape is None:
        raise MEDH5ValidationError("no probability maps were supplied", code="E410")
    chosen = storage_dtype(planes, decide, dtype)
    data = (
        np.stack([arr.astype(chosen) for arr in planes])
        if planes
        else np.zeros((0, *shape), dtype=chosen)
    )
    return AnnotationPayload(
        kind="probmap",
        datasets={"data": data},
        attrs=attrs,
        stacked_axes=1,
        class_ids=class_ids,
    )


class ProbmapAnnotation(VoxelAnnotation):
    """Reader for ``kind = "probmap"``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        try:
            return self.group["data"]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `probmap` requires a `data` dataset",
                code="E410",
            ) from None

    @property
    def normalized(self) -> bool:
        return bool(self.group.attrs.get("normalized", False))

    @property
    def threshold(self) -> float:
        """The declared decision threshold (§7.5), or the 0.5 default."""
        return float(self.group.attrs.get("threshold", DEFAULT_THRESHOLD))

    def _position(self, class_id: int) -> int | None:
        try:
            return self.class_ids.index(class_id)
        except ValueError:
            return None

    def probabilities(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.float32]:
        """``(C, *roi_shape)`` float probabilities for the requested classes."""
        ids = self.resolve_classes(classes)
        window = self._roi(roi)
        out = np.zeros((len(ids), *self._roi_shape(window)), dtype=np.float32)
        for i, class_id in enumerate(ids):
            position = self._position(class_id)
            if position is not None:
                out[i] = np.asarray(self.data[(position, *window)], dtype=np.float32)
        return out

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        position = self._position(class_id)
        if position is None:
            return np.zeros(self._roi_shape(roi), dtype=bool)
        return contains_at(np.asarray(self.data[(position, *roi)]), self.threshold)

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["normalized"] = self.normalized
        out["threshold"] = self.threshold
        return out


__all__ = [
    "DEFAULT_THRESHOLD",
    "STORAGE_DTYPES",
    "ProbmapAnnotation",
    "contains_at",
    "encode_probmap",
    "storage_dtype",
]
