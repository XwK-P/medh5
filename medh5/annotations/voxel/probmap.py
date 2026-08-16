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
from medh5.annotations.voxel.payload import VoxelPayload
from medh5.errors import MEDH5ValidationError

DEFAULT_THRESHOLD = 0.5


def encode_probmap(
    probabilities: Mapping[int, npt.NDArray[Any]],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    dtype: npt.DTypeLike = np.float16,
    normalized: bool = False,
) -> VoxelPayload:
    """Stack per-class probability volumes on a leading class axis."""
    class_ids = tuple(sorted(int(c) for c in probabilities))
    shape = spatial_shape
    planes = []
    for class_id in class_ids:
        arr = np.asarray(probabilities[class_id], dtype=np.float64)
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
        planes.append(arr.astype(dtype))
    if shape is None:
        raise MEDH5ValidationError("no probability maps were supplied", code="E410")
    data = np.stack(planes) if planes else np.zeros((0, *shape), dtype=np.dtype(dtype))
    return VoxelPayload(
        kind="probmap",
        datasets={"data": data},
        attrs={"normalized": bool(normalized)},
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
        return (
            np.asarray(self.data[(position, *roi)], dtype=np.float32) >= self.threshold
        )

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["normalized"] = self.normalized
        out["threshold"] = self.threshold
        return out


__all__ = ["DEFAULT_THRESHOLD", "ProbmapAnnotation", "encode_probmap"]
