"""The intermediate a voxel encoder produces before anything touches HDF5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class VoxelPayload:
    """Datasets and kind-specific attributes for one encoded voxel annotation.

    Keeping encoders pure --- arrays in, arrays out --- is what lets the
    transcoding matrix be tested without a file, and what lets the writer decide
    chunking and codecs in one place instead of in five.
    """

    kind: str
    datasets: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    stacked_axes: int = 0
    """Leading axes of ``data`` that must get chunk extent 1 (spec §14.1)."""

    class_ids: tuple[int, ...] = ()

    @property
    def data(self) -> npt.NDArray[Any]:
        return self.datasets["data"]

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.datasets.values())

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "datasets": {
                name: {"shape": list(a.shape), "dtype": a.dtype.str}
                for name, a in self.datasets.items()
            },
            "nbytes": self.nbytes,
        }


Masks = Mapping[int, npt.NDArray[np.bool_]]
"""class id -> boolean occupancy over the grid's spatial shape."""


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
        out[int(class_id)] = arr
    if shape is None:
        raise MEDH5ValidationError("no masks were supplied", code="E410")
    return out, tuple(shape)


__all__ = ["Masks", "VoxelPayload", "normalize_masks"]
