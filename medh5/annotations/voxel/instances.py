"""``instances``: objects with identity, boxes and cropped masks (spec §7.4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import Instance, VoxelAnnotation, _instance


@dataclass(slots=True)
class InstanceInput:
    """One object handed to :func:`encode_instances`."""

    class_id: int
    instance_id: int
    mask: npt.NDArray[np.bool_] | None = None
    box: npt.NDArray[np.float32] | None = None
    crop: npt.NDArray[np.bool_] | None = None
    score: float | None = None


def encode_instances(
    objects: Sequence[InstanceInput],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    store_masks: bool = True,
    class_ids: Sequence[int] | None = None,
) -> Any:
    """Pack objects into boxes, ids and bit-packed crops (spec §7.4).

    ``class_ids`` declares the classes examined; with no objects it is what
    makes "examined, none found" expressible.
    """
    return _core.encode_instances(
        objects, spatial_shape, store_masks=store_masks, class_ids=class_ids
    )


def instances_from_masks(
    masks: Mapping[int, npt.NDArray[Any]], *, start_id: int = 1
) -> list[InstanceInput]:
    """One object per class mask, minting ids from *start_id*."""
    return [
        InstanceInput(class_id=class_id, instance_id=instance_id, mask=mask)
        for class_id, instance_id, mask in _core.instances_from_masks(
            masks, start_id=start_id
        )
    ]


class InstancesAnnotation(VoxelAnnotation):
    """Objects: each a class, a sample-scoped id, a box and maybe a mask."""

    __slots__ = ()

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        found: npt.NDArray[np.float32] = self._handle.boxes
        return found

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.object_class_ids
        return found

    @property
    def instance_ids(self) -> npt.NDArray[np.uint64]:
        found: npt.NDArray[np.uint64] = self._handle.instance_ids
        return found

    @property
    def scores(self) -> npt.NDArray[np.float32] | None:
        found: npt.NDArray[np.float32] | None = self._handle.scores
        return found

    @property
    def has_masks(self) -> bool:
        return bool(self._handle.has_masks)

    @property
    def n_objects(self) -> int:
        return int(self._handle.n_objects)

    def crop(self, index: int) -> npt.NDArray[np.bool_] | None:
        found: npt.NDArray[np.bool_] | None = self._handle.crop(int(index))
        return found

    def instance(self, instance_id: int) -> Instance:
        return _instance(self._handle.instance(int(instance_id)))

    def tracking(self) -> dict[int, int]:
        """``instance_id`` -> row index, the join key for §7.4 tracking."""
        found: dict[int, int] = self._handle.tracking()
        return found


__all__ = [
    "InstanceInput",
    "InstancesAnnotation",
    "encode_instances",
    "instances_from_masks",
]
