"""``instances``: per-object box + bbox-local bit-packed mask (spec §7.4).

Storage is proportional to **object volume, not image volume**: 0.08 MiB versus
3.57 MiB for the same 200-structure phantom stored as per-class dense volumes, a
45x reduction.  It is the natural encoding for lesions, nodules and cells, and
the only voxel encoding that carries object identity.

``instance_id`` is sample-scoped and longitudinal: the same physical object
observed at several timepoints reuses its id in every annotation describing it,
so lesion tracking, growth curves and per-lesion response are a **join on this
column** rather than a separate structure.  An object present at one timepoint
and absent at another is represented by its absence --- and a *resolved* lesion
is distinguishable from an *unexamined* one only through ``annotated_class_ids``,
which is why coverage is required rather than optional.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import Instance, VoxelAnnotation, instance_id_dtype
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import box_to_slices, slices_to_box


@dataclass(slots=True)
class InstanceInput:
    """One object handed to :func:`encode_instances`."""

    class_id: int
    instance_id: int
    mask: npt.NDArray[np.bool_] | None = None
    box: npt.NDArray[np.float32] | None = None
    crop: npt.NDArray[np.bool_] | None = None
    score: float | None = None


def _tight_slices(mask: npt.NDArray[np.bool_]) -> tuple[slice, ...] | None:
    slices: list[slice] = []
    for axis in range(mask.ndim):
        axes = tuple(i for i in range(mask.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=axes))
        if present.size == 0:
            return None
        slices.append(slice(int(present[0]), int(present[-1]) + 1))
    return tuple(slices)


def encode_instances(
    objects: Sequence[InstanceInput],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    store_masks: bool = True,
    class_ids: Sequence[int] | None = None,
) -> AnnotationPayload:
    """Pack objects into boxes, ids, offsets and one concatenated bit stream.

    *class_ids* declares the classes the annotation can express, which is not
    the same as the classes that happen to have an object.  A class searched for
    and not found must stay in ``class_ids`` --- dropping it would turn
    "verified absent" into "never looked for" (spec §11.3).
    """
    if not objects:
        raise MEDH5ValidationError("no instances were supplied", code="E410")
    ndim = None
    boxes: list[npt.NDArray[np.float32]] = []
    crops: list[npt.NDArray[np.bool_]] = []
    object_classes: list[int] = []
    instance_ids: list[int] = []
    scores: list[float] = []
    has_scores = any(o.score is not None for o in objects)

    for obj in objects:
        crop = obj.crop
        box = obj.box
        if obj.mask is not None:
            mask = np.asarray(obj.mask, dtype=bool)
            if spatial_shape is not None and mask.shape != tuple(spatial_shape):
                raise MEDH5ValidationError(
                    f"instance {obj.instance_id}: mask shape {mask.shape} != "
                    f"grid {tuple(spatial_shape)}",
                    code="E405",
                )
            slices = _tight_slices(mask)
            if slices is None:
                raise MEDH5ValidationError(
                    f"instance {obj.instance_id} has an empty mask", code="E404"
                )
            crop = mask[slices]
            box = slices_to_box(slices)
        if box is None:
            raise MEDH5ValidationError(
                f"instance {obj.instance_id} needs either a mask or a box", code="E410"
            )
        box_arr = np.asarray(box, dtype=np.float32)
        if np.any(box_arr[:, 0] > box_arr[:, 1]):
            raise MEDH5ValidationError(
                f"instance {obj.instance_id}: box has lo > hi", code="E406"
            )
        if ndim is None:
            ndim = box_arr.shape[0]
        elif box_arr.shape[0] != ndim:
            raise MEDH5ValidationError(
                "instances disagree on dimensionality", code="E405"
            )
        boxes.append(box_arr)
        object_classes.append(int(obj.class_id))
        instance_ids.append(int(obj.instance_id))
        scores.append(float(obj.score) if obj.score is not None else float("nan"))
        if store_masks:
            if crop is None:
                raise MEDH5ValidationError(
                    f"instance {obj.instance_id}: store_masks=True needs a mask or "
                    f"crop",
                    code="E410",
                )
            crops.append(np.asarray(crop, dtype=bool))

    datasets: dict[str, npt.NDArray[Any]] = {
        "boxes": np.stack(boxes).astype(np.float32),
        "class_ids": np.asarray(object_classes, dtype=np.uint16),
        # §7.4 permits uint32 or uint64; the width follows the data.  Hard-casting
        # to uint32 silently wrapped an id minted from a 64-bit key -- 2**32 + 7
        # became 7 -- so one object took another's identity in the field that is
        # the entire longitudinal join.
        "instance_ids": np.asarray(instance_ids, dtype=instance_id_dtype(instance_ids)),
    }
    if has_scores:
        datasets["scores"] = np.asarray(scores, dtype=np.float32)
    if store_masks:
        packed = [np.packbits(c.reshape(-1)) for c in crops]
        offsets = np.zeros(len(packed) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum([p.size for p in packed], dtype=np.uint64)
        datasets["mask_offsets"] = offsets
        datasets["mask_shapes"] = np.asarray([c.shape for c in crops], dtype=np.int32)
        datasets["mask_data"] = (
            np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8)
        )
    declared = (
        tuple(sorted({int(c) for c in class_ids}))
        if class_ids is not None
        else tuple(sorted(set(object_classes)))
    )
    return AnnotationPayload(
        kind="instances",
        datasets=datasets,
        attrs={},
        stacked_axes=0,
        class_ids=declared,
    )


def instances_from_masks(
    masks: dict[int, npt.NDArray[np.bool_]], *, start_id: int = 1
) -> list[InstanceInput]:
    """One object per class --- what a converter can honestly infer from masks.

    Splitting a class mask into connected components would **invent** object
    identity, so it is not done here: a caller who wants per-lesion ids supplies
    them.
    """
    return [
        InstanceInput(class_id=class_id, instance_id=start_id + i, mask=mask)
        for i, (class_id, mask) in enumerate(sorted(masks.items()))
    ]


class InstancesAnnotation(VoxelAnnotation):
    """Reader for ``kind = "instances"``."""

    __slots__ = ()

    def _dataset(self, name: str, required: bool = True) -> Any:
        if name in self.group:
            return self.group[name]
        if required:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `instances` requires a {name!r} dataset",
                code="E410",
            )
        return None

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("boxes")[...], dtype=np.float32)

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._dataset("class_ids")[...], dtype=np.uint16)

    @property
    def instance_ids(self) -> npt.NDArray[np.uint64]:
        """Object ids, widened to ``uint64`` and never narrowed.

        §7.4 permits ``uint32`` or ``uint64`` on disk.  Reading through a
        ``uint32`` cast undid the encoder's widening: a stored ``uint64`` id of
        ``2**32 + 7`` came back as ``7``, and so did every ``instances()`` row
        and every longitudinal join built on it, with nothing raised.
        """
        return np.asarray(self._dataset("instance_ids")[...], dtype=np.uint64)

    @property
    def scores(self) -> npt.NDArray[np.float32] | None:
        node = self._dataset("scores", required=False)
        return None if node is None else np.asarray(node[...], dtype=np.float32)

    @property
    def has_masks(self) -> bool:
        return "mask_data" in self.group

    @property
    def n_objects(self) -> int:
        return int(self._dataset("boxes").shape[0])

    def crop(self, index: int) -> npt.NDArray[np.bool_] | None:
        """Decode one object's bbox-local mask."""
        if not self.has_masks:
            return None
        offsets = self._dataset("mask_offsets")
        shapes = self._dataset("mask_shapes")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        shape = tuple(int(v) for v in shapes[index])
        packed = np.asarray(self._dataset("mask_data")[start:stop], dtype=np.uint8)
        n = int(np.prod(shape, dtype=np.int64))
        return np.unpackbits(packed)[:n].astype(bool).reshape(shape)

    def instances(self) -> Iterator[Instance]:
        boxes = self.boxes
        classes = self.object_class_ids
        ids = self.instance_ids
        scores = self.scores
        for i in range(boxes.shape[0]):
            yield Instance(
                index=i,
                instance_id=int(ids[i]),
                class_id=int(classes[i]),
                box=boxes[i],
                mask=self.crop(i),
                score=None
                if scores is None or not np.isfinite(scores[i])
                else float(scores[i]),
            )

    def instance(self, instance_id: int) -> Instance:
        for obj in self.instances():
            if obj.instance_id == instance_id:
                return obj
        raise KeyError(f"annotation {self.ann_id!r} has no instance {instance_id}")

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        shape = self._roi_shape(roi)
        out = np.zeros(shape, dtype=bool)
        classes = self.object_class_ids
        boxes = self.boxes
        for index in np.flatnonzero(classes == class_id):
            # Unclipped, deliberately: these slices are the coordinate frame the
            # stored crop was cut in, and `box_to_slices(..., grid_shape)` moves
            # `start` for a box hanging off the near edge.  The crop was then
            # read from its own element 0 rather than from the first in-bounds
            # one, shifting the decoded mask by exactly the overhang --- silently,
            # for boxes a resample pushed slightly out of bounds, which is the
            # case the clip was added for.  Intersecting with `roi` below does
            # the clipping, because `roi` is already inside the array.
            obj_slices = box_to_slices(boxes[index])
            local: list[slice] = []
            target: list[slice] = []
            empty = False
            for axis, (want, have) in enumerate(zip(roi, obj_slices, strict=True)):
                lo = max(want.start, have.start)
                hi = min(want.stop, have.stop)
                if hi <= lo:
                    empty = True
                    break
                local.append(slice(lo - have.start, hi - have.start))
                target.append(slice(lo - want.start, hi - want.start))
                del axis
            if empty:
                continue
            crop = self.crop(int(index))
            if crop is None:
                out[tuple(target)] = True
            else:
                out[tuple(target)] |= crop[tuple(local)]
        return out

    def tracking(self) -> dict[int, int]:
        """``instance_id -> class_id`` --- the join key for longitudinal tracking."""
        return {
            int(i): int(c)
            for i, c in zip(self.instance_ids, self.object_class_ids, strict=True)
        }

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["objects"] = self.n_objects
        out["has_masks"] = self.has_masks
        out["instance_ids"] = [int(v) for v in self.instance_ids]
        return out


__all__ = [
    "InstanceInput",
    "InstancesAnnotation",
    "encode_instances",
    "instances_from_masks",
]
