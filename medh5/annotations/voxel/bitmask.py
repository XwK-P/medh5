"""``bitmask``: one bit per class per voxel, packed into ``uint64`` planes (§7.3).

Raw cost is ``8 * ceil(C/64)`` bytes per voxel regardless of how many classes
are actually present, versus ``2L`` for ``uint16`` layers --- so ``layers`` wins
on size while ``L < 4 * ceil(C/64)``.  What ``bitmask`` wins is the "which
classes are at this voxel" query: O(P) reads independent of C, against one read
per layer.

Bit order is LSB-first within each ``uint64`` and the value is interpreted with
**integer shifts**, never byte offsets, so the encoding is byte-order
independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5.annotations.base import AnnotationHeader, VoxelAnnotation
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel.payload import Masks, normalize_masks, popcounts
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.labels.labelset import LabelSet

BITS_PER_PLANE = 64


def encode_bitmask(
    masks: Masks, spatial_shape: tuple[int, ...] | None = None
) -> AnnotationPayload:
    """Pack class masks into ``ceil(C/64)`` ``uint64`` bitplanes."""
    resolved, shape = normalize_masks(masks, spatial_shape)
    class_ids = tuple(sorted(resolved))
    n_planes = max(1, -(-len(class_ids) // BITS_PER_PLANE))
    data = np.zeros((n_planes, *shape), dtype=np.uint64)
    for position, class_id in enumerate(class_ids):
        plane, bit = divmod(position, BITS_PER_PLANE)
        data[plane][resolved[class_id]] |= np.uint64(1) << np.uint64(bit)
    return AnnotationPayload(
        kind="bitmask",
        datasets={
            "data": data,
            "bit_class_ids": np.asarray(class_ids, dtype=np.uint16),
        },
        attrs={},
        stacked_axes=1,
        class_ids=class_ids,
    )


class BitmaskAnnotation(VoxelAnnotation):
    """Reader for ``kind = "bitmask"``.

    ``bit_class_ids`` is read once per open annotation and kept, as
    ``layers`` and ``instances`` keep theirs and for the same reason: a
    ``Sample`` is read-only and ``amend`` replaces the inode, so the table
    cannot change under an open reader, while re-reading it per accessor cost
    one HDF5 read per class on every ``dense()``.
    """

    __slots__ = ("_bit_table",)

    def __init__(
        self,
        ann_id: str,
        group: h5py.Group,
        header: AnnotationHeader,
        grids: Mapping[str, Grid] | None = None,
        label_set: LabelSet | None = None,
    ) -> None:
        super().__init__(ann_id, group, header, grids, label_set)
        self._bit_table: tuple[npt.NDArray[np.uint16], dict[int, int]] | None = None

    def _table(self) -> tuple[npt.NDArray[np.uint16], dict[int, int]]:
        """``(bit_class_ids, class id -> bit position)``, read once."""
        if self._bit_table is None:
            try:
                ids = np.asarray(self.group["bit_class_ids"][...], dtype=np.uint16)
            except KeyError:
                raise MEDH5ValidationError(
                    f"annotation {self.ann_id!r}: `bitmask` requires `bit_class_ids`",
                    code="E410",
                ) from None
            self._bit_table = (ids, {int(c): i for i, c in enumerate(ids)})
        return self._bit_table

    @property
    def data(self) -> Any:
        try:
            return self.group["data"]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `bitmask` requires a `data` dataset",
                code="E410",
            ) from None

    @property
    def bit_class_ids(self) -> npt.NDArray[np.uint16]:
        return self._table()[0]

    @property
    def n_planes(self) -> int:
        return int(self.data.shape[0])

    @property
    def position_of(self) -> dict[int, int]:
        return self._table()[1]

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        position = self.position_of.get(class_id)
        if position is None:
            return np.zeros(self._roi_shape(roi), dtype=bool)
        plane, bit = divmod(position, BITS_PER_PLANE)
        block = np.asarray(self.data[(plane, *roi)], dtype=np.uint64)
        hit: npt.NDArray[np.bool_] = (
            block & (np.uint64(1) << np.uint64(bit))
        ) != np.uint64(0)
        return hit

    def dense(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """``(C, *roi)`` occupancy, reading each bitplane once.

        One ``uint64`` plane carries 64 classes, so asking per class re-reads
        and re-decompresses the same words up to 64 times.  Grouping by plane
        makes a 200-class read four plane reads and 200 mask operations on
        data already in cache.
        """
        ids = self.resolve_classes(classes)
        window = self._roi(roi)
        out = np.zeros((len(ids), *self._roi_shape(window)), dtype=bool)
        by_plane: dict[int, list[tuple[int, int]]] = {}
        for position, class_id in enumerate(ids):
            slot = self.position_of.get(class_id)
            if slot is None:
                continue
            plane, bit = divmod(slot, BITS_PER_PLANE)
            by_plane.setdefault(plane, []).append((position, bit))
        for plane, entries in sorted(by_plane.items()):
            block = np.asarray(self.data[(plane, *window)], dtype=np.uint64)
            for position, bit in entries:
                out[position] = (block & (np.uint64(1) << np.uint64(bit))) != np.uint64(
                    0
                )
        return out

    def _counts_from_planes(self) -> dict[int, int] | None:
        """One population count per plane answers all 64 of its classes."""
        ids = self.bit_class_ids
        if ids.size == 0:
            return {}
        counts = popcounts(self.data)
        out: dict[int, int] = {}
        for position, class_id in enumerate(ids):
            plane, bit = divmod(position, BITS_PER_PLANE)
            if plane < counts.shape[0]:
                out[int(class_id)] = int(counts[plane, bit])
        return out

    def classes_at(self, voxel: tuple[int, ...]) -> tuple[int, ...]:
        """Every class present at one voxel, in O(P) reads.

        This is the query ``bitmask`` exists for: ``layers`` answers it in O(L)
        reads and per-class dense encodings in O(C).
        """
        window = tuple(slice(int(v), int(v) + 1) for v in voxel)
        ids = self.bit_class_ids
        out: list[int] = []
        for plane in range(self.n_planes):
            word = int(np.asarray(self.data[(plane, *window)]).reshape(-1)[0])
            if word == 0:
                continue
            for bit in range(BITS_PER_PLANE):
                if word >> bit & 1:
                    position = plane * BITS_PER_PLANE + bit
                    if position < ids.size:
                        out.append(int(ids[position]))
        return tuple(out)

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["planes"] = self.n_planes
        return out


__all__ = ["BITS_PER_PLANE", "BitmaskAnnotation", "encode_bitmask"]
