"""The sampling index: what a patch sampler would otherwise recompute (spec §14.3).

Foreground patch sampling in 0.x loaded a full mask and ran ``np.argwhere`` ---
9.2 ms and memory proportional to the volume, per sample, per epoch.  A cached
subsample of foreground coordinates answers the same question in 0.52 ms and 48
KiB, and, more importantly, in **O(1) in volume size**: the 0.x approach cached
``argwhere`` output per (file, class) in process memory, which at 512^3 with 200
classes is tens of GiB and simply cannot exist.

Every index object carries ``source_digest``, so a stale entry is detectable
rather than silently wrong.  Readers ignore a stale entry and rebuild; it is not
a file error, and there is no invalidation protocol to get wrong (§13.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_int, as_str, set_attrs
from medh5.annotations.base import VoxelAnnotation
from medh5.errors import MEDH5ValidationError
from medh5.storage.codecs import CodecProfile, dataset_kwargs

DEFAULT_MAX_COORDS = 4096
"""Coordinates cached per class.

Measured sufficient: with 4096 uniformly drawn foreground centres per class a
sampler's empirical spatial distribution is indistinguishable from sampling the
full foreground, while the index stays 48 KiB per class instead of megabytes.
"""

DEFAULT_OCCUPANCY_FACTOR = 8
SLAB_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class IndexPayload:
    """The datasets of one annotation's index entry."""

    ann_id: str
    class_ids: npt.NDArray[np.uint16]
    voxel_counts: npt.NDArray[np.int64]
    class_bboxes: npt.NDArray[np.float32]
    fg_coords: dict[int, npt.NDArray[np.int32]]
    occupancy: npt.NDArray[np.bool_] | None = None
    source_digest: str | None = None
    max_coords: int = DEFAULT_MAX_COORDS
    seed: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


def _sample_foreground_coords(
    mask: npt.NDArray[np.bool_], max_coords: int, rng: np.random.Generator
) -> npt.NDArray[np.int32]:
    """Uniformly subsample foreground voxel coordinates in bounded memory.

    Picks ``max_coords`` ordinals out of the foreground count first, then walks
    the volume in slabs collecting exactly those --- so peak memory is one slab
    plus the sample, not one index per foreground voxel.
    """
    total = int(mask.sum())
    if total == 0:
        return np.zeros((0, mask.ndim), dtype=np.int32)
    if total <= max_coords:
        return np.argwhere(mask).astype(np.int32)

    picks = np.sort(rng.choice(total, size=max_coords, replace=False))
    row_bytes = max(1, int(np.prod(mask.shape[1:], dtype=np.int64)))
    step = max(1, SLAB_BYTES // row_bytes)
    out = np.zeros((max_coords, mask.ndim), dtype=np.int32)
    filled = 0
    offset = 0
    for start in range(0, mask.shape[0], step):
        slab = mask[start : start + step]
        flat = np.flatnonzero(slab.reshape(-1))
        if flat.size:
            lo = np.searchsorted(picks, offset, side="left")
            hi = np.searchsorted(picks, offset + flat.size, side="left")
            if hi > lo:
                chosen = flat[picks[lo:hi] - offset]
                coords = np.stack(np.unravel_index(chosen, slab.shape), axis=1)
                coords[:, 0] += start
                out[filled : filled + coords.shape[0]] = coords
                filled += coords.shape[0]
        offset += flat.size
    return out[:filled]


def _occupancy(mask: npt.NDArray[np.bool_], factor: int) -> npt.NDArray[np.bool_]:
    """Low-resolution "is there anything in this block" map for rejection sampling.

    Pad to a multiple of the factor, reshape so each axis splits into (block,
    within-block), and reduce with ``any`` over the within-block axes.  The
    obvious Python loop over ``np.ndindex`` is one interpreted iteration and
    one array slice per *coarse voxel*: 80 ms per class at 256³ and about
    0.6 s at 512³, so ``build_index`` on a 200-class annotation spent minutes
    building a map measured in kilobytes.
    """
    shape = mask.shape
    coarse = tuple(max(1, -(-n // factor)) for n in shape)
    padding = tuple(
        (0, blocks * factor - n) for blocks, n in zip(coarse, shape, strict=True)
    )
    padded = (
        np.pad(mask, padding, constant_values=False)
        if any(after for _, after in padding)
        else mask
    )
    split: tuple[int, ...] = ()
    for blocks in coarse:
        split = (*split, blocks, factor)
    reduced: npt.NDArray[np.bool_] = padded.reshape(split).any(
        axis=tuple(range(1, len(split), 2))
    )
    return reduced


def build_index(
    annotation: VoxelAnnotation,
    *,
    classes: Sequence[int | str] | None = None,
    max_coords: int = DEFAULT_MAX_COORDS,
    occupancy: int | None = DEFAULT_OCCUPANCY_FACTOR,
    seed: int = 0,
    source_digest: str | None = None,
) -> IndexPayload:
    """Compute the sampling index for one voxel annotation."""
    from medh5.geometry.affine import slices_to_box

    ids = annotation.resolve_classes(classes)
    rng = np.random.default_rng(seed)
    window = annotation._roi(None)
    n_spatial = len(annotation.spatial_shape)

    counts = np.zeros(len(ids), dtype=np.int64)
    bboxes = np.zeros((len(ids), n_spatial, 2), dtype=np.float32)
    coords: dict[int, npt.NDArray[np.int32]] = {}
    occ_planes: list[npt.NDArray[np.bool_]] = []

    for i, class_id in enumerate(ids):
        mask = annotation._dense_class(class_id, window)
        counts[i] = int(mask.sum())
        if counts[i]:
            slices = []
            for axis in range(mask.ndim):
                axes = tuple(a for a in range(mask.ndim) if a != axis)
                present = np.flatnonzero(mask.any(axis=axes))
                slices.append(slice(int(present[0]), int(present[-1]) + 1))
            bboxes[i] = slices_to_box(slices)
        else:
            bboxes[i] = np.nan
        coords[int(class_id)] = _sample_foreground_coords(mask, max_coords, rng)
        if occupancy:
            occ_planes.append(_occupancy(mask, occupancy))

    return IndexPayload(
        ann_id=annotation.ann_id,
        class_ids=np.asarray(ids, dtype=np.uint16),
        voxel_counts=counts,
        class_bboxes=bboxes,
        fg_coords=coords,
        occupancy=np.stack(occ_planes) if occ_planes else None,
        source_digest=source_digest,
        max_coords=max_coords,
        seed=seed,
        stats={"total_foreground": int(counts.sum())},
    )


def write_index(
    root: h5py.Group,
    payload: IndexPayload,
    *,
    codec: str | CodecProfile | None = None,
) -> h5py.Group:
    """Write an index entry under ``index/<ann_id>``.

    Datasets go through the label codec, like every other dataset the writer
    stores, and ``occupancy`` is chunked one class per chunk --- the unit a
    reader asks about.  It was written uncompressed and contiguous: C × (N/8)³
    booleans, about 52 MB for 200 classes at 512³, of what is almost entirely
    ``False``.  Datasets under the compression threshold stay contiguous, as
    they do everywhere else.
    """
    index_root = root.require_group("index")
    if payload.ann_id in index_root:
        del index_root[payload.ann_id]
    group = index_root.create_group(payload.ann_id)

    def store(
        parent: h5py.Group,
        name: str,
        data: npt.NDArray[Any],
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        parent.create_dataset(
            name,
            data=data,
            **dataset_kwargs(
                data.shape, data.dtype, profile=codec, role="label", chunks=chunks
            ),
        )

    store(group, "class_ids", payload.class_ids)
    store(group, "voxel_counts", payload.voxel_counts)
    store(group, "class_bboxes", payload.class_bboxes)
    coords = group.create_group("fg_coords")
    for class_id, arr in payload.fg_coords.items():
        store(coords, str(class_id), arr)
    if payload.occupancy is not None:
        occupancy = payload.occupancy
        store(group, "occupancy", occupancy, (1, *occupancy.shape[1:]))
    set_attrs(
        group,
        {
            "source_digest": payload.source_digest,
            "max_coords": int(payload.max_coords),
            "seed": int(payload.seed),
        },
    )
    return group


class SamplingIndex:
    """Reader for one ``index/<ann_id>`` entry.

    The class table, the counts and each class's coordinate dataset are read
    once per open and kept: a :class:`~medh5.sample.Sample` is a read-only
    view, so they cannot change under it.  Reading them per call is what made
    the draw the design calls O(1) cost 190 HDF5 reads at 63 classes --- one
    ``class_ids`` read per class checked, and two more per count looked up ---
    the pattern P-02 and P-05 had already removed from the annotation readers.
    Coordinates are *not* all loaded: a draw reads the one row it picked.
    """

    __slots__ = (
        "_class_ids",
        "_coord_nodes",
        "_counts",
        "_positions",
        "ann_id",
        "group",
    )

    def __init__(self, ann_id: str, group: h5py.Group) -> None:
        self.ann_id = ann_id
        self.group = group
        self._class_ids: tuple[int, ...] | None = None
        self._positions: dict[int, int] = {}
        self._counts: dict[int, int] | None = None
        self._coord_nodes: dict[int, h5py.Dataset] = {}

    @property
    def class_ids(self) -> tuple[int, ...]:
        if self._class_ids is None:
            self._class_ids = tuple(int(c) for c in self.group["class_ids"][...])
            self._positions = {c: i for i, c in enumerate(self._class_ids)}
        return self._class_ids

    def _position(self, class_id: int) -> int | None:
        """Row of *class_id* in the class table, or ``None``."""
        if self._class_ids is None:
            self.class_ids  # noqa: B018 - fills the table and its positions
        return self._positions.get(int(class_id))

    def has_class(self, class_id: int) -> bool:
        return self._position(class_id) is not None

    @property
    def source_digest(self) -> str | None:
        value = self.group.attrs.get("source_digest")
        return as_str(value) if value is not None else None

    @property
    def max_coords(self) -> int:
        return as_int(self.group.attrs.get("max_coords", DEFAULT_MAX_COORDS))

    @property
    def voxel_counts(self) -> dict[int, int]:
        if self._counts is None:
            counts = self.group["voxel_counts"][...]
            self._counts = {
                cid: int(n) for cid, n in zip(self.class_ids, counts, strict=True)
            }
        return dict(self._counts)

    def bbox(self, class_id: int) -> npt.NDArray[np.float32] | None:
        position = self._position(class_id)
        if position is None:
            raise ValueError(f"index {self.ann_id!r} has no class {class_id}")
        box = np.asarray(self.group["class_bboxes"][position], dtype=np.float32)
        return None if np.isnan(box).any() else box

    def _coord_node(self, class_id: int) -> h5py.Dataset:
        key = int(class_id)
        node = self._coord_nodes.get(key)
        if node is None:
            group = self.group["fg_coords"]
            if str(key) not in group:
                raise KeyError(
                    f"index {self.ann_id!r} has no coordinates for class {class_id}"
                )
            node = self._coord_nodes[key] = group[str(key)]
        return node

    def coords(self, class_id: int) -> npt.NDArray[np.int32]:
        return np.asarray(self._coord_node(class_id)[...], dtype=np.int32)

    def sample_foreground(
        self, class_id: int, n: int = 1, rng: np.random.Generator | None = None
    ) -> npt.NDArray[np.int32]:
        """Draw *n* foreground voxel coordinates in O(1) time and memory."""
        pool = self._coord_node(class_id)
        size = int(pool.shape[0])
        if size == 0:
            raise MEDH5ValidationError(
                f"index {self.ann_id!r}: class {class_id} has no foreground voxels"
            )
        generator = rng if rng is not None else np.random.default_rng()
        picks = generator.integers(0, size, size=n)
        if n == 1:
            # One row, not the pool: the draw that made sampling O(1) should
            # not read 4096 coordinates to use one of them.
            drawn: npt.NDArray[np.int32] = np.asarray(
                pool[int(picks[0])], dtype=np.int32
            )[None]
            return drawn
        rows: npt.NDArray[np.int32] = np.asarray(pool[...], dtype=np.int32)[picks]
        return rows

    def class_weights(self, mode: str = "inverse_frequency") -> dict[int, float]:
        """Sampling weights derived from ``voxel_counts``."""
        counts = self.voxel_counts
        if mode == "uniform":
            return dict.fromkeys(counts, 1.0)
        if mode != "inverse_frequency":
            raise MEDH5ValidationError(f"unknown weighting mode {mode!r}")
        weights = {c: (1.0 / n if n else 0.0) for c, n in counts.items()}
        total = sum(weights.values())
        return (
            {c: w / total for c, w in weights.items()}
            if total > 0
            else dict.fromkeys(counts, 0.0)
        )

    def is_current(self, annotation_digest: str) -> bool:
        return self.source_digest == annotation_digest

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.ann_id,
            "classes": list(self.class_ids),
            "voxel_counts": {str(k): v for k, v in self.voxel_counts.items()},
            "max_coords": self.max_coords,
            "has_occupancy": "occupancy" in self.group,
            "source_digest": self.source_digest,
        }


def read_indices(root: h5py.Group) -> dict[str, SamplingIndex]:
    if "index" not in root:
        return {}
    node = root["index"]
    return {name: SamplingIndex(name, node[name]) for name in sorted(node)}


__all__ = [
    "DEFAULT_MAX_COORDS",
    "DEFAULT_OCCUPANCY_FACTOR",
    "IndexPayload",
    "SamplingIndex",
    "build_index",
    "read_indices",
    "write_index",
]
