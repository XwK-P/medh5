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
    """Low-resolution "is there anything in this block" map for rejection sampling."""
    coarse = tuple(max(1, -(-n // factor)) for n in mask.shape)
    out = np.zeros(coarse, dtype=bool)
    for block in np.ndindex(*coarse):
        window = tuple(
            slice(i * factor, min((i + 1) * factor, n))
            for i, n in zip(block, mask.shape, strict=True)
        )
        out[block] = bool(mask[window].any())
    return out


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
    window = annotation._roi(None)  # noqa: SLF001 - same-package internal
    n_spatial = len(annotation.spatial_shape)

    counts = np.zeros(len(ids), dtype=np.int64)
    bboxes = np.zeros((len(ids), n_spatial, 2), dtype=np.float32)
    coords: dict[int, npt.NDArray[np.int32]] = {}
    occ_planes: list[npt.NDArray[np.bool_]] = []

    for i, class_id in enumerate(ids):
        mask = annotation._dense_class(class_id, window)  # noqa: SLF001
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


def write_index(root: h5py.Group, payload: IndexPayload) -> h5py.Group:
    """Write an index entry under ``index/<ann_id>``."""
    index_root = root.require_group("index")
    if payload.ann_id in index_root:
        del index_root[payload.ann_id]
    group = index_root.create_group(payload.ann_id)
    group.create_dataset("class_ids", data=payload.class_ids)
    group.create_dataset("voxel_counts", data=payload.voxel_counts)
    group.create_dataset("class_bboxes", data=payload.class_bboxes)
    coords = group.create_group("fg_coords")
    for class_id, arr in payload.fg_coords.items():
        coords.create_dataset(str(class_id), data=arr)
    if payload.occupancy is not None:
        group.create_dataset("occupancy", data=payload.occupancy)
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
    """Reader for one ``index/<ann_id>`` entry."""

    __slots__ = ("ann_id", "group")

    def __init__(self, ann_id: str, group: h5py.Group) -> None:
        self.ann_id = ann_id
        self.group = group

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.group["class_ids"][...])

    @property
    def source_digest(self) -> str | None:
        value = self.group.attrs.get("source_digest")
        return as_str(value) if value is not None else None

    @property
    def max_coords(self) -> int:
        return as_int(self.group.attrs.get("max_coords", DEFAULT_MAX_COORDS))

    @property
    def voxel_counts(self) -> dict[int, int]:
        counts = self.group["voxel_counts"][...]
        return {cid: int(n) for cid, n in zip(self.class_ids, counts, strict=True)}

    def bbox(self, class_id: int) -> npt.NDArray[np.float32] | None:
        position = self.class_ids.index(int(class_id))
        box = np.asarray(self.group["class_bboxes"][position], dtype=np.float32)
        return None if np.isnan(box).any() else box

    def coords(self, class_id: int) -> npt.NDArray[np.int32]:
        node = self.group["fg_coords"]
        key = str(int(class_id))
        if key not in node:
            raise KeyError(
                f"index {self.ann_id!r} has no coordinates for class {class_id}"
            )
        return np.asarray(node[key][...], dtype=np.int32)

    def sample_foreground(
        self, class_id: int, n: int = 1, rng: np.random.Generator | None = None
    ) -> npt.NDArray[np.int32]:
        """Draw *n* foreground voxel coordinates in O(1) time and memory."""
        pool = self.coords(class_id)
        if pool.shape[0] == 0:
            raise MEDH5ValidationError(
                f"index {self.ann_id!r}: class {class_id} has no foreground voxels"
            )
        generator = rng if rng is not None else np.random.default_rng()
        picks = generator.integers(0, pool.shape[0], size=n)
        drawn: npt.NDArray[np.int32] = pool[picks]
        return drawn

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
