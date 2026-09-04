"""Overlap analysis, graph colouring and encoding selection (spec §7.6).

The defining constraint of medical segmentation is that a voxel may belong to
several classes at once --- a lesion inside a liver segment inside the liver ---
and that there may be hundreds of classes.  Storing one boolean volume per class
costs 3.57 MiB and 117 ms per 64^3 all-class patch read on the 200-structure
phantom; colouring the class **overlap graph** and storing one integer volume per
colour costs 0.55 MiB and 6.7 ms for the same content.

The colouring is the whole trick: classes that never co-occur at a voxel can
share one dense labelmap, so the number of volumes tracks the *overlap depth*
rather than the class count.  On the phantom, mean degree 3.4 coloured into 5
layers --- one volume per 40 classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.voxel.payload import Masks, normalize_masks

LOCALIZED_BBOX_FRACTION = 0.05
"""A class is 'localized' when its bounding box covers at most this much volume."""

SPARSE_FILL = 1e-3
"""Below this mean fill an ``instances`` encoding beats any dense one."""


@dataclass(slots=True)
class OverlapStats:
    """Measured properties of a set of class masks, and what they imply."""

    class_ids: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    counts: dict[int, int]
    edges: frozenset[tuple[int, int]]
    colouring: dict[int, int]
    localized: frozenset[int]
    n_labelled_voxels: int

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.spatial_shape, dtype=np.int64))

    @property
    def total_foreground(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def fill(self) -> float:
        """Mean per-class density: how much of the volume a class occupies."""
        if not self.class_ids or self.n_voxels == 0:
            return 0.0
        return self.total_foreground / (self.n_classes * self.n_voxels)

    @property
    def depth(self) -> float:
        """Mean number of labels on a labelled voxel."""
        if self.n_labelled_voxels == 0:
            return 0.0
        return self.total_foreground / self.n_labelled_voxels

    @property
    def n_layers(self) -> int:
        return (max(self.colouring.values()) + 1) if self.colouring else 0

    @property
    def mean_degree(self) -> float:
        if not self.class_ids:
            return 0.0
        return 2.0 * len(self.edges) / len(self.class_ids)

    @property
    def is_edgeless(self) -> bool:
        return not self.edges

    @property
    def n_planes(self) -> int:
        return -(-self.n_classes // 64) if self.n_classes else 0

    def summary(self) -> dict[str, Any]:
        return {
            "classes": self.n_classes,
            "voxels": self.n_voxels,
            "fill": self.fill,
            "depth": self.depth,
            "layers": self.n_layers,
            "planes": self.n_planes,
            "edges": len(self.edges),
            "mean_degree": self.mean_degree,
            "localized": sorted(self.localized),
            "counts": {str(k): v for k, v in sorted(self.counts.items())},
        }


def analyse(masks: Masks, spatial_shape: tuple[int, ...] | None = None) -> OverlapStats:
    """Measure counts, the overlap graph, a greedy colouring and localization.

    Pairwise overlap is computed only on voxels carrying more than one label.
    A naive implementation is O(C^2) full-volume ``AND``s --- 20 000 passes over
    a 4 M-voxel array for 200 classes.  Restricting to the multi-labelled voxels
    first makes the quadratic part run over a vector that is usually thousands of
    elements long instead of millions.
    """
    resolved, shape = normalize_masks(masks, spatial_shape)
    class_ids = tuple(sorted(resolved))
    counts = {cid: int(resolved[cid].sum()) for cid in class_ids}
    n_voxels = int(np.prod(shape, dtype=np.int64))

    depth_map = np.zeros(shape, dtype=np.uint16)
    for cid in class_ids:
        depth_map += resolved[cid]
    labelled = int(np.count_nonzero(depth_map))
    multi = depth_map > 1

    edges: set[tuple[int, int]] = set()
    if bool(multi.any()):
        picked = {cid: resolved[cid][multi] for cid in class_ids}
        active = [cid for cid in class_ids if bool(picked[cid].any())]
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                if bool(np.any(picked[a] & picked[b])):
                    edges.add((a, b) if a < b else (b, a))

    localized: set[int] = set()
    for cid in class_ids:
        if counts[cid] == 0:
            continue
        box_volume = _bbox_volume(resolved[cid])
        if box_volume <= LOCALIZED_BBOX_FRACTION * n_voxels:
            localized.add(cid)

    return OverlapStats(
        class_ids=class_ids,
        spatial_shape=shape,
        counts=counts,
        edges=frozenset(edges),
        colouring=greedy_colour(class_ids, edges),
        localized=frozenset(localized),
        n_labelled_voxels=labelled,
    )


def _bbox_volume(mask: npt.NDArray[np.bool_]) -> int:
    volume = 1
    for axis in range(mask.ndim):
        axes = tuple(i for i in range(mask.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=axes))
        if present.size == 0:
            return 0
        volume *= int(present[-1] - present[0] + 1)
    return volume


def greedy_colour(
    class_ids: Sequence[int], edges: frozenset[tuple[int, int]] | set[tuple[int, int]]
) -> dict[int, int]:
    """Colour the overlap graph greedily, highest degree first (spec §7.6).

    Greedy-by-degree is not optimal --- graph colouring is NP-hard --- but it is
    within one or two colours of optimal on the sparse, locally clustered graphs
    anatomy produces, and it is O(V + E).  The validator's W908 fires when the
    stored layer count is far from what this returns, so a hand-built file with a
    poor colouring is visible rather than merely inefficient.
    """
    neighbours: dict[int, set[int]] = {cid: set() for cid in class_ids}
    for a, b in edges:
        if a in neighbours and b in neighbours:
            neighbours[a].add(b)
            neighbours[b].add(a)
    order = sorted(class_ids, key=lambda c: (-len(neighbours[c]), c))
    colour: dict[int, int] = {}
    for cid in order:
        taken = {colour[n] for n in neighbours[cid] if n in colour}
        chosen = 0
        while chosen in taken:
            chosen += 1
        colour[cid] = chosen
    return {cid: colour[cid] for cid in sorted(class_ids)}


def layers_from_colouring(colouring: Mapping[int, int]) -> tuple[tuple[int, ...], ...]:
    """Group a colouring into per-layer class-id tuples, ascending."""
    if not colouring:
        return ()
    n_layers = max(colouring.values()) + 1
    buckets: list[list[int]] = [[] for _ in range(n_layers)]
    for cid in sorted(colouring):
        buckets[colouring[cid]].append(cid)
    return tuple(tuple(b) for b in buckets)


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Raw (pre-compression) bytes per encoding, for one set of masks."""

    labelmap: int | None
    layers: int
    bitmask: int
    instances: int
    probmap: int
    detail: dict[str, Any] = field(default_factory=dict)

    def best(self) -> str:
        options = {
            "layers": self.layers,
            "bitmask": self.bitmask,
            "instances": self.instances,
        }
        if self.labelmap is not None:
            options["labelmap"] = self.labelmap
        return min(options, key=lambda k: options[k])


def label_dtype_size(class_ids: Sequence[int], *, ignore: bool = False) -> int:
    """1 for ``uint8`` when the ids fit, else 2 for ``uint16`` (spec §7.1)."""
    ceiling = max(class_ids) if class_ids else 0
    return 1 if (ceiling <= 254 and not ignore) else 2


def cost_model(stats: OverlapStats, *, ignore: bool = False) -> CostModel:
    """Raw bytes each encoding would consume for the analysed masks.

    ``layers`` beats ``bitmask`` on size whenever ``L < 4 * ceil(C/64)`` for
    ``uint16`` layers --- the crossover the spec names --- and this makes that
    concrete rather than asserted.
    """
    n_voxels = stats.n_voxels
    itemsize = label_dtype_size(stats.class_ids, ignore=ignore)
    labelmap = n_voxels * itemsize if stats.is_edgeless else None
    layers = stats.n_layers * n_voxels * itemsize
    bitmask = stats.n_planes * n_voxels * 8
    instances = sum(_instance_cost(stats, cid) for cid in stats.class_ids)
    probmap = stats.n_classes * n_voxels * 2
    return CostModel(
        labelmap=labelmap,
        layers=layers,
        bitmask=bitmask,
        instances=instances,
        probmap=probmap,
        detail={
            "itemsize": itemsize,
            "n_layers": stats.n_layers,
            "n_planes": stats.n_planes,
            "crossover_layers": 4 * stats.n_planes * (2 // itemsize),
        },
    )


def _instance_cost(stats: OverlapStats, class_id: int) -> int:
    count = stats.counts.get(class_id, 0)
    if count == 0:
        return 0
    # bit-packed crop, approximated by the foreground count when the bbox is
    # tight; plus the per-object box/id/offset overhead.
    return count // 8 + 64


def select_encoding(
    masks: Masks | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    stats: OverlapStats | None = None,
    soft: bool = False,
    prefer: str | None = None,
    ignore: bool = False,
) -> tuple[str, OverlapStats]:
    """Choose an encoding by measurement (spec §7.6).

    Returns the chosen ``kind`` and the statistics that justified it, so a writer
    can record *why* --- an encoding chosen by a rule nobody can inspect is a
    rule nobody will trust.

    *ignore* says an in-band ignore region will be written with the masks.
    The reserved id is ``65535``, so ``labelmap`` and ``layers`` then need
    ``uint16`` planes whatever the class ids are (§7.1), and the crossover
    against ``bitmask`` moves: the choice has to be made with the region in
    the picture, or it is made for a payload that is not the one written.
    """
    if soft:
        measured = stats or analyse(masks or {}, spatial_shape)
        return "probmap", measured
    measured = stats if stats is not None else analyse(masks or {}, spatial_shape)
    if prefer is not None and prefer != "auto":
        return prefer, measured

    nonempty = [cid for cid in measured.class_ids if measured.counts[cid] > 0]
    all_localized = bool(nonempty) and all(
        cid in measured.localized for cid in nonempty
    )
    if all_localized and measured.fill < SPARSE_FILL:
        return "instances", measured
    if measured.is_edgeless:
        return "labelmap", measured
    itemsize = label_dtype_size(measured.class_ids, ignore=ignore)
    if measured.n_layers < (8 // itemsize) * measured.n_planes:
        return "layers", measured
    return "bitmask", measured


__all__ = [
    "LOCALIZED_BBOX_FRACTION",
    "SPARSE_FILL",
    "CostModel",
    "OverlapStats",
    "analyse",
    "cost_model",
    "greedy_colour",
    "label_dtype_size",
    "layers_from_colouring",
    "select_encoding",
]
