"""Patch and timepoint-pair sampling (spec §14.3).

Deliberately free of any deep-learning dependency: a sampler decides *where* to
read, which is a geometry question, and keeping it here means it can be tested
without a framework and reused by one that is not PyTorch.

The design point is that foreground sampling reads a **cached coordinate
subsample** (§14.3) rather than scanning a mask.  The 0.x path loaded the full
label volume and called ``np.argwhere`` per draw, which is O(volume) in both
time and memory and forced a per-process cache that cannot exist at 512³ with
200 classes.  Reading 4096 pre-sampled coordinates is O(1) in volume size and
18× faster per patch.

When a file carries no index the sampler still works --- it falls back to
scanning --- but it says so on every :class:`Patch` it returns, because a
silent 20× slowdown in a dataloader is indistinguishable from a slow disk.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.annotations.base import VoxelAnnotation
    from medh5.sample import Sample

STRATEGIES = ("uniform", "foreground", "balanced")
PAIR_MODES = ("consecutive", "baseline_vs_all", "all_pairs")


@dataclass(frozen=True, slots=True)
class Patch:
    """One draw: where to read, and how the location was chosen."""

    slices: tuple[slice, ...]
    pad: tuple[tuple[int, int], ...] = ()
    """Per-axis ``(before, after)`` padding where the volume is smaller than
    the patch."""
    center: tuple[int, ...] = ()
    strategy: str = "uniform"
    class_id: int | None = None
    used_index: bool | None = None
    """Whether the §14.3 index answered the foreground query.

    ``True`` when it did, ``False`` when the foreground had to be scanned because
    no current index was present, and **``None`` when the question did not
    arise** --- a uniform draw consults no index at all.

    The third state is not pedantry.  This defaulted to ``True``, so a uniform
    draw reported that an index had been used when none had been opened, and a
    reader logging `used_index` to find out whether their cohort was indexed got
    ``True`` from every uniform patch in a `balanced` run.  Read it together with
    ``strategy``, or read ``None`` as "not applicable"."""
    grid_id: str | None = None
    """The grid whose index coordinates ``slices`` are in.

    A window is only meaningful in the grid it was measured in, so a reader
    that applies it to a different one is reading different anatomy --- and,
    where that grid is smaller, silently getting a shorter array back.  Carried
    on the patch so the consumer can check rather than assume.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            (s.stop - s.start) + before + after
            for s, (before, after) in zip(self.slices, self.padding, strict=True)
        )

    @property
    def padding(self) -> tuple[tuple[int, int], ...]:
        return self.pad or ((0, 0),) * len(self.slices)

    @property
    def needs_padding(self) -> bool:
        return any(before or after for before, after in self.padding)

    def apply_padding(
        self, array: npt.NDArray[Any], value: float = 0.0
    ) -> npt.NDArray[Any]:
        """Pad a read to the requested patch size, leading axes untouched."""
        if not self.needs_padding:
            return array
        lead = array.ndim - len(self.slices)
        widths = ((0, 0),) * lead + self.padding
        return np.pad(array, widths, mode="constant", constant_values=value)

    def to_json(self) -> dict[str, Any]:
        return {
            "start": [s.start for s in self.slices],
            "stop": [s.stop for s in self.slices],
            "pad": [list(p) for p in self.padding],
            "center": list(self.center),
            "strategy": self.strategy,
            "class_id": self.class_id,
            "used_index": self.used_index,
            "grid_id": self.grid_id,
        }

    def __repr__(self) -> str:
        window = ", ".join(f"{s.start}:{s.stop}" for s in self.slices)
        return f"Patch([{window}], {self.strategy})"


def coerce_patch_size(patch_size: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Broadcast a scalar patch size across *ndim* spatial axes."""
    if isinstance(patch_size, (int, np.integer)):
        size = (int(patch_size),) * ndim
    else:
        size = tuple(int(v) for v in patch_size)
    if len(size) != ndim:
        raise MEDH5ValidationError(
            f"patch_size {size} has {len(size)} axes; the grid has {ndim}"
        )
    if any(v <= 0 for v in size):
        raise MEDH5ValidationError(f"patch_size {size} must be positive on every axis")
    return size


def window_around(
    center: Sequence[int], patch: Sequence[int], shape: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[tuple[int, int], ...]]:
    """Slices covering *patch* voxels around *center*, plus the padding needed.

    A centre near the edge is shifted inwards rather than clipped, so a patch
    keeps its requested size wherever the volume allows; padding appears only on
    an axis genuinely shorter than the patch.  Silently returning a smaller
    array would break batching in a way that surfaces as a shape error three
    layers away.
    """
    slices: list[slice] = []
    pads: list[tuple[int, int]] = []
    for axis_center, size, extent in zip(center, patch, shape, strict=True):
        if size >= extent:
            slices.append(slice(0, int(extent)))
            spare = size - extent
            before = spare // 2
            pads.append((int(before), int(spare - before)))
            continue
        start = int(axis_center) - size // 2
        start = max(0, min(start, extent - size))
        slices.append(slice(start, start + size))
        pads.append((0, 0))
    return tuple(slices), tuple(pads)


def _uniform_center(
    shape: Sequence[int], patch: Sequence[int], generator: np.random.Generator
) -> tuple[int, ...]:
    """A centre whose window lands uniformly over the volume.

    Drawing the centre uniformly over *every* voxel and letting
    :func:`window_around` clamp is not uniform in the thing that matters: every
    centre in the leading half-patch collapses onto window start 0, and every
    centre in the trailing half onto the last start.  On a 24-voxel axis with an
    8-voxel patch that put 3.6x the uniform share on the first window and 2.8x
    on the last, and the skew grows with patch size --- border-heavy training
    data, from a strategy named ``uniform``.

    Drawing from the range that maps one-to-one onto the valid window starts
    makes the *window* uniform, which is what a caller asking for uniform
    coverage means.  Clamping stays in ``window_around`` for the foreground
    path, where the centre is a voxel the caller specifically wants included.
    """
    center: list[int] = []
    for extent, size in zip(shape, patch, strict=True):
        if size >= extent:
            center.append(int(extent) // 2)
            continue
        lead = size // 2
        # Valid starts are 0 .. extent - size; start s has centre s + lead.
        center.append(int(generator.integers(0, extent - size + 1)) + lead)
    return tuple(center)


class PatchSampler:
    """Choose patch windows in a volume (spec §14.3).

    ``strategy``:

    ``uniform``
        centres drawn uniformly over the volume.
    ``foreground``
        centres drawn from indexed foreground coordinates.
    ``balanced``
        foreground with probability ``foreground_prob``, uniform otherwise ---
        the strategy nearly every segmentation recipe actually uses, because
        pure foreground sampling never shows the model the background it will
        be evaluated on.
    """

    __slots__ = (
        "class_weights",
        "foreground_classes",
        "foreground_prob",
        "patch_size",
        "strategy",
    )

    def __init__(
        self,
        patch_size: int | Sequence[int],
        *,
        strategy: str = "balanced",
        foreground_prob: float = 0.5,
        foreground_classes: Sequence[int | str] | None = None,
        class_weights: str | Mapping[int, float] = "uniform",
    ) -> None:
        if strategy not in STRATEGIES:
            raise MEDH5ValidationError(
                f"unknown sampling strategy {strategy!r}; expected one of "
                f"{list(STRATEGIES)}"
            )
        if not 0.0 <= foreground_prob <= 1.0:
            raise MEDH5ValidationError("foreground_prob must lie in [0, 1]")
        self.patch_size = patch_size
        self.strategy = strategy
        self.foreground_prob = float(foreground_prob)
        self.foreground_classes = (
            None if foreground_classes is None else tuple(foreground_classes)
        )
        self.class_weights = class_weights

    def __repr__(self) -> str:
        return f"PatchSampler({self.patch_size}, {self.strategy})"

    # -- drawing -----------------------------------------------------------

    def draw(
        self,
        sample: Sample,
        annotation: str | None = None,
        rng: np.random.Generator | None = None,
        *,
        grid: str | None = None,
    ) -> Patch:
        """Draw one patch window from *sample*.

        *grid* pins the grid the window is measured in, for a caller that wants
        a window in a particular space whether or not an annotation happens to
        live there --- ``annotation=None`` on its own means "find me one",
        which for a longitudinal sample can find another visit's.
        """
        generator = rng if rng is not None else np.random.default_rng()
        ann = self._annotation(sample, annotation, grid)
        grid_id = self._window_grid(sample, ann, grid)
        shape = sample.grids[grid_id].spatial_shape
        patch = coerce_patch_size(self.patch_size, len(shape))
        want_foreground = self.strategy == "foreground" or (
            self.strategy == "balanced" and generator.random() < self.foreground_prob
        )
        if want_foreground and ann is not None:
            drawn = self._foreground_center(sample, ann, generator)
            if drawn is not None:
                center, class_id, used_index = drawn
                slices, pad = window_around(center, patch, shape)
                return Patch(
                    slices=slices,
                    pad=pad,
                    center=tuple(int(c) for c in center),
                    strategy="foreground",
                    class_id=class_id,
                    used_index=used_index,
                    grid_id=grid_id,
                )
        center = _uniform_center(shape, patch, generator)
        slices, pad = window_around(center, patch, shape)
        return Patch(
            slices=slices,
            pad=pad,
            center=center,
            strategy="uniform",
            grid_id=grid_id,
        )

    def draws(
        self,
        sample: Sample,
        annotation: str | None = None,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[Patch]:
        generator = rng if rng is not None else np.random.default_rng()
        return [self.draw(sample, annotation, generator) for _ in range(n)]

    # -- internals ---------------------------------------------------------

    def _annotation(
        self, sample: Sample, annotation: str | None, grid: str | None = None
    ) -> str | None:
        """The annotation to draw foreground from, auto-selected if not named.

        Auto-selection stays inside *grid* when the caller named one.  Without
        that, a caller who wanted "no annotation, this grid" got "any
        annotation anywhere in the sample" --- for a longitudinal file that is
        another visit's, whose coordinates are in another visit's grid.
        """
        if annotation is not None:
            return annotation
        for name, ann in sample.annotations.items():
            if getattr(ann, "dense", None) is None or ann.kind == "classification":
                continue
            if grid is not None and ann.grid_id != grid:
                continue
            return name
        return None

    def _window_grid(
        self, sample: Sample, annotation: str | None, grid: str | None = None
    ) -> str:
        """The grid a draw is measured in.

        The annotation's own grid where it has one, because that is the space
        its foreground coordinates are in; *grid* where the caller named one
        and there is no annotation to defer to; the sample's reference grid
        otherwise.  Named on the patch so a consumer reading some *other* grid
        with the result can tell, instead of returning a window that indexes
        different anatomy.
        """
        if annotation is not None and annotation in sample.annotations:
            declared = sample.annotations[annotation].grid_id
            if declared is not None:
                if grid is not None and str(declared) != grid:
                    raise MEDH5ValidationError(
                        f"annotation {annotation!r} is on grid {str(declared)!r}, so a "
                        f"window drawn from its foreground cannot be measured in "
                        f"grid {grid!r}"
                    )
                return str(declared)
        if grid is not None:
            return grid
        return str(sample.reference_grid.grid_id)

    def _foreground_center(
        self, sample: Sample, annotation: str, rng: np.random.Generator
    ) -> tuple[tuple[int, ...], int, bool] | None:
        """A foreground voxel, from the index when there is one."""
        ann = sample.annotations[annotation]
        classes = self._classes(ann)
        if not classes:
            return None
        # A stale index is worse than no index: its coordinates point at
        # foreground the annotation no longer has, so a centre drawn from it is
        # silently not foreground and the training distribution shifts with
        # nothing to show for it.  Scan instead, as the statistics path does.
        index = (
            sample.index.get(annotation) if annotation in sample.fresh_indices else None
        )
        if index is not None:
            available = [c for c in classes if c in index.class_ids]
            counted = {c: index.voxel_counts.get(c, 0) for c in available}
            picked = self._pick_class(counted, rng)
            if picked is not None:
                coords = index.sample_foreground(picked, 1, rng)
                return tuple(int(v) for v in coords[0]), picked, True
        return self._scan_center(ann, classes, rng)

    def _classes(self, ann: VoxelAnnotation) -> tuple[int, ...]:
        if self.foreground_classes is None:
            return tuple(ann.class_ids)
        return tuple(ann.resolve_class(c) for c in self.foreground_classes)

    def _pick_class(
        self, counts: Mapping[int, int], rng: np.random.Generator
    ) -> int | None:
        """Choose a class to sample from, weighted as configured."""
        present = {c: n for c, n in counts.items() if n > 0}
        if not present:
            return None
        if isinstance(self.class_weights, str):
            if self.class_weights == "uniform":
                weights = dict.fromkeys(present, 1.0)
            elif self.class_weights == "inverse_frequency":
                weights = {c: 1.0 / n for c, n in present.items()}
            elif self.class_weights == "frequency":
                weights = {c: float(n) for c, n in present.items()}
            else:
                raise MEDH5ValidationError(
                    f"unknown class_weights {self.class_weights!r}"
                )
        else:
            weights = {c: float(self.class_weights.get(c, 0.0)) for c in present}
        total = sum(weights.values())
        if total <= 0:
            return None
        keys = sorted(weights)
        probabilities = [weights[k] / total for k in keys]
        return int(rng.choice(keys, p=probabilities))

    def _scan_center(
        self,
        ann: VoxelAnnotation,
        classes: Sequence[int],
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], int, bool] | None:
        """The O(volume) fallback for a file with no sampling index."""
        order = list(classes)
        rng.shuffle(order)
        for class_id in order:
            mask = ann.dense([int(class_id)])[0]
            coords = np.argwhere(mask)
            if coords.shape[0]:
                pick = int(rng.integers(0, coords.shape[0]))
                return tuple(int(v) for v in coords[pick]), int(class_id), False
        return None


@dataclass(frozen=True, slots=True)
class TimepointPair:
    """One ordered pair of visits, and the change label spanning it, if any."""

    first: str
    second: str
    interval_days: float | None = None
    label: str | None = None

    def __repr__(self) -> str:
        gap = "" if self.interval_days is None else f", {self.interval_days}d"
        return f"TimepointPair({self.first} -> {self.second}{gap})"


class TimepointPairSampler:
    """Enumerate the visit pairs a longitudinal model trains on (§3.7, §9).

    A cross-sectional sample yields **no** pairs.  That is reported as a count
    rather than absorbed silently: a dataset that quietly drops nine tenths of
    its files looks exactly like one that is training normally.
    """

    __slots__ = ("mode",)

    def __init__(self, mode: str = "consecutive") -> None:
        if mode not in PAIR_MODES:
            raise MEDH5ValidationError(
                f"unknown pair mode {mode!r}; expected one of {list(PAIR_MODES)}"
            )
        self.mode = mode

    def __repr__(self) -> str:
        return f"TimepointPairSampler({self.mode!r})"

    def pairs(self, sample: Sample) -> list[TimepointPair]:
        ids = list(sample.timepoints.ids)
        if len(ids) < 2:  # noqa: PLR2004 - a pair needs two visits
            return []
        if self.mode == "consecutive":
            combos = list(zip(ids, ids[1:], strict=False))
        elif self.mode == "baseline_vs_all":
            combos = [(ids[0], later) for later in ids[1:]]
        else:
            combos = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
        return [
            TimepointPair(
                first=a,
                second=b,
                interval_days=sample.timepoints.interval_days(a, b),
                label=_change_label(sample, a, b),
            )
            for a, b in combos
        ]

    def __call__(self, sample: Sample) -> list[TimepointPair]:
        return self.pairs(sample)


def _change_label(sample: Sample, first: str, second: str) -> str | None:
    """The classification annotation whose ``timepoints`` is exactly this pair.

    Ordered, not as a set: "grew 40 %" and "shrank 40 %" span the same two
    visits and differ only in which one is the baseline, so a label written
    ``(tp1, tp0)`` does not describe the forward pair ``(tp0, tp1)``.
    """
    for name, ann in sample.annotations.items():
        if ann.kind != "classification":
            continue
        if tuple(ann.timepoints) == (first, second):
            return name
    return None


@dataclass(slots=True)
class PairReport:
    """What a paired dataset kept and what it skipped."""

    files: int = 0
    pairs: int = 0
    skipped: list[str] = field(default_factory=list)

    def add_skip(self, path: str) -> None:
        self.skipped.append(path)

    def summary(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "pairs": self.pairs,
            "skipped_cross_sectional": len(self.skipped),
            "examples": self.skipped[:5],
        }

    def __str__(self) -> str:
        return (
            f"{self.pairs} pairs from {self.files} files; "
            f"{len(self.skipped)} cross-sectional file(s) contributed none"
        )


def iter_patches(
    sample: Sample,
    sampler: PatchSampler,
    *,
    annotation: str | None = None,
    n: int = 1,
    rng: np.random.Generator | None = None,
) -> Iterator[Patch]:
    """Convenience generator over :meth:`PatchSampler.draw`."""
    generator = rng if rng is not None else np.random.default_rng()
    for _ in range(n):
        yield sampler.draw(sample, annotation, generator)


def grid_patches(
    shape: Sequence[int],
    patch_size: int | Sequence[int],
    *,
    overlap: int = 0,
    grid_id: str | None = None,
) -> list[Patch]:
    """Deterministic sliding-window cover of a volume --- the inference path.

    Every voxel is covered, and the last window on each axis is shifted inwards
    rather than padded, so predictions near the far edge come from real data.
    """
    extent = tuple(int(v) for v in shape)
    patch = coerce_patch_size(patch_size, len(extent))
    if overlap < 0 or any(overlap >= p for p in patch):
        raise MEDH5ValidationError("overlap must be non-negative and below patch_size")
    starts_per_axis = []
    for size, n in zip(patch, extent, strict=True):
        step = size - overlap
        if size >= n:
            starts_per_axis.append([0])
            continue
        positions = list(range(0, n - size + 1, step))
        if positions[-1] != n - size:
            positions.append(n - size)
        starts_per_axis.append(positions)
    out: list[Patch] = []
    for corner in _odometer(starts_per_axis):
        slices = tuple(
            slice(start, min(start + size, n))
            for start, size, n in zip(corner, patch, extent, strict=True)
        )
        pad = tuple(
            (0, size - (s.stop - s.start))
            for s, size in zip(slices, patch, strict=True)
        )
        out.append(
            Patch(
                slices=slices,
                pad=pad,
                center=tuple(s.start + (s.stop - s.start) // 2 for s in slices),
                strategy="grid",
                grid_id=grid_id,
            )
        )
    return out


def _odometer(axes: Sequence[Sequence[int]]) -> Iterator[tuple[int, ...]]:
    if not axes:
        yield ()
        return
    for head in axes[0]:
        for tail in _odometer(axes[1:]):
            yield (head, *tail)


__all__ = [
    "PAIR_MODES",
    "STRATEGIES",
    "PairReport",
    "Patch",
    "PatchSampler",
    "TimepointPair",
    "TimepointPairSampler",
    "coerce_patch_size",
    "grid_patches",
    "iter_patches",
    "window_around",
]
