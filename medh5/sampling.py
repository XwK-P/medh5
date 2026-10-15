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

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.errors import MEDH5ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.sample import Sample

STRATEGIES: tuple[str, ...] = _core.SAMPLING_STRATEGIES
PAIR_MODES: tuple[str, ...] = _core.SAMPLING_PAIR_MODES

Seed = int | Sequence[int] | np.random.Generator | None
"""What seeds a draw: an integer, a sequence of integers --- ``(seed, epoch,
index)`` --- a ``numpy.random.Generator``, which supplies one seed and
advances, or ``None`` for fresh entropy.  The engine draws from its own
generator, so a seed gives the same patches from Python, Rust and the command
line."""


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

    ``True`` when it did, ``False`` when the labels were scanned instead ---
    the annotation has no index, its index is stale, or the index lacks a
    class the draw could pick --- and **``None`` when the question did not
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
    size: tuple[int, ...] = _core.sampling_coerce_patch_size(patch_size, ndim)
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
    found: tuple[tuple[slice, ...], tuple[tuple[int, int], ...]] = (
        _core.sampling_window_around(
            [int(v) for v in center], [int(v) for v in patch], [int(v) for v in shape]
        )
    )
    return found


_CONFIG = (
    "patch_size",
    "strategy",
    "foreground_prob",
    "foreground_classes",
    "class_weights",
)


class PatchSampler:
    """Choose patch windows in a volume (spec §14.3).

    ``strategy``:

    ``uniform``
        windows drawn uniformly over the volume --- uniform in the *window*,
        not the centre, so the border is not over-represented.
    ``foreground``
        centres drawn from indexed foreground coordinates.
    ``balanced``
        foreground with probability ``foreground_prob``, uniform otherwise ---
        the strategy nearly every segmentation recipe actually uses, because
        pure foreground sampling never shows the model the background it will
        be evaluated on.

    ``foreground_classes`` are the classes a foreground draw may target, by id
    or label-set key (every class of the annotation when ``None``).  A class
    the annotation cannot express has no voxels in it and counts zero, so one
    list can serve a whole cohort.  ``class_weights`` picks among them:
    ``"uniform"``, ``"inverse_frequency"``, ``"frequency"``, or a mapping of
    class id to a finite weight of at least 0 --- anything else is refused
    here, not at the first draw.

    A stale index is never trusted: its coordinates point at foreground the
    annotation no longer has, so the sampler scans instead --- and says so on
    the patch (``used_index=False``).
    """

    __slots__ = (*_CONFIG, "_built")
    _built: _core.PatchSamplerHandle | None

    def __init__(
        self,
        patch_size: int | Sequence[int],
        *,
        strategy: str = "balanced",
        foreground_prob: float = 0.5,
        foreground_classes: Sequence[int | str] | None = None,
        class_weights: str | Mapping[int, float] = "uniform",
    ) -> None:
        object.__setattr__(self, "_built", None)
        self.patch_size = patch_size
        self.strategy = strategy
        self.foreground_prob = float(foreground_prob)
        self.foreground_classes = (
            None if foreground_classes is None else tuple(foreground_classes)
        )
        self.class_weights = class_weights
        self._engine()  # refuse a bad configuration now, not at the first draw

    def __setattr__(self, name: str, value: Any) -> None:
        # Once configured, a change is checked as `__init__` checks one: set
        # after construction, a bad strategy was accepted and refused at the
        # next draw.  A refused change leaves the sampler as it was.
        configured = name in _CONFIG and all(hasattr(self, n) for n in _CONFIG)
        before = getattr(self, name, None)
        object.__setattr__(self, name, value)
        if name in _CONFIG:
            object.__setattr__(self, "_built", None)
        if configured:
            try:
                self._engine()
            except BaseException:
                object.__setattr__(self, name, before)
                object.__setattr__(self, "_built", None)
                raise

    def __getstate__(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _CONFIG}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_built", None)
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def _engine(self) -> Any:
        built = self._built
        if built is None:
            built = _core.PatchSamplerHandle(
                self.patch_size,
                strategy=self.strategy,
                foreground_prob=self.foreground_prob,
                foreground_classes=self.foreground_classes,
                class_weights=self.class_weights,
            )
            object.__setattr__(self, "_built", built)
        return built

    def __repr__(self) -> str:
        return str(self._engine().repr())

    # -- drawing -----------------------------------------------------------

    def draw(
        self,
        sample: Sample,
        annotation: str | None = None,
        rng: Seed = None,
        *,
        grid: str | None = None,
    ) -> Patch:
        """Draw one patch window from *sample*.

        *rng* seeds the draw (see :data:`Seed`).  *grid* pins the grid the
        window is measured in, for a caller that wants a window in a particular
        space whether or not an annotation happens to live there ---
        ``annotation=None`` on its own means "find me one", which for a
        longitudinal sample can find another visit's.
        """
        return Patch(**self._engine().draw(sample, annotation, rng, grid=grid))

    def draws(
        self,
        sample: Sample,
        annotation: str | None = None,
        n: int = 1,
        rng: Seed = None,
        *,
        grid: str | None = None,
    ) -> list[Patch]:
        """*n* draws from one generator seeded by *rng*, each measured in
        *grid* when one is named (as :meth:`draw` takes it)."""
        found = self._engine().draws(sample, annotation, n, rng, grid=grid)
        return [Patch(**f) for f in found]

    # -- the decisions a draw makes, for callers that need one alone ---------

    def _annotation(
        self, sample: Sample, annotation: str | None, grid: str | None = None
    ) -> str | None:
        """The annotation to draw foreground from, auto-selected if not named.

        Auto-selection stays inside *grid* when the caller named one, and never
        picks a ``mask`` (no classes, so it can only answer "no foreground").
        """
        found: str | None = self._engine().annotation(sample, annotation, grid)
        return found

    def _window_grid(
        self, sample: Sample, annotation: str | None, grid: str | None = None
    ) -> str:
        """The grid a draw is measured in: the annotation's own where it has
        one, *grid* where the caller named one, the reference grid otherwise."""
        found: str = self._engine().window_grid(sample, annotation, grid)
        return found

    def _pick_class(self, counts: Mapping[int, int], rng: Seed) -> int | None:
        """Choose a class to sample from, weighted as configured."""
        found: int | None = self._engine().pick_class(counts, rng)
        return found


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

    A pair's ``label`` is the classification annotation whose ``timepoints``
    is exactly that ordered pair: "grew 40 %" and "shrank 40 %" span the same
    two visits and differ only in which one is the baseline.
    """

    __slots__ = ("mode",)

    def __init__(self, mode: str = "consecutive") -> None:
        _core.sampling_check_pair_mode(mode)
        self.mode = mode

    def __repr__(self) -> str:
        return f"TimepointPairSampler({self.mode!r})"

    def pairs(self, sample: Sample) -> list[TimepointPair]:
        return [
            TimepointPair(first, second, interval, label)
            for first, second, interval, label in _core.sampling_pairs(
                self.mode, sample
            )
        ]

    def __call__(self, sample: Sample) -> list[TimepointPair]:
        return self.pairs(sample)


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
    rng: Seed = None,
    grid: str | None = None,
) -> Iterator[Patch]:
    """The patches of :meth:`PatchSampler.draws`, drawn one per ``next()``.

    A long run holds one patch at a time, not *n*.  The annotation and the
    grid are checked when this is called, before the first patch.
    """
    draws = sampler._engine().iter_draws(sample, annotation, n, rng, grid=grid)
    return (Patch(**fields) for fields in draws)


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
    return [
        Patch(**fields)
        for fields in _core.sampling_grid_patches(
            [int(v) for v in shape], patch_size, overlap=overlap, grid_id=grid_id
        )
    ]


BLEND_MODES: tuple[str, ...] = ("constant", "gaussian")
"""How overlapping patch predictions are weighted when stitched back:
``constant`` averages them, ``gaussian`` trusts each patch most at its centre,
where its model saw the most context."""

_FLOOR = 1e-6
"""The least weight, relative to the largest: a covered voxel is never divided
by zero, however far into a patch's corner it lies."""


def blend_weights(
    patch_size: Sequence[int],
    *,
    mode: str = "gaussian",
    sigma_scale: float = 0.125,
) -> npt.NDArray[np.float32]:
    """The weight of each voxel of a patch when patches are stitched back.

    ``gaussian`` is a separable Gaussian centred on the patch, with a standard
    deviation of ``sigma_scale`` times each axis's extent, scaled so its
    largest weight is 1; ``constant`` is 1 everywhere.  Every weight is at
    least ``1e-6`` of the largest.
    """
    size = tuple(int(n) for n in patch_size)
    if any(n < 1 for n in size):
        raise MEDH5ValidationError(f"a patch has at least one voxel per axis: {size}")
    if mode not in BLEND_MODES:
        raise MEDH5ValidationError(
            f"unknown blend mode {mode!r}; expected one of {list(BLEND_MODES)}"
        )
    if mode == "constant":
        return np.ones(size, dtype=np.float32)
    if not (math.isfinite(sigma_scale) and sigma_scale > 0):
        raise MEDH5ValidationError(
            f"sigma_scale is a positive fraction of the patch, not {sigma_scale!r}"
        )
    weights = np.ones((), dtype=np.float64)
    for n in size:
        x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        axis = np.exp(-0.5 * (x / (sigma_scale * n)) ** 2)
        weights = np.multiply.outer(weights, axis / axis.max())
    weights /= weights.max()
    return np.maximum(weights, _FLOOR).astype(np.float32)


class PatchAggregator:
    """Stitch patch predictions back into one volume.

    One dense accumulator over the volume's spatial shape (times
    ``channels``, when the predictions have a channel axis before the spatial
    ones), and one of weights: :meth:`add` weights a prediction by
    :func:`blend_weights`, cuts off the padding a patch carries past the
    volume, and adds it in; :meth:`result` divides.  The accumulator holds
    ``channels`` volumes of ``dtype`` plus one of weights, whatever the number
    of patches.

    A patch from another grid (``grid_id``, taken from the first patch when
    not given) is refused: its slices index other voxels.  ``count`` is the
    number of patches added, and ``coverage`` marks the voxels one covered.
    """

    __slots__ = (
        "_sum",
        "_weight",
        "_weights",
        "channels",
        "count",
        "coverage",
        "dtype",
        "grid_id",
        "mode",
        "sigma_scale",
        "spatial_shape",
    )

    def __init__(
        self,
        spatial_shape: Sequence[int],
        *,
        channels: int | None = None,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
        dtype: npt.DTypeLike = np.float32,
        grid_id: str | None = None,
    ) -> None:
        self.spatial_shape = tuple(int(n) for n in spatial_shape)
        self.channels = None if channels is None else int(channels)
        if self.channels is not None and self.channels < 1:
            raise MEDH5ValidationError("channels is at least 1, or None")
        blend_weights(
            (1,) * len(self.spatial_shape), mode=mode, sigma_scale=sigma_scale
        )
        self.mode = mode
        self.sigma_scale = float(sigma_scale)
        self.dtype = np.dtype(dtype)
        self.grid_id = grid_id
        lead = () if self.channels is None else (self.channels,)
        self._sum: npt.NDArray[Any] = np.zeros(
            (*lead, *self.spatial_shape), dtype=self.dtype
        )
        self._weight = np.zeros(self.spatial_shape, dtype=self.dtype)
        self._weights: dict[tuple[int, ...], npt.NDArray[np.float32]] = {}
        self.count = 0
        self.coverage = np.zeros(self.spatial_shape, dtype=bool)

    def _blend(self, size: tuple[int, ...]) -> npt.NDArray[np.float32]:
        found = self._weights.get(size)
        if found is None:
            found = blend_weights(size, mode=self.mode, sigma_scale=self.sigma_scale)
            self._weights[size] = found
        return found

    def add(self, prediction: npt.ArrayLike, patch: Patch) -> None:
        """Add one patch's prediction: shaped as the patch, padding included,
        after ``channels`` when there are channels."""
        if patch.grid_id is not None:
            if self.grid_id is None:
                self.grid_id = patch.grid_id
            elif patch.grid_id != self.grid_id:
                raise MEDH5ValidationError(
                    f"a patch on grid {patch.grid_id!r} cannot be stitched into "
                    f"a volume on grid {self.grid_id!r}: its slices index other "
                    "voxels"
                )
        if len(patch.slices) != len(self.spatial_shape):
            raise MEDH5ValidationError(
                f"a {len(patch.slices)}-D patch cannot be stitched into a "
                f"{len(self.spatial_shape)}-D volume"
            )
        array = np.asarray(prediction)
        lead = () if self.channels is None else (self.channels,)
        expected = (*lead, *patch.shape)
        if array.shape != expected:
            raise MEDH5ValidationError(
                f"a prediction for {patch!r} is shaped {expected}"
                + ("" if self.channels is None else " (channels first)")
                + f", not {array.shape}"
            )
        if any(
            s.start < 0 or s.stop > n
            for s, n in zip(patch.slices, self.spatial_shape, strict=True)
        ):
            raise MEDH5ValidationError(
                f"{patch!r} reaches outside the volume {self.spatial_shape}"
            )
        inner = tuple(
            slice(before, before + (s.stop - s.start))
            for s, (before, _) in zip(patch.slices, patch.padding, strict=True)
        )
        weights = self._blend(patch.shape)[inner]
        region = tuple(patch.slices)
        lead_axes = (slice(None),) * (array.ndim - len(region))
        cut = np.asarray(array[(*lead_axes, *inner)], dtype=np.float64) * weights
        self._sum[(*lead_axes, *region)] += cut.astype(self.dtype, copy=False)
        self._weight[region] += weights
        self.coverage[region] = True
        self.count += 1

    def result(self, *, uncovered: float = float("nan")) -> npt.NDArray[Any]:
        """The stitched volume; voxels no patch covered are *uncovered*."""
        out = np.full(self._sum.shape, uncovered, dtype=self.dtype)
        covered = np.broadcast_to(self.coverage, self._sum.shape)
        weight = np.broadcast_to(self._weight, self._sum.shape)
        np.divide(self._sum, weight, out=out, where=covered)
        return out


__all__ = [
    "BLEND_MODES",
    "PAIR_MODES",
    "STRATEGIES",
    "Seed",
    "PairReport",
    "Patch",
    "PatchAggregator",
    "PatchSampler",
    "TimepointPair",
    "TimepointPairSampler",
    "blend_weights",
    "coerce_patch_size",
    "grid_patches",
    "iter_patches",
    "window_around",
]
