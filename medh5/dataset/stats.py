"""Streaming cohort statistics: intensity moments and class frequencies.

Normalisation constants and class weights are cohort properties, and computing
them by loading the cohort is how a preprocessing step comes to need more RAM
than the training it precedes.  Everything here is streaming: one pass per
file, constant memory per worker, and a merge step that is exact rather than
approximate.

Two things this deliberately does *not* do.  It does not average per-file means
--- that weights a 40-slice scan the same as a 900-slice one; the Welford merge
weights by voxel count.  And it does not treat an unexamined class as a zero:
``§11.3`` distinguishes "looked for and absent" from "never looked for", so
frequencies are reported over the samples that actually examined each class.

Class statistics are *voxel* statistics, so they come from voxel annotations
only.  A sample-level classification or a set of boxes names classes too, and
counting those as "examined, 0 voxels" made a one-bit diagnosis the heaviest
segmentation class in the cohort.  And the moments are over the voxels an image
holds data in: a ``valid_mask`` (§4.4) keeps the scanner's padding outside the
reconstruction circle out of the mean.

Intensity moments are over **physical** values by default --- ``stored × slope +
intercept`` (§4.2) --- because that is what the loaders hand a model with
``physical=True``, and a z-score computed over the numbers the file *stores*
normalises the wrong distribution.  A CT stored ``int16`` with slope 2 and
intercept −1024 has a stored mean near 100 and a physical mean near −824; the
statistics used to report the former while the tensors carried the latter.
``physical=False`` measures the stored values for the caller who wants them.

The moments are over **finite** values.  A NaN or an infinity --- PET outside
its field of view, resampled data --- is counted in ``Moments.nonfinite`` and
left out: folded in, it made every moment NaN, let the minimum and maximum
depend on the order of the files, and wrote ``null``, which a reload read as a
mean of 0 and a standard deviation of 0 (normalising with ``(0, 1)``).  A
statistics file whose moments are ``null`` is now refused on reload.

A class id means its key in its label set, so ``ClassStats.key`` records it,
and statistics in which one id carries two keys --- 1 is ``liver`` in some
samples and ``spleen`` in others --- are refused rather than counted as one
class under one loss weight (``dataset check`` reports the clash as C102).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core, _deprecation


@dataclass(slots=True)
class Moments:
    """Count, mean and M2 for one image key --- a Welford accumulator over the
    finite values folded in."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    nonfinite: int = 0
    """Values left out of the moments because they are NaN or infinite."""

    def _set(self, raw: tuple[int, float, float, float, float, int]) -> None:
        (self.count, self.mean, self.m2, self.minimum, self.maximum, self.nonfinite) = (
            raw
        )

    def update(self, values: npt.NDArray[Any]) -> None:
        """Fold in one block of values (any shape); a NaN or an infinity is
        counted in ``nonfinite``, not folded in."""
        self._set(
            _core.dataset_moments_update(
                self, np.asarray(values, dtype=np.float64).ravel()
            )
        )

    def merge(self, other: Moments) -> None:
        """Chan-Golub-LeVeque parallel merge --- exact, not an approximation."""
        self._set(_core.dataset_moments_merge(self, other))

    @property
    def std(self) -> float:
        return float(_core.dataset_moments_std(self))

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_moments_json(self)
        return found

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Moments:
        """The moments *doc* describes; a ``null`` moment over any value is
        refused (``ValueError``), never read as 0."""
        out = cls()
        out._set(_core.dataset_moments_from_json(doc))
        return out


@dataclass(slots=True)
class ClassStats:
    """How often a class occurs, over the samples that examined it.

    ``present_in`` and ``examined_in`` count *samples*: a longitudinal sample
    that examined a class at two visits examined it once, as a cohort member.
    ``present_in`` counts the samples that examined the class *and* found it,
    so ``prevalence`` is at most 1: a class drawn where nobody looked for it
    says nothing about how often it occurs (§11.3).  ``voxels`` counts every
    voxel of it, examined or not.  ``key`` is what the id means in the label
    set of the samples measured, ``None`` where none declares it.
    """

    class_id: int
    voxels: int = 0
    present_in: int = 0
    examined_in: int = 0
    key: str | None = None

    @property
    def examined_prevalence(self) -> float | None:
        """Fraction of *examining* samples in which the class was found;
        ``None`` when no sample examined it, which says nothing about how
        often it occurs (§11.3)."""
        return self.present_in / self.examined_in if self.examined_in else None

    @property
    def prevalence(self) -> float:
        """:attr:`examined_prevalence`, with 0.0 for a class no sample examined.

        That 0.0 reads as "looked for everywhere, found nowhere" for a class
        nobody looked for, so it raises a ``FutureWarning`` there: medh5 3.0
        returns ``None``.
        """
        found = self.examined_prevalence
        if found is None:
            _deprecation.warn(
                "medh5 3.0 returns None for the prevalence of a class no sample "
                "examined: use examined_prevalence",
                category=FutureWarning,
                stacklevel=3,
            )
            return 0.0
        return found

    def to_json(self) -> dict[str, Any]:
        """The engine's JSON: ``prevalence`` is ``null`` where no sample examined
        the class."""
        found: dict[str, Any] = _core.dataset_class_stats_json(self)
        return found


@dataclass(slots=True)
class DatasetStats:
    """Everything one streaming pass over a cohort learned."""

    samples: int = 0
    images: dict[str, Moments] = field(default_factory=dict)
    classes: dict[int, ClassStats] = field(default_factory=dict)
    total_voxels: int = 0
    failures: tuple[str, ...] = ()
    physical: bool = True
    """Whether the image moments are over physical values (rescale applied)."""

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> DatasetStats:
        out = cls()
        out._set(raw)
        return out

    def _set(self, raw: dict[str, Any]) -> None:
        self.samples = int(raw["samples"])
        self.physical = bool(raw["physical"])
        self.total_voxels = int(raw["total_voxels"])
        self.failures = tuple(raw["failures"])
        images: dict[str, Moments] = {}
        for key, state in raw["images"].items():
            moments = self.images.get(key) or Moments()
            moments._set(state)
            images[key] = moments
        self.images = images
        classes: dict[int, ClassStats] = {}
        for class_id, (voxels, present_in, examined_in, key) in raw["classes"].items():
            stats = self.classes.get(class_id) or ClassStats(class_id)
            stats.voxels, stats.present_in, stats.examined_in, stats.key = (
                voxels,
                present_in,
                examined_in,
                key,
            )
            classes[class_id] = stats
        self.classes = classes

    def merge(self, other: DatasetStats) -> None:
        """Fold another pass in.  Statistics over physical values do not merge
        with statistics over stored ones, and a class id the two give
        different keys is refused (``MEDH5ValidationError``)."""
        self._set(_core.dataset_stats_merge(self, other))

    def normalization(self, image_key: str) -> tuple[float, float]:
        """``(mean, std)`` for a z-score transform, or ``(0, 1)`` if unseen."""
        mean, std = _core.dataset_stats_normalization(self, image_key)
        return float(mean), float(std)

    def class_weights(self, *, scheme: str = "inverse_frequency") -> dict[int, float]:
        """Loss weights from measured voxel frequencies.

        ``inverse_frequency`` weights by 1/count and normalises to a mean of 1,
        so switching schemes does not silently rescale the learning rate.

        A class with no voxels in the cohort gets **no** weight, and a warning
        names it: the inverse of zero is not a weight, and flooring it made an
        unseen class the whole loss.
        """
        found: dict[int, float] = _core.dataset_stats_class_weights(self, scheme=scheme)
        return found

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_stats_json(self)
        return found

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> DatasetStats:
        return cls._from_raw(_core.dataset_stats_from_json(doc))


def stats_for(
    path: str | os.PathLike[str],
    *,
    images: Sequence[str] | None = None,
    annotations: Sequence[str] | None = None,
    sample_stride: int = 1,
    physical: bool = True,
) -> DatasetStats:
    """One file's contribution, read plane by plane rather than whole.

    ``sample_stride`` subsamples along the first spatial axis.  A stride of 4
    over a 900-slice CT reads a quarter of the bytes for a mean that differs in
    the fourth decimal --- but it is opt-in, because "close enough" is a
    decision for the caller to take knowingly.

    ``physical`` applies each image's rescale (§4.2) before measuring, which is
    the default because it is what the loaders do; ``False`` measures the
    stored values.
    """
    return DatasetStats._from_raw(
        _core.dataset_stats_for(
            os.fspath(path),
            images=None if images is None else list(images),
            annotations=None if annotations is None else list(annotations),
            sample_stride=sample_stride,
            physical=physical,
        )
    )


def compute_stats(
    paths: Sequence[str | os.PathLike[str]],
    *,
    images: Sequence[str] | None = None,
    annotations: Sequence[str] | None = None,
    workers: int = 1,
    sample_stride: int = 1,
    physical: bool = True,
) -> DatasetStats:
    """Stream a whole cohort, optionally across *workers* threads.

    Workers return accumulators, not arrays, and the results are merged in
    path order, so the statistics do not depend on the worker count.  A file
    that cannot be read is listed in ``failures``, not fatal; a file giving a
    class id another key than the files before it is refused, naming the file.
    """
    return DatasetStats._from_raw(
        _core.dataset_compute_stats(
            [os.fspath(p) for p in paths],
            images=None if images is None else list(images),
            annotations=None if annotations is None else list(annotations),
            workers=workers,
            sample_stride=sample_stride,
            physical=physical,
        )
    )


__all__ = [
    "ClassStats",
    "DatasetStats",
    "Moments",
    "compute_stats",
    "stats_for",
]
