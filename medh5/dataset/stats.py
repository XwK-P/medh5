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
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5Error


@dataclass(slots=True)
class Moments:
    """Count, mean and M2 for one image key --- a Welford accumulator."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: npt.NDArray[Any]) -> None:
        block = np.asarray(values, dtype=np.float64).ravel()
        if block.size == 0:
            return
        self.merge(
            Moments(
                count=int(block.size),
                mean=float(block.mean()),
                m2=float(((block - block.mean()) ** 2).sum()),
                minimum=float(block.min()),
                maximum=float(block.max()),
            )
        )

    def merge(self, other: Moments) -> None:
        """Chan-Golub-LeVeque parallel merge --- exact, not an approximation."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            self.minimum, self.maximum = other.minimum, other.maximum
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / total
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.count = total
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count > 1 else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": None if self.count == 0 else self.minimum,
            "max": None if self.count == 0 else self.maximum,
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Moments:
        count = int(doc["count"])
        std = float(doc.get("std", 0.0))
        return cls(
            count=count,
            mean=float(doc.get("mean", 0.0)),
            m2=std * std * count,
            minimum=float(doc["min"]) if doc.get("min") is not None else math.inf,
            maximum=float(doc["max"]) if doc.get("max") is not None else -math.inf,
        )


@dataclass(slots=True)
class ClassStats:
    """How often a class occurs, over the samples that examined it."""

    class_id: int
    voxels: int = 0
    present_in: int = 0
    examined_in: int = 0

    @property
    def prevalence(self) -> float:
        """Fraction of *examining* samples in which the class was found."""
        return self.present_in / self.examined_in if self.examined_in else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "voxels": self.voxels,
            "present_in": self.present_in,
            "examined_in": self.examined_in,
            "prevalence": self.prevalence,
        }


@dataclass(slots=True)
class DatasetStats:
    """Everything one streaming pass over a cohort learned."""

    samples: int = 0
    images: dict[str, Moments] = field(default_factory=dict)
    classes: dict[int, ClassStats] = field(default_factory=dict)
    total_voxels: int = 0
    failures: tuple[str, ...] = ()

    def merge(self, other: DatasetStats) -> None:
        self.samples += other.samples
        self.total_voxels += other.total_voxels
        self.failures = (*self.failures, *other.failures)
        for key, moments in other.images.items():
            self.images.setdefault(key, Moments()).merge(moments)
        for class_id, stats in other.classes.items():
            mine = self.classes.setdefault(class_id, ClassStats(class_id))
            mine.voxels += stats.voxels
            mine.present_in += stats.present_in
            mine.examined_in += stats.examined_in

    def normalization(self, image_key: str) -> tuple[float, float]:
        """``(mean, std)`` for a z-score transform, or ``(0, 1)`` if unseen."""
        moments = self.images.get(image_key)
        if moments is None or moments.count == 0:
            return 0.0, 1.0
        return moments.mean, moments.std or 1.0

    def class_weights(self, *, scheme: str = "inverse_frequency") -> dict[int, float]:
        """Loss weights from measured voxel frequencies.

        ``inverse_frequency`` weights by 1/count and normalises to a mean of 1,
        so switching schemes does not silently rescale the learning rate.
        """
        counts = {c: max(s.voxels, 1) for c, s in self.classes.items()}
        if not counts:
            return {}
        if scheme == "inverse_frequency":
            raw = {c: 1.0 / v for c, v in counts.items()}
        elif scheme == "inverse_sqrt":
            raw = {c: 1.0 / math.sqrt(v) for c, v in counts.items()}
        elif scheme == "uniform":
            raw = dict.fromkeys(counts, 1.0)
        else:
            raise MEDH5Error(
                f"unknown weighting scheme {scheme!r}; expected "
                "inverse_frequency, inverse_sqrt or uniform"
            )
        scale = len(raw) / sum(raw.values())
        return {c: v * scale for c, v in sorted(raw.items())}

    def to_json(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "total_voxels": self.total_voxels,
            "images": {k: v.to_json() for k, v in sorted(self.images.items())},
            "classes": [
                s.to_json()
                for s in sorted(self.classes.values(), key=lambda s: s.class_id)
            ],
            "failures": list(self.failures),
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> DatasetStats:
        return cls(
            samples=int(doc.get("samples", 0)),
            total_voxels=int(doc.get("total_voxels", 0)),
            images={k: Moments.from_json(v) for k, v in doc.get("images", {}).items()},
            classes={
                int(c["class_id"]): ClassStats(
                    class_id=int(c["class_id"]),
                    voxels=int(c.get("voxels", 0)),
                    present_in=int(c.get("present_in", 0)),
                    examined_in=int(c.get("examined_in", 0)),
                )
                for c in doc.get("classes", ())
            },
            failures=tuple(doc.get("failures", ())),
        )


def stats_for(
    path: str | os.PathLike[str],
    *,
    images: Sequence[str] | None = None,
    annotations: Sequence[str] | None = None,
    sample_stride: int = 1,
) -> DatasetStats:
    """One file's contribution, read plane by plane rather than whole.

    ``sample_stride`` subsamples along the first spatial axis.  A stride of 4
    over a 900-slice CT reads a quarter of the bytes for a mean that differs in
    the fourth decimal --- but it is opt-in, because "close enough" is a
    decision for the caller to take knowingly.
    """
    import medh5

    out = DatasetStats(samples=1)
    with medh5.open(path) as sample:
        for key, image in sample.images.items():
            if images is not None and key not in images:
                continue
            moments = out.images.setdefault(key, Moments())
            for block in _blocks(image, sample_stride):
                moments.update(block)
        fresh = _fresh_indices(sample)
        for key, annotation in sample.annotations.items():
            if annotations is not None and key not in annotations:
                continue
            examined = {int(c) for c in annotation.annotated_class_ids}
            present = _counts(sample, key, annotation, fresh)
            for class_id in examined | {int(c) for c in annotation.class_ids}:
                stats = out.classes.setdefault(class_id, ClassStats(class_id))
                if class_id in examined:
                    stats.examined_in += 1
                voxels = int(present.get(class_id, 0))
                stats.voxels += voxels
                if voxels:
                    stats.present_in += 1
        for grid in sample.grids.values():
            out.total_voxels += int(np.prod(grid.spatial_shape))
    return out


def _fresh_indices(sample: Any) -> frozenset[str]:
    """Annotation ids whose sampling index is present and current."""
    from medh5.integrity.verify import stale_index_entries

    stale = set(stale_index_entries(sample.root))
    return frozenset(name for name in sample.index if name not in stale)


def _counts(
    sample: Any, key: str, annotation: Any, fresh: frozenset[str]
) -> dict[int, int]:
    """Per-class voxel counts --- from the index when it can be trusted.

    The sampling index already stores exact counts (§14.3), so a cohort pass
    over indexed files reads a few hundred bytes per annotation instead of
    decompressing every mask.  A *stale* index is not used: counts for the
    annotation as it was before somebody edited it would be worse than slow.
    """
    if key in fresh:
        return {int(c): int(n) for c, n in sample.index[key].voxel_counts.items()}
    counter = getattr(annotation, "voxel_counts", None)
    return {} if counter is None else {int(c): int(n) for c, n in counter().items()}


def _blocks(image: Any, stride: int) -> Iterable[npt.NDArray[Any]]:
    """Slabs of an image along its first spatial axis."""
    shape = image.shape
    if not shape:
        return
    step = max(1, stride)
    for start in range(0, shape[0], step):
        yield np.asarray(image.dataset[start])


def compute_stats(
    paths: Sequence[str | os.PathLike[str]],
    *,
    images: Sequence[str] | None = None,
    annotations: Sequence[str] | None = None,
    workers: int = 1,
    sample_stride: int = 1,
) -> DatasetStats:
    """Stream a whole cohort, optionally across processes.

    Workers return accumulators, not arrays: the merge is over a few hundred
    bytes per file however large the volumes were.
    """
    total = DatasetStats()
    if workers <= 1:
        for path in paths:
            total.merge(_guarded(path, images, annotations, sample_stride))
        return total
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_guarded, os.fspath(p), images, annotations, sample_stride)
            for p in paths
        ]
        for future in futures:
            total.merge(future.result())
    return total


def _guarded(
    path: str | os.PathLike[str],
    images: Sequence[str] | None,
    annotations: Sequence[str] | None,
    sample_stride: int,
) -> DatasetStats:
    """One file, with failure recorded rather than raised.

    A cohort pass that aborts on file 3 000 has computed nothing usable; the
    failures come back in the result and the caller decides.
    """
    try:
        return stats_for(
            path, images=images, annotations=annotations, sample_stride=sample_stride
        )
    except (MEDH5Error, OSError) as exc:
        return DatasetStats(failures=(f"{os.fspath(path)}: {exc}",))


__all__ = [
    "ClassStats",
    "DatasetStats",
    "Moments",
    "compute_stats",
    "stats_for",
]
