"""Reproducing the performance targets on the reader's own hardware.

The numbers in the performance guide and in §14 of the specification were
measured on one machine.  Published without a way to re-run them they are
marketing; this module is the way to re-run them, so a claim like "18× faster
foreground sampling" can be checked rather than believed.

Every metric here is one the performance guide sets a target for.  A
measurement below target is reported as such and the exit status says so --- a
benchmark that always passes is a benchmark nobody reads.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medh5 import _core

TARGETS: dict[str, tuple[float, str]] = dict(_core.BENCH_TARGETS)
"""Metric -> (upper bound in ms, description): the performance guide's targets.

The many-class row holds the foreground draw to its O(1) claim where the class
count is large.  The eight-class sample met the target at 0.9 ms while a
63-class file cost 7.6 ms per draw, and nothing measured it.
"""

MANY_CLASSES: int = _core.BENCH_MANY_CLASSES


@dataclass(slots=True)
class Measurement:
    """One timed metric, and whether it met its target."""

    name: str
    value: float
    unit: str = "ms"
    target: float | None = None
    description: str = ""
    detail: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Measurement:
        return cls(
            name=str(doc["name"]),
            value=float(doc["value"]),
            unit=str(doc.get("unit", "ms")),
            target=doc.get("target"),
            description=str(doc.get("description", "")),
            detail=dict(doc.get("detail") or {}),
        )

    @property
    def ok(self) -> bool:
        return self.target is None or self.value <= self.target

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "target": self.target,
            "ok": self.ok,
            "description": self.description,
            "detail": self.detail or {},
        }

    def __str__(self) -> str:
        return str(_core.bench_line(self.to_json()))


def timed(fn: Callable[[], Any], *, repeats: int = 20, warmup: int = 3) -> float:
    """Median milliseconds per call.

    The median, not the mean: one page fault or one scheduler preemption in
    twenty runs moves a mean and not a median, and the question here is what a
    dataloader gets typically, not what it gets in the worst case.
    """
    return float(_core.bench_timed(fn, repeats=repeats, warmup=warmup))


def benchmark_file(
    path: str | os.PathLike[str],
    *,
    annotation: str | None = None,
    patch: int = 64,
    repeats: int = 20,
) -> list[Measurement]:
    """Every targeted metric on one file (and the paired-read metrics when it
    is longitudinal)."""
    return [
        Measurement.from_json(doc)
        for doc in _core.bench_benchmark_file(
            os.fspath(path), annotation=annotation, patch=patch, repeats=repeats
        )
    ]


def synthetic_pair(
    directory: str | os.PathLike[str],
    *,
    shape: tuple[int, int, int] = (64, 96, 96),
    codec: str = "training",
    seed: int = 20260815,
) -> Path:
    """A two-timepoint sample with a registration, for the paired metrics."""
    return Path(
        _core.bench_synthetic_pair(
            os.fspath(directory), shape=list(shape), codec=codec, seed=seed
        )
    )


def throughput(
    paths: Sequence[str | os.PathLike[str]],
    *,
    patch: int = 96,
    batches: int = 32,
    batch_size: int = 2,
    workers: int = 0,
    annotation: str | None = None,
) -> Measurement:
    """Sustained patches/s through the real dataloader (target: ≥ 400/s).

    Measured end to end --- open, sample, read, decompress, collate --- because
    that is the number that decides whether a GPU waits.
    """
    from torch.utils.data import DataLoader

    from medh5.sampling import PatchSampler
    from medh5.torch import PatchDataset, collate, worker_init_fn

    annotations: dict[str, list[str]] | None = {annotation: []} if annotation else None
    dataset = PatchDataset(
        [os.fspath(p) for p in paths],
        PatchSampler(patch, strategy="balanced"),
        samples_per_volume=max(1, (batches * batch_size) // max(1, len(paths))),
        annotations=annotations,
        label_format="onehot" if annotations else "none",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
    )
    seen = 0
    start = None
    elapsed = 0.0
    for batch in loader:
        if start is None:
            # The first batch pays for worker startup --- on spawn platforms
            # that is a fresh interpreter and a torch import per worker, which
            # is a one-off cost and not what "sustained throughput" means.
            start = time.perf_counter()
            continue
        seen += len(batch["meta"]["path"])
        elapsed = time.perf_counter() - start
        if seen >= batches * batch_size:
            break
    rate = seen / elapsed if elapsed > 0 else 0.0
    return Measurement(
        "patch_throughput",
        rate,
        unit="patches/s",
        description=f"{patch}³ patches, {workers} workers, steady state",
        detail={"patches": seen, "seconds": elapsed, "workers": workers},
    )


def synthetic_sample(
    directory: str | os.PathLike[str],
    *,
    shape: tuple[int, int, int] = (192, 256, 256),
    classes: int = 8,
    codec: str = "training",
    index: bool = True,
    seed: int = 20260815,
    name: str = "bench.medh5",
) -> Path:
    """Write a sample shaped like the one the published numbers were measured
    on: a CT, ``classes`` overlapping organs, a sampling index."""
    return Path(
        _core.bench_synthetic_sample(
            os.fspath(directory),
            shape=list(shape),
            classes=classes,
            codec=codec,
            index=index,
            seed=seed,
            name=name,
        )
    )


def synthetic_many_class_sample(
    directory: str | os.PathLike[str], *, classes: int = MANY_CLASSES
) -> Path:
    """A sample with *classes* small classes, for the many-class draw."""
    return Path(
        _core.bench_synthetic_many_class_sample(os.fspath(directory), classes=classes)
    )


def many_class_measurement(
    path: str | os.PathLike[str], *, patch: int = 64, repeats: int = 20
) -> Measurement:
    """The foreground draw on a many-class sample: O(1) whatever the classes."""
    return Measurement.from_json(
        _core.bench_many_class_measurement(
            os.fspath(path), patch=patch, repeats=repeats
        )
    )


def report(measurements: Sequence[Measurement]) -> str:
    """The text table the ``medh5 bench`` command prints."""
    return str(_core.bench_report([m.to_json() for m in measurements]))


__all__ = [
    "MANY_CLASSES",
    "TARGETS",
    "Measurement",
    "benchmark_file",
    "many_class_measurement",
    "report",
    "synthetic_many_class_sample",
    "synthetic_pair",
    "synthetic_sample",
    "throughput",
    "timed",
]
