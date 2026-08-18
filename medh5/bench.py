"""Reproducing the performance targets on the reader's own hardware (plan §4.3).

The numbers in the plan and in §14 of the specification were measured on one
machine.  Published without a way to re-run them they are marketing; this
module is the way to re-run them, so a claim like "18× faster foreground
sampling" can be checked rather than believed.

Every metric here is one the plan set a target for.  A measurement below target
is reported as such and the exit status says so --- a benchmark that always
passes is a benchmark nobody reads.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TARGETS: dict[str, tuple[float, str]] = {
    "patch_labels_ms": (10.0, "64³ patch, multi-class labels only"),
    "foreground_sample_ms": (1.0, "foreground centre sampling"),
    "meta_read_ms": (2.0, "metadata-only read"),
    "open_to_first_patch_ms": (15.0, "full open() → first patch"),
}
"""Metric -> (upper bound in ms, description).  Plan §4.3."""


@dataclass(slots=True)
class Measurement:
    """One timed metric, and whether it met its target."""

    name: str
    value: float
    unit: str = "ms"
    target: float | None = None
    description: str = ""
    detail: dict[str, Any] | None = None

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
        goal = (
            "" if self.target is None else f"  (target ≤ {self.target:g} {self.unit})"
        )
        mark = " " if self.ok else "!"
        return f"{mark} {self.name:26s} {self.value:8.3f} {self.unit}{goal}"


def timed(fn: Callable[[], Any], *, repeats: int = 20, warmup: int = 3) -> float:
    """Median milliseconds per call.

    The median, not the mean: one page fault or one scheduler preemption in
    twenty runs moves a mean and not a median, and the question here is what a
    dataloader gets typically, not what it gets in the worst case.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def benchmark_file(
    path: str | os.PathLike[str],
    *,
    annotation: str | None = None,
    patch: int = 64,
    repeats: int = 20,
) -> list[Measurement]:
    """Run the §4.3 metrics against one existing sample."""
    import medh5
    from medh5.document import read_document
    from medh5.sampling import PatchSampler

    text = os.fspath(path)
    out: list[Measurement] = []

    with medh5.open(text) as sample:
        ann_id = annotation or next(
            (
                name
                for name, ann in sample.annotations.items()
                if ann.kind not in ("classification",)
            ),
            None,
        )
        image_id = sorted(sample.images)[0]
        shape = sample.images[image_id].grid.spatial_shape
        window = tuple(
            slice(
                max(0, n // 2 - patch // 2), max(0, n // 2 - patch // 2) + min(patch, n)
            )
            for n in shape
        )

        if ann_id is not None:
            ann = sample.annotations[ann_id]
            classes = list(ann.class_ids)
            out.append(
                Measurement(
                    "patch_labels_ms",
                    timed(
                        lambda: ann.dense(classes, roi=list(window)), repeats=repeats
                    ),
                    target=TARGETS["patch_labels_ms"][0],
                    description=TARGETS["patch_labels_ms"][1],
                    detail={"classes": len(classes), "kind": ann.kind},
                )
            )
            sampler = PatchSampler(patch, strategy="foreground")
            rng = np.random.default_rng(0)
            indexed = ann_id in sample.index
            out.append(
                Measurement(
                    "foreground_sample_ms",
                    timed(lambda: sampler.draw(sample, ann_id, rng), repeats=repeats),
                    target=TARGETS["foreground_sample_ms"][0],
                    description=TARGETS["foreground_sample_ms"][1],
                    detail={"used_index": indexed},
                )
            )
        out.append(
            Measurement(
                "image_patch_ms",
                timed(lambda: sample.images[image_id].read(window), repeats=repeats),
                description=f"{patch}³ image patch",
                detail={"image": image_id, "shape": list(shape)},
            )
        )

    def read_meta() -> Any:
        with medh5.open(text) as handle:
            return read_document(handle.root)

    out.append(
        Measurement(
            "meta_read_ms",
            timed(read_meta, repeats=repeats),
            target=TARGETS["meta_read_ms"][0],
            description=TARGETS["meta_read_ms"][1],
        )
    )

    def open_and_patch() -> Any:
        with medh5.open(text) as handle:
            return handle.images[image_id].read(window)

    out.append(
        Measurement(
            "open_to_first_patch_ms",
            timed(open_and_patch, repeats=repeats),
            target=TARGETS["open_to_first_patch_ms"][0],
            description=TARGETS["open_to_first_patch_ms"][1],
        )
    )
    return out


def throughput(
    paths: Sequence[str | os.PathLike[str]],
    *,
    patch: int = 96,
    batches: int = 32,
    batch_size: int = 2,
    workers: int = 0,
    annotation: str | None = None,
) -> Measurement:
    """Sustained patches/s through the real dataloader (plan §4.3: ≥ 400/s, 8 workers).

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
    shape: tuple[int, ...] = (192, 256, 256),
    classes: int = 8,
    codec: str = "training",
    index: bool = True,
    seed: int = 20260815,
) -> Path:
    """Write a sample shaped like the one the published numbers were measured on."""
    import medh5
    from medh5.labels.labelset import LabelClass, LabelSet

    rng = np.random.default_rng(seed)
    root = Path(os.fspath(directory))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bench.medh5"
    label_set = LabelSet(
        "bench",
        version="1.0.0",
        classes=[
            LabelClass(i + 1, f"c{i + 1}", f"Class {i + 1}") for i in range(classes)
        ],
    )
    masks = {}
    for i in range(classes):
        mask = np.zeros(shape, dtype=bool)
        corner = [int(rng.integers(0, max(1, n - n // 4))) for n in shape]
        window = tuple(slice(c, c + n // 4) for c, n in zip(corner, shape, strict=True))
        mask[window] = True
        masks[i + 1] = mask
    with medh5.create(path, sample_id="bench", codec=codec) as writer:
        writer.label_set(label_set)
        writer.add_grid(
            "g", shape=shape, spacing=(1.0, 1.0, 1.0), patch_hint=(64, 64, 64)
        )
        writer.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="g",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        writer.add_segmentation("organs", grid="g", masks=masks)
        if index:
            writer.build_index()
        writer.deidentification(method="synthetic")
    return path


def report(measurements: Sequence[Measurement]) -> str:
    lines = [str(m) for m in measurements]
    failed = [m for m in measurements if not m.ok]
    lines.append("")
    lines.append(
        "all targets met"
        if not failed
        else f"{len(failed)} metric(s) below target: "
        + ", ".join(m.name for m in failed)
    )
    return "\n".join(lines)


__all__ = [
    "TARGETS",
    "Measurement",
    "benchmark_file",
    "report",
    "synthetic_sample",
    "throughput",
    "timed",
]
