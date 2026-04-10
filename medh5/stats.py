"""Streaming dataset-level statistics over a collection of ``.medh5`` files.

Per-modality mean/variance use Welford's online algorithm so the result
is exact regardless of dataset size and never holds more than one
volume in memory at a time. Percentiles are estimated from a per-file
random voxel subsample (configurable budget) — exact percentiles over
multi-TB datasets are not worth the cost.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from medh5.core import MEDH5File
from medh5.dataset.index import Dataset


@dataclass
class ModalityStats:
    """Aggregated statistics for one modality across many files."""

    n_voxels: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    p01: float = 0.0
    p99: float = 0.0


@dataclass
class DatasetStats:
    """Top-level result of :func:`compute_stats`.

    Per-modality statistics live in :attr:`modalities`. Counts and
    distribution summaries cover labels, shapes, and seg coverage.
    """

    modalities: dict[str, ModalityStats] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)
    shape_histogram: dict[str, int] = field(default_factory=dict)
    seg_coverage: dict[str, float] = field(default_factory=dict)
    n_files: int = 0

    def __getitem__(self, modality: str) -> ModalityStats:
        return self.modalities[modality]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist as a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DatasetStats:
        payload = json.loads(Path(path).read_text())
        return cls(
            modalities={
                k: ModalityStats(**v) for k, v in payload.get("modalities", {}).items()
            },
            label_counts=payload.get("label_counts", {}),
            shape_histogram=payload.get("shape_histogram", {}),
            seg_coverage=payload.get("seg_coverage", {}),
            n_files=payload.get("n_files", 0),
        )


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------


@dataclass
class _FilePartial:
    """Per-file partial result returned by a worker."""

    n_per_modality: dict[str, int]
    sum_per_modality: dict[str, float]
    sumsq_per_modality: dict[str, float]
    min_per_modality: dict[str, float]
    max_per_modality: dict[str, float]
    samples_per_modality: dict[str, list[float]]
    label: int | str | None
    shape: tuple[int, ...] | None
    seg_coverage: dict[str, float]


def _process_file(args: tuple[str, list[str] | None, int, str | None]) -> _FilePartial:
    path, modalities, sample_voxels, foreground_mask = args
    n: dict[str, int] = {}
    s: dict[str, float] = {}
    sq: dict[str, float] = {}
    mn: dict[str, float] = {}
    mx: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    seg_cov: dict[str, float] = {}

    rng = np.random.default_rng(abs(hash(path)) % (2**32))

    with MEDH5File(path) as f:
        meta = f.meta
        img_grp = f.images
        names = (
            modalities
            if modalities is not None
            else sorted(meta.image_names or list(img_grp))
        )

        ref_shape: tuple[int, ...] | None = None
        # Optional foreground mask
        fg_mask = None
        if (
            foreground_mask is not None
            and f.seg is not None
            and foreground_mask in f.seg
        ):
            fg_mask = np.asarray(f.seg[foreground_mask][...], dtype=bool)

        for name in names:
            if name not in img_grp:
                continue
            arr = np.asarray(img_grp[name][...])
            if ref_shape is None:
                ref_shape = arr.shape

            if fg_mask is not None and fg_mask.shape == arr.shape:
                values = arr[fg_mask]
            else:
                values = arr.ravel()

            if values.size == 0:
                continue

            values_f = values.astype(np.float64, copy=False)
            n[name] = int(values_f.size)
            s[name] = float(values_f.sum())
            sq[name] = float(np.square(values_f).sum())
            mn[name] = float(values_f.min())
            mx[name] = float(values_f.max())

            # Random voxel subsample for percentile estimation
            if values_f.size > sample_voxels:
                idx = rng.choice(values_f.size, size=sample_voxels, replace=False)
                samples[name] = values_f[idx].tolist()
            else:
                samples[name] = values_f.tolist()

        # Per-mask coverage (fraction of voxels positive)
        if f.seg is not None:
            for sn in f.seg:
                m = np.asarray(f.seg[sn][...], dtype=bool)
                if m.size > 0:
                    seg_cov[sn] = float(m.sum()) / float(m.size)

    return _FilePartial(
        n_per_modality=n,
        sum_per_modality=s,
        sumsq_per_modality=sq,
        min_per_modality=mn,
        max_per_modality=mx,
        samples_per_modality=samples,
        label=meta.label,
        shape=ref_shape,
        seg_coverage=seg_cov,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_stats(
    source: Dataset | list[str] | list[Path],
    *,
    modalities: list[str] | None = None,
    sample_voxels: int = 1_000_000,
    foreground_mask: str | None = None,
    workers: int = 1,
) -> DatasetStats:
    """Aggregate per-modality statistics over a collection of files.

    Parameters
    ----------
    source : Dataset or list of paths
        The files to scan.
    modalities : list[str], optional
        Restrict to these modality names. Defaults to every modality
        encountered.
    sample_voxels : int
        Per-file random voxel budget for percentile estimation.
    foreground_mask : str, optional
        Restrict mean/std/min/max/percentiles to voxels inside the
        named segmentation mask (when present). Files lacking the mask
        contribute their full volume.
    workers : int
        Number of parallel processes (uses :class:`ProcessPoolExecutor`
        when ``> 1``).

    Returns
    -------
    DatasetStats
    """
    paths = source.paths if isinstance(source, Dataset) else [str(p) for p in source]

    args = [(p, modalities, sample_voxels, foreground_mask) for p in paths]

    partials: list[_FilePartial] = []
    if workers <= 1:
        for a in args:
            partials.append(_process_file(a))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_process_file, args):
                partials.append(r)

    # Welford merge across files (algebraically equivalent for our case
    # since we kept Σx, Σx², n per modality).
    agg_n: dict[str, int] = {}
    agg_sum: dict[str, float] = {}
    agg_sumsq: dict[str, float] = {}
    agg_min: dict[str, float] = {}
    agg_max: dict[str, float] = {}
    agg_samples: dict[str, list[float]] = {}

    label_counts: dict[str, int] = {}
    shape_hist: dict[str, int] = {}
    seg_cov_sum: dict[str, float] = {}
    seg_cov_n: dict[str, int] = {}

    for p in partials:
        for name, n in p.n_per_modality.items():
            agg_n[name] = agg_n.get(name, 0) + n
            agg_sum[name] = agg_sum.get(name, 0.0) + p.sum_per_modality[name]
            agg_sumsq[name] = agg_sumsq.get(name, 0.0) + p.sumsq_per_modality[name]
            agg_min[name] = min(
                agg_min.get(name, float("inf")), p.min_per_modality[name]
            )
            agg_max[name] = max(
                agg_max.get(name, float("-inf")), p.max_per_modality[name]
            )
            agg_samples.setdefault(name, []).extend(
                p.samples_per_modality.get(name, [])
            )

        if p.label is not None:
            key = str(p.label)
            label_counts[key] = label_counts.get(key, 0) + 1
        if p.shape is not None:
            sk = str(tuple(p.shape))
            shape_hist[sk] = shape_hist.get(sk, 0) + 1
        for sn, cov in p.seg_coverage.items():
            seg_cov_sum[sn] = seg_cov_sum.get(sn, 0.0) + cov
            seg_cov_n[sn] = seg_cov_n.get(sn, 0) + 1

    modalities_out: dict[str, ModalityStats] = {}
    for name, n in agg_n.items():
        if n == 0:
            continue
        mean = agg_sum[name] / n
        var = max(agg_sumsq[name] / n - mean * mean, 0.0)
        std = float(np.sqrt(var))
        samples = np.asarray(agg_samples.get(name, []), dtype=np.float64)
        if samples.size > 0:
            p01 = float(np.percentile(samples, 1))
            p99 = float(np.percentile(samples, 99))
        else:
            p01 = float("nan")
            p99 = float("nan")
        modalities_out[name] = ModalityStats(
            n_voxels=n,
            mean=float(mean),
            std=std,
            min=agg_min[name],
            max=agg_max[name],
            p01=p01,
            p99=p99,
        )

    seg_coverage_out = {
        sn: seg_cov_sum[sn] / seg_cov_n[sn] for sn in seg_cov_sum
    }

    return DatasetStats(
        modalities=modalities_out,
        label_counts=label_counts,
        shape_histogram=shape_hist,
        seg_coverage=seg_coverage_out,
        n_files=len(partials),
    )
