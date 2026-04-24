# Datasets and statistics

## Manifest

`Dataset` is a metadata-only manifest of a directory of `.medh5` files. It
scans with `read_meta()` (no array reads), so building a manifest over
thousands of files is fast and cheap.

```python
from medh5.dataset import Dataset

ds = Dataset.from_directory("data/", recursive=True)

len(ds)
ds.records                        # list[DatasetRecord]
ds.records[0].path
ds.records[0].label
ds.records[0].shape
ds.records[0].spacing
ds.records[0].coord_system
ds.records[0].patch_size
ds.records[0].review_status
```

Filter and persist:

```python
labeled = ds.filter(lambda r: r.label is not None)
labeled.save("manifest.json")
ds2 = Dataset.load("manifest.json")
```

Loaded manifests track per-file mtime and size. `ds.is_stale()` returns
`True` if any file has changed on disk since the manifest was saved.

## Splitting

`make_splits` produces reproducible train/val/test splits (or k-fold) with
optional stratification and grouping.

```python
from medh5.dataset import make_splits

splits = make_splits(
    ds,
    ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    stratify_by="label",
    group_by="extra.patient_id",     # dotted path into extra
    seed=42,
)
splits["train"]                       # Dataset
splits["train"].save("train.json")
```

K-fold:

```python
folds = make_splits(ds, k_folds=5, seed=42)
# folds is list[dict[str, Dataset]] — one entry per fold with "train" and "val"
```

Grouping ensures all records with the same group key end up in the same
split (e.g. all scans from one patient stay together). Stratification
balances the distribution of the stratify key across splits, respecting the
group constraint where both are given.

## Statistics

`compute_stats` streams per-modality mean, std, min, max, and percentiles
using Welford accumulation per file and Chan's parallel-variance merge
across files. Large integer volumes no longer suffer catastrophic
cancellation on variance.

```python
from medh5.stats import compute_stats

stats = compute_stats(ds, workers=4)

stats["CT"].mean
stats["CT"].std
stats["CT"].min
stats["CT"].max
stats["CT"].p01
stats["CT"].p99

stats.label_counts
stats.shape_histogram
stats.seg_coverage      # fraction of foreground voxels per named mask
```

Foreground-restricted stats:

```python
stats = compute_stats(ds, foreground_seg="tumor", workers=4)
```

### Determinism

Per-file percentile sampling uses a stable BLAKE2b digest of the file path
as its seed, so results are reproducible across runs and Python invocations
regardless of `PYTHONHASHSEED`.
