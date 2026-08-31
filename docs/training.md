# Training

PyTorch datasets, patch samplers, the MONAI adapter, and the performance
numbers behind them.

```bash
pip install "medh5[torch]"
```

## Datasets

All four take the same reading arguments: `images`, `annotations`,
`label_format`, `physical`, `dtype`, `timepoint`.

```python
from medh5.torch import (
    VolumeDataset, PatchDataset, GridPatchDataset, PairedPatchDataset,
    collate, worker_init_fn,
)
```

### VolumeDataset — whole volumes

```python
ds = VolumeDataset(paths,
                   images=["CT", "PET"],
                   annotations={"organs": ["liver", "lesion"]},
                   label_format="onehot",     # or labelmap, instances, none
                   physical=True,
                   timepoint="tp0")
```

### PatchDataset — random patches

```python
from medh5.sampling import PatchSampler

sampler = PatchSampler((96, 96, 96), strategy="balanced",
                       foreground_classes=["liver", "lesion"],
                       foreground_prob=0.5,
                       class_weights="inverse_frequency")

ds = PatchDataset(paths, sampler,
                  images=["CT"],
                  annotations={"organs": ["liver", "lesion"]},
                  samples_per_volume=8,
                  seed=0)

ds.set_epoch(epoch)   # re-seed between epochs
```

### GridPatchDataset — deterministic tiling, for inference

```python
ds = GridPatchDataset(paths, patch_size=(96, 96, 96), overlap=16,
                      images=["CT"])
```

Every patch position, in order, with a recorded pad — so you can stitch the
output back together.

### PairedPatchDataset — the same place at two visits

```python
from medh5.sampling import TimepointPairSampler

ds = PairedPatchDataset(paths, sampler,
                        pair_sampler=TimepointPairSampler("consecutive"),
                        align="transform",
                        annotation="lesions")
ds.report      # how many pairs were aligned by a transform, how many had none
```

See [Longitudinal](longitudinal.md#paired-sampling-for-training).

## A batch

```python
batch = next(iter(loader))

batch["images"]["CT"]        # (B, *patch) float32
batch["label"]["organs"]     # (B, C, *patch) float32 — one-hot over the classes asked for
batch["meta"]["subject_id"]  # list[str] — kept as a list, not stacked
batch["meta"]["patch"]["start"], ["stop"], ["pad"], ["center"]
batch["meta"]["patch"]["strategy"], ["class_id"], ["used_index"]
```

`collate` stacks tensors and leaves everything else as lists. When two samples
disagree on a tensor's shape it names the key that disagreed rather than
raising from inside `torch.stack`.

`used_index` is worth logging: `False` means the sampler fell back to scanning
the volume because there was no sampling index, which is the difference between
0.9 ms and several hundred.

## The DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=2, num_workers=8,
                    worker_init_fn=worker_init_fn,
                    collate_fn=collate,
                    persistent_workers=True)
```

`worker_init_fn` is required for `num_workers > 0`. HDF5 handles must not cross
a `fork`: the handle cache is keyed by PID, and a forked child **abandons** the
parent's handles rather than closing them — closing a descriptor the parent
still owns is how a dataloader corrupts a file it is only reading.

A 10-epoch soak over the cache leaves the handle count and the descriptor count
flat; there is a test that asserts it.

## Sampling strategies

| Strategy | |
|---|---|
| `uniform` | any position, uniformly |
| `foreground` | centred on a foreground voxel of a chosen class |
| `balanced` | `foreground_prob` of the time foreground, otherwise uniform |

`class_weights` picks which class a foreground draw targets: `uniform`,
`inverse_frequency`, `frequency`, or an explicit `{class_id: weight}` mapping.

Foreground sampling is O(1) in the volume **if the file carries a sampling
index**:

```
$ medh5 index build cohort/*.medh5
```

Without one the sampler scans, still works, and records `used_index=False`.

## Normalisation from the training split only

```python
from medh5.dataset import Manifest, compute_stats

manifest = Manifest.load("cohort.json")
train = [e.path for e in manifest
         if any(c["set_id"] == "cv5" and c["partition"] == "train" for c in e.splits)]
stats = compute_stats(train, images=["CT"], workers=8)
mean, std = stats.normalization("CT")
weights = stats.class_weights(scheme="inverse_frequency")
```

Or from the shell:

```
$ medh5 dataset stats cohort.json --partition train --set-id cv5 -o stats.json
```

See [Cohorts](cohorts.md).

## MONAI

```bash
pip install "medh5[monai]"
```

```python
from medh5.monai import to_metatensor, from_metatensor, meta_dict, affine_for

with medh5.open(path) as s:
    tensor = to_metatensor(s, "CT")       # MetaTensor with the correct affine
```

`Spacingd`, `Orientationd`, `SaveImaged` and the rest work unmodified, because
the affine is right. `to_metatensor(..., roi=...)` shifts the origin to the
ROI, so a patch keeps its world position.

The affine construction (`affine_for`, `convert_affine`) does not import MONAI,
so the geometry is testable — and tested — in an environment without it.

## Performance

Measured on a 192×256×256 synthetic CT with eight classes.

| Metric | Target | 0.x | Measured |
|---|---|---|---|
| 64³ patch, multi-class labels only | ≤ 10 ms | 117 ms | **4.0 ms** |
| Foreground centre sampling *(indexed)* | ≤ 1 ms, O(1) memory | 9.2 ms, O(volume) | **0.90 ms** |
| Metadata-only read | ≤ 2 ms | ~1.5 ms | **0.21 ms** |
| Full `open()` → first patch | ≤ 15 ms | ~120 ms | **2.4 ms** |
| Sustained 96³ throughput | ≥ 400 patches/s | ~60 | **600–850** (4 workers) |

```
$ medh5 bench                       # builds a synthetic sample and measures
$ medh5 bench case.medh5 --patch 64 --repeats 20 --workers 4 --json
```

Two things to know before quoting these.

**The sampling row needs an index.** `bench` calls `build_index()` on the
sample it builds, so 0.90 ms is the indexed path — the one you get after
`medh5 index build`, not the one you get by default. Unindexed, the same draw
scans the labels: 30 ms on this volume, 312 ms at 512³, growing with the volume
while the indexed draw stays flat. `used_index` in the batch metadata says which
you measured.

**`bench` does not check the throughput target.** The first four rows carry a
target it verifies and reports against; throughput depends on worker count, so
it is measured and printed without one. `medh5 bench` with no `--workers` runs
single-process and reports around 330 patches/s — below the 400 in the table,
and still followed by *all targets met*, which is a statement about the four
checked rows. Pass `--workers 4` to reproduce the number above.

Two decisions are behind the label-read number: each stacked plane is chunked
separately, so one layer reads without the others; and a multi-class `dense()`
reads **by plane rather than by class**, so a 200-class annotation packed into
four layers is four reads and not two hundred.

## Codec profiles

Storage is a training parameter. `training` decompresses fastest; `archive` is
smallest.

```
$ medh5 recompress cohort/*.medh5 --profile training
```

Every stored byte changes and no `content_id` does — the digest is over
content, not over its encoding — so a recompressed cohort is still verifiably
the same data. See [Storage](file-format.md).
