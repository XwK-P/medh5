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

for epoch in range(epochs):
    ds.set_epoch(epoch)   # new patches this epoch --- call it every epoch
    ...
```

Each item's patch is drawn from `(seed, epoch, index)`, so a run is
reproducible and **a loop that never calls `set_epoch` draws the same patches
every epoch**. The epoch lives in shared memory, so `set_epoch` reaches every
`DataLoader` worker — `persistent_workers=True` included, under `fork` and
`spawn`. (Before 1.4.1 persistent workers kept the epoch they were started
with and repeated epoch 0's draw for the whole run.)

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
ds.report      # files, pairs, and cross-sectional files that contributed none
```

`report` does **not** resolve transforms. With `align="transform"`, a pair whose
frames have no registration is still counted as a pair, and the failure surfaces
from `__getitem__` as `MEDH5ValidationError` — part way into an epoch. See
[Longitudinal studies](../guides/longitudinal.md#train-on-the-pairs) for a
preflight that resolves the pairs itself.

Two grids **without** a `frame_uid` are never treated as registered, whatever
their coordinates: nothing says their world coordinates agree (§3.3). The NIfTI
and nnU-Net importers write no frame, so their longitudinal samples need a
transform — or `align="none"`, which reads the same index window from both
visits.


See [Longitudinal](../guides/longitudinal.md#train-on-the-pairs).

## A batch

```python
batch = next(iter(loader))

batch["images"]["CT"]        # (B, *patch) float32
batch["valid"]["CT"]         # (B, *patch) bool — where the image holds data
batch["label"]["organs"]     # (B, C, *patch) float32 — one-hot over the classes asked for
batch["ignore"]["organs"]    # (B, *patch) bool — voxels no loss may score
batch["meta"]["annotated"]["organs"]  # (B, C) bool — was each class examined?
batch["meta"]["subject_id"]  # list[str] — kept as a list, not stacked
batch["meta"]["patch"]["start"], ["stop"], ["pad"], ["center"]
batch["meta"]["patch"]["strategy"], ["class_id"], ["used_index"]
```

Three keys carry the file's contracts to the loss, because a loss can only
honour what it is handed:

- **`ignore[ann]`** is `True` on the §7.7 ignore region, under whichever
  encoding stores it — in band for `labelmap` and `layers`, a sibling mask for
  the rest — and on any padding added to reach the patch size.
  `label_format="labelmap"` also writes `65535` there, so
  `CrossEntropyLoss(ignore_index=65535)` works directly. The one-hot planes are
  `0` there: mask the loss with `ignore`.
- **`valid[image]`** is the image's `valid_mask` (§4.4) where it declares one,
  all `True` otherwise, and never the padding.
- **`meta["annotated"][ann]`** holds one flag per label channel, in the order you
  asked for the classes: whether that class was *examined* (§11.3). A `0` in a
  class nobody looked for is not a negative.

```python
loss = criterion(logits, target)                       # (B, C, *patch), unreduced
keep = ~batch["ignore"]["organs"] & batch["valid"]["CT"]
loss = loss * keep[:, None] * batch["meta"]["annotated"]["organs"][..., None, None, None]
```

`collate` stacks tensors and leaves everything else as lists. When two samples
disagree on a tensor's shape it names the key that disagreed rather than
raising from inside `torch.stack`.

`used_index` is worth logging. `False` means the sampler fell back to scanning
the volume because there was no sampling index — the difference between 0.9 ms
and several hundred. `None` means the draw was uniform and consulted no index, so
there is nothing to report.

## The DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=2, num_workers=8,
                    worker_init_fn=worker_init_fn,
                    collate_fn=collate,
                    persistent_workers=True)
```

`worker_init_fn` drops handles inherited across a `fork`. It is **recommended
but not required for correctness**: the handle cache is PID-keyed and re-checks
ownership on every access, so a forked worker abandons the parent's handles on
first use rather than reading through or closing them. The callback just does
that reset eagerly, at worker start, instead of lazily.

If you need your one `worker_init_fn` slot for seeding or other setup, call it
from your own:

```python
from medh5.torch import worker_init_fn as medh5_worker_init

def init(worker_id):
    medh5_worker_init(worker_id)
    seed_everything(worker_id)
```

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

`to_dict(sample, images, annotations)` builds a dictionary-transform item.
Annotations arrive as `int64` label volumes with their own grid's affine, and
every voxel of the §7.7 ignore region is `65535` whichever encoding stores it —
pass `ignore_index=65535` to the loss.

The affine construction (`affine_for`, `convert_affine`) does not import MONAI,
so the geometry is testable — and tested — in an environment without it.

## Related

- **[Tune performance](../guides/performance.md)** — the measured numbers, and
  the four levers behind them.
- **[Storage](storage.md)** — codec profiles, chunking, and the sampling index
  these datasets read.
- **[Longitudinal studies](../guides/longitudinal.md)** — what `PairedPatchDataset`
  is for.
