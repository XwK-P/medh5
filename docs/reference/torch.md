# PyTorch and MONAI

PyTorch datasets, patch samplers, collation and the handle cache, and the MONAI
adapter. What each performance lever is worth is in
[Tune performance](../guides/performance.md).

```bash
pip install "medh5[torch]"
```

## Datasets

All four take the same reading arguments: `images`, `annotations`,
`label_format` (`onehot`, `labelmap`, `instances` or `none`), `physical`,
`dtype`, `timepoint`.

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
output back together. The tiling itself is `medh5.sampling.grid_patches(shape,
patch_size, overlap=...)`, which needs no torch, for an inference loop of your
own.

### PairedPatchDataset — the same place at two visits

```python
from medh5.sampling import TimepointPairSampler

ds = PairedPatchDataset(paths, sampler,
                        pair_sampler=TimepointPairSampler("consecutive"),
                        align="transform",
                        annotation="lesions",
                        samples_per_pair=4)
ds.report      # files, pairs, and cross-sectional files that contributed none
```

`TimepointPairSampler` modes are `consecutive`, `baseline_vs_all` and
`all_pairs`. Each item carries both visits' patches, `meta["pair"]` and
`meta["interval_days"]`, and — where a change label spans exactly that pair —
`item["label"][ann_id]`, that classification's `{class_key: value}`. `label=`
names the change-label annotation explicitly instead.

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
the volume because there was no current sampling index — the difference between
0.03 ms and 650 ms per draw at 512³. `None` means the draw was uniform and
consulted no index, so there is nothing to report.

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

### Keeping a file's items together

Each worker keeps up to 32 files open (`set_cache_size(n)` changes it; call it
in your `worker_init_fn` to size each worker's cache). `shuffle=True` scatters a
file's `samples_per_volume` items across the epoch, so with more files than the
cache holds most items open their file again. `FileGroupedSampler` shuffles
**files** instead, and yields each file's items together:

```python
from medh5.torch import FileGroupedSampler, set_cache_size

loader = DataLoader(ds, batch_size=4, sampler=FileGroupedSampler(ds, seed=0),
                    num_workers=8, worker_init_fn=worker_init_fn,
                    collate_fn=collate, persistent_workers=True)
```

The order is a function of `(seed, epoch)`, and the epoch is the dataset's:
`ds.set_epoch(epoch)` redraws the patches and reorders the files. Pass it
instead of `shuffle=True`, not with it. It works with every dataset here, and
with workers `samples_per_volume` should be a multiple of `batch_size`, since a
file whose items straddle two batches is opened by both workers.

The cache is shared by the threads of a process and locked. A handle is held
for the length of an item, so a thread-based loader never has one closed under
it by another thread's eviction.

A 10-epoch soak over the cache leaves the handle count and the descriptor count
flat; there is a test that asserts it.

### On a network filesystem

HDF5's file locking is unreliable on NFS, Lustre and GPFS. For a training job
that only reads, turn it off in the job's environment, before Python starts:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

medh5 does not set it for you (§14.4): it also removes the protection between
two writers. See [Tune performance](../guides/performance.md#on-a-network-filesystem).

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

## Clinical tasks (format 1.1)

`ClinicalTaskDataset` turns the rows of a [task manifest](../spec/task-cache-1.md)
--- a subject at a cutoff --- into items; `collate_clinical` batches them.
Every decision about what a row may read is the engine's, made once by the
task's preflight; the dataset reads only that.

```python
from torch.utils.data import DataLoader
from medh5.torch import ClinicalTaskDataset, collate_clinical

train = ClinicalTaskDataset("cohort/progression.task.json", partition="train")
batch = next(iter(DataLoader(train, batch_size=8, collate_fn=collate_clinical)))
```

| Argument | |
|---|---|
| `task` | A `TaskManifest` or the path of one |
| `partition` | One partition of the task's split |
| `statuses` | Preflight statuses to keep (`("eligible",)`) |
| `concepts` | A `ConceptVocabulary` --- fitted on the training partition when omitted, refused when fitted on anything else (T405) |
| `documents` | An event-level feature cache's path, or an encoder with `encode(text)` and `dim` |
| `row_features` | A patient-level cache, validated against the task before any row reads it |
| `strict` | Refuse a task whose preflight has findings (default), or keep only the unaffected rows |

A batch, for `B` rows:

| Key | Shape | Meaning |
|---|---|---|
| `images[slot]` | `(B, C, *patch)` | The slot's window on its image's own grid; zeros where the slot is empty |
| `present[slot]` | `(B,)` bool | Modality availability: an eligible image filled the slot |
| `valid[slot]` | `(B, *patch)` bool | Field of view: inside the image and its valid region, never the padding |
| `image_time[slot][...]` | `(B, 2)`, `(B,)` | The image's acquisition (`start_age_h`) and availability (`available_age_h`), as ages before the cutoff, with `start_known` and `available_known` |
| `label[slot]`, `annotated[slot]` | `(B, K, *patch)`, `(B, K)` | Supervision for the slot's `classes`, and whether each was examined (coverage) |
| `ignore[slot]` | `(B, *patch)` bool | Voxels a loss must not score: ignore regions and padding |
| `events[...]` | `(B, N)`, `(B, N, 2)` | The admitted versions in input order, padded to the longest history; `mask` marks real events, `length` counts them (below) |
| `documents[...]` | `(B, M, D)`, `(B, M)`, `(B, M, 2)` | `features` of the admitted documents, `event` (the owning version's position in `events`), `start_age_h` and `available_age_h` with their `*_known`; `mask` and `length` |
| `target["value"]`, `target["observed"]` | `(B,)` | The label, and whether it is observed (a censored row is not) |
| `meta` | list | Row id, subject, partition, cutoff, fingerprint, the admitted event ids, and per slot the visit that filled it |

The event sequence keeps what 1.1 §5 distinguishes, and never collapses a time
to one number:

| `events[...]` | Shape | Meaning |
|---|---|---|
| `concept`, `kind`, `status`, `temporal_type` | `(B, N)` int | Vocabulary index (0 padding, 1 unseen); the kind, status and temporal type as their vocabulary's index + 1 (0 padding) |
| `value`, `has_value` | `(B, N)` | The value, normalised by the fitted statistics; whether there is one --- an absent value is not 0 |
| `comparator` | `(B, N)` int | `eq`, `lt`, ... as `COMPARATORS` index + 1 (`eq` when a value has none); 0 without a value |
| `missing` | `(B, N)` bool | An expected result missing for a source reason: absence, never a negative |
| `start_age_h`, `end_age_h`, `available_age_h` | `(B, N, 2)` | Effective start, effective end and availability as ages before the cutoff in hours: `[..., 0]` the least, `[..., 1]` the most, equal for an exact instant |
| `start_known`, `end_known`, `available_known` | `(B, N)` bool | Whether that time is known; an unknown one is zeros, which are not a time |
| `tie_group` | `(B, N)` int | Versions whose ordering times overlap share a group: their order is the storage tie-break, not evidence |
| `plan` | `(B, N)` bool | Admitted as a plan (`plans` policy): its start is ahead of the cutoff |

An age is negative only for what a version available at the cutoff itself recorded about later --- a
plan's start, a course's recorded end; every `available_age_h` is at least 0. A `static` event has no
effective time (`temporal_type`), which is not the same as an `unknown` one (strict selection never
orders an unknown time; `order_by = "available"` admits it, as unknown).

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

The affine is passed through in the grid's own `coord_system` — usually LPS —
and labelled with it in the metadata MONAI reads, rather than silently converted.
Pass `space="RAS"` when a consumer needs RAS: the conversion is a sign flip on
the first two world axes of the affine, never a flip of the voxels.
`from_metatensor(tensor)` returns the array and its metadata.

`to_dict(sample, images, annotations)` builds a dictionary-transform item, with
the same `space=` and `physical=` options.
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
- **[Train on clinical tasks](../guides/clinical-training.md)** — what
  `ClinicalTaskDataset` is for, and the masks it keeps apart.
