# PyTorch and MONAI

PyTorch datasets, patch samplers, collation and the handle cache, and the MONAI
adapter. What each performance lever is worth is in
[Tune performance](../guides/performance.md).

```bash
pip install "medh5[torch]"
```

## Datasets

`VolumeDataset`, `PatchDataset` and `GridPatchDataset` take the same reading
arguments: `images`, `annotations`, `label_format` (`onehot`, `labelmap`,
`instances` or `none`), `physical`, `dtype`, `timepoint`. A patch is drawn, and
a cover planned, on the grid the images they read share, so `timepoint="tp1"`
reads the follow-up's grid; `images=` given beside it must be at that visit. `PairedPatchDataset` takes `images`, `physical` and
`dtype`, and refuses the other three: a pair chooses its own visits.

`annotations` names voxel annotations (segmentations) only: every label format,
`instances` included, is read from voxels, and a `boxes`, `points` or other
geometric annotation is refused by name (`MEDH5ValidationError`), as it is by
MONAI's `to_dict`.

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

Every patch position, in order, with a recorded pad, and each item's
`meta["cover"]` — `{"file": i, "index": k, "of": n}`, the *k*-th of the *n*
patches covering `paths[i]`. `GridStitcher` stitches the predictions back,
in whatever order the loader delivers them, and hands back each file's volume
as soon as its last patch arrives:

```python
from medh5.torch import GridPatchDataset, GridStitcher

ds = GridPatchDataset(paths, patch_size=(8, 8, 8), overlap=2, images=["CT"])
stitcher = GridStitcher(ds, mode="gaussian")      # or "constant": a plain mean
loader = DataLoader(ds, batch_size=4, collate_fn=collate)
for batch in loader:
    predictions = batch["images"]["CT"]           # your model's output here
    for path, volume in stitcher.add(predictions, batch["meta"]):
        print(path, volume.shape)                 # one file, stitched
stitcher.finish()
```

A prediction is shaped as its patch, after `channels` when the model returns
channels (`GridStitcher(ds, channels=3)`); the padding a patch carries past the
volume is cut off. Overlaps are blended with Gaussian weights, highest at a
patch's centre, or averaged (`mode="constant"`): `medh5.sampling.BLEND_MODES`
lists the two, and `medh5.sampling.blend_weights(patch_size, mode=...)` returns
a patch's weights. `stitcher.pending` maps each
file not yet returned to the patches it is missing, and `finish()` refuses,
naming every such file, rather than return a volume with holes —
`finish(partial=True)` returns what arrived, `NaN` where nothing did. One
accumulator is open per file in flight: a float32 volume per channel, and one
of weights.

The tiling itself is `medh5.sampling.grid_patches(shape, patch_size,
overlap=...)`, and the stitching `medh5.sampling.PatchAggregator`, which need no
torch, for an inference loop of your own:

```python
from medh5.sampling import PatchAggregator, grid_patches

volume = s.images["CT"].read(physical=True)
aggregator = PatchAggregator(volume.shape, mode="gaussian")
for patch in grid_patches(volume.shape, (8, 8, 8), overlap=2):
    aggregator.add(patch.apply_padding(volume[patch.slices]), patch)
assert np.allclose(aggregator.result(), volume, atol=1e-3)
```

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
names the change-label annotation explicitly instead: a classification, carried
only by the pair it spans. `annotation=` is drawn from at its own visit, and
every other visit selects its own. An item's `images` and
`valid` are keyed by timepoint first (`item["images"]["tp0"]["CT_tp0"]`), so
`collate` batches pairs of the same two visits, and names the key when a batch
mixes others.

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
  `label_format="labelmap"` is an `int64` plane of the requested classes' ids,
  every other class background, and writes `65535` there, so it is a
  `CrossEntropyLoss(ignore_index=65535)` target as it stands, given a logit per
  class id up to the largest requested. The one-hot planes are `0` there: mask
  the loss with `ignore`.
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
raising from inside `torch.stack`, and it refuses by name a key some items of
the batch carry and others do not.

`used_index` is worth logging. `False` means the sampler fell back to scanning
the volume because no current sampling index held the classes it could draw —
the difference between 0.03 ms and 2.4 s per draw at 512³. `None` means the
draw was uniform and consulted no index, so there is nothing to report.

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
instead of `shuffle=True`, not with it. It works with the four datasets above
and with `ClinicalTaskDataset`, whose rows it groups by subject (every row of a
subject reads the same sources); each says which items read which file
(`file_groups()`). With workers `samples_per_volume` should be a multiple of
`batch_size`, since a file whose items straddle two batches is opened by both
workers.

Under DDP, `FileGroupedSampler` takes the place of `DistributedSampler`, which
deals items round-robin and so hands every rank a patch of every file. Each
rank takes one contiguous share of the epoch's order — the same length on
every rank, every epoch, so no collective waits on a rank that ran out — and a
file is split between ranks only where a share ends: at most
`num_replicas - 1` files an epoch are read by two neighbouring ranks.
`num_replicas` and `rank` default to `torch.distributed`'s world size and rank,
when a process group is initialised:

```python
from medh5.torch import FileGroupedSampler

sampler = FileGroupedSampler(ds, seed=0, num_replicas=2, rank=1)
loader = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate)
for epoch in range(epochs):
    ds.set_epoch(epoch)
    sampler.set_epoch(epoch)     # on every rank, so the ranks agree on the order
    for batch in loader:
        pass
```

The epoch's order is extended cyclically to a multiple of `num_replicas`, as
`DistributedSampler` pads; `drop_last=True` cuts it to one instead.

The cache is shared by the threads of a process and locked. A handle is held
for the length of an item, so a thread-based loader never has one closed under
it by another thread's eviction.

The cache is `medh5.torch.CACHE`, a `HandleCache`, and `open_cached(path,
sample_key=None)` opens a sample — or a member of a `.medh5c` collection —
through it. The datasets read through `CACHE.lease(path, sample_key,
content_id=...)`, which holds the handle until the block ends. Given the
`content_id` a task's row pins, a lease reopens a handle whose file `amend`
replaced after it was opened, and refuses a sample that is another version
all the same (`T302`): what the row pins is what it reads. Every new lease
re-checks the source, with one `stat` against the file's identity when its
handle was opened, so a worker with the file cached and one without agree; a
lease in progress finishes on the bytes it opened. `VolumeDataset`,
`PatchDataset`, `GridPatchDataset` and `PairedPatchDataset` lease without a
pin, so after an `amend` during training one epoch can read both versions of a
file: a worker that has the old one open keeps reading it until the handle is
evicted. `clear()` (and `worker_init_fn`) abandons handles inherited across a
`fork`, and closes the idle ones the process opened itself.

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
It is applied to every candidate class's voxel count, from the index or from a
scan alike; a class weighted zero is never drawn, and a draw with no candidate
that has both foreground and weight is a uniform one.

Foreground sampling is O(1) in the volume **if the file carries a sampling
index**:

```
$ medh5 index build cohort/*.medh5
```

Without one the sampler scans, still works, and records `used_index=False`.

## Clinical tasks (format 1.1)

`ClinicalTaskDataset` turns the rows of a [task manifest](../spec/task-cache-1.md)
— a subject at a cutoff — into items; `collate_clinical` batches them.
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
| `concepts` | A `ConceptVocabulary` — fitted on the training partition when omitted, refused when fitted on anything else (T405). It holds each concept's statistics in the one unit its training values were in (a concept whose training values are in two units is refused at fit time) and the text values each concept took |
| `documents` | An event-level feature cache's path, or an encoder with `encode(text)` and `dim` |
| `row_features` | A patient-level cache, validated against the task before any row reads it |
| `base` | The directory relative source URIs resolve against — the manifest's own directory when omitted |
| `strict` | Refuse a task whose preflight has findings (default), or keep only the unaffected rows |
| `preflight` | This manifest's `Preflight`, when it has been run already; one of another task, or another instance of this one, is refused (T404). Run here when omitted |
| `class_ignore` | Add `class_ignore[slot]`, the ignore mask per class (below); off by default |

A batch, for `B` rows:

| Key | Shape | Meaning |
|---|---|---|
| `images[slot]` | `(B, C, *patch)` | The slot's window on its image's own grid; zeros where the slot is empty, with the channel count of the images that fill the slot anywhere in the task (refused, `T102`, when they disagree). A slot without a `patch` reads whole volumes, which batch only when they agree: give it a `patch`, or use `batch_size=1` |
| `present[slot]` | `(B,)` bool | Modality availability: an eligible image filled the slot |
| `valid[slot]` | `(B, *patch)` bool | Field of view: inside the image and its valid region, never the padding |
| `image_time[slot][...]` | `(B, 2)`, `(B,)` | The image's acquisition (`start_age_h`) and availability (`available_age_h`), as ages before the cutoff, with `start_known` and `available_known` |
| `label[slot]`, `annotated[slot]` | `(B, K, *patch)`, `(B, K)` | Supervision for the slot's `classes`, and whether each was examined (coverage) |
| `ignore[slot]` | `(B, *patch)` bool | Voxels a loss must not score: padding, and the ignore regions of the annotations that label or examined one of the slot's `classes` (an annotation of other classes only has no say). It is one mask for every class, so with several classes a region one annotation ignores is masked for the others' supervision too |
| `class_ignore[slot]` | `(B, K, *patch)` bool | With `class_ignore=True`: channel *k* is the padding and the ignore regions of the annotations that label or examined class *k* only (1.0 §7.7: a region belongs to its annotation), in the slot's class order; all `True` where the slot is empty. `ignore` is the union of its channels |
| `events[...]` | `(B, N)`, `(B, N, 2)` | The admitted versions in input order, padded to the longest history; `mask` marks real events, `length` counts them (below) |
| `documents[...]` | `(B, M, D)`, `(B, M)`, `(B, M, 2)` | `features` of the admitted documents, `event` (the owning version's position in `events`), `start_age_h` and `available_age_h` with their `*_known`; `mask` and `length` |
| `target["value"]`, `target["observed"]` | `(B,)` | The label, and whether it is observed (a censored row is not) |
| `meta` | list | Row id, subject, partition, cutoff, fingerprint, the admitted event ids, and per slot the visit that filled it |

The event sequence keeps what 1.1 §5 distinguishes, and never collapses a time
to one number:

| `events[...]` | Shape | Meaning |
|---|---|---|
| `concept`, `kind`, `status`, `temporal_type` | `(B, N)` int | Vocabulary index (0 padding, 1 unseen); the kind, status and temporal type as their vocabulary's index + 1 (0 padding) |
| `value`, `has_value` | `(B, N)` | The value, normalised by the fitted statistics; whether there is one — an absent value is not 0 |
| `unit` | `(B, N)` int | 2 when the value is in the unit its concept was fitted in (and so normalised); 1 when it is in another, or its concept was not fitted — then `value` is 0; 0 without a value |
| `value_index` | `(B, N)` int | The text value (`value_text`) as a fitted `(concept, value)` pair's index: 0 none, 1 a value the concept never took in training, fitted pairs from 2 (`ConceptVocabulary.n_values` in all) |
| `comparator` | `(B, N)` int | `eq`, `lt`, ... as `COMPARATORS` index + 1 (`eq` when a value has none); 0 without a value |
| `missing` | `(B, N)` bool | An expected result missing for a source reason: absence, never a negative |
| `start_age_h`, `end_age_h`, `available_age_h` | `(B, N, 2)` | Effective start, effective end and availability as ages before the cutoff in hours: `[..., 0]` the least, `[..., 1]` the most, equal for an exact instant |
| `start_known`, `end_known`, `available_known` | `(B, N)` bool | Whether that time is known; an unknown one is zeros, which are not a time |
| `tie_group` | `(B, N)` int | Versions whose ordering times overlap share a group: their order is the storage tie-break, not evidence |
| `plan` | `(B, N)` bool | Admitted as a plan (`plans` policy): `planned`, or its start not definitely at or before the cutoff |

An age is negative only for what a version available at the cutoff itself recorded about later — a
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

The affine is in millimetres, which is what MONAI and the NIfTI files it saves
read: a grid in `m` or `um` is scaled, and one in `px`, which has no size,
stays in pixels (`meta["medh5"]["units"]` names the grid's unit). It is passed
through in the grid's own `coord_system` — usually LPS — and labelled with it in
the metadata MONAI reads, rather than silently converted. Pass `space="RAS"`
when a consumer needs RAS: the conversion is a sign flip on the first two world
axes of the affine, never a flip of the voxels. `from_metatensor(tensor)`
returns the array and its geometry — spacing, origin and direction in the
tensor's space, and their `units` — and refuses a tensor whose metadata states
no affine or no space, which has none to return.

`to_dict(sample, images, annotations)` builds a dictionary-transform item, with
the same `space=` and `physical=` options. Every tensor's `filename_or_obj` is
the sample's path, so `SaveImage` names its output after the sample.
Annotations arrive as `int64` label volumes with their own grid's affine, and
every voxel of the §7.7 ignore region is `65535` whichever encoding stores it —
pass `ignore_index=65535` to the loss.

An identifier a sample does not have — a `frame_uid`, `value_units`,
`modality`, `timepoint`, `subject_id`, or the `sample_key` of a file outside a
collection — is `"none"` in `meta["medh5"]`, MONAI's own spelling: its
`list_data_collate` refuses a `None`, so samples without a frame of reference
(every NIfTI and nnU-Net import) could not be batched. `from_metatensor` reads
`"none"` back as `None`.

### A reader for `LoadImage`

`MedH5Reader` is a MONAI `ImageReader`, so `LoadImage` and `LoadImaged` open a
`.medh5` — or a collection member, written `path::key` as a split names it —
with the affine `to_metatensor` gives:

```python
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
from medh5.monai import MedH5Reader

load = Compose([
    LoadImaged("image", reader=MedH5Reader("CT")),
    LoadImaged("label", reader=MedH5Reader(annotation="organs"), dtype=None),
    EnsureChannelFirstd(["image", "label"]),
])
item = load({"image": "case.medh5", "label": "case.medh5"})
member = LoadImaged("image", reader=MedH5Reader("CT"))({"image": "shard.medh5c::case_0001"})
```

`MedH5Reader()` without an image reads the sample's only image, and refuses a
sample holding several, naming them. `annotation=` reads a voxel annotation as
`to_dict` does — `int64`, `65535` in the ignore region — so pass `dtype=None` to
`LoadImaged`, which otherwise casts to `float32`. Several files given to one
call stack as channels, only when they share one grid. The reader leases each
file from the per-process handle cache, so a worker opens a file once however
many items read it.

`LoadMedH5d` is `to_dict` as a dictionary transform: one key naming the sample,
and one output key per image and annotation, read from one lease:

```python
from monai.transforms import Compose, EnsureChannelFirstd, Spacingd
from medh5.monai import LoadMedH5d

pipeline = Compose([
    LoadMedH5d("path", images=["CT"], annotations=["organs"]),
    EnsureChannelFirstd(["CT", "organs"]),
    Spacingd(["CT", "organs"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
])
item = pipeline({"path": "case.medh5"})
```

`import medh5.monai` imports no MONAI: the reader and the transform subclass
MONAI's, and are loaded when they are first asked for. The affine construction
(`affine_for`, `convert_affine`) does not import MONAI either, so the geometry is
testable — and tested — in an environment without it.

## Related

- **[Tune performance](../guides/performance.md)** — the measured numbers, and
  the four levers behind them.
- **[Storage](storage.md)** — codec profiles, chunking, and the sampling index
  these datasets read.
- **[Longitudinal studies](../guides/longitudinal.md)** — what `PairedPatchDataset`
  is for.
- **[Train on clinical tasks](../guides/clinical-training.md)** — what
  `ClinicalTaskDataset` is for, and the masks it keeps apart.
