# Train on clinical tasks

A prediction made at a moment may only use what was known at that moment. This page builds a
**task** over clinical samples --- which subjects, which cutoffs, what a row may read, what it
predicts --- checks it before training, and turns it into real batches: images from different
visits, missing modalities, histories of different lengths, document features, and a target that may
be unobserved. The contract is [Task and cache contract 1](../spec/task-cache-1.md); every decision
about what a row may see is the engine's, made once at preflight.

The cohort is the one [`docs/examples/clinical_longitudinal.py`](../examples/clinical_longitudinal.py)
writes: six subjects, each with a baseline and a three-month CT, an MR at follow-up for some, a
laboratory value, a report revised two days after the scan, and RECIST responses at day 90 and,
for some, day 200.

## Describe the task

A task names its **sources** --- each pinned to the `content_id` it was built from --- assigns every
**subject** to a partition before any row exists, and lists the **rows**: a subject at a cutoff.
Slots are stable names for modalities, never filenames; the target is an event concept read after
the cutoff.

```python
from medh5.clinical import DAY, HOUR
from medh5.task import Slot, SourceRef, Target, TaskManifest

task = TaskManifest.new(
    "progression-180d", "1",
    identity_namespace="example-site",
    slots=[
        Slot("ct", "CT", required=True, patch=(8, 16, 16), roi="eligible_instances",
             classes=(3,)),
        Slot("mr", "MR", patch=(8, 16, 16)),
    ],
    target=Target("progression", "1", "org.example.recist", "overall_response",
                  positive=("PD",), negative=("SD", "PR", "CR"),
                  horizon_us=180 * DAY, min_follow_up_us=60 * DAY, kind="assessment"),
    split=("fold-0", ["train", "val"]),          # the training partition first
    base="cohort",
)
for subject, partition in [("P-01", "train"), ("P-02", "train"), ("P-05", "val")]:
    source = SourceRef.pin(f"cohort/{subject}.medh5", uri=f"{subject}.medh5")
    task.add_subject(subject, [source], partition=partition)
    task.add_row(f"{subject}@24h", subject, cutoff_us=24 * HOUR)
    task.add_row(f"{subject}@d95", subject, cutoff_us=95 * DAY)
task.save("cohort/my.task.json")

assert task.validate() == []
assert task.task_fingerprint != task.manifest_fingerprint
```

The **task fingerprint** identifies the definition --- policy, slots, target --- and not the rows, so
two spellings of one task share it. The **row fingerprint** adds the subject, the pinned source
versions and the cutoff: an example's identity depends on the data it reads, not on a patient id.

## Check it before training

Preflight opens every source once, checks every pin --- recomputing the clinical datasets' digests,
not trusting the stored root --- and says per row what it may read and why it is in or out. No voxel
or report text is read.

```python
from medh5.task import TaskManifest

task = TaskManifest.load("cohort/progression.task.json")
report = task.preflight()
assert report.ok
assert report.counts == {"eligible": 10, "excluded": 2}

early = report.row("P-02@24h")
assert [e.event_id for e in early.events] == ["lab0", "ct0", "report0_v1"]
assert early.slots["ct"].image_id == "CT_tp0" and not early.slots["mr"].available

late = report.row("P-01@d95")
assert (late.status, late.reasons) == ("excluded", ("prevalent_target",))
```

```bash
medh5 task validate cohort/progression.task.json
medh5 task preflight cohort/progression.task.json
```

| Status | Meaning |
|---|---|
| `eligible` | Certified inputs, every required slot filled, a usable target |
| `uncertifiable` | A later revision's availability is unknown or straddles the cutoff: what the row would read cannot be vouched for |
| `excluded` | A required slot is empty, the outcome had already happened (`prevalent_target`), or the target is censored and the task excludes censored rows |
| `error` | Its sources are wrong: a changed pin (T302), a fragment of another subject (T303), clocks that disagree (T304), unreconciled duplicates (T305) |

A changed source fails its pin; nothing is re-pinned behind your back. Pin again on purpose when the
change is meant.

## Train on it

```python
from torch.utils.data import DataLoader
from medh5.torch import ClinicalTaskDataset, collate_clinical

train = ClinicalTaskDataset("cohort/progression.task.json", partition="train")
loader = DataLoader(train, batch_size=8, collate_fn=collate_clinical)
batch = next(iter(loader))

batch["images"]["ct"].shape            # (7, 1, 8, 16, 16): every train row
batch["present"]["mr"]                 # which rows have an eligible MR
batch["valid"]["ct"]                   # field of view: in the image, never the padding
batch["annotated"]["ct"]               # coverage: was the class examined at all
batch["events"]["concept"].shape       # (7, longest history in the batch)
batch["events"]["mask"]                # which positions are events, not padding
batch["target"]["value"], batch["target"]["observed"]
[m["visits"]["ct"]["timepoint"] for m in batch["meta"]]   # tp0 at 24 h, tp1 at day 95

# The validation rows, with the vocabulary fitted on the training partition.
val = ClinicalTaskDataset("cohort/progression.task.json", partition="val",
                          concepts=train.concepts)
assert train.concepts.fitted_on["partition"] == "train"
```

What each part of a batch means, and the masks that keep it honest:

- **Images** come from the newest image of the slot's modality that the row's selection admits ---
  baseline at 24 h, follow-up at day 95 --- read as a window on the image's own grid. Nothing is
  resampled. With `roi="eligible_instances"` the window is centred on an annotation *available at the
  cutoff*; before the lesion masks were drawn, the crop falls back to the grid centre
  (`meta["visits"]["ct"]["roi"] == "center_fallback"`), because centring it on a mask drawn later
  would leak the future through the crop.
- **`present`** is modality availability. A missing MR is zeros with `present = False`: a zero-filled
  modality is not an observed negative.
- **`valid`** is the field of view: inside the image and its valid region, never the padding.
- **`label`, `annotated` and `ignore`** are segmentation supervision for the slot's classes. Like the
  target, supervision may come from after the cutoff, and it never enters an input. `annotated` says
  whether a class was examined: a 0 in an unexamined class is not a negative.
- **`events`** is the admitted history in clinical order --- concept index, kind, normalised value,
  hours before the cutoff --- padded to the longest history in the batch, with `mask` marking real
  events. A revised report contributes the version known at the cutoff.
- **`target`** is read from the full history: `observed = False` for a censored row, whose inputs
  still train but whose loss term should be masked.

**Learned preprocessing is fitted on the training partition.** The concept vocabulary and its
per-concept value statistics record the split they were fitted on (`fitted_on`); a vocabulary fitted
on anything else is refused (T405).

## Read documents

Give the dataset an encoder, and each row gets one feature per admitted document --- the version known
at its cutoff --- read when the item is built:

```python
from medh5.cache import HashingTextEncoder
from medh5.torch import ClinicalTaskDataset

ds = ClinicalTaskDataset("cohort/progression.task.json", partition="train",
                         documents=HashingTextEncoder(dim=32))
item = ds[0]
item["documents"]["features"].shape    # (admitted documents, 32)
item["meta"]["document_events"]        # ['report0_v1'] at 24 h
```

`HashingTextEncoder` is a deterministic, dependency-free fixture, not a language model. A real encoder
goes in the same place --- anything with `encode(text) -> array` and a `dim` --- or, better, into a
cache.

## Cache features

An **event-level** cache holds one feature per event version of a pinned source. A version is
immutable, so its feature is shared by every row and every cutoff; each row's selection decides which
of them it may read.

```python
from medh5.cache import HashingTextEncoder, build_document_cache
from medh5.task import TaskManifest
from medh5.torch import ClinicalTaskDataset

task = TaskManifest.load("cohort/progression.task.json")
report = build_document_cache(task, "cohort/reports.medh5cache", HashingTextEncoder(dim=32))
assert report.ok

ds = ClinicalTaskDataset(task, partition="train", documents="cohort/reports.medh5cache")
```

The cache records the encoder and its immutable revision, the preprocessing, the output dtype and
shape, every source version each entry read, and checksums over itself and every entry. It is
validated before a dataset reads it, and it is told apart from its sources in the way it fails:

```python
import medh5
from medh5.cache import HashingTextEncoder, build_document_cache, validate_cache
from medh5.clinical import DAY, Event
from medh5.task import TaskManifest

task = TaskManifest.load("cohort/progression.task.json")
build_document_cache(task, "cohort/reports.medh5cache", HashingTextEncoder(dim=32))

with medh5.amend("cohort/P-02.medh5") as w:            # a source changes...
    w.add_event(Event("note9", "note9", "other", "point", "final",
                      effective_start_us=300 * DAY, available_us=300 * DAY))

report = validate_cache("cohort/reports.medh5cache")
assert report.stale and not report.corrupt             # ...so its entries are stale
assert not task.preflight().ok                         # and the task's pin fails (T302)
```

```bash
medh5 cache validate cohort/reports.medh5cache --task cohort/progression.task.json
```

- **Stale** (T403): a source no longer has the content an entry pins. Rebuild the entry.
- **Corrupt** (T401, T402): the cache's own bytes no longer match their checksums. Rebuild the cache.
- **Not this task's** (T404--T406): built for another task or cutoff, fitted on another split, or ---
  for a patient-level feature --- encoding event versions its row does not admit, such as a
  whole-history embedding.

A cache never redefines its sources: a failed validation rejects the cache, and the samples are
untouched.

## Fragments and collection members

A subject may be spread over several samples --- one per export, or members of a `.medh5c` shard ---
on one clock. List them all as the subject's sources; preflight checks that every fragment is that
subject, that their clocks agree, and that an event present in several fragments is the same event.
Record the duplicates once:

```bash
medh5 task reconcile cohort/progression.task.json --out cohort/reconciled.task.json
```

A member of a collection is named by the shard and its key, and pins the member's own `content_id`
--- packing does not change it:

```python
from medh5.collection import pack
from medh5.task import SourceRef

pack(["cohort/P-05.medh5", "cohort/P-06.medh5"], "cohort/val.medh5c", keys=["P-05", "P-06"])
member = SourceRef.pin("cohort/val.medh5c", sample_key="P-05", uri="val.medh5c")
assert member.locator == "val.medh5c::P-05"
assert member.check(base="cohort") == []
assert member.content_id == SourceRef.pin("cohort/P-05.medh5").content_id
```

[`docs/examples/clinical_collection.py`](../examples/clinical_collection.py) trains on a subject split
across two members of a shard.

## Workers

Handles follow the 1.0 §14.4 rules. Every source --- file or member --- is opened through the
per-process cache of [`medh5.torch`](../reference/torch.md), keyed by `(path, sample_key)`, so a
forked worker abandons its parent's handles instead of reading through them; a feature cache is
opened in the process that reads it and is abandoned, never closed, across a fork.

<!-- illustrative -->
```python
from torch.utils.data import DataLoader
from medh5.torch import collate_clinical, worker_init_fn

loader = DataLoader(train, batch_size=8, num_workers=4, persistent_workers=True,
                    worker_init_fn=worker_init_fn, collate_fn=collate_clinical)
```

The dataset holds only what pickles: the manifest, the row views from preflight, the fitted
vocabulary and the cache's path. Workers re-read nothing that preflight decided.
