# From clinical history to a training batch

<!-- sequential -->

A model that predicts at a moment may only use what was known at that moment. In this tutorial you
write two subjects whose scans and reads carry *when they happened* and *when they became known*,
ask a question of them at a cutoff, check which rows can be trained on and why one cannot, and load
a batch. About fifteen minutes, with PyTorch installed:

```bash
pip install "medh5[torch]"
```

Every block below runs after the one before it, in one Python session.

## Write two subjects

Each subject is one file: a baseline CT, a follow-up CT ninety days later, and the response a
radiologist read at follow-up. The history sits on a `utc` clock, so every time is an instant in
microseconds since 1970, made from an aware `datetime` by `utc_us`.

```python
import datetime as dt
from pathlib import Path

import numpy as np

import medh5
from medh5.clinical import DAY, HOUR, MINUTE, Clock, Event, Link, utc_us

RECIST = "org.example.recist"


def write_subject(path, subject_id, scanned, response):
    """Two CT visits and the response read at the second; returns the first scan's time."""
    t0 = utc_us(scanned)
    t1 = t0 + 90 * DAY
    rng = np.random.default_rng(0)
    with medh5.create(path, sample_id=subject_id, subject_id=subject_id) as w:
        w.set_clock(Clock.utc("clock"))
        for tp, day in (("tp0", 0), ("tp1", 90)):
            w.add_timepoint(tp, days_from_baseline=day)
            w.add_grid(f"ct_{tp}", shape=(12, 24, 24), spacing=(2.0, 0.8, 0.8), timepoint=tp)
            volume = rng.normal(40.0, 12.0, (12, 24, 24)).astype(np.int16)
            w.add_image(f"CT_{tp}", volume, grid=f"ct_{tp}", modality="CT")
        # A scan is acquired at t and reaches the archive an hour later.
        for event_id, tp, t in (("ct0", "tp0", t0), ("ct1", "tp1", t1)):
            w.add_event(Event(event_id, event_id, "imaging", "point", "final",
                              effective_start_us=t, available_us=t + HOUR,
                              timepoint_id=tp))
            w.add_link(Link.between(("event", event_id), "describes", ("image", f"CT_{tp}")))
        # The response is read at follow-up and signed the next day.
        w.add_event(Event("recist1", "recist1", "assessment", "point", "final",
                          effective_start_us=t1, available_us=t1 + DAY,
                          code_system=RECIST, code="overall_response", value_text=response))
    return t0


Path("history").mkdir(exist_ok=True)
utc = dt.timezone.utc
scanned = {
    "A": write_subject("history/A.medh5", "A", dt.datetime(2024, 3, 4, 9, tzinfo=utc), "PD"),
    "B": write_subject("history/B.medh5", "B", dt.datetime(2024, 5, 13, 14, tzinfo=utc), "SD"),
}
```

Subject A progressed (`PD`); subject B is stable (`SD`). The files are format 1.1, because they
carry the `clinical` profile; one without a history is still written as 1.0.

Each event has two times, never one: `effective_start_us` is when it happened, `available_us` when
it became known. Ask the file what was known half an hour after B's baseline scan, and a day after:

```python
with medh5.open("history/B.medh5") as subject:
    assert subject.version == "1.1" and "clinical" in subject.profiles
    early = subject.clinical.select(scanned["B"] + 30 * MINUTE)
    later = subject.clinical.select(scanned["B"] + DAY)

assert early.event_ids == []          # scanned, but not yet in the archive
assert later.event_ids == ["ct0"]     # the follow-up and its read are still to come
assert later.excluded == {"after_cutoff": 2}
```

## Describe the task

A **task** says what is predicted, from what, at which moments. Here: does the disease progress
within 180 days of the cutoff? The input is one CT **slot**, and the **target** is read from the
history *after* the cutoff, which is the one place the future is allowed. Each **row** is a subject
at a cutoff; each **source** is pinned to the `content_id` of the file it was built from.

```python
from medh5.task import Slot, SourceRef, Target, TaskManifest

task = TaskManifest.new(
    "progression-180d", "1",
    identity_namespace="tutorial",
    slots=[Slot("ct", "CT", required=True, patch=(8, 16, 16))],
    target=Target("progression", "1", RECIST, "overall_response",
                  positive=("PD",), negative=("SD", "PR", "CR"),
                  horizon_us=180 * DAY, min_follow_up_us=60 * DAY, kind="assessment"),
    base="history",
)
for subject_id, t0 in scanned.items():
    source = SourceRef.pin(f"history/{subject_id}.medh5", uri=f"{subject_id}.medh5")
    task.add_subject(subject_id, [source])
    task.add_row(f"{subject_id}@day1", subject_id, cutoff_us=t0 + DAY)
# One row too early: half an hour after B's scan.
task.add_row("B@30min", "B", cutoff_us=scanned["B"] + 30 * MINUTE)

assert task.validate() == []
task.save("history/progression.task.json")
```

## Check it before training

Preflight opens every source once, checks every pin, and decides for each row what it may read and
whether it can be trained on. No voxel is read.

```python
report = task.preflight()
assert report.ok, report.findings
assert report.counts == {"eligible": 2, "excluded": 1}

row = report.row("A@day1")
assert row.status == "eligible"
assert [e.event_id for e in row.events] == ["ct0"]
assert row.slots["ct"].image_id == "CT_tp0"
assert (row.target.status, row.target.value) == ("positive", 1.0)

too_early = report.row("B@30min")
assert (too_early.status, too_early.reasons) == ("excluded", ("missing_required_slot:ct",))
```

`B@30min` is out because its one required slot is empty: the scan had happened, but it was not yet
known, so nothing at that cutoff may read it. Its target is still defined; a row is excluded for
what it cannot see, not for what it would predict. The same check runs from the shell:

```bash
medh5 task preflight history/progression.task.json
```

## Load a batch

`ClinicalTaskDataset` serves the eligible rows exactly as preflight decided, and
`collate_clinical` pads what differs between rows: histories of different lengths, a modality one
row has and another lacks.

```python
from torch.utils.data import DataLoader

from medh5.torch import ClinicalTaskDataset, collate_clinical

rows = ClinicalTaskDataset(task, preflight=report)
loader = DataLoader(rows, batch_size=4, collate_fn=collate_clinical)
batch = next(iter(loader))

assert [m["row_id"] for m in batch["meta"]] == ["A@day1", "B@day1"]
assert tuple(batch["images"]["ct"].shape) == (2, 1, 8, 16, 16)
assert batch["target"]["value"].tolist() == [1.0, 0.0]
assert batch["target"]["observed"].tolist() == [True, True]
```

That is the whole path: a history with two times per event, a task over it, a preflight that says
why a row is out, and a batch built only from what each row was allowed to know.

## Where next

- [Clinical history beside the images](../guides/clinical.md): documents, revisions, and adding a
  history to a 1.0 file.
- [Train on clinical tasks](../guides/clinical-training.md): several slots and missing
  modalities, regions of interest, splits, document features, and feature caches.
- [Task and cache contract 1](../spec/task-cache-1.md): what a task file is, field by field.
