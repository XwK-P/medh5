"""Feature caches over a clinical task, and every way one stops being usable.

Builds the cohort of ``clinical_longitudinal.py`` and two caches over it:

- an **event-level** cache of report features --- one per event version of a
  pinned source, shared by every row and every cutoff;
- a **patient-level** cache --- one feature per training row, pooled from the
  report versions that row admits, normalised with statistics fitted on the
  training partition, and pinned to the row's cutoff and selected versions.

Then it breaks them, one way at a time, and shows that validation tells the
failures apart --- because each needs a different fix:

==========================  ======  =========================================
what happened               code    what to do
==========================  ======  =========================================
a source was amended        T403    stale: rebuild the entries of that source
a cache entry's bytes rot   T402    corrupt: rebuild the cache
a whole-history embedding   T406    inadmissible at the row's cutoff
fitted on the wrong split   T405    refit on the training partition
==========================  ======  =========================================

Run it::

    python docs/examples/clinical_cache.py [outdir]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import medh5
from medh5.cache import (
    CacheWriter,
    FeatureCache,
    HashingTextEncoder,
    build_document_cache,
    fitted_on,
    validate_cache,
)
from medh5.clinical import DAY, Event
from medh5.task import Preflight, SourceRef, TaskManifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_longitudinal import progression_task, write_cohort  # noqa: E402

ENCODER = HashingTextEncoder(dim=16)


def report_features(task: TaskManifest, report: Preflight, events: FeatureCache) -> dict[str, Any]:
    """Per training row: the mean of its admitted report features."""
    rows = {}
    for row in report.eligible(task.training_partition):
        features = []
        for event, fragment in zip(row.events, row.event_fragments, strict=True):
            if event.kind != "document":
                continue
            source = row.sources[fragment]
            found = events.event_feature(source.content_id, event.event_id)
            assert found is not None, event.event_id
            features.append(found)
        rows[row.row_id] = (row, np.mean(features, axis=0).astype(np.float32))
    return rows


def patient_cache(
    path: Path,
    task: TaskManifest,
    rows: dict[str, Any],
    *,
    whole_history: dict[str, list[str]] | None = None,
    partition: str | None = None,
) -> None:
    """A patient-level cache, its normalisation fitted on ``partition``."""
    stacked = np.stack([feature for _, feature in rows.values()])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    with CacheWriter(
        path,
        level="patient",
        task=task,
        fitted_on=fitted_on(task, partition),
        **{**ENCODER.header(), "preprocessing": {"pooling": "mean", "normalise": "train-z"}},
    ) as w:
        for row_id, (row, feature) in rows.items():
            versions = [e.event_id for e in row.events]
            if whole_history is not None:
                versions = whole_history[row.subject_id]
            w.add(
                row_id,
                ((feature - mean) / std).astype(np.float32),
                sources=list(row.sources),
                row_id=row_id,
                row_fingerprint=row.fingerprint,
                cutoff_us=row.cutoff_us,
                event_versions=versions,
            )


def all_versions(path: Path) -> list[str]:
    with medh5.open(path) as sample:
        assert sample.clinical is not None
        return [e.event_id for e in sample.clinical.events]


def codes(report: Any) -> list[str]:
    return sorted({f.code for f in report.findings})


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    task = progression_task(out, write_cohort(out))
    task.save(out / "progression.task.json")
    preflight = task.preflight()
    assert preflight.ok

    events_path = out / "reports.medh5cache"
    assert build_document_cache(task, events_path, ENCODER).ok
    with FeatureCache.open(events_path) as events:
        rows = report_features(task, preflight, events)
    patient_path = out / "rows.medh5cache"
    patient_cache(patient_path, task, rows)
    results: dict[str, Any] = {
        "fresh": (codes(validate_cache(events_path)), codes(validate_cache(patient_path, task=task)))
    }

    # 1. A whole-history embedding: each row encodes every report version of
    #    its subject, including the amendment a 24 h row cannot have read.
    everything = {s.subject_id: all_versions(out / f"{s.subject_id}.medh5") for s in task.subjects}
    leaky = out / "leaky.medh5cache"
    patient_cache(leaky, task, rows, whole_history=everything)
    results["whole history"] = codes(validate_cache(leaky, task=task))

    # 2. Statistics fitted on the validation partition, not the training one.
    misfit = out / "misfit.medh5cache"
    patient_cache(misfit, task, rows, partition="val")
    results["fitted on val"] = codes(validate_cache(misfit, task=task))

    # 3. A cache entry's bytes change on disk.
    rotten = out / "rotten.medh5cache"
    shutil.copyfile(events_path, rotten)
    with h5py.File(rotten, "r+") as f:
        name = sorted(f["entries"])[0]
        f["entries"][name][0] += 1.0
    report = validate_cache(rotten)
    results["corrupted bytes"] = (codes(report), len(report.corrupt), len(report.stale))

    # 4. A source is amended: its entries are stale, the others are not, and
    #    the task's pin fails until it is renewed on purpose.
    with medh5.amend(out / "P-02.medh5") as w:
        w.add_event(Event("note9", "note9", "other", "point", "final",
                          effective_start_us=300 * DAY, available_us=300 * DAY))
    report = validate_cache(events_path)
    results["amended source"] = (codes(report), len(report.stale), report.entries)
    results["task after amend"] = codes(task.preflight())

    repinned = TaskManifest(task.to_json(), base=out)
    doc = repinned.to_json()
    for subject in doc["subjects"]:
        for source in subject["sources"]:
            fresh = SourceRef.pin(out / source["uri"], uri=source["uri"],
                                  source_id=source["source_id"])
            source["content_id"] = fresh.content_id
    repinned = TaskManifest(doc, base=out)
    results["repinned task"] = (
        codes(repinned.preflight()),
        repinned.task_fingerprint == task.task_fingerprint,
        repinned.manifest_fingerprint == task.manifest_fingerprint,
        repinned.row_fingerprint("P-02@24h") == task.row_fingerprint("P-02@24h"),
        repinned.row_fingerprint("P-01@24h") == task.row_fingerprint("P-01@24h"),
    )
    assert build_document_cache(repinned, events_path, ENCODER).ok
    results["rebuilt"] = codes(validate_cache(events_path))
    return results


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for key, value in main(target).items():
        print(f"{key:>17}: {value}")
