"""A subject split across two collection members, reconciled and trained on.

Two sites each exported part of one subject's history: site A the baseline CT
and the reports, site B the follow-up CT and the later responses.  Both
exported the same pre-baseline laboratory value.  The two exports are members
of one ``.medh5c`` shard, beside a second subject's single sample.

The task names each member by shard and key, pinned to the member's own
``content_id`` (packing does not change it).  Preflight refuses the subject
until the duplicated event is **reconciled** --- one content, recorded in the
manifest --- and then fills the CT slot from whichever fragment holds the
newest eligible image: site A's baseline at 24 h, site B's follow-up at day 95.
The batch is loaded through worker processes, which open members through the
PID-keyed handle cache.

Run it::

    python docs/examples/clinical_collection.py [outdir]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.clinical import DAY, HOUR, Clock, Document, Event, Link
from medh5.collection import pack
from medh5.task import Slot, SourceRef, Target, TaskManifest

SHAPE = (12, 24, 24)
RECIST = "org.example.recist"
CLOCK = Clock.relative("clock-P-10", "acquisition start of the baseline CT at site A")


def lab0() -> Event:
    """The value both sites exported: one event, one content."""
    return Event(
        "lab0",
        "lab0",
        "observation",
        "point",
        "final",
        effective_start_us=-48 * HOUR,
        available_us=-47 * HOUR,
        code_system="http://loinc.org",
        code="2160-0",
        value_num=1.1,
        unit="mg/dL",
    )


def scan(w: Any, image_id: str, timepoint: str, days: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w.add_timepoint(timepoint, days_from_baseline=days)
    w.add_grid(f"ct_{timepoint}", shape=SHAPE, spacing=(2.0, 0.8, 0.8), timepoint=timepoint)
    volume = np.rint(rng.normal(40.0, 12.0, SHAPE)).astype(np.int16)
    w.add_image(image_id, volume, grid=f"ct_{timepoint}", modality="CT")


def site_a(path: Path) -> None:
    """Baseline CT, the reports, and the shared laboratory value."""
    with medh5.create(path, sample_id="P-10-site-a", subject_id="P-10") as w:
        scan(w, "CT_tp0", "tp0", 0, seed=1)
        w.set_clock(CLOCK)
        w.add_event(lab0())
        w.add_event(
            Event("ct0", "ct0", "imaging", "point", "final", effective_start_us=0,
                  available_us=HOUR, timepoint_id="tp0")
        )
        w.add_link(Link.between(("event", "ct0"), "describes", ("image", "CT_tp0")))
        for version, available, status in (("v1", 4, "preliminary"), ("v2", 48, "amended")):
            w.add_event(
                Event(f"report0_{version}", "report0", "document", "point", status,
                      effective_start_us=0, available_us=available * HOUR)
            )
            w.add_document(Document(f"report0_text_{version}", f"Baseline CT, read {version}."))
            w.add_link(
                Link.between(("event", f"report0_{version}"), "describes",
                             ("document", f"report0_text_{version}"))
            )
        w.add_link(Link.between(("event", "report0_v2"), "supersedes", ("event", "report0_v1")))


def site_b(path: Path) -> None:
    """Follow-up CT and the responses, on the same subject clock."""
    with medh5.create(path, sample_id="P-10-site-b", subject_id="P-10") as w:
        scan(w, "CT_tp1", "tp1", 90, seed=2)
        w.set_clock(CLOCK)
        w.add_event(lab0())
        w.add_event(
            Event("ct1", "ct1", "imaging", "point", "final", effective_start_us=90 * DAY,
                  available_us=90 * DAY + HOUR, timepoint_id="tp1")
        )
        w.add_link(Link.between(("event", "ct1"), "describes", ("image", "CT_tp1")))
        for event_id, day, value in (("recist1", 90, "SD"), ("recist2", 200, "PD")):
            w.add_event(
                Event(event_id, event_id, "assessment", "point", "final",
                      effective_start_us=day * DAY, available_us=(day + 1) * DAY,
                      code_system=RECIST, code="overall_response", value_text=value)
            )


def other_subject(path: Path) -> None:
    """A second subject, in one sample."""
    clock = Clock.relative("clock-P-11", "acquisition start of the baseline CT")
    with medh5.create(path, sample_id="P-11", subject_id="P-11") as w:
        scan(w, "CT_tp0", "tp0", 0, seed=3)
        w.set_clock(clock)
        w.add_event(
            Event("ct0", "ct0", "imaging", "point", "final", effective_start_us=0,
                  available_us=HOUR, timepoint_id="tp0")
        )
        w.add_link(Link.between(("event", "ct0"), "describes", ("image", "CT_tp0")))
        w.add_event(
            Event("recist1", "recist1", "assessment", "point", "final",
                  effective_start_us=90 * DAY, available_us=91 * DAY,
                  code_system=RECIST, code="overall_response", value_text="PD")
        )


def main(out: Path, *, workers: int = 2) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from medh5.torch import ClinicalTaskDataset, collate_clinical, worker_init_fn

    out.mkdir(parents=True, exist_ok=True)
    parts = {"P-10-a": out / "a.medh5", "P-10-b": out / "b.medh5", "P-11": out / "c.medh5"}
    site_a(parts["P-10-a"])
    site_b(parts["P-10-b"])
    other_subject(parts["P-11"])
    shard = pack(list(parts.values()), out / "cohort.medh5c", keys=list(parts))

    task = TaskManifest.new(
        "progression-180d",
        "1",
        identity_namespace="example-network",
        slots=[Slot("ct", "CT", required=True, patch=(8, 16, 16))],
        target=Target(
            "progression", "1", RECIST, "overall_response", positive=("PD",),
            negative=("SD", "PR", "CR"), horizon_us=180 * DAY, min_follow_up_us=60 * DAY,
        ),
        split=("fold-0", ["train", "val"]),
        base=out,
    )
    members = {key: SourceRef.pin(shard, sample_key=key, uri=shard.name) for key in parts}
    task.add_subject("P-10", [members["P-10-a"], members["P-10-b"]], partition="train")
    task.add_subject("P-11", [members["P-11"]], partition="train")
    for subject in ("P-10", "P-11"):
        task.add_row(f"{subject}@24h", subject, 24 * HOUR)
        task.add_row(f"{subject}@d95", subject, 95 * DAY)

    # Both fragments hold `lab0`: preflight refuses the subject until the
    # manifest records the one content they share.
    before = task.preflight()
    unreconciled = sorted({f.code for f in before.findings})
    task.reconcile()
    after = task.preflight()
    assert after.ok, [str(f) for f in after.findings]
    task.save(out / "network.task.json")

    dataset = ClinicalTaskDataset(task, preflight=after)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=workers,
        worker_init_fn=worker_init_fn if workers else None,
        collate_fn=collate_clinical,
    )
    batch = next(iter(loader))
    return {
        "members": [m.locator for m in members.values()],
        "before reconciling": unreconciled,
        "preflight": after.counts,
        "rows": [m["row_id"] for m in batch["meta"]],
        "ct source": [m["visits"]["ct"]["source"] for m in batch["meta"]],
        "ct image": [m["visits"]["ct"]["image_id"] for m in batch["meta"]],
        "events": batch["events"]["length"].tolist(),
        "target": batch["target"]["value"].tolist(),
        "observed": batch["target"]["observed"].tolist(),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for key, value in main(target).items():
        print(f"{key:>18}: {value}")
