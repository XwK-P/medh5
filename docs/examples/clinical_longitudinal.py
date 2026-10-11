"""MEDH5 1.1 end to end: clinical context beside the images, a strict
prospective task over it, and a real training batch.

Each subject is one ``.medh5`` file holding two CT visits (baseline ``tp0``
and a three-month follow-up ``tp1``), an MR at follow-up for some subjects,
and --- under the ``clinical`` profile --- the history around them on one
subject clock (hours from the baseline CT):

==============  ===============  ======================================
event           effective        available
==============  ===============  ======================================
``lab0``        -48 h            -47 h (a creatinine, before the scan)
``ct0``         0 (``tp0``)      1 h
``report0_v1``  0                4 h (preliminary read)
``report0_v2``  0                48 h (amended; supersedes v1)
``ct1``         day 90 (``tp1``) day 90 + 1 h
``response1``   day 90           day 91 (lesion assessment at ``tp1``)
``recist1``     day 90           day 91 (overall response)
``seg0``        day 92           day 92 (the lesion masks drawn)
``recist2``     day 200          day 201 (some subjects)
==============  ===============  ======================================

The task asks, at a cutoff, whether the disease progresses (RECIST PD) within
180 days.  Rows sit at **24 h** --- the preliminary report is known, its
amendment is not --- and at **day 95**, after the follow-up.  Strict
selection decides what each row may see; the target is read from the full
history, after the cutoff.

Run it::

    python docs/examples/clinical_longitudinal.py [outdir]

It writes the cohort, reopens and validates every file, preflights the task,
and prints one collated training batch.  ``tests/integrations/
test_clinical_training.py`` runs it and checks the batch.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.clinical import (
    ASSESSMENT_SYSTEM,
    DAY,
    HOUR,
    LESION_PRESENCE,
    Clock,
    Document,
    Event,
    Link,
)
from medh5.labels import LabelClass, LabelSet
from medh5.task import Slot, SourceRef, Target, TaskManifest
from medh5.validate import validate_file

SHAPE = (12, 24, 24)
LESION = 3
RECIST = "org.example.recist"
REPORT_V1 = (
    "CT chest, baseline. A 14 mm nodule in the right lower lobe. Preliminary read."
)
REPORT_V2 = (
    "CT chest, baseline. A 14 mm nodule in the right lower lobe; no "
    "lymphadenopathy. Final read, amended."
)


@dataclass(frozen=True)
class Subject:
    """What differs between the example's subjects."""

    partition: str
    mr: bool
    """An MR at follow-up: the optional slot is filled only then."""
    recist1: str
    """Overall response at day 90."""
    recist2: str | None
    """At day 200; ``None`` leaves a day-95 row without follow-up (censored)."""
    annotated_tp1: bool = True
    """Whether anyone drew the follow-up lesion mask."""


COHORT: dict[str, Subject] = {
    "P-01": Subject("train", mr=True, recist1="PD", recist2=None),
    "P-02": Subject("train", mr=False, recist1="SD", recist2="PD"),
    "P-03": Subject("train", mr=True, recist1="SD", recist2="SD"),
    "P-04": Subject("train", mr=False, recist1="PR", recist2=None, annotated_tp1=False),
    "P-05": Subject("val", mr=True, recist1="CR", recist2="SD"),
    "P-06": Subject("val", mr=False, recist1="PD", recist2="PD"),
}


def label_set() -> LabelSet:
    return LabelSet(
        "example-clinical-v1",
        version="1.0.0",
        classes=[LabelClass(LESION, "lesion", "Lesion")],
    )


def lesion_mask(center: Sequence[int], radius: int) -> Any:
    z, y, x = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    cz, cy, cx = center
    return (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2


def ct(rng: np.random.Generator, lesion: Any | None) -> Any:
    volume = rng.normal(40.0, 12.0, SHAPE)
    if lesion is not None:
        volume[lesion] += 80.0
    return np.rint(volume).astype(np.int16)


def point(
    event_id: str,
    kind: str,
    effective_us: int,
    available_us: int,
    status: str = "final",
    record_id: str | None = None,
    **fields: Any,
) -> Event:
    """A point event known exactly, imported by ``act_import``."""
    return Event(
        event_id,
        record_id or event_id,
        kind,
        "point",
        status,
        effective_start_us=effective_us,
        available_us=available_us,
        prov="act_import",
        **fields,
    )


def response(event_id: str, day: int, value: str) -> Event:
    """An overall response read on ``day``, available the day after."""
    return point(
        event_id,
        "assessment",
        day * DAY,
        (day + 1) * DAY,
        code_system=RECIST,
        code="overall_response",
        value_text=value,
    )


def clinical_history(w: Any, subject: Subject, *, resolved: bool) -> None:
    """The subject's clinical records, added to the writer ``w``."""
    w.set_clock(Clock.relative("clock", "acquisition start of the baseline CT (tp0)"))
    w.add_event(
        point(
            "lab0",
            "observation",
            -48 * HOUR,
            -47 * HOUR,
            code_system="http://loinc.org",
            code="2160-0",
            value_num=1.1,
            unit="mg/dL",
        )
    )
    w.add_event(point("ct0", "imaging", 0, HOUR, timepoint_id="tp0"))
    w.add_event(point("report0_v1", "document", 0, 4 * HOUR, "preliminary", "report0"))
    w.add_event(point("report0_v2", "document", 0, 48 * HOUR, "amended", "report0"))
    w.add_event(point("ct1", "imaging", 90 * DAY, 90 * DAY + HOUR, timepoint_id="tp1"))
    if subject.mr:
        w.add_event(
            point("mr1", "imaging", 90 * DAY, 90 * DAY + 2 * HOUR, timepoint_id="tp1")
        )
    w.add_event(
        point(
            "response1",
            "assessment",
            90 * DAY,
            91 * DAY,
            code_system=ASSESSMENT_SYSTEM,
            code=LESION_PRESENCE,
            value_text="resolved" if resolved else "present",
            timepoint_id="tp1",
        )
    )
    w.add_event(response("recist1", 90, subject.recist1))
    if subject.recist2 is not None:
        w.add_event(response("recist2", 200, subject.recist2))
    # The lesion masks were drawn on day 92, after the follow-up read: before
    # then no row may use them as an input, nor centre a crop on them.
    w.add_event(point("seg0", "procedure", 92 * DAY, 92 * DAY, "completed"))
    for version in ("v1", "v2"):
        text = REPORT_V1 if version == "v1" else REPORT_V2
        w.add_document(
            Document(f"report0_text_{version}", text, language="en", source_type="radiology")
        )
    links = [
        Link.between(("event", "ct0"), "describes", ("image", "CT_tp0")),
        Link.between(("event", "ct1"), "describes", ("image", "CT_tp1")),
        Link.between(("event", "report0_v1"), "describes", ("document", "report0_text_v1")),
        Link.between(("event", "report0_v2"), "describes", ("document", "report0_text_v2")),
        Link.between(("event", "report0_v2"), "supersedes", ("event", "report0_v1")),
        Link.between(
            ("event", "response1"),
            "assesses",
            ("instance", "1"),
            asserted_by="response1",
            target_annotation_id="lesions_tp0",
        ),
        Link.between(
            ("event", "seg0"), "describes", ("annotation", "lesions_tp0"), asserted_by="seg0"
        ),
    ]
    if subject.annotated_tp1:
        links.append(
            Link.between(
                ("event", "seg0"), "describes", ("annotation", "lesions_tp1"), asserted_by="seg0"
            )
        )
    if subject.mr:
        links.append(Link.between(("event", "mr1"), "describes", ("image", "MR_tp1")))
    for link in links:
        w.add_link(link)


def write_subject(path: Path, subject_id: str, subject: Subject, *, seed: int) -> str:
    """One subject's file; returns its ``content_id``."""
    rng = np.random.default_rng(seed)
    resolved = subject.recist1 in ("CR", "PR")
    # Off-centre, so a crop centred on the lesion differs from the grid centre.
    before = lesion_mask((6, 8, 15), 3)
    after = None if resolved else lesion_mask((6, 9, 15), 3)
    with medh5.create(path, sample_id=subject_id, subject_id=subject_id) as w:
        w.label_set(label_set())
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow-up", days_from_baseline=90)
        for tp, lesion in (("tp0", before), ("tp1", after)):
            w.add_grid(
                f"ct_{tp}",
                shape=SHAPE,
                spacing=(2.0, 0.8, 0.8),
                timepoint=tp,
                frame_uid=f"{subject_id}-{tp}",
            )
            w.add_image(
                f"CT_{tp}",
                ct(rng, lesion),
                grid=f"ct_{tp}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        if subject.mr:
            w.add_grid(
                "mr_tp1",
                shape=(8, 16, 16),
                spacing=(3.0, 1.2, 1.2),
                timepoint="tp1",
                frame_uid=f"{subject_id}-tp1",
            )
            mr = rng.normal(300.0, 40.0, (8, 16, 16)).astype(np.float32)
            w.add_image("MR_tp1", mr, grid="mr_tp1", modality="MR")
        w.add_segmentation(
            "lesions_tp0",
            grid="ct_tp0",
            instances=[InstanceInput(LESION, 1, mask=before)],
            annotated_classes=[LESION],
        )
        if subject.annotated_tp1:
            w.add_segmentation(
                "lesions_tp1",
                grid="ct_tp1",
                instances=[] if after is None else [InstanceInput(LESION, 1, mask=after)],
                annotated_classes=[LESION],
            )
        importer = w.software("example-clinical-import", "1.0")
        w.activity("import", agent=importer, activity_id="act_import")
        clinical_history(w, subject, resolved=resolved)
        content_id = w.commit()
    assert content_id is not None
    return content_id


def write_cohort(out: Path) -> dict[str, Path]:
    """Every subject of :data:`COHORT`, one file each."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for seed, (subject_id, subject) in enumerate(sorted(COHORT.items())):
        paths[subject_id] = out / f"{subject_id}.medh5"
        write_subject(paths[subject_id], subject_id, subject, seed=seed)
    return paths


def progression_task(out: Path, paths: dict[str, Path]) -> TaskManifest:
    """Progression within 180 days, asked at 24 h and at day 95."""
    task = TaskManifest.new(
        "progression-180d",
        "1",
        identity_namespace="example-site",
        description="Progression (RECIST PD) within 180 days of the cutoff",
        slots=[
            Slot(
                "ct",
                "CT",
                required=True,
                patch=(8, 16, 16),
                roi="eligible_instances",
                classes=(LESION,),
            ),
            Slot("mr", "MR", patch=(8, 16, 16)),
        ],
        target=Target(
            "progression",
            "1",
            RECIST,
            "overall_response",
            positive=("PD",),
            negative=("SD", "PR", "CR"),
            horizon_us=180 * DAY,
            min_follow_up_us=60 * DAY,
            kind="assessment",
        ),
        split=("fold-0", ["train", "val"]),
        base=out,
    )
    for subject_id, path in sorted(paths.items()):
        task.add_subject(
            subject_id,
            [SourceRef.pin(path, uri=path.name, source_id=subject_id)],
            partition=COHORT[subject_id].partition,
        )
        task.add_row(f"{subject_id}@24h", subject_id, 24 * HOUR)
        task.add_row(f"{subject_id}@d95", subject_id, 95 * DAY)
    return task


def main(out: Path) -> dict[str, Any]:
    """Write, reopen, validate, preflight, and load one training batch."""
    from torch.utils.data import DataLoader

    from medh5.torch.clinical import ClinicalTaskDataset, collate_clinical

    paths = write_cohort(out)
    for path in paths.values():
        with medh5.open(path) as s:
            assert s.version == "1.1" and "clinical" in s.profiles, s.profiles
            assert s.clinical is not None
            known = s.clinical.select(24 * HOUR).event_ids
            # At 24 h the preliminary report is known; its amendment, the
            # follow-up CT and the response are not.
            assert "report0_v1" in known and "report0_v2" not in known, known
            assert "ct1" not in known and "response1" not in known, known
        report = validate_file(path, level="integrity")
        assert report.ok, [str(d) for d in report.errors]

    task = progression_task(out, paths)
    task.save(out / "progression.task.json")
    preflight = task.preflight()
    assert preflight.ok, [str(f) for f in preflight.findings]

    train = ClinicalTaskDataset(task, partition="train", preflight=preflight)
    loader = DataLoader(train, batch_size=8, shuffle=False, collate_fn=collate_clinical)
    batch = next(iter(loader))
    return {
        "preflight": preflight.counts,
        "train rows": [r.row_id for r in train.rows],
        "batch rows": [m["row_id"] for m in batch["meta"]],
        "ct visit": [m["visits"]["ct"]["timepoint"] for m in batch["meta"]],
        "ct roi": [m["visits"]["ct"]["roi"] for m in batch["meta"]],
        "mr present": batch["present"]["mr"].tolist(),
        "events": batch["events"]["length"].tolist(),
        "event batch": tuple(batch["events"]["concept"].shape),
        "target": batch["target"]["value"].tolist(),
        "observed": batch["target"]["observed"].tolist(),
        "coverage": batch["annotated"]["ct"][:, 0].tolist(),
        "ct batch": tuple(batch["images"]["ct"].shape),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for key, value in main(target).items():
        print(f"{key:>11}: {value}")
    print(f"written under {target}")
