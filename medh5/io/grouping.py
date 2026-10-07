"""Grouping study-scoped sources into subject-scoped samples (spec §3.7).

A MEDH5 sample is one **subject** at one or more timepoints; DICOM trees, 0.x
files and nnU-Net datasets are all organised by *study* or by *case*.  Bridging
that is the one genuinely lossy step in importing, so the rules are explicit:

* Identity comes from a **declared key** --- ``PatientID``, a 0.x
  ``extra.patient_id``, an explicit mapping.  Never from a filename, a date, or
  an accession number: those correlate with identity often enough to look like
  they work and not often enough to be right, and a wrong merge puts two
  patients in one sample, which §2.2 forbids outright.
* When identity cannot be established the converter **falls back to one sample
  per study**, names the affected inputs, and records the fallback.  A file that
  is one visit of a patient is still a valid sample; a file that silently merges
  two patients is not.
* A declared key the source **contradicts** is not established either.  Two
  studies under one ``PatientID`` whose birth dates or sexes differ are two
  people as far as the evidence goes --- an anonymiser that writes a constant
  ``PatientID`` makes this ordinary --- so they fall back to one sample per study,
  recorded as a guess with the values that disagreed.
* Timepoint **order** comes from a date when there is one.  Where there is not,
  the order is a guess and is reported as such --- ordering by mtime is a
  plausible heuristic and an indefensible ground truth.
* Instance correspondence across merged studies is **never** inferred (§7.4).
  Each study's objects keep their own ids; asserting that lesion 2 at baseline
  is lesion 2 at follow-up would fabricate exactly the tracking the format
  exists to record honestly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core
from medh5.io.report import ConversionReport

FALLBACK_PREFIX: str = _core.IO_FALLBACK_PREFIX


@dataclass(frozen=True, slots=True)
class Occasion:
    """One study/visit of one subject, before it becomes a timepoint."""

    key: str
    """The source's own identifier --- a StudyInstanceUID, a file path."""
    subject_id: str | None = None
    date: str | None = None
    order_hint: float | None = None
    """A last-resort ordering value (an mtime); using it is reported as a guess."""
    payload: Any = None
    demographics: Mapping[str, str] = field(default_factory=dict)
    """Facts about the person that no visit can change --- ``PatientBirthDate``,
    ``PatientSex`` --- where the source states them.  Two occasions under one
    subject key that disagree here are not one person (see
    :func:`group_by_subject`)."""

    def __repr__(self) -> str:
        return f"Occasion({self.key!r}, subject={self.subject_id!r}, {self.date})"


@dataclass(slots=True)
class SubjectGroup:
    """Occasions belonging to one subject, in timepoint order."""

    subject_id: str
    occasions: list[Occasion] = field(default_factory=list)
    ordered_by: str = "date"
    """``date``, ``given`` or ``order_hint`` --- the last is a guess."""

    @property
    def is_longitudinal(self) -> bool:
        return len(self.occasions) > 1

    def timepoint_ids(self) -> list[str]:
        return [f"tp{i}" for i in range(len(self.occasions))]

    def days_from_baseline(self) -> list[int | None]:
        """Intervals in days, or ``None`` where a date is missing."""
        found: list[int | None] = _core.io_days_from_baseline(
            [o.date for o in self.occasions]
        )
        return found

    def to_json(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "ordered_by": self.ordered_by,
            "occasions": [o.key for o in self.occasions],
            "days_from_baseline": self.days_from_baseline(),
        }

    def __len__(self) -> int:
        return len(self.occasions)

    def __repr__(self) -> str:
        return f"SubjectGroup({self.subject_id!r}, {len(self)} occasions)"


def group_by_subject(
    occasions: Iterable[Occasion],
    *,
    mode: str = "subject",
    report: ConversionReport | None = None,
) -> list[SubjectGroup]:
    """Group occasions into subjects (``mode="subject"``) or leave them apart.

    ``mode="study"`` produces one group per occasion, which is the honest
    default for sources with no reliable subject key.  A subject key the
    sources contradict (two birth dates, two sexes) groups nothing: each of its
    studies becomes its own subject, recorded as a guess with the values that
    disagreed.  Order within a subject comes from dates, else from order
    hints (a guess), else is kept as given (a guess).
    """
    entries = list(occasions)
    groups, notes = _core.io_group_by_subject(entries, mode=mode)
    if report is not None:
        report._extend(notes)
    return [
        SubjectGroup(
            subject_id=subject_id,
            occasions=[entries[i] for i in positions],
            ordered_by=ordered_by,
        )
        for subject_id, positions, ordered_by in groups
    ]


def contradictions(occasions: Sequence[Occasion]) -> dict[str, list[str]]:
    """Demographics on which *occasions* disagree: ``{field: [values]}``.

    Only stated values count.  A study that omits a birth date contradicts
    nothing; one that states a different one contradicts the rest.
    """
    found: dict[str, list[str]] = _core.io_contradictions(list(occasions))
    return found


def output_name(group: SubjectGroup, used: set[str], *, safe: Any = None) -> str:
    """A unique filename stem for one group; it is added to *used*.

    The subject key is the name.  In ``study`` mode several groups can share a
    subject --- that is the point of the mode --- so a collision falls back to
    the occasion's own key, and then to a counter.  Naming files after the
    subject alone would silently overwrite every visit but the last.
    """
    return str(
        _core.io_output_name(
            group.subject_id, [o.key for o in group.occasions], used, safe
        )
    )


def note_instance_ids(group: SubjectGroup, log: ConversionReport) -> None:
    """Record that objects were *not* joined across merged studies (§7.4)."""
    log._extend(_core.io_note_instance_ids(group.subject_id, len(group.occasions)))


def build_occasions(
    items: Sequence[Any],
    *,
    key: Callable[[Any], str],
    subject: Callable[[Any], str | None],
    date: Callable[[Any], str | None] | None = None,
    order_hint: Callable[[Any], float | None] | None = None,
) -> list[Occasion]:
    """Adapt any sequence of source objects into occasions."""
    return [
        Occasion(
            key=key(item),
            subject_id=subject(item),
            date=None if date is None else date(item),
            order_hint=None if order_hint is None else order_hint(item),
            payload=item,
        )
        for item in items
    ]


__all__ = [
    "FALLBACK_PREFIX",
    "Occasion",
    "SubjectGroup",
    "build_occasions",
    "contradictions",
    "group_by_subject",
    "note_instance_ids",
    "output_name",
]
