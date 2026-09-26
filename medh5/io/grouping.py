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

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medh5.io._common import sanitize_stem
from medh5.io.report import ConversionReport

FALLBACK_PREFIX = "study"


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
        dates = [_parse_date(o.date) for o in self.occasions]
        if not dates or dates[0] is None:
            return [None] * len(dates)
        return [None if d is None else (d - dates[0]).days for d in dates]

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


def _parse_date(value: str | None) -> Any:
    """Parse a DICOM ``YYYYMMDD`` or an ISO date; ``None`` when unparseable."""
    if not value:
        return None
    from datetime import date, datetime

    text = str(value).strip()
    for form in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(form) + 2], form).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def group_by_subject(
    occasions: Iterable[Occasion],
    *,
    mode: str = "subject",
    report: ConversionReport | None = None,
) -> list[SubjectGroup]:
    """Group occasions into subjects (``mode="subject"``) or leave them apart.

    ``mode="study"`` produces one group per occasion, which is the honest
    default for sources with no reliable subject key.
    """
    if mode not in ("subject", "study"):
        raise ValueError(f"unknown grouping mode {mode!r}")
    entries = list(occasions)
    log = report
    contradicted = _contradicted(entries, log)

    def subject_of(occasion: Occasion) -> str | None:
        """The occasion's key, unless the source contradicts it."""
        if occasion.subject_id in contradicted:
            return None
        return occasion.subject_id

    if mode == "study":
        return [
            SubjectGroup(
                subject_id=subject_of(o) or f"{FALLBACK_PREFIX}:{o.key}",
                occasions=[o],
                ordered_by="given",
            )
            for o in entries
        ]

    identified = [o for o in entries if subject_of(o)]
    unidentified = [o for o in entries if not o.subject_id]
    if unidentified and log is not None:
        log.warn(
            "grouping",
            f"{len(unidentified)} input(s) carry no subject key, so each became "
            "its own sample; identity is never inferred from filenames or dates",
            {"inputs": [o.key for o in unidentified][:20]},
        )
    unidentified.extend(o for o in entries if o.subject_id and not subject_of(o))

    groups: dict[str, SubjectGroup] = {}
    for occasion in identified:
        assert occasion.subject_id is not None
        group = groups.setdefault(
            occasion.subject_id, SubjectGroup(subject_id=occasion.subject_id)
        )
        group.occasions.append(occasion)
    for occasion in unidentified:
        key = f"{FALLBACK_PREFIX}:{occasion.key}"
        groups[key] = SubjectGroup(subject_id=key, occasions=[occasion])

    out = []
    for group in groups.values():
        _order(group, log)
        out.append(group)
    return sorted(out, key=lambda g: g.subject_id)


def _contradicted(
    occasions: Sequence[Occasion], log: ConversionReport | None
) -> set[str]:
    """Subject keys the source contradicts, each recorded as a guess.

    Every study of such a key falls back to its own subject, not a majority:
    which of them belong together is exactly what the key was supposed to say
    and cannot.
    """
    by_subject: dict[str, list[Occasion]] = {}
    for occasion in occasions:
        if occasion.subject_id:
            by_subject.setdefault(occasion.subject_id, []).append(occasion)
    out: set[str] = set()
    for subject, members in by_subject.items():
        conflicts = contradictions(members)
        if not conflicts:
            continue
        out.add(subject)
        if log is not None:
            log.guess(
                "identity",
                f"subject key {subject!r} names {len(members)} studies whose "
                f"{' and '.join(sorted(conflicts))} disagree, so they are not "
                "one person on the evidence; each study was given its own "
                f"subject ({FALLBACK_PREFIX}:<study>) rather than grouped under "
                "the key",
                {
                    "subject": subject,
                    "conflicts": conflicts,
                    "occasions": [o.key for o in members],
                },
            )
    return out


def contradictions(occasions: Sequence[Occasion]) -> dict[str, list[str]]:
    """Demographics on which *occasions* disagree: ``{field: [values]}``.

    Only stated values count.  A study that omits a birth date contradicts
    nothing; one that states a different one contradicts the rest.
    """
    seen: dict[str, set[str]] = {}
    for occasion in occasions:
        for name, value in occasion.demographics.items():
            text = str(value).strip().upper()
            if text:
                seen.setdefault(name, set()).add(text)
    return {name: sorted(values) for name, values in seen.items() if len(values) > 1}


def _order(group: SubjectGroup, log: ConversionReport | None) -> None:
    """Sort a subject's occasions, saying which signal did the sorting."""
    if len(group) < 2:
        group.ordered_by = "given"
        return
    if all(_parse_date(o.date) is not None for o in group.occasions):
        group.occasions.sort(key=lambda o: _parse_date(o.date))
        group.ordered_by = "date"
        return
    if all(o.order_hint is not None for o in group.occasions):
        group.occasions.sort(key=lambda o: o.order_hint or 0.0)
        group.ordered_by = "order_hint"
        if log is not None:
            log.guess(
                "timepoint_order",
                f"subject {group.subject_id!r} has no study dates; its "
                f"{len(group)} visits were ordered by a file timestamp, which is "
                "a heuristic and not evidence",
                {
                    "subject": group.subject_id,
                    "occasions": [o.key for o in group.occasions],
                },
            )
        return
    group.ordered_by = "given"
    if log is not None:
        log.guess(
            "timepoint_order",
            f"subject {group.subject_id!r} has neither dates nor timestamps; its "
            "visits kept the order they were supplied in",
            {"subject": group.subject_id},
        )


def output_name(group: SubjectGroup, used: set[str], *, safe: Any = None) -> str:
    """A unique filename stem for one group.

    The subject key is the name.  In ``study`` mode several groups can share a
    subject --- that is the point of the mode --- so a collision falls back to
    the occasion's own key, and then to a counter.  Naming files after the
    subject alone would silently overwrite every visit but the last.
    """
    clean = safe or _default_safe
    base = clean(group.subject_id)
    if base.startswith("study") and group.occasions:
        base = clean(_key_stem(group.occasions[0].key)) or base
    if base not in used:
        used.add(base)
        return base
    if group.occasions:
        candidate = f"{base}_{clean(_key_stem(group.occasions[0].key))}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    used.add(f"{base}_{index}")
    return f"{base}_{index}"


_UID = re.compile(r"^\d+(\.\d+)+$")


def _key_stem(key: str) -> str:
    """An occasion key as a file-name stem: a path's stem, a UID whole.

    ``Path(uid).stem`` reads a StudyInstanceUID's last component as a file
    extension, so every study of one root named its file after the same
    truncated UID.
    """
    return key if _UID.match(key) else Path(key).stem


def _default_safe(text: str) -> str:
    return sanitize_stem(text, limit=120)


def note_instance_ids(group: SubjectGroup, log: ConversionReport) -> None:
    """Record that objects were *not* joined across merged studies (§7.4)."""
    if group.is_longitudinal:
        log.decision(
            "instance_ids",
            f"subject {group.subject_id!r} merged {len(group)} studies; each "
            "study's objects kept independent instance ids, because asserting "
            "correspondence across visits would fabricate tracking ground truth",
            {"subject": group.subject_id, "occasions": len(group)},
        )


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
