"""What a conversion decided, and where it had to guess.

Importing data is where a format either preserves what a source said or quietly
substitutes something plausible.  Every converter here records the second kind
of step, because those are exactly the ones that are invisible in the output and
expensive to discover later: which encoding was chosen, which class ids were
minted, where a half-voxel convention was changed, whether timepoint order was
inferred rather than read.

The report is a first-class output, not logging.  ``medh5 convert`` and
``medh5 migrate`` write it as JSON alongside the files, so a curator can review
a cohort's conversions without re-running them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core

SEVERITIES: tuple[str, ...] = _core.IO_SEVERITIES
"""``decision`` was determined by the data; ``guess`` was not."""


@dataclass(frozen=True, slots=True)
class Note:
    """One thing a conversion did that the source did not fully determine."""

    kind: str
    message: str
    severity: str = "info"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.io_note_json(self)
        return result

    def __str__(self) -> str:
        return str(_core.io_note_line(self))


@dataclass(slots=True)
class ConversionReport:
    """Every note from one conversion, plus what it produced.

    The text and JSON forms are the engine's, so a converter's report reads
    exactly as the native ``medh5 migrate`` prints one.  Details JSON has no
    form for are written as their ``str()``, so the JSON always serialises.
    """

    source: str = ""
    converter: str = ""
    outputs: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    def add(
        self,
        kind: str,
        message: str,
        severity: str = "info",
        detail: Mapping[str, Any] | None = None,
    ) -> Note:
        """Record a note.

        *detail* is an explicit mapping rather than ``**kwargs`` because its
        keys are data, and a converter naturally wants to record one called
        ``kind`` --- which would collide with this method's own parameter.
        """
        note = Note(
            kind=kind, message=message, severity=severity, detail=dict(detail or {})
        )
        self.notes.append(note)
        return note

    def decision(
        self, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> Note:
        """Something the data determined --- auditable, but not a guess."""
        return self.add(kind, message, "decision", detail)

    def guess(
        self, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> Note:
        """Something the source did not say and the converter had to assume."""
        return self.add(kind, message, "guess", detail)

    def warn(
        self, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> Note:
        return self.add(kind, message, "warning", detail)

    @property
    def guesses(self) -> tuple[Note, ...]:
        return tuple(n for n in self.notes if n.severity == "guess")

    @property
    def warnings(self) -> tuple[Note, ...]:
        return tuple(n for n in self.notes if n.severity == "warning")

    @property
    def ok(self) -> bool:
        """Whether the conversion needed no warning.  Guesses are not failures."""
        return bool(_core.io_report_ok(self))

    def of_kind(self, kind: str) -> tuple[Note, ...]:
        return tuple(n for n in self.notes if n.kind == kind)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.io_report_json(self)
        return result

    def format(self, *, verbose: bool = False) -> str:
        return str(_core.io_report_format(self, verbose=verbose))

    def __str__(self) -> str:
        return self.format()

    def _extend(self, notes: Sequence[Mapping[str, Any]]) -> None:
        """Append notes an engine step recorded, as field mappings."""
        self.notes.extend(Note(**n) for n in notes)

    def _update(self, fields: Mapping[str, Any]) -> ConversionReport:
        """Take the engine's report, keeping this object (and its lists)."""
        self.source = fields["source"]
        self.converter = fields["converter"]
        self.outputs[:] = fields["outputs"]
        self.notes[:] = [Note(**n) for n in fields["notes"]]
        return self

    @classmethod
    def _from_fields(cls, fields: Mapping[str, Any]) -> ConversionReport:
        return cls()._update(fields)


def merge_reports(
    reports: Sequence[ConversionReport], converter: str = ""
) -> ConversionReport:
    """One report over a whole cohort."""
    out = ConversionReport(converter=converter or "batch", source="<many>")
    for report in reports:
        out.outputs.extend(report.outputs)
        out.notes.extend(report.notes)
    return out


__all__ = ["SEVERITIES", "ConversionReport", "Note", "merge_reports"]
