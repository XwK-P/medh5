"""What a conversion decided, and where it had to guess (plan §6).

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

SEVERITIES = ("info", "decision", "guess", "warning")
"""``decision`` was determined by the data; ``guess`` was not."""


@dataclass(frozen=True, slots=True)
class Note:
    """One thing a conversion did that the source did not fully determine."""

    kind: str
    message: str
    severity: str = "info"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"{self.severity.upper():8s} {self.kind}: {self.message}"


@dataclass(slots=True)
class ConversionReport:
    """Every note from one conversion, plus what it produced."""

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
        return not self.warnings

    def of_kind(self, kind: str) -> tuple[Note, ...]:
        return tuple(n for n in self.notes if n.kind == kind)

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "converter": self.converter,
            "outputs": list(self.outputs),
            "ok": self.ok,
            "counts": {
                severity: sum(1 for n in self.notes if n.severity == severity)
                for severity in SEVERITIES
            },
            "notes": [n.to_json() for n in self.notes],
        }

    def format(self, *, verbose: bool = False) -> str:
        head = (
            f"{self.converter}: {len(self.outputs)} output(s), "
            f"{len(self.guesses)} guess(es), {len(self.warnings)} warning(s)"
        )
        lines = [head]
        for note in self.notes:
            if verbose or note.severity in ("guess", "warning"):
                lines.append("  " + str(note))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


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
