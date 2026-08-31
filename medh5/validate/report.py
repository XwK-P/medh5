"""The validation report model (spec §15).

A validator emits ``(code, severity, location, message)``.  Codes are stable API
--- :mod:`medh5.errors` owns the table --- and ``location`` is an HDF5 path or a
JSON pointer into ``/meta``, so a diagnostic always names the object to look at.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from medh5.errors import CODES, Severity

Level = Literal["structural", "semantic", "integrity", "strict"]

LEVELS: tuple[Level, ...] = ("structural", "semantic", "integrity", "strict")

LEVEL_ORDER = {name: i for i, name in enumerate(LEVELS)}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One finding."""

    code: str
    location: str
    message: str
    severity: Severity = "error"
    level: Level = "structural"

    @property
    def summary(self) -> str:
        return CODES[self.code].summary if self.code in CODES else ""

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
            "level": self.level,
        }

    def __str__(self) -> str:
        return f"{self.severity.upper():7s} {self.code} {self.location}: {self.message}"


@dataclass(slots=True)
class Report:
    """Everything a validation pass found."""

    path: str
    level: Level = "structural"
    profiles: tuple[str, ...] = ()
    diagnostics: list[Diagnostic] = field(default_factory=list)
    checked: dict[str, Any] = field(default_factory=dict)

    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        self.diagnostics.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    def _effective_severity(self, diagnostic: Diagnostic) -> Severity:
        """Severity *at this level*, which is not always the table's severity.

        §15.1 defines ``strict`` as the other levels "with warnings promoted to
        errors", so at ``strict`` there are no warnings.  The promotion has to
        reach the counts, not just the verdict: reporting ``FAILED ... (0 errors,
        2 warnings)`` contradicts itself, and a CI job gating on ``errors == 0``
        --- with the count right there in the JSON --- passed files the same
        payload called not-ok.  Each diagnostic keeps its measured
        ``severity``, so what the table says is still visible.
        """
        if self.level == "strict":
            return "error"
        return diagnostic.severity

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(
            d for d in self.diagnostics if self._effective_severity(d) == "error"
        )

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(
            d for d in self.diagnostics if self._effective_severity(d) == "warning"
        )

    @property
    def promoted(self) -> tuple[Diagnostic, ...]:
        """Diagnostics the table calls warnings and this level treats as errors."""
        if self.level != "strict":
            return ()
        return tuple(d for d in self.diagnostics if d.severity == "warning")

    @property
    def codes(self) -> tuple[str, ...]:
        """Every code emitted, sorted and deduplicated --- the conformance key."""
        return tuple(sorted({d.code for d in self.diagnostics}))

    @property
    def ok(self) -> bool:
        """Whether the file conforms.

        At ``strict`` a warning is a failure; at every other level it is advice.
        Both cases fall out of `errors`, which already applies the level's
        promotion, so the verdict and the counts cannot disagree.
        """
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "level": self.level,
            "profiles": list(self.profiles),
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "promoted": len(self.promoted),
            "diagnostics": [d.to_json() for d in self.diagnostics],
            "checked": self.checked,
        }

    def dumps(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json(), indent=indent)

    def format(self, *, verbose: bool = False) -> str:
        """Human-readable text, one diagnostic per line."""
        head = (
            f"{self.path}: {'OK' if self.ok else 'FAILED'} "
            f"[{self.level}] profiles={','.join(self.profiles) or '-'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )
        lines = [head]
        for diagnostic in self.diagnostics:
            lines.append("  " + str(diagnostic))
            if verbose and diagnostic.summary:
                lines.append(f"          -> {diagnostic.summary}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Report({self.path!r}, ok={self.ok}, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


def merge(reports: Sequence[Report], path: str = "<multiple>") -> Report:
    """Combine several reports, prefixing each diagnostic's location with its file."""
    out = Report(path=path, level=reports[0].level if reports else "structural")
    for report in reports:
        for diagnostic in report.diagnostics:
            out.add(
                Diagnostic(
                    code=diagnostic.code,
                    location=f"{report.path}:{diagnostic.location}",
                    message=diagnostic.message,
                    severity=diagnostic.severity,
                    level=diagnostic.level,
                )
            )
    return out


__all__ = ["LEVELS", "LEVEL_ORDER", "Diagnostic", "Level", "Report", "merge"]
