"""Cohort-level checks: what is wrong *between* files, not inside one.

``medh5 validate`` answers "is this file legal".  Every question that matters
for training is about the cohort: do all these files mean the same thing by
class 3, was this split computed before or after the cohort last changed, is
the same subject in train and test, is a class annotated in a tenth of the
files and absent from the rest.  None of those is visible from inside a single
sample, and each of them silently corrupts a result.

Findings carry the same shape as the validator's --- a code, a severity, a
location and a message --- but the codes are cohort codes (``C1xx``) and belong
to this tool, not to the format.  A file is not non-conforming because the
cohort around it is inconsistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core
from medh5.dataset.manifest import Manifest

SEVERITIES: tuple[str, ...] = _core.CHECK_SEVERITIES

CHECK_CODES: dict[str, str] = dict(_core.CHECK_CODES)
"""Cohort codes.  Distinct from the format's E/W table on purpose (§15.2)."""


@dataclass(slots=True)
class Finding:
    code: str
    severity: str
    message: str
    where: tuple[str, ...] = ()

    def _doc(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "where": list(self.where),
        }

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_finding_json(self._doc())
        return found

    def __str__(self) -> str:
        return str(_core.dataset_finding_line(self._doc()))


@dataclass(slots=True)
class CohortReport:
    manifest_sha256: str
    samples: int
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[int, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> CohortReport:
        return cls(
            manifest_sha256=str(doc["manifest_sha256"]),
            samples=int(doc["samples"]),
            findings=[
                Finding(
                    code=f["code"],
                    severity=f["severity"],
                    message=f["message"],
                    where=tuple(f.get("where") or ()),
                )
                for f in doc.get("findings", ())
            ],
            coverage={int(k): dict(v) for k, v in (doc.get("coverage") or {}).items()},
        )

    def add(
        self, code: str, severity: str, message: str, where: Sequence[str] = ()
    ) -> Finding:
        finding = Finding(code, severity, message, tuple(where))
        self.findings.append(finding)
        return finding

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def _doc(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "samples": self.samples,
            "findings": [f._doc() for f in self.findings],
            "coverage": {str(k): v for k, v in self.coverage.items()},
        }

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_check_json(self._doc())
        return found

    def format(self) -> str:
        return str(_core.dataset_check_format(self._doc()))


def check(
    manifest: Manifest,
    *,
    set_id: str | None = None,
    deep: bool = False,
) -> CohortReport:
    """Every cohort-level check, over a manifest.

    ``deep`` re-reads each file's ``content_id`` rather than trusting size and
    mtime --- slower, and the only way to be sure a file is the one that was
    scanned.
    """
    return CohortReport.from_json(
        _core.dataset_check(manifest._doc(), set_id=set_id, deep=deep)
    )


__all__ = ["CHECK_CODES", "SEVERITIES", "CohortReport", "Finding", "check"]
