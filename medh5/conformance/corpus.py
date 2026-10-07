"""The conformance corpus: every case, how it is built, what it must report.

The cases and their builders are the format engine's
(``crates/medh5/src/conformance/build.rs``): valid cases are written by the
writer, invalid ones by mutating a valid file, so the corpus holds the same
files whichever frontend builds it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medh5 import _core
from medh5.validate.report import Level

SEED: int = _core.CONFORMANCE_SEED


@dataclass(frozen=True, slots=True)
class Case:
    """One corpus entry."""

    name: str
    description: str
    clause: str
    level: Level = "semantic"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    suffix: str = ".medh5"
    """``.medh5c`` for a collection case (§2.1); the corpus runner honours it."""
    mutated: bool = False
    """Built by editing a committed file, so its digests are deliberately stale.

    Mutation is how invalid cases are made --- the writer refuses to produce
    them --- but it leaves ``content_id`` covering the pre-mutation bytes.  The
    flag says so, rather than letting a consumer read "no expected errors" as
    "this file also verifies".
    """

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Case:
        """A case from its manifest record (``expected.json``)."""
        return cls(
            name=str(record.get("name", "")),
            description=str(record.get("description", "")),
            clause=str(record.get("clause", "")),
            level=record.get("level", "semantic"),
            errors=tuple(record.get("expect_errors") or ()),
            warnings=tuple(record.get("expect_warnings") or ()),
            suffix=str(record.get("file_suffix", ".medh5")),
            mutated=bool(record.get("mutated", False)),
        )

    @property
    def valid(self) -> bool:
        return not self.errors

    def build(self, path: str | os.PathLike[str]) -> None:
        """Write this case's file to *path*."""
        _core.conformance_build_case(self.name, os.fspath(path))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "clause": self.clause,
            "level": self.level,
            "file_suffix": self.suffix,
            "valid": self.valid,
            "mutated": self.mutated,
            "expect_errors": sorted(self.errors),
            "expect_warnings": sorted(self.warnings),
        }


@dataclass(slots=True)
class CaseResult:
    """What running one case produced."""

    case: Case
    path: str
    got_errors: tuple[str, ...] = ()
    got_warnings: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    error: str | None = None
    details: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> CaseResult:
        record = doc["case"]
        known = _BY_NAME.get(record.get("name", ""))
        return cls(
            case=known if known is not None else Case.from_record(record),
            path=doc["path"],
            got_errors=tuple(doc.get("got_errors") or ()),
            got_warnings=tuple(doc.get("got_warnings") or ()),
            missing=tuple(doc.get("missing") or ()),
            unexpected=tuple(doc.get("unexpected") or ()),
            error=doc.get("error"),
            details=list(doc.get("details") or ()),
        )

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and self.error is None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.case.name,
            "ok": self.ok,
            "expect_errors": sorted(self.case.errors),
            "got_errors": sorted(self.got_errors),
            "expect_warnings": sorted(self.case.warnings),
            "got_warnings": sorted(self.got_warnings),
            "missing": sorted(self.missing),
            "unexpected": sorted(self.unexpected),
            "error": self.error,
        }


CASES: tuple[Case, ...] = tuple(
    Case.from_record(record) for record in _core.conformance_cases()
)
_BY_NAME: dict[str, Case] = {c.name: c for c in CASES}


def case_by_name(name: str) -> Case:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown conformance case {name!r}") from None


def build_corpus(
    outdir: str | os.PathLike[str], *, names: list[str] | tuple[str, ...] | None = None
) -> Path:
    """Write every case (or those *names*) and an ``expected.json`` beside them;
    returns the manifest's path."""
    return Path(
        _core.conformance_build_corpus(
            os.fspath(outdir), names=None if names is None else list(names)
        )
    )


def run_corpus(
    outdir: str | os.PathLike[str], *, names: list[str] | tuple[str, ...] | None = None
) -> list[CaseResult]:
    """Build the corpus into *outdir* and check this validator against it."""
    return [
        CaseResult.from_json(doc)
        for doc in _core.conformance_run_corpus(
            os.fspath(outdir), names=None if names is None else list(names)
        )
    ]


__all__ = [
    "CASES",
    "SEED",
    "Case",
    "CaseResult",
    "build_corpus",
    "case_by_name",
    "run_corpus",
]
