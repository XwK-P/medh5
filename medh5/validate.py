"""Validation (spec §15).

Four levels, each a superset of the last:

``structural``
    layout, required attributes, dtypes, shapes, identifier syntax, JSON Schema
``semantic``
    cross-references resolve, geometry consistency, class ids in the label set,
    encoding invariants, profile requirements
``integrity``
    per-object digests, ``content_id``, index ``source_digest`` currency
``strict``
    all of the above, with warnings promoted to failures

A validation pass never raises on a bad file --- it reports.  Curation needs to
see everything wrong with a file at once, and a validator that stops at the
first problem turns one review cycle into ten.

A validator emits ``(code, severity, location, message)``.  Codes are stable API
--- :mod:`medh5.errors` exposes the table --- and ``location`` is an HDF5 path
or a JSON pointer into ``/meta``, so a diagnostic always names the object to
look at.

The rules are the format engine's (``medh5`` crate, ``validate`` module): the
command line, this package and the Rust SDK run the same code, so a file one
of them accepts the others accept.  The classes here are the report it
returns, in the shape 1.x returned it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from medh5 import _core
from medh5.errors import CODES, Severity

Level = Literal["structural", "semantic", "integrity", "strict"]
"""A validation level (§15.1); each includes the checks of the levels before it."""

LEVELS: tuple[Level, ...] = _core.LEVELS

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

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Diagnostic:
        return cls(
            code=doc["code"],
            location=doc["location"],
            message=doc["message"],
            severity=doc.get("severity", "error"),
            level=doc.get("level", "structural"),
        )

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

    def _positions(self) -> tuple[list[int], list[int], list[int]]:
        """Where the errors, the warnings and the promoted warnings are in
        `diagnostics`, at this level --- the engine's judgement.

        §15.1 defines ``strict`` as the other levels "with warnings promoted to
        errors", so at ``strict`` there are no warnings.  The promotion reaches
        the counts, not just the verdict: reporting ``FAILED ... (0 errors, 2
        warnings)`` contradicts itself.  Each diagnostic keeps its measured
        ``severity``, so what the table says is still visible.
        """
        return _core.validate_report_positions(self)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(self.diagnostics[i] for i in self._positions()[0])

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(self.diagnostics[i] for i in self._positions()[1])

    @property
    def promoted(self) -> tuple[Diagnostic, ...]:
        """Diagnostics the table calls warnings and this level treats as errors."""
        return tuple(self.diagnostics[i] for i in self._positions()[2])

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
        found: dict[str, Any] = _core.validate_report_json(self)
        return found

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Report:
        """A report from its JSON form (what the engine returns)."""
        return cls(
            path=doc["path"],
            level=doc.get("level", "structural"),
            profiles=tuple(doc.get("profiles") or ()),
            diagnostics=[Diagnostic.from_json(d) for d in doc.get("diagnostics", ())],
            checked=dict(doc.get("checked") or {}),
        )

    def dumps(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json(), indent=indent)

    def format(self, *, verbose: bool = False) -> str:
        """Human-readable text, one diagnostic per line."""
        return str(_core.validate_report_format(self, verbose=verbose))

    def __repr__(self) -> str:
        return (
            f"Report({self.path!r}, ok={self.ok}, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


def merge(reports: Sequence[Report], path: str = "<multiple>") -> Report:
    """Combine several reports, prefixing each diagnostic's location with its file."""
    return Report.from_json(_core.validate_merge(list(reports), path))


def validate_file(
    path: str | os.PathLike[str],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
    codes_revision: int | None = None,
) -> Report:
    """Validate one ``.medh5`` sample or ``.medh5c`` collection file.

    The file is held to the profiles it declares and to *profiles* besides: a
    requirement adds to what a file claims and never waives a claim.  A file
    that cannot be opened is reported (E001), not raised; an unknown *level*
    or profile raises ``ValueError``.

    *codes_revision* reports only the codes that revision of the code table
    allocated (§15.2), so a ``strict`` verdict does not move when a later
    revision adds a warning; a revision the table does not have raises
    ``ValueError``.
    """
    return Report.from_json(
        _core.validate_file(
            os.fspath(path),
            level=level,
            profiles=profiles,
            codes_revision=codes_revision,
        )
    )


def validate_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
    codes_revision: int | None = None,
) -> list[Report]:
    """Validate many files, one report each, as :func:`validate_file` does."""
    return [
        Report.from_json(doc)
        for doc in _core.validate_paths(
            [os.fspath(p) for p in paths],
            level=level,
            profiles=profiles,
            codes_revision=codes_revision,
        )
    ]


def validate_root(
    sample: Any,
    *,
    path: str | None = None,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
    errors_only: bool = False,
) -> Report:
    """Validate an open :class:`~medh5.sample.Sample` --- a file, or a member
    of a collection --- held to the profiles it declares and to *profiles*
    besides.

    ``errors_only`` skips the warning-only checks that read bulk data, which
    is what the writer's commit gate asks for.
    """
    return Report.from_json(
        _core.validate_root(
            sample,
            path=path,
            level=level,
            profiles=profiles,
            errors_only=errors_only,
        )
    )


def rules_for(level: Level) -> tuple[str, ...]:
    """The rules a pass at *level* runs, in order.  An unknown *level* is a
    ``ValueError``, as it is to the validators."""
    return tuple(_core.rules_for(level))


__all__ = [
    "LEVELS",
    "LEVEL_ORDER",
    "Diagnostic",
    "Level",
    "Report",
    "merge",
    "rules_for",
    "validate_file",
    "validate_paths",
    "validate_root",
]
