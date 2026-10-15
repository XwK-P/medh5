"""The conformance corpus (spec §15).

Every case is a file plus the exact set of diagnostic codes a conforming
validator must emit for it.  Valid cases prove the format is writable; invalid
cases --- at least one per error code, built by mutating a valid file --- prove
the validator actually catches what the spec says it must.

The corpus is the contract a third-party implementation is measured against.
``build_corpus`` writes the files and an ``expected.json`` beside them;
``run_corpus`` checks *this* validator against it; ``publish`` writes the whole
distributable suite (cases, codes, schemas, the task-and-cache fixtures,
checksums, instructions) and ``score`` measures an implementation that is not
this one --- in any language --- from the codes it reports back.

The cases and their builders are the format engine's
(``crates/medh5/src/conformance/build.rs``): valid cases are written by the
writer, invalid ones by mutating a valid file, so the corpus holds the same
files whichever frontend builds it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medh5 import _core
from medh5.validate import Level

SEED: int = _core.CONFORMANCE_SEED
SCHEMA: str = _core.CONFORMANCE_SCHEMA
CHECKSUMS: str = _core.CONFORMANCE_CHECKSUMS


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
    """``.medh5c`` for a collection case (§2.2); the corpus runner honours it."""
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
        """Whether the file conforms: it expects no error."""
        return bool(self.to_json()["valid"])

    def build(self, path: str | os.PathLike[str]) -> None:
        """Write this case's file to *path*."""
        _core.conformance_build_case(self.name, os.fspath(path))

    def to_json(self) -> dict[str, Any]:
        """The case's entry in ``expected.json``."""
        found: dict[str, Any] = _core.conformance_case_json(self)
        return found


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
        """Whether the validator reported exactly the expected codes."""
        return bool(self.to_json()["ok"])

    def to_json(self) -> dict[str, Any]:
        """The result as ``conformance run --json`` prints it."""
        found: dict[str, Any] = _core.conformance_result_json(self)
        return found


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


# -- the distributable suite ------------------------------------------------------


def publish(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> Path:
    """Write the whole suite into *outdir* --- the cases, ``expected.json``, the
    code table, the JSON Schemas (the sample document, the clinical profile,
    the task and cache contract), the task-and-cache fixtures under
    ``companion/`` when the clinical cases they read are among the cases, a
    checksum file and instructions, everything a third-party implementation
    needs; returns *outdir*."""
    return Path(
        _core.conformance_publish(
            os.fspath(outdir), names=None if names is None else list(names)
        )
    )


def check_checksums(root: str | os.PathLike[str]) -> tuple[str, ...]:
    """Files whose checksum does not match ``SHA256SUMS`` (empty when intact)."""
    return tuple(_core.conformance_check_checksums(os.fspath(root)))


def load_manifest(root: str | os.PathLike[str]) -> dict[str, Any]:
    """The suite's ``expected.json``."""
    found: dict[str, Any] = _core.conformance_load_manifest(os.fspath(root))
    return found


def score(
    root: str | os.PathLike[str], submitted: Sequence[dict[str, Any]]
) -> list[CaseResult]:
    """Score a foreign validator's results against the published expectations.

    A case with no submitted result is a failure, not a skip: silence about a
    file you were given is the same as failing to diagnose it.
    """
    return [
        CaseResult.from_json(doc)
        for doc in _core.conformance_score(os.fspath(root), list(submitted))
    ]


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    """The ``--json`` summary of a run or a score."""
    found: dict[str, Any] = _core.conformance_summary(list(results))
    return found


__all__ = [
    "CASES",
    "CHECKSUMS",
    "SCHEMA",
    "SEED",
    "Case",
    "CaseResult",
    "build_corpus",
    "case_by_name",
    "check_checksums",
    "load_manifest",
    "publish",
    "run_corpus",
    "score",
    "summarize",
]
