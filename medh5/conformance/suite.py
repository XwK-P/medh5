"""The distributable conformance suite: publish it, check it, score against it.

``publish`` writes the cases, ``expected.json``, the code table, the
sample-document schema, a checksum file and instructions --- everything a
third-party implementation, in any language, needs.  ``score`` measures such an
implementation from the codes it reports back.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from medh5 import _core
from medh5.conformance.corpus import CaseResult

SCHEMA: str = _core.CONFORMANCE_SCHEMA
CHECKSUMS: str = _core.CONFORMANCE_CHECKSUMS


def publish(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> Path:
    """Write the whole suite into *outdir*; returns it."""
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
    failures = [r for r in results if not r.ok]
    return {
        "cases": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "failures": [r.to_json() for r in failures],
    }


__all__ = [
    "CHECKSUMS",
    "SCHEMA",
    "check_checksums",
    "load_manifest",
    "publish",
    "score",
    "summarize",
]
