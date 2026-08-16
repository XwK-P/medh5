"""The conformance corpus (spec §15, implementation plan phase 0).

Every case is a file plus the exact set of diagnostic codes a conforming
validator must emit for it.  Valid cases prove the format is writable; invalid
cases --- one per error code, built by mutating a valid file --- prove the
validator actually catches what the spec says it must.

The corpus is the contract a third-party implementation is measured against:
``build_corpus`` writes the files and an ``expected.json`` beside them, and
``run_corpus`` checks any validator against that manifest.
"""

from __future__ import annotations

from medh5.conformance.corpus import (
    CASES,
    Case,
    CaseResult,
    build_corpus,
    case_by_name,
    run_corpus,
)

__all__ = [
    "CASES",
    "Case",
    "CaseResult",
    "build_corpus",
    "case_by_name",
    "run_corpus",
]
