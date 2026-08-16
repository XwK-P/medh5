"""The conformance corpus (spec §15, implementation plan phase 0).

Every case is a file plus the exact set of diagnostic codes a conforming
validator must emit for it.  Valid cases prove the format is writable; invalid
cases --- one per error code, built by mutating a valid file --- prove the
validator actually catches what the spec says it must.

The corpus is the contract a third-party implementation is measured against.
``build_corpus`` writes the files and an ``expected.json`` beside them;
``run_corpus`` checks *this* validator against it; ``publish`` writes the whole
distributable suite (cases, codes, schema, checksums, instructions) and ``score``
measures an implementation that is not this one --- in any language --- from the
codes it reports back.
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
from medh5.conformance.suite import (
    check_checksums,
    load_manifest,
    publish,
    score,
    summarize,
)

__all__ = [
    "CASES",
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
