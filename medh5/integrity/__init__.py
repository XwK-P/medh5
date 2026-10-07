"""Content addressing and verification (spec §13)."""

from __future__ import annotations

from medh5.integrity.digest import (
    DEFAULT_ALGO,
    DIGEST_ALGOS,
    array_digest,
    attrs_digest,
    canonical_attrs,
    collect_digests,
    compute_content_id,
    dataset_digest,
    digest_bytes,
    group_digest,
    parse_digest,
    relative_path,
)
from medh5.integrity.repair import Diagnosis, Repair, diagnose, fix, fix_paths
from medh5.integrity.verify import (
    VerifyResult,
    stale_index_entries,
    verify_object,
    verify_root,
)

__all__ = [
    "DEFAULT_ALGO",
    "DIGEST_ALGOS",
    "Diagnosis",
    "Repair",
    "VerifyResult",
    "array_digest",
    "attrs_digest",
    "canonical_attrs",
    "collect_digests",
    "compute_content_id",
    "dataset_digest",
    "diagnose",
    "digest_bytes",
    "fix",
    "fix_paths",
    "group_digest",
    "parse_digest",
    "relative_path",
    "stale_index_entries",
    "verify_object",
    "verify_root",
]
