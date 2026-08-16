"""Content addressing and verification (spec §13)."""

from __future__ import annotations

from medh5.integrity.digest import (
    DEFAULT_ALGO,
    DIGEST_ALGOS,
    attrs_digest,
    canonical_attrs,
    compute_content_id,
    dataset_digest,
    digest_bytes,
    group_digest,
    parse_digest,
    relative_path,
    stamp_digests,
)
from medh5.integrity.verify import (
    VerifyResult,
    stale_index_entries,
    verify_object,
    verify_root,
)

__all__ = [
    "DEFAULT_ALGO",
    "DIGEST_ALGOS",
    "VerifyResult",
    "attrs_digest",
    "canonical_attrs",
    "compute_content_id",
    "dataset_digest",
    "digest_bytes",
    "group_digest",
    "parse_digest",
    "relative_path",
    "stale_index_entries",
    "stamp_digests",
    "verify_object",
    "verify_root",
]
