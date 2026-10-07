"""Quality records (spec §11.2).

Status changes are **activities**, not fields with private history: the audit
trail is the provenance graph, so a quality record says what is true now and
the graph says how it got that way.
"""

from __future__ import annotations

from medh5._core import (
    ISSUE_SEVERITY,
    QUALITY_FIELDS,
    QUALITY_STATUS,
    Agreement,
    Issue,
    QualityRecord,
    dice_agreement,
    quality_from_json,
    quality_to_json,
)

__all__ = [
    "ISSUE_SEVERITY",
    "QUALITY_FIELDS",
    "QUALITY_STATUS",
    "Agreement",
    "Issue",
    "QualityRecord",
    "dice_agreement",
    "quality_from_json",
    "quality_to_json",
]
