"""Review / curation helpers for ``.medh5`` files.

Tracks annotation status with an audit trail stored under the ``extra``
metadata attribute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

from medh5.exceptions import MEDH5FileError, MEDH5ValidationError
from medh5.integrity import _CHECKSUM_ATTR, write_checksum
from medh5.meta import read_meta

_REVIEW_STATUSES: tuple[str, ...] = ("pending", "reviewed", "flagged", "rejected")

_SUFFIX = ".medh5"


@dataclass
class ReviewStatus:
    """Lightweight view of the curation/review state stored under ``extra``."""

    status: str = "pending"
    annotator: str | None = None
    timestamp: str | None = None
    notes: str | None = None
    history: list[dict[str, Any]] | None = None


def _validate_suffix(path: Path) -> None:
    if path.suffix != _SUFFIX:
        raise MEDH5ValidationError(
            f"File must have '{_SUFFIX}' extension, got '{path.suffix}'"
        )


def set_review_status(
    path: str | Path,
    *,
    status: str,
    annotator: str | None = None,
    notes: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Record a review/curation status under ``extra["review"]``.

    The previous state is appended to ``extra["review"]["history"]``
    so the audit trail is preserved across calls.

    Parameters
    ----------
    path : str or Path
        ``.medh5`` file to update.
    status : str
        One of ``"pending"``, ``"reviewed"``, ``"flagged"``, ``"rejected"``.
    annotator : str, optional
        Name of the annotator making the change.
    notes : str, optional
        Free-form notes for this status entry.
    timestamp : str, optional
        ISO-8601 timestamp. Defaults to ``datetime.now(UTC)``.

    Raises
    ------
    MEDH5ValidationError
        On unknown status.
    """
    if status not in _REVIEW_STATUSES:
        raise MEDH5ValidationError(
            f"Unknown review status '{status}'. Choose from: {list(_REVIEW_STATUSES)}"
        )
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    path = Path(path)
    _validate_suffix(path)

    try:
        with h5py.File(str(path), "a") as f:
            meta = read_meta(f)
            extra = dict(meta.extra) if meta.extra else {}
            review = dict(extra.get("review") or {})
            history = list(review.get("history") or [])
            # Snapshot the previous active state (if any) into history.
            if review.get("status") is not None:
                history.append(
                    {
                        "status": review.get("status"),
                        "annotator": review.get("annotator"),
                        "timestamp": review.get("timestamp"),
                        "notes": review.get("notes"),
                    }
                )
            review.update(
                {
                    "status": status,
                    "annotator": annotator,
                    "timestamp": timestamp,
                    "notes": notes,
                    "history": history,
                }
            )
            extra["review"] = review
            f.attrs["extra"] = json.dumps(extra)
            if _CHECKSUM_ATTR in f.attrs:
                write_checksum(f)
    except MEDH5ValidationError:
        raise
    except OSError as exc:
        raise MEDH5FileError(f"Failed to update '{path}': {exc}") from exc


def get_review_status(path: str | Path) -> ReviewStatus:
    """Return the current review state stored under ``extra["review"]``.

    Returns a default :class:`ReviewStatus` (``status="pending"``) when
    the file has no review metadata yet.
    """
    path = Path(path)
    _validate_suffix(path)

    try:
        with h5py.File(str(path), "r") as f:
            meta = read_meta(f)
    except OSError as exc:
        raise MEDH5FileError(f"Failed to read '{path}': {exc}") from exc

    if not meta.extra or "review" not in meta.extra:
        return ReviewStatus()
    review = meta.extra["review"]
    return ReviewStatus(
        status=review.get("status", "pending"),
        annotator=review.get("annotator"),
        timestamp=review.get("timestamp"),
        notes=review.get("notes"),
        history=review.get("history"),
    )
