"""Review / curation helpers for ``.medh5`` files.

Tracks annotation status with an audit trail stored under the ``extra``
metadata attribute.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

from medh5.exceptions import MEDH5FileError, MEDH5ValidationError
from medh5.integrity import _CHECKSUM_ATTR, write_checksum
from medh5.meta import _validate_suffix, read_meta

_REVIEW_STATUSES: tuple[str, ...] = ("pending", "reviewed", "flagged", "rejected")
_REVIEW_SCHEMA_VERSION = 1


def _looks_like_already_open(exc: OSError) -> bool:
    """Heuristic: does *exc* look like HDF5's single-open-per-process error?"""
    msg = str(exc).lower()
    return "already open" in msg or "unable to lock file" in msg


def _wrap_open_or_lock_error(
    path: Path,
    exc: OSError,
    *,
    action: str,
    before_phrase: str,
) -> MEDH5FileError:
    """Build the MEDH5FileError for an open/lock OSError raised by h5py.

    *action* is the infinitive verb used in the generic
    ``"Failed to {action} '{path}'"`` message; *before_phrase* is the
    gerund used in the already-open hint
    (``"before {before_phrase}"``).
    """
    if _looks_like_already_open(exc):
        return MEDH5FileError(
            f"'{path}' is already open in this process; close other "
            f"MEDH5File handles before {before_phrase}"
        )
    return MEDH5FileError(f"Failed to {action} '{path}': {exc}")


@dataclass
class ReviewStatus:
    """Lightweight view of the curation/review state stored under ``extra``."""

    status: str = "pending"
    annotator: str | None = None
    timestamp: str | None = None
    notes: str | None = None
    history: list[dict[str, Any]] | None = None


def set_review_status(
    path: str | Path,
    *,
    status: str,
    annotator: str | None = None,
    notes: str | None = None,
    timestamp: str | None = None,
    on_reopened: Callable[[Path], None] | None = None,
) -> ReviewStatus:
    """Record a review/curation status under ``extra["review"]``.

    The previous state is appended to ``extra["review"]["history"]``
    so the audit trail is preserved across calls.

    Requires exclusive write access: HDF5 refuses to reopen a file
    already open elsewhere in the same process. Close any lingering
    :class:`MEDH5File` handles (or drop references held by lazy
    consumers) before calling.

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
    on_reopened : callable, optional
        Invoked with *path* after the write handle has been closed and
        only when the update succeeded. Lazy-read consumers use this to
        re-acquire handles or rebind cached views. See
        :func:`medh5.open_shared` for a ref-counted reader that pairs
        well with this hook.

    Returns
    -------
    ReviewStatus
        The freshly persisted review state, so callers can refresh UI
        without re-reading the file.

    Raises
    ------
    MEDH5ValidationError
        On unknown status.
    MEDH5FileError
        If the file cannot be opened, or is already open in this
        process by another handle.
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
            # Snapshot the previous state (or implicit "pending") so the
            # audit trail covers the sample's pre-review life.
            history.append(
                {
                    "status": review.get("status") or "pending",
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
                    "schema_version": _REVIEW_SCHEMA_VERSION,
                }
            )
            extra["review"] = review
            f.attrs["extra"] = json.dumps(extra)
            if _CHECKSUM_ATTR in f.attrs:
                write_checksum(f)
    except MEDH5ValidationError:
        raise
    except OSError as exc:
        raise _wrap_open_or_lock_error(
            path,
            exc,
            action="update",
            before_phrase="setting review status",
        ) from exc

    if on_reopened is not None:
        on_reopened(path)

    return ReviewStatus(
        status=status,
        annotator=annotator,
        timestamp=timestamp,
        notes=notes,
        history=history,
    )


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
