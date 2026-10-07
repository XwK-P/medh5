"""Diagnosing and repairing derived data: indices and digests (§13.3, §14.3).

Repairs never touch ground truth.  A stale index is rebuilt; digests are
rewritten only when asked, and the rewrite is recorded as an activity --- a
digest mismatch can mean corruption as easily as an edit, so restamping it
is a decision a person makes, not a default.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core


@dataclass(slots=True)
class Diagnosis:
    """What is wrong with one file, before anything is done about it."""

    path: str
    mismatched: tuple[str, ...] = ()
    undigested: tuple[str, ...] = ()
    unattested: tuple[str, ...] = ()
    stale_index: tuple[str, ...] = ()
    missing_index: tuple[str, ...] = ()
    content_id_ok: bool | None = None

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Diagnosis:
        return cls(
            path=doc["path"],
            mismatched=tuple(doc.get("mismatched") or ()),
            undigested=tuple(doc.get("undigested") or ()),
            unattested=tuple(doc.get("unattested") or ()),
            stale_index=tuple(doc.get("stale_index") or ()),
            missing_index=tuple(doc.get("missing_index") or ()),
            content_id_ok=doc.get("content_id_ok"),
        )

    @property
    def needs_index(self) -> bool:
        """A *stale* index is a defect.  An absent one is a choice (§14.3)."""
        return bool(self.stale_index)

    @property
    def needs_digests(self) -> bool:
        return (
            bool(self.mismatched)
            or bool(self.unattested)
            or self.content_id_ok is False
        )

    @property
    def clean(self) -> bool:
        return not self.needs_index and not self.needs_digests

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mismatched": list(self.mismatched),
            "undigested": list(self.undigested),
            "unattested": list(self.unattested),
            "stale_index": list(self.stale_index),
            "missing_index": list(self.missing_index),
            "content_id_ok": self.content_id_ok,
            "needs_index": self.needs_index,
            "needs_digests": self.needs_digests,
        }


@dataclass(slots=True)
class Repair:
    """What was actually done to one file."""

    path: str
    diagnosis: Diagnosis
    rebuilt_index: tuple[str, ...] = ()
    rewrote_digests: bool = False
    content_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Repair:
        return cls(
            path=doc["path"],
            diagnosis=Diagnosis.from_json(doc["diagnosis"]),
            rebuilt_index=tuple(doc.get("rebuilt_index") or ()),
            rewrote_digests=bool(doc.get("rewrote_digests", False)),
            content_id=doc.get("content_id"),
            notes=list(doc.get("notes") or ()),
        )

    @property
    def changed(self) -> bool:
        return bool(self.rebuilt_index) or self.rewrote_digests

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "diagnosis": self.diagnosis.to_json(),
            "rebuilt_index": list(self.rebuilt_index),
            "rewrote_digests": self.rewrote_digests,
            "content_id": self.content_id,
            "changed": self.changed,
            "notes": list(self.notes),
        }


def diagnose(path: str | os.PathLike[str]) -> Diagnosis:
    """What is stale, mismatched or missing in one file."""
    return Diagnosis.from_json(_core.diagnose(os.fspath(path)))


def fix(
    path: str | os.PathLike[str],
    *,
    rebuild_index: bool = False,
    rewrite_digests: bool = False,
    reason: str | None = None,
    performed_by: str | None = None,
    max_coords: int | None = None,
) -> Repair:
    """Repair what was asked: rebuild stale indices, rewrite digests (with a
    *reason*, recorded as a provenance activity).  Copy-on-write."""
    return Repair.from_json(
        _core.fix(
            os.fspath(path),
            rebuild_index=rebuild_index,
            rewrite_digests=rewrite_digests,
            reason=reason,
            performed_by=performed_by,
            max_coords=max_coords,
        )
    )


def fix_paths(paths: Sequence[str | os.PathLike[str]], **options: Any) -> list[Repair]:
    """:func:`fix` over many files."""
    return [
        Repair.from_json(doc)
        for doc in _core.fix_paths([os.fspath(p) for p in paths], **options)
    ]


__all__ = ["Diagnosis", "Repair", "diagnose", "fix", "fix_paths"]
