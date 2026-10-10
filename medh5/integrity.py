"""Content addressing, verification and repair (spec §13, §14.3).

A dataset digest covers the object path, dtype, shape and the **decompressed**
little-endian bytes, so recompression changes every stored byte and no
digest.  ``content_id`` is a Merkle root over the *stored* digests, ``meta``
and canonical attributes: an edited dataset breaks its object digest and
leaves the root matching --- verify per object, never only the root.

:func:`verify_root` re-digests every object (or the *partial* list) and
compares with what the file stores; it reports rather than raising, so a
curator sees every mismatch at once.

Repairs never touch ground truth.  A stale index is rebuilt; digests are
rewritten only when asked, and the rewrite is recorded as an activity --- a
digest mismatch can mean corruption as easily as an edit, so restamping it
is a decision a person makes, not a default.

The functions taking stored objects accept the ``Dataset`` and ``Group``
views this package hands out (``Sample.root``, ``Image.dataset``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core

# -- digests and content_id (§13.1-§13.2) -------------------------------------

DEFAULT_ALGO: str = _core.DEFAULT_ALGO
DIGEST_ALGOS: tuple[str, ...] = _core.DIGEST_ALGOS
STREAM_BYTES: int = _core.STREAM_BYTES
"""Datasets are hashed in slabs of at most this many bytes."""

parse_digest = _core.parse_digest
digest_bytes = _core.digest_bytes
array_digest = _core.array_digest
dataset_digest = _core.dataset_digest
canonical_attrs = _core.canonical_attrs
attrs_digest = _core.attrs_digest
relative_path = _core.relative_path
group_digest = _core.group_digest
compute_content_id = _core.compute_content_id
collect_digests = _core.collect_digests

# -- verification (§13) ---------------------------------------------------------

ATTESTED_GROUPS: tuple[str, ...] = _core.ATTESTED_GROUPS


@dataclass(slots=True)
class VerifyResult:
    """Outcome of a verification pass."""

    checked: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    undigested: tuple[str, ...] = ()
    malformed: tuple[str, ...] = ()
    content_id_declared: str | None = None
    content_id_computed: str | None = None
    stale_index: tuple[str, ...] = ()
    unattested: tuple[str, ...] = ()
    """Paths inside an object a declared ``content_id`` covers that no line
    of it binds there.

    ``content_id`` is a root over the digests that are *present*, so a dataset
    added to an annotation without one changes what the object means while
    the root still matches.  A line binds bytes to the path it names, so a
    path that is not its object's own --- an alias sorting before it, a soft
    link --- could be relinked to other covered bytes under the same root
    (``E704``).  The writer digests every dataset and makes no links, so none
    of its files has either.
    """
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def content_id_ok(self) -> bool | None:
        """Whether the root ``content_id`` matches, or ``None`` when unjudged:
        a file that declares none, or a *partial* pass."""
        if self.content_id_declared is None or self.content_id_computed is None:
            return None
        return self.content_id_declared == self.content_id_computed

    @property
    def ok(self) -> bool:
        return (
            not self.mismatched
            and not self.malformed
            and not self.unattested
            and self.content_id_ok is not False
        )

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": len(self.checked),
            "mismatched": list(self.mismatched),
            "undigested": list(self.undigested),
            "unattested": list(self.unattested),
            "malformed": list(self.malformed),
            "content_id_ok": self.content_id_ok,
            "stale_index": list(self.stale_index),
        }


verify_object = _core.verify_object
stale_index_entries = _core.stale_index_entries
raw_chunks = _core.raw_chunks
subtrees_identical = _core.subtrees_identical


def verify_root(
    root: Any,
    attr_names: Any = None,
    *,
    partial: Any = None,
    check_content_id: bool = True,
) -> VerifyResult:
    """Verify a sample root (``Sample.root``, or a collection member's)."""
    return VerifyResult(
        **_core.verify_root(
            root,
            attr_names,
            partial=None if partial is None else list(partial),
            check_content_id=check_content_id,
        )
    )


# -- diagnosis and repair (§13.3, §14.3) ----------------------------------------


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


__all__ = [
    "ATTESTED_GROUPS",
    "DEFAULT_ALGO",
    "DIGEST_ALGOS",
    "STREAM_BYTES",
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
    "raw_chunks",
    "relative_path",
    "stale_index_entries",
    "subtrees_identical",
    "verify_object",
    "verify_root",
]
