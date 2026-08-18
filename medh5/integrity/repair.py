"""``medh5 fix`` --- rebuilding what is derived, restamping what is claimed.

Two very different operations share one command, and the difference matters.

**Rebuilding an index** recomputes a cache from the data it caches (§14.3).
Nothing is asserted that was not already true; a stale index is a performance
bug, and rebuilding it is repair in the ordinary sense.

**Rewriting digests** is not repair.  A digest that no longer matches is
evidence that the bytes changed since whoever wrote them signed for them, and
recomputing it does not undo the change --- it destroys the evidence.  There
are legitimate reasons to do it (an external tool edited the file and cannot
restamp; a curator has independently confirmed the new content), and there is
exactly one honest way: say so in the file.  So ``rewrite_digests`` records a
provenance activity naming what was restamped and what it did *not* verify,
and refuses to run silently.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import medh5
from medh5.errors import MEDH5ValidationError


@dataclass(slots=True)
class Diagnosis:
    """What is wrong with one file, before anything is done about it."""

    path: str
    mismatched: tuple[str, ...] = ()
    undigested: tuple[str, ...] = ()
    stale_index: tuple[str, ...] = ()
    missing_index: tuple[str, ...] = ()
    content_id_ok: bool | None = None

    @property
    def needs_index(self) -> bool:
        return bool(self.stale_index or self.missing_index)

    @property
    def needs_digests(self) -> bool:
        return bool(self.mismatched) or self.content_id_ok is False

    @property
    def clean(self) -> bool:
        return not self.needs_index and not self.needs_digests

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mismatched": list(self.mismatched),
            "undigested": list(self.undigested),
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
    """What a file needs, without touching it."""
    from medh5.annotations.base import VoxelAnnotation
    from medh5.integrity.verify import stale_index_entries

    with medh5.open(path) as sample:
        result = sample.verify()
        indexable = {
            name
            for name, annotation in sample.annotations.items()
            if isinstance(annotation, VoxelAnnotation) and annotation.kind != "mask"
        }
        present = set(sample.index)
        return Diagnosis(
            path=os.fspath(path),
            mismatched=tuple(result.mismatched),
            undigested=tuple(result.undigested),
            stale_index=tuple(stale_index_entries(sample.root)),
            missing_index=tuple(sorted(indexable - present)) if present else (),
            content_id_ok=result.content_id_ok,
        )


def fix(
    path: str | os.PathLike[str],
    *,
    rebuild_index: bool = False,
    rewrite_digests: bool = False,
    reason: str | None = None,
    performed_by: str | None = None,
    max_coords: int | None = None,
) -> Repair:
    """Repair one file.  Neither flag means diagnose and change nothing.

    ``missing_index`` is only rebuilt for annotations that already had one:
    building an index where the curator never asked for one is a decision about
    their storage budget, not a repair.  ``medh5 index build`` is the command
    that adds one.
    """
    diagnosis = diagnose(path)
    repair = Repair(path=os.fspath(path), diagnosis=diagnosis)
    if not rebuild_index and not rewrite_digests:
        return repair

    if rewrite_digests and not reason:
        raise MEDH5ValidationError(
            "rewriting digests re-attests content this tool did not verify; "
            "pass a reason, which is recorded in the file's provenance"
        )

    if rebuild_index and not rewrite_digests and diagnosis.needs_digests:
        # An amend restamps every digest from the bytes it copies, so rebuilding
        # an index in a file that already has a mismatch would recompute the
        # mismatched digest as a side effect --- destroying the evidence with no
        # reason, no provenance and `rewrote_digests` reporting False.  The
        # guard the module is built around belongs on this path too.
        raise MEDH5ValidationError(
            f"{os.fspath(path)}: cannot rebuild the index without restamping "
            f"digests, and this file has {len(diagnosis.mismatched)} that no "
            "longer match its bytes"
            + (
                ""
                if diagnosis.content_id_ok is not False
                else " (and a stale content_id)"
            )
            + ". Rebuilding would recompute them and the mismatch would vanish "
            "unrecorded. Restore the content, or pass rewrite_digests with a "
            "reason so the re-attestation is written into the file."
        )

    # `stale_index` and `missing_index` together are the whole of what a repair
    # may touch.  Passing an empty list through as `None` used to mean "every
    # indexable annotation", so a curator who deliberately shipped a file with
    # no index got one built for everything --- a decision about their storage
    # budget, not a repair.  `medh5 index build` is the command that adds one.
    names = (
        sorted({*diagnosis.stale_index, *diagnosis.missing_index})
        if rebuild_index
        else []
    )
    if not names and not rewrite_digests:
        repair.notes.append("nothing to rebuild: no sampling index is stale or missing")
        return repair

    with medh5.amend(path) as writer:
        if names:
            kwargs: dict[str, Any] = {}
            if max_coords is not None:
                kwargs["max_coords"] = max_coords
            repair.rebuilt_index = writer.build_index(names, **kwargs)
        if rewrite_digests:
            agent = (
                writer.person(performed_by)
                if performed_by
                else writer.software("medh5", medh5.__version__)
            )
            writer.activity(
                # `other`, not a new activity type: §11.1's vocabulary describes
                # what produced the *data*, and a repair tool did not produce any.
                "other",
                agent=agent,
                tool="medh5 fix --rewrite-digests",
                params={
                    "reason": reason,
                    "restamped": sorted(diagnosis.mismatched) or "all",
                    "verified_content": False,
                },
            )
            repair.rewrote_digests = True
            repair.notes.append(
                "digests were recomputed from the current bytes; this asserts "
                "nothing about whether those bytes are correct"
            )
    # The amend rewrote the file, so every digest and the content_id are now
    # stamped from what is actually there --- which is precisely why the note
    # above is needed.
    with medh5.open(path) as sample:
        repair.content_id = sample.content_id
    return repair


def fix_paths(
    paths: Sequence[str | os.PathLike[str]],
    **options: Any,
) -> list[Repair]:
    return [fix(path, **options) for path in paths]


__all__ = ["Diagnosis", "Repair", "diagnose", "fix", "fix_paths"]
