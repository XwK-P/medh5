"""Verification: per-object digests, ``content_id`` and index currency (§13).

``verify_root`` re-digests every object (or the *partial* list) and compares
with what the file stores; it reports rather than raising, so a curator sees
every mismatch at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medh5 import _core

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
    """Undigested datasets inside an object a declared ``content_id`` covers.

    ``content_id`` is a root over the digests that are *present*, so a dataset
    added to an annotation without one changes what the object means while
    the root still matches.  The writer digests every dataset, so none of its
    files carries one.
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


__all__ = [
    "ATTESTED_GROUPS",
    "VerifyResult",
    "raw_chunks",
    "stale_index_entries",
    "subtrees_identical",
    "verify_object",
    "verify_root",
]
