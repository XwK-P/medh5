"""Auditing split claims across a cohort (spec §12.3).

A split claim in a file is a **membership claim, not an authority**: the
dataset-level manifest decides, and the in-file copy exists so that a single
file is debuggable on its own.  That design only pays off if the claims can be
checked against each other, which is what this module does --- and it is
necessarily a *cross-file* operation, so it lives here rather than in the
per-file validator.

Two findings matter, and they are not the same thing:

``W906`` --- **conflicting claims.**  Two files claim the same ``set_id`` against
different ``manifest_sha256`` values.  One of them predates a re-split, and any
training run that mixes them is using two different partitions at once.

**Subject leakage.**  Two files carrying the same grouping key (§12.2) --- or
the same subject under two keys --- land in different partitions of one split.
A sample never spans subjects (§3.7), so assigning whole files is subject-safe
*by construction* --- but only if the assignment itself respected the grouping,
and nothing prevents a hand-edited manifest from splitting a patient's baseline
into ``train`` and their follow-up into ``test``.  That is the single most
common evaluation error in medical AI, it inflates every reported metric, and
it is invisible in any one file.  It gets its own report rather than being
folded into W906, because the remedy is different: a conflicting claim needs a
re-stamp, leakage needs a re-split.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core
from medh5._core import SplitClaim


@dataclass(frozen=True, slots=True)
class Membership:
    """One file's claim to one partition of one split."""

    path: str
    sample_id: str
    subject_id: str
    group_id: str
    claim: SplitClaim

    @property
    def set_id(self) -> str:
        return self.claim.set_id

    @property
    def partition(self) -> str:
        return self.claim.partition

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.audit_membership_json(self)
        return result


@dataclass(frozen=True, slots=True)
class Leak:
    """One unit of anatomy appearing in more than one partition of one split.

    The unit is a connected component of ``subject_id`` and grouping key: two
    files are in one unit if they share either, transitively.  ``group_id``
    names the unit by its first grouping key; ``groups`` and ``subjects`` list
    everything it joined.
    """

    set_id: str
    group_id: str
    partitions: tuple[str, ...]
    paths: tuple[str, ...]
    groups: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.audit_leak_json(self)
        return result

    def __str__(self) -> str:
        result: str = _core.audit_leak_line(self)
        return result


@dataclass(frozen=True, slots=True)
class Conflict:
    """One ``set_id`` claimed against more than one manifest (W906)."""

    set_id: str
    manifests: tuple[str, ...]
    paths_by_manifest: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.audit_conflict_json(self)
        return result

    def __str__(self) -> str:
        result: str = _core.audit_conflict_line(self)
        return result


@dataclass(slots=True)
class SplitAudit:
    """What a cohort's split claims say, and where they disagree."""

    memberships: tuple[Membership, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    leaks: tuple[Leak, ...] = ()
    unclaimed: tuple[str, ...] = ()
    """Files carrying no split claim at all --- not an error, but easy to lose."""
    unreadable: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        result: bool = _core.audit_ok(self)
        return result

    @property
    def set_ids(self) -> tuple[str, ...]:
        result: tuple[str, ...] = _core.audit_set_ids(self)
        return result

    def partitions(self, set_id: str) -> dict[str, tuple[str, ...]]:
        """``partition -> sample ids`` for one split."""
        result: dict[str, tuple[str, ...]] = _core.audit_partitions(self, set_id)
        return result

    def counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = _core.audit_counts(self)
        return result

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.audit_json(self)
        return result

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> SplitAudit:
        """The audit the engine reported, as its field values."""
        return cls(
            memberships=tuple(Membership(**m) for m in fields["memberships"]),
            conflicts=tuple(Conflict(**c) for c in fields["conflicts"]),
            leaks=tuple(Leak(**leak) for leak in fields["leaks"]),
            unclaimed=tuple(fields["unclaimed"]),
            unreadable=tuple(fields["unreadable"]),
        )


def audit_splits(paths: Sequence[str | os.PathLike[str]]) -> SplitAudit:
    """Read every file's claims and cross-check them (spec §12.3).

    A file that cannot be read is listed under ``unreadable`` rather than
    stopping the audit: one bad file must not hide a leak in the others.
    """
    return SplitAudit.from_fields(_core.audit_splits([os.fspath(p) for p in paths]))


def anatomy_units(pairs: Iterable[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    """Grouping key -> every grouping key sharing anatomy with it, sorted.

    *pairs* are ``(subject_id, group_id)``.  A split is subject-safe only if
    no subject straddles two partitions, and a grouping key is a *coarsening*
    of subjects only when every file of a subject names the same key.  When two
    visits of one subject were curated under two ``group_id`` values, grouping by
    the key alone deals them as strangers: that is the case
    :func:`medh5.dataset.split.make_splits` refuses to produce, and an audit
    that groups by the key alone cannot see it.  So the unit is the connected
    component of subjects and keys (union--find): two files are one unit if
    they share either, transitively.
    """
    result: dict[str, tuple[str, ...]] = _core.audit_anatomy_units(list(pairs))
    return result


__all__ = [
    "Conflict",
    "Leak",
    "Membership",
    "SplitAudit",
    "anatomy_units",
    "audit_splits",
]
