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

**Subject leakage.**  Two files carrying the same grouping key (§12.2) land in
different partitions of one split.  A sample never spans subjects (§3.7), so
assigning whole files is subject-safe *by construction* --- but only if the
assignment itself respected the grouping, and nothing prevents a hand-edited
manifest from splitting a patient's baseline into ``train`` and their follow-up
into ``test``.  That is the single most common evaluation error in medical AI,
it inflates every reported metric, and it is invisible in any one file.  It gets
its own report rather than being folded into W906, because the remedy is
different: a conflicting claim needs a re-stamp, leakage needs a re-split.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.curation.identity import SplitClaim


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
        return {
            "path": self.path,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "group_id": self.group_id,
            **self.claim.to_json(),
        }


@dataclass(frozen=True, slots=True)
class Leak:
    """One grouping key appearing in more than one partition of one split."""

    set_id: str
    group_id: str
    partitions: tuple[str, ...]
    paths: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "group_id": self.group_id,
            "partitions": list(self.partitions),
            "paths": list(self.paths),
        }

    def __str__(self) -> str:
        return (
            f"{self.set_id}: group {self.group_id!r} is in "
            f"{', '.join(self.partitions)} ({len(self.paths)} files)"
        )


@dataclass(frozen=True, slots=True)
class Conflict:
    """One ``set_id`` claimed against more than one manifest (W906)."""

    set_id: str
    manifests: tuple[str, ...]
    paths_by_manifest: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "manifests": list(self.manifests),
            "paths_by_manifest": {
                k: list(v) for k, v in self.paths_by_manifest.items()
            },
        }

    def __str__(self) -> str:
        return (
            f"{self.set_id}: {len(self.manifests)} different manifest hashes "
            f"({', '.join(m[:12] for m in self.manifests)})"
        )


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
        return not self.conflicts and not self.leaks and not self.unreadable

    @property
    def set_ids(self) -> tuple[str, ...]:
        return tuple(sorted({m.set_id for m in self.memberships}))

    def partitions(self, set_id: str) -> dict[str, tuple[str, ...]]:
        """``partition -> sample ids`` for one split."""
        out: dict[str, list[str]] = {}
        for member in self.memberships:
            if member.set_id == set_id:
                out.setdefault(member.partition, []).append(member.sample_id)
        return {k: tuple(sorted(v)) for k, v in sorted(out.items())}

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            set_id: {k: len(v) for k, v in self.partitions(set_id).items()}
            for set_id in self.set_ids
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sets": self.set_ids,
            "counts": self.counts(),
            "conflicts": [c.to_json() for c in self.conflicts],
            "leaks": [leak.to_json() for leak in self.leaks],
            "unclaimed": list(self.unclaimed),
            "unreadable": [{"path": p, "error": e} for p, e in self.unreadable],
        }


def audit_splits(paths: Sequence[str | os.PathLike[str]]) -> SplitAudit:
    """Read every file's claims and cross-check them (spec §12.3)."""
    memberships: list[Membership] = []
    unclaimed: list[str] = []
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        try:
            found = list(_memberships_of(path))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop an audit
            unreadable.append((os.fspath(path), f"{type(exc).__name__}: {exc}"))
            continue
        if not found:
            unclaimed.append(os.fspath(path))
        memberships.extend(found)
    return SplitAudit(
        memberships=tuple(memberships),
        conflicts=_conflicts(memberships),
        leaks=_leaks(memberships),
        unclaimed=tuple(sorted(unclaimed)),
        unreadable=tuple(sorted(unreadable)),
    )


def _memberships_of(path: str | os.PathLike[str]) -> Iterator[Membership]:
    """Every claim in a file, whether it holds one sample or many."""
    from medh5.collection import open_any
    from medh5.sample import Sample

    text = os.fspath(path)
    with open_any(path) as opened:
        samples = (
            [("", opened)]
            if isinstance(opened, Sample)
            else [(key, opened[key]) for key in opened]
        )
        for key, sample in samples:
            document = sample.document
            identity = document.identity
            location = text if not key else f"{text}::{key}"
            for claim in document.splits:
                yield Membership(
                    path=location,
                    sample_id=identity.sample_id,
                    subject_id=identity.subject_id,
                    group_id=document.cohort.grouping_key(identity.subject_id),
                    claim=claim,
                )


def _conflicts(memberships: Sequence[Membership]) -> tuple[Conflict, ...]:
    by_set: dict[str, dict[str, list[str]]] = {}
    for member in memberships:
        digest = member.claim.manifest_sha256
        if not digest:
            continue
        by_set.setdefault(member.set_id, {}).setdefault(digest, []).append(member.path)
    out = []
    for set_id, by_manifest in sorted(by_set.items()):
        if len(by_manifest) > 1:
            out.append(
                Conflict(
                    set_id=set_id,
                    manifests=tuple(sorted(by_manifest)),
                    paths_by_manifest={
                        k: tuple(sorted(v)) for k, v in sorted(by_manifest.items())
                    },
                )
            )
    return tuple(out)


def _leaks(memberships: Sequence[Membership]) -> tuple[Leak, ...]:
    by_group: dict[tuple[str, str], dict[str, list[str]]] = {}
    for member in memberships:
        key = (member.set_id, member.group_id)
        by_group.setdefault(key, {}).setdefault(member.partition, []).append(
            member.path
        )
    out = []
    for (set_id, group_id), by_partition in sorted(by_group.items()):
        if len(by_partition) > 1:
            out.append(
                Leak(
                    set_id=set_id,
                    group_id=group_id,
                    partitions=tuple(sorted(by_partition)),
                    paths=tuple(sorted(p for v in by_partition.values() for p in v)),
                )
            )
    return tuple(out)


__all__ = ["Conflict", "Leak", "Membership", "SplitAudit", "audit_splits"]
