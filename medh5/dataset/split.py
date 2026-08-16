"""Splitting a cohort so the split means something (spec §12.3).

Three rules, and they are the whole module:

1. **Group before you split.** The unit is a *group* --- ``cohort.group_id`` if
   the file declares one, otherwise the subject --- never a file.  A subject
   with a baseline and two follow-ups is one unit; splitting per file puts the
   same anatomy in train and test and reports a score that is partly memory.
2. **Stratify on what you can see.** Stratification balances a metadata field
   across partitions.  It is best-effort by construction: groups are indivisible
   and a group's stratum is its majority, so exact balance is not always
   reachable, and ``Split.balance`` says what was actually achieved rather than
   what was asked for.
3. **A claim in a file is not the split.** ``write_claims`` stamps each sample
   with its partition *and the manifest digest it came from*, so a later reader
   can tell a current claim from one that predates a re-split.

Assignment is deterministic given ``(manifest digest, seed, parameters)``: the
same cohort split twice on two machines produces the same partitions.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.curation.identity import PARTITIONS
from medh5.dataset.manifest import Entry, Manifest
from medh5.errors import MEDH5ValidationError

DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass(slots=True)
class Assignment:
    """Where one group went."""

    group: str
    partition: str
    fold: int | None = None
    stratum: str | None = None
    entries: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "partition": self.partition,
            "fold": self.fold,
            "stratum": self.stratum,
            "entries": list(self.entries),
        }


@dataclass(slots=True)
class Split:
    """A complete assignment of a cohort's groups."""

    set_id: str
    manifest_sha256: str
    assignments: list[Assignment] = field(default_factory=list)
    group_by: str = "group_id"
    stratify_by: str | None = None
    seed: int = 0
    k_folds: int | None = None
    ratios: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))

    def partition_of(self, entry: Entry) -> str | None:
        key = str(entry.field(self.group_by))
        for assignment in self.assignments:
            if assignment.group == key:
                return assignment.partition
        return None

    def fold_of(self, entry: Entry) -> int | None:
        key = str(entry.field(self.group_by))
        for assignment in self.assignments:
            if assignment.group == key:
                return assignment.fold
        return None

    def paths(self, partition: str) -> tuple[str, ...]:
        return tuple(
            path
            for a in self.assignments
            if a.partition == partition
            for path in a.entries
        )

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for assignment in self.assignments:
            out[assignment.partition] = out.get(assignment.partition, 0) + len(
                assignment.entries
            )
        return dict(sorted(out.items()))

    def balance(self) -> dict[str, dict[str, int]]:
        """Achieved stratum counts per partition --- what balance really came out."""
        out: dict[str, dict[str, int]] = {}
        for assignment in self.assignments:
            bucket = out.setdefault(assignment.partition, {})
            stratum = assignment.stratum or "-"
            bucket[stratum] = bucket.get(stratum, 0) + len(assignment.entries)
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    @property
    def underfilled(self) -> tuple[str, ...]:
        """Partitions that were asked for and got nothing.

        Four groups at 0.7/0.15/0.15 is 2.8/0.6/0.6, and no integer allocation
        of four indivisible groups fills all three.  That is arithmetic, not a
        bug --- but an empty test set is the kind of thing that is noticed after
        the results are written up, so it is reported rather than left to be
        inferred from a table of counts.
        """
        if self.k_folds is not None:
            return ()
        counts = self.counts
        return tuple(
            p for p, share in self.ratios.items() if share > 0 and not counts.get(p)
        )

    def leaks(self) -> tuple[str, ...]:
        """Groups that ended up in more than one partition.

        Structurally impossible here --- a group is assigned once --- so this is
        a self-check, not a feature.  It exists because "impossible" is what
        every leakage bug was called before it shipped.
        """
        seen: dict[str, str] = {}
        bad: list[str] = []
        for assignment in self.assignments:
            previous = seen.setdefault(assignment.group, assignment.partition)
            if previous != assignment.partition:
                bad.append(assignment.group)
        return tuple(sorted(set(bad)))

    def to_json(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "manifest_sha256": self.manifest_sha256,
            "group_by": self.group_by,
            "stratify_by": self.stratify_by,
            "seed": self.seed,
            "k_folds": self.k_folds,
            "ratios": dict(self.ratios),
            "counts": self.counts,
            "underfilled": list(self.underfilled),
            "balance": self.balance(),
            "assignments": [a.to_json() for a in self.assignments],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Split:
        return cls(
            set_id=str(doc["set_id"]),
            manifest_sha256=str(doc["manifest_sha256"]),
            assignments=[
                Assignment(
                    group=str(a["group"]),
                    partition=str(a["partition"]),
                    fold=a.get("fold"),
                    stratum=a.get("stratum"),
                    entries=tuple(a.get("entries", ())),
                )
                for a in doc.get("assignments", ())
            ],
            group_by=str(doc.get("group_by", "group_id")),
            stratify_by=doc.get("stratify_by"),
            seed=int(doc.get("seed", 0)),
            k_folds=doc.get("k_folds"),
            ratios=dict(doc.get("ratios", DEFAULT_RATIOS)),
        )


def _rank(set_id: str, seed: int, group: str) -> str:
    """A stable pseudo-random ordering key for a group.

    Hashing rather than shuffling: the order depends only on the inputs, so it
    does not drift with Python's hash randomisation, the order the filesystem
    happened to return files in, or the version of ``random`` in use.
    """
    return hashlib.sha256(f"{set_id}:{seed}:{group}".encode()).hexdigest()


def _stratum_of(entries: Sequence[Entry], by: str | None) -> str | None:
    """A group's stratum: the majority value among its samples.

    A group is indivisible, so when its samples disagree --- two visits recorded
    on different scanners --- something has to give, and the majority is the
    honest summary.  Ties break on the lexically first value for determinism.
    """
    if by is None:
        return None
    tally: dict[str, int] = {}
    for entry in entries:
        value = str(entry.field(by))
        tally[value] = tally.get(value, 0) + 1
    return max(sorted(tally), key=lambda k: tally[k])


def make_splits(
    manifest: Manifest,
    *,
    set_id: str = "default",
    group_by: str = "group_id",
    stratify_by: str | None = None,
    ratios: Mapping[str, float] | None = None,
    k_folds: int | None = None,
    seed: int = 0,
) -> Split:
    """Assign every group in *manifest* to a partition, or to a fold.

    With ``k_folds`` the partitions are ``fold-0 … fold-(k-1)`` recorded as
    ``holdout`` claims with a ``fold`` number; without it, groups go to
    ``train``/``val``/``test`` in the given ratios.
    """
    if not len(manifest):
        raise MEDH5ValidationError("cannot split an empty manifest")
    shares = dict(ratios or DEFAULT_RATIOS)
    if k_folds is None:
        unknown = sorted(set(shares) - set(PARTITIONS))
        if unknown:
            raise MEDH5ValidationError(
                f"unknown partition(s) {unknown}; expected {list(PARTITIONS)}"
            )
        total = sum(shares.values())
        if total <= 0:
            raise MEDH5ValidationError("split ratios must sum to more than zero")
        shares = {k: v / total for k, v in shares.items()}
    elif k_folds < 2:
        raise MEDH5ValidationError("k_folds must be at least 2")

    grouped = manifest.groups(group_by)
    digest = manifest.sha256()
    split = Split(
        set_id=set_id,
        manifest_sha256=digest,
        group_by=group_by,
        stratify_by=stratify_by,
        seed=seed,
        k_folds=k_folds,
        ratios=shares,
    )

    strata: dict[str | None, list[tuple[str, list[Entry]]]] = {}
    for group, entries in grouped.items():
        stratum = _stratum_of(entries, stratify_by)
        strata.setdefault(stratum, []).append((group, entries))

    # Order within each stratum, then interleave the strata round-robin.  Two
    # things have to hold at once: each stratum must be spread across the
    # partitions (that is what stratifying means), and the *global* ratios must
    # still be met.  Dealing each stratum with its own tally satisfies the
    # first and wrecks the second --- every small stratum rounds toward train,
    # and a four-group cohort ends up entirely in train.  Interleaving, then
    # dealing once against a single running tally, satisfies both.
    ordered_strata = {
        stratum: sorted(members, key=lambda kv: _rank(set_id, seed, kv[0]))
        for stratum, members in sorted(strata.items(), key=lambda kv: str(kv[0]))
    }
    interleaved: list[tuple[str | None, str, list[Entry]]] = []
    depth = max((len(v) for v in ordered_strata.values()), default=0)
    names = list(ordered_strata)
    for position in range(depth):
        # Rotate which stratum leads each round.  Without it the stratum that
        # sorts first always occupies the same positions in the deal, and the
        # small partitions --- the ones with one or two groups to give away ---
        # fill from that stratum every time.
        for offset in range(len(names)):
            stratum = names[(position + offset) % len(names)]
            members = ordered_strata[stratum]
            if position < len(members):
                group, entries = members[position]
                interleaved.append((stratum, group, entries))

    if k_folds is not None:
        for position, (stratum, group, entries) in enumerate(interleaved):
            split.assignments.append(
                Assignment(
                    group=group,
                    partition="holdout",
                    fold=position % k_folds,
                    stratum=stratum,
                    entries=tuple(e.path for e in entries),
                )
            )
    else:
        for stratum, group, entries, partition in _deal(interleaved, shares):
            split.assignments.append(
                Assignment(
                    group=group,
                    partition=partition,
                    stratum=stratum,
                    entries=tuple(e.path for e in entries),
                )
            )
    split.assignments.sort(key=lambda a: (a.partition, a.fold or 0, a.group))
    return split


def _deal(
    ordered: Sequence[tuple[str | None, str, list[Entry]]],
    shares: Mapping[str, float],
) -> list[tuple[str | None, str, list[Entry], str]]:
    """Hand out groups by largest remaining deficit against the target shares.

    Not "first 70 % to train": with 6 groups and a 0.7/0.15/0.15 split, slicing
    by index gives train everything.  Dealing by deficit gives the closest
    integer allocation those ratios admit --- and ``Split.underfilled`` says so
    when the closest is still zero for some partition.
    """
    names = [p for p in PARTITIONS if p in shares and shares[p] > 0]
    assigned: dict[str, int] = dict.fromkeys(names, 0)
    out: list[tuple[str | None, str, list[Entry], str]] = []
    for position, (stratum, group, entries) in enumerate(ordered):
        done = position + 1
        target = {p: shares[p] * done for p in names}
        partition = max(names, key=lambda p: (target[p] - assigned[p], -names.index(p)))
        assigned[partition] += 1
        out.append((stratum, group, entries, partition))
    return out


def write_claims(
    split: Split,
    manifest: Manifest,
    *,
    assigned_by: str | None = None,
    fold: int | None = None,
) -> tuple[str, ...]:
    """Stamp each sample with its partition and the manifest digest (§12.3).

    With ``k_folds``, ``fold`` selects which fold is the validation set for this
    claim; every other fold becomes ``train``.  Without it the claim is the
    partition as assigned.

    Amending rewrites each file, so this is not free --- and it is optional.  A
    split is fully usable as a JSON file; the claim exists for the case where a
    sample travels away from its manifest and has to carry its provenance.
    """
    import medh5
    from medh5.curation.identity import SplitClaim

    written: list[str] = []
    by_path: dict[str, Assignment] = {}
    for assignment in split.assignments:
        for path in assignment.entries:
            by_path[path] = assignment

    for entry in manifest.entries:
        placed = by_path.get(entry.path)
        if placed is None:
            continue
        if entry.key is not None:
            # A sample inside a collection cannot be amended in place without
            # rewriting the shard; say so rather than silently skipping it.
            raise MEDH5ValidationError(
                f"{entry.path}#{entry.key} is inside a collection --- unpack it "
                "before writing split claims"
            )
        partition = placed.partition
        claim_fold = placed.fold
        if split.k_folds is not None:
            if fold is None:
                raise MEDH5ValidationError(
                    "a k-fold split needs --fold N to say which fold is validation"
                )
            partition = "val" if placed.fold == fold else "train"
        claim = SplitClaim(
            set_id=split.set_id,
            partition=partition,
            fold=claim_fold,
            assigned_by=assigned_by,
            manifest_sha256=split.manifest_sha256,
        )
        with medh5.amend(entry.path) as writer:
            writer.split(**claim.to_json())
        written.append(entry.path)
    return tuple(written)


def load(path: str | os.PathLike[str]) -> Split:
    import json
    from pathlib import Path

    return Split.from_json(json.loads(Path(os.fspath(path)).read_text()))


__all__ = [
    "DEFAULT_RATIOS",
    "Assignment",
    "Split",
    "load",
    "make_splits",
    "write_claims",
]
