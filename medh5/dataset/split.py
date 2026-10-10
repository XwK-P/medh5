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

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core
from medh5.dataset.manifest import Entry, Manifest

DEFAULT_RATIOS: dict[str, float] = dict(_core.dataset_default_ratios())


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

    def _doc(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "manifest_sha256": self.manifest_sha256,
            "group_by": self.group_by,
            "stratify_by": self.stratify_by,
            "seed": self.seed,
            "k_folds": self.k_folds,
            "ratios": dict(self.ratios),
            "assignments": [a.to_json() for a in self.assignments],
        }

    def partition_of(self, entry: Entry) -> str | None:
        found: str | None = _core.dataset_split_partition_of(
            self._doc(), entry._fields()
        )
        return found

    def fold_of(self, entry: Entry) -> int | None:
        found: int | None = _core.dataset_split_fold_of(self._doc(), entry._fields())
        return found

    def paths(self, partition: str) -> tuple[str, ...]:
        """The entries of *partition*: a file's path, or ``path::key`` for a
        sample inside a collection (``open_any(path, key=key)`` opens it).

        Two members of one collection in different partitions were both its
        path, so a loader building per-partition file lists put the whole
        collection in both.  A split written before 2.0 carries paths alone.
        """
        return tuple(_core.dataset_split_paths(self._doc(), partition))

    @property
    def counts(self) -> dict[str, int]:
        found: dict[str, int] = _core.dataset_split_counts(self._doc())
        return found

    def balance(self) -> dict[str, dict[str, int]]:
        """Achieved stratum counts per partition --- what balance really came out."""
        found: dict[str, dict[str, int]] = _core.dataset_split_balance(self._doc())
        return found

    @property
    def empty_folds(self) -> tuple[int, ...]:
        """Folds that were asked for and got no groups.

        `--k-folds 5` over three groups yields three folds, and a cross
        validation that quietly runs three ways instead of five is not one
        anybody would notice until they compared results with a colleague.
        """
        return tuple(_core.dataset_split_empty_folds(self._doc()))

    @property
    def underfilled(self) -> tuple[str, ...]:
        """Partitions that were asked for and got nothing.

        Four groups at 0.7/0.15/0.15 is 2.8/0.6/0.6, and no integer allocation
        of four indivisible groups fills all three.  That is arithmetic, not a
        bug --- but an empty test set is the kind of thing that is noticed after
        the results are written up, so it is reported rather than left to be
        inferred from a table of counts.
        """
        return tuple(_core.dataset_split_underfilled(self._doc()))

    def leaks(self) -> tuple[str, ...]:
        """Groups, and entries, that ended up in more than one partition.

        Structurally impossible here --- a group is assigned once --- so this is
        a self-check, not a feature, and the check a split loaded from a file
        needs.  It exists because "impossible" is what every leakage bug was
        called before it shipped.
        """
        return tuple(_core.dataset_split_leaks(self._doc()))

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_split_json(self._doc())
        return found

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
    ``train``/``val``/``test`` in the given ratios.  Deterministic given the
    manifest digest, *seed* and the parameters.
    """
    return Split.from_json(
        _core.dataset_make_splits(
            manifest._doc(),
            set_id=set_id,
            group_by=group_by,
            stratify_by=stratify_by,
            ratios=None if ratios is None else dict(ratios),
            k_folds=k_folds,
            seed=seed,
        )
    )


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

    Everything the write needs --- a ``fold`` the split has, no sample inside a
    collection --- is checked before the first file is amended, so a refusal
    leaves every file as it was.  A fold outside the split's range was not a
    refusal at all: every file was written as ``train``.

    Amending rewrites each file, so this is not free --- and it is optional.  A
    split is fully usable as a JSON file; the claim exists for the case where a
    sample travels away from its manifest and has to carry its provenance.
    """
    return tuple(
        _core.dataset_write_claims(
            split._doc(), manifest._doc(), assigned_by=assigned_by, fold=fold
        )
    )


def load(path: str | os.PathLike[str]) -> Split:
    """Read a split written by ``medh5 dataset split -o``."""
    return Split.from_json(_core.dataset_split_load(os.fspath(path)))


__all__ = [
    "DEFAULT_RATIOS",
    "Assignment",
    "Split",
    "load",
    "make_splits",
    "write_claims",
]
