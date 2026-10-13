"""Cohort tools: manifests, splits, statistics, cross-file checks.

The unit under test is never one file.  Everything here is about a property no
sample can carry on its own --- who agrees on a label set, who is in which
partition, what the intensity distribution is over a whole study.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5 import _core
from medh5.cli import main
from medh5.dataset.check import check
from medh5.dataset.manifest import Manifest, entries_for, find, scan
from medh5.dataset.split import make_splits, write_claims
from medh5.dataset.stats import DatasetStats, Moments, compute_stats, stats_for
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from tests.helpers import SHAPE, block, write_sample
from tests.kits import Flat, Hierarchy, Organs


@pytest.fixture
def cohort(tmp_path: Path, label_set: LabelSet) -> Path:
    """Six samples over four subjects, two sites, one label set."""
    root = tmp_path / "cohort"
    root.mkdir()
    plan = [
        ("subj-A", "site-A", ("tp0", "tp1")),
        ("subj-A", "site-A", ("tp0",)),
        ("subj-B", "site-A", ("tp0",)),
        ("subj-C", "site-B", ("tp0",)),
        ("subj-D", "site-B", ("tp0",)),
        ("subj-D", "site-B", ("tp0",)),
    ]
    masks = {1: block(SHAPE, (2, 2, 2), 8), 3: block(SHAPE, (4, 4, 4), 3)}
    for index, (subject, site, timepoints) in enumerate(plan):
        path = root / f"case-{index}.medh5"
        write_sample(
            path,
            label_set=label_set,
            masks=masks,
            timepoints=timepoints,
            sample_id=f"case-{index}",
        )
        with medh5.amend(path) as writer:
            writer.identity(subject_id=subject)
            writer.cohort(site_id=site, dataset_id="test")
    return root


@pytest.fixture
def manifest(cohort: Path) -> Manifest:
    built, failures = scan(cohort)
    assert not failures
    return built


def _subject(path: Path) -> str:
    with medh5.open(path) as sample:
        return sample.identity.subject_id


class TestManifest:
    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
    def test_a_scan_follows_linked_directories_once_ignoring_case_and_appledouble(
        self, tmp_path, sample_path
    ):
        """A symlinked directory was passed over, and `CASE.MEDH5` with it,
        while `._case.medh5` --- a Mac's AppleDouble file --- failed the scan
        as a sample that would not open."""
        root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
        (root / "real").mkdir(parents=True)
        elsewhere.mkdir()
        for name in ("real/a.medh5", "B.MEDH5"):
            (root / name).write_bytes(sample_path.read_bytes())
        (elsewhere / "c.medh5").write_bytes(sample_path.read_bytes())
        (root / "._a.medh5").write_bytes(b"\x00\x05\x16\x07 AppleDouble")
        (root / "linked").symlink_to(elsewhere, target_is_directory=True)
        (root / "real" / "up").symlink_to(root, target_is_directory=True)
        built, failures = scan(root)
        assert failures == ()
        assert [Path(e.path).relative_to(root).as_posix() for e in built] == [
            "B.MEDH5",
            "linked/c.medh5",
            "real/a.medh5",
        ]
        assert [p.name for p in find(root)] == ["B.MEDH5", "c.medh5", "a.medh5"]

    def test_a_scan_reads_metadata_only(self, cohort):
        built, failures = scan(cohort)
        assert len(built) == 6
        assert not failures
        assert built.subjects == ("subj-A", "subj-B", "subj-C", "subj-D")

    def test_an_entry_carries_what_splitting_needs(self, manifest):
        entry = manifest.by_path(
            str(next(iter(sorted(Path(manifest.root).glob("*.medh5")))))
        )
        assert entry is not None
        assert entry.subject_id.startswith("subj-")
        assert entry.site_id in ("site-A", "site-B")
        assert entry.label_set_digest
        assert entry.class_ids == (1, 3)
        assert "CT_tp0" in entry.images

    def test_the_group_id_defaults_to_the_subject(self, manifest):
        for entry in manifest:
            assert entry.group_id == entry.subject_id

    def test_S12_2_an_explicit_group_id_wins(self, tmp_path, label_set):
        path = write_sample(tmp_path / "g.medh5", label_set=label_set)
        with medh5.amend(path) as writer:
            writer.cohort(group_id="family-7")
        assert entries_for(path)[0].group_id == "family-7"

    def test_round_trips_through_json(self, manifest, tmp_path):
        target = manifest.save(tmp_path / "m.json")
        again = Manifest.load(target)
        assert [e.to_json() for e in again] == [e.to_json() for e in manifest]
        assert again.sha256() == manifest.sha256()

    def test_a_save_replaces_the_manifest_whole(self, manifest, tmp_path):
        """A manifest was written in place, so a write cut short --- a full
        disk, a crash --- left it truncated; it is replaced whole now."""
        target = manifest.save(tmp_path / "m.json")
        before = target.stat().st_ino
        assert manifest.save(target) == target
        assert target.stat().st_ino != before, "a new file, not the old rewritten"
        assert Manifest.load(target).sha256() == manifest.sha256()
        assert [p.name for p in tmp_path.glob(".*")] == []

    def test_the_digest_ignores_where_and_when(self, manifest):
        """A cohort scanned on another machine must digest the same."""
        moved = Manifest(
            entries=[
                replace(e, path=f"/elsewhere/{Path(e.path).name}", mtime=0.0, size=0)
                for e in manifest
            ],
            root="/elsewhere",
            generator="medh5 999",
        )
        assert moved.sha256() == manifest.sha256()

    def test_the_digest_survives_writing_the_claims_it_authorises(
        self, manifest, cohort
    ):
        """A digest over content would be stale the instant a claim is written.

        `write_claims` rewrites every file, so a content-covering digest could
        never match a re-scan --- and `dataset check` would report C201 on the
        split it had just made.
        """
        before = manifest.sha256()
        write_claims(make_splits(manifest), manifest)
        after, _ = scan(cohort)
        assert after.sha256() == before
        assert not [f for f in check(after).findings if f.code == "C201"]

    def test_the_digest_changes_with_the_cohort(self, manifest, cohort, label_set):
        smaller = manifest.filter(lambda e: e.site_id == "site-A")
        assert len(smaller) == 3
        assert smaller.sha256() != manifest.sha256()

        write_sample(cohort / "new.medh5", label_set=label_set, sample_id="new")
        wider, _ = scan(cohort)
        assert wider.sha256() != manifest.sha256()

    def test_staleness_is_noticed(self, manifest, cohort):
        assert not manifest.stale()
        victim = sorted(cohort.glob("*.medh5"))[0]
        with medh5.amend(victim) as writer:
            writer.cohort(scanner_id="new")
        assert manifest.stale() == (str(victim),)

    def test_a_broken_file_does_not_abort_the_scan(self, cohort):
        """... and `on_error="warn"`, the default, warns: it returned the
        failures and told nobody."""
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        with pytest.warns(UserWarning, match="1 path"):
            built, failures = scan(cohort)
        assert len(built) == 6
        assert len(failures) == 1 and "broken.medh5" in failures[0]

    def test_strict_mode_raises_instead(self, cohort):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        with pytest.raises(MEDH5Error):
            scan(cohort, on_error="raise")

    def test_a_root_that_cannot_be_listed_is_a_failure(self, tmp_path):
        """A directory that could not be listed was passed over in silence:
        the manifest was short of everything under it, with no failure to say
        so, even on a strict scan."""
        missing = tmp_path / "no-such-cohort"
        with pytest.warns(UserWarning, match="1 path"):
            built, failures = scan(missing)
        assert len(built) == 0
        # In the system's own words: Windows says it cannot find the path.
        with pytest.raises(FileNotFoundError) as listing:
            os.listdir(missing)
        assert len(failures) == 1 and listing.value.strerror in failures[0]
        with pytest.raises(OSError, match="no-such-cohort"):
            scan(missing, on_error="raise")

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
        reason="POSIX file modes, which root ignores",
    )
    def test_a_subdirectory_that_cannot_be_listed_is_a_failure(self, cohort):
        hidden = cohort / "site-C"
        hidden.mkdir()
        hidden.chmod(0)
        try:
            built, failures = scan(cohort)
            assert len(built) == 6
            assert len(failures) == 1
            assert "site-C" in failures[0] and "Permission denied" in failures[0]
            with pytest.raises(OSError, match="site-C"):
                scan(cohort, on_error="raise")
        finally:
            hidden.chmod(0o755)

    def test_an_unknown_error_mode_is_refused(self, cohort):
        """Anything but ``"raise"`` read as ``"warn"``, so a misspelt strict
        scan went on past every broken file."""
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        with pytest.raises(MEDH5ValidationError, match="on_error"):
            scan(cohort, on_error="raises")

    def test_a_collection_fans_out_into_entries(self, cohort, tmp_path):
        shard = tmp_path / "shard.medh5c"
        medh5.pack(sorted(cohort.glob("*.medh5")), shard)
        entries = entries_for(shard)
        assert len(entries) == 6
        assert all(e.key for e in entries)

    def test_find_is_ordered_and_typed(self, cohort):
        found = find(cohort)
        assert found == sorted(found)
        assert all(p.suffix == ".medh5" for p in found)

    def test_an_unknown_field_names_the_alternatives(self, manifest):
        with pytest.raises(MEDH5ValidationError, match="subject_id"):
            manifest[0].field("nonsense")

    def test_dotted_names_reach_the_same_field(self, manifest):
        entry = manifest[0]
        assert entry.field("cohort.site_id") == entry.field("site_id")
        assert entry.field("identity.subject_id") == entry.subject_id

    def test_manifest_fields_are_fields(self, tmp_path: Path):
        from medh5.dataset.manifest import scan

        with Flat.open_writer(tmp_path / "a.medh5"):
            pass
        manifest, _ = scan(tmp_path)
        entry = manifest.entries[0]
        assert entry.field("cohort.site_id") is None
        assert entry.field("subject_id") == "a"
        with pytest.raises(MEDH5ValidationError, match="not a manifest field"):
            entry.field("to_json")


class TestSplits:
    def test_groups_never_straddle_partitions(self, manifest):
        split = make_splits(manifest, seed=7)
        assert not split.leaks()
        placed: dict[str, str] = {}
        for assignment in split.assignments:
            assert placed.setdefault(assignment.group, assignment.partition) == (
                assignment.partition
            )

    def test_S12_3_a_subject_with_two_files_stays_together(self, manifest):
        split = make_splits(manifest, seed=3)
        by_subject: dict[str, set[str]] = {}
        for assignment in split.assignments:
            for path in assignment.entries:
                entry = manifest.by_path(path)
                assert entry is not None
                by_subject.setdefault(entry.subject_id, set()).add(assignment.partition)
        assert all(len(v) == 1 for v in by_subject.values())
        assert sum(len(a.entries) for a in split.assignments) == 6

    def test_S12_3_a_subject_split_across_two_groups_is_refused(self, manifest):
        """`group_id` is declared per file, so two visits can disagree about it.

        The subject is then two groups, they are dealt independently, and the
        same anatomy lands in two partitions. `leaks()` cannot see it --- each
        group really was assigned once --- so it is refused at construction.
        """
        entries = list(manifest.entries)
        subject = entries[0].subject_id
        twin = next(e for e in entries[1:] if e.subject_id == subject)
        moved = replace(twin, group_id="OTHER-COHORT")
        split_manifest = Manifest(
            entries=[e for e in entries if e is not twin] + [moved]
        )
        with pytest.raises(MEDH5ValidationError, match="C204") as caught:
            make_splits(split_manifest, seed=3)
        assert subject in str(caught.value)

    def test_the_same_inputs_give_the_same_split(self, manifest):
        first = make_splits(manifest, seed=11)
        second = make_splits(manifest, seed=11)
        assert first.to_json()["assignments"] == second.to_json()["assignments"]

    def test_a_different_seed_gives_a_different_split(self, manifest):
        a = make_splits(manifest, seed=1).to_json()["assignments"]
        b = make_splits(manifest, seed=2).to_json()["assignments"]
        assert a != b

    def test_dealing_by_deficit_beats_slicing_by_index(self, manifest):
        """Slicing 4 groups at 70/15/15 by index gives train all of them."""
        split = make_splits(manifest, seed=5)
        assert split.counts["train"] < 6
        assert len(split.counts) > 1

    def test_a_partition_that_gets_nothing_says_so(self, manifest):
        """4 indivisible groups cannot fill 0.7/0.15/0.15; that must not be silent."""
        split = make_splits(manifest, seed=5)
        assert split.underfilled == ("test",)
        assert set(split.counts) == {"train", "val"}

    def test_a_cohort_large_enough_fills_everything(self, cohort, label_set):
        for extra in range(6):
            write_sample(
                cohort / f"more-{extra}.medh5",
                label_set=label_set,
                sample_id=f"more-{extra}",
            )
            with medh5.amend(cohort / f"more-{extra}.medh5") as writer:
                writer.identity(subject_id=f"subj-{extra}x")
        built, _ = scan(cohort)
        split = make_splits(built, seed=5)
        assert not split.underfilled
        assert set(split.counts) == {"train", "val", "test"}

    def test_stratification_spreads_a_field(self, manifest):
        split = make_splits(manifest, stratify_by="site_id", seed=4)
        sites = {
            partition: set(counts) for partition, counts in split.balance().items()
        }
        assert any(len(v) > 1 for v in sites.values()), sites

    def test_stratifying_does_not_starve_the_small_partitions(self, manifest):
        """Per-stratum tallies round every small stratum toward train.

        Four groups in two strata, dealt independently, put all four in train:
        each stratum's own 0.7/0.15/0.15 rounds that way twice.  The tally has
        to be global for the ratios to mean anything.
        """
        plain = make_splits(manifest, seed=0)
        stratified = make_splits(manifest, stratify_by="site_id", seed=0)
        assert set(stratified.counts) == set(plain.counts)
        assert stratified.underfilled == plain.underfilled
        assert stratified.counts["train"] < sum(stratified.counts.values())

    def test_each_stratum_is_spread_across_the_partitions(self, cohort, label_set):
        """...and the strata still have to be spread, not merely counted."""
        for extra in range(6):
            path = cohort / f"more-{extra}.medh5"
            write_sample(path, label_set=label_set, sample_id=f"more-{extra}")
            with medh5.amend(path) as writer:
                writer.identity(subject_id=f"subj-{extra}x")
                writer.cohort(site_id="site-A" if extra % 2 else "site-B")
        built, _ = scan(cohort)
        split = make_splits(built, stratify_by="site_id", seed=0)
        balance = split.balance()
        assert len(balance["train"]) > 1, balance

    def test_a_groups_stratum_is_its_majority(self, cohort):
        """A group is indivisible, so a disagreeing group needs one answer."""
        paths = sorted(cohort.glob("*.medh5"))
        with medh5.amend(paths[1]) as writer:
            writer.cohort(site_id="site-Z")
        built, _ = scan(cohort)
        split = make_splits(built, stratify_by="site_id", seed=0)
        strata = {a.group: a.stratum for a in split.assignments}
        assert strata["subj-A"] in ("site-A", "site-Z")

    def test_k_folds_deals_every_group(self, manifest):
        split = make_splits(manifest, k_folds=2, seed=0)
        folds = {a.fold for a in split.assignments}
        assert folds == {0, 1}
        assert all(a.partition == "holdout" for a in split.assignments)

    @pytest.mark.parametrize("k", [5, 10**9])
    def test_more_folds_than_groups_is_refused(self, manifest, k):
        """A 5-fold CV that quietly runs 4 ways is not noticed until later ---
        and listing a billion empty folds took the memory the process had."""
        with pytest.raises(MEDH5ValidationError, match="at most 4-fold"):
            make_splits(manifest, k_folds=k, seed=0)
        assert len(make_splits(manifest, k_folds=4, seed=0).assignments) == 4

    def test_enough_groups_fills_every_fold(self, manifest):
        assert make_splits(manifest, k_folds=2, seed=0).empty_folds == ()

    def test_a_ratio_split_has_no_folds_to_report(self, manifest):
        assert make_splits(manifest, seed=0).empty_folds == ()

    def test_ratios_do_not_apply_to_folds(self, manifest):
        """With k_folds the ratios went unused and unchecked into the split."""
        with pytest.raises(MEDH5ValidationError, match="ratios and k_folds"):
            make_splits(manifest, k_folds=2, ratios={"train": -1.0, "bogus": 0.5})
        assert make_splits(manifest, k_folds=2).ratios == {}

    def test_a_group_in_two_folds_is_a_leak(self, manifest):
        """Every k-fold assignment is `holdout`, so a group or an entry in two
        folds was in one partition as `leaks()` compared them."""
        from medh5.dataset.split import Assignment, Split

        split = Split(
            "cv",
            manifest.sha256(),
            [
                Assignment("g1", "holdout", fold=0, entries=("a.medh5",)),
                Assignment("g1", "holdout", fold=1, entries=("a.medh5",)),
                Assignment("g2", "holdout", fold=1, entries=("b.medh5",)),
            ],
            k_folds=2,
        )
        assert split.leaks() == ("a.medh5", "g1")
        assert make_splits(manifest, k_folds=2, seed=0).leaks() == ()

    def test_k_folds_below_two_is_refused(self, manifest):
        with pytest.raises(MEDH5ValidationError, match="at least 2"):
            make_splits(manifest, k_folds=1)

    def test_an_empty_manifest_is_refused(self):
        with pytest.raises(MEDH5ValidationError, match="empty"):
            make_splits(Manifest())

    def test_an_unknown_partition_is_refused(self, manifest):
        with pytest.raises(MEDH5ValidationError, match="unknown partition"):
            make_splits(manifest, ratios={"trian": 1.0})

    def test_ratios_are_normalised(self, manifest):
        split = make_splits(manifest, ratios={"train": 8, "val": 2}, seed=0)
        assert set(split.counts) <= {"train", "val"}
        assert sum(split.counts.values()) == 6

    @pytest.mark.parametrize(
        "ratios",
        [
            {"train": float("nan"), "test": 1.0},
            {"train": float("inf"), "test": 1.0},
            {"train": -1.0, "val": 2.0},
        ],
    )
    def test_a_ratio_is_a_finite_number_of_at_least_zero(self, manifest, ratios):
        """NaN and infinity panicked the engine (a ``BaseException`` no
        ``except Exception`` catches); a negative share was written down."""
        with pytest.raises(MEDH5ValidationError, match="finite number"):
            make_splits(manifest, ratios=ratios)

    def test_finite_shares_whose_sum_overflows_still_split(self, manifest):
        split = make_splits(manifest, ratios={"train": 1e308, "test": 1e308}, seed=0)
        assert set(split.counts) == {"train", "test"}
        assert sum(split.counts.values()) == 6

    def test_the_command_line_refuses_a_ratio_that_is_not_a_number(
        self, cohort, tmp_path, capsys
    ):
        from medh5.cli import EXIT_ERROR, main

        manifest_path = tmp_path / "m.json"
        assert main(["dataset", "index", str(cohort), "-o", str(manifest_path)]) == 0
        code = main(
            ["dataset", "split", str(manifest_path), "--ratios", "train=nan,test=1"]
        )
        assert code == EXIT_ERROR
        assert "finite number" in capsys.readouterr().err

    def test_the_split_records_the_manifest_it_came_from(self, manifest):
        split = make_splits(manifest)
        assert split.manifest_sha256 == manifest.sha256()

    def test_S12_3_claims_carry_the_manifest_digest(self, manifest):
        split = make_splits(manifest, seed=0)
        written = write_claims(split, manifest, assigned_by="test")
        assert len(written) == 6
        with medh5.open(written[0]) as sample:
            claim = sample.document.splits[0]
            assert claim.manifest_sha256 == manifest.sha256()
            assert claim.partition in ("train", "val", "test")

    def test_re_splitting_replaces_the_claim_it_does_not_stack(self, manifest):
        write_claims(make_splits(manifest, seed=1), manifest)
        write_claims(make_splits(manifest, seed=2), manifest)
        with medh5.open(manifest[0].path) as sample:
            claims = [c for c in sample.document.splits if c.set_id == "default"]
            assert len(claims) == 1

    def test_a_second_set_lives_alongside_the_first(self, manifest):
        write_claims(make_splits(manifest, set_id="cv5"), manifest)
        write_claims(make_splits(manifest, set_id="site-holdout"), manifest)
        with medh5.open(manifest[0].path) as sample:
            assert {c.set_id for c in sample.document.splits} == {"cv5", "site-holdout"}

    def test_a_k_fold_claim_needs_a_validation_fold(self, manifest):
        split = make_splits(manifest, k_folds=2)
        with pytest.raises(MEDH5ValidationError, match="--fold"):
            write_claims(split, manifest)
        written = write_claims(split, manifest, fold=0)
        with medh5.open(written[0]) as sample:
            claim = sample.document.splits[0]
            assert claim.partition in ("train", "val")
            assert claim.fold in (0, 1)

    def test_a_sample_inside_a_collection_is_refused_not_skipped(
        self, cohort, tmp_path
    ):
        shard = tmp_path / "shard.medh5c"
        medh5.pack(sorted(cohort.glob("*.medh5")), shard)
        built, _ = scan(shard)
        with pytest.raises(MEDH5ValidationError, match="unpack"):
            write_claims(make_splits(built), built)

    def test_S12_3_members_of_one_collection_are_named_one_by_one(
        self, cohort, tmp_path
    ):
        """A split's entries held the file's path alone, so the members of one
        collection placed in two partitions put the whole collection in both
        per-partition file lists."""
        shard = tmp_path / "shard.medh5c"
        medh5.pack(sorted(cohort.glob("*.medh5")), shard)
        built, _ = scan(shard)
        split = make_splits(built, ratios={"train": 0.5, "test": 0.5}, seed=0)
        placed = {name: split.paths(name) for name in ("train", "test")}
        assert all(placed.values()), placed
        named = [entry for entries in placed.values() for entry in entries]
        assert sorted(named) == sorted(f"{e.path}::{e.key}" for e in built)
        assert not set(placed["train"]) & set(placed["test"])
        assert split.leaks() == ()
        for entry in built:
            assert f"{entry.path}::{entry.key}" in placed[split.partition_of(entry)]

    def test_S12_3_a_manifest_the_split_was_not_made_from_is_refused(
        self, manifest, cohort, label_set
    ):
        """A split made before the cohort grew stamped its old digest --- stale
        on arrival (C201) --- and passed over the new sample in silence."""
        split = make_splits(manifest, seed=0)
        write_sample(cohort / "new.medh5", label_set=label_set, sample_id="new")
        wider, _ = scan(cohort)
        with pytest.raises(MEDH5ValidationError, match="new.medh5"):
            write_claims(split, wider)
        assert all(self._claims(entry.path) == [] for entry in wider)
        site_a = manifest.filter(lambda e: e.site_id == "site-A")
        assert len(write_claims(split, site_a)) == 3

    def test_partition_of_follows_the_assignments_without_rebuilding_them(
        self, manifest
    ):
        """Each call rebuilt the whole split in the engine: the documented
        per-entry loop took 34 s at 3,000 subjects."""
        split = make_splits(manifest, k_folds=2, seed=0)
        for entry in manifest:
            doc, fields = split._doc(), entry._fields()
            assert split.partition_of(entry) == (
                _core.dataset_split_partition_of(doc, fields)
            )
            assert split.fold_of(entry) == _core.dataset_split_fold_of(doc, fields)
        entry = manifest[0]
        moved = next(a for a in split.assignments if a.group == entry.group_id)
        moved.fold = 1 - (moved.fold or 0)
        assert split.fold_of(entry) == moved.fold
        split.assignments = []
        assert split.partition_of(entry) is None

    def test_S12_3_an_entry_in_two_partitions_is_a_leak(self, manifest):
        """`leaks()` checked groups alone: a loaded split placing one file
        under two groups in two partitions passed it."""
        from medh5.dataset.split import Assignment, Split

        split = Split(
            "s",
            manifest.sha256(),
            [
                Assignment("g1", "train", entries=("a.medh5", "b.medh5")),
                Assignment("g2", "test", entries=("b.medh5",)),
            ],
        )
        assert split.leaks() == ("b.medh5",)

    @staticmethod
    def _claims(path: str) -> list[tuple[str, str]]:
        with medh5.open(path) as sample:
            return [(c.set_id, c.partition) for c in sample.document.splits]

    @pytest.mark.parametrize("fold", [2, 7, -1])
    def test_S12_3_a_fold_the_split_lacks_is_refused_before_any_write(
        self, manifest, fold
    ):
        """Fold 7 of a 2-fold split matched no assignment, so every file was
        written as `train` and the split had no validation partition."""
        split = make_splits(manifest, k_folds=2, seed=0)
        with pytest.raises(MEDH5ValidationError, match="not a fold of this 2-fold"):
            write_claims(split, manifest, fold=fold)
        assert all(self._claims(entry.path) == [] for entry in manifest)

    def test_S12_3_a_file_amending_would_refuse_stops_the_write_before_it_starts(
        self, cohort
    ):
        """Only the fold and collection membership were checked first: a later
        minor version, met at the last file, was refused after every file
        before it had claimed the split."""
        import h5py

        last = sorted(cohort.glob("*.medh5"))[-1]
        with h5py.File(last, "r+") as f:
            f.attrs["medh5_version"] = "1.9"
        built, failures = scan(cohort)
        assert not failures
        with pytest.raises(medh5.errors.MEDH5VersionError, match="1.9"):
            write_claims(make_splits(built, seed=0), built)
        assert all(self._claims(entry.path) == [] for entry in built)

    def test_S12_3_a_file_the_validator_rejects_stops_the_write(self, cohort):
        """A commit refuses to write a file the validator rejects; that was
        met file by file too."""
        import h5py

        last = sorted(cohort.glob("*.medh5"))[-1]
        with h5py.File(last, "r+") as f:
            doc = json.loads(f["meta"][()])
            doc["provenance"]["activities"].append(
                {"id": "ghost-run", "type": "import", "agent": "ghost"}
            )
            del f["meta"]
            f.create_dataset("meta", data=json.dumps(doc))
        built, failures = scan(cohort)
        assert not failures
        with pytest.raises(MEDH5Error, match="fails validation") as exc:
            write_claims(make_splits(built, seed=0), built)
        assert exc.value.code == "E605"
        assert all(self._claims(entry.path) == [] for entry in built)

    def test_S12_3_a_file_failing_its_digest_stops_the_write_before_it_starts(
        self, cohort
    ):
        """A commit refuses to restamp a digest its source already failed
        (E701); that was met at the commit of the file it fell on, after every
        file before it had claimed the split."""
        import h5py

        last = sorted(cohort.glob("*.medh5"))[-1]
        with h5py.File(last, "r+") as f:
            f["images/CT_tp0"][0, 0, 0] += 1
        built, failures = scan(cohort)
        assert not failures
        with pytest.raises(
            MEDH5Error, match=r"does not match its digest \(images/CT_tp0\)"
        ) as exc:
            write_claims(make_splits(built, seed=0), built)
        assert exc.value.code == "E701"
        assert all(self._claims(entry.path) == [] for entry in built)

    def test_a_split_that_leaks_is_not_written(self, manifest):
        """An entry in two partitions was written as whichever came last."""
        from medh5.dataset.split import Assignment, Split

        first, second = manifest[0].path, manifest[1].path
        split = Split(
            "s",
            manifest.sha256(),
            [
                Assignment("g1", "train", entries=(first, second)),
                Assignment("g2", "test", entries=(second,)),
            ],
        )
        with pytest.raises(MEDH5ValidationError, match="more than one partition"):
            write_claims(split, manifest)
        assert all(self._claims(entry.path) == [] for entry in manifest)

    def test_a_fold_is_not_ignored(self, manifest):
        """A fold for a ratio split, or one no group is in, was written as if
        it named the validation set."""
        from medh5.dataset.split import Split

        with pytest.raises(MEDH5ValidationError, match="has none"):
            write_claims(make_splits(manifest, seed=0), manifest, fold=0)
        legacy = make_splits(manifest, k_folds=2, seed=0).to_json()
        legacy["k_folds"] = 5
        with pytest.raises(MEDH5ValidationError, match="holds no group"):
            write_claims(Split.from_json(legacy), manifest, fold=3)
        assert all(self._claims(entry.path) == [] for entry in manifest)

    def test_S12_3_a_refused_write_leaves_every_file_as_it_was(self, cohort):
        """The member of a collection was refused after the files before it
        had been amended: half a cohort claimed the split."""
        members = sorted(cohort.glob("case-[45].medh5"))
        medh5.pack(members, cohort / "zz-shard.medh5")
        for member in members:
            member.unlink()
        built, failures = scan(cohort)
        assert not failures
        assert [e.key is not None for e in built][-1], "the member must come last"
        with pytest.raises(MEDH5ValidationError, match="unpack"):
            write_claims(make_splits(built, seed=0), built)
        plain = [entry.path for entry in built if entry.key is None]
        assert len(plain) == 4
        assert all(self._claims(path) == [] for path in plain)


class TestStats:
    def test_welford_merges_exactly(self):
        rng = np.random.default_rng(0)
        data = rng.normal(50, 12, 5000)
        whole = Moments()
        whole.update(data)
        parts = Moments()
        for chunk in np.array_split(data, 7):
            piece = Moments()
            piece.update(chunk)
            parts.merge(piece)
        assert parts.count == whole.count
        assert parts.mean == pytest.approx(whole.mean)
        assert parts.std == pytest.approx(whole.std)

    def test_a_merge_weights_by_voxels_not_by_file(self):
        """Averaging per-file means would give both files equal say."""
        small, large = Moments(), Moments()
        small.update(np.zeros(10))
        large.update(np.full(1000, 100.0))
        total = Moments()
        total.merge(small)
        total.merge(large)
        assert total.mean == pytest.approx(100 * 1000 / 1010)

    def test_stats_match_a_direct_computation(self, sample_path):
        result = stats_for(sample_path)
        with medh5.open(sample_path) as sample:
            raw = sample.images["CT_tp0"].read()
        moments = result.images["CT_tp0"]
        assert moments.count == raw.size
        assert moments.mean == pytest.approx(float(raw.mean()), rel=1e-9)
        assert moments.std == pytest.approx(float(raw.std()), rel=1e-6)

    def test_class_counts_come_from_the_index_when_it_is_fresh(self, longitudinal_path):
        result = stats_for(longitudinal_path)
        with medh5.open(longitudinal_path) as sample:
            direct = sample.annotations["organs_tp0"].voxel_counts()
        for class_id, count in direct.items():
            assert result.classes[class_id].voxels >= count

    def test_a_stale_index_is_not_trusted(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "s.medh5", label_set=label_set, masks=masks, index=True
        )
        smaller = {1: block(SHAPE, (2, 2, 2), 2)}
        with medh5.amend(path) as writer:
            writer.remove_annotation("organs_tp0")
            writer.add_segmentation("organs_tp0", grid="ct_tp0", masks=smaller)
        result = stats_for(path)
        assert result.classes[1].voxels == int(smaller[1].sum())

    def test_S11_3_an_unexamined_class_is_not_a_zero(self, tmp_path, label_set):
        """Prevalence is over the samples that looked, not over all of them."""
        looked = write_sample(
            tmp_path / "a.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2), 4)},
            annotated=["liver", "spleen"],
        )
        did_not = write_sample(tmp_path / "b.medh5", label_set=label_set)
        result = compute_stats([looked, did_not])
        assert result.classes[2].examined_in == 1
        assert result.classes[2].present_in == 0
        assert result.classes[1].prevalence == 1.0

    def test_S11_3_prevalence_is_over_the_samples_that_looked(
        self, tmp_path, label_set
    ):
        """A class drawn where nobody looked for it counted as found, over
        only the samples that looked: three drawings and one examination
        that found nothing made its prevalence 3.0."""
        paths = []
        for i in range(3):
            path = tmp_path / f"drawn-{i}.medh5"
            masks = {1: block(SHAPE, (2, 2, 2), 4), 2: block(SHAPE, (2, 14, 2), 4)}
            write_sample(path, label_set=label_set, masks=masks, annotated=["liver"])
            paths.append(path)
        looked = tmp_path / "looked.medh5"
        write_sample(
            looked,
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2), 4)},
            annotated=["liver", "spleen"],
        )
        result = compute_stats([*paths, looked])
        spleen = result.classes[2]
        assert (spleen.present_in, spleen.examined_in) == (0, 1)
        assert spleen.prevalence == 0.0
        assert spleen.voxels == 3 * 4**3

    def test_a_collection_member_is_measured_by_its_locator(
        self, cohort, tmp_path, capsys
    ):
        """A split names a collection's members `path::key`, and `dataset
        stats` opened every entry's path as a sample, which a shard is not."""
        shard = tmp_path / "shard.medh5c"
        medh5.pack(sorted(cohort.glob("case-[01].medh5")), shard)
        built, _ = scan(shard)
        located = [f"{e.path}::{e.key}" for e in built]
        result = compute_stats(located)
        assert (result.samples, result.failures) == (2, ())
        manifest_path = tmp_path / "shard.json"
        built.save(manifest_path)
        assert main(["dataset", "stats", str(manifest_path)]) == 0
        assert "unreadable" not in capsys.readouterr().out

    def test_workers_agree_with_one_process(self, cohort):
        paths = sorted(str(p) for p in cohort.glob("*.medh5"))
        serial = compute_stats(paths)
        parallel = compute_stats(paths, workers=2)
        assert parallel.samples == serial.samples
        assert parallel.images["CT_tp0"].mean == pytest.approx(
            serial.images["CT_tp0"].mean
        )

    def test_an_unreadable_file_is_reported_not_raised(self, cohort):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        result = compute_stats(sorted(str(p) for p in cohort.glob("*.medh5")))
        assert result.samples == 6
        assert len(result.failures) == 1

    def test_normalization_is_ready_to_use(self, sample_path):
        mean, std = stats_for(sample_path).normalization("CT_tp0")
        assert std > 0
        mean, std = stats_for(sample_path).normalization("nope")
        assert (mean, std) == (0.0, 1.0)

    def test_class_weights_normalise_to_a_mean_of_one(self, cohort):
        result = compute_stats(sorted(str(p) for p in cohort.glob("*.medh5")))
        weights = result.class_weights()
        assert len(weights) >= 2
        assert sum(weights.values()) == pytest.approx(len(weights))
        rare = min(result.classes.values(), key=lambda s: s.voxels)
        common = max(result.classes.values(), key=lambda s: s.voxels)
        assert weights[rare.class_id] > weights[common.class_id]

    def test_an_unknown_scheme_is_named(self, sample_path):
        with pytest.raises(MEDH5Error, match="inverse_frequency"):
            stats_for(sample_path).class_weights(scheme="magic")

    @staticmethod
    def _measured(path: Path, image: Any) -> Path:
        with medh5.create(path, sample_id=path.stem, subject_id=path.stem) as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("PT", image, grid="g", modality="PT")
        return path

    def test_a_non_finite_voxel_is_counted_and_left_out(self, tmp_path):
        """A NaN made every moment NaN, the extremes depend on the order of the
        files, and the `null` it was written as reloaded as mean 0, std 0."""
        odd = np.arange(64, dtype=np.float32).reshape(4, 4, 4) - 10
        odd[0, 0, 0], odd[1, 1, 1] = np.nan, np.inf
        flat = np.full((4, 4, 4), 5.0, np.float32)
        paths = [
            self._measured(tmp_path / "odd.medh5", odd),
            self._measured(tmp_path / "flat.medh5", flat),
        ]
        finite = np.concatenate([odd[np.isfinite(odd)], flat.ravel()]).astype(float)
        forward, backward = compute_stats(paths), compute_stats(paths[::-1])
        for result in (forward, backward):
            pt = result.images["PT"]
            assert (pt.count, pt.nonfinite) == (finite.size, 2)
            assert (pt.minimum, pt.maximum) == (finite.min(), finite.max())
            assert pt.mean == pytest.approx(finite.mean())
            assert pt.std == pytest.approx(finite.std())
        doc = json.loads(json.dumps(forward.to_json()))
        assert doc["images"]["PT"]["nonfinite"] == 2
        back = DatasetStats.from_json(doc).normalization("PT")
        assert back == pytest.approx(forward.normalization("PT"))
        doc["images"]["PT"]["mean"] = None
        with pytest.raises(ValueError, match="'PT'.*no mean"):
            DatasetStats.from_json(doc)

    def test_S5_1_a_class_id_meaning_two_things_is_refused(self, tmp_path, capsys):
        """1 was "liver" in one label set and "spleen" in another, and both
        were counted as one class, whose voxels fed one loss weight."""

        def labelled(name: str, key: str) -> Path:
            vocabulary = LabelSet(
                f"vocab-{key}", version="1", classes=[LabelClass(1, key, key)]
            )
            masks = {1: block(SHAPE, (2, 2, 2), 4)}
            path = tmp_path / f"{name}.medh5"
            return write_sample(path, label_set=vocabulary, masks=masks)

        a, b, c = (
            labelled("a", "liver"),
            labelled("b", "spleen"),
            labelled("c", "liver"),
        )
        same = compute_stats([a, c])
        assert (same.classes[1].key, same.classes[1].voxels) == ("liver", 2 * 4**3)
        assert same.to_json()["classes"][0]["key"] == "liver"
        for paths, named in (([a, b], "b.medh5"), ([b, c, a], "c.medh5")):
            for workers in (1, 2):
                with pytest.raises(MEDH5ValidationError, match="C102") as exc:
                    compute_stats(paths, workers=workers)
                assert named in str(exc.value)
        merged = compute_stats([a])
        with pytest.raises(MEDH5ValidationError, match="'liver' in these"):
            merged.merge(compute_stats([b]))
        manifest = tmp_path / "m.json"
        Manifest(entries=entries_for(a) + entries_for(b)).save(manifest)
        capsys.readouterr()
        assert main(["dataset", "stats", str(manifest)]) == 1
        assert "class 1 is 'spleen'" in capsys.readouterr().err

    def test_stats_are_the_plane_by_plane_fold_to_the_bit(self, tmp_path):
        """A read per plane decompressed each chunk once per plane it holds;
        reading a chunk's planes at once changes no bit of the moments."""
        shape = (24, 40, 36)  # 64 KiB and more is chunked
        stored = np.random.default_rng(5).integers(-999, 999, shape).astype(np.int16)
        fov = np.ones(shape, bool)
        fov[:, :2] = False
        path = tmp_path / "s.medh5"
        with medh5.create(path, sample_id="s") as w:
            w.add_grid(
                "g", shape=shape, spacing=(1.0, 1.0, 1.0), chunk_hint=(5, 40, 36)
            )
            w.add_mask("fov", fov, grid="g")
            w.add_image(
                "CT",
                stored,
                grid="g",
                modality="CT",
                rescale_slope=0.37,
                rescale_intercept=-12.5,
                valid_mask="fov",
            )
            w.add_image("raw", stored, grid="g", modality="CT")
        with medh5.open(path) as sample:
            assert sample.images["CT"].chunks[0] > 1, "a read must span planes"
            for stride in (1, 4):
                for physical in (True, False):
                    expected = {}
                    for key in ("CT", "raw"):
                        moments = Moments()
                        for z in range(0, shape[0], stride):
                            roi = (slice(z, z + 1), slice(None), slice(None))
                            plane = np.asarray(
                                sample.images[key].read(roi, physical=physical), float
                            )
                            valid = sample.valid_region(key, roi)
                            moments.update(plane[np.broadcast_to(valid, plane.shape)])
                        expected[key] = moments
                    found = stats_for(path, sample_stride=stride, physical=physical)
                    assert found.images == expected

    @pytest.mark.parametrize("stride", [0, -1])
    def test_a_stride_below_one_is_refused(self, sample_path, stride):
        """0 read every plane, as 1 does, and -1 raised `OverflowError`."""
        with pytest.raises(MEDH5ValidationError, match="sample_stride"):
            stats_for(sample_path, sample_stride=stride)
        with pytest.raises(MEDH5ValidationError, match="sample_stride"):
            compute_stats([sample_path], sample_stride=stride)

    def test_an_unreadable_file_fails_the_command_and_is_named_on_stderr(
        self, cohort, tmp_path, capsys
    ):
        """`dataset stats` exited 0 over part of the cohort, and listed what it
        could not read on stdout, among the numbers."""
        manifest = tmp_path / "m.json"
        assert main(["dataset", "index", str(cohort), "-o", str(manifest)]) == 0
        victim = sorted(cohort.glob("*.medh5"))[0]
        victim.write_bytes(b"not hdf5")
        capsys.readouterr()
        assert main(["dataset", "stats", str(manifest), "--json"]) == 1
        out = capsys.readouterr()
        payload = json.loads(out.out)
        assert (payload["samples"], len(payload["failures"])) == (5, 1)
        assert f"medh5: unreadable: {victim}" in out.err

    def test_stats_round_trip_through_json(self, sample_path):
        result = stats_for(sample_path)
        again = type(result).from_json(json.loads(json.dumps(result.to_json())))
        assert again.images["CT_tp0"].mean == pytest.approx(
            result.images["CT_tp0"].mean
        )
        assert again.classes.keys() == result.classes.keys()


class TestCohortCheck:
    def test_a_clean_cohort_passes(self, manifest):
        report = check(manifest)
        assert report.ok
        assert not report.errors

    def test_C102_a_class_id_meaning_two_things_is_an_error(self, cohort, tmp_path):
        other = LabelSet(
            "other-v1",
            version="1.0.0",
            classes=[LabelClass(1, "kidney", "Kidney"), LabelClass(3, "cyst", "Cyst")],
        )
        write_sample(
            cohort / "odd.medh5",
            label_set=other,
            masks={1: block(SHAPE, (2, 2, 2), 4)},
            sample_id="odd",
        )
        built, _ = scan(cohort)
        report = check(built)
        codes = {f.code for f in report.findings}
        assert "C101" in codes and "C102" in codes
        assert not report.ok

    def _with_vocabulary(self, cohort: Path, name: str, label_set: LabelSet) -> None:
        write_sample(
            cohort / f"{name}.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2), 4)},
            sample_id=name,
        )

    @staticmethod
    def _bumped(label_set: LabelSet) -> LabelSet:
        """The same classes under the next version."""
        return LabelSet(label_set.id, version="2.0.0", classes=list(label_set.classes))

    def test_C102_two_versions_that_agree_on_a_class_do_not_clash(
        self, cohort, label_set
    ):
        """Keyed on `id@version`, a version bump that kept every class was a
        C102 error on each id the two versions share."""
        self._with_vocabulary(cohort, "v2", self._bumped(label_set))
        built, _ = scan(cohort)
        report = check(built)
        assert "C101" in {f.code for f in report.warnings}
        assert "C102" not in {f.code for f in report.findings}
        assert report.ok

    def test_C102_one_id_and_version_with_two_meanings_clashes(self, cohort):
        """Keyed on `id@version`, two vocabularies under one name and version
        were one label set, and class 1 meaning two organs passed."""
        relabelled = LabelSet(
            "test-v1",
            version="1.0.0",
            classes=[LabelClass(1, "tumour", "Tumour"), LabelClass(3, "lesion", "L")],
        )
        self._with_vocabulary(cohort, "relabelled", relabelled)
        built, _ = scan(cohort)
        report = check(built)
        clash = next(f for f in report.errors if f.code == "C102")
        assert "[1]" in clash.message
        assert "'liver'" in clash.message and "'tumour'" in clash.message
        assert entries_for(cohort / "relabelled.medh5")[0].class_keys == {1: "tumour"}

    def test_C102_a_manifest_without_class_keys_compares_label_sets(
        self, cohort, label_set
    ):
        """A manifest indexed before 2.0 records no class keys: nothing can
        confirm two vocabularies agree, so a shared id is still reported."""
        self._with_vocabulary(cohort, "v2", self._bumped(label_set))
        built, _ = scan(cohort)
        legacy = Manifest(entries=[replace(e, class_keys={}) for e in built])
        assert "C102" in {f.code for f in check(legacy).errors}

    def test_C103_a_sample_without_a_label_set_is_flagged(self, cohort):
        write_sample(cohort / "bare.medh5", sample_id="bare")
        built, _ = scan(cohort)
        assert "C103" in {f.code for f in check(built).findings}

    def test_C201_a_claim_from_another_manifest_is_an_error(self, manifest, cohort):
        write_claims(make_splits(manifest), manifest)
        write_sample(cohort / "new.medh5", sample_id="new")
        wider, _ = scan(cohort)
        report = check(wider)
        assert "C201" in {f.code for f in report.errors}

    def test_C201_counts_samples_not_claims(self, cohort):
        """A sample with two stale claims was counted, and listed, twice."""
        for path in sorted(cohort.glob("*.medh5"))[:2]:
            with medh5.amend(path) as writer:
                for set_id in ("s1", "s2"):
                    writer.split(
                        set_id=set_id, partition="train", manifest_sha256="0" * 64
                    )
        built, _ = scan(cohort)
        stale = next(f for f in check(built).errors if f.code == "C201")
        assert stale.message.startswith("2 sample(s)")
        assert len(stale.where) == len(set(stale.where)) == 2

    def test_C202_a_unit_in_two_folds_is_a_leak(self, manifest, cohort):
        """k-fold claims are `train` in every fold but the validation one, so a
        subject in folds 0 and 3 read `train` twice and was no leak."""
        a, b = [p for p in sorted(cohort.glob("*.medh5")) if _subject(p) == "subj-A"]
        digest = manifest.sha256()
        for path, fold in ((a, 0), (b, 3)):
            with medh5.amend(path) as writer:
                writer.split(
                    set_id="cv", partition="train", fold=fold, manifest_sha256=digest
                )
        built, _ = scan(cohort)
        assert "C202" in {f.code for f in check(built).errors}
        with medh5.amend(b) as writer:
            writer.split(set_id="cv", partition="train", fold=0, manifest_sha256=digest)
        built, _ = scan(cohort)
        assert "C202" not in {f.code for f in check(built).findings}

    def test_C202_a_subject_in_two_partitions_is_caught(self, manifest, cohort):
        """The leak splitting is designed to prevent, checked independently."""
        split = make_splits(manifest)
        write_claims(split, manifest)
        paths = sorted(cohort.glob("*.medh5"))
        a, b = [p for p in paths if _subject(p) == "subj-A"]
        with medh5.open(a) as sample:
            mine = sample.document.splits[0]
        other = "test" if mine.partition != "test" else "train"
        with medh5.amend(b) as writer:
            writer.split(
                set_id=mine.set_id,
                partition=other,
                manifest_sha256=mine.manifest_sha256,
            )
        built, _ = scan(cohort)
        report = check(built)
        assert "C202" in {f.code for f in report.errors}

    def test_C203_partial_claims_are_a_warning(self, manifest, cohort):
        split = make_splits(manifest)
        subset = manifest.filter(lambda e: e.site_id == "site-A")
        write_claims(split, subset)
        built, _ = scan(cohort)
        assert "C203" in {f.code for f in check(built).warnings}

    def test_C301_partial_coverage_is_reported_with_the_numbers(
        self, cohort, label_set
    ):
        write_sample(
            cohort / "partial.medh5",
            label_set=label_set,
            masks={2: block(SHAPE, (2, 14, 2), 4)},
            annotated=["spleen"],
            sample_id="partial",
        )
        built, _ = scan(cohort)
        report = check(built)
        assert "C301" in {f.code for f in report.findings}
        assert report.coverage[2]["examined_in"] == 1
        assert report.coverage[1]["of"] == 7

    def test_C302_a_class_named_and_examined_nowhere(self, manifest, cohort, label_set):
        """`present_in` counted the samples that *named* a class, which every
        sample examining it does, so C302 could never fire.  A manifest knows
        naming and examination; occurrence is `compute_stats`'s."""
        assert "C302" not in {f.code for f in check(manifest).findings}
        write_sample(
            cohort / "drawn.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2), 4), 2: block(SHAPE, (2, 14, 2), 4)},
            annotated=["liver"],
            sample_id="drawn",
        )
        built, _ = scan(cohort)
        report = check(built)
        assert report.coverage[2] == {"examined_in": 0, "named_in": 1, "of": 7}
        finding = next(f for f in report.findings if f.code == "C302")
        assert finding.severity == "info" and "[2]" in finding.message

    def test_C401_a_changed_file_is_noticed(self, manifest, cohort):
        victim = sorted(cohort.glob("*.medh5"))[0]
        with medh5.amend(victim) as writer:
            writer.cohort(scanner_id="changed")
        assert "C401" in {f.code for f in check(manifest).findings}

    def test_deep_checks_the_content_id_not_the_mtime(self, manifest, cohort):
        """Touching a file must not read as a change; editing it must."""
        victim = Path(sorted(cohort.glob("*.medh5"))[0])
        victim.touch()
        shallow = check(manifest)
        assert "C401" in {f.code for f in shallow.findings}
        rescanned, _ = scan(cohort)
        deep = check(rescanned, deep=True)
        assert "C401" not in {f.code for f in deep.findings}

    def test_C402_a_missing_file_is_an_error(self, manifest, cohort):
        sorted(cohort.glob("*.medh5"))[0].unlink()
        report = check(manifest)
        assert "C402" in {f.code for f in report.errors}

    def test_C501_a_partly_deidentified_cohort_is_flagged(self, cohort):
        path = cohort / "identified.medh5"
        with medh5.create(path, sample_id="identified", subject_id="subj-E") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1, 1, 1))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        built, _ = scan(cohort)
        assert "C501" in {f.code for f in check(built).warnings}

    def test_the_report_round_trips_and_formats(self, manifest):
        report = check(manifest)
        payload = json.loads(json.dumps(report.to_json()))
        assert payload["samples"] == 6
        assert "OK" in report.format()

    def test_a_collection_is_one_file_of_named_members(self, cohort, tmp_path):
        """Findings named the shard once per member, never the member, and
        counted the members of one missing shard as missing files."""
        bare = [
            write_sample(tmp_path / f"bare-{n}.medh5", sample_id=f"bare-{n}")
            for n in "ab"
        ]
        shard = tmp_path / "shard.medh5c"
        medh5.pack(bare, shard)
        built = Manifest(entries=scan(cohort)[0].entries + scan(shard)[0].entries)
        members = sorted(f"{e.path}::{e.key}" for e in built if e.key)
        assert len(members) == 2
        unlabelled = next(f for f in check(built).findings if f.code == "C103")
        assert sorted(unlabelled.where) == members
        shard.unlink()
        missing = next(f for f in check(built).errors if f.code == "C402")
        assert missing.where == (str(shard),)
        assert missing.message.startswith("1 manifest file(s)")

    def test_C202_groups_by_the_grouping_key(self, tmp_path: Path):
        """`dataset check` grouped by subject; `medh5 splits` by the §12.2 key.

        A family or longitudinal group straddling two partitions was a LEAK in
        one tool and clean in the other, on the same files.
        """
        from medh5.dataset.check import check
        from medh5.dataset.manifest import scan

        root = tmp_path / "cohort"
        root.mkdir()
        for name, subject, partition in (
            ("a", "subj-A", "train"),
            ("b", "subj-B", "test"),
        ):
            path = root / f"{name}.medh5"
            with medh5.create(path, sample_id=name, subject_id=subject) as w:
                w.cohort(group_id="family-7")
                w.add_grid("g", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0))
                w.add_image(
                    "CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT"
                )
                w.split(set_id="cv5", partition=partition)
        manifest, _ = scan(root)
        report = check(manifest)
        finding = next(f for f in report.errors if f.code == "C202")
        assert "family-7" in finding.where
        assert "subj-A" in finding.message and "subj-B" in finding.message


class TestCohortWorkflow:
    """The command sequence the cohort guide documents, end to end."""

    def test_S12_3_the_manifest_records_claims_once_it_is_re_indexed(
        self, cohort, tmp_path, capsys
    ):
        """`--write-claims` writes into the files, not the manifest, so
        `stats --partition` right after it found no claim --- and told the
        caller to run the `--write-claims` they had just run."""
        manifest = str(tmp_path / "cohort.json")
        index = ["dataset", "index", str(cohort), "-o", manifest]
        assert main(index) == 0
        split = ["dataset", "split", manifest, "--k-folds", "2", "--set-id", "cv"]
        assert main([*split, "--write-claims", "--fold", "0"]) == 0
        stats = ["dataset", "stats", manifest, "--partition", "train", "--set-id", "cv"]
        capsys.readouterr()
        assert main(stats) == 1
        assert "re-run `medh5 dataset index`" in capsys.readouterr().err
        assert main(index) == 0
        assert main(stats) == 0
        assert main(["dataset", "check", manifest, "--deep"]) == 0


class TestStatisticsArePhysical:
    """§4.2: the loaders read `stored × slope + intercept`; so do the statistics."""

    SLOPE, INTERCEPT = 2.0, -1024.0

    def _write(self, path: Path) -> Any:
        stored = np.full(Hierarchy.SHAPE, 100, dtype=np.int16)
        stored[0] = 90  # a spread, so std is not trivially zero
        with medh5.create(path, sample_id=path.stem) as w:
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image(
                "CT",
                stored,
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=self.SLOPE,
                rescale_intercept=self.INTERCEPT,
            )
        return stored

    def test_S4_2_normalization_matches_what_the_loader_reads(
        self, tmp_path: Path
    ) -> None:
        from medh5.dataset.stats import DatasetStats, stats_for

        path = tmp_path / "ct.medh5"
        stored = self._write(path)
        physical = stored.astype(np.float64) * self.SLOPE + self.INTERCEPT
        stats = stats_for(path)
        assert stats.physical is True
        mean, std = stats.normalization("CT")
        assert mean == pytest.approx(float(physical.mean()))
        assert std == pytest.approx(float(physical.std()))
        moments = stats.images["CT"]
        assert moments.minimum == pytest.approx(float(physical.min()))
        assert moments.maximum == pytest.approx(float(physical.max()))
        with medh5.open(path) as sample:
            loaded = sample.images["CT"].read(physical=True)
        assert mean == pytest.approx(float(loaded.mean()))
        # The convention travels with the numbers.
        back = DatasetStats.from_json(stats.to_json())
        assert back.physical is True
        assert back.normalization("CT")[0] == pytest.approx(mean)

    def test_S4_2_stored_values_stay_available(self, tmp_path: Path) -> None:
        from medh5.dataset.stats import compute_stats, stats_for

        path = tmp_path / "ct.medh5"
        stored = self._write(path)
        one = stats_for(path, physical=False)
        assert one.physical is False
        assert one.normalization("CT")[0] == pytest.approx(float(stored.mean()))
        many = compute_stats([path], physical=False)
        assert many.physical is False
        assert many.normalization("CT")[0] == pytest.approx(float(stored.mean()))

    def test_merging_the_two_conventions_is_refused(self) -> None:
        from medh5.dataset.stats import DatasetStats

        left = DatasetStats(samples=1, physical=True)
        with pytest.raises(MEDH5Error, match="physical"):
            left.merge(DatasetStats(samples=1, physical=False))
        # A failure record carries no samples and merges into either.
        left.merge(DatasetStats(failures=("x: broken",), physical=False))
        assert left.physical is True

    def test_cli_stored_flag(self, tmp_path: Path, capsys: Any) -> None:
        from medh5.dataset.manifest import scan

        self._write(tmp_path / "ct.medh5")
        manifest, _ = scan(tmp_path)
        manifest.save(tmp_path / "cohort.json")
        assert main(["dataset", "stats", str(tmp_path / "cohort.json")]) == 0
        assert "physical" in capsys.readouterr().out
        assert (
            main(["dataset", "stats", str(tmp_path / "cohort.json"), "--stored"]) == 0
        )
        assert "stored" in capsys.readouterr().out
