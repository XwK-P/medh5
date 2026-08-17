"""Cohort tools: manifests, splits, statistics, cross-file checks (plan §5).

The unit under test is never one file.  Everything here is about a property no
sample can carry on its own --- who agrees on a label set, who is in which
partition, what the intensity distribution is over a whole study.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.dataset.check import check
from medh5.dataset.manifest import Manifest, entries_for, find, scan
from medh5.dataset.split import make_splits, write_claims
from medh5.dataset.stats import Moments, compute_stats, stats_for
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.labels.labelset import LabelClass, LabelSet
from tests.v1.conftest import SHAPE, block, write_sample


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


class TestManifest:
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
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        built, failures = scan(cohort)
        assert len(built) == 6
        assert len(failures) == 1 and "broken.medh5" in failures[0]

    def test_strict_mode_raises_instead(self, cohort):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        with pytest.raises(MEDH5Error):
            scan(cohort, on_error="raise")

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

    def test_a_cohort_large_enough_fills_everything(self, manifest, cohort, label_set):
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
        assert any(len(v) > 1 for v in sites.values()) or len(sites) > 1

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

    def test_each_stratum_is_spread_across_the_partitions(
        self, manifest, cohort, label_set
    ):
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

    def test_a_groups_stratum_is_its_majority(self, manifest, cohort):
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

    def test_more_folds_than_groups_says_so(self, manifest):
        """A 5-fold CV that quietly runs 4 ways is not noticed until later."""
        split = make_splits(manifest, k_folds=8, seed=0)
        assert len(split.assignments) == 4
        assert split.empty_folds == (4, 5, 6, 7)

    def test_enough_groups_fills_every_fold(self, manifest):
        assert make_splits(manifest, k_folds=2, seed=0).empty_folds == ()

    def test_a_ratio_split_has_no_folds_to_report(self, manifest):
        assert make_splits(manifest, seed=0).empty_folds == ()

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
        self, manifest, cohort, tmp_path
    ):
        shard = tmp_path / "shard.medh5c"
        medh5.pack(sorted(cohort.glob("*.medh5")), shard)
        built, _ = scan(shard)
        with pytest.raises(MEDH5ValidationError, match="unpack"):
            write_claims(make_splits(built), built)


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

    def test_C501_a_partly_deidentified_cohort_is_flagged(self, cohort, label_set):
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


def _subject(path: Path) -> str:
    with medh5.open(path) as sample:
        return sample.identity.subject_id
