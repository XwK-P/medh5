"""``medh5 fix`` and ``medh5 scrub`` --- repair and de-identification (plan §5).

The two commands share a shape and not much else: one rebuilds what is derived
and one removes what should never have been written.  Both are tested on the
same principle --- that a tool which changes a file must say what it changed,
and must not claim more than it did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.curation import scrub as scrubber
from medh5.errors import MEDH5ValidationError
from medh5.integrity.repair import diagnose, fix
from tests.v1.conftest import SHAPE, block, write_sample


@pytest.fixture
def indexed(tmp_path: Path, label_set, masks) -> Path:
    return write_sample(
        tmp_path / "case.medh5", label_set=label_set, masks=masks, index=True
    )


class TestFix:
    def test_a_healthy_file_needs_nothing(self, indexed):
        assert diagnose(indexed).clean
        assert not fix(indexed).changed

    def test_an_external_edit_shows_up_as_a_digest_mismatch(self, indexed):
        """The case `fix` exists for: something edited the data past the writer."""
        _edit_mask_in_place(indexed)
        diagnosis = diagnose(indexed)
        assert diagnosis.mismatched == ("annotations/organs_tp0/data",)
        assert diagnosis.needs_digests

    def test_the_content_id_alone_does_not_catch_an_edited_dataset(self, indexed):
        """`content_id` is a Merkle over stored digests, not over the bytes.

        Editing a dataset without restamping it changes the data but not the
        digest attribute the root hashes, so the root still matches while the
        object does not --- which is why `verify` checks both and why an
        integrity pass is per-object, not just a single top-level comparison.
        """
        _edit_mask_in_place(indexed)
        diagnosis = diagnose(indexed)
        assert diagnosis.content_id_ok is True
        assert diagnosis.mismatched

    def test_a_stale_index_is_found_and_rebuilt(self, indexed):
        """Restamping an external edit is what leaves the index behind."""
        _edit_mask_in_place(indexed)
        fix(indexed, rewrite_digests=True, reason="edited by an external tool")
        diagnosis = diagnose(indexed)
        assert diagnosis.stale_index == ("organs_tp0",)
        assert diagnosis.needs_index

        repair = fix(indexed, rebuild_index=True)
        assert repair.rebuilt_index == ("organs_tp0",)
        assert not diagnose(indexed).stale_index
        with medh5.open(indexed) as sample:
            counts = sample.index["organs_tp0"].voxel_counts
            direct = sample.annotations["organs_tp0"].voxel_counts()
            assert counts == direct

    def test_removing_an_annotation_takes_its_index_with_it(self, indexed):
        """Not stale --- gone.  A stale index is a mismatch, not an absence."""
        with medh5.amend(indexed) as writer:
            writer.remove_annotation("organs_tp0")
            writer.add_segmentation(
                "organs_tp0", grid="ct_tp0", masks={1: block(SHAPE, (2, 2, 2), 3)}
            )
        assert diagnose(indexed).stale_index == ()

    def test_diagnosing_changes_nothing(self, indexed):
        with medh5.open(indexed) as sample:
            before = sample.content_id
        diagnose(indexed)
        with medh5.open(indexed) as sample:
            assert sample.content_id == before

    def test_no_flags_means_report_only(self, indexed):
        _edit_mask_in_place(indexed)
        repair = fix(indexed)
        assert repair.diagnosis.needs_digests
        assert not repair.changed
        assert diagnose(indexed).needs_digests

    def test_rewriting_digests_without_a_reason_is_refused(self, indexed):
        """Restamping destroys evidence; it must be a decision, not a default."""
        with pytest.raises(MEDH5ValidationError, match="reason"):
            fix(indexed, rewrite_digests=True)

    def test_rewriting_digests_records_what_it_did_not_verify(self, indexed):
        repair = fix(
            indexed, rewrite_digests=True, reason="reconstructed by an external tool"
        )
        assert repair.rewrote_digests
        assert any("asserts nothing" in note for note in repair.notes)
        with medh5.open(indexed) as sample:
            activities = [a for a in sample.document.provenance.activities if a.tool]
            restamp = [a for a in activities if "rewrite-digests" in str(a.tool)]
            assert restamp, "the restamp must be in the file's own provenance"
            assert restamp[0].params["verified_content"] is False
            assert "external tool" in restamp[0].params["reason"]

    def test_a_restamped_file_verifies_again(self, indexed):
        fix(indexed, rewrite_digests=True, reason="test")
        with medh5.open(indexed) as sample:
            assert sample.verify().ok


class TestScrubFinds:
    @pytest.fixture
    def dirty(self, tmp_path: Path, label_set) -> Path:
        """A sample a careless converter produced."""
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s1", subject_id="Doe^Jane") as w:
            w.add_timepoint("tp0", date="2026-02-03", study_uid="1.2.840.113619.2.1")
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="1.2.840.10008.3.1.2.9",
            )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.acquisition(
                "CT",
                kvp=120,
                PatientName="Doe^Jane",
                InstitutionName="St Elsewhere",
                StudyDate="20260203",
            )
            w.extra("source", {"AccessionNumber": "A-99213", "notes": "x" * 250})
            w.person("Dr Alice Roe")
        return path

    def test_every_rule_fires_where_it_should(self, dirty):
        report = scrubber.scan(dirty)
        found = {(f.rule, f.where) for f in report.findings}
        assert ("person_name", "identity.subject_id") in found
        assert ("identifier", "acquisition.CT.PatientName") in found
        assert ("identifier", "acquisition.CT.InstitutionName") in found
        assert ("identifier", "extra.source.AccessionNumber") in found
        assert ("date", "acquisition.CT.StudyDate") in found
        assert ("uid", "grids.g.frame_uid") in found
        assert ("uid", "timepoints[0].study_uid") in found
        assert ("free_text", "extra.source.notes") in found
        assert any(f.rule == "staff_name" for f in report.findings)

    def test_a_clean_sample_is_clean(self, sample_path):
        assert scrubber.scan(sample_path).clean

    def test_scanning_changes_nothing(self, dirty):
        with medh5.open(dirty) as sample:
            before = sample.content_id
        scrubber.scan(dirty)
        with medh5.open(dirty) as sample:
            assert sample.content_id == before

    def test_a_pseudonymised_uid_is_not_flagged(self, sample_path):
        """`pseudo:` and UUIDs must not read as DICOM UIDs."""
        report = scrubber.scan(sample_path)
        assert not [f for f in report.findings if f.rule == "uid"]

    def test_the_report_says_what_it_did_not_look_at(self, dirty):
        report = scrubber.scan(dirty)
        assert any("pixel data" in note for note in report.not_checked)
        assert "NOT checked" in report.format()

    def test_strict_widens_what_is_actionable(self, dirty):
        basic = scrubber.scan(dirty, profile="basic")
        strict = scrubber.scan(dirty, profile="strict")
        assert len(strict.actionable) > len(basic.actionable)
        assert len(strict.findings) == len(basic.findings)

    def test_an_unknown_profile_is_refused(self, dirty):
        with pytest.raises(MEDH5ValidationError, match="unknown profile"):
            scrubber.scan(dirty, profile="paranoid")


class TestScrubApplies:
    @pytest.fixture
    def dirty(self, tmp_path: Path) -> Path:
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s1", subject_id="subj-A") as w:
            w.add_timepoint("tp0", date="2026-02-03", study_uid="1.2.840.113619.2.1")
            w.add_timepoint("tp1", index=1, date="2026-05-04", days_from_baseline=90)
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="1.2.840.10008.3.1.2.9",
            )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.acquisition("CT", kvp=120, PatientName="Doe^Jane", StudyDate="20260203")
            w.extra("source", {"AccessionNumber": "A-99213", "series": "1.2.3.4.5"})
        return path

    def test_S11_4_identifiers_are_removed_and_physics_is_kept(self, dirty):
        scrubber.apply(dirty)
        with medh5.open(dirty) as sample:
            acquisition = sample.document.acquisition["CT"]
            assert "PatientName" not in acquisition
            assert acquisition["kvp"] == 120
            assert "AccessionNumber" not in sample.document.extra["source"]

    def test_uids_are_pseudonymised_not_deleted(self, dirty):
        """Deleting a frame UID would break registration; a pseudonym does not."""
        report = scrubber.apply(dirty)
        with medh5.open(dirty) as sample:
            frame = sample.grids["g"].frame_uid
            assert frame is not None and frame.startswith("pseudo:")
            assert sample.document.timepoints["tp0"].study_uid.startswith("pseudo:")
            assert sample.document.extra["source"]["series"].startswith("pseudo:")
        assert report.uid_map["1.2.840.10008.3.1.2.9"] == frame

    def test_the_same_uid_maps_the_same_way_everywhere(self, dirty, tmp_path):
        """A cohort has to stay joinable after being scrubbed file by file."""
        assert scrubber.pseudonymise("1.2.3") == scrubber.pseudonymise("1.2.3")
        assert scrubber.pseudonymise("1.2.3") != scrubber.pseudonymise("1.2.3", "salt")
        assert scrubber.pseudonymise("1.2.3", "s") == scrubber.pseudonymise(
            "1.2.3", "s"
        )

    def test_dates_are_dropped_when_no_shift_is_given(self, dirty):
        scrubber.apply(dirty)
        with medh5.open(dirty) as sample:
            assert sample.document.timepoints["tp0"].date is None
            assert "StudyDate" not in sample.document.acquisition["CT"]

    def test_a_shift_preserves_the_interval(self, dirty):
        scrubber.apply(dirty, date_shift_days=-117)
        with medh5.open(dirty) as sample:
            first = sample.document.timepoints["tp0"].date
            second = sample.document.timepoints["tp1"].date
            assert first == "2025-10-09"
            assert second == "2026-01-07"
            assert sample.document.acquisition["CT"]["StudyDate"] == "20251009"
            assert sample.document.timepoints["tp1"].days_from_baseline == 90

    def test_S11_4_the_attestation_says_what_was_not_checked(self, dirty):
        scrubber.apply(dirty, date_shift_days=-117, performed_by="RAD-07")
        with medh5.open(dirty) as sample:
            record = sample.document.deidentification
            assert record is not None
            assert record.method == "medh5-scrub"
            assert "container metadata only" in record.profile
            assert record.date_shift_days == -117
            assert record.burned_in_annotation_checked is False
            types = [a.type for a in sample.document.provenance.activities]
            assert "deidentify" in types

    def test_the_file_is_still_valid_afterwards(self, dirty):
        from medh5.validate import validate_file

        scrubber.apply(dirty, date_shift_days=-30)
        assert not validate_file(dirty).errors

    def test_scrubbing_twice_is_stable(self, dirty):
        scrubber.apply(dirty, date_shift_days=-117)
        with medh5.open(dirty) as sample:
            first = sample.document.to_json()
        second_run = scrubber.apply(dirty, date_shift_days=-117)
        with medh5.open(dirty) as sample:
            second = sample.document.to_json()
        assert first["timepoints"] == second["timepoints"]
        assert first["acquisition"] == second["acquisition"]
        assert not [f for f in second_run.findings if f.rule == "identifier"]


def _edit_mask_in_place(path: Path) -> None:
    """Change an annotation's bytes behind the writer's back.

    Exactly what an external tool does, and the reason a stale index and a
    mismatched digest are different problems from a malformed file.
    """
    import h5py

    with h5py.File(path, "a") as handle:
        node = handle["annotations"]["organs_tp0"]["data"]
        data = node[...]
        assert data.any(), "the fixture must have foreground to erase"
        data[data != 0] = 0
        node[...] = data
