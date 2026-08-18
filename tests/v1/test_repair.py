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

    def test_S13_3_rebuilding_an_index_will_not_launder_a_digest_mismatch(
        self, indexed
    ):
        """An amend restamps every digest, so the rebuild path needs the guard too.

        Recomputing a mismatched digest does not undo the edit that caused it,
        it destroys the evidence of it --- and on this path it did so with no
        reason, no provenance activity, and `rewrote_digests` reporting False.
        """
        _edit_mask_in_place(indexed)
        assert diagnose(indexed).mismatched

        with pytest.raises(MEDH5ValidationError, match="no longer match"):
            fix(indexed, rebuild_index=True)
        assert diagnose(indexed).mismatched, "the evidence survived the refusal"

        # The deliberate path still works, and still says what it did not verify.
        repair = fix(
            indexed,
            rebuild_index=True,
            rewrite_digests=True,
            reason="edited by an external tool, content confirmed",
        )
        assert repair.rewrote_digests
        assert not diagnose(indexed).mismatched
        assert any("asserts nothing" in note for note in repair.notes)

    def test_S14_3_rebuilding_does_not_index_what_was_never_indexed(
        self, tmp_path, label_set, masks
    ):
        """`fix` repairs; it does not decide a curator's storage budget for them.

        An empty rebuild list used to reach `build_index` as None, which means
        "every indexable annotation", so a file deliberately shipped without an
        index got one built for everything.
        """
        path = write_sample(
            tmp_path / "bare.medh5", label_set=label_set, masks=masks, index=False
        )
        diagnosis = diagnose(path)
        assert diagnosis.stale_index == () and diagnosis.missing_index == ()

        repair = fix(path, rebuild_index=True)
        assert repair.rebuilt_index == ()
        assert not repair.changed
        with medh5.open(path) as sample:
            assert sorted(sample.index) == []

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


class TestScrubCoverage:
    """What the sweep must not miss.  A false negative here is the worst
    outcome this module has, because the file then carries an attestation."""

    def _with(self, tmp_path, name, **acquisition):
        path = tmp_path / f"{name}.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.acquisition("CT", **acquisition)
        return path

    @pytest.mark.parametrize(
        "key",
        [
            "PatientName",
            "patientname",
            "PATIENTNAME",
            "patient_name",
            "Patient Name",
            "patient__name",
        ],
    )
    def test_a_key_is_matched_however_it_is_spelled(self, tmp_path, key):
        path = self._with(tmp_path, f"k{abs(hash(key))}", **{key: "Doe^Jane"})
        assert scrubber.scan(path).actionable

    @pytest.mark.parametrize(
        "key",
        [
            "AdditionalPatientHistory",
            "PatientComments",
            "Occupation",
            "EthnicGroup",
            "MilitaryRank",
            "CountryOfResidence",
            "StationName",
            "DeviceSerialNumber",
            "ClinicalTrialSubjectID",
            "ContentCreatorName",
            "VerifyingObserverName",
            "RequestedProcedureID",
            "InstitutionName",
            "CurrentPatientLocation",
            "PatientInstitutionResidence",
        ],
    )
    def test_the_PS3_15_E1_attributes_are_reported(self, tmp_path, key):
        """The denylist started far too short --- 7 of 43 probes were caught."""
        path = self._with(tmp_path, f"e{abs(hash(key))}", **{key: "SENSITIVE"})
        assert [f for f in scrubber.scan(path).findings if key in f.where], key

    def test_S3_4_a_frame_uid_is_pseudonymised_everywhere_it_is_named(self, tmp_path):
        """Grids are not the only place a FrameOfReferenceUID appears.

        A world-space annotation names one and a transform names two.  Rewriting
        the grids alone left the real UID in a file certified de-identified and
        --- because the frame graph is keyed on the string --- disconnected the
        grids from the transform relating them, so `transform_between` answered
        None and every longitudinal loader silently dropped the pair.
        """
        import h5py

        uid = "1.2.840.113619.2.55.3.604688119.868.1234567890.123"
        later = uid + ".9"
        path = tmp_path / "frames.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            for tp, frame in (("tp0", uid), ("tp1", later)):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}", np.zeros(SHAPE, np.int16), grid=f"g_{tp}", modality="CT"
                )
            w.add_boxes(
                "lesions",
                boxes=[[[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]]],
                class_ids=[1],
                space="world",
                frame_uid=uid,
            )
            w.add_transform(
                "t", kind="affine", matrix=np.eye(4), from_frame=uid, to_frame=later
            )

        reported = {f.where for f in scrubber.scan(path).findings if f.rule == "uid"}
        assert reported == {
            "grids.g_tp0.frame_uid",
            "grids.g_tp1.frame_uid",
            "annotations.lesions.frame_uid",
            "transforms.t.from_frame",
            "transforms.t.to_frame",
        }

        scrubber.apply(path, salt="pepper")

        surviving = []
        with h5py.File(path, "r") as handle:

            def collect(name, obj):
                for key, value in obj.attrs.items():
                    text = value.decode() if isinstance(value, bytes) else str(value)
                    if uid in text:
                        surviving.append(f"{name}@{key}")

            handle.visititems(collect)
        assert surviving == [], "a certified file still holding the real UID"

        with medh5.open(path) as sample:
            assert sample.transform_between("tp0", "tp1") is not None, (
                "the rename kept the frame graph connected"
            )
            assert sample.verify().ok
            assert not scrubber.scan(path).actionable

    def test_a_quasi_identifier_is_reported_but_kept_under_basic(self, tmp_path):
        """PatientWeight drives a PET SUV; removing it by default would break
        quantitative imaging to buy privacy the caller may already have."""
        path = self._with(tmp_path, "quasi", kvp=120, PatientWeight=82.0)
        report = scrubber.scan(path)
        assert [f.rule for f in report.findings] == ["quasi_identifier"]
        assert not report.actionable

        scrubber.apply(path, profile="basic")
        with medh5.open(path) as sample:
            assert sample.document.acquisition["CT"]["PatientWeight"] == 82.0
            assert "retained for review" in sample.document.deidentification.profile

    def test_strict_removes_it_and_says_so(self, tmp_path):
        path = self._with(tmp_path, "quasi-strict", kvp=120, PatientWeight=82.0)
        assert scrubber.scan(path, profile="strict").actionable
        scrubber.apply(path, profile="strict")
        with medh5.open(path) as sample:
            acquisition = sample.document.acquisition["CT"]
            assert "PatientWeight" not in acquisition
            assert acquisition["kvp"] == 120
            assert "quasi-identifiers removed" in (
                sample.document.deidentification.profile
            )

    def test_a_uid_key_holding_a_non_uid_is_reported(self, tmp_path):
        """It cannot be pseudonymised safely, so it must not pass silently."""
        path = self._with(tmp_path, "uidkey", StudyInstanceUID="not-a-uid")
        findings = scrubber.scan(path).findings
        assert [f.rule for f in findings] == ["uid"]

    def test_too_deep_is_reported_not_skipped(self, tmp_path):
        """A silent stop would leave an attestation over uninspected data."""
        payload = {"PatientName": "Doe^Jane"}
        for _ in range(scrubber.MAX_DEPTH + 1):
            payload = {"level": payload}
        path = tmp_path / "deep.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.extra("src", payload)
        report = scrubber.scan(path)
        assert not report.clean
        assert "too_deep" in {f.rule for f in report.findings}

    @pytest.mark.parametrize(
        "payload",
        [
            {"items": [{"PatientName": "Doe^Jane"}]},
            {"items": [[{"PatientName": "Doe^Jane"}]]},
            {"1": {"PatientName": "Doe^Jane"}},
            {"StudyDate": 20260203},
            {"names": ["Doe^Jane"]},
        ],
    )
    def test_the_walk_reaches_awkward_structures(self, tmp_path, payload):
        path = tmp_path / f"w{abs(hash(str(payload)))}.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.extra("src", payload)
        assert scrubber.scan(path).actionable

    def test_nothing_actionable_survives_apply(self, tmp_path):
        """The contract: after --apply, a re-scan finds nothing left to do."""
        path = tmp_path / "full.medh5"
        with medh5.create(path, sample_id="s", subject_id="Doe^Jane") as w:
            w.add_timepoint("tp0", date="2026-02-03", study_uid="1.2.840.113619.2.1")
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="1.2.840.10008.3.1.2.9",
            )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.acquisition("CT", kvp=120, PatientName="Doe^Jane", StudyDate="20260203")
            w.extra("src", {"AccessionNumber": "A-99213", "series": "1.2.3.4.5"})
        assert len(scrubber.scan(path).actionable) >= 5

        scrubber.apply(path, date_shift_days=-117)
        assert not scrubber.scan(path).actionable

    def test_a_swept_file_passes_scrub_as_a_gate(self, tmp_path):
        """`scrub` exits non-zero on findings, so its own output must pass."""
        path = self._with(
            tmp_path, "gate", kvp=120, PatientName="Doe^Jane", StudyDate="20260203"
        )
        scrubber.apply(path, date_shift_days=-117)
        assert not scrubber.scan(path).actionable


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
