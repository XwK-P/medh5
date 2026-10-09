"""De-identification: the identifier sweep, its fixes and its record (§11.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.curation import scrub as scrubber
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file
from tests.helpers import SHAPE, write_dicom_series
from tests.kits import Numbered, Organs


class TestW8Deidentification:
    """F-08: the attestation, the scope it covers, and the exit code."""

    def _dirty(self, path: Path) -> Path:
        from medh5.curation import Issue

        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.identity(PatientName="Doe^Jane")
            w.cohort(
                dataset_id="d",
                InstitutionName="St Elsewhere",
                ReferringPhysicianName="Smith^John",
            )
            w.add_timepoint("tp0", date="2026-02-03", study_uid="1.2.840.113619.2.55")
            w.add_grid(
                "g",
                shape=Organs.SHAPE,
                spacing=(1.0, 1.0, 1.0),
                frame_uid="1.2.840.10008.5.1.4.1.1.2",
            )
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
            tool = w.software("conv")
            w.person("Brown^Ann")
            w.activity(
                "import", agent=tool, tool="ctp", params={"OperatorsName": "Lee^Bo"}
            )
            w.acquisition("CT", PatientID="MRN-1")
            w.set_quality(
                "q", status="draft", issues=[Issue(code="x", note="Green^Sam")]
            )
            w.extra("dicom", {"PatientID": "MRN-1"})
        return path

    def test_F08_S11_4_apply_acts_on_everything_it_flags(self, tmp_path: Path):
        """A strict apply used to leave three identifiers it had itself flagged.

        `apply` cleaned `extra` and `acquisition` only, then wrote a record
        whose profile string reads "quasi-identifiers removed" and exited 0 ---
        over a file still carrying a patient name, an institution and a
        referring physician, in `identity.extra` and `cohort`.
        """
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "dirty.medh5")
        before = scrub.scan(path, profile="strict")
        flagged = {f.where for f in before.actionable}
        assert {
            "identity.extra.PatientName",
            "cohort.InstitutionName",
            "cohort.ReferringPhysicianName",
            "provenance.activities[act_import_1].params.OperatorsName",
            "quality.q.issues[0].note",
            "provenance.agents[p2]",
        } <= flagged

        report = scrub.apply(path, profile="strict", date_shift_days=-30, salt="pep")
        assert report.applied
        assert not report.remaining_actionable, report.remaining
        assert report.ok

        # The independent re-scan, which is the check the tool now makes itself.
        assert not scrub.scan(path, profile="strict").actionable

    def test_F08_the_record_states_what_the_run_achieved(self, tmp_path: Path):
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "record.medh5")
        scrub.apply(path, profile="strict", date_shift_days=-30)
        with medh5.open(path) as sample:
            activity = sample.document.provenance.activities_by_type("deidentify")[0]
            assert activity.params["remaining_actionable"] == 0
            assert activity.params["changes"] >= activity.params["remaining"]

    def test_F08_apply_is_idempotent_and_stays_green(self, tmp_path: Path):
        """A rule that fires on its own output can never go green in a pipeline."""
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "twice.medh5")
        first = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        second = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        assert first.ok and second.ok
        assert not second.remaining_actionable

    def test_F08_the_ids_a_join_needs_are_reported_and_never_rewritten(
        self, tmp_path: Path
    ):
        from medh5.curation import scrub
        from medh5.curation.scrub import UNFIXABLE_LOCATIONS

        path = tmp_path / "named.medh5"
        with medh5.create(path, sample_id="s1", subject_id="Doe^Jane") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
        report = scrub.scan(path, profile="strict")
        named = [f for f in report.findings if f.where == "identity.subject_id"]
        assert named and not named[0].actionable
        assert "identity.subject_id" in UNFIXABLE_LOCATIONS
        scrub.apply(path, profile="strict")
        with medh5.open(path) as sample:
            assert sample.identity.subject_id == "Doe^Jane"

    def test_F08_the_cli_exit_code_follows_the_re_scan(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._dirty(tmp_path / "cli.medh5")
        assert main(["scrub", str(path)]) == 1
        assert (
            main(["scrub", str(path), "--apply", "--profile", "strict", "--by", "R7"])
            == 0
        )
        capsys.readouterr()
        assert main(["scrub", str(path), "--profile", "strict"]) == 0


class TestW15Deidentification:
    """F-14, L-36: the scan and the clean see every string, and say so."""

    @pytest.fixture
    def dicom_import(self, tmp_path: Path) -> dict[str, Any]:
        pytest.importorskip("pydicom")
        from medh5.io.dicom import from_dicom

        info = write_dicom_series(
            tmp_path / "dcm",
            patient_id="MRN12345",
            study_uid="1.2.826.0.1.3680043.8.498.111",
            study_date="20230105",
        )
        out = tmp_path / "case.medh5"
        from_dicom(tmp_path / "dcm", out)
        return {"path": out, **info}

    def test_F14_S11_4_the_importer_says_where_the_ids_came_from(self, dicom_import):
        from medh5.curation import scrub

        path = dicom_import["path"]
        with medh5.open(path) as sample:
            identity = sample.document.identity
            assert identity.subject_id == "MRN12345"
            assert identity.extra["id_source"]["subject_id"] == "dicom:PatientID"
        report = scrub.scan(path)
        rules = {(f.rule, f.where) for f in report.findings}
        assert ("identity", "identity.subject_id") in rules
        assert ("identity", "identity.sample_id") in rules
        assert ("uid", "provenance.activities[act_import_1].inputs[0]") in rules

    def test_F14_a_strict_apply_fails_while_the_ids_are_record_numbers(
        self, dicom_import
    ):
        """It used to exit 0 over the PatientID and a real SeriesInstanceUID."""
        from medh5.curation import scrub

        path = dicom_import["path"]
        report = scrub.apply(path, profile="strict", salt="pep", date_shift_days=-30)
        assert not report.ok
        assert {f.where for f in report.open_identity} == {
            "identity.sample_id",
            "identity.subject_id",
        }
        assert "--pseudonymise-ids" in report.format()
        # The real UID the timepoint was pseudonymised from is not left beside it.
        assert dicom_import["series_uid"].encode() not in path.read_bytes()

    def test_F14_pseudonymised_ids_leave_no_identifier_bytes(self, dicom_import):
        from medh5.curation import scrub

        path = dicom_import["path"]
        report = scrub.apply(
            path,
            profile="strict",
            salt="pep",
            date_shift_days=-30,
            pseudonymise_ids=True,
        )
        assert report.ok, report.format()
        data = path.read_bytes()
        for needle in (
            b"MRN12345",
            b"1.2.826.0.1.3680043.8.498.111",
            dicom_import["series_uid"].encode(),
            dicom_import["frame_uid"].encode(),
            b"20230105",
            b"2023-01-05",
        ):
            assert needle not in data, needle
        with medh5.open(path) as sample:
            document = sample.document
            series = document.timepoints["tp0"].series_uids["CT_tp0"]
            # One pseudonym per UID, wherever the UID was named.
            assert document.provenance.activities[0].inputs == (f"dicom:{series}",)
            assert series == report.uid_map[dicom_import["series_uid"]]
            identity = document.identity
            assert identity.subject_id == report.uid_map["MRN12345"]
            assert identity.extra["id_source"]["subject_id"] == "pseudonym"
            assert document.deidentification.id_mapping == "external"
            assert "ids pseudonymised" in document.deidentification.profile
            assert "voxel data not examined" in document.deidentification.profile
            assert sample.verify().ok
        assert validate_file(path).ok
        again = scrub.apply(path, profile="strict", salt="pep", pseudonymise_ids=True)
        assert again.ok

    def test_F14_pseudonymising_ids_needs_a_salt(self, dicom_import):
        """An unsalted hash of a record number is reversed by hashing them all."""
        from medh5.curation import scrub

        with pytest.raises(MEDH5ValidationError, match="salt"):
            scrub.apply(dicom_import["path"], pseudonymise_ids=True)

    def test_F14_a_file_named_after_the_record_number_is_reported(
        self, dicom_import, tmp_path: Path
    ):
        from medh5.curation import scrub

        path = tmp_path / "MRN12345.medh5"
        dicom_import["path"].rename(path)
        assert "file_name" in {f.rule for f in scrub.scan(path).findings}
        report = scrub.apply(path, profile="strict", salt="pep", pseudonymise_ids=True)
        assert [f.rule for f in report.remaining] == ["file_name"]
        assert not report.ok

    def test_F14_the_cli_flag(self, dicom_import, capsys):
        from medh5.cli import main

        path = str(dicom_import["path"])
        assert main(["scrub", path, "--pseudonymise-ids"]) != 0
        assert main(["scrub", path, "--apply", "--profile", "strict"]) != 0
        assert (
            main(
                [
                    "scrub",
                    path,
                    "--apply",
                    "--profile",
                    "strict",
                    "--salt",
                    "pep",
                    "--pseudonymise-ids",
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert main(["scrub", path, "--profile", "strict"]) == 0

    def test_F14_reminting_an_id_retires_its_recorded_source(self, tmp_path: Path):
        """A re-minted id is not a DICOM id, and must not be reported as one."""
        from medh5.curation import scrub

        path = tmp_path / "remint.medh5"
        with Numbered.writer(path) as w:
            w.identity(id_source={"sample_id": "dicom:PatientID"})
        assert "identity" in {f.rule for f in scrub.scan(path).findings}
        with medh5.amend(path) as w:
            w.identity(sample_id="case-001")
        with medh5.open(path) as sample:
            assert "id_source" not in sample.document.identity.extra
        assert "identity" not in {f.rule for f in scrub.scan(path).findings}

    @pytest.mark.parametrize("name", ["Müller^Hans", "山田^太郎", "Ødegård^Åse"])
    def test_L36_a_person_name_in_any_script(self, tmp_path: Path, name: str):
        from medh5.curation import scrub

        path = tmp_path / "pn.medh5"
        with Numbered.writer(path) as w:
            w.extra("src", {"reader": name})
        assert [f.rule for f in scrub.scan(path).actionable] == ["person_name"]

    def test_L36_an_age_over_89_and_an_organization(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "age.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", subject_age_years=93.0)
            w.organization("St Elsewhere General")
        found = {f.rule: f for f in scrub.scan(path).findings}
        assert not found["age"].actionable
        assert not found["organization_name"].actionable
        assert found["age"].where == "timepoints[0].subject_age_years"
        assert scrub.apply(path, profile="strict").ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].subject_age_years == 90.0
            names = [a.name for a in sample.document.provenance.agents]
            assert "St Elsewhere General" not in names

    def test_L36_an_identifying_attribute_on_an_hdf5_object(self, tmp_path: Path):
        """The key rules apply to attribute names as they do to `/meta` keys."""
        from medh5.curation import scrub

        path = Numbered.plain(tmp_path / "attr.medh5")
        with h5py.File(path, "r+") as handle:
            image = handle["images/CT"]
            image.attrs["PatientName"] = "Doe^Jane"
            image.attrs["x_source"] = json.dumps({"OperatorsName": "Lee^Bo", "kv": 120})
        found = {(f.rule, f.where) for f in scrub.scan(path).actionable}
        assert ("identifier", "images.CT.PatientName") in found
        assert ("identifier", "images.CT.x_source.OperatorsName") in found
        assert scrub.apply(path).ok
        with h5py.File(path, "r") as handle:
            attrs = handle["images/CT"].attrs
            assert "PatientName" not in attrs
            assert json.loads(attrs["x_source"]) == {"kv": 120}
        assert b"Doe^Jane" not in path.read_bytes()
        assert b"Lee^Bo" not in path.read_bytes()

    def test_L36_every_string_slot_is_removed_or_reported(self, tmp_path: Path):
        """The invariant the location lists could not keep.

        A distinct person name is planted in every string slot a fully populated
        sample has --- `/meta`, attributes, string datasets, an unknown group ---
        and a strict apply must, for each one, either remove it from the file's
        bytes or still report it.
        """
        from medh5.curation import Agreement, Issue, scrub

        planted: dict[str, str] = {}

        def plant(slot: str) -> str:
            index = len(planted)
            name = f"Slot{chr(97 + index // 26)}{chr(97 + index % 26)}^Planted"
            planted[slot] = name
            return name

        path = tmp_path / "full.medh5"
        vocabulary = LabelSet(
            "vocab",
            version="1.0.0",
            classes=[
                LabelClass(1, "lesion", plant("label name"), category=plant("cat")),
                LabelClass(2, "organ", "Organ"),
            ],
        )
        with medh5.create(path, sample_id="s1", subject_id="subj-1") as w:
            w.identity(bodypart=plant("bodypart"), note=plant("identity extra"))
            w.cohort(
                dataset_id=plant("dataset_id"),
                site_id=plant("site_id"),
                scanner_id=plant("scanner_id"),
                group_id=plant("group_id"),
                acquisition_protocol=plant("protocol"),
                other=plant("cohort extra"),
            )
            w.add_timepoint(
                "tp0",
                label=plant("tp label"),
                description=plant("tp description"),
                study_uid=plant("study_uid"),
                series_uids={"CT": plant("series_uids")},
            )
            w.label_set(vocabulary)
            w.person(plant("person"), qualification=plant("qualification"))
            w.organization(plant("organization"))
            tool = w.software(plant("software"), version=plant("version"))
            act = w.activity(
                "import",
                agent=tool,
                tool=plant("tool"),
                inputs=[plant("input")],
                outputs=[plant("output")],
                params={"note": plant("params")},
            )
            w.add_grid(
                "g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image(
                "CT",
                np.zeros(Numbered.SHAPE, np.int16),
                grid="g",
                modality="CT",
                prov=act,
                value_units=plant("value_units"),
            )
            w.set_quality(
                "q",
                status="draft",
                issues=[Issue(code=plant("issue code"), note=plant("issue note"))],
                agreement=[
                    Agreement(metric=plant("metric"), value=0.5, against=plant("vs"))
                ],
            )
            w.split(set_id=plant("set_id"), partition="train", assigned_by=plant("by"))
            w.acquisition("CT", note=plant("acquisition"))
            w.extra("vendor", {"a": plant("extra"), "b": [{"c": plant("nested")}]})
            w.add_boxes(
                "boxes",
                [[[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]]],
                [1],
                grid="g",
                attributes=[{"reader": plant("box attributes")}],
            )
            w.add_points("pts", [[1.0, 2.0, 3.0]], grid="g", names=[plant("points")])
            w.add_classification(
                "cls",
                {1: 1.0},
                schemes=[plant("schemes")],
                scheme_values=[plant("scheme_values")],
            )
        with h5py.File(path, "r+") as handle:
            handle.attrs["x_note"] = plant("root attribute")
            image = handle["images/CT"]
            image.attrs["x_operator"] = plant("object attribute")
            image.attrs["x_list"] = np.array(
                ["fine", plant("string list attribute")], dtype=h5py.string_dtype()
            )
            group = handle.create_group("x_vendor")
            group.attrs["who"] = plant("unknown group attribute")
            group.create_dataset(
                "notes",
                data=np.array(
                    ["fine", plant("vlen strings")], dtype=h5py.string_dtype()
                ),
            )
            group.create_dataset(
                "fixed", data=np.array([plant("fixed strings").encode()], dtype="S40")
            )

        before = path.read_bytes()
        assert all(name.encode() in before for name in planted.values())
        scanned = scrub.scan(path, profile="strict")
        values = " ".join(f.value or "" for f in scanned.findings)
        assert [slot for slot, name in planted.items() if name not in values] == []

        report = scrub.apply(path, profile="strict", salt="pep")
        after = path.read_bytes()
        remaining = " ".join(f.value or "" for f in report.remaining)
        left = {slot for slot, name in planted.items() if name.encode() in after}
        assert {slot for slot in left if planted[slot] not in remaining} == set()
        # What stays is what --apply must not rewrite: the shared vocabulary and
        # the id a split manifest joins on --- each reported, never silent.
        assert left == {"label name", "cat", "set_id"}
        assert report.ok
        assert validate_file(path).ok


class TestW15ScrubRules:
    """The rules the traversal applies, one place each."""

    def _with_inputs(self, path: Path, *refs: str) -> Path:
        with Numbered.writer(path) as w:
            tool = w.software("converter")
            w.activity("import", agent=tool, inputs=list(refs))
        return path

    def test_F14_a_source_path_keeps_its_file_name_or_goes(self, tmp_path: Path):
        """Export directories routinely name the patient or the site."""
        from medh5.curation import scrub

        refs = (
            "/exports/StElsewhere/MRN12345/ct.nii.gz",
            "nifti:C:\\exports\\site\\Doe^Jane.nii.gz",
            "annotations/seg",
        )
        basic = self._with_inputs(tmp_path / "basic.medh5", *refs)
        found = scrub.scan(basic).findings
        assert [f.rule for f in found if "inputs[0]" in f.where] == ["path"]
        assert "person_name" in {f.rule for f in found if "inputs[1]" in f.where}
        assert not [f for f in found if "inputs[2]" in f.where], "internal refs"
        assert scrub.apply(basic).ok
        with medh5.open(basic) as sample:
            inputs = sample.document.provenance.activities[0].inputs
        assert inputs == ("ct.nii.gz", "nifti:", "annotations/seg")

        strict = self._with_inputs(tmp_path / "strict.medh5", *refs)
        assert scrub.apply(strict, profile="strict").ok
        with medh5.open(strict) as sample:
            inputs = sample.document.provenance.activities[0].inputs
        assert inputs[:2] == ("<path removed>", "nifti:<path removed>")
        assert b"MRN12345" not in strict.read_bytes()

    def test_F14_a_date_without_a_shift_is_removed(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "date.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", date="2023-01-05")
        assert scrub.apply(path).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date is None

    def test_L36_mapping_keys_are_read_as_text(self, tmp_path: Path):
        """`{"Doe^Jane": ...}` names a person as surely as a value does."""
        from medh5.curation import scrub

        path = tmp_path / "keys.medh5"
        uid = "1.2.840.113619.2.55.3.1"
        with Numbered.writer(path) as w:
            w.extra("src", {"Doe^Jane": {"kv": 120}, uid: "CT", "2023-01-05": 1})
        found = {(f.rule, f.actionable) for f in scrub.scan(path).findings}
        assert ("person_name", True) in found
        assert ("uid", True) in found
        assert ("date", False) in found
        report = scrub.apply(path)
        assert report.ok
        with medh5.open(path) as sample:
            source = sample.document.extra["src"]
        assert "Doe^Jane" not in source
        assert source[report.uid_map[uid]] == "CT"

    def test_L36_a_compound_dataset_with_text_is_reported_not_skipped(
        self, tmp_path: Path
    ):
        from medh5.curation import scrub

        path = Numbered.plain(tmp_path / "compound.medh5")
        record = np.dtype([("n", "i4"), ("who", h5py.string_dtype())])
        with h5py.File(path, "r+") as handle:
            handle.create_dataset("x_log", data=np.array([(1, "Doe^Jane")], record))
        (finding,) = [f for f in scrub.scan(path).findings if f.where == "x_log"]
        assert finding.rule == "unreadable" and not finding.actionable

    def test_L36_an_id_is_reviewed_and_never_renamed(self, tmp_path: Path):
        """Renaming one end of a reference breaks it; a person re-mints ids."""
        from medh5.curation import scrub

        uid = "1.2.840.113619.2.55.3"
        path = tmp_path / "ids.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid(uid, shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid=uid, modality="CT"
            )
        report = scrub.scan(path, profile="strict")
        where = {f.where: f for f in report.findings if f.rule == "uid"}
        assert f"grids.{uid}" in where and "images.CT.grid" in where
        assert not any(f.actionable for f in where.values())
        assert scrub.apply(path, profile="strict").ok
        with medh5.open(path) as sample:
            assert sample.images["CT"].grid_id == uid
        assert validate_file(path).ok

    def test_F14_the_group_id_follows_the_subject_id(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "group.medh5"
        with Numbered.writer(path) as w:
            w.cohort(group_id="subj-1")
        report = scrub.apply(path, salt="pep", pseudonymise_ids=True)
        with medh5.open(path) as sample:
            subject = sample.identity.subject_id
            assert subject == report.uid_map["subj-1"]
            assert sample.document.cohort.group_id == subject

    def test_L36_a_rewritten_string_dataset_keeps_its_filters(self, tmp_path: Path):
        """Widening a fixed-length dataset kept its compression and dropped the
        rest: a checksummed column lost its Fletcher32 to a scrub."""
        import hdf5plugin

        from medh5.curation import scrub

        path = Numbered.plain(tmp_path / "filters.medh5")
        uid = b"1.2.840.1.2"  # 11 bytes, in a 12-byte column: its pseudonym is 39
        with h5py.File(path, "r+") as handle:
            group = handle.create_group("x_vendor")
            narrow = group.create_dataset(
                "narrow",
                data=np.array([uid, b"ok"], dtype="S12"),
                chunks=(2,),
                shuffle=True,
                fletcher32=True,
                compression="gzip",
            )
            narrow.attrs["about"] = "series"
            group.create_dataset(
                "codec",
                data=np.array([uid, b"ok"], dtype="S12"),
                chunks=(2,),
                **hdf5plugin.Zstd(clevel=5),
            )
            group.create_dataset(
                "wide", data=np.array([uid, b"ok"], dtype="S64"), chunks=(2,)
            )

        def pipeline(dataset: Any) -> list[int]:
            plist = dataset.id.get_create_plist()
            return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]

        with h5py.File(path, "r") as handle:
            before = {n: pipeline(handle[f"x_vendor/{n}"]) for n in handle["x_vendor"]}
        report = scrub.apply(path)
        assert report.ok
        pseudonym = report.uid_map[uid.decode()].encode()
        with h5py.File(path, "r") as handle:
            group = handle["x_vendor"]
            assert sorted(group) == ["codec", "narrow", "wide"]
            for name, filters in before.items():
                assert pipeline(group[name]) == filters, name
                assert group[name].chunks == (2,)
                assert list(group[name][...]) == [pseudonym, b"ok"]
            assert group["narrow"].dtype.itemsize == len(pseudonym)
            assert group["wide"].dtype == np.dtype("S64")
            assert group["narrow"].attrs["about"] == "series"

    def test_scan_document_reads_only_the_document(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "doc.medh5"
        with Numbered.writer(path) as w:
            w.extra("src", {"PatientName": "Doe^Jane"})
        with medh5.open(path) as sample:
            report = scrub.scan_document(
                sample.document, scrub.ScrubReport(path=str(path))
            )
        assert [f.rule for f in report.findings] == ["identifier"]


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

    def test_S11_4_the_original_uid_does_not_survive_in_freed_space(self, dirty):
        """A scrubbed file must not still contain the UID it pseudonymised.

        The amend copies each object and *then* rewrites the attribute, and HDF5
        never reclaims what it supersedes -- so the released file carried the
        real FrameOfReferenceUID in freed space, findable with ``strings``,
        while every API read returned the pseudonym.  A UID links back to the
        originating study in the source PACS, so this is the difference between
        a de-identified file and one that only looks it.
        """
        original = b"1.2.840.10008.3.1.2.9"
        assert original in dirty.read_bytes()
        scrubber.apply(dirty)
        assert original not in dirty.read_bytes()
        with medh5.open(dirty) as sample:
            assert sample.grids["g"].frame_uid.startswith("pseudo:")

    def test_S11_4_id_mapping_is_external_only_when_uids_were_mapped(self, tmp_path):
        """`external` is the strongest claim §11.4 offers; don't make it for free."""
        path = tmp_path / "nouid.medh5"
        with medh5.create(path, sample_id="s1", subject_id="subj-A") as w:
            w.add_timepoint("tp0", date="2026-02-03")
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="not-a-dicom-uid",
            )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        scrubber.apply(path, salt="pepper", date_shift_days=-117)
        with medh5.open(path) as sample:
            assert sample.document.deidentification.id_mapping == "none"

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


class TestB08ClinicalIsNotScrubbed:
    """The clinical profile is outside what this tool examines (1.1 §6:
    de-identification covers the body as well as the metadata).  Its text is
    packed UTF-8, which the string sweep never read, and a date shift moved
    `/meta` and not the clinical clock --- yet a strict scan of a clinical
    sample with a name in its report said `clean`, and apply attested a
    de-identification and kept the name.  Until the tool covers the profile, a
    clinical sample is never clean and is not scrubbed."""

    @staticmethod
    def _clinical(tmp_path: Path) -> Path:
        from medh5.clinical import HOUR, Document, Event, Link
        from tests.kits import History

        path = tmp_path / "clinical.medh5"
        note = Event(
            "note",
            "note",
            "document",
            "point",
            "final",
            effective_start_us=0,
            available_us=HOUR,
        )
        History.write(
            path,
            events=[note],
            documents=[Document("note_text", "Seen with Smith^Alice today.")],
            links=[
                Link.between(("event", "note"), "describes", ("document", "note_text"))
            ],
        )
        return path

    @pytest.mark.parametrize("profile", ["basic", "strict"])
    def test_B08_a_clinical_sample_is_never_clean(self, tmp_path: Path, profile):
        report = scrubber.scan(self._clinical(tmp_path), profile=profile)
        assert not report.clean
        found = [f for f in report.findings if f.rule == "clinical"]
        assert len(found) == 1 and not found[0].actionable
        assert "not examined" in found[0].detail
        assert any("clinical" in item for item in report.not_checked)

    @pytest.mark.parametrize(
        "options", [{}, {"profile": "strict"}, {"date_shift_days": -117}]
    )
    def test_B08_apply_refuses_a_clinical_sample(self, tmp_path: Path, options):
        path = self._clinical(tmp_path)
        with medh5.open(path) as sample:
            before = sample.content_id
        with pytest.raises(MEDH5ValidationError, match="clinical strip"):
            scrubber.apply(path, **options)
        with medh5.open(path) as sample:
            assert sample.content_id == before
            assert sample.document.deidentification is None

    def test_B08_the_imaging_projection_is_scrubbed(self, tmp_path: Path):
        from medh5.clinical import strip

        projection = tmp_path / "imaging.medh5"
        strip(self._clinical(tmp_path), projection)
        assert not [
            f for f in scrubber.scan(projection).findings if f.rule == "clinical"
        ]
        report = scrubber.apply(projection, profile="strict")
        assert report.applied
