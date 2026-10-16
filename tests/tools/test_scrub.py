"""De-identification: the identifier sweep, its fixes and its record (§11.4)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.curation import scrub as scrubber
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file
from tests.helpers import SHAPE, encode_attr, write_dicom_series
from tests.kits import Numbered, Organs


def _untimed(findings: list[Any]) -> list[Any]:
    """*findings* but the file's own `created`, which a strict scan reports
    on every file a strict scrub has not cut (10.3)."""
    return [f for f in findings if f.where != "root.created"]


class TestDeidentificationAttestation:
    """The attestation, the scope it covers, and the exit code."""

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

    def test_S11_4_apply_acts_on_everything_it_flags(self, tmp_path: Path):
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

    def test_S11_4_the_record_states_what_the_run_achieved(self, tmp_path: Path):
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "record.medh5")
        scrub.apply(path, profile="strict", date_shift_days=-30, salt="pep")
        with medh5.open(path) as sample:
            activity = sample.document.provenance.activities_by_type("deidentify")[0]
            assert activity.params["remaining_actionable"] == 0
            assert activity.params["changes"] >= activity.params["remaining"]

    def test_apply_is_idempotent_and_stays_green(self, tmp_path: Path):
        """A rule that fires on its own output can never go green in a pipeline."""
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "twice.medh5")
        first = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        second = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        assert first.ok and second.ok
        assert not second.remaining_actionable

    def test_S11_4_the_ids_a_join_needs_are_reported_and_never_rewritten(
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

    def test_the_cli_exit_code_follows_the_re_scan(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._dirty(tmp_path / "cli.medh5")
        assert main(["scrub", str(path)]) == 1
        assert (
            main(
                [
                    "scrub",
                    str(path),
                    "--apply",
                    "--profile",
                    "strict",
                    "--by",
                    "R7",
                    "--salt",
                    "pep",
                ]
            )  # fmt: skip
            == 0
        )
        capsys.readouterr()
        assert main(["scrub", str(path), "--profile", "strict"]) == 0

    @pytest.mark.parametrize(
        "flags",
        [
            ["--salt", "pep"],
            ["--date-shift-days", "-30"],
            ["--by", "R7"],
            ["--salt", "pep", "--by", "R7"],
        ],
    )
    def test_an_apply_option_without_apply_is_refused(
        self, tmp_path: Path, capsys, flags: list[str]
    ):
        """Without `--apply` these were dropped without a word, and a scan
        meant as a salted, shifted scrub exited as a scan."""
        from medh5.cli import main

        path = self._dirty(tmp_path / "flags.medh5")
        with medh5.open(path) as sample:
            before = sample.content_id
        assert main(["scrub", str(path), *flags]) == 2
        err = capsys.readouterr().err
        assert "without --apply" in err
        assert all(flag in err for flag in flags[::2])
        with medh5.open(path) as sample:
            assert sample.content_id == before


class TestDeidentification:
    """The scan and the clean see every string, and say so."""

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

    def test_S11_4_the_importer_says_where_the_ids_came_from(self, dicom_import):
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

    def test_S11_4_a_strict_apply_fails_while_the_ids_are_record_numbers(
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

    def test_S11_4_pseudonymised_ids_leave_no_identifier_bytes(self, dicom_import):
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

    def test_S11_4_pseudonymising_ids_needs_a_salt(self, dicom_import):
        """An unsalted hash of a record number is reversed by hashing them all."""
        from medh5.curation import scrub

        with pytest.raises(MEDH5ValidationError, match="salt"):
            scrub.apply(dicom_import["path"], pseudonymise_ids=True)

    def test_S11_4_a_file_named_after_the_record_number_is_reported(
        self, dicom_import, tmp_path: Path
    ):
        from medh5.curation import scrub

        path = tmp_path / "MRN12345.medh5"
        dicom_import["path"].rename(path)
        assert "file_name" in {f.rule for f in scrub.scan(path).findings}
        report = scrub.apply(path, profile="strict", salt="pep", pseudonymise_ids=True)
        assert [f.rule for f in report.remaining] == ["file_name"]
        assert not report.ok

    def test_the_cli_flag(self, dicom_import, capsys):
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

    def test_S11_4_reminting_an_id_retires_its_recorded_source(self, tmp_path: Path):
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
    def test_S11_4_a_person_name_in_any_script(self, tmp_path: Path, name: str):
        from medh5.curation import scrub

        path = tmp_path / "pn.medh5"
        with Numbered.writer(path) as w:
            w.extra("src", {"reader": name})
        assert [f.rule for f in scrub.scan(path).actionable] == ["person_name"]

    def test_S11_4_an_age_over_89_and_an_organization(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "age.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", subject_age_years=93.0)
            w.organization("St Elsewhere General")
        found = {f.rule: f for f in scrub.scan(path).findings}
        assert not found["age"].actionable
        assert not found["organization_name"].actionable
        assert found["age"].where == "timepoints[0].subject_age_years"
        assert scrub.apply(path, profile="strict", salt="pep").ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].subject_age_years == 90.0
            names = [a.name for a in sample.document.provenance.agents]
            assert "St Elsewhere General" not in names

    def test_S11_4_an_identifying_attribute_on_an_hdf5_object(self, tmp_path: Path):
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

    def test_S11_4_a_replaced_id_is_replaced_wherever_a_string_holds_it(
        self, tmp_path: Path
    ):
        """`--pseudonymise-ids` replaced the ids in `identity` alone, and the
        record number outlived its pseudonym in a label, a note and an
        attribute --- which no later scan can tell is one."""
        from medh5.curation import scrub

        record = "MRN-000042"
        path = tmp_path / "ids.medh5"
        with medh5.create(path, sample_id=record, subject_id=record) as w:
            w.add_timepoint("tp0", label=f"baseline of {record}")
            w.add_grid(
                "g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
            w.extra("src", {"note": f"case {record}.nii", "other": f"{record}1"})
        with h5py.File(path, "r+") as handle:
            handle["images/CT"].attrs["x_case"] = f"{record}/ct"
        report = scrub.apply(path, profile="strict", salt="pep", pseudonymise_ids=True)
        assert report.ok, report.format()
        pseudonym = report.uid_map[record]
        with medh5.open(path) as sample:
            document = sample.document
            assert document.timepoints["tp0"].label == f"baseline of {pseudonym}"
            assert document.extra["src"]["note"] == f"case {pseudonym}.nii"
            # A longer token is another value, and is left as it is.
            assert document.extra["src"]["other"] == f"{record}1"
        with h5py.File(path, "r") as handle:
            assert handle["images/CT"].attrs["x_case"] == f"{pseudonym}/ct"
        assert re.findall(rb"MRN-000042(?!1)", path.read_bytes()) == []

    def test_S11_4_an_id_too_short_to_tell_from_a_word_is_not_sought(
        self, tmp_path: Path
    ):
        from medh5.curation import scrub

        path = tmp_path / "short.medh5"
        with Numbered.writer(path) as w:
            w.extra("src", {"note": "s1 of subj-1"})
        report = scrub.apply(path, salt="pep", pseudonymise_ids=True)
        with medh5.open(path) as sample:
            note = sample.document.extra["src"]["note"]
        assert note == f"s1 of {report.uid_map['subj-1']}"

    def test_S11_4_a_replaced_id_in_an_object_name_is_left_for_a_person(
        self, tmp_path: Path
    ):
        """A name is never rewritten, so the run that knows the old id says
        where it still is --- a later scan cannot know to look."""
        from medh5.curation import scrub

        path = tmp_path / "named.medh5"
        with medh5.create(path, sample_id="MRN-000042", subject_id="subj-1") as w:
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
        with h5py.File(path, "r+") as handle:
            handle.create_group("x_MRN-000042")
        report = scrub.apply(path, profile="strict", salt="pep", pseudonymise_ids=True)
        assert [f.where for f in report.open_identity] == ["x_MRN-000042"]
        assert not report.ok

    def test_S11_4_a_strict_name_pseudonym_is_salted_and_recorded(self, tmp_path: Path):
        """Strict replaced a staff or site name with an unsalted hash of it,
        which hashing candidate names reverses, and kept no key to it."""
        from medh5.curation import scrub

        path = tmp_path / "staff.medh5"
        with Numbered.writer(path) as w:
            w.organization("St Elsewhere General")
            w.person("Dr Alice Roe")
        report = scrub.apply(path, profile="strict", salt="pep")
        assert report.ok
        names = ("St Elsewhere General", "Dr Alice Roe")
        expected = {name: scrub.pseudonymise(name, "pep") for name in names}
        assert {name: report.uid_map[name] for name in names} == expected
        with medh5.open(path) as sample:
            agents = {a.name for a in sample.document.provenance.agents}
        assert set(expected.values()) <= agents
        assert scrub.pseudonymise("Dr Alice Roe") not in agents

    def test_S11_4_the_record_and_an_organization_are_scanned(self, tmp_path: Path):
        """Neither was read: a name in the record's profile passed every scan,
        under the attestation that said the file was clean."""
        from medh5.curation import scrub

        path = tmp_path / "record.medh5"
        with Numbered.writer(path) as w:
            person = w.person("Roe", organization="1.2.840.113619.2.55.7")
            w.deidentification(method="manual", profile="checked by Doe^Jane")
        found = {(f.rule, f.where): f.actionable for f in scrub.scan(path).findings}
        assert found[("person_name", "deidentification.profile")] is True
        assert found[("uid", f"provenance.agents[{person.id}].organization")] is False
        report = scrub.apply(path)
        assert report.ok
        assert b"Doe^Jane" not in path.read_bytes()

    def test_S11_4_every_string_slot_is_removed_or_reported(self, tmp_path: Path):
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


class TestScrubRules:
    """The rules the traversal applies, one place each."""

    def _with_inputs(self, path: Path, *refs: str) -> Path:
        with Numbered.writer(path) as w:
            tool = w.software("converter")
            w.activity("import", agent=tool, inputs=list(refs))
        return path

    def test_S11_4_a_source_path_keeps_its_file_name_or_goes(self, tmp_path: Path):
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

    def test_S11_4_a_date_without_a_shift_is_removed(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "date.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", date="2023-01-05")
        assert scrub.apply(path).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date is None

    def test_S11_4_mapping_keys_are_read_as_text(self, tmp_path: Path):
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

    def test_S11_4_a_compound_dataset_with_text_is_reported_not_skipped(
        self, tmp_path: Path
    ):
        from medh5.curation import scrub

        path = Numbered.plain(tmp_path / "compound.medh5")
        record = np.dtype([("n", "i4"), ("who", h5py.string_dtype())])
        with h5py.File(path, "r+") as handle:
            handle.create_dataset("x_log", data=np.array([(1, "Doe^Jane")], record))
        (finding,) = [f for f in scrub.scan(path).findings if f.where == "x_log"]
        assert finding.rule == "unreadable" and not finding.actionable

    def test_S11_4_an_id_is_reviewed_and_never_renamed(self, tmp_path: Path):
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

    def test_S12_2_the_group_id_follows_the_subject_id(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "group.medh5"
        with Numbered.writer(path) as w:
            w.cohort(group_id="subj-1")
        report = scrub.apply(path, salt="pep", pseudonymise_ids=True)
        with medh5.open(path) as sample:
            subject = sample.identity.subject_id
            assert subject == report.uid_map["subj-1"]
            assert sample.document.cohort.group_id == subject

    def test_a_rewritten_string_dataset_keeps_its_filters(self, tmp_path: Path):
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
        # The same places, and the time the file was written, which strict
        # cuts to its year.
        timed = [f.where for f in strict.findings if f.rule == "tool_timestamp"]
        assert timed == ["root.created"]
        assert len(strict.findings) == len(basic.findings) + 1

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

    @pytest.mark.parametrize(
        ("key", "value", "rule", "actionable"),
        [
            ("nhs_number", "943 476 5919", "identifier", True),
            ("birth_date", "19500302", "date", True),
            ("date_of_birth", "1950-03-02", "date", True),
            ("study_uid", "not-a-uid", "uid", False),
            ("series_uid", "not-a-uid", "uid", False),
            ("frame_uid", "not-a-uid", "uid", False),
        ],
    )
    def test_S11_4_a_table_entry_written_with_underscores_is_matched(
        self, tmp_path, key, value, rule, actionable
    ):
        """A lookup compares a key with its underscores and spaces removed,
        and the tables held these as written: none could ever match, so an
        NHS number scanned clean and a date of birth was only text."""
        path = self._with(tmp_path, key, **{key: value})
        found = [
            (f.rule, f.actionable)
            for f in scrubber.scan(path).findings
            if f.where == f"acquisition.CT.{key}"
        ]
        assert found == [(rule, actionable)]

    def test_S11_4_every_key_table_is_stored_as_a_lookup_compares(self):
        tables = (
            scrubber.IDENTIFYING_KEYS,
            scrubber.QUASI_IDENTIFYING_KEYS,
            scrubber.DATE_KEYS,
            scrubber.UID_KEYS,
        )
        for table in tables:
            spelled = {k for k in table if k != k.replace("_", "").replace(" ", "")}
            assert spelled | {k for k in table if k != k.lower()} == set()

    def test_S11_4_a_pseudonym_under_a_uid_key_is_what_the_rule_asks_for(
        self, tmp_path
    ):
        """The rule reported the pseudonym `--apply` wrote, so a scrubbed file
        never scanned clean."""
        uid = "1.2.840.113619.2.55.3.7"
        path = self._with(tmp_path, "uid-key", StudyInstanceUID=uid)
        assert [f.rule for f in scrubber.scan(path).actionable] == ["uid"]
        report = scrubber.apply(path, salt="pep")
        assert report.ok
        with medh5.open(path) as sample:
            acquisition = sample.document.acquisition["CT"]
            assert acquisition["StudyInstanceUID"] == report.uid_map[uid]
        assert scrubber.scan(path).clean

    def test_S11_4_a_top_level_key_is_a_key_like_any_other(self, tmp_path):
        """`extra.src.PatientID` was removed and `extra.PatientID` scanned
        clean: the top level read its keys as ids and its values as text."""
        path = tmp_path / "top.medh5"
        with Numbered.writer(path) as w:
            w.extra("PatientID", "MRN12345")
            w.extra("AccessionNumber", {"value": "A-99213"})
            w.extra("StudyDate", "20200131")
            w.extra("Doe^Jane", {"kv": 120})
            w.extra("org.example", {"kv": 120})
        found = {(f.rule, f.where): f.actionable for f in scrubber.scan(path).findings}
        assert found == {
            ("identifier", "extra.PatientID"): True,
            ("identifier", "extra.AccessionNumber"): True,
            ("date", "extra.StudyDate"): True,
            ("person_name", "extra.Doe^Jane"): True,
        }
        assert scrubber.apply(path, date_shift_days=-10, withhold_shift=True).ok
        with medh5.open(path) as sample:
            extra = sample.document.extra
        assert extra["StudyDate"] == "20200121"
        assert extra["org.example"] == {"kv": 120}
        assert not {"PatientID", "AccessionNumber", "Doe^Jane"} & set(extra)
        assert b"MRN12345" not in path.read_bytes()

    def test_S11_4_a_value_that_is_wholly_a_date_is_one_under_any_key(self, tmp_path):
        """No table names every date attribute: a
        `RadiopharmaceuticalStartDateTime` survived a strict apply that
        reported `ok`."""
        path = tmp_path / "dated.medh5"
        with Numbered.writer(path) as w:
            w.software("conv", version="20240115")
            w.extra(
                "src",
                {
                    "RadiopharmaceuticalStartDateTime": "20200131102200",
                    "injected": "2020-01-31T10:22:00Z",
                    "visits": ["2020-01-31", "2020-02-03"],
                    "note": "seen on 2020-01-31",
                    "code": "52988006",
                },
            )
        found = {
            f.where: f.actionable
            for f in scrubber.scan(path, profile="strict").findings
            if f.rule == "date"
        }
        assert found == {
            "extra.src.RadiopharmaceuticalStartDateTime": True,
            "extra.src.injected": True,
            "extra.src.visits[0]": True,
            "extra.src.visits[1]": True,
            # Text holding a date, and a build's date, are a person's call.
            "extra.src.note": False,
            "provenance.agents[s1].version": False,
        }
        report = scrubber.apply(path, profile="strict", date_shift_days=10)
        assert report.ok
        with medh5.open(path) as sample:
            src = sample.document.extra["src"]
            assert sample.document.provenance.agents[0].version == "20240115"
        assert src["RadiopharmaceuticalStartDateTime"] == "20200210102200"
        assert src["injected"] == "2020-02-10T10:22:00Z"
        assert src["visits"] == ["2020-02-10", "2020-02-13"]
        assert src["note"] == "seen on 2020-01-31"
        assert src["code"] == "52988006"

    def test_S11_4_a_uid_inside_text_is_pseudonymised_as_the_field_is(self, tmp_path):
        """Left in a description, the real UID maps its pseudonym straight
        back."""
        uid = "1.2.840.113619.2.55.3.1234"
        path = tmp_path / "text.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", study_uid=uid)
            w.extra("src", {"description": f"derived from {uid} (recon 2)"})
        found = {(f.rule, f.where, f.actionable) for f in scrubber.scan(path).findings}
        assert ("uid", "extra.src.description", True) in found
        report = scrubber.apply(path, salt="pep")
        assert report.ok
        pseudonym = report.uid_map[uid]
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].study_uid == pseudonym
            described = sample.document.extra["src"]["description"]
        assert described == f"derived from {pseudonym} (recon 2)"
        assert uid.encode() not in path.read_bytes()

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

        scrubber.apply(path, date_shift_days=-117, withhold_shift=True)
        assert not scrubber.scan(path).actionable

    def test_a_swept_file_passes_scrub_as_a_gate(self, tmp_path):
        """`scrub` exits non-zero on findings, so its own output must pass."""
        path = self._with(
            tmp_path, "gate", kvp=120, PatientName="Doe^Jane", StudyDate="20260203"
        )
        scrubber.apply(path, date_shift_days=-117, withhold_shift=True)
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
        scrubber.apply(path, salt="pepper", date_shift_days=-117, withhold_shift=True)
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

    def test_S11_4_a_cohort_scrubbed_file_by_file_still_joins(self, tmp_path):
        """A cohort has to stay joinable after being scrubbed file by file."""

        def visit(name: str) -> Path:
            path = tmp_path / f"{name}.medh5"
            with medh5.create(path, sample_id=name, subject_id="subj-A") as w:
                w.add_timepoint("tp0", study_uid="1.2.840.113619.2.55.3")
                w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
                w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            return path

        def study(path: Path) -> str:
            with medh5.open(path) as sample:
                uid: str = sample.document.timepoints["tp0"].study_uid
            return uid

        first, second, other = visit("a"), visit("b"), visit("c")
        scrubber.apply(first, salt="pepper")
        scrubber.apply(second, salt="pepper")
        scrubber.apply(other, salt="salt")
        assert study(first).startswith(scrubber.PSEUDONYM_PREFIX)
        assert study(first) == study(second)
        assert study(other) != study(first)

    def test_dates_are_dropped_when_no_shift_is_given(self, dirty):
        scrubber.apply(dirty)
        with medh5.open(dirty) as sample:
            assert sample.document.timepoints["tp0"].date is None
            assert "StudyDate" not in sample.document.acquisition["CT"]

    def test_a_shift_preserves_the_interval(self, dirty):
        scrubber.apply(dirty, date_shift_days=-117, withhold_shift=True)
        with medh5.open(dirty) as sample:
            first = sample.document.timepoints["tp0"].date
            second = sample.document.timepoints["tp1"].date
            assert first == "2025-10-09"
            assert second == "2026-01-07"
            assert sample.document.acquisition["CT"]["StudyDate"] == "20251009"
            assert sample.document.timepoints["tp1"].days_from_baseline == 90

    def test_S11_4_the_attestation_says_what_was_not_checked(self, dirty):
        scrubber.apply(
            dirty, date_shift_days=-117, performed_by="RAD-07", withhold_shift=False
        )
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

        scrubber.apply(dirty, date_shift_days=-30, withhold_shift=True)
        assert not validate_file(dirty).errors

    def test_scrubbing_twice_is_stable(self, dirty):
        scrubber.apply(dirty, date_shift_days=-117, withhold_shift=True)
        with medh5.open(dirty) as sample:
            first = sample.document.to_json()
        second_run = scrubber.apply(dirty, date_shift_days=-117, withhold_shift=True)
        with medh5.open(dirty) as sample:
            second = sample.document.to_json()
        assert first["timepoints"] == second["timepoints"]
        assert first["acquisition"] == second["acquisition"]
        assert not [f for f in second_run.findings if f.rule == "identifier"]


class TestScrubContainer:
    """The HDF5 file around `/meta`: every name a link holds, every type that
    can carry a string, and every copy of the document."""

    RECORD = "MRN-000042"

    def _named(self, path: Path) -> Path:
        """A sample whose ids are a record number nothing marks as one."""
        with medh5.create(path, sample_id=self.RECORD, subject_id=self.RECORD) as w:
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
        return path

    @pytest.mark.parametrize("alias", ["x_acme/meta", "a_first/meta", "meta_copy"])
    def test_S11_4_another_name_for_meta_is_reported_and_removed(
        self, tmp_path: Path, alias: str
    ):
        """A hard link to `/meta` is the document under a second name.

        An amendment wrote the new document at `/meta` and carried the old one
        across under the alias, so `--pseudonymise-ids` left the record number
        there --- while the apply reported `ok` and a fresh strict scan was
        clean.
        """
        path = self._named(tmp_path / "alias.medh5")
        with h5py.File(path, "r+") as handle:
            handle[alias] = handle["meta"]
        where = alias.replace("/", ".")
        scanned = _untimed(scrubber.scan(path, profile="strict").findings)
        assert [(f.rule, f.where, f.actionable, f.value) for f in scanned] == [
            ("identity", where, True, self.RECORD)
        ]

        report = scrubber.apply(
            path, profile="strict", salt="pep", pseudonymise_ids=True
        )
        assert report.ok, report.format()
        assert self.RECORD.encode() not in path.read_bytes()
        assert scrubber.scan(path, profile="strict").clean
        with h5py.File(path, "r") as handle:
            assert alias not in handle and "meta" in handle
        with medh5.open(path) as sample:
            assert sample.identity.subject_id == report.uid_map[self.RECORD]

    def test_S11_4_the_command_line_gate_sees_the_alias(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._named(tmp_path / "gate.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_acme/meta"] = handle["meta"]
        strict = ["scrub", str(path), "--profile", "strict"]
        assert main(strict) == 1
        assert "another name for /meta" in capsys.readouterr().out
        assert main([*strict, "--apply", "--salt", "pep", "--pseudonymise-ids"]) == 0
        assert main(strict) == 0

    def test_S11_4_a_soft_link_to_meta_reads_the_new_document(self, tmp_path: Path):
        """A soft link names `/meta` by path, so it follows the document."""
        path = self._named(tmp_path / "soft.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_document"] = h5py.SoftLink("/meta")
        assert not _untimed(scrubber.scan(path, profile="strict").findings)
        report = scrubber.apply(
            path, profile="strict", salt="pep", pseudonymise_ids=True
        )
        assert report.ok
        assert self.RECORD.encode() not in path.read_bytes()
        with h5py.File(path, "r") as handle:
            document = json.loads(handle["x_document"][()])
        assert document["identity"]["subject_id"] == report.uid_map[self.RECORD]

    def test_S11_4_a_copy_of_the_document_is_reported_and_removed(self, tmp_path: Path):
        """A copy keeps every identifier `--apply` replaces in `/meta`."""
        path = self._named(tmp_path / "copy.medh5")
        with h5py.File(path, "r+") as handle:
            document = handle["meta"][()]
            handle.create_group("x_backup").create_dataset("meta", data=document)
            # Found once, at its first name; removed under every name.
            handle["x_backup/again"] = handle["x_backup/meta"]
            handle["images/CT"].attrs["x_document"] = document
        scanned = _untimed(scrubber.scan(path, profile="strict").findings)
        assert {(f.rule, f.where, f.actionable, f.value) for f in scanned} == {
            ("identity", "x_backup.again", True, self.RECORD),
            ("identity", "images.CT.x_document", True, self.RECORD),
        }
        report = scrubber.apply(
            path, profile="strict", salt="pep", pseudonymise_ids=True
        )
        assert report.ok, report.format()
        assert self.RECORD.encode() not in path.read_bytes()
        with h5py.File(path, "r") as handle:
            assert list(handle["x_backup"]) == []
            assert "x_document" not in handle["images/CT"].attrs

    @staticmethod
    def _plant(handle: h5py.File, carrier: str) -> tuple[str, str]:
        """A person name in one carrier: where, and by which rule, it is
        reported."""
        name = "Doe^Jane"
        if carrier == "a second hard link":
            handle.create_group("x_a").create_dataset("table", data=np.arange(3))
            handle[f"x_b/{name}"] = handle["x_a/table"]
            return f"x_b.{name}", "person_name"
        if carrier == "a soft link":
            handle[f"x_{name}"] = h5py.SoftLink("/images/CT")
            return f"x_{name}", "person_name"
        if carrier == "a soft link's target":
            handle["x_points"] = h5py.SoftLink(f"/x_{name}/gone")
            return "x_points (link target)", "person_name"
        if carrier == "a committed type's name":
            handle[f"x_{name}"] = np.dtype("i4")
            return f"x_{name}", "person_name"
        if carrier == "a committed type's attribute":
            handle["x_row"] = np.dtype("i4")
            handle["x_row"].attrs["PatientName"] = name
            return "x_row.PatientName", "identifier"
        if carrier == "a nested compound":
            inner = np.dtype([("who", h5py.string_dtype())])
            row = np.dtype([("n", "i4"), ("about", inner)])
            handle.create_dataset("x_log", data=np.array([(1, (name,))], row))
            return "x_log", "unreadable"
        assert carrier == "an array of strings in a compound"
        row = np.dtype([("n", "i4"), ("who", h5py.string_dtype(), (2,))])
        handle.create_dataset("x_log", data=np.array([(1, (name, "ok"))], row))
        return "x_log", "unreadable"

    @pytest.mark.parametrize(
        "carrier",
        [
            "a second hard link",
            "a soft link",
            "a soft link's target",
            "a committed type's name",
            "a committed type's attribute",
            "a nested compound",
            "an array of strings in a compound",
        ],
    )
    def test_S11_4_every_name_and_string_carrier_is_examined(
        self, tmp_path: Path, carrier: str
    ):
        """Each scanned clean before and after a strict apply: the walk reached
        each object once, by its first name, and read only flat compounds."""
        path = Numbered.plain(tmp_path / "carrier.medh5")
        with h5py.File(path, "r+") as handle:
            reported = self._plant(handle, carrier)
        found = {
            (f.where, f.rule) for f in scrubber.scan(path, profile="strict").findings
        }
        assert reported in found
        report = scrubber.apply(path, profile="strict", salt="pep")
        # Gone from the file's bytes, or still reported: never silent.
        if b"Doe^Jane" in path.read_bytes():
            assert reported in {(f.where, f.rule) for f in report.remaining}

    def test_S11_4_the_walk_ends_where_links_go_round(self, tmp_path: Path):
        """A hard link back to a group, or to the root, is walked once --- and
        every name on the way is still read."""
        path = Numbered.plain(tmp_path / "cycle.medh5")
        with h5py.File(path, "r+") as handle:
            group = handle.create_group("x_loop")
            group["itself"] = group
            group["root"] = handle["/"]
            group["Doe^Jane"] = group
        found = {
            (f.where, f.rule)
            for f in _untimed(scrubber.scan(path, profile="strict").findings)
        }
        assert found == {("x_loop.Doe^Jane", "person_name")}

    def test_S11_4_a_dataset_outside_the_imaging_groups_is_named_if_not_read(
        self, tmp_path: Path
    ):
        """A DICOM header kept as bytes scanned clean: only strings were read,
        and the report did not say what was not."""
        path = Numbered.plain(tmp_path / "bytes.medh5")
        with h5py.File(path, "r+") as handle:
            vendor = handle.create_group("x_acme")
            vendor["header"] = np.frombuffer(b"PatientName=Doe^Jane", np.uint8)
            vendor["blob"] = np.void(b"PatientName=Doe^Jane")
            vendor["calibration"] = np.linspace(0.0, 1.0, 4)
        report = scrubber.scan(path)
        found = {f.where: (f.rule, f.actionable) for f in report.findings}
        assert found == {
            "x_acme.header": ("unreadable", False),
            "x_acme.blob": ("unreadable", False),
        }
        (note,) = [n for n in report.not_checked if "non-string datasets" in n]
        assert all(
            name in note
            for name in ("x_acme/blob", "x_acme/calibration", "x_acme/header")
        )
        # The imaging groups hold numbers by design, and are not listed.
        assert "images/" not in note


class TestScrubDates:
    """A date is found, and shifted, in every form a converter writes one."""

    def _with(self, path: Path, *, date: str | None = None, **extra: Any) -> Path:
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", **({} if date is None else {"date": date}))
            w.extra("src", extra)
        return path

    def test_S11_4_a_date_time_is_found_and_shifted_with_its_time(self, tmp_path):
        """A word boundary after the day missed every ISO date-time.

        `1` and `T` are both word characters, so `2020-01-31T10:22:00` had
        none: in free text it was not reported, under a date key it was
        dropped, and a timepoint's date-time was deleted rather than shifted
        --- while the attestation recorded the shift and the run reported
        nothing left.
        """
        path = self._with(
            tmp_path / "dt.medh5",
            date="2020-01-31T10:00:00",
            note="scanned 2020-01-31T10:22:00 at the north wing",
            AcquisitionDateTime="2020-01-31T10:22:00Z",
            StudyDate="2020-01-31 and 2020-02-03",
            ContentDate="20200131102200.5+0100",
        )
        found = {f.where: f for f in scrubber.scan(path).findings if f.rule == "date"}
        assert set(found) == {
            "timepoints[0].date",
            "extra.src.note",
            "extra.src.AcquisitionDateTime",
            "extra.src.StudyDate",
            "extra.src.ContentDate",
        }
        assert not found["extra.src.note"].actionable, "free text is a person's call"
        report = scrubber.apply(path, date_shift_days=10, withhold_shift=True)
        assert report.ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date == "2020-02-10T10:00:00"
            src = sample.document.extra["src"]
        assert src["AcquisitionDateTime"] == "2020-02-10T10:22:00Z"
        assert src["StudyDate"] == "2020-02-10 and 2020-02-13"
        assert src["ContentDate"] == "20200210102200.5+0100"
        # Reported for review, never rewritten: no rule can tell what a
        # sentence means by a date.
        assert not report.remaining_actionable
        assert "extra.src.note" in {f.where for f in report.remaining}
        assert src["note"] == "scanned 2020-01-31T10:22:00 at the north wing"

    def test_S11_4_a_date_in_a_file_name_is_found(self, tmp_path):
        path = tmp_path / "name.medh5"
        with Numbered.writer(path) as w:
            tool = w.software("converter")
            w.activity("import", agent=tool, inputs=["nifti:CT_2020-01-31.nii.gz"])
        assert "date" in {f.rule for f in scrubber.scan(path).findings}

    def test_S11_4_digits_of_another_script_do_not_crash_the_shift(self, tmp_path):
        """`\\d` matched Devanagari digits, and the shift sliced one in two.

        The panic escaped as `PanicException`, which is not an `Exception`.
        Such a value is not a date this tool can move, so it is dropped.
        """
        path = self._with(tmp_path / "deva.medh5", StudyDate="२०२००१३१")
        report = scrubber.apply(path, date_shift_days=5, withhold_shift=True)
        assert report.ok
        with medh5.open(path) as sample:
            assert "StudyDate" not in sample.document.extra["src"]
        with pytest.raises(MEDH5ValidationError) as exc:
            self._with(tmp_path / "tp.medh5", date="२०२०-०१-३१")
        assert exc.value.code == "E604"

    def test_S11_4_an_ontology_code_is_not_a_date(self, tmp_path):
        """Eight digits that name no day are a code, not a date.

        `52988006` (SNOMED CT) was reported as a date in the label set, a
        finding no `--apply` can clear, so the scrub gate stayed red.
        """
        from medh5.labels import OntologyCode

        path = tmp_path / "codes.medh5"
        with Numbered.writer(path) as w:
            w.label_set(
                LabelSet(
                    "codes",
                    version="1.0.0",
                    classes=[
                        LabelClass(
                            1,
                            "lesion",
                            "Lesion",
                            codes=(OntologyCode("SNOMED-CT", "52988006"),),
                        )
                    ],
                )
            )
        assert scrubber.scan(path).clean

    def test_S11_4_a_shift_of_nothing_is_refused(self, tmp_path):
        """Recorded, a shift of 0 moved no date and silenced every later scan."""
        from medh5.cli import main

        path = self._with(tmp_path / "zero.medh5", date="2020-01-31")
        with medh5.open(path) as sample:
            before = sample.content_id
        with pytest.raises(MEDH5ValidationError, match="shifts nothing"):
            scrubber.apply(path, date_shift_days=0)
        assert main(["scrub", str(path), "--apply", "--date-shift-days", "0"]) == 2
        with medh5.open(path) as sample:
            assert sample.content_id == before
            assert sample.document.deidentification is None

    def test_S11_4_a_recorded_shift_of_nothing_is_no_shift(self, tmp_path):
        """A file that recorded 0 is read as unshifted, and is not rewritten
        to say so."""
        path = tmp_path / "zero-record.medh5"
        with Numbered.writer(path) as w:
            w.add_timepoint("tp0", date="2020-01-31")
            w.deidentification(method="manual", date_shift_days=0)
        found = [f for f in scrubber.scan(path).findings if f.rule == "date"]
        assert [(f.where, f.actionable) for f in found] == [
            ("timepoints[0].date", True)
        ]
        with medh5.open(path) as sample:
            assert sample.document.deidentification.date_shift_days == 0
        assert scrubber.apply(path, date_shift_days=-30, withhold_shift=False).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date == "2020-01-01"
            assert sample.document.deidentification.date_shift_days == -30

    def test_S11_4_a_date_added_after_the_shift_is_reported(self, tmp_path):
        """The record silenced the date scan, so a visit added later kept its
        real date under an attestation that every date had moved."""
        path = tmp_path / "later.medh5"
        with medh5.create(path, sample_id="s1", subject_id="subj-1") as w:
            w.add_timepoint("tp0", date="2020-01-31")
            w.add_grid(
                "g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
        assert scrubber.apply(path, date_shift_days=-10, withhold_shift=False).ok
        with medh5.amend(path) as w:
            # Shifted by hand, as the record says: it agrees.
            w.add_timepoint("tp1", date="2020-04-20", days_from_baseline=90)
            # The real date, ten days from what the others imply.
            w.add_timepoint("tp2", date="2020-07-29", days_from_baseline=180)
        dates = [f for f in scrubber.scan(path).findings if f.rule == "date"]
        assert [(f.where, f.actionable) for f in dates] == [
            ("timepoints[2].date", False)
        ]
        assert "10 day(s) off" in dates[0].detail
        # A run with the recorded shift leaves the dates alone, and says so.
        again = scrubber.apply(path, date_shift_days=-10, withhold_shift=False)
        assert [f.where for f in again.remaining if f.rule == "date"] == [
            "timepoints[2].date"
        ]

    def test_S11_4_a_second_shift_that_differs_is_refused(self, tmp_path):
        """It was ignored: the dates stayed where the first shift put them, and
        the caller who asked for another was told nothing."""
        from medh5.cli import main

        path = self._with(tmp_path / "twice.medh5", date="2020-01-31")
        assert scrubber.apply(path, date_shift_days=117, withhold_shift=False).ok
        with medh5.open(path) as sample:
            before = sample.content_id
        with pytest.raises(MEDH5ValidationError, match="records a date shift of 117"):
            scrubber.apply(path, date_shift_days=-30)
        assert main(["scrub", str(path), "--apply", "--date-shift-days", "-30"]) == 1
        with medh5.open(path) as sample:
            assert sample.content_id == before
        # The recorded shift, or none, is the one the dates already had.
        assert scrubber.apply(path, date_shift_days=117, withhold_shift=False).ok
        assert scrubber.apply(path, withhold_shift=False).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date == "2020-05-27"
            assert sample.document.deidentification.date_shift_days == 117


class TestClinicalScrub:
    """10.6: the clinical profile is examined --- its clock, its ids and its
    text, one document at a time --- and `apply` writes it again as a new
    sample's records (D15): times moved by the one shift, a `utc` clock become
    `shifted_utc`, every id re-minted from the salt.  Text is never rewritten:
    whoever de-identified it attests that, by the digests of its bytes."""

    NOTE = "Seen with Smith^Alice today."
    CLEAN = "Seen in clinic today; no change."

    @staticmethod
    def _clinical(
        tmp_path: Path,
        *,
        text: str = NOTE,
        clock: Any = None,
        name: str = "visit.medh5",
        subject_id: str = "P-01",
    ) -> Path:
        from medh5.clinical import HOUR, Document, Event, Link
        from tests.kits import History

        path = tmp_path / name
        note = Event(
            "note",
            "note",
            "document",
            "point",
            "final",
            effective_start_us=0,
            available_us=HOUR,
            encounter_id="ENC-7781",
        )
        History.write(
            path,
            subject_id=subject_id,
            clock=clock,
            events=[note],
            documents=[Document("note_text", text)],
            links=[
                Link.between(("event", "note"), "describes", ("document", "note_text"))
            ],
        )
        return path

    @staticmethod
    def _utc(tmp_path: Path, **options: Any) -> Path:
        from medh5.clinical import Clock

        options.setdefault("text", TestClinicalScrub.CLEAN)
        return TestClinicalScrub._clinical(tmp_path, clock=Clock.utc("ehr"), **options)

    @staticmethod
    def _rules(report: Any, rule: str) -> list[Any]:
        return [f for f in report.findings if f.rule == rule]

    @staticmethod
    def _plant(path: Path, document_id: str, old: bytes, new: bytes) -> None:
        """Replace *old* with *new* (as long) in one document's stored bytes."""
        with h5py.File(path, "r+") as handle:
            table = handle["clinical/documents"]
            ids = table["document_id/data"][()]
            offsets = table["document_id/offsets"][()]
            names = [
                bytes(ids[a:b]).decode()
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
            row = names.index(document_id)
            text = table["text/data"]
            # Plain ints: NumPy 1's uint64 plus an int is a float, which h5py
            # refuses as an index.
            start, end = (int(o) for o in table["text/offsets"][row : row + 2])
            stored = bytes(text[start:end])
            assert len(old) == len(new) and old in stored
            at = start + stored.index(old)
            text[at : at + len(new)] = np.frombuffer(new, np.uint8)

    @pytest.mark.parametrize("profile", ["basic", "strict"])
    def test_S11_4_a_clinical_sample_is_never_clean(self, tmp_path: Path, profile):
        report = scrubber.scan(self._clinical(tmp_path), profile=profile)
        assert not report.clean
        unattested = self._rules(report, "clinical")
        # One per document, and one for the short text columns.
        assert sorted(f.where for f in unattested) == [
            "clinical/documents/note_text",
            "clinical/documents/rep_text_v1",
            "clinical/documents/rep_text_v2",
            "clinical/events",
        ]
        assert not any(f.actionable for f in unattested)
        assert all("not attested" in f.detail for f in unattested)
        (name,) = [
            f for f in self._rules(report, "clinical_text") if "name" in f.detail
        ]
        assert name.where == "clinical/documents/note_text"
        assert name.value == "Smith^Alice" and "[10, 21)" in name.detail
        ids = {f.where: f for f in self._rules(report, "clinical_id")}
        assert ids["clinical/meta.clock.id"].value == "clock"
        assert ids["clinical/events.encounter_id"].value == "ENC-7781"

    def test_S6_text_is_scanned_one_document_at_a_time(self, tmp_path: Path):
        """Each document is read on its own: one whose bytes do not read is
        reported, and the others are still examined."""
        path = self._clinical(tmp_path)
        self._plant(path, "rep_text_v1", b"B", b"\xff")
        report = scrubber.scan(path)
        (unread,) = self._rules(report, "unreadable")
        assert unread.where == "clinical/documents/rep_text_v1"
        assert any(
            f.where == "clinical/documents/note_text"
            for f in self._rules(report, "clinical_text")
        )
        with pytest.raises(MEDH5ValidationError, match="does not read"):
            scrubber.apply(path, salt="pep", attest_text="philter")

    def test_S6_a_name_in_a_report_blocks_apply(self, tmp_path: Path):
        path = self._clinical(tmp_path)
        with medh5.open(path) as sample:
            before = sample.content_id
        with pytest.raises(MEDH5ValidationError, match="a name, a UID") as refused:
            scrubber.apply(path, salt="pep", attest_text="philter")
        assert "note_text" in str(refused.value)
        with medh5.open(path) as sample:
            assert sample.content_id == before
            assert sample.document.deidentification is None

    def test_S6_unattested_text_refuses_apply(self, tmp_path: Path):
        path = self._clinical(tmp_path, text=self.CLEAN)
        with pytest.raises(MEDH5ValidationError, match="attest-text"):
            scrubber.apply(path, salt="pep")
        with medh5.open(path) as sample:
            held = sample.clinical.records()
        report = scrubber.apply(path, salt="pep", attest_text="philter 2.1")
        assert report.ok and self._rules(report, "clinical")
        assert not [f for f in report.remaining if f.rule == "clinical"]
        assert any("philter 2.1" in n for n in scrubber.scan(path).not_checked)
        assert validate_file(path).ok
        with medh5.open(path) as sample:
            # A relative clock and ids left alone: nothing under clinical/
            # was touched, only /meta records the attestation.
            assert sample.clinical.records() == held
            (attested,) = [
                a
                for a in sample.document.provenance.activities_by_type("deidentify")
                if a.params.get("scope") == "clinical_text"
            ]
            texts = {
                d.document_id: sample.clinical.text(d.document_id)
                for d in held.documents
            }
        assert attested.params["tool"] == "philter 2.1"
        digests = attested.params["digests"]
        assert sorted(digests) == [
            "clinical/documents/note_text",
            "clinical/documents/rep_text_v1",
            "clinical/documents/rep_text_v2",
            "clinical/events",
        ]
        for document_id, text in texts.items():
            digest = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
            assert digests[f"clinical/documents/{document_id}"] == digest

    def test_S6_an_attestation_lapses_when_the_text_changes(self, tmp_path: Path):
        path = self._clinical(tmp_path, text=self.CLEAN)
        scrubber.apply(path, salt="pep", attest_text="philter")
        assert not self._rules(scrubber.scan(path), "clinical")
        self._plant(path, "note_text", b"clinic", b"clinik")
        (lapsed,) = self._rules(scrubber.scan(path), "clinical")
        assert lapsed.where == "clinical/documents/note_text"

    def test_S3_utc_becomes_shifted_utc_by_the_one_shift(self, tmp_path: Path):
        from medh5.clinical import DAY

        path = self._utc(tmp_path)
        assert self._rules(scrubber.scan(path), "clinical_clock")[0].actionable
        with pytest.raises(MEDH5ValidationError, match="--date-shift-days"):
            scrubber.apply(path, salt="pep", attest_text="philter")
        with pytest.raises(MEDH5ValidationError, match="--salt"):
            scrubber.apply(
                path, date_shift_days=-30, attest_text="philter", withhold_shift=True
            )
        with medh5.open(path) as sample:
            before = sample.clinical.records()
        out = tmp_path / "shared.medh5"
        report = scrubber.apply(
            path,
            salt="pep",
            date_shift_days=-30,
            withhold_shift=True,
            attest_text="philter",
            out=out,
        )
        assert report.ok and validate_file(out).ok
        assert not self._rules(scrubber.scan(out), "clinical_clock")
        with medh5.open(out) as sample:
            after = sample.clinical.records()
            assert sample.document.deidentification.date_shift_days is None
        assert after.clock.reference == "shifted_utc"
        assert after.clock.id.startswith("pseudo.clock.")
        moved = {report.uid_map[f"event:{e.event_id}"]: e for e in before.events}
        assert len(moved) == len(after.events)
        for event in after.events:
            was = moved[event.event_id]
            for name in ("effective_start_us", "effective_end_us", "available_us"):
                old, new = getattr(was, name), getattr(event, name)
                if old is None:
                    assert new is None
                else:
                    assert new == (old[0] - 30 * DAY, old[1] - 30 * DAY)
            assert event.value_text == was.value_text
        # The text is the same bytes, so every span still points at it.
        assert sorted(d.text for d in after.documents) == sorted(
            d.text for d in before.documents
        )

    def test_S3_fragments_scrubbed_with_one_salt_share_a_clock(self, tmp_path: Path):
        options = {
            "date_shift_days": -30,
            "withhold_shift": True,
            "attest_text": "philter",
        }
        clocks = []
        for name, salt in (("a", "pep"), ("b", "pep"), ("c", "other")):
            path = self._utc(tmp_path, name=f"{name}.medh5")
            scrubber.apply(path, salt=salt, **options)
            with medh5.open(path) as sample:
                clocks.append(sample.clinical.clock.id)
        assert clocks[0] == clocks[1] != clocks[2]

    def test_S3_two_subjects_sharing_a_clock_id_get_different_ones(
        self, tmp_path: Path
    ):
        # Each subject takes its own shift from the range, so one new id for
        # both would claim their shifted times share a clock.
        clocks = []
        for subject in ("P-01", "P-02"):
            path = self._utc(tmp_path, name=f"{subject}.medh5", subject_id=subject)
            scrubber.apply(
                path,
                salt="pep",
                date_shift_range=(-400, -30),
                withhold_shift=True,
                attest_text="philter",
            )
            with medh5.open(path) as sample:
                clocks.append(sample.clinical.clock.id)
        assert clocks[0] != clocks[1]

    def test_S7_3_reminted_ids_keep_every_link_resolving(self, tmp_path: Path):
        path = self._clinical(tmp_path, text=self.CLEAN)
        report = scrubber.apply(
            path, salt="pep", pseudonymise_ids=True, attest_text="philter"
        )
        result = validate_file(path)
        assert result.ok, result
        assert not [d for d in result.diagnostics if d.code == "E813"]
        with medh5.open(path) as sample:
            records = sample.clinical.records()
            sample_id = sample.document.identity.sample_id
        assert sample_id.startswith("pseudo:")
        # A relative clock keeps its times; its id is re-minted all the same.
        assert records.clock.reference == "relative"
        assert records.clock.id.startswith("pseudo.clock.")
        events = {e.event_id for e in records.events}
        documents = {d.document_id for d in records.documents}
        assert all(i.startswith("pseudo.event.") for i in events)
        assert all(i.startswith("pseudo.document.") for i in documents)
        for link in records.links:
            for kind, found in (
                (link.source_type, link.source_id),
                (link.target_type, link.target_id),
            ):
                if kind == "event":
                    assert found in events
                elif kind == "document":
                    assert found in documents
        mapped = report.uid_map
        assert any(
            (link.source_id, link.target_id)
            == (mapped["event:rep_v2"], mapped["event:rep_v1"])
            for link in records.links
            if link.relation == "supersedes"
        )
        (record,) = {
            e.record_id
            for e in records.events
            if e.event_id in (mapped["event:rep_v1"], mapped["event:rep_v2"])
        }
        assert record == mapped["record:rep"]
        encounters = {e.encounter_id for e in records.events} - {None}
        assert encounters == {mapped["encounter:ENC-7781"]}
        # The attestation followed the documents to their new ids.
        assert not [f for f in report.remaining if f.rule == "clinical"]
        again = scrubber.apply(
            path, salt="pep", pseudonymise_ids=True, attest_text="philter"
        )
        assert again.ok
        with medh5.open(path) as sample:
            assert sample.clinical.records() == records

    def test_S6_an_attestation_follows_the_reminted_documents(self, tmp_path: Path):
        """Attested by one run and derived by a later one, each text keeps the
        tool that attested it, under its new id."""
        path = self._clinical(tmp_path, text=self.CLEAN)
        scrubber.apply(path, salt="pep", attest_text="philter")
        report = scrubber.apply(path, salt="pep", pseudonymise_ids=True)
        assert report.ok
        assert not [f for f in report.remaining if f.rule == "clinical"]
        with medh5.open(path) as sample:
            activities = sample.document.provenance.activities_by_type("deidentify")
            documents = {d.document_id for d in sample.clinical.documents}
        carried = [a for a in activities if a.params.get("scope") == "clinical_text"]
        assert [a.params["tool"] for a in carried] == ["philter", "philter"]
        digests = carried[-1].params["digests"]
        keys = {k.removeprefix("clinical/documents/") for k in digests}
        assert keys - {"clinical/events"} == documents

    def test_S9_a_shifted_cutoff_admits_the_same_versions(self, tmp_path: Path):
        from medh5.clinical import DAY, HOUR

        path = self._utc(tmp_path)
        cutoffs = [-47 * HOUR, 0, 3 * HOUR, 5 * HOUR, 49 * HOUR, 91 * DAY]
        with medh5.open(path) as sample:
            before = [sample.clinical.select(t).event_ids for t in cutoffs]
        report = scrubber.apply(
            path,
            salt="pep",
            date_shift_range=(-400, -100),
            attest_text="philter",
        )
        with medh5.open(path) as sample:
            first = sample.clinical.records().events
        shift = None
        with medh5.open(path) as sample:
            for event in first:
                if event.event_id == report.uid_map["event:ct0"]:
                    shift = event.effective_start_us[0]
            assert shift is not None and shift % DAY == 0 and shift < 0
            after = [set(sample.clinical.select(t + shift).event_ids) for t in cutoffs]
        # The same versions; ties order by id, and the ids are new.
        assert after == [{report.uid_map[f"event:{e}"] for e in ids} for ids in before]
        assert any(before)

    def test_S11_4_the_clinical_rule_still_blocks_the_guide_gate(self, tmp_path: Path):
        """The guide's gate: what is not examined, and a name or a date in the
        clinical text, block until a person has dealt with it."""
        not_examined = {"clinical", "unreadable", "too_deep"}
        review = {"clinical_text"}

        def blocked(report: Any) -> bool:
            return bool(
                report.actionable
                or report.open_identity
                or any(f.rule in not_examined | review for f in report.findings)
            )

        unattested = scrubber.scan(self._clinical(tmp_path, text=self.CLEAN))
        assert {"clinical"} <= {f.rule for f in unattested.findings}
        assert blocked(unattested)
        named = self._clinical(tmp_path, name="named.medh5")
        assert blocked(scrubber.scan(named))
        clean = self._clinical(tmp_path, text=self.CLEAN, name="clean.medh5")
        scrubber.apply(clean, salt="pep", pseudonymise_ids=True, attest_text="philter")
        assert not blocked(scrubber.scan(clean, profile="basic"))

    def test_S11_4_an_undeclared_clinical_group_is_not_examined_either(
        self, tmp_path: Path
    ):
        """The check read the declaration, not the file: a ``clinical`` group
        whose profile was not declared (E803) scanned without a word about
        it, and ``apply`` was refused only by the commit's validation."""
        path = self._clinical(tmp_path)
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_profiles"] = encode_attr(["core"])
        report = scrubber.scan(path)
        found = [f for f in report.findings if f.rule == "clinical"]
        assert len(found) == 1 and not found[0].actionable
        assert "does not declare" in found[0].detail
        with pytest.raises(MEDH5ValidationError, match="clinical strip"):
            scrubber.apply(path)

    def test_S11_4_the_imaging_projection_is_scrubbed(self, tmp_path: Path):
        from medh5.clinical import strip

        projection = tmp_path / "imaging.medh5"
        strip(self._clinical(tmp_path), projection)
        assert not [
            f
            for f in scrubber.scan(projection).findings
            if f.rule.startswith("clinical")
        ]
        report = scrubber.apply(projection, profile="strict", salt="pep")
        assert report.applied


class TestScrubPaths:
    """`scrub_paths`, the batch: every path, every report, no silent option."""

    def test_every_path_is_processed_and_every_report_kept(self, tmp_path: Path):
        """The first refusal raised at once: the reports of the files already
        rewritten --- their `uid_map` the only key to the pseudonyms --- went
        with it, and the files after it were never scrubbed."""
        first = Numbered.plain(tmp_path / "a.medh5")
        clinical = TestClinicalScrub._clinical(tmp_path)
        last = Numbered.plain(tmp_path / "b.medh5")
        with pytest.raises(MEDH5ValidationError, match="clinical strip") as refused:
            scrubber.scrub_paths(
                [first, clinical, last], apply_changes=True, salt="pep"
            )
        reports = refused.value.reports  # type: ignore[attr-defined]
        assert [report is None for report in reports] == [False, True, False]
        assert reports[0].applied and reports[2].applied
        with medh5.open(last) as sample:
            assert sample.document.deidentification is not None

    def test_a_misspelt_option_cannot_become_a_scan(self, tmp_path: Path):
        """`apply=True` went into `**options`, unread, and the files were
        scanned instead of scrubbed."""
        path = Numbered.plain(tmp_path / "a.medh5")
        with pytest.raises(TypeError, match="apply"):
            scrubber.scrub_paths([path], apply=True)  # type: ignore[call-arg]
        with medh5.open(path) as sample:
            assert sample.document.deidentification is None

    @pytest.mark.parametrize(
        "option",
        [
            {"salt": "pep"},
            {"date_shift_days": -30},
            {"performed_by": "R7"},
            {"pseudonymise_ids": True},
        ],
    )
    def test_an_apply_option_without_apply_changes_is_refused(
        self, tmp_path: Path, option: dict[str, Any]
    ):
        path = Numbered.plain(tmp_path / "a.medh5")
        with pytest.raises(MEDH5ValidationError, match="apply_changes=True"):
            scrubber.scrub_paths([path], **option)

    def test_a_scan_and_an_apply(self, tmp_path: Path):
        paths = [Numbered.plain(tmp_path / f"{name}.medh5") for name in "ab"]
        scanned = scrubber.scrub_paths(paths, profile="strict")
        assert [(r.applied, r.profile) for r in scanned] == [(False, "strict")] * 2
        applied = scrubber.scrub_paths(
            paths, apply_changes=True, date_shift_days=-30, withhold_shift=True
        )
        assert [(r.applied, r.ok) for r in applied] == [(True, True)] * 2


def _dated(path: Path, *, subject_id: str = "subj-1", sample_id: str = "s1") -> Path:
    """A sample with one dated timepoint, of *subject_id*."""
    with medh5.create(path, sample_id=sample_id, subject_id=subject_id) as w:
        w.add_timepoint("tp0", date="2020-01-31")
        w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
        w.add_image("CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT")
    return path


def _deidentify(path: Path) -> Any:
    with medh5.open(path) as sample:
        return sample.document.provenance.activities_by_type("deidentify")[-1]


class TestScrubOptions:
    """10.1: the options are checked once, by the engine, for both doors."""

    @pytest.mark.parametrize(
        ("options", "flags", "message"),
        [
            ({"salt": "pep"}, ["--salt", "pep"], "--salt (salt) would do nothing"),
            (
                {"pseudonymise_ids": True, "withhold_shift": True},
                ["--pseudonymise-ids", "--withhold-shift"],
                "--pseudonymise-ids (pseudonymise_ids) and --withhold-shift",
            ),
            (
                {"apply_changes": True, "pseudonymise_ids": True},
                ["--apply", "--pseudonymise-ids"],
                "needs --salt (salt)",
            ),
            (
                {"apply_changes": True, "date_shift_days": 0},
                ["--apply", "--date-shift-days", "0"],
                "shifts nothing",
            ),
            (
                {"apply_changes": True, "date_shift_range": (-5, -10), "salt": "p"},
                ["--apply", "--date-shift-range", "-5:-10", "--salt", "p"],
                "holds no nonzero shift",
            ),
            (
                {"apply_changes": True, "date_shift_range": (-10, -5)},
                ["--apply", "--date-shift-range", "-10:-5"],
                "--date-shift-range (date_shift_range) needs --salt",
            ),
        ],
    )
    def test_S11_4_the_cli_and_the_api_refuse_alike(
        self, tmp_path: Path, capsys, options, flags, message
    ):
        """Each kept its own list --- Python four options, the command line
        three --- and they disagreed."""
        from medh5.cli import main

        path = Numbered.plain(tmp_path / "a.medh5")
        with pytest.raises(MEDH5Error) as refused:
            scrubber.scrub_paths([path], **options)
        said = str(refused.value)
        assert message in said
        assert main(["scrub", str(path), *flags]) == 2
        assert f"medh5: {said}" in capsys.readouterr().err
        with medh5.open(path) as sample:
            assert sample.document.deidentification is None

    def test_S11_4_out_names_one_file(self, tmp_path: Path, capsys):
        from medh5.cli import main

        paths = [str(Numbered.plain(tmp_path / f"{n}.medh5")) for n in "ab"]
        out = str(tmp_path / "x.medh5")
        assert main(["scrub", *paths, "--apply", "--out", out]) == 2
        assert "use --out-dir (out_dir)" in capsys.readouterr().err
        assert not Path(out).exists()


class TestScrubSalt:
    """10.2 (D26): a name is never hashed without a salt."""

    def test_S11_4_strict_refuses_to_hash_a_name_without_a_salt(self, tmp_path: Path):
        """Strict wrote "John Smith" as `sha256("John Smith")[:32]`: hashing
        candidate names reversed it."""
        path = tmp_path / "staff.medh5"
        with Numbered.writer(path) as w:
            w.person("John Smith")
        with medh5.open(path) as sample:
            before = sample.content_id
        with pytest.raises(MEDH5ValidationError, match="needs --salt") as refused:
            scrubber.apply(path, profile="strict")
        assert "provenance.agents[" in str(refused.value)
        with medh5.open(path) as sample:
            assert sample.content_id == before
        # Under basic a person's name is reported and kept, so nothing is hashed.
        assert scrubber.apply(path, withhold_shift=True).applied
        # The person --by names is pseudonymised under strict too.
        plain = Numbered.plain(tmp_path / "plain.medh5")
        with pytest.raises(MEDH5ValidationError, match="the person --by names"):
            scrubber.apply(plain, profile="strict", performed_by="RAD-07")

    def test_S11_4_salted_names_are_in_the_uid_map(self, tmp_path: Path):
        """A software agent named like a person became an unsalted pseudonym,
        whatever the salt."""
        path = tmp_path / "named.medh5"
        with Numbered.writer(path) as w:
            w.software("Doe^Jane")
            w.person("John Smith")
        with pytest.raises(MEDH5ValidationError, match="needs --salt"):
            scrubber.apply(path)
        report = scrubber.apply(path, profile="strict", salt="pep")
        for name in ("Doe^Jane", "John Smith"):
            assert report.uid_map[name] == scrubber.pseudonymise(name, "pep")
        with medh5.open(path) as sample:
            names = {a.name for a in sample.document.provenance.agents}
        assert scrubber.pseudonymise("Doe^Jane", "pep") in names
        assert scrubber.pseudonymise("Doe^Jane") not in names

    def test_S11_4_an_unsalted_uid_pseudonym_is_reported(self, tmp_path: Path):
        def with_uid(path: Path) -> Path:
            with Numbered.writer(path) as w:
                w.add_timepoint("tp0", study_uid="1.2.840.113619.2.55.3")
            return path

        report = scrubber.apply(with_uid(tmp_path / "uid.medh5"))
        assert report.ok
        (found,) = [f for f in report.remaining if f.rule == "unsalted_pseudonym"]
        assert not found.actionable and "3.0 requires one" in found.detail
        assert found in report.findings
        report = scrubber.apply(with_uid(tmp_path / "salted.medh5"), salt="pep")
        assert not [f for f in report.remaining if f.rule == "unsalted_pseudonym"]


class TestToolTimestamps:
    """10.3: when a tool ran can date the acquisition."""

    def _timed(self, path: Path) -> Path:
        with medh5.create(path, sample_id="s1", subject_id="subj-1") as w:
            w.add_timepoint("tp0", date="2020-01-31")
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
            tool = w.software("conv")
            w.activity(
                "import",
                agent=tool,
                started="2020-01-31T10:22:00Z",
                ended="2020-01-31T10:24:00Z",
            )
        return path

    def test_S11_4_strict_truncates_tool_times(self, tmp_path: Path):
        path = self._timed(tmp_path / "timed.medh5")
        found = {
            f.where: f.actionable
            for f in scrubber.scan(path, profile="strict").findings
            if f.rule == "tool_timestamp"
        }
        assert found == {
            "root.created": True,
            "provenance.activities[act_import_1].started": True,
            "provenance.activities[act_import_1].ended": True,
        }
        report = scrubber.apply(path, profile="strict", salt="pep", date_shift_days=-9)
        assert report.ok, report.format()
        assert "root.created cut to its year" in report.actions
        with h5py.File(path, "r") as handle:
            created = handle.attrs["created"]
        created = created.decode() if isinstance(created, bytes) else str(created)
        assert re.fullmatch(r"\d{4}-01-01T00:00:00Z", created)
        with medh5.open(path) as sample:
            (imported,) = sample.document.provenance.activities_by_type("import")
            assert sample.document.deidentification.date.endswith("-01-01T00:00:00Z")
        assert (imported.started, imported.ended) == ("2020-01-01T00:00:00Z",) * 2
        assert validate_file(path).ok
        assert not scrubber.scan(path, profile="strict").findings

    def test_S11_4_basic_reports_tool_times_after_a_shift(self, tmp_path: Path):
        path = self._timed(tmp_path / "timed.medh5")
        report = scrubber.apply(path, date_shift_days=-9, withhold_shift=True)
        assert report.ok
        timed = [f for f in report.remaining if f.rule == "tool_timestamp"]
        assert {f.where for f in timed} == {
            "provenance.activities[act_import_1].started",
            "provenance.activities[act_import_1].ended",
        }
        assert not any(f.actionable for f in timed)
        assert "unshifted beside shifted dates" in timed[0].detail
        with medh5.open(path) as sample:
            (imported,) = sample.document.provenance.activities_by_type("import")
        assert imported.started == "2020-01-31T10:22:00Z"

    def test_S11_4_basic_leaves_tool_times_alone_without_a_shift(self, tmp_path: Path):
        path = self._timed(tmp_path / "timed.medh5")
        assert not [
            f for f in scrubber.scan(path).findings if f.rule == "tool_timestamp"
        ]
        report = scrubber.apply(path)
        assert not [f for f in report.remaining if f.rule == "tool_timestamp"]


class TestDateSpellings:
    """10.4: a date or a time in a form no shift moves is reported."""

    def _with(self, path: Path, **extra: Any) -> Path:
        with Numbered.writer(path) as w:
            w.extra("src", extra)
        return path

    def test_S11_4_a_locale_date_is_reported(self, tmp_path: Path):
        path = self._with(
            tmp_path / "locale.medh5",
            seen="31.01.2020",
            us="01/31/2020",
            either="03/04/2020",
            nonsense="13/13/2020",
            dotted_month_first="01.31.2020",
        )
        found = {
            f.where: f.actionable
            for f in scrubber.scan(path).findings
            if f.rule == "locale_date"
        }
        assert found == {
            "extra.src.seen": False,
            "extra.src.us": False,
            "extra.src.either": False,
        }
        report = scrubber.apply(path)
        with medh5.open(path) as sample:
            assert sample.document.extra["src"]["seen"] == "31.01.2020"
        assert {f.where for f in report.remaining if f.rule == "locale_date"} == set(
            found
        )

    def test_S11_4_a_dicom_time_is_reported(self, tmp_path: Path):
        path = self._with(
            tmp_path / "time.medh5",
            AcquisitionTime="102200.123",
            study_time="102200",
            ContentTime="256000",
        )
        found = {
            f.where for f in scrubber.scan(path).findings if f.rule == "dicom_time"
        }
        assert found == {"extra.src.AcquisitionTime", "extra.src.study_time"}

    def test_S11_4_a_six_digit_id_is_not_a_time(self, tmp_path: Path):
        path = self._with(tmp_path / "id.medh5", SeriesNumber="102200", ref="102200")
        assert scrubber.scan(path).clean


class TestWithheldShift:
    """10.5 (D14): the amount of a shift restores every date it moved."""

    def test_S11_4_a_withheld_shift_leaves_no_amount_in_the_file(self, tmp_path: Path):
        path = _dated(tmp_path / "w.medh5")
        report = scrubber.apply(
            path, date_shift_days=-117, salt="pep", withhold_shift=True
        )
        assert report.ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date == "2019-10-06"
            record = sample.document.deidentification
            assert record.date_shift_days is None
            assert "date_shift_days" not in json.dumps(record.to_json())
        params = _deidentify(path).params
        assert (params["dates"], params["shift"]) == ("shifted", "withheld")
        assert re.fullmatch(r"[0-9a-f]{64}", params["shift_commitment"])
        assert b"-117" not in path.read_bytes()

    def test_S11_4_a_withheld_shift_still_passes_the_date_scan(self, tmp_path: Path):
        path = _dated(tmp_path / "w.medh5")
        scrubber.apply(path, date_shift_days=-117, salt="pep", withhold_shift=True)
        assert scrubber.scan(path).clean
        with medh5.amend(path) as w:
            w.add_timepoint("tp1", date="2020-04-30", days_from_baseline=90)
        (late,) = [f for f in scrubber.scan(path).findings if f.rule == "date"]
        assert late.where == "timepoints[1].date" and not late.actionable
        assert "withholds" in late.detail

    def test_S11_4_the_same_salt_and_shift_reapply_another_is_refused(
        self, tmp_path: Path
    ):
        path = _dated(tmp_path / "w.medh5")
        options = {"date_shift_days": -117, "salt": "pep", "withhold_shift": True}
        assert scrubber.apply(path, **options).ok
        assert scrubber.apply(path, **options).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date == "2019-10-06"
            before = sample.content_id
        for other in ({"date_shift_days": -30}, {"salt": "other"}):
            with pytest.raises(MEDH5ValidationError, match="withholds a date shift"):
                scrubber.apply(path, **{**options, **other})
        with medh5.open(path) as sample:
            assert sample.content_id == before
        # The amount it withholds can be recorded only by the run that knows it.
        assert scrubber.apply(path, **{**options, "withhold_shift": False}).ok
        with medh5.open(path) as sample:
            assert sample.document.deidentification.date_shift_days == -117

    def test_S11_4_one_subjects_files_share_a_shift_from_a_range(self, tmp_path: Path):
        first = _dated(tmp_path / "a.medh5", sample_id="a")
        second = _dated(tmp_path / "b.medh5", sample_id="b")
        scrubber.apply(first, date_shift_range=(-400, -100), salt="pep")
        # The other file's ids pseudonymised: its subject is still the same.
        scrubber.apply(
            second, date_shift_range=(-400, -100), salt="pep", pseudonymise_ids=True
        )
        dates = []
        for path in (first, second):
            with medh5.open(path) as sample:
                dates.append(sample.document.timepoints["tp0"].date)
                assert sample.document.deidentification.date_shift_days is None
            assert _deidentify(path).params["shift"] == "withheld"
        assert dates[0] == dates[1] != "2020-01-31"
        import datetime

        moved = datetime.date.fromisoformat(dates[0]) - datetime.date(2020, 1, 31)
        assert -400 <= moved.days <= -100
        # A re-run with the range finds the same shift, and is accepted.
        assert scrubber.apply(first, date_shift_range=(-400, -100), salt="pep").ok

    def test_S11_4_two_subjects_get_different_shifts(self, tmp_path: Path):
        dates = set()
        for subject in ("subj-1", "subj-2", "subj-3"):
            path = _dated(tmp_path / f"{subject}.medh5", subject_id=subject)
            scrubber.apply(path, date_shift_range=(-1000, 1000), salt="pep")
            with medh5.open(path) as sample:
                dates.add(sample.document.timepoints["tp0"].date)
        assert len(dates) == 3 and "2020-01-31" not in dates

    def test_S11_4_a_recorded_shift_is_reported(self, tmp_path: Path):
        path = _dated(tmp_path / "w.medh5")
        with pytest.warns(FutureWarning, match="^medh5 3.0 withholds the date shift"):
            scrubber.apply(path, date_shift_days=-117)
        (basic,) = [
            f for f in scrubber.scan(path).findings if f.rule == "date_shift_recorded"
        ]
        assert (basic.value, basic.actionable) == ("-117", False)
        assert "date_shift_recorded" in scrubber.STRICT_RULES
        strict = scrubber.scan(path, profile="strict")
        assert [f.actionable for f in strict.findings if f.rule == basic.rule] == [True]
        # A strict scrub withholds it, and leaves the dates where they are.
        assert scrubber.apply(path, profile="strict", salt="pep").ok
        with medh5.open(path) as sample:
            assert sample.document.deidentification.date_shift_days is None
            assert sample.document.timepoints["tp0"].date == "2019-10-06"
        assert _deidentify(path).params["shift"] == "withheld"
        assert not scrubber.scan(path, profile="strict").findings

    def test_S11_4_out_leaves_the_source_byte_identical(self, tmp_path: Path):
        source = _dated(tmp_path / "src.medh5", sample_id="MRN-000042")
        before = source.read_bytes()
        out = tmp_path / "out.medh5"
        report = scrubber.apply(
            source, date_shift_days=-5, salt="pep", withhold_shift=True, out=out
        )
        assert report.ok and report.path == str(out)
        assert source.read_bytes() == before
        with medh5.open(out) as sample:
            assert sample.document.timepoints["tp0"].date == "2020-01-26"
        with pytest.raises(MEDH5ValidationError, match="does not overwrite one"):
            scrubber.apply(source, withhold_shift=True, out=out)
        # A directory: named by the pseudonymised id, the source untouched.
        target = tmp_path / "shared"
        target.mkdir()
        [report] = scrubber.scrub_paths(
            [source],
            apply_changes=True,
            salt="pep",
            pseudonymise_ids=True,
            withhold_shift=True,
            out_dir=target,
        )
        pseudonym = scrubber.pseudonymise("MRN-000042", "pep")
        assert report.path == str(target / f"{pseudonym[len('pseudo:') :]}.medh5")
        assert source.read_bytes() == before
        assert [p.name for p in target.iterdir()] == [Path(report.path).name]

    def test_S11_4_a_burned_in_check_survives_apply(self, tmp_path: Path):
        path = tmp_path / "checked.medh5"
        with Numbered.writer(path) as w:
            w.deidentification(method="manual", burned_in_annotation_checked=True)
        scrubber.apply(path)
        with medh5.open(path) as sample:
            record = sample.document.deidentification
        assert record.method == "medh5-scrub"
        assert record.burned_in_annotation_checked is True
        assert record.date is not None
        assert _deidentify(path).params["prior_method"] == "manual"

    def test_the_cli_withholds_and_takes_a_range(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = _dated(tmp_path / "w.medh5")
        argv = ["scrub", str(path), "--apply", "--salt", "pep"]
        assert main([*argv, "--date-shift-range", "-20:-10"]) == 0
        assert _deidentify(path).params["shift"] == "withheld"
        recorded = _dated(tmp_path / "r.medh5")
        assert main(["scrub", str(recorded), "--apply", "--date-shift-days", "-3"]) == 0
        assert "medh5 3.0 withholds it by default" in capsys.readouterr().err
        quiet = _dated(tmp_path / "q.medh5")
        assert (
            main(["scrub", str(quiet), "--apply", "--date-shift-days=-3"])
            + main(["scrub", str(quiet), "--apply", "--withhold-shift"])
            == 0
        )


class TestScrubCollection:
    """W22-07.1 (E8): a collection is scrubbed member by member, into a new
    collection, and its source is only read."""

    IDS = (("MRN-0001", "subj-1"), ("MRN-0002", "subj-1"), ("MRN-0003", "subj-2"))

    @staticmethod
    def _collection(tmp_path: Path, *extra: Path) -> Path:
        from medh5.collection import pack

        paths = [
            _dated(tmp_path / f"{sample}.medh5", sample_id=sample, subject_id=subject)
            for sample, subject in TestScrubCollection.IDS
        ]
        return pack([*paths, *extra], tmp_path / "cohort.medh5c")

    @staticmethod
    def _members(path: Path) -> dict[str, Any]:
        """Each member's key, ids, first date and stored `content_id`."""
        from medh5.collection import open_collection

        with open_collection(path) as collection:
            return {
                key: (
                    collection[key].document.identity.sample_id,
                    collection[key].document.identity.subject_id,
                    collection[key].document.timepoints["tp0"].date,
                    collection[key].content_id,
                )
                for key in collection
            }

    @staticmethod
    def _scratch(tmp_path: Path) -> list[str]:
        return [p.name for p in tmp_path.iterdir() if ".scrub-" in p.name]

    def test_S2_2_a_scrubbed_collection_is_a_new_collection(self, tmp_path: Path):
        source = self._collection(tmp_path)
        before = self._members(source)
        out = tmp_path / "shared.medh5c"
        report = scrubber.apply(
            source,
            out=out,
            salt="pep",
            pseudonymise_ids=True,
            date_shift_range=(-400, -100),
        )
        assert report.applied and report.ok and report.path == str(out)
        assert validate_file(out).ok
        after = self._members(out)
        # Keyed by each member's pseudonym, as `--out-dir` names a file.
        pseudonyms = {
            scrubber.pseudonymise(sample, "pep"): sample for sample, _ in self.IDS
        }
        assert sorted(after) == sorted(p.removeprefix("pseudo:") for p in pseudonyms)
        for key, (sample_id, subject_id, date, content_id) in after.items():
            old = pseudonyms[sample_id]
            assert key == sample_id.removeprefix("pseudo:")
            assert subject_id == scrubber.pseudonymise(dict(self.IDS)[old], "pep")
            assert date != "2020-01-31" and content_id != before[old][3]
        # One report: each member's findings under its source key, what a
        # re-scan still finds under the key it was written as.
        assert {f.where.split("::")[0] for f in report.findings} == {
            sample for sample, _ in self.IDS
        }
        assert (
            f"MRN-0001 written as {scrubber.pseudonymise('MRN-0001', 'pep')[7:]}"
            in (report.actions)
        )
        assert set(report.uid_map) >= {"MRN-0001", "MRN-0002", "MRN-0003", "subj-1"}
        assert b"MRN-000" not in out.read_bytes()
        assert not self._scratch(tmp_path)

    def test_S11_4_members_of_one_subject_share_a_shift(self, tmp_path: Path):
        source = self._collection(tmp_path)
        out = tmp_path / "shared.medh5c"
        scrubber.apply(source, out=out, salt="pep", date_shift_range=(-400, -100))
        dates = {sample: date for sample, _, date, _ in self._members(out).values()}
        assert dates["MRN-0001"] == dates["MRN-0002"] != dates["MRN-0003"]
        # The members agree with the samples scrubbed one by one.
        alone = _dated(tmp_path / "alone.medh5", subject_id="subj-2")
        scrubber.apply(alone, salt="pep", date_shift_range=(-400, -100))
        with medh5.open(alone) as sample:
            assert sample.document.timepoints["tp0"].date == dates["MRN-0003"]

    def test_S2_2_a_refused_member_leaves_no_output(self, tmp_path: Path):
        clinical = TestClinicalScrub._clinical(tmp_path)
        source = self._collection(tmp_path, clinical)
        out = tmp_path / "shared.medh5c"
        with pytest.raises(MEDH5ValidationError, match="nothing was written") as no:
            scrubber.apply(source, out=out, salt="pep", withhold_shift=True)
        said = str(no.value)
        assert "member 'visit' of" in said and "--skip-refused" in said
        assert f"{source}::visit" in said and ".scrub-" not in said
        assert not out.exists() and not self._scratch(tmp_path)
        # Left out and listed, the others written.
        report = scrubber.apply(
            source, out=out, salt="pep", withhold_shift=True, skip_refused=True
        )
        assert sorted(self._members(out)) == ["MRN-0001", "MRN-0002", "MRN-0003"]
        (refused,) = [f for f in report.findings if f.rule == "refused_member"]
        assert refused.where == "visit" and "a name, a UID" in refused.detail
        assert "visit left out: refused" in report.actions
        assert not self._scratch(tmp_path)

    def test_S2_2_the_source_collection_is_byte_identical(self, tmp_path: Path):
        source = self._collection(tmp_path)
        before = source.read_bytes()
        scanned = scrubber.scan(source)
        assert not scanned.applied
        assert [f.where for f in scanned.findings] == [
            f"{sample}::timepoints[0].date" for sample, _ in self.IDS
        ]
        report = scrubber.apply(
            source, out=tmp_path / "shared.medh5c", withhold_shift=True, salt="pep"
        )
        assert report.ok
        assert source.read_bytes() == before

    def test_S2_2_a_collection_is_never_rewritten_in_place(self, tmp_path: Path):
        source = self._collection(tmp_path)
        with pytest.raises(MEDH5ValidationError, match="give --out"):
            scrubber.apply(source)
        with pytest.raises(MEDH5ValidationError, match="the collection being scrubbed"):
            scrubber.apply(source, out=source)
        # In a directory, under its own name.
        target = tmp_path / "shared"
        target.mkdir()
        [report] = scrubber.scrub_paths(
            [source], apply_changes=True, withhold_shift=True, out_dir=target
        )
        assert report.path == str(target / source.name)

    def test_S11_4_a_member_keyed_by_a_record_number_is_reported(self, tmp_path: Path):
        """A member's key is a lone sample's file name: the identity rule
        reads it as it reads a name, and a pseudonymised run re-keys it."""
        from medh5.collection import pack

        path = tmp_path / "1.2.840.10008.5.medh5"
        _dated(path, sample_id="s1")
        source = pack([path], tmp_path / "c.medh5c", keys=["1.2.840.10008.5"])
        (keyed,) = [f for f in scrubber.scan(source).findings if f.rule == "file_name"]
        assert keyed.where == "1.2.840.10008.5::member key"
        assert keyed.value == "1.2.840.10008.5" and "member key" in keyed.detail
        assert "rename" not in keyed.detail
        report = scrubber.apply(
            source,
            out=tmp_path / "shared.medh5c",
            salt="pep",
            pseudonymise_ids=True,
            withhold_shift=True,
        )
        assert not [f for f in report.remaining if f.rule == "file_name"]

    def test_S11_4_each_members_reminted_ids_are_kept(self, tmp_path: Path):
        """Two subjects' histories on clocks of one id re-mint apart, so the
        key to the pseudonyms keeps each under its member; a pseudonym of an
        id, the salt's alone, stays one entry."""
        from medh5.clinical import HOUR, Event
        from medh5.collection import pack
        from tests.kits import History

        visit = Event(
            "visit", "visit", "observation", "point", "final", 0, available_us=HOUR
        )
        paths = []
        for subject in ("P-01", "P-02"):
            path = tmp_path / f"{subject}.medh5"
            History.write(path, subject_id=subject, events=[visit], lesion=False)
            paths.append(path)
        source = pack(paths, tmp_path / "c.medh5c")
        report = scrubber.apply(
            source,
            out=tmp_path / "shared.medh5c",
            salt="pep",
            pseudonymise_ids=True,
            withhold_shift=True,
            attest_text="reviewed",
        )
        clocks = {k: v for k, v in report.uid_map.items() if "clock:" in k}
        assert sorted(k.split("::")[0] for k in clocks) == ["P-01", "P-02"]
        assert len(set(clocks.values())) == 2
        assert "event:visit" not in report.uid_map
        assert (
            report.uid_map["P-01::event:visit"] != report.uid_map["P-02::event:visit"]
        )
        assert report.uid_map["P-01"] == scrubber.pseudonymise("P-01", "pep")

    def test_S2_2_a_key_as_long_as_a_key_may_be(self, tmp_path: Path):
        """A key may be 255 bytes, longer than a file name with its suffix
        can be: the member is scrubbed under it all the same."""
        from medh5.collection import open_collection, pack

        key = "k" * 255
        source = pack([_dated(tmp_path / "s.medh5")], tmp_path / "c.medh5c", keys=[key])
        out = tmp_path / "shared.medh5c"
        report = scrubber.apply(source, out=out, salt="pep", withhold_shift=True)
        assert report.ok
        with open_collection(out) as collection:
            assert list(collection) == [key]
        assert {f.where.split("::")[0] for f in report.findings} == {key}

    def test_the_cli_scrubs_a_collection(self, tmp_path: Path, capsys):
        from medh5.cli import main

        source = self._collection(tmp_path)
        out = tmp_path / "shared.medh5c"
        assert main(["scrub", str(source), "--apply", "--withhold-shift"]) == 1
        assert "give --out" in capsys.readouterr().err
        flags = ["--apply", "--salt", "pep", "--withhold-shift", "--out", str(out)]
        assert main(["scrub", str(source), *flags, "--json"]) == 0
        [report] = json.loads(capsys.readouterr().out)
        assert report["path"] == str(out) and report["ok"]
        assert main(["scrub", str(source), "--skip-refused"]) == 2
        assert "--skip-refused (skip_refused)" in capsys.readouterr().err


class TestRedactText:
    """W22-07.2 (E7): a redacted clinical document is a new document, owned
    by a new version of its event, and every span into it moves with it."""

    NOTE = "Seen with Smith^Alice today; review on 2020-02-14."

    @staticmethod
    def _sample(
        tmp_path: Path,
        *,
        notes: Sequence[tuple[str, str]] = (("note", NOTE),),
        spans: Sequence[tuple[str, str]] = (("note", "today"),),
        name: str = "visit.medh5",
    ) -> Path:
        """One document event per note, each owning its text, and a link
        citing each *spans* entry's words in its note."""
        from medh5.clinical import HOUR, Document, Event, Link
        from tests.kits import History

        texts = dict(notes)
        events = [
            Event(
                key,
                key,
                "document",
                "point",
                "final",
                effective_start_us=0,
                available_us=HOUR,
            )
            for key in texts
        ]
        links = [
            Link.between(("event", key), "describes", ("document", f"{key}_text"))
            for key in texts
        ]
        for key, words in spans:
            text = texts[key].encode()
            start = text.index(words.encode())
            links.append(
                Link.between(
                    ("document", f"{key}_text"),
                    "describes",
                    ("image", "CT_tp0"),
                    span=(start, start + len(words.encode())),
                )
            )
        path = tmp_path / name
        History.write(
            path,
            events=events,
            documents=[Document(f"{key}_text", text) for key, text in notes],
            links=links,
        )
        return path

    @staticmethod
    def _cited(records: Any) -> list[str]:
        """What every span in *records* cites, in order."""
        texts = {d.document_id: d.text.encode() for d in records.documents}
        return [
            texts[link.source_id][link.source_span[0] : link.source_span[1]].decode()
            for link in records.links
            if link.source_span is not None
        ]

    def test_S6_redaction_writes_a_new_document_and_version(self, tmp_path: Path):
        path = self._sample(tmp_path)
        before = path.read_bytes()
        with medh5.open(path) as sample:
            held = sample.clinical.records()
        out = tmp_path / "shared.medh5"
        report = scrubber.apply(
            path, salt="pep", redact_text=True, attest_text="reviewed", out=out
        )
        assert report.ok and path.read_bytes() == before
        with medh5.open(out) as sample:
            records = sample.clinical.records()
            activities = sample.document.provenance.activities_by_type("deidentify")
        note = report.uid_map["document:note_text"]
        (document,) = [d for d in records.documents if d.document_id == note]
        assert document.text == "Seen with [NAME] today; review on [DATE]."
        # Owned by a new version of its event: a new id, the same record and
        # the source's availability.
        (event,) = [
            e for e in records.events if e.event_id == report.uid_map["event:note"]
        ]
        (was,) = [e for e in held.events if e.event_id == "note"]
        assert event.record_id == report.uid_map["record:note"]
        assert event.available_us == was.available_us
        assert any(
            (link.source_id, link.relation, link.target_id)
            == (event.event_id, "describes", note)
            for link in records.links
        )
        # A document without a hit is the same bytes under its new id.
        texts = sorted(d.text for d in records.documents if d.document_id != note)
        assert texts == sorted(
            d.text for d in held.documents if d.document_id != "note_text"
        )
        # The redaction is recorded, apart from the attestation, with the
        # digests of the bytes it wrote.
        digest = "sha256:" + hashlib.sha256(document.text.encode()).hexdigest()
        (redaction,) = [
            a for a in activities if a.params.get("scope") == "clinical_text_redaction"
        ]
        assert redaction.params["tool"] == "medh5 scrub"
        assert redaction.params["placeholders"]["name"] == "[NAME]"
        assert redaction.params["digests"] == {f"clinical/documents/{note}": digest}
        (attested,) = [
            a for a in activities if a.params.get("scope") == "clinical_text"
        ]
        assert attested.params["digests"][f"clinical/documents/{note}"] == digest
        assert any("redacted" in action for action in report.actions)

    def test_S6_spans_are_remapped_through_the_redaction(self, tmp_path: Path):
        path = self._sample(tmp_path)
        with medh5.open(path) as sample:
            assert self._cited(sample.clinical.records()) == ["today"]
        report = scrubber.apply(
            path, salt="pep", redact_text=True, attest_text="reviewed"
        )
        assert report.ok
        with medh5.open(path) as sample:
            assert self._cited(sample.clinical.records()) == ["today"]

    def test_S6_a_span_inside_a_redacted_range_is_refused(self, tmp_path: Path):
        path = self._sample(tmp_path, spans=[("note", "Smith^Alice today")])
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="overlap the range") as no:
            scrubber.apply(path, salt="pep", redact_text=True, attest_text="reviewed")
        said = str(no.value)
        assert "document:note_text -describes-> image:CT_tp0" in said
        assert "[10, 21)" in said and "Smith" not in said
        assert path.read_bytes() == before

    def test_S6_a_redacted_sample_validates_clean(self, tmp_path: Path):
        path = self._sample(tmp_path)
        report = scrubber.apply(
            path,
            salt="pep",
            redact_text=True,
            attest_text="reviewed",
            pseudonymise_ids=True,
        )
        assert report.ok
        result = validate_file(path)
        assert result.ok, result
        rescan = scrubber.scan(path)
        assert not [f for f in rescan.findings if f.rule.startswith("clinical")]
        # Re-applied, nothing is left to redact: the records are the same.
        with medh5.open(path) as sample:
            held = sample.clinical.records()
        again = scrubber.apply(
            path, salt="pep", redact_text=True, pseudonymise_ids=True
        )
        assert again.ok
        with medh5.open(path) as sample:
            assert sample.clinical.records() == held

    def test_S6_redaction_is_read_one_document_at_a_time(self, tmp_path: Path):
        """Each document is redacted from its own bytes, and its spans by its
        own edits; one whose bytes do not read refuses the run by name."""
        notes = [("note", self.NOTE), ("letter", "Dear Jones^Bob, all is well.")]
        spans = [("note", "today"), ("letter", "all is well")]
        path = self._sample(tmp_path, notes=notes, spans=spans)
        out = tmp_path / "shared.medh5"
        report = scrubber.apply(
            path, salt="pep", redact_text=True, attest_text="reviewed", out=out
        )
        with medh5.open(out) as sample:
            records = sample.clinical.records()
        letter = report.uid_map["document:letter_text"]
        (redacted,) = [d.text for d in records.documents if d.document_id == letter]
        assert redacted == "Dear [NAME], all is well."
        assert sorted(self._cited(records)) == ["all is well", "today"]
        TestClinicalScrub._plant(path, "letter_text", b"w", b"\xff")
        with pytest.raises(MEDH5ValidationError, match="does not read") as no:
            scrubber.apply(path, salt="pep", redact_text=True, attest_text="reviewed")
        assert "letter_text" in str(no.value) and "note_text" not in str(no.value)

    def test_S6_redacted_text_still_needs_an_attestation(self, tmp_path: Path):
        path = self._sample(tmp_path)
        with pytest.raises(MEDH5ValidationError, match="--attest-text TOOL"):
            scrubber.apply(path, salt="pep", redact_text=True)
        with pytest.raises(MEDH5ValidationError, match="--salt"):
            scrubber.apply(path, redact_text=True, attest_text="reviewed")
        with pytest.raises(MEDH5ValidationError, match="--redact-text"):
            scrubber.apply(path, salt="pep", attest_text="reviewed")
        with pytest.raises(
            MEDH5ValidationError, match=r"--redact-text \(redact_text\)"
        ):
            scrubber.scrub_paths([path], redact_text=True)

    def test_the_cli_flag(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._sample(tmp_path)
        flags = ["--apply", "--salt", "pep", "--attest-text", "reviewed"]
        assert main(["scrub", str(path), *flags, "--redact-text", "--json"]) == 0
        [report] = json.loads(capsys.readouterr().out)
        assert report["ok"]
        with medh5.open(path) as sample:
            texts = [d.text for d in sample.clinical.records().documents]
        assert any("[NAME]" in text for text in texts)
