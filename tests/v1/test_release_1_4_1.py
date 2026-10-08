"""What 1.4.1 changed, held to by the reproductions that found it.

Every test here is one of the third audit's findings, written as the shortest
program that shows it, and every one of them fails on 1.4.0.  The third audit
went after what the files are *used for* --- shipped to a collaborator after a
scrub, fed to a training loop, recompressed by a curator, amended by a tool one
version behind --- where the first two had gone after what the writer was asked
for.  The gates were green throughout, because they test what the code was built
to do.

Test names cite the finding as well as the clause, because the audit page is
where the reasoning lives and the id is what joins the two.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5FileError, MEDH5ValidationError, MEDH5VersionError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file

SHAPE = (8, 12, 12)


def _label_set(n: int = 3) -> LabelSet:
    return LabelSet(
        "rel-1.4.1",
        version="1.0.0",
        classes=[LabelClass(i, f"c{i}", f"C{i}") for i in range(1, n + 1)],
    )


def _writer(path: Path, *, shape: tuple[int, ...] = SHAPE, **options: Any) -> Any:
    options.setdefault("codec", "portable")
    w = medh5.create(path, sample_id="s1", subject_id="subj-1", **options)
    w.add_grid("g", shape=shape, spacing=(2.0, 1.0, 1.0))
    w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
    w.label_set(_label_set())
    return w


def _plain(path: Path, **options: Any) -> Path:
    with _writer(path, **options):
        pass
    return path


# --------------------------------------------------------------------------
# W14 --- one gate for every rewrite
# --------------------------------------------------------------------------


class TestW14RewriteGate:
    """F-20, F-22, L-27, L-31, Q-15: what every door into a file checks."""

    def test_F20_S16_amend_refuses_a_future_major(self, tmp_path: Path):
        """`open` refused a 2.0 file; `amend` restamped it 1.0 and carried on."""
        path = _plain(tmp_path / "v2.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        before = path.read_bytes()
        with pytest.raises(MEDH5VersionError):
            medh5.amend(path)
        assert path.read_bytes() == before

    def test_F20_S16_amend_keeps_a_later_minor(self, tmp_path: Path):
        """A 1.1 file's objects are carried through, so it stays a 1.1 file."""
        path = _plain(tmp_path / "v11.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "1.1"
            handle.create_group("x_future")
        with medh5.amend(path):
            pass
        with h5py.File(path, "r") as handle:
            assert handle.attrs["medh5_version"] == "1.1"
            assert "x_future" in handle

    def test_F20_the_scrub_and_fix_doors_are_the_same_gate(self, tmp_path: Path):
        from medh5.curation import scrub

        path = _plain(tmp_path / "v2s.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            scrub.apply(path)

    @pytest.fixture
    def secret(self, tmp_path: Path) -> Path:
        path = tmp_path / "id_rsa"
        path.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----hunter2" + b"A" * 4000)
        return path

    def _external_storage(self, tmp_path: Path, secret: Path) -> Path:
        path = _plain(tmp_path / "ext.medh5")
        size = secret.stat().st_size
        with h5py.File(path, "r+") as handle:
            handle.create_dataset(
                "x_vendor_notes",
                shape=(size,),
                dtype="u1",
                external=[(secret, 0, size)],
            )
        return path

    def test_F22_external_storage_is_refused_by_every_tool(
        self, tmp_path: Path, secret: Path
    ):
        """A crafted file made `recompress` copy a local file into its output."""
        from medh5.storage import recompress

        path = self._external_storage(tmp_path, secret)
        out = tmp_path / "shared.medh5"
        for door in (
            lambda: medh5.open(path),
            lambda: medh5.amend(path),
            lambda: recompress(path, "portable", out=out),
        ):
            with pytest.raises(MEDH5FileError, match="not self-contained"):
                door()
        assert not out.exists()
        assert validate_file(path).codes == ("E001",)

    def test_F22_an_external_link_is_refused(self, tmp_path: Path):
        other = _plain(tmp_path / "other.medh5")
        path = _plain(tmp_path / "linked.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_link"] = h5py.ExternalLink(str(other), "/images")
        with pytest.raises(MEDH5FileError, match="external link"):
            medh5.open(path)

    def test_F22_a_virtual_dataset_is_refused(self, tmp_path: Path):
        source = tmp_path / "source.h5"
        with h5py.File(source, "w") as handle:
            handle["data"] = np.arange(16, dtype="u1")
        path = _plain(tmp_path / "virtual.medh5")
        layout = h5py.VirtualLayout(shape=(16,), dtype="u1")
        layout[:] = h5py.VirtualSource(str(source), "data", shape=(16,))
        with h5py.File(path, "r+") as handle:
            handle.create_virtual_dataset("x_view", layout)
        with pytest.raises(MEDH5FileError, match="virtual dataset"):
            medh5.open(path)

    @pytest.mark.skipif(os.name == "nt", reason="Windows cannot replace an open file")
    def test_F22_the_check_is_remembered_for_the_file_it_read(
        self, tmp_path: Path, secret: Path
    ):
        """A path replaced between the open and the check must not inherit it.

        The memo was keyed by a stat of the *path*, taken after the open.  A
        replacement landing in between was recorded as checked while the handle
        scanned the old file, and its next open skipped the check.  The race
        itself is the engine's to hold (`h5::ops::tests::
        f22_the_check_is_remembered_for_the_file_it_read`); through the public
        door, a file checked and remembered does not vouch for its replacement.
        """
        target = _plain(tmp_path / "target.medh5")
        bad = self._external_storage(tmp_path, secret)
        with medh5.open(target) as sample:  # checked, and remembered
            os.replace(bad, target)  # the path now names another file
            assert "CT" in sample.images  # the file read is clean
        with pytest.raises(MEDH5FileError, match="not self-contained"):
            medh5.open(target)

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
        reason="POSIX file modes, which root ignores",
    )
    def test_Q15_a_read_only_sample_can_still_be_amended(self, tmp_path: Path):
        """The temporary took the target's `0o444` and HDF5 could not write it."""
        path = _plain(tmp_path / "read-only.medh5")
        os.chmod(path, 0o444)
        writer = medh5.amend(path)
        try:
            temporary = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
            assert temporary
            assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in temporary)
        finally:
            writer.commit()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444

    def test_L31_unpack_refuses_a_member_name_that_is_a_path(self, tmp_path: Path):
        from medh5.collection import pack, unpack

        shard = tmp_path / "s.medh5c"
        pack([_plain(tmp_path / "a.medh5")], shard, keys=["good"])
        with h5py.File(shard, "r+") as handle:
            handle.copy("samples/good", handle["samples"], name="..\\..\\evil")
        with pytest.raises(MEDH5ValidationError, match="evil"):
            unpack(shard, tmp_path / "out")
        assert not list(tmp_path.rglob("evil*"))

    def test_L27_S14_amend_keeps_a_portable_file_portable(self, tmp_path: Path):
        """New datasets in an amended `portable` file were Blosc2."""
        from medh5.storage import describe_filters

        path = _plain(tmp_path / "port.medh5", codec="portable")
        big = np.random.default_rng(0).random((64, 64, 64)) > 0.5
        with medh5.amend(path) as w:
            w.add_grid("g2", shape=big.shape, spacing=(1.0, 1.0, 1.0))
            w.add_mask("new", big, grid="g2")
        with medh5.open(path) as sample:
            stored = sample.root["annotations/new/data"]
            assert describe_filters(stored).startswith("gzip")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_Q15_the_temporary_file_is_never_more_permissive(self, tmp_path: Path):
        path = _plain(tmp_path / "perm.medh5")
        os.chmod(path, 0o600)
        writer = medh5.amend(path)
        try:
            temporary = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
            assert temporary
            assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in temporary)
        finally:
            writer.commit()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


# --------------------------------------------------------------------------
# W15 --- de-identification that covers the file
# --------------------------------------------------------------------------


class TestW15Deidentification:
    """F-14, L-36: the scan and the clean see every string, and say so."""

    @pytest.fixture
    def dicom_import(self, tmp_path: Path) -> dict[str, Any]:
        pytest.importorskip("pydicom")
        from medh5.io.dicom import from_dicom
        from tests.v1.conftest import write_dicom_series

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
        with _writer(path) as w:
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
        with _writer(path) as w:
            w.extra("src", {"reader": name})
        assert [f.rule for f in scrub.scan(path).actionable] == ["person_name"]

    def test_L36_an_age_over_89_and_an_organization(self, tmp_path: Path):
        from medh5.curation import scrub

        path = tmp_path / "age.medh5"
        with _writer(path) as w:
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

        path = _plain(tmp_path / "attr.medh5")
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
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image(
                "CT",
                np.zeros(SHAPE, np.int16),
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
        with _writer(path) as w:
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
        with _writer(path) as w:
            w.add_timepoint("tp0", date="2023-01-05")
        assert scrub.apply(path).ok
        with medh5.open(path) as sample:
            assert sample.document.timepoints["tp0"].date is None

    def test_L36_mapping_keys_are_read_as_text(self, tmp_path: Path):
        """`{"Doe^Jane": ...}` names a person as surely as a value does."""
        from medh5.curation import scrub

        path = tmp_path / "keys.medh5"
        uid = "1.2.840.113619.2.55.3.1"
        with _writer(path) as w:
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

        path = _plain(tmp_path / "compound.medh5")
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
            w.add_grid(uid, shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid=uid, modality="CT")
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
        with _writer(path) as w:
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

        path = _plain(tmp_path / "filters.medh5")
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
        with _writer(path) as w:
            w.extra("src", {"PatientName": "Doe^Jane"})
        with medh5.open(path) as sample:
            report = scrub.scan_document(
                sample.document, scrub.ScrubReport(path=str(path))
            )
        assert [f.rule for f in report.findings] == ["identifier"]


# --------------------------------------------------------------------------
# W17 --- constants that must not change an answer
# --------------------------------------------------------------------------


class TestW17Precision:
    """F-15, F-21, P-10, Q-14."""

    @pytest.mark.parametrize("n", range(2, 11))
    def test_F15_S7_5_a_vote_fraction_survives_its_storage(self, tmp_path: Path, n):
        """float16 put k/n just under the threshold k/n for n = 3, 5, 6, 7, 9, 10."""
        for k in range(1, n):
            votes = np.zeros(SHAPE)
            for level in range(1, n + 1):
                votes[level % SHAPE[0]] = level / n
            threshold = k / n
            path = tmp_path / f"pm{n}_{k}.medh5"
            with _writer(path) as w:
                w.add_segmentation(
                    "soft", grid="g", probabilities={1: votes}, threshold=threshold
                )
            with medh5.open(path) as sample:
                annotation = sample.annotations["soft"]
                np.testing.assert_array_equal(
                    annotation.dense([1])[0], votes >= threshold, err_msg=f"{k}/{n}"
                )
                assert annotation.data.dtype in (np.float16, np.float32)

    def test_F15_float16_is_kept_where_it_is_exact(self, tmp_path: Path):
        path = tmp_path / "half.medh5"
        votes = np.zeros(SHAPE)
        votes[2:4] = 0.5
        with _writer(path) as w:
            w.add_segmentation(
                "soft", grid="g", probabilities={1: votes}, threshold=0.5
            )
        with medh5.open(path) as sample:
            assert sample.annotations["soft"].data.dtype == np.float16

    def test_F21_S7_7_the_in_band_ignore_check_has_no_size_cap(self, tmp_path: Path):
        """E411 fired on a correct 512×512×256 labelmap: the check declined to
        look past 64M elements and then reported what it had not found.

        The many-slab path, with the only ignore voxel in the last slab, is the
        engine's Rust test (`annotations::payload::tests::s7_7_scans_reach_the_
        last_slab`, budgets down to one byte); this holds the validator to it.
        """
        path = tmp_path / "ignore.medh5"
        liver = np.zeros(SHAPE, bool)
        liver[:2] = True
        region = np.zeros(SHAPE, bool)
        region[-1, -1, -1] = True
        with _writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver},
                ignore=region,
                encoding="labelmap",
            )
        report = validate_file(path)
        assert not [c for c in report.codes if c in ("E411", "W904")], report.codes

    def test_F21_a_ct_sized_labelmap_with_an_ignore_region_validates(
        self, tmp_path: Path
    ):
        """The reproduction at its real size: 256×512×512, just past 64M elements.

        The datasets are grown in place with HDF5 chunks that are never
        written, which read back as the fill value --- so the file costs
        neither the memory nor the disk of a real CT, and the validator still
        has 67M elements to scan.  1.4.0 reported E411 and W904 here.
        """
        big = (256, 512, 512)
        liver = np.zeros(SHAPE, bool)
        liver[:2] = True
        region = np.zeros(SHAPE, bool)
        region[-1] = True
        path = tmp_path / "ct.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: np.zeros(SHAPE, bool)},
                ignore=region,
                annotated_classes=[1],
                encoding="labelmap",
            )
        with h5py.File(path, "r+") as handle:
            handle["grids/g"].attrs["shape"] = np.asarray(big, dtype=np.int64)
            for name, dtype in (
                ("images/CT", np.int16),
                ("annotations/seg/data", np.uint16),
            ):
                attrs = dict(handle[name].attrs)
                del handle[name]
                grown = handle.create_dataset(
                    name,
                    shape=big,
                    dtype=dtype,
                    chunks=(16, 128, 128),
                    compression="gzip",
                    fillvalue=0,
                )
                for key, value in attrs.items():
                    grown.attrs[key] = value
            data = handle["annotations/seg/data"]
            data[:8, :64, :64] = 1
            data[-4:, :64, :64] = 65535
        codes = validate_file(path).codes
        assert "E411" not in codes and "W904" not in codes, codes
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].has_ignore_region

    def test_P10_the_overlap_graph_is_read_in_slabs(self, tmp_path: Path):
        """The slab-bounded read is the engine's (`overlap_edges_within`, held
        at a 64-byte and a 1-byte budget by its Rust test); what W908 judges
        from that graph is checked here: one overlap needs two layers, so two
        layers are not too many."""
        path = tmp_path / "layers.medh5"
        liver = np.zeros(SHAPE, bool)
        liver[1:6] = True
        lesion = np.zeros(SHAPE, bool)
        lesion[5, 3:5, 3:5] = True
        spleen = np.zeros(SHAPE, bool)
        spleen[7] = True
        with _writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: lesion, 3: spleen},
                encoding="layers",
            )
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].kind == "layers"
            assert sample.root["annotations/seg/data"].shape[0] == 2
        assert "W908" not in validate_file(path).codes

    @pytest.mark.parametrize("class_id", [0, -1, 65535, 70000])
    def test_Q14_S5_3_every_encoder_refuses_an_id_out_of_range(
        self, tmp_path: Path, class_id: int
    ):
        """numpy 2 raised OverflowError at the uint16 cast; numpy 1.24 wrapped."""
        from medh5.annotations.geometric import encode_boxes
        from medh5.annotations.voxel import (
            InstanceInput,
            encode_instances,
            encode_masks,
            encode_probmap,
        )

        mask = np.zeros(SHAPE, bool)
        mask[1, 1, 1] = True
        for encode in (
            lambda: encode_masks({class_id: mask}, "labelmap", SHAPE),
            lambda: encode_probmap({class_id: mask.astype(float)}, SHAPE),
            lambda: encode_instances([InstanceInput(class_id, 1, mask=mask)], SHAPE),
            lambda: encode_boxes([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], [class_id]),
        ):
            with pytest.raises(MEDH5ValidationError) as caught:
                encode()
            assert caught.value.code == "E303"


# --------------------------------------------------------------------------
# F-23 --- one class, several assertions
# --------------------------------------------------------------------------


class TestF23ClassificationRows:
    def test_F23_S9_one_class_per_lesion(self, tmp_path: Path):
        """ "Lesion 1 malignant, lesion 2 benign" needs the class twice."""
        path = tmp_path / "rows.medh5"
        with _writer(path) as w:
            w.add_classification(
                "malignancy",
                [("c1", 1.0, 1), ("c1", 0.0, 2)],
                scope="instance",
            )
        with medh5.open(path) as sample:
            annotation = sample.annotations["malignancy"]
            assert len(list(annotation.assertions())) == 2
            by_unit = annotation.by_scope_id()
            assert [(a.class_id, a.value) for a in by_unit[1]] == [(1, 1.0)]
            assert [(a.class_id, a.value) for a in by_unit[2]] == [(1, 0.0)]

    def test_F23_rows_carry_schemes(self, tmp_path: Path):
        path = tmp_path / "schemes.medh5"
        with _writer(path) as w:
            w.add_classification(
                "grade",
                [("c1", 1.0, 1, "birads", "4"), ("c2", 1.0, 2, "birads", "2")],
                scope="instance",
            )
        with medh5.open(path) as sample:
            assertions = sample.annotations["grade"].assertions()
            assert [a.scheme_value for a in assertions] == ["4", "2"]

    def test_F23_rows_of_different_widths_are_refused(self, tmp_path: Path):
        with _writer(tmp_path / "mixed.medh5") as w:
            with pytest.raises(MEDH5ValidationError) as caught:
                w.add_classification(
                    "x", [("c1", 1.0, 1), ("c2", 0.0)], scope="instance"
                )
            assert caught.value.code == "E405"
            w.abort()

    def test_F23_a_column_given_twice_is_refused(self, tmp_path: Path):
        with _writer(tmp_path / "twice.medh5") as w:
            with pytest.raises(MEDH5ValidationError, match="once"):
                w.add_classification(
                    "x", [("c1", 1.0, 1)], scope="instance", scope_ids=[1]
                )
            w.abort()


# --------------------------------------------------------------------------
# W16 --- what training code receives
# --------------------------------------------------------------------------


def _two_classes() -> tuple[Any, Any, Any]:
    """Two disjoint classes and an ignore region touching neither."""
    liver = np.zeros(SHAPE, bool)
    liver[1:5, 1:6, 1:6] = True
    spleen = np.zeros(SHAPE, bool)
    spleen[1:5, 7:11, 7:11] = True
    region = np.zeros(SHAPE, bool)
    region[6:, :, :] = True
    return liver, spleen, region


def _encoded(path: Path, encoding: str, *, overlap: bool = False) -> Any:
    """One annotation ``seg`` under *encoding*, carrying the ignore region."""
    from medh5.annotations.voxel import InstanceInput

    liver, spleen, region = _two_classes()
    if overlap:
        region = region.copy()
        region[3:5, 2:4, 2:4] = True  # inside the liver
    with _writer(path) as w:
        if encoding == "instances":
            w.add_segmentation(
                "seg",
                grid="g",
                instances=[
                    InstanceInput(1, 1, mask=liver),
                    InstanceInput(2, 2, mask=spleen),
                ],
                ignore=region,
            )
        elif encoding == "probmap":
            w.add_segmentation(
                "seg",
                grid="g",
                probabilities={1: liver.astype(float), 2: spleen.astype(float)},
                ignore=region,
            )
        else:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: spleen},
                ignore=region,
                encoding=encoding,
            )
    return region


ENCODINGS = ("labelmap", "layers", "bitmask", "instances", "probmap")


class TestW16Loaders:
    """F-16, F-17, F-18, F-24, L-34: the file's contracts reach the model."""

    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_F16_S7_7_the_ignore_region_reaches_the_item(
        self, tmp_path: Path, encoding: str
    ):
        """Every loader item dropped it; `layers` & co. then trained it as 0."""
        from medh5.torch import VolumeDataset

        path = tmp_path / f"{encoding}.medh5"
        region = _encoded(path, encoding)
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].kind == encoding
            np.testing.assert_array_equal(sample.ignore_region("seg"), region)
        for label_format in ("onehot", "labelmap"):
            item = VolumeDataset(
                [path],
                images=["CT"],
                annotations={"seg": [1, 2]},
                label_format=label_format,
            )[0]
            np.testing.assert_array_equal(item["ignore"]["seg"].numpy(), region)
            assert item["meta"]["annotated"]["seg"].tolist() == [True, True]
            assert item["valid"]["CT"].numpy().all()
            if label_format == "labelmap":
                label = item["label"]["seg"].numpy()
                assert set(np.unique(label[region]).tolist()) == {65535}
                assert set(np.unique(label[~region]).tolist()) == {0, 1, 2}

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_F16_padding_is_ignored_and_never_valid(
        self, tmp_path: Path, encoding: str
    ):
        """Padding was labelled 0 --- background --- in every format."""
        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset

        path = tmp_path / f"pad-{encoding}.medh5"
        _encoded(path, encoding)
        size = (10, 16, 16)  # larger than the grid on every axis
        item = PatchDataset(
            [path],
            PatchSampler(size, strategy="uniform"),
            images=["CT"],
            annotations={"seg": [1, 2]},
            label_format="labelmap",
        )[0]
        pads = item["meta"]["patch"]["pad"]
        outside = np.ones(size, bool)
        outside[
            tuple(
                slice(before, n - after)
                for (before, after), n in zip(pads, size, strict=True)
            )
        ] = False
        assert outside.any()
        assert item["ignore"]["seg"].numpy()[outside].all()
        assert not item["valid"]["CT"].numpy()[outside].any()
        assert (item["label"]["seg"].numpy()[outside] == 65535).all()
        assert item["valid"]["CT"].numpy()[~outside].all()

    def test_F16_S4_4_the_valid_mask_reaches_the_item(self, tmp_path: Path):
        from medh5.torch import VolumeDataset

        fov = np.zeros(SHAPE, bool)
        fov[:, 2:10, 2:10] = True
        path = tmp_path / "fov.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_mask("fov", fov, grid="g")
            w.add_image(
                "CT",
                np.zeros(SHAPE, np.int16),
                grid="g",
                modality="CT",
                valid_mask="fov",
            )
        item = VolumeDataset([path], images=["CT"])[0]
        np.testing.assert_array_equal(item["valid"]["CT"].numpy(), fov)

    def test_F16_S11_3_coverage_is_per_channel(self, tmp_path: Path):
        """A 0 in a class nobody looked for is not a negative."""
        from medh5.torch import VolumeDataset

        liver, spleen, _ = _two_classes()
        path = tmp_path / "partial.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks={1: liver, 2: spleen}, annotated_classes=[1]
            )
        item = VolumeDataset([path], images=["CT"], annotations={"seg": [2, 1]})[0]
        assert item["meta"]["annotated"]["seg"].tolist() == [False, True]

    def test_F16_the_batch_collates(self, tmp_path: Path):
        from torch.utils.data import DataLoader

        from medh5.torch import VolumeDataset, collate

        paths = []
        for n in range(2):
            path = tmp_path / f"b{n}.medh5"
            _encoded(path, "layers")
            paths.append(path)
        dataset = VolumeDataset(paths, images=["CT"], annotations={"seg": [1, 2]})
        for collate_fn in (collate, None):
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))
            assert tuple(batch["ignore"]["seg"].shape) == (2, *SHAPE)
            assert tuple(batch["valid"]["CT"].shape) == (2, *SHAPE)
            assert tuple(batch["meta"]["annotated"]["seg"].shape) == (2, 2)

    def test_F16_the_monai_bridge_writes_65535_under_every_encoding(
        self, tmp_path: Path
    ):
        pytest.importorskip("monai")
        import medh5.monai as bridge

        for encoding in ENCODINGS:
            path = tmp_path / f"monai-{encoding}.medh5"
            region = _encoded(path, encoding)
            with medh5.open(path) as sample:
                label = bridge.to_dict(sample, images=["CT"], annotations=["seg"])
                values = np.asarray(label["seg"])[region]
            assert set(np.unique(values).tolist()) == {65535}, encoding

    def test_F17_persistent_workers_see_the_epoch(self, tmp_path: Path):
        """The documented setup drew epoch 0's patches for the whole run."""
        from torch.utils.data import DataLoader

        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset, collate, worker_init_fn

        path = _plain(tmp_path / "big.medh5", shape=(32, 64, 64))
        dataset = PatchDataset(
            [path],
            PatchSampler((8, 8, 8), strategy="uniform"),
            images=["CT"],
            samples_per_volume=6,
        )
        loader = DataLoader(
            dataset,
            batch_size=6,
            num_workers=2,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

        def draw(epoch: int) -> list[tuple[int, ...]]:
            dataset.set_epoch(epoch)
            batch = next(iter(loader))
            return sorted(tuple(s) for s in batch["meta"]["patch"]["start"])

        first, second, again = draw(0), draw(1), draw(0)
        assert first != second
        assert first == again
        assert dataset.epoch == 0

    def test_F17_the_epoch_is_an_attribute_as_before(self, tmp_path: Path):
        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset

        dataset = PatchDataset(
            [_plain(tmp_path / "e.medh5")], PatchSampler(4), images=["CT"]
        )
        dataset.epoch = 3
        assert dataset.epoch == 3

    def _visits(self, path: Path, *, frames: tuple[Any, Any], register: bool) -> Path:
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_timepoint("tp0", index=0, days_from_baseline=0)
            w.add_timepoint("tp1", index=1, days_from_baseline=90)
            for n, (origin, frame) in enumerate(
                zip(((0, 0, 0), (-40, 15, 7)), frames, strict=True)
            ):
                w.add_grid(
                    f"g{n}",
                    shape=(16, 32, 32),
                    spacing=(2.0, 1.0, 1.0),
                    origin=origin,
                    timepoint=f"tp{n}",
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT{n}",
                    np.zeros((16, 32, 32), np.int16),
                    grid=f"g{n}",
                    modality="CT",
                )
            if register:
                w.add_transform(
                    "t", kind="affine", from_frame="F0", to_frame="F1", matrix=np.eye(4)
                )
        return path

    def test_F18_S3_3_two_frameless_grids_are_not_registered(self, tmp_path: Path):
        """`None == None` let unregistered visits through as aligned."""
        from medh5.sampling import PatchSampler
        from medh5.torch import PairedPatchDataset

        path = self._visits(tmp_path / "nif.medh5", frames=(None, None), register=False)
        dataset = PairedPatchDataset(
            [path], PatchSampler(8, strategy="uniform"), align="transform"
        )
        with pytest.raises(MEDH5ValidationError, match="frame of reference"):
            dataset[0]
        loose = PairedPatchDataset([path], PatchSampler(8), align="none")
        assert loose[0]["meta"]["aligned"] == "none"

    def test_F18_registered_and_shared_frames_still_map(self, tmp_path: Path):
        from medh5.sampling import PatchSampler
        from medh5.torch import PairedPatchDataset

        for name, frames, register in (
            ("reg", ("F0", "F1"), True),
            ("shared", ("F", "F"), False),
        ):
            path = self._visits(
                tmp_path / f"{name}.medh5", frames=frames, register=register
            )
            item = PairedPatchDataset([path], PatchSampler(8), align="transform")[0]
            assert item["meta"]["aligned"] == "transform"
            assert set(item["valid"]) == {"tp0", "tp1"}

    def test_F24_class_weights_come_from_voxel_classes(self, tmp_path: Path):
        """A sample-level diagnosis became the heaviest segmentation class."""
        from medh5.dataset.stats import compute_stats

        path = tmp_path / "dx.medh5"
        liver = np.zeros(SHAPE, bool)
        liver[:4, :8, :8] = True  # 256 voxels
        spleen = np.zeros(SHAPE, bool)
        spleen[4:5, :8, :8] = True  # 64 voxels
        with _writer(path) as w:
            w.add_segmentation("organs", grid="g", masks={1: liver, 2: spleen})
            w.add_classification("dx", {3: 1.0})
        stats = compute_stats([path])
        assert set(stats.classes) == {1, 2}
        weights = stats.class_weights()
        assert set(weights) == {1, 2}
        assert weights[2] / weights[1] == pytest.approx(256 / 64)

    def test_F24_an_absent_class_gets_no_weight_and_a_warning(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        liver, _, _ = _two_classes()
        path = tmp_path / "absent.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks={1: liver}, annotated_classes=[1, 2]
            )
        stats = compute_stats([path])
        assert stats.classes[2].examined_in == 1 and stats.classes[2].voxels == 0
        with pytest.warns(UserWarning, match=r"\[2\]"):
            assert set(stats.class_weights()) == {1}

    def test_F24_present_and_examined_count_samples(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        liver, _, _ = _two_classes()
        path = tmp_path / "long.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(_label_set())
            for n in range(2):
                w.add_timepoint(f"tp{n}", index=n)
                w.add_grid(
                    f"g{n}", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint=f"tp{n}"
                )
                w.add_image(
                    f"CT{n}", np.zeros(SHAPE, np.int16), grid=f"g{n}", modality="CT"
                )
                w.add_segmentation(f"seg{n}", grid=f"g{n}", masks={1: liver})
        stats = compute_stats([path])
        assert (stats.classes[1].present_in, stats.classes[1].examined_in) == (1, 1)
        assert stats.classes[1].voxels == 2 * int(liver.sum())

    def test_L34_S4_4_moments_honour_the_valid_mask(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        shape = (8, 32, 32)
        image = np.full(shape, -3024, np.int16)
        yy, xx = np.mgrid[:32, :32]
        circle = (yy - 15.5) ** 2 + (xx - 15.5) ** 2 < 14**2
        image[:, circle] = 40
        path = tmp_path / "fov.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_mask("fov", np.broadcast_to(circle, shape).copy(), grid="g")
            w.add_image("CT", image, grid="g", modality="CT", valid_mask="fov")
            w.add_grid("small", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            w.add_mask("other", np.ones((2, 2, 2), bool), grid="small")
        stats = compute_stats([path])
        moments = stats.images["CT"]
        assert moments.mean == pytest.approx(40.0)
        assert moments.minimum == 40
        assert moments.count == int(circle.sum()) * shape[0]
        assert stats.total_voxels == int(np.prod(shape))


# --------------------------------------------------------------------------
# W18 --- converters that refuse what they cannot place
# --------------------------------------------------------------------------


class TestW18Converters:
    """F-19, F-25."""

    @pytest.fixture(autouse=True)
    def _pydicom(self) -> None:
        pytest.importorskip("pydicom")

    def _tilted(self, root: Path, degrees: float) -> Path:
        import pydicom

        from tests.v1.conftest import write_dicom_series

        write_dicom_series(
            root,
            patient_id="P1",
            study_uid="1.2.826.0.1.3680043.8.498.99",
            study_date="20240102",
            shape=(20, 16, 20),
            spacing=(2.5, 0.8, 0.9),
        )
        shear = 2.5 * np.tan(np.radians(degrees))  # mm per slice, in plane
        for path in sorted(root.glob("*.dcm")):
            ds = pydicom.dcmread(path)
            x, y, z = (float(v) for v in ds.ImagePositionPatient)
            k = round((-10.0 - x) / 2.5)
            ds.ImagePositionPatient = [x, y + k * shear, z]
            ds.save_as(path)
        return root

    def test_F19_S3_a_tilted_gantry_stack_is_refused(self, tmp_path: Path):
        """It imported as an orthonormal grid, 17 mm off at the last slice."""
        from medh5.io.dicom import from_dicom

        source = self._tilted(tmp_path / "tilt", 20.0)
        with pytest.raises(MEDH5ValidationError, match=r"20\.0 degrees") as caught:
            from_dicom(source, tmp_path / "tilt.medh5")
        assert "17.3 mm" in str(caught.value)
        assert not (tmp_path / "tilt.medh5").exists()

    def test_F19_an_untilted_stack_imports_and_says_it_was_checked(
        self, tmp_path: Path
    ):
        from medh5.io.dicom import from_dicom

        source = self._tilted(tmp_path / "flat", 0.0)
        report = from_dicom(source, tmp_path / "flat.medh5")
        assert "slice_alignment" in {n.kind for n in report.notes}
        with medh5.open(tmp_path / "flat.medh5") as sample:
            assert sample.images["CT_tp0"].shape == (20, 16, 20)

    def _studies(self, root: Path, demographics: list[tuple[str, str]]) -> Path:
        import pydicom

        from tests.v1.conftest import write_dicom_series

        for n, (sex, born) in enumerate(demographics, start=1):
            directory = root / f"s{n}"
            write_dicom_series(
                directory,
                patient_id="ANONYMOUS",
                study_uid=f"1.2.826.0.1.3680043.8.498.{n}",
                study_date=f"202{n}0101",
                seed=n,
            )
            for path in directory.glob("*.dcm"):
                ds = pydicom.dcmread(path)
                ds.PatientSex = sex
                ds.PatientBirthDate = born
                ds.save_as(path)
        return root

    def test_F25_S2_2_contradicted_demographics_are_not_one_subject(
        self, tmp_path: Path
    ):
        """Two patients under an anonymiser's constant PatientID became one."""
        from medh5.io.dicom import from_dicom

        source = self._studies(tmp_path / "a", [("F", "19500101"), ("M", "19900101")])
        report = from_dicom(source, tmp_path / "out")
        assert len(report.outputs) == 2
        guesses = [n for n in report.notes if n.kind == "identity"]
        assert len(guesses) == 1 and guesses[0].severity == "guess"
        assert guesses[0].detail["conflicts"] == {
            "PatientBirthDate": ["19500101", "19900101"],
            "PatientSex": ["F", "M"],
        }
        subjects = set()
        for output in report.outputs:
            with medh5.open(output) as sample:
                assert len(sample.timepoints) == 1
                subjects.add(sample.identity.subject_id)
                assert (
                    sample.identity.extra["id_source"]["subject_id"]
                    == "dicom:StudyInstanceUID"
                )
        assert len(subjects) == 2
        # Named by the whole StudyInstanceUID, not by one missing its last part.
        assert sorted(Path(p).name for p in report.outputs) == [
            "1.2.826.0.1.3680043.8.498.1.medh5",
            "1.2.826.0.1.3680043.8.498.2.medh5",
        ]

    def test_F25_consistent_demographics_still_group(self, tmp_path: Path):
        from medh5.io.dicom import from_dicom

        source = self._studies(tmp_path / "b", [("F", "19500101"), ("F", "")])
        out = tmp_path / "one.medh5"
        report = from_dicom(source, out)
        assert report.outputs == [str(out)]
        with medh5.open(out) as sample:
            assert sample.timepoints.ids == ("tp0", "tp1")
        assert "identity" not in {n.kind for n in report.notes}

    def test_F25_a_uid_in_a_file_name_is_reported_by_scrub(self, tmp_path: Path):
        from medh5.curation import scrub
        from medh5.io.dicom import from_dicom

        source = self._studies(tmp_path / "c", [("F", "19500101"), ("M", "19900101")])
        report = from_dicom(source, tmp_path / "out")
        findings = scrub.scan(report.outputs[0]).findings
        assert "file_name" in {f.rule for f in findings}


# --------------------------------------------------------------------------
# W19 --- the writer and the validator agree
# --------------------------------------------------------------------------


class TestW19WriterEqualsValidator:
    """L-19...L-25, L-37, S-13: one definition of a valid file."""

    def test_L19_S10_3_a_field_off_its_grid_is_refused_at_the_call(
        self, tmp_path: Path
    ):
        """A (3,4,4,4) field on a 16³ grid was written, then read as T(x) = x."""
        with medh5.create(tmp_path / "disp.medh5", sample_id="s") as w:
            w.add_timepoint("tp0", index=0)
            w.add_timepoint("tp1", index=1)
            for n in range(2):
                w.add_grid(
                    f"g{n}",
                    shape=(16, 16, 16),
                    spacing=(1.0, 1.0, 1.0),
                    frame_uid=f"F{n}",
                    timepoint=f"tp{n}",
                )
                w.add_image(
                    f"CT{n}", np.zeros((16,) * 3, np.int16), grid=f"g{n}", modality="CT"
                )
            with pytest.raises(MEDH5ValidationError, match="lattice") as caught:
                w.add_transform(
                    "t",
                    kind="displacement",
                    from_frame="F0",
                    to_frame="F1",
                    field=np.ones((3, 4, 4, 4), np.float32),
                    field_grid="g0",
                )
            assert caught.value.code == "E503"
            with pytest.raises(MEDH5ValidationError, match="components"):
                w.add_transform(
                    "b",
                    kind="bspline",
                    from_frame="F0",
                    to_frame="F1",
                    control_points=np.zeros((2, 5, 5), np.float32),
                    cp_grid="g0",
                )
            w.abort()

    def test_L19_commit_refuses_what_the_validator_rejects(self, tmp_path: Path):
        """Anything written through the handle is held to the same rules."""
        path = _plain(tmp_path / "keep.medh5")
        before = path.read_bytes()
        writer = medh5.amend(path)
        del writer.handle["images/CT"].attrs["grid"]
        with pytest.raises(MEDH5ValidationError, match="fail validation") as caught:
            writer.commit()
        assert caught.value.code.startswith("E")
        assert path.read_bytes() == before, "the refused commit replaced the file"
        assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name]

    def test_L20_S5_1_a_ref_label_set_is_written(self, tmp_path: Path):
        """The writer refused what the validator accepts: E402 under form=ref."""
        path = tmp_path / "ref.medh5"
        mask = np.zeros(SHAPE, bool)
        mask[1, 2, 2] = True
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.label_set(_label_set().as_ref("https://example.org/vocab.json"))
            w.add_segmentation("seg", grid="g", masks={1: mask})
        assert validate_file(path).ok

    def test_L21_the_schema_check_needs_no_optional_dependency(self):
        """1.4.1 made jsonschema a core dependency so E005 was checked
        everywhere.  2.0 compiles the validator into the engine: NumPy is the
        only runtime dependency, and the check is always there."""
        from importlib.metadata import requires

        from medh5.document import validate_against_schema

        core = [r for r in requires("medh5") or () if "extra ==" not in r]
        assert [re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0] for r in core] == ["numpy"]
        assert validate_against_schema({"identity": {}})

    def test_L21_S2_4_commit_checks_the_schema(self, tmp_path: Path):
        """A class key like `Left-Kidney` failed E005 only where jsonschema was."""
        with _writer(tmp_path / "schema.medh5") as w:
            w.extra("x", {"fine": 1})
            w.document.label_set = LabelSet(
                "bad",
                version="1",
                classes=[LabelClass(1, "Left-Kidney", "Left kidney")],
            )
            with pytest.raises(MEDH5ValidationError) as caught:
                w.commit()
            assert caught.value.code == "E005"
            w.abort()

    @pytest.mark.parametrize(
        ("field", "value"), [("units", "cm"), ("units", "inch"), ("time_units", "min")]
    )
    def test_L22_S3_2_units_outside_the_list_are_refused(
        self, tmp_path: Path, field: str, value: str
    ):
        with medh5.create(tmp_path / "u.medh5", sample_id="s") as w:
            options: dict[str, Any] = {field: value}
            if field == "time_units":
                options.update(
                    axis_kinds=("time", "spatial", "spatial", "spatial"),
                    time_values=[0.0, 1.0],
                )
                shape: tuple[int, ...] = (2, *SHAPE)
            else:
                shape = SHAPE
            with pytest.raises(MEDH5ValidationError, match=value) as caught:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), **options)
            assert caught.value.code == "E109"
            w.abort()

    def test_L23_one_source_and_a_matching_encoding(self, tmp_path: Path):
        """`masks=` was dropped when `probabilities=` came too."""
        from medh5.annotations.voxel import InstanceInput

        liver, spleen, _ = _two_classes()
        with _writer(tmp_path / "l23.medh5") as w:
            for call in (
                {"masks": {1: liver}, "probabilities": {2: spleen.astype(float)}},
                {"probabilities": {1: liver.astype(float)}, "encoding": "layers"},
                {"instances": [InstanceInput(1, 1, mask=liver)], "encoding": "bitmask"},
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    w.add_segmentation("s", grid="g", **call)
                assert caught.value.code == "E404"
            kind, _ = w.add_segmentation(
                "p",
                grid="g",
                probabilities={1: liver.astype(float)},
                encoding="probmap",
            )
            assert kind == "probmap"

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_L24_S7_7_an_overlapping_region_reads_back_whole(
        self, tmp_path: Path, encoding: str
    ):
        """Under `labelmap` 56 of 64 ignored voxels came back; `bitmask` gave 64."""
        path = tmp_path / f"ovl-{encoding}.medh5"
        region = _encoded(path, encoding, overlap=True)
        liver, spleen, _ = _two_classes()
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            np.testing.assert_array_equal(sample.ignore_region("seg"), region)
            np.testing.assert_array_equal(annotation.dense([1])[0], liver)
            np.testing.assert_array_equal(annotation.dense([2])[0], spleen)
            assert annotation.header.ignore_mask == "seg_ignore"
        assert validate_file(path).ok

    def test_L24_a_region_clear_of_the_classes_stays_in_band(self, tmp_path: Path):
        path = tmp_path / "band.medh5"
        _encoded(path, "labelmap")
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].header.ignore_mask is None
            assert "seg_ignore" not in sample.annotations

    def _lesions(self, path: Path) -> Path:
        from medh5.annotations.voxel import InstanceInput

        first = np.zeros(SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        with _writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 11, mask=first),
                    InstanceInput(1, 22, mask=second),
                ],
            )
        return path

    def test_L25_S7_4_a_transcode_does_not_drop_identity_silently(self, tmp_path: Path):
        """After instances -> labelmap, tracks() was empty and nothing said so."""
        path = self._lesions(tmp_path / "tc.medh5")
        with medh5.amend(path) as w:
            with pytest.raises(MEDH5ValidationError, match="drop_identity") as caught:
                w.transcode_annotation("les", "labelmap")
            assert caught.value.code == "E404"
        with medh5.amend(path) as w:
            w.transcode_annotation("les", "labelmap", drop_identity=True)
        with medh5.open(path) as sample:
            assert sample.annotations["les"].kind == "labelmap"
            (activity,) = sample.document.provenance.activities_by_type("transcode")
            assert activity.params["dropped"] == "instance identity"
            assert activity.params["objects"] == 2
            assert activity.outputs == ("annotations/les",)
        assert validate_file(path).ok

    def test_L25_the_cli_flag(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._lesions(tmp_path / "cli.medh5")
        args = ["seg", "convert", str(path), "les", "--to", "layers"]
        assert main(args) != 0
        assert main([*args, "--drop-identity"]) == 0
        capsys.readouterr()
        with medh5.open(path) as sample:
            assert sample.annotations["les"].kind == "layers"

    def test_L37_S7_4_examined_and_none_found(self, tmp_path: Path):
        """`instances=[]` was refused (E410): a resolved lesion read unexamined."""
        from medh5.annotations.voxel import InstanceInput

        lesion = np.zeros(SHAPE, bool)
        lesion[2:4, 2:4, 2:4] = True
        path = tmp_path / "resolved.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(_label_set())
            for n in range(2):
                w.add_timepoint(f"tp{n}", index=n)
                w.add_grid(
                    f"g{n}", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint=f"tp{n}"
                )
                w.add_image(
                    f"CT{n}", np.zeros(SHAPE, np.int16), grid=f"g{n}", modality="CT"
                )
            w.add_segmentation(
                "les0", grid="g0", instances=[InstanceInput(1, 7, mask=lesion)]
            )
            w.add_segmentation("les1", grid="g1", instances=[], annotated_classes=[1])
        report = validate_file(path, level="strict")
        assert not [c for c in report.codes if c.startswith("E")], report.codes
        with medh5.open(path) as sample:
            empty = sample.annotations["les1"]
            assert (empty.class_ids, empty.annotated_class_ids) == ((1,), (1,))
            assert not empty.dense().any()
            assert list(empty.instances()) == []
            tracking = sample.tracks()
            assert tracking.state_at(7, "tp1") == "resolved"
            assert tracking.is_resolved(7)

    def test_L37_an_empty_list_alone_says_nothing_and_is_refused(self, tmp_path: Path):
        with _writer(tmp_path / "bare.medh5") as w:
            with pytest.raises(
                MEDH5ValidationError, match="annotated_classes"
            ) as caught:
                w.add_segmentation("les", grid="g", instances=[])
            assert caught.value.code == "E410"
            w.abort()

    def test_property_12_no_tool_writes_what_the_validator_rejects(
        self, tmp_path: Path
    ):
        """Every corpus file, amended as it is: refused, or written valid.

        The writer and the validator each enforced a hand-kept list of rules,
        and the two drifted (L-19...L-22).  With ``commit()`` running the
        validator there is one list --- which this checks from the outside, over
        the conformance corpus: a no-op amend of each of its 117 files either
        refuses the file or writes one the validator passes, and a valid case
        comes back valid.
        """
        from medh5.conformance import CASES
        from medh5.errors import MEDH5Error

        refused = written = 0
        for case in CASES:
            path = tmp_path / f"{case.name}{case.suffix}"
            case.build(path)
            try:
                with medh5.amend(path):
                    pass
            except (MEDH5Error, OSError):
                refused += 1
                assert not case.valid or case.suffix == ".medh5c", case.name
                continue
            written += 1
            errors = [c for c in validate_file(path).codes if c.startswith("E")]
            assert errors == [], (case.name, errors)
        assert refused and written
