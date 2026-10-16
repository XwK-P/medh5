"""DICOM series import (``medh5.io.dicom``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.report import ConversionReport
from medh5.validate import validate_file
from tests.helpers import write_dicom_series

pydicom = pytest.importorskip("pydicom")


class TestDicomSameModality:
    """A study with several series of one modality imports, named by series."""

    def test_two_series_of_one_modality_import_side_by_side(
        self, tmp_path: Path
    ) -> None:
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom

        study = generate_uid()
        root = tmp_path / "dicom"
        first = write_dicom_series(
            root / "a",
            patient_id="P1",
            study_uid=study,
            study_date="20260101",
            modality="MR",
            seed=1,
        )
        second = write_dicom_series(
            root / "b",
            patient_id="P1",
            study_uid=study,
            study_date="20260101",
            modality="MR",
            seed=2,
        )
        report = from_dicom(root, tmp_path / "out.medh5")
        assert report.of_kind("image_ids"), "the numbering is recorded"
        by_uid = {first["series_uid"]: first, second["series_uid"]: second}
        ordered = sorted(by_uid)  # SeriesInstanceUID order
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["MR_1_tp0", "MR_2_tp0"]
            assert sorted(sample.grids) == ["mr_1_tp0", "mr_2_tp0"]
            uids = sample.timepoints["tp0"].series_uids
            assert uids == {"MR_1_tp0": ordered[0], "MR_2_tp0": ordered[1]}
            assert sample.images["MR_1_tp0"].grid_id == "mr_1_tp0"

    def test_a_single_series_keeps_its_short_name(self, tmp_path: Path) -> None:
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom

        series = write_dicom_series(
            tmp_path / "dicom",
            patient_id="P1",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        report = from_dicom(tmp_path / "dicom", tmp_path / "out.medh5")
        assert not report.of_kind("image_ids")
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["CT_tp0"]
            assert sample.timepoints["tp0"].series_uids == {
                "CT_tp0": series["series_uid"]
            }


class TestConverters:
    """A tilted gantry, demographics that contradict one PatientID, and a UID in
    a file name: what the DICOM import refuses or reports."""

    def _tilted(self, root: Path, degrees: float) -> Path:
        import pydicom

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

    def test_S3_a_tilted_gantry_stack_is_refused(self, tmp_path: Path):
        """It imported as an orthonormal grid, 17 mm off at the last slice."""
        from medh5.io.dicom import from_dicom

        source = self._tilted(tmp_path / "tilt", 20.0)
        with pytest.raises(MEDH5ValidationError, match=r"20\.0 degrees") as caught:
            from_dicom(source, tmp_path / "tilt.medh5")
        assert "17.3 mm" in str(caught.value)
        assert not (tmp_path / "tilt.medh5").exists()

    def test_S3_an_untilted_stack_imports_and_says_it_was_checked(self, tmp_path: Path):
        from medh5.io.dicom import from_dicom

        source = self._tilted(tmp_path / "flat", 0.0)
        report = from_dicom(source, tmp_path / "flat.medh5")
        assert "slice_alignment" in {n.kind for n in report.notes}
        with medh5.open(tmp_path / "flat.medh5") as sample:
            assert sample.images["CT_tp0"].shape == (20, 16, 20)

    def _studies(self, root: Path, demographics: list[tuple[str, str]]) -> Path:
        import pydicom

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

    def test_S2_2_contradicted_demographics_are_not_one_subject(self, tmp_path: Path):
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

    def test_S2_2_consistent_demographics_still_group(self, tmp_path: Path):
        from medh5.io.dicom import from_dicom

        source = self._studies(tmp_path / "b", [("F", "19500101"), ("F", "")])
        out = tmp_path / "one.medh5"
        report = from_dicom(source, out)
        assert report.outputs == [str(out)]
        with medh5.open(out) as sample:
            assert sample.timepoints.ids == ("tp0", "tp1")
        assert "identity" not in {n.kind for n in report.notes}

    def test_S11_4_a_uid_in_a_file_name_is_reported_by_scrub(self, tmp_path: Path):
        from medh5.curation import scrub
        from medh5.io.dicom import from_dicom

        source = self._studies(tmp_path / "c", [("F", "19500101"), ("M", "19900101")])
        report = from_dicom(source, tmp_path / "out")
        findings = scrub.scan(report.outputs[0]).findings
        assert "file_name" in {f.rule for f in findings}

    def test_S2_3_a_non_ascii_patient_id_names_an_ascii_sample(self, tmp_path: Path):
        """A sample id is `[A-Za-z0-9_.-]`, and the id was made by keeping every
        alphanumeric character: `Müller^Hans` aborted the import (E003)."""
        import pydicom

        from medh5.io.dicom import from_dicom

        root = tmp_path / "dicom"
        series = write_dicom_series(
            root, patient_id="X", study_uid="1.2.826.0.1.3680043.8.498.7",
            study_date="20240102", shape=(2, 4, 4),
        )  # fmt: skip
        for path in series["paths"]:
            ds = pydicom.dcmread(path)
            ds.SpecificCharacterSet = "ISO_IR 192"
            ds.PatientID = "Müller^Hans"
            ds.save_as(path, enforce_file_format=True)
        report = from_dicom(root, f"{tmp_path / 'out'}{os.sep}")
        (written,) = report.outputs
        assert Path(written).name == "Muller_Hans.medh5"
        with medh5.open(written) as sample:
            assert sample.identity.sample_id == "Muller_Hans"
            assert sample.identity.subject_id == "Müller^Hans", "the id is kept whole"


def _restack(paths: list[str], along: list[float]) -> None:
    """Move the slices of a `write_dicom_series` stack, in writing order, to
    *along* millimetres along their normal (``-x``), to four decimals."""
    import pydicom

    for path, distance in zip(sorted(paths), along, strict=True):
        ds = pydicom.dcmread(path)
        _, y, z = (float(v) for v in ds.ImagePositionPatient)
        ds.ImagePositionPatient = [-float(f"{distance:.4f}"), y, z]
        ds.save_as(path, enforce_file_format=True)


def _instance_uids(root: Path) -> list[str]:
    """The SOPInstanceUIDs of a `write_dicom_series` stack, in writing order."""
    return [str(pydicom.dcmread(p).SOPInstanceUID) for p in sorted(root.glob("*.dcm"))]


class TestSlicePositions:
    """Every slice is held to its place in the regular stack, within a tenth of
    a slice and at least 0.01 mm.  Each gap was held to 1 % of the median gap
    instead: gaps within it drifted slices off the grid unseen, and positions
    merely rounded to 0.01 mm were refused (F07)."""

    @staticmethod
    def _series(root: Path, along: Any) -> Any:
        from medh5.io.dicom import scan_dicom

        along = [float(z) for z in along]
        written = write_dicom_series(
            root, patient_id="P", study_uid="1.2.826.0.1.3680043.8.498.5",
            study_date="20240102", shape=(len(along), 4, 4),
        )  # fmt: skip
        _restack(written["paths"], along)
        (series,) = scan_dicom(root)
        return series

    def test_S3_3_drift_that_every_gap_hides_is_refused(self, tmp_path: Path):
        """Twenty gaps 0.9 % long, then twenty 0.9 % short: each within 1 % of
        the median, and the middle slice 0.45 mm --- 1.8 tenths of a slice ---
        from where the grid put it."""
        from medh5.cli import EXIT_ERROR, main
        from medh5.io.dicom import read_series

        gaps = [2.5225] * 20 + [2.4775] * 20
        along = np.concatenate([[0.0], np.cumsum(gaps)]) + 10.0
        series = self._series(tmp_path / "drift", along)
        with pytest.raises(MEDH5ValidationError, match="irregular slice gaps") as exc:
            read_series(series)
        found = re.search(
            r"the slice at index (\d+) of 41 \(instance ([\d.]+)\) ([\d.]+) mm off, "
            r"beyond the ([\d.]+) mm tolerance",
            str(exc.value),
        )
        assert found, str(exc.value)
        index, uid, off, tolerance = found.groups()
        assert float(off) > float(tolerance) >= 0.25
        assert uid == _instance_uids(tmp_path / "drift")[int(index)]
        out = tmp_path / "drift.medh5"
        assert main(["convert", "from-dicom", str(tmp_path / "drift"), str(out)]) == (
            EXIT_ERROR
        )
        assert not out.exists()

    @pytest.mark.parametrize(
        ("step", "decimals", "count"), [(0.625, 2, 120), (1.25, 1, 60)]
    )
    def test_S3_3_rounded_positions_are_a_regular_stack(
        self, tmp_path: Path, step: float, decimals: int, count: int
    ):
        """Positions rounded per slice accumulate nothing: 0.625 mm slices
        written to 0.01 mm have gaps of 0.62 and 0.63 mm, and were refused."""
        from medh5.io.dicom import read_series

        along = np.round(10.0 + step * np.arange(count), decimals)
        gaps = np.diff(along)
        median = np.median(gaps)
        assert np.max(np.abs(gaps - median)) > 0.01 * median, "1 % of the median gap"
        report = ConversionReport()
        _, geometry = read_series(self._series(tmp_path / "r", along), report=report)
        assert geometry["spacing"][0] == pytest.approx(step, abs=1e-3)
        (note,) = report.of_kind("slice_spacing")
        rounding = 0.5 * 10.0**-decimals
        assert note.detail["max_through_plane_error_mm"] <= 2 * rounding + 1e-6
        assert note.detail["tolerance_mm"] == pytest.approx(0.1 * step, abs=1e-4)
        # Where the grid puts each slice is within the stated bound of the file.
        grid = geometry["origin"][0] - np.arange(count) * geometry["spacing"][0]
        assert np.max(np.abs(grid + along)) <= note.detail["tolerance_mm"]

    def test_S3_3_a_regular_stack_keeps_its_spacing(self, tmp_path: Path):
        """The median gap, where it places every slice: what 2.0.1 wrote."""
        from medh5.io.dicom import read_series, scan_dicom

        write_dicom_series(
            tmp_path / "ct", patient_id="P", study_uid="1.2.826.0.1.3680043.8.498.6",
            study_date="20240102",
        )  # fmt: skip
        report = ConversionReport()
        _, geometry = read_series(scan_dicom(tmp_path / "ct")[0], report=report)
        assert geometry["spacing"][0] == 2.5
        (note,) = report.of_kind("slice_spacing")
        assert note.detail["estimate"] == "median gap"
        assert note.detail["max_through_plane_error_mm"] == 0.0

    def test_S3_3_a_missing_slice_is_named(self, tmp_path: Path):
        from medh5.io.dicom import read_series

        along = [10.0 + 2.5 * k for k in range(8) if k != 4]
        series = self._series(tmp_path / "gap", along)
        with pytest.raises(MEDH5ValidationError, match="missing 1 slice") as exc:
            read_series(series)
        assert "is 5 mm along the slice normal, 2 times the 2.5 mm" in str(exc.value)

    def test_S3_3_a_duplicate_slice_is_named(self, tmp_path: Path):
        """Two copies of one file were reported as irregular gaps."""
        import shutil

        from medh5.io.dicom import read_series, scan_dicom

        root = tmp_path / "dup"
        written = write_dicom_series(
            root, patient_id="P", study_uid="1.2.826.0.1.3680043.8.498.8",
            study_date="20240102", shape=(5, 4, 4),
        )  # fmt: skip
        copied = sorted(written["paths"])[2]
        shutil.copy(copied, root / "copy.dcm")
        uid = str(pydicom.dcmread(copied).SOPInstanceUID)
        with pytest.raises(MEDH5ValidationError, match="coincident positions") as exc:
            read_series(scan_dicom(root)[0])
        assert f"instances {uid} and {uid}" in str(exc.value)


def _rt_dose(path: Path, like: str, frames: int = 3) -> None:
    """An RT Dose grid beside a CT: multi-frame, with a patient position."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid

    reference = pydicom.dcmread(like)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = RTDoseStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    for tag in ("PatientID", "StudyInstanceUID", "StudyDate", "FrameOfReferenceUID"):
        setattr(ds, tag, getattr(reference, tag))
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "RTDOSE"
    ds.ImagePositionPatient = list(reference.ImagePositionPatient)
    ds.ImageOrientationPatient = list(reference.ImageOrientationPatient)
    ds.PixelSpacing = [4.0, 4.0]
    ds.NumberOfFrames = frames
    ds.GridFrameOffsetVector = [-2.5 * k for k in range(frames)]
    ds.DoseGridScaling = 1e-4
    ds.DoseUnits, ds.DoseType = "GY", "PHYSICAL"
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((frames, 2, 2), np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


class TestPerSliceRescale:
    """A PET series whose slices each state their own rescale (§4.2).

    §4.2 stores one modality LUT per image, so a series whose slices disagree
    is refused --- unless ``per_slice_rescale="apply"`` turns each slice into
    its physical values, which the image then stores with no LUT at all.
    """

    SLOPES = (0.5, 0.5, 0.75, 1.25, 1.25, 2.0)
    INTERCEPTS = (0.0, 0.0, 0.0, -3.0, -3.0, 0.0)

    @pytest.fixture
    def pet(self, tmp_path: Path) -> dict[str, Any]:
        root = tmp_path / "dicom"
        series = write_dicom_series(
            root / "pt", patient_id="P1", study_uid="1.2.826.0.1.3680043.8.498.7",
            study_date="20240102", modality="PT", shape=(6, 4, 5),
        )  # fmt: skip
        # The files sort in geometric order: file k is slice k of the grid.
        for path, slope, intercept in zip(
            series["paths"], self.SLOPES, self.INTERCEPTS, strict=True
        ):
            ds = pydicom.dcmread(path)
            ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
            ds.save_as(path)
        return {"root": root, **series}

    def test_S4_2_per_slice_rescale_is_refused_naming_the_option(self, pet, tmp_path):
        from medh5.io.dicom import from_dicom, read_series, scan_dicom

        with pytest.raises(MEDH5ValidationError) as caught:
            read_series(scan_dicom(pet["root"])[0])
        message = str(caught.value)
        assert "RescaleSlope/Intercept" in message
        assert 'per_slice_rescale="apply"' in message
        assert "--per-slice-rescale apply" in message
        with pytest.raises(MEDH5ValidationError, match="--per-slice-rescale apply"):
            from_dicom(pet["root"], tmp_path / "s.medh5")
        assert not (tmp_path / "s.medh5").exists()

    def test_S4_2_per_slice_rescale_applied_stores_physical_values(self, pet, tmp_path):
        from medh5.io.dicom import from_dicom

        out = tmp_path / "s.medh5"
        report = from_dicom(pet["root"], out, per_slice_rescale="apply")
        stored = pet["volume"].astype(np.float64)
        slope = np.asarray(self.SLOPES)[:, None, None]
        intercept = np.asarray(self.INTERCEPTS)[:, None, None]
        expected = (stored * slope + intercept).astype(np.float32)
        with medh5.open(out) as sample:
            image = sample.images["PT_tp0"]
            assert not image.is_rescaled
            values = image.read()
            assert values.dtype == np.float32
            assert np.array_equal(values, expected)
            # SUV is never computed: what PET was, it stays.
            assert image.value_type == "intensity"
            assert image.value_units is None
            (activity,) = sample.document.provenance.activities_by_type("import")
            assert activity.params["per_slice_rescale"] == "applied"
        (note,) = report.of_kind("per_slice_rescale")
        assert note.severity == "decision"
        assert note.detail["series_uid"] == pet["series_uid"]
        assert note.detail["rescales"] == [
            {"slope": 0.5, "intercept": 0.0, "slices": [0, 1]},
            {"slope": 0.75, "intercept": 0.0, "slices": [2]},
            {"slope": 1.25, "intercept": -3.0, "slices": [3, 4]},
            {"slope": 2.0, "intercept": 0.0, "slices": [5]},
        ]
        (scale,) = report.of_kind("value_scale")
        assert pet["series_uid"] in scale.message
        assert report.ok
        assert validate_file(out).ok

    def test_S4_2_uniform_rescale_is_unchanged_by_apply(self, tmp_path):
        from medh5.io.dicom import from_dicom

        root = tmp_path / "dicom"
        write_dicom_series(
            root, patient_id="P1", study_uid="1.2.826.0.1.3680043.8.498.8",
            study_date="20240102", modality="PT", shape=(3, 4, 5),
        )  # fmt: skip
        refused = from_dicom(root, tmp_path / "refuse.medh5")
        applied = from_dicom(root, tmp_path / "apply.medh5", per_slice_rescale="apply")
        assert not applied.of_kind("per_slice_rescale")
        assert [str(n) for n in refused.notes] == [str(n) for n in applied.notes]
        with (
            medh5.open(tmp_path / "refuse.medh5") as one,
            medh5.open(tmp_path / "apply.medh5") as other,
        ):
            assert other.images["PT_tp0"].is_rescaled
            (activity,) = other.document.provenance.activities_by_type("import")
            assert "per_slice_rescale" not in activity.params
            assert one.content_id == other.content_id

    @pytest.mark.parametrize("mode", ["refuse", "apply"])
    def test_S4_2_slice_without_rescale_is_refused(self, pet, mode):
        from medh5.io.dicom import read_series, scan_dicom

        path = pet["paths"][2]
        ds = pydicom.dcmread(path)
        del ds.RescaleSlope
        del ds.RescaleIntercept
        ds.save_as(path)
        with pytest.raises(MEDH5ValidationError, match="RescaleSlope") as caught:
            read_series(scan_dicom(pet["root"])[0], per_slice_rescale=mode)
        assert str(ds.SOPInstanceUID) in str(caught.value)
        assert "states no RescaleSlope/Intercept" in str(caught.value)

    def test_S4_2_the_command_line_applies_a_per_slice_rescale(
        self, pet, tmp_path, capsys
    ):
        from medh5.cli import EXIT_ERROR, EXIT_OK, main

        out = tmp_path / "s.medh5"
        command = ["convert", "from-dicom", str(pet["root"]), str(out)]
        assert main(command) == EXIT_ERROR
        assert "--per-slice-rescale apply" in capsys.readouterr().err
        assert not out.exists()
        assert main([*command, "--per-slice-rescale", "apply"]) == EXIT_OK
        with medh5.open(out) as sample:
            assert sample.images["PT_tp0"].read().dtype == np.float32

    def test_per_slice_rescale_must_be_refuse_or_apply(self, pet, tmp_path):
        from medh5.io.dicom import from_dicom

        with pytest.raises(ValueError, match="per_slice_rescale"):
            from_dicom(pet["root"], tmp_path / "s.medh5", per_slice_rescale="yes")


class TestOneSeriesDoesNotStopTheTree:
    """A series the importer cannot read is left out with a warning, and the
    series beside it are imported.  An RT Dose grid, a localizer, a transfer
    syntax nothing decodes --- each aborted the whole tree, the dose grid as a
    4-D grid nobody declared."""

    @pytest.fixture
    def ct(self, tmp_path: Path) -> dict[str, Any]:
        root = tmp_path / "dicom"
        series = write_dicom_series(
            root / "ct", patient_id="P1", study_uid="1.2.826.0.1.3680043.8.498.9",
            study_date="20240102", shape=(3, 4, 4),
        )  # fmt: skip
        return {"root": root, **series}

    def _second(self, ct: dict[str, Any], *, modality: str = "MR") -> dict[str, Any]:
        return write_dicom_series(
            ct["root"] / "second", patient_id="P1",
            study_uid="1.2.826.0.1.3680043.8.498.9", study_date="20240102",
            modality=modality, shape=(3, 4, 4), seed=5,
        )  # fmt: skip

    def _only_the_ct(self, out: Path, ct: dict[str, Any]) -> None:
        with medh5.open(out) as sample:
            assert sorted(sample.images) == ["CT_tp0"]
            assert sample.timepoints["tp0"].series_uids == {
                "CT_tp0": ct["series_uid"]
            }, "the timepoint names only what was written"
            assert np.array_equal(sample.images["CT_tp0"].read(), ct["volume"])

    def test_S3_an_rt_dose_beside_its_ct_is_skipped_with_a_note(self, ct, tmp_path):
        from medh5.cli import EXIT_OK, main
        from medh5.io.dicom import from_dicom

        _rt_dose(ct["root"] / "dose.dcm", ct["paths"][0])
        out = tmp_path / "out.medh5"
        report = from_dicom(ct["root"], out)
        (note,) = report.of_kind("non_image")
        assert note.severity == "info" and "RT Dose Storage" in note.message
        assert report.ok, "the dose grid is the ordinary company of a CT"
        self._only_the_ct(out, ct)
        cli = tmp_path / "cli.medh5"
        assert main(["convert", "from-dicom", str(ct["root"]), str(cli)]) == EXIT_OK

    def test_S3_a_multi_frame_object_is_skipped_with_a_warning(self, ct, tmp_path):
        from pydicom.uid import NuclearMedicineImageStorage

        from medh5.io.dicom import from_dicom

        nm = tmp_path / "nm.dcm"
        _rt_dose(nm, ct["paths"][0])
        ds = pydicom.dcmread(nm)
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = (
            NuclearMedicineImageStorage
        )
        ds.Modality = "NM"
        ds.save_as(str(ct["root"] / "nm.dcm"), enforce_file_format=True)
        out = tmp_path / "out.medh5"
        report = from_dicom(ct["root"], out)
        (warning,) = report.of_kind("unsupported")
        assert warning.severity == "warning" and "multi-frame" in warning.message
        self._only_the_ct(out, ct)

    def test_S3_1_a_localizer_is_left_out_by_name(self, ct, tmp_path):
        """One SeriesInstanceUID in three orientations is no volume."""
        from medh5.cli import EXIT_ERROR, main
        from medh5.io.dicom import from_dicom

        localizer = self._second(ct)
        for path, orientation in zip(
            sorted(localizer["paths"])[1:3],
            ([1, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, -1]),
            strict=True,
        ):
            ds = pydicom.dcmread(path)
            ds.ImageOrientationPatient = orientation
            ds.save_as(path, enforce_file_format=True)
        out = tmp_path / "out.medh5"
        report = from_dicom(ct["root"], out)
        (warning,) = report.of_kind("series")
        assert localizer["series_uid"] in warning.message
        assert "ImageOrientationPatient" in warning.message
        assert warning.detail["series_uid"] == localizer["series_uid"]
        assert not report.ok
        self._only_the_ct(out, ct)
        # The command line writes the file, and its exit status says so.
        cli = tmp_path / "cli.medh5"
        assert main(["convert", "from-dicom", str(ct["root"]), str(cli)]) == EXIT_ERROR
        assert cli.is_file()

    def test_S3_a_transfer_syntax_nothing_decodes_is_left_out(self, ct, tmp_path):
        from pydicom.encaps import encapsulate
        from pydicom.uid import MPEG2MPML

        from medh5.io.dicom import from_dicom

        second = self._second(ct)
        for path in second["paths"]:
            ds = pydicom.dcmread(path)
            ds.file_meta.TransferSyntaxUID = MPEG2MPML
            ds.PixelData = encapsulate([ds.PixelData])
            ds["PixelData"].VR = "OB"
            ds["PixelData"].is_undefined_length = True
            ds.save_as(path, enforce_file_format=True)
        out = tmp_path / "out.medh5"
        report = from_dicom(ct["root"], out)
        (warning,) = report.of_kind("series")
        assert "MPEG2 Main Profile / Main Level" in warning.message
        assert "no installed pixel-data decoder" in warning.message
        self._only_the_ct(out, ct)

    def test_S3_a_slice_that_does_not_decode_is_refused_by_name(self, ct, tmp_path):
        """pydicom's own `ValueError` named no file."""
        from medh5.io.dicom import from_dicom, read_series, scan_dicom

        second = self._second(ct)
        broken = sorted(second["paths"])[1]
        ds = pydicom.dcmread(broken)
        ds.PixelData = ds.PixelData[:10]
        ds.save_as(broken, enforce_file_format=True)
        wanted = next(
            s for s in scan_dicom(ct["root"]) if s.series_uid == second["series_uid"]
        )
        with pytest.raises(MEDH5ValidationError, match="could not be decoded") as exc:
            read_series(wanted)
        assert str(ds.SOPInstanceUID) in str(exc.value) and broken in str(exc.value)
        out = tmp_path / "out.medh5"
        report = from_dicom(ct["root"], out)
        assert report.of_kind("series")
        self._only_the_ct(out, ct)

    def test_S3_when_no_series_imports_the_refusals_are_raised(self, ct, tmp_path):
        from medh5.io.dicom import from_dicom

        second = self._second(ct)
        for path in [*ct["paths"], *second["paths"]]:
            ds = pydicom.dcmread(path)
            ds.PixelSpacing = [0.5 + 0.1 * int(ds.InstanceNumber), 0.5]
            ds.save_as(path, enforce_file_format=True)
        out = tmp_path / "out.medh5"
        with pytest.raises(MEDH5ValidationError, match="none of the 2 series") as exc:
            from_dicom(ct["root"], out)
        assert ct["series_uid"] in str(exc.value)
        assert second["series_uid"] in str(exc.value)
        assert not out.exists()


class TestDicom:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> dict[str, Any]:
        """One patient, two studies; the first holds a CT and a co-registered PT."""
        from pydicom.uid import generate_uid

        root = tmp_path / "dicom"
        first, second = generate_uid(), generate_uid()
        ct0 = write_dicom_series(
            root / "v1" / "ct",
            patient_id="PSEUDO-001",
            study_uid=first,
            study_date="20260101",
            seed=1,
        )
        pt0 = write_dicom_series(
            root / "v1" / "pt",
            patient_id="PSEUDO-001",
            study_uid=first,
            study_date="20260101",
            modality="PT",
            frame_uid=ct0["frame_uid"],
            seed=2,
        )
        ct1 = write_dicom_series(
            root / "v2" / "ct",
            patient_id="PSEUDO-001",
            study_uid=second,
            study_date="20260401",
            seed=3,
        )
        return {"root": root, "ct0": ct0, "pt0": pt0, "ct1": ct1}

    def test_S3_3_a_single_slice_series_uses_its_declared_thickness(self, tmp_path):
        """One slice offers no increment to measure, but it does declare an extent.

        Assuming 1 mm over a declared 5 mm gives the grid a physical size the
        source never claimed, and every later resample or export inherits it.
        """
        from pydicom.uid import generate_uid

        from medh5.io.dicom import read_series, scan_dicom
        from medh5.io.report import ConversionReport

        root = tmp_path / "single"
        write_dicom_series(
            root,
            patient_id="PSEUDO-004",
            study_uid=generate_uid(),
            study_date="20260101",
            shape=(1, 16, 20),
            spacing=(2.5, 0.8, 0.9),
        )
        report = ConversionReport(converter="test")
        _, geometry = read_series(scan_dicom(root)[0], report=report)
        # the fixture writes SliceThickness as twice the increment, so 5 mm
        assert geometry["spacing"] == [5.0, 0.8, 0.9]
        assert report.of_kind("slice_spacing"), "the fallback is recorded, not silent"

    def test_scan_finds_every_series(self, tree):
        from medh5.io.dicom import scan_dicom, select_series

        series = scan_dicom(tree["root"])
        assert len(series) == 3
        assert {s.modality for s in series} == {"CT", "PT"}
        assert all(s.patient_id == "PSEUDO-001" for s in series)
        assert "6 slices" in repr(series[0])
        assert set(select_series(series)) == {"CT", "PT"}
        assert series[0].to_json()["slices"] == 6

    def test_S3_3_slices_are_ordered_by_geometry_not_InstanceNumber(self, tree):
        """The fixture numbers its instances backwards on purpose (§3.3)."""
        from medh5.io.dicom import read_series, scan_dicom

        series = next(
            s
            for s in scan_dicom(tree["root"])
            if s.series_uid == tree["ct0"]["series_uid"]
        )
        volume, geometry = read_series(series)
        assert geometry["shape"] == (6, 16, 20)
        # The slice normal is -x and the fixture steps -x with k, so ascending
        # projection recovers the writing order --- and *not* the descending
        # InstanceNumber, which would have reversed the volume.
        assert np.array_equal(volume, tree["ct0"]["volume"])
        assert not np.array_equal(volume, tree["ct0"]["volume"][::-1])

    def test_S3_2_spacing_is_measured_not_taken_from_SliceThickness(self, tree):
        from medh5.io.dicom import read_series, scan_dicom

        series = next(s for s in scan_dicom(tree["root"]) if s.modality == "CT")
        report = ConversionReport()
        _, geometry = read_series(series, report=report)
        assert np.isclose(geometry["spacing"][0], 2.5)
        note = report.of_kind("slice_spacing")[0]
        assert note.detail["thickness"] == 5.0, "the slab is twice the increment"

    def test_S4_2_a_per_slice_rescale_is_refused_not_taken_from_slice_zero(self, tree):
        """§4.2 stores one modality LUT for the series, so there has to be one.

        A PET series with a per-slice rescale is ordinary, and collapsing it to
        slice 0's slope reports the wrong activity on every other slice with
        nothing in the file to say so.
        """
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[3]
        ds = pydicom.dcmread(str(target))
        ds.RescaleSlope, ds.RescaleIntercept = 2.0, 0.0
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5ValidationError, match="RescaleSlope"):
            read_series(series)

    def test_S3_1_a_slice_rotated_from_the_rest_is_refused(self, tree):
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[3]
        ds = pydicom.dcmread(str(target))
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5ValidationError, match="ImageOrientationPatient"):
            read_series(series)

    def test_S3_2_a_slice_with_its_own_pixel_spacing_is_refused(self, tree):
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[2]
        ds = pydicom.dcmread(str(target))
        ds.PixelSpacing = [1.5, 1.5]
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5ValidationError, match="PixelSpacing"):
            read_series(series)

    def test_a_slice_missing_a_tag_refuses_rather_than_raising_raw(self, tree):
        """`scan_dicom` groups by series without requiring these tags.

        A slice can therefore reach the agreement check missing one outright,
        and the raw `AttributeError` that produced was both a CLI traceback and
        an exception no caller could catch as `MEDH5Error` like every other
        input problem this converter reports.
        """
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[3]
        ds = pydicom.dcmread(str(target))
        del ds.ImageOrientationPatient
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5Error, match="no usable ImageOrientationPatient"):
            read_series(series)

    def test_a_converter_refusal_does_not_borrow_a_format_code(self, tree):
        """§15.2's codes describe conditions in a MEDH5 file.

        A DICOM series is not one yet, and no code in the table means "these
        slices disagree" --- borrowing E204 would have reported a modality-LUT
        problem as malformed `channel_names`.
        """
        import pydicom

        from medh5.errors import MEDH5ValidationError
        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[3]
        ds = pydicom.dcmread(str(target))
        ds.RescaleSlope = 2.0
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5ValidationError) as caught:
            read_series(series)
        assert caught.value.code is None

    def test_S3_2_a_series_with_no_pixel_spacing_is_refused_not_assumed(self, tree):
        """Every slice omitting `PixelSpacing` used to be the dangerous case.

        One slice omitting it disagrees with the others and was caught; *all*
        of them omitting it meant they agreed on the 1 mm default, so the stack
        was written with an in-plane size the source never stated.
        """
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        for path in sorted((tree["root"] / "v1" / "ct").glob("*.dcm")):
            ds = pydicom.dcmread(str(path))
            del ds.PixelSpacing
            ds.save_as(str(path))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5Error, match="no usable PixelSpacing"):
            read_series(series)

    @pytest.mark.parametrize(
        ("tag", "value"),
        [
            ("ImageOrientationPatient", [0, 0, 1, 0, 1]),
            ("ImageOrientationPatient", [0, 0, 1, 0, 1, 0, 5]),
            ("PixelSpacing", [0.8]),
            ("PixelSpacing", [0.8, 0.9, 1.0]),
        ],
    )
    def test_a_tag_of_the_wrong_length_is_refused(self, tree, tag, value):
        """Cardinality, not just parseability.

        These extract perfectly well, and every slice carries the same wrong
        length -- so the agreement check sees nothing to disagree about. What
        followed was either a raw exception outside `MEDH5Error` (five values
        reach `np.cross`) or, for a three-value `PixelSpacing`, silence: it was
        read as its first two elements and the third discarded, giving the grid
        an in-plane size nobody wrote down.
        """
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        for path in sorted((tree["root"] / "v1" / "ct").glob("*.dcm")):
            ds = pydicom.dcmread(str(path))
            setattr(ds, tag, value)
            ds.save_as(str(path))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        # A single-value tag arrives from pydicom as a scalar rather than a
        # one-element sequence, so it is refused by the extraction guard rather
        # than the length check. Either way it names the tag and is a
        # `MEDH5Error`, which is the contract under test.
        with pytest.raises(MEDH5Error, match=tag):
            read_series(series)

    def test_an_irregular_stack_is_refused(self, tmp_path):
        import pydicom
        from pydicom.uid import generate_uid

        from medh5.io.dicom import read_series, scan_dicom

        root = tmp_path / "bad"
        write_dicom_series(
            root, patient_id="p", study_uid=generate_uid(), study_date="20260101"
        )
        victim = sorted(root.glob("*.dcm"))[2]
        dataset = pydicom.dcmread(str(victim))
        position = list(dataset.ImagePositionPatient)
        position[0] = float(position[0]) + 1.7
        dataset.ImagePositionPatient = position
        dataset.save_as(str(victim), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="irregular slice gaps"):
            read_series(scan_dicom(root)[0])

    def test_S3_7_studies_of_one_patient_become_one_longitudinal_sample(
        self, tree, tmp_path
    ):
        from medh5.io.dicom import from_dicom

        report = from_dicom(tree["root"], tmp_path / "subject.medh5")
        with medh5.open(tmp_path / "subject.medh5") as sample:
            assert sample.identity.subject_id == "PSEUDO-001"
            assert sample.timepoints.ids == ("tp0", "tp1")
            assert sample.timepoints["tp1"].days_from_baseline == 90
            assert sorted(sample.images) == ["CT_tp0", "CT_tp1", "PT_tp0"]
            assert sample.grids["pt_tp0"].frame_uid == sample.grids["ct_tp0"].frame_uid
            assert sample.grids["ct_tp1"].timepoint == "tp1"
        assert report.of_kind("instance_ids")

    def test_S4_2_the_rescale_is_stored_not_applied(self, tree, tmp_path):
        from medh5.io.dicom import from_dicom

        from_dicom(tree["root"], tmp_path / "s.medh5")
        with medh5.open(tmp_path / "s.medh5") as sample:
            image = sample.images["CT_tp0"]
            assert image.is_rescaled
            stored = image.read()
            assert np.allclose(image.read(physical=True), stored - 1024.0)

    def test_S11_4_only_named_acquisition_tags_are_copied(self, tree, tmp_path):
        from medh5.io.dicom import from_dicom

        from_dicom(tree["root"], tmp_path / "s.medh5")
        with medh5.open(tmp_path / "s.medh5") as sample:
            acquisition = sample.document.acquisition["CT_tp0"]
            assert acquisition["ConvolutionKernel"] == "B30f"
            assert "PatientName" not in acquisition
            assert "PatientID" not in acquisition

    def test_S11_4_a_converted_file_says_it_is_not_de_identified(
        self, tree, tmp_path, capsys
    ):
        """As information, which the validator repeats (W903): a warning fails
        a conversion, and this one was on every conversion, so `medh5 convert
        from-dicom` exited 1 on every success."""
        from medh5.cli import EXIT_OK, main
        from medh5.io.dicom import from_dicom

        report = from_dicom(tree["root"], tmp_path / "s.medh5")
        (note,) = report.of_kind("deidentification")
        assert note.severity == "info" and report.ok
        assert "W903" in validate_file(tmp_path / "s.medh5").codes
        out = tmp_path / "cli.medh5"
        assert main(["convert", "from-dicom", str(tree["root"]), str(out)]) == EXIT_OK
        assert "de-identif" in capsys.readouterr().out and out.is_file()

    @pytest.mark.parametrize("how", ["separator", "existing"])
    def test_an_out_directory_holds_one_sample_too(self, tree, tmp_path, how):
        """`out/` names a directory whatever the number of samples: with one
        subject in the tree it was written as a file named `out`, and an
        existing directory ended in an `OSError` traceback."""
        from medh5.cli import EXIT_OK, main

        folder = tmp_path / "out"
        if how == "existing":
            folder.mkdir()
        named = str(folder) + (os.sep if how == "separator" else "")
        assert main(["convert", "from-dicom", str(tree["root"]), named]) == EXIT_OK
        (written,) = folder.glob("*.medh5")
        with medh5.open(written) as sample:
            assert sample.identity.subject_id == "PSEUDO-001"

    def test_S3_3_slices_of_two_matrix_sizes_are_refused_by_name(self, tree):
        """They reached `np.stack`, whose `ValueError` named nothing."""
        import pydicom

        from medh5.io.dicom import read_series, scan_dicom

        target = sorted((tree["root"] / "v1" / "ct").glob("*.dcm"))[3]
        ds = pydicom.dcmread(str(target))
        ds.Rows, ds.Columns = 8, 10
        ds.PixelData = np.zeros((8, 10), np.uint16).tobytes()
        ds.save_as(str(target))
        wanted = tree["ct0"]["series_uid"]
        series = next(s for s in scan_dicom(tree["root"]) if s.series_uid == wanted)
        with pytest.raises(MEDH5ValidationError, match="Rows/Columns") as caught:
            read_series(series)
        assert caught.value.code is None

    def test_group_by_study_keeps_the_visits_apart(self, tree, tmp_path):
        from medh5.io.dicom import from_dicom

        from_dicom(tree["root"], tmp_path / "out", group_by="study")
        written = sorted((tmp_path / "out").glob("*.medh5"))
        assert len(written) == 2
        with medh5.open(written[0]) as sample:
            assert sample.timepoints.ids == ("tp0",)

    def test_selecting_by_modality_and_series(self, tree, tmp_path):
        from medh5.io.dicom import from_dicom

        from_dicom(tree["root"], tmp_path / "ct.medh5", modalities=["CT"])
        with medh5.open(tmp_path / "ct.medh5") as sample:
            assert sorted(sample.images) == ["CT_tp0", "CT_tp1"]
        from_dicom(
            tree["root"],
            tmp_path / "one.medh5",
            series_uids=[tree["ct0"]["series_uid"]],
        )
        with medh5.open(tmp_path / "one.medh5") as sample:
            assert len(sample.images) == 1

    def test_an_empty_tree_is_refused(self, tmp_path):
        from medh5.io.dicom import from_dicom

        (tmp_path / "empty").mkdir()
        with pytest.raises(MEDH5ValidationError, match="no DICOM"):
            from_dicom(tmp_path / "empty", tmp_path / "x.medh5")

    def test_dicom_a_broken_enhanced_object_and_monochrome1_are_reported(
        self, tmp_path: Path
    ):
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        from medh5.io.dicom import from_dicom, read_series, scan_dicom
        from medh5.io.report import ConversionReport

        root = tmp_path / "dicom"
        study = generate_uid()
        series = write_dicom_series(
            root / "ct", patient_id="P", study_uid=study, study_date="20260101"
        )
        for file in series["paths"]:
            ds = pydicom.dcmread(file)
            ds.PhotometricInterpretation = "MONOCHROME1"
            ds.save_as(file, enforce_file_format=True)
        enhanced = Dataset()
        enhanced.file_meta = FileMetaDataset()
        enhanced.file_meta.MediaStorageSOPClassUID = pydicom.uid.EnhancedMRImageStorage
        enhanced.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        enhanced.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        enhanced.SOPClassUID = pydicom.uid.EnhancedMRImageStorage
        enhanced.SOPInstanceUID = enhanced.file_meta.MediaStorageSOPInstanceUID
        enhanced.Modality = "MR"
        enhanced.PatientID = "P"
        enhanced.StudyInstanceUID = study
        enhanced.SeriesInstanceUID = generate_uid()
        enhanced.NumberOfFrames = 4
        (root / "mr").mkdir()
        enhanced.save_as(str(root / "mr" / "enh.dcm"), enforce_file_format=True)

        report = ConversionReport(converter="test")
        found = scan_dicom(root, report=report)
        assert sorted(s.modality for s in found) == ["CT", "MR"]
        assert not report.of_kind("unsupported")

        report = ConversionReport(converter="test")
        read_series(next(s for s in found if s.modality == "CT"), report=report)
        assert report.of_kind("photometric")

        report = from_dicom(root, tmp_path / "out.medh5")
        assert not report.ok  # both are warnings, and warnings fail the command
        # The enhanced object states no frame's position: left out, saying so.
        (left_out,) = report.of_kind("series")
        assert "MR series" in left_out.message
        assert "frame 1 of instance" in left_out.message
        assert "ImagePositionPatient" in left_out.message
        with medh5.open(tmp_path / "out.medh5") as sample:
            acquisition = sample.document.acquisition["CT_tp0"]
            assert acquisition["PhotometricInterpretation"] == "MONOCHROME1"


_LEGACY = {
    "CT": ("LegacyConvertedEnhancedCTImage", "CTImageStorage"),
    "MR": ("LegacyConvertedEnhancedMRImage", "MRImageStorage"),
    "PT": ("LegacyConvertedEnhancedPETImage", "PositronEmissionTomographyImageStorage"),
}


"""The highdicom class converting a classic series of each modality, and the
storage class its slices must declare for it."""


def _enhanced(paths: list[str], modality: str = "CT") -> Any:
    """The classic slices at *paths* as one legacy-converted enhanced object."""
    hd = pytest.importorskip("highdicom")
    import pydicom.uid
    from pydicom.uid import generate_uid

    converter, storage = _LEGACY[modality]
    slices = []
    for path in paths:
        ds = pydicom.dcmread(path)
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = getattr(
            pydicom.uid, storage
        )
        # What highdicom 0.23 (the floor) reads off every slice, where later
        # versions default it.
        ds.setdefault("ReferringPhysicianName", "")
        ds.setdefault("ImageType", ["ORIGINAL", "PRIMARY", "AXIAL"])
        slices.append(ds)
    return getattr(hd.legacy, converter)(
        slices,
        series_instance_uid=generate_uid(),
        series_number=2,
        sop_instance_uid=generate_uid(),
        instance_number=1,
    )


def _save(dataset: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), enforce_file_format=True)
    return path


def _frame(dataset: Any, index: int) -> Any:
    """Frame *index*'s own functional groups."""
    return dataset.PerFrameFunctionalGroupsSequence[index]


def _concatenation(
    dataset: Any, directory: Path, split: int | tuple[int, ...]
) -> list[Path]:
    """*dataset* as a concatenation of instances split before each frame in
    *split* --- ``4`` makes frames ``[:4]`` and ``[4:]`` --- the last part
    written first, so file order is not the concatenation's."""
    from copy import deepcopy

    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid

    frames = int(dataset.NumberOfFrames)
    size = int(dataset.Rows) * int(dataset.Columns) * int(dataset.BitsAllocated) // 8
    concatenation = generate_uid()
    written = []
    edges = [0, *((split,) if isinstance(split, int) else split), frames]
    bounds = list(zip(edges[:-1], edges[1:], strict=True))
    for number, (start, stop) in enumerate(bounds, start=1):
        part = deepcopy(dataset)
        part.SOPInstanceUID = part.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        part.NumberOfFrames = stop - start
        part.PerFrameFunctionalGroupsSequence = Sequence(
            list(dataset.PerFrameFunctionalGroupsSequence)[start:stop]
        )
        part.PixelData = dataset.PixelData[start * size : stop * size]
        part.ConcatenationUID = concatenation
        part.SOPInstanceUIDOfConcatenationSource = dataset.SOPInstanceUID
        part.InConcatenationNumber = number
        part.InConcatenationTotalNumber = len(bounds)
        part.ConcatenationFrameOffsetNumber = start
        written.append(_save(part, directory / f"part{len(bounds) + 1 - number}.dcm"))
    return written


class TestEnhancedMultiFrame:
    """Enhanced and legacy-converted multi-frame objects (PS3.3 C.7.6.16).

    One object carries every slice as a frame, and each frame's position,
    orientation, spacing and rescale in its functional groups --- the frame's
    own or the shared ones.  A stack of frames imports as the classic series
    it was converted from would: the same grid, the same values.
    """

    @staticmethod
    def _classic(tmp_path: Path, modality: str = "CT", **options: Any) -> Any:
        from pydicom.uid import generate_uid

        return write_dicom_series(
            tmp_path / "classic" / modality.lower(),
            patient_id="P1",
            study_uid=generate_uid(),
            study_date="20260101",
            modality=modality,
            **options,
        )

    @staticmethod
    def _imported(root: Path, out: Path, **options: Any) -> Path:
        from medh5.io.dicom import from_dicom

        report = from_dicom(root, out, **options)
        assert not report.of_kind("series"), report.of_kind("series")
        return out

    @staticmethod
    def _same(left: Path, right: Path, image: str, grid: str) -> None:
        with medh5.open(left) as a, medh5.open(right) as b:
            ga, gb = a.grids[grid], b.grids[grid]
            assert ga.shape == gb.shape
            assert np.allclose(ga.spacing, gb.spacing)
            assert np.allclose(ga.origin, gb.origin)
            assert np.allclose(ga.direction, gb.direction)
            assert ga.frame_uid == gb.frame_uid
            ia, ib = a.images[image], b.images[image]
            assert ia.digest == ib.digest
            assert ia.rescale == ib.rescale
            assert np.array_equal(ia.read(), ib.read())
            assert a.document.acquisition[image] == b.document.acquisition[image]

    def test_enhanced_ct_imports_as_the_classic_series(self, tmp_path):
        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        left = self._imported(tmp_path / "classic", tmp_path / "classic.medh5")
        right = self._imported(tmp_path / "enhanced", tmp_path / "enhanced.medh5")
        self._same(left, right, "CT_tp0", "ct_tp0")
        with medh5.open(right) as sample:
            assert sample.timepoints["tp0"].series_uids == {
                "CT_tp0": str(enhanced.SeriesInstanceUID)
            }
        assert validate_file(right).ok

    @pytest.mark.parametrize("modality", ["MR", "PT"])
    def test_enhanced_mr_and_pet_import_as_their_classic_series(
        self, tmp_path, modality
    ):
        classic = self._classic(tmp_path, modality)
        # The classic MR and PET image modules define no rescale, and older
        # highdicom drops it on conversion; the fixture writes one for all.
        for path in classic["paths"]:
            ds = pydicom.dcmread(path)
            del ds.RescaleSlope, ds.RescaleIntercept
            ds.save_as(path)
        _save(_enhanced(classic["paths"], modality), tmp_path / "enhanced" / "x.dcm")
        left = self._imported(tmp_path / "classic", tmp_path / "classic.medh5")
        right = self._imported(tmp_path / "enhanced", tmp_path / "enhanced.medh5")
        self._same(left, right, f"{modality}_tp0", f"{modality.lower()}_tp0")

    def test_scan_finds_an_enhanced_object_as_one_series(self, tmp_path):
        from medh5.io.dicom import scan_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        path = _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        report = ConversionReport(converter="test")
        (series,) = scan_dicom(tmp_path / "enhanced", report=report)
        assert not report.of_kind("unsupported")
        assert series.series_uid == str(enhanced.SeriesInstanceUID)
        assert series.paths == [str(path)]
        assert series.n_slices == 6 and series.stack is None
        assert sorted(index for _, index in series.frames) == list(range(6))

    def test_enhanced_two_stacks_are_two_series(self, tmp_path):
        from medh5.io.dicom import from_dicom, scan_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        positions = []
        for index in range(6):
            content = _frame(enhanced, index).FrameContentSequence[0]
            content.StackID = "1" if index < 3 else "2"
            content.InStackPositionNumber = index % 3 + 1
            plane = _frame(enhanced, index).PlanePositionSequence[0]
            positions.append(float(plane.ImagePositionPatient[0]))
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        stacks = scan_dicom(tmp_path / "enhanced")
        assert [s.stack for s in stacks] == ["1", "2"]
        assert [s.n_slices for s in stacks] == [3, 3]
        report = from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")
        assert not report.of_kind("series")
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["CT_1_tp0", "CT_2_tp0"]
            first, second = sample.grids["ct_1_tp0"], sample.grids["ct_2_tp0"]
            assert first.shape == second.shape == (3, 16, 20)
            # Each stack is placed by its own frames. The slice normal of
            # this orientation is -x, so a stack starts at its largest x.
            assert first.origin[0] == max(positions[:3])
            assert second.origin[0] == max(positions[3:])
            volume = np.stack(
                [pydicom.dcmread(p).pixel_array for p in classic["paths"]]
            )
            both = np.concatenate(
                [sample.images["CT_1_tp0"].read(), sample.images["CT_2_tp0"].read()]
            )
            # Classic slice k lies at origin - k * spacing: stacked by position,
            # the slices run the other way.
            assert np.array_equal(np.sort(both, axis=None), np.sort(volume, axis=None))

    def test_an_irregular_stack_is_refused(self, tmp_path):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        plane = _frame(enhanced, 2).PlanePositionSequence[0]
        position = [float(v) for v in plane.ImagePositionPatient]
        position[0] += 1.1
        plane.ImagePositionPatient = position
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(MEDH5ValidationError, match="irregular slice gaps"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_frames_out_of_their_stack_order_are_refused(self, tmp_path):
        """``InStackPositionNumber`` says where a frame sits in its stack; a
        number its position contradicts is refused, not re-sorted by guess."""
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        for index in range(6):  # the frames are in geometric order
            _frame(enhanced, index).FrameContentSequence[0].InStackPositionNumber = (
                index + 1
            )
        first = _frame(enhanced, 0).FrameContentSequence[0]
        third = _frame(enhanced, 2).FrameContentSequence[0]
        first.InStackPositionNumber, third.InStackPositionNumber = (
            third.InStackPositionNumber,
            first.InStackPositionNumber,
        )
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(MEDH5ValidationError, match="InStackPositionNumber"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_a_stack_whose_numbers_skip_a_frame_is_refused(self, tmp_path):
        """``InStackPositionNumber`` counts a stack's frames from 1 (C.7.6.16.2.2.4);
        a gap says a plane is missing, though the positions left are regular."""
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        for index, number in enumerate([1, 2, 3, 4, 5, 7]):
            _frame(enhanced, index).FrameContentSequence[
                0
            ].InStackPositionNumber = number
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(MEDH5ValidationError, match=r"InStackPositionNumber.*\[6\]"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_a_frame_with_no_position_is_refused_with_the_reason(self, tmp_path):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        del _frame(enhanced, 4).PlanePositionSequence
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(
            MEDH5ValidationError, match="frame 5 .*ImagePositionPatient"
        ):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_frames_are_decoded_one_at_a_time(self, tmp_path, monkeypatch):
        """No frame is decoded with the rest: an enhanced object of thousands
        of frames is held once, as the volume, not twice."""
        from pydicom.dataset import Dataset

        classic = self._classic(tmp_path)
        _save(_enhanced(classic["paths"]), tmp_path / "enhanced" / "ct.dcm")

        def whole(self: Any) -> Any:
            raise AssertionError("the whole pixel array was decoded")

        monkeypatch.setattr(Dataset, "pixel_array", property(whole))
        out = self._imported(tmp_path / "enhanced", tmp_path / "out.medh5")
        with medh5.open(out) as sample:
            assert sample.images["CT_tp0"].shape == (6, 16, 20)

    def test_enhanced_per_frame_rescale_follows_per_slice_rescale(self, tmp_path):
        """Frames stating their own rescale are a series of slices that do:
        refused, unless ``per_slice_rescale="apply"``, which stores each
        frame's physical values as the classic import does."""
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path, "CT", shape=(6, 4, 5))
        for k, path in enumerate(classic["paths"]):
            ds = pydicom.dcmread(path)
            ds.RescaleSlope = (0.5, 0.5, 0.75, 1.25, 1.25, 2.0)[k]
            ds.RescaleIntercept = (0.0, 0.0, 0.0, -3.0, -3.0, 0.0)[k]
            ds.save_as(path)
        _save(_enhanced(classic["paths"]), tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(MEDH5ValidationError, match="per_slice_rescale"):
            from_dicom(tmp_path / "enhanced", tmp_path / "refused.medh5")
        left = self._imported(
            tmp_path / "classic", tmp_path / "classic.medh5", per_slice_rescale="apply"
        )
        right = self._imported(
            tmp_path / "enhanced",
            tmp_path / "enhanced.medh5",
            per_slice_rescale="apply",
        )
        self._same(left, right, "CT_tp0", "ct_tp0")
        with medh5.open(right) as sample:
            assert not sample.images["CT_tp0"].is_rescaled

    @pytest.mark.parametrize(
        ("rescale_type", "value_type", "units"),
        [("MGML", "quantitative", "MGML"), ("US", "intensity", None)],
    )
    def test_a_ct_takes_its_units_from_its_rescale_type(
        self, tmp_path, rescale_type, value_type, units
    ):
        """A CT's rescale gives HU unless ``RescaleType`` says otherwise
        (C.8.15.3.10): a material map in mg/ml is not HU, and ``US`` is no unit."""
        classic = self._classic(tmp_path)
        for path in classic["paths"]:
            ds = pydicom.dcmread(path)
            ds.RescaleType = rescale_type
            ds.save_as(path)
        enhanced = _enhanced(classic["paths"])
        # highdicom writes HU whatever the slices say.
        shared = enhanced.SharedFunctionalGroupsSequence[0]
        shared.PixelValueTransformationSequence[0].RescaleType = rescale_type
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        for name in ("classic", "enhanced"):
            out = self._imported(tmp_path / name, tmp_path / f"{name}.medh5")
            with medh5.open(out) as sample:
                image = sample.images["CT_tp0"]
                assert (image.value_type, image.value_units) == (value_type, units)

    def test_frames_of_different_rescale_types_are_refused(self, tmp_path):
        from copy import deepcopy

        from pydicom.sequence import Sequence

        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        shared = enhanced.SharedFunctionalGroupsSequence[0]
        shared.PixelValueTransformationSequence[0].RescaleType = "MGML"
        item = deepcopy(shared.PixelValueTransformationSequence[0])
        item.RescaleType = "HU"
        _frame(enhanced, 2).PixelValueTransformationSequence = Sequence([item])
        _save(enhanced, tmp_path / "enhanced" / "ct.dcm")
        with pytest.raises(MEDH5ValidationError, match="RescaleType"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_enhanced_acquisition_attributes_come_from_the_functional_groups(
        self, tmp_path
    ):
        """An enhanced MR states its echo time per frame, as
        ``EffectiveEchoTime``, and its repetition time in the shared groups."""
        from pydicom.dataset import Dataset

        classic = self._classic(tmp_path, "MR")
        enhanced = _enhanced(classic["paths"], "MR")
        timing = Dataset()
        timing.RepetitionTime = 2200.0
        timing.FlipAngle = 9.0
        enhanced.SharedFunctionalGroupsSequence[
            0
        ].MRTimingAndRelatedParametersSequence = [timing]
        for index in range(6):
            echo = Dataset()
            echo.EffectiveEchoTime = 2.5
            _frame(enhanced, index).MREchoSequence = [echo]
        _save(enhanced, tmp_path / "enhanced" / "mr.dcm")
        out = self._imported(tmp_path / "enhanced", tmp_path / "out.medh5")
        with medh5.open(out) as sample:
            acquisition = sample.document.acquisition["MR_tp0"]
        assert acquisition["RepetitionTime"] == 2200.0
        assert acquisition["FlipAngle"] == 9.0
        assert acquisition["EchoTime"] == 2.5

    def test_a_concatenation_is_one_object(self, tmp_path):
        from medh5.io.dicom import scan_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        parts = _concatenation(enhanced, tmp_path / "enhanced", split=4)
        (series,) = scan_dicom(tmp_path / "enhanced")
        assert series.n_slices == 6
        assert sorted(series.paths) == sorted(str(p) for p in parts)
        left = self._imported(tmp_path / "classic", tmp_path / "classic.medh5")
        right = self._imported(tmp_path / "enhanced", tmp_path / "enhanced.medh5")
        self._same(left, right, "CT_tp0", "ct_tp0")

    def test_a_concatenation_missing_a_part_is_refused(self, tmp_path):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        first, second = _concatenation(enhanced, tmp_path / "enhanced", split=4)
        second.unlink()
        with pytest.raises(MEDH5ValidationError, match="concatenation .* 1 of its 2"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_a_concatenation_split_between_its_stacks_imports_each(self, tmp_path):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        for index in range(6):
            content = _frame(enhanced, index).FrameContentSequence[0]
            content.StackID = "1" if index < 3 else "2"
            content.InStackPositionNumber = index % 3 + 1
        # One part holds the first stack's frames, the other the second's:
        # each stack is complete, and so is the object.
        _concatenation(enhanced, tmp_path / "enhanced", split=3)
        from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["CT_1_tp0", "CT_2_tp0"]

    def test_a_concatenation_with_no_total_is_held_to_numbers_from_one(self, tmp_path):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        parts = _concatenation(enhanced, tmp_path / "enhanced", split=4)
        for path in parts:
            part = pydicom.dcmread(path)
            del part.InConcatenationTotalNumber  # optional (Type 3)
            part.save_as(path)
        from_dicom(tmp_path / "enhanced", tmp_path / "whole.medh5")
        parts[0].unlink()
        with pytest.raises(MEDH5ValidationError, match=r"numbered \[2\]"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    @pytest.mark.parametrize(
        ("totals", "match"),
        [((None, 3), "2 of its 3"), ((2, 3), "states its total as")],
    )
    def test_a_concatenation_total_is_read_from_every_part(
        self, tmp_path, totals, match
    ):
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        parts = _concatenation(enhanced, tmp_path / "enhanced", split=4)
        for path, total in zip(parts, totals, strict=True):
            part = pydicom.dcmread(path)
            if total is None:
                del part.InConcatenationTotalNumber
            else:
                part.InConcatenationTotalNumber = total
            part.save_as(path)
        with pytest.raises(MEDH5ValidationError, match=match):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_a_concatenation_whose_offsets_skip_a_frame_is_refused(self, tmp_path):
        """``ConcatenationFrameOffsetNumber`` places each part's first frame in
        the object: the second part of a 4 + 2 split starts at 4, and one
        starting at 5 says a frame between them is missing."""
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        _, second = _concatenation(enhanced, tmp_path / "enhanced", split=4)
        part = pydicom.dcmread(second)
        part.ConcatenationFrameOffsetNumber = 5
        part.save_as(second)
        with pytest.raises(MEDH5ValidationError, match="frame offset"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_a_part_stating_no_offset_leaves_the_others_checked(self, tmp_path):
        """Offsets 0, none and 5 over three parts of two frames: the third part
        starts at 4 whatever the second states, so 5 is refused."""
        from medh5.io.dicom import from_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        _, second, third = _concatenation(enhanced, tmp_path / "enhanced", (2, 4))
        part = pydicom.dcmread(second)
        del part.ConcatenationFrameOffsetNumber
        part.save_as(second)
        part = pydicom.dcmread(third)
        part.ConcatenationFrameOffsetNumber = 5
        part.save_as(third)
        with pytest.raises(MEDH5ValidationError, match="frame offset"):
            from_dicom(tmp_path / "enhanced", tmp_path / "out.medh5")

    def test_an_enhanced_ultrasound_volume_is_still_skipped(self, tmp_path):
        from pydicom.uid import EnhancedUSVolumeStorage

        from medh5.io.dicom import scan_dicom

        classic = self._classic(tmp_path)
        enhanced = _enhanced(classic["paths"])
        enhanced.SOPClassUID = enhanced.file_meta.MediaStorageSOPClassUID = (
            EnhancedUSVolumeStorage
        )
        _save(enhanced, tmp_path / "enhanced" / "us.dcm")
        report = ConversionReport(converter="test")
        assert scan_dicom(tmp_path / "enhanced", report=report) == []
        (warning,) = report.of_kind("unsupported")
        assert "Enhanced US Volume" in warning.message
