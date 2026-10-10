"""DICOM series import (``medh5.io.dicom``)."""

from __future__ import annotations

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


class TestF06DicomSameModality:
    """A study with several series of one modality imports, named by series."""

    def test_two_series_of_one_modality_import_side_by_side(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("pydicom")
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
        pytest.importorskip("pydicom")
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


class TestW18Converters:
    """F-19, F-25."""

    @pytest.fixture(autouse=True)
    def _pydicom(self) -> None:
        pytest.importorskip("pydicom")

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

    def test_a_converted_file_says_it_is_not_de_identified(self, tree, tmp_path):
        from medh5.io.dicom import from_dicom

        report = from_dicom(tree["root"], tmp_path / "s.medh5")
        assert any(w.kind == "deidentification" for w in report.warnings)
        assert "W903" in validate_file(tmp_path / "s.medh5").codes

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

    def test_dicom_enhanced_objects_and_monochrome1_are_reported(self, tmp_path: Path):
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        from medh5.io.dicom import from_dicom, read_series, scan_dicom
        from medh5.io.report import ConversionReport

        root = tmp_path / "dicom"
        series = write_dicom_series(
            root / "ct", patient_id="P", study_uid=generate_uid(), study_date="20260101"
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
        enhanced.SeriesInstanceUID = generate_uid()
        enhanced.NumberOfFrames = 4
        (root / "mr").mkdir()
        enhanced.save_as(str(root / "mr" / "enh.dcm"), enforce_file_format=True)

        report = ConversionReport(converter="test")
        found = scan_dicom(root, report=report)
        assert [s.modality for s in found] == ["CT"]
        unsupported = report.of_kind("unsupported")
        assert unsupported and unsupported[0].severity == "warning"
        assert "MR" not in [s.modality for s in found]

        report = ConversionReport(converter="test")
        read_series(found[0], report=report)
        assert report.of_kind("photometric")

        report = from_dicom(root, tmp_path / "out.medh5")
        assert not report.ok  # both are warnings, and warnings fail the command
        with medh5.open(tmp_path / "out.medh5") as sample:
            acquisition = sample.document.acquisition["CT_tp0"]
            assert acquisition["PhotometricInterpretation"] == "MONOCHROME1"
