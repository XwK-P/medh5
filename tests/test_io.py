"""Tests for medh5.io NIfTI and DICOM converters."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------


def _write_nifti(path, data, affine):
    img = nib.Nifti1Image(np.asarray(data), affine)
    nib.save(img, str(path))


def _make_affine(spacing, origin):
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = spacing[0]
    aff[1, 1] = spacing[1]
    aff[2, 2] = spacing[2]
    aff[:3, 3] = origin
    return aff


# ---------------------------------------------------------------------------
# NIfTI -> medh5
# ---------------------------------------------------------------------------


class TestFromNifti:
    def test_single_modality_roundtrip(self, tmp_path):
        from medh5.io import from_nifti

        ct = (np.arange(8 * 16 * 16) % 200).reshape(8, 16, 16).astype(np.int16)
        aff = _make_affine([1.5, 0.75, 0.75], [10.0, -20.0, 5.0])
        ct_path = tmp_path / "ct.nii.gz"
        _write_nifti(ct_path, ct, aff)

        out = tmp_path / "sample.medh5"
        from_nifti(images={"CT": ct_path}, out_path=out, label=1)

        sample = MEDH5File.read(out)
        assert sample.images["CT"].shape == ct.shape
        np.testing.assert_array_equal(sample.images["CT"], ct)
        np.testing.assert_allclose(sample.meta.spatial.spacing, [1.5, 0.75, 0.75])
        np.testing.assert_allclose(sample.meta.spatial.origin, [10.0, -20.0, 5.0])
        assert sample.meta.spatial.coord_system == "RAS"
        assert sample.meta.label == 1

    def test_multi_modality_with_seg(self, tmp_path):
        from medh5.io import from_nifti

        rng = np.random.default_rng(0)
        ct = rng.standard_normal((6, 12, 12)).astype(np.float32)
        pet = rng.standard_normal((6, 12, 12)).astype(np.float32)
        tumor = (rng.random((6, 12, 12)) > 0.7).astype(np.uint8)
        aff = _make_affine([2.0, 1.0, 1.0], [0.0, 0.0, 0.0])

        ct_p = tmp_path / "ct.nii.gz"
        pet_p = tmp_path / "pet.nii.gz"
        tumor_p = tmp_path / "tumor.nii.gz"
        _write_nifti(ct_p, ct, aff)
        _write_nifti(pet_p, pet, aff)
        _write_nifti(tumor_p, tumor, aff)

        out = tmp_path / "multi.medh5"
        from_nifti(
            images={"CT": ct_p, "PET": pet_p},
            seg={"tumor": tumor_p},
            out_path=out,
        )

        sample = MEDH5File.read(out)
        assert set(sample.images.keys()) == {"CT", "PET"}
        assert sample.seg is not None
        assert "tumor" in sample.seg
        assert sample.seg["tumor"].dtype == bool
        np.testing.assert_array_equal(sample.seg["tumor"], tumor.astype(bool))

    def test_grid_mismatch_raises(self, tmp_path):
        from medh5.io import from_nifti

        ct = np.zeros((4, 8, 8), dtype=np.float32)
        pet = np.zeros((4, 8, 9), dtype=np.float32)  # mismatched
        aff = np.eye(4)
        ct_p = tmp_path / "ct.nii.gz"
        pet_p = tmp_path / "pet.nii.gz"
        _write_nifti(ct_p, ct, aff)
        _write_nifti(pet_p, pet, aff)

        with pytest.raises(MEDH5ValidationError, match="shape mismatch"):
            from_nifti(images={"CT": ct_p, "PET": pet_p}, out_path=tmp_path / "x.medh5")

    def test_empty_images_raises(self, tmp_path):
        from medh5.io import from_nifti

        with pytest.raises(MEDH5ValidationError, match="at least one"):
            from_nifti(images={}, out_path=tmp_path / "x.medh5")


# ---------------------------------------------------------------------------
# medh5 -> NIfTI
# ---------------------------------------------------------------------------


class TestToNifti:
    def test_image_and_seg_roundtrip(self, tmp_path):
        from medh5.io import from_nifti, to_nifti

        ct = (np.arange(4 * 8 * 8) % 100).reshape(4, 8, 8).astype(np.int16)
        tumor = (ct > 50).astype(np.uint8)
        aff = _make_affine([1.5, 0.75, 0.75], [3.0, 4.0, 5.0])
        ct_p = tmp_path / "ct.nii.gz"
        tumor_p = tmp_path / "tumor.nii.gz"
        _write_nifti(ct_p, ct, aff)
        _write_nifti(tumor_p, tumor, aff)

        medh5_path = tmp_path / "s.medh5"
        from_nifti(
            images={"CT": ct_p},
            seg={"tumor": tumor_p},
            out_path=medh5_path,
        )

        out_dir = tmp_path / "export"
        written = to_nifti(medh5_path, out_dir)

        assert "image:CT" in written
        assert "seg:tumor" in written

        ct_back = nib.load(str(written["image:CT"]))
        np.testing.assert_array_equal(np.asarray(ct_back.dataobj), ct)
        # Affine round-trip (within float64 tolerance)
        np.testing.assert_allclose(ct_back.affine, aff, atol=1e-6)

        seg_back = nib.load(str(written["seg:tumor"]))
        np.testing.assert_array_equal(
            np.asarray(seg_back.dataobj).astype(bool), tumor.astype(bool)
        )

    def test_unknown_modality_raises(self, tmp_path):
        from medh5.io import to_nifti

        ct = np.zeros((2, 4, 4), dtype=np.float32)
        path = tmp_path / "s.medh5"
        MEDH5File.write(path, images={"CT": ct})
        with pytest.raises(MEDH5ValidationError, match="not found"):
            to_nifti(path, tmp_path / "out", modalities=["MISSING"])


# ---------------------------------------------------------------------------
# DICOM -> medh5
# ---------------------------------------------------------------------------


def _make_dicom_series(directory, n_slices=4, rows=8, cols=8):
    """Create a synthetic axial CT-style series with monotonic IPP."""
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    pixel_spacing = [0.75, 0.75]
    slice_thickness = 1.5
    written_paths = []
    rng = np.random.default_rng(123)
    pixel_planes = []

    for i in range(n_slices):
        plane = rng.integers(0, 2000, size=(rows, cols), dtype=np.uint16)
        pixel_planes.append(plane)
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(
            str(directory / f"slice_{i:03d}.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        ds.PatientID = "P001"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.Manufacturer = "Acme"
        ds.StudyDate = "20260101"
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = slice_thickness
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * slice_thickness]
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = plane.tobytes()

        ds.save_as(str(directory / f"slice_{i:03d}.dcm"), enforce_file_format=True)
        written_paths.append(directory / f"slice_{i:03d}.dcm")

    return written_paths, np.stack(pixel_planes, axis=0)


class TestFromDicom:
    def test_series_to_medh5(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "series"
        _written, expected_volume = _make_dicom_series(dicom_dir, n_slices=5)

        out = tmp_path / "ct.medh5"
        from_dicom(dicom_dir, out, modality_name="CT", checksum=True)

        sample = MEDH5File.read(out)
        assert "CT" in sample.images
        assert sample.images["CT"].shape == expected_volume.shape
        np.testing.assert_array_equal(sample.images["CT"], expected_volume)
        # Spacing should match the synthetic header
        np.testing.assert_allclose(sample.meta.spatial.spacing, [1.5, 0.75, 0.75])
        assert sample.meta.spatial.coord_system == "LPS"
        assert sample.meta.extra is not None
        assert "dicom" in sample.meta.extra
        assert sample.meta.extra["dicom"]["PatientID"] == "P001"
        assert sample.meta.extra["dicom"]["Modality"] == "CT"
        assert MEDH5File.verify(out) is True

    def test_missing_dir_raises(self, tmp_path):
        from medh5.io import from_dicom

        with pytest.raises(MEDH5ValidationError, match="not found"):
            from_dicom(tmp_path / "nope", tmp_path / "x.medh5")

    def test_empty_dir_raises(self, tmp_path):
        from medh5.io import from_dicom

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(MEDH5ValidationError, match="No DICOM"):
            from_dicom(empty, tmp_path / "x.medh5")
