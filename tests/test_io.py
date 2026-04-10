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

    def test_resample_to_reference_modality(self, tmp_path):
        pytest.importorskip("SimpleITK")
        from medh5.io import from_nifti

        ct = np.ones((8, 16, 16), dtype=np.float32)
        pet = np.ones((4, 8, 8), dtype=np.float32) * 5.0
        tumor = np.zeros((4, 8, 8), dtype=np.uint8)
        tumor[1:3, 2:6, 2:6] = 1

        ct_p = tmp_path / "ct.nii.gz"
        pet_p = tmp_path / "pet.nii.gz"
        tumor_p = tmp_path / "tumor.nii.gz"
        _write_nifti(ct_p, ct, _make_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
        _write_nifti(pet_p, pet, _make_affine([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]))
        _write_nifti(tumor_p, tumor, _make_affine([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]))

        out = tmp_path / "resampled.medh5"
        from_nifti(
            images={"CT": ct_p, "PET": pet_p},
            seg={"tumor": tumor_p},
            out_path=out,
            resample_to="CT",
        )

        sample = MEDH5File.read(out)
        assert sample.images["CT"].shape == (8, 16, 16)
        assert sample.images["PET"].shape == (8, 16, 16)
        assert sample.seg is not None
        assert sample.seg["tumor"].shape == (8, 16, 16)
        assert sample.seg["tumor"].dtype == bool
        assert sample.seg["tumor"].any()

    def test_affine_mismatch_raises(self, tmp_path):
        from medh5.io import from_nifti

        ct = np.zeros((4, 8, 8), dtype=np.float32)
        pet = np.zeros((4, 8, 8), dtype=np.float32)
        ct_p = tmp_path / "ct.nii.gz"
        pet_p = tmp_path / "pet.nii.gz"
        _write_nifti(ct_p, ct, _make_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
        _write_nifti(pet_p, pet, _make_affine([1.0, 1.0, 1.0], [5.0, 0.0, 0.0]))

        with pytest.raises(MEDH5ValidationError, match="affine mismatch"):
            from_nifti(images={"CT": ct_p, "PET": pet_p}, out_path=tmp_path / "x.medh5")

    def test_seg_affine_mismatch_raises(self, tmp_path):
        from medh5.io import from_nifti

        ct = np.zeros((4, 8, 8), dtype=np.float32)
        tumor = np.zeros((4, 8, 8), dtype=np.uint8)
        ct_p = tmp_path / "ct.nii.gz"
        tumor_p = tmp_path / "tumor.nii.gz"
        _write_nifti(ct_p, ct, _make_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
        _write_nifti(
            tumor_p,
            tumor,
            _make_affine([1.0, 1.0, 1.0], [0.0, 2.0, 0.0]),
        )

        with pytest.raises(MEDH5ValidationError, match="affine does not match"):
            from_nifti(
                images={"CT": ct_p},
                seg={"tumor": tumor_p},
                out_path=tmp_path / "x.medh5",
            )

    def test_resample_to_external_reference(self, tmp_path):
        pytest.importorskip("SimpleITK")
        from medh5.io import from_nifti

        ct = np.ones((4, 4, 4), dtype=np.float32)
        ref = np.zeros((8, 8, 8), dtype=np.float32)
        ct_p = tmp_path / "ct.nii.gz"
        ref_p = tmp_path / "ref.nii.gz"
        _write_nifti(ct_p, ct, _make_affine([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]))
        _write_nifti(ref_p, ref, _make_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))

        out = tmp_path / "external_ref.medh5"
        from_nifti(images={"CT": ct_p}, out_path=out, resample_to=ref_p)
        sample = MEDH5File.read(out)
        assert sample.images["CT"].shape == (8, 8, 8)

    def test_import_seg_nifti_replace_and_mismatch(self, tmp_path):
        pytest.importorskip("SimpleITK")
        from medh5.io import import_seg_nifti

        medh5_path = tmp_path / "sample.medh5"
        MEDH5File.write(
            medh5_path,
            images={"CT": np.zeros((8, 8, 8), dtype=np.float32)},
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        same_grid_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        same_grid_mask[0, 0, 0] = 1
        same_grid_path = tmp_path / "same.nii.gz"
        _write_nifti(same_grid_path, same_grid_mask, np.eye(4))
        import_seg_nifti(medh5_path, same_grid_path, name="tumor")

        replacement = np.zeros((8, 8, 8), dtype=np.uint8)
        replacement[1, 1, 1] = 1
        replacement_path = tmp_path / "replacement.nii.gz"
        _write_nifti(replacement_path, replacement, np.eye(4))
        import_seg_nifti(medh5_path, replacement_path, name="tumor", replace=True)
        sample = MEDH5File.read(medh5_path)
        assert sample.seg is not None
        assert sample.seg["tumor"][1, 1, 1]

        mismatched = np.zeros((4, 4, 4), dtype=np.uint8)
        mismatched_path = tmp_path / "mismatch.nii.gz"
        _write_nifti(mismatched_path, mismatched, np.diag([2.0, 2.0, 2.0, 1.0]))
        with pytest.raises(MEDH5ValidationError, match="Edited mask grid"):
            import_seg_nifti(medh5_path, mismatched_path, name="bad", resample=False)


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

    def test_unknown_seg_raises(self, tmp_path):
        from medh5.io import to_nifti

        path = tmp_path / "s.medh5"
        MEDH5File.write(
            path,
            images={"CT": np.zeros((2, 4, 4), dtype=np.float32)},
            seg={"tumor": np.zeros((2, 4, 4), dtype=bool)},
        )
        with pytest.raises(MEDH5ValidationError, match="Segmentation 'missing'"):
            to_nifti(path, tmp_path / "out", seg=["missing"])


# ---------------------------------------------------------------------------
# DICOM -> medh5
# ---------------------------------------------------------------------------


def _make_dicom_series(
    directory,
    n_slices=4,
    rows=8,
    cols=8,
    *,
    series_uid=None,
    pixel_spacing=(0.75, 0.75),
    slice_thickness=1.5,
    rescale_slope=None,
    rescale_intercept=None,
    position_step=None,
    multiframe=False,
):
    """Create a synthetic axial CT-style series with monotonic IPP."""
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study_uid = generate_uid()
    pixel_spacing = list(pixel_spacing)
    position_step = slice_thickness if position_step is None else position_step
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
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * position_step]
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        if rescale_slope is not None:
            ds.RescaleSlope = rescale_slope
        if rescale_intercept is not None:
            ds.RescaleIntercept = rescale_intercept
        if multiframe:
            ds.NumberOfFrames = 2
        ds.PixelData = plane.tobytes()

        ds.save_as(str(directory / f"slice_{i:03d}.dcm"), enforce_file_format=True)
        written_paths.append(directory / f"slice_{i:03d}.dcm")

    return written_paths, np.stack(pixel_planes, axis=0), series_uid


class TestFromDicom:
    def test_series_to_medh5(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "series"
        _written, expected_volume, series_uid = _make_dicom_series(
            dicom_dir, n_slices=5
        )

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
        assert sample.meta.extra["dicom"]["selected_series_uid"] == series_uid
        assert MEDH5File.verify(out) is True

    def test_modality_lut_applied(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "series_lut"
        _written, expected_volume, _series_uid = _make_dicom_series(
            dicom_dir,
            n_slices=3,
            rescale_slope=2.0,
            rescale_intercept=-1000.0,
        )

        out = tmp_path / "ct_lut.medh5"
        from_dicom(dicom_dir, out)
        sample = MEDH5File.read(out)
        expected = expected_volume.astype(np.float64) * 2.0 - 1000.0
        np.testing.assert_array_equal(sample.images["CT"], expected)

    def test_extra_tags_preserve_sequence_values(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "series_tags"
        _make_dicom_series(dicom_dir, n_slices=2)
        out = tmp_path / "ct_tags.medh5"
        from_dicom(dicom_dir, out, extra_tags=["PixelSpacing"])
        sample = MEDH5File.read(out)
        assert sample.meta.extra is not None
        assert sample.meta.extra["dicom"]["PixelSpacing"] == [0.75, 0.75]

    def test_selects_largest_series_deterministically(self, tmp_path):
        from medh5.io import from_dicom

        root = tmp_path / "multi"
        major_uid = "1.2.3.4.10"
        minor_uid = "1.2.3.4.2"
        _make_dicom_series(root / "a", n_slices=2, series_uid=minor_uid)
        _written, expected_volume, _ = _make_dicom_series(
            root / "b", n_slices=4, series_uid=major_uid
        )

        out = tmp_path / "selected.medh5"
        from_dicom(root, out)
        sample = MEDH5File.read(out)
        np.testing.assert_array_equal(sample.images["CT"], expected_volume)
        assert sample.meta.extra is not None
        assert sample.meta.extra["dicom"]["selected_series_uid"] == major_uid

    def test_series_uid_filter(self, tmp_path):
        from medh5.io import from_dicom

        root = tmp_path / "series_uid"
        _make_dicom_series(root / "a", n_slices=2, series_uid="1.2.3")
        _written, expected_volume, _ = _make_dicom_series(
            root / "b", n_slices=3, series_uid="1.2.4"
        )
        out = tmp_path / "selected_uid.medh5"
        from_dicom(root, out, series_uid="1.2.4")
        sample = MEDH5File.read(out)
        np.testing.assert_array_equal(sample.images["CT"], expected_volume)

    def test_inconsistent_slice_spacing_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "bad_spacing"
        _make_dicom_series(dicom_dir, n_slices=3, position_step=1.5)
        # overwrite one slice with a different ImagePositionPatient spacing
        ds = pydicom.dcmread(str(dicom_dir / "slice_002.dcm"))
        ds.ImagePositionPatient = [0.0, 0.0, 10.0]
        ds.save_as(str(dicom_dir / "slice_002.dcm"), enforce_file_format=True)

        with pytest.raises(MEDH5ValidationError, match="slice spacing"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

    def test_multiframe_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "multiframe"
        _make_dicom_series(dicom_dir, n_slices=1, multiframe=True)
        with pytest.raises(MEDH5ValidationError, match="Multi-frame"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

    def test_series_uid_not_found_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "series_uid_missing"
        _make_dicom_series(dicom_dir, n_slices=2, series_uid="1.2.3")
        with pytest.raises(MEDH5ValidationError, match="SeriesInstanceUID"):
            from_dicom(dicom_dir, tmp_path / "x.medh5", series_uid="9.9.9")

    def test_missing_pixel_data_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "no_pixeldata"
        written, _volume, _uid = _make_dicom_series(dicom_dir, n_slices=1)
        ds = pydicom.dcmread(str(written[0]))
        del ds.PixelData
        ds.save_as(str(written[0]), enforce_file_format=True)

        with pytest.raises(MEDH5ValidationError, match="No DICOM files with PixelData"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

    def test_missing_orientation_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "missing_iop"
        written, _volume, _uid = _make_dicom_series(dicom_dir, n_slices=2)
        ds = pydicom.dcmread(str(written[0]))
        del ds.ImageOrientationPatient
        ds.save_as(str(written[0]), enforce_file_format=True)

        with pytest.raises(MEDH5ValidationError, match="ImageOrientationPatient"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

    def test_bad_pixel_spacing_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "bad_pixel_spacing"
        _make_dicom_series(dicom_dir, n_slices=2, pixel_spacing=(0.0, 1.0))
        with pytest.raises(MEDH5ValidationError, match="Invalid PixelSpacing"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

    def test_unsupported_photometric_raises(self, tmp_path):
        from medh5.io import from_dicom

        dicom_dir = tmp_path / "rgb"
        written, _volume, _uid = _make_dicom_series(dicom_dir, n_slices=1)
        ds = pydicom.dcmread(str(written[0]))
        ds.PhotometricInterpretation = "RGB"
        ds.save_as(str(written[0]), enforce_file_format=True)

        with pytest.raises(MEDH5ValidationError, match="PhotometricInterpretation"):
            from_dicom(dicom_dir, tmp_path / "x.medh5")

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
