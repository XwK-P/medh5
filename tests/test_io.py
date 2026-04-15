"""Tests for medh5.io NIfTI and DICOM converters."""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# nnU-Net v2 ⇄ medh5
# ---------------------------------------------------------------------------


def _build_nnunetv2_dataset(
    root,
    *,
    cases=("c001", "c002"),
    channel_names=(("0", "T1"), ("1", "T2")),
    labels=(("background", 0), ("tumor", 1), ("edema", 2)),
    include_test=False,
    test_cases=("t001",),
    extra_json=None,
):
    """Create a synthetic nnU-Net v2 dataset on disk and return (root, arrays).

    ``arrays`` is a dict mapping ``{case_id: {"channel_{idx}": arr, "label": arr}}``
    so tests can verify byte-identical round-trips.
    """
    import json as _json

    root = Path(root)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    aff = _make_affine([1.5, 1.0, 1.0], [4.0, -3.0, 2.0])
    arrays: dict = {}
    rng = np.random.default_rng(42)
    n_channels = len(channel_names)

    for cid in cases:
        arrays[cid] = {}
        for idx_str, _name in channel_names:
            arr = rng.standard_normal((6, 8, 10)).astype(np.float32)
            arrays[cid][f"channel_{idx_str}"] = arr
            img_path = root / "imagesTr" / f"{cid}_{int(idx_str):04d}.nii.gz"
            _write_nifti(img_path, arr, aff)
        lab = np.zeros((6, 8, 10), dtype=np.uint8)
        lab[1:3, 1:4, 1:5] = 1
        lab[3:5, 4:7, 5:9] = 2
        arrays[cid]["label"] = lab
        _write_nifti(root / "labelsTr" / f"{cid}.nii.gz", lab, aff)

    if include_test:
        (root / "imagesTs").mkdir(parents=True, exist_ok=True)
        for cid in test_cases:
            arrays[cid] = {}
            for idx_str, _name in channel_names:
                arr = rng.standard_normal((6, 8, 10)).astype(np.float32)
                arrays[cid][f"channel_{idx_str}"] = arr
                _write_nifti(
                    root / "imagesTs" / f"{cid}_{int(idx_str):04d}.nii.gz", arr, aff
                )

    payload = {
        "channel_names": dict(channel_names),
        "labels": dict(labels),
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": root.name,
    }
    if extra_json is not None:
        payload.update(extra_json)
    (root / "dataset.json").write_text(_json.dumps(payload))

    # Return the expected number of channels as a convenience.
    return root, arrays, n_channels


class TestFromNnunetv2:
    def test_minimal_dataset_import(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset001_Foo"
        _build_nnunetv2_dataset(src)

        out = tmp_path / "converted"
        written = from_nnunetv2(src, out)

        assert len(written["train"]) == 2
        assert len(written["test"]) == 0
        assert (out / "imagesTr" / "c001.medh5").is_file()
        assert (out / "imagesTr" / "c002.medh5").is_file()

        sample = MEDH5File.read(out / "imagesTr" / "c001.medh5")
        assert set(sample.images.keys()) == {"T1", "T2"}
        assert sample.seg is not None
        assert set(sample.seg.keys()) == {"tumor", "edema"}
        assert sample.seg["tumor"].dtype == bool
        assert sample.seg["edema"].any()
        # background is not stored as a seg entry
        assert "background" not in sample.seg
        assert sample.meta.spatial.coord_system == "RAS"
        np.testing.assert_allclose(sample.meta.spatial.spacing, [1.5, 1.0, 1.0])
        # extra["nnunetv2"] round-trips
        assert sample.meta.extra is not None
        nnu = sample.meta.extra["nnunetv2"]
        assert nnu["channel_names"] == {"0": "T1", "1": "T2"}
        assert nnu["labels"] == {"background": 0, "tumor": 1, "edema": 2}
        assert nnu["file_ending"] == ".nii.gz"

    def test_include_test_flag(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset002_Bar"
        _build_nnunetv2_dataset(src, include_test=True)

        out = tmp_path / "converted"
        written = from_nnunetv2(src, out)
        assert len(written["test"]) == 1
        test_sample = MEDH5File.read(out / "imagesTs" / "t001.medh5")
        assert test_sample.seg is None

        # Now skip the test split.
        out_no_test = tmp_path / "converted_no_test"
        written_no_test = from_nnunetv2(src, out_no_test, include_test=False)
        assert len(written_no_test["test"]) == 0
        assert not (out_no_test / "imagesTs").exists()

    def test_region_labels_rejected(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset003_Region"
        _build_nnunetv2_dataset(
            src,
            labels=(("background", 0), ("whole_tumor", 1)),
            extra_json={
                "labels": {"background": 0, "whole_tumor": [1, 2]},
                "regions_class_order": [1, 2],
            },
        )
        with pytest.raises(MEDH5ValidationError, match="[Rr]egion"):
            from_nnunetv2(src, tmp_path / "out")

    def test_missing_dataset_json_raises(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "empty"
        src.mkdir()
        with pytest.raises(MEDH5ValidationError, match="dataset.json"):
            from_nnunetv2(src, tmp_path / "out")

    def test_missing_channel_raises(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset004_Missing"
        _build_nnunetv2_dataset(src)
        # Remove one channel file for case c001.
        (src / "imagesTr" / "c001_0001.nii.gz").unlink()
        with pytest.raises(MEDH5ValidationError, match="missing"):
            from_nnunetv2(src, tmp_path / "out")

    def test_label_grid_mismatch_raises(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset005_Bad"
        _build_nnunetv2_dataset(src)
        # Overwrite the label with a wrong-shaped volume.
        bad = np.zeros((4, 4, 4), dtype=np.uint8)
        _write_nifti(src / "labelsTr" / "c001.nii.gz", bad, np.eye(4))
        with pytest.raises(MEDH5ValidationError, match="label"):
            from_nnunetv2(src, tmp_path / "out")

    def test_bad_labels_mapping_raises(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset006_BadLabels"
        _build_nnunetv2_dataset(
            src,
            extra_json={"labels": {"background": 0, "tumor": 1, "edema": 3}},
        )
        with pytest.raises(MEDH5ValidationError, match="consecutive"):
            from_nnunetv2(src, tmp_path / "out")

    def test_undeclared_label_value_rejected(self, tmp_path):
        from medh5.io import from_nnunetv2

        src = tmp_path / "Dataset007_StrayVal"
        _build_nnunetv2_dataset(src)
        # Paint a voxel with a class value not declared in dataset.json.
        aff = _make_affine([1.5, 1.0, 1.0], [4.0, -3.0, 2.0])
        stray = np.zeros((6, 8, 10), dtype=np.uint8)
        stray[0, 0, 0] = 7  # not in {0, 1, 2}
        _write_nifti(src / "labelsTr" / "c001.nii.gz", stray, aff)
        with pytest.raises(MEDH5ValidationError, match="undeclared values"):
            from_nnunetv2(src, tmp_path / "out")


class TestToNnunetv2:
    def test_roundtrip_preserves_images_and_labels(self, tmp_path):
        from medh5.io import from_nnunetv2, to_nnunetv2

        src = tmp_path / "Dataset010_Round"
        _build_nnunetv2_dataset(src, include_test=True)

        medh5_dir = tmp_path / "medh5_dir"
        from_nnunetv2(src, medh5_dir, include_test=True)

        out = tmp_path / "Dataset010_Roundtrip"
        dataset_json_path = to_nnunetv2(medh5_dir, out)
        assert dataset_json_path.is_file()

        # Training images + labels survive byte-identical.
        for cid in ("c001", "c002"):
            for idx in (0, 1):
                orig = nib.load(str(src / "imagesTr" / f"{cid}_{idx:04d}.nii.gz"))
                back = nib.load(str(out / "imagesTr" / f"{cid}_{idx:04d}.nii.gz"))
                np.testing.assert_array_equal(
                    np.asarray(back.dataobj), np.asarray(orig.dataobj)
                )
                np.testing.assert_allclose(back.affine, orig.affine, atol=1e-6)
            orig_lab = nib.load(str(src / "labelsTr" / f"{cid}.nii.gz"))
            back_lab = nib.load(str(out / "labelsTr" / f"{cid}.nii.gz"))
            np.testing.assert_array_equal(
                np.asarray(back_lab.dataobj), np.asarray(orig_lab.dataobj)
            )

        # Test images survive.
        for idx in (0, 1):
            orig = nib.load(str(src / "imagesTs" / f"t001_{idx:04d}.nii.gz"))
            back = nib.load(str(out / "imagesTs" / f"t001_{idx:04d}.nii.gz"))
            np.testing.assert_array_equal(
                np.asarray(back.dataobj), np.asarray(orig.dataobj)
            )

        # dataset.json re-parses with equivalent core fields.
        import json as _json

        orig_json = _json.loads((src / "dataset.json").read_text())
        back_json = _json.loads(dataset_json_path.read_text())
        assert back_json["channel_names"] == orig_json["channel_names"]
        assert back_json["labels"] == orig_json["labels"]
        assert back_json["numTraining"] == orig_json["numTraining"]
        assert back_json["file_ending"] == orig_json["file_ending"]

    def test_export_flat_medh5_without_nnunet_meta(self, tmp_path):
        from medh5.io import to_nnunetv2

        # Write a single .medh5 file with no nnunetv2 metadata.
        images = {"CT": np.zeros((4, 6, 8), dtype=np.float32)}
        seg_mask = np.zeros((4, 6, 8), dtype=bool)
        seg_mask[1:3, 2:4, 3:6] = True
        seg = {"tumor": seg_mask}
        MEDH5File.write(
            tmp_path / "case1.medh5",
            images=images,
            seg=seg,
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        out = tmp_path / "DatasetFlat"
        dataset_json_path = to_nnunetv2(tmp_path, out)
        assert dataset_json_path.is_file()
        assert (out / "imagesTr" / "case1_0000.nii.gz").is_file()
        assert (out / "labelsTr" / "case1.nii.gz").is_file()

        import json as _json

        payload = _json.loads(dataset_json_path.read_text())
        assert payload["channel_names"] == {"0": "CT"}
        assert payload["labels"] == {"background": 0, "tumor": 1}
        assert payload["numTraining"] == 1

    def test_export_empty_src_raises(self, tmp_path):
        from medh5.io import to_nnunetv2

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(MEDH5ValidationError, match="No training"):
            to_nnunetv2(empty, tmp_path / "out")

    def test_extra_seg_mask_rejected_on_export(self, tmp_path):
        from medh5.io import to_nnunetv2

        # Write a .medh5 file carrying nnU-Net metadata plus a seg mask whose
        # name is not declared in ``labels``. Export must refuse rather than
        # silently drop the stray mask.
        images_tr = tmp_path / "imagesTr"
        images_tr.mkdir()
        images = {"T1": np.zeros((4, 6, 8), dtype=np.float32)}
        tumor = np.zeros((4, 6, 8), dtype=bool)
        tumor[1, 1, 1] = True
        rogue = np.zeros((4, 6, 8), dtype=bool)
        rogue[2, 2, 2] = True
        nnunet_meta = {
            "channel_names": {"0": "T1"},
            "labels": {"background": 0, "tumor": 1},
            "numTraining": 1,
            "file_ending": ".nii.gz",
        }
        MEDH5File.write(
            images_tr / "case1.medh5",
            images=images,
            seg={"tumor": tumor, "rogue": rogue},
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            extra={"nnunetv2": nnunet_meta},
        )
        with pytest.raises(MEDH5ValidationError, match="not declared"):
            to_nnunetv2(tmp_path, tmp_path / "out")

    def test_channel_set_mismatch_rejected_on_export(self, tmp_path):
        from medh5.io import to_nnunetv2

        # .medh5 declares two channels in nnU-Net metadata but only stores one.
        images_tr = tmp_path / "imagesTr"
        images_tr.mkdir()
        images = {"T1": np.zeros((4, 6, 8), dtype=np.float32)}
        nnunet_meta = {
            "channel_names": {"0": "T1", "1": "T2"},
            "labels": {"background": 0, "tumor": 1},
            "numTraining": 1,
            "file_ending": ".nii.gz",
        }
        # ``to_nnunetv2`` resolves channel order from the first file's
        # metadata, so the mismatch must be caught by ``_write_case_nifti``.
        # Use two files so the resolver sees the full channel list first.
        full_images = {
            "T1": np.zeros((4, 6, 8), dtype=np.float32),
            "T2": np.zeros((4, 6, 8), dtype=np.float32),
        }
        MEDH5File.write(
            images_tr / "case0.medh5",
            images=full_images,
            seg={"tumor": np.zeros((4, 6, 8), dtype=bool)},
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            extra={"nnunetv2": nnunet_meta},
        )
        MEDH5File.write(
            images_tr / "case1.medh5",
            images=images,
            seg={"tumor": np.zeros((4, 6, 8), dtype=bool)},
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            extra={"nnunetv2": nnunet_meta},
        )
        with pytest.raises(MEDH5ValidationError, match="channels do not match"):
            to_nnunetv2(tmp_path, tmp_path / "out")
