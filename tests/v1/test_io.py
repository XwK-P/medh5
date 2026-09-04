"""Converters: NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net v2 (plan §7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5FileError, MEDH5SchemaError, MEDH5ValidationError
from medh5.io.grouping import Occasion, SubjectGroup, group_by_subject
from medh5.io.report import ConversionReport, Note, merge_reports
from medh5.validate import validate_file
from tests.v1.conftest import write_legacy_sample

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")

from medh5.io.nifti import (  # noqa: E402
    convert_world,
    from_nifti,
    import_seg_nifti,
    read_nifti,
    to_nifti,
)

SHAPE_XYZ = (24, 20, 12)
AFFINE = np.array(
    [[0.8, 0.0, 0.0, -10.0], [0.0, 0.9, 0.0, -20.0], [0.0, 0.0, 2.0, 5.0], [0, 0, 0, 1]]
)


@pytest.fixture
def volumes(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(11)
    ct = rng.integers(-1000, 1500, SHAPE_XYZ).astype(np.int16)
    liver = np.zeros(SHAPE_XYZ, dtype=np.uint8)
    liver[4:14, 4:12, 2:8] = 1
    lesion = np.zeros(SHAPE_XYZ, dtype=np.uint8)
    lesion[6:9, 6:9, 3:5] = 1
    paths = {}
    for name, array in (("ct", ct), ("liver", liver), ("lesion", lesion)):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, AFFINE), str(path))
        paths[name] = path
    paths["_ct"] = ct  # type: ignore[assignment]
    paths["_liver"] = liver  # type: ignore[assignment]
    return paths


class TestReport:
    def test_a_guess_is_not_a_failure_and_a_warning_is(self):
        report = ConversionReport(converter="test")
        report.decision("encoding", "chose layers", {"kind": "layers"})
        report.guess("order", "ordered by mtime")
        assert report.ok
        assert len(report.guesses) == 1
        report.warn("geometry", "spacing disagreed")
        assert not report.ok
        assert "1 warning" in report.format()
        assert "GUESS" in report.format(verbose=False)

    def test_detail_may_carry_a_key_called_kind(self):
        """The report's own parameter names must not shadow a converter's data."""
        report = ConversionReport()
        note = report.decision("encoding", "m", {"kind": "layers"})
        assert note.detail == {"kind": "layers"}
        assert note.kind == "encoding"

    def test_json_and_merge(self):
        first = ConversionReport(converter="a", outputs=["x"])
        first.warn("k", "m")
        second = ConversionReport(converter="b", outputs=["y"])
        merged = merge_reports([first, second], converter="batch")
        assert merged.outputs == ["x", "y"]
        assert not merged.ok
        payload = merged.to_json()
        assert payload["counts"]["warning"] == 1
        assert Note("k", "m").to_json()["severity"] == "info"
        assert first.of_kind("k")


class TestNiftiGeometryIsNeverInvented:
    """§3.3: a grid this converter made up is indistinguishable from a measured one."""

    def _write(self, path, sform_code, qform_code, sform=None, qform=None):
        data = np.zeros((6, 7, 8), np.int16)
        image = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        image.set_sform(sform, code=sform_code)
        image.set_qform(qform, code=qform_code)
        nib.save(image, str(path))
        return path

    def test_a_file_declaring_no_spatial_mapping_is_refused(self, tmp_path):
        """`sform_code == qform_code == 0` means "voxel indices only".

        nibabel still returns an affine, rebuilt from pixdim, and importing it
        mints a world grid nobody measured -- silently, with the conversion
        report saying "0 guesses".
        """
        src = self._write(tmp_path / "nocode.nii.gz", 0, 0)
        with pytest.raises(MEDH5ValidationError, match="no spatial mapping"):
            from_nifti({"IMG": src}, tmp_path / "out.medh5", sample_id="s")

    def test_the_pixdim_fallback_is_available_but_recorded_as_a_guess(self, tmp_path):
        src = self._write(tmp_path / "nocode.nii.gz", 0, 0)
        report = from_nifti(
            {"IMG": src},
            tmp_path / "out.medh5",
            sample_id="s",
            assume_geometry=True,
        )
        assert any("no spatial mapping" in n.message for n in report.guesses)

    def test_an_sform_qform_disagreement_is_reported(self, tmp_path):
        """The classic signature of a file one tool updated and another did not.

        Preferring the sform is conventional and defensible; reporting it as no
        decision at all is not -- a reader preferring the qform puts the volume
        somewhere else, and a cohort conversion never surfaced that some files
        carried contradictory geometry.
        """
        src = self._write(
            tmp_path / "disagree.nii.gz",
            2,
            1,
            sform=np.diag([2.0, 2.0, 2.0, 1.0]),
            qform=np.diag([3.0, 3.0, 3.0, 1.0]),
        )
        report = from_nifti({"IMG": src}, tmp_path / "out.medh5", sample_id="s")
        assert any("sform and qform" in n.message for n in report.guesses)

    def test_a_well_formed_file_reports_no_geometry_guess(self, tmp_path):
        src = self._write(
            tmp_path / "ok.nii.gz",
            2,
            2,
            sform=np.diag([2.0, 0.8, 0.8, 1.0]),
            qform=np.diag([2.0, 0.8, 0.8, 1.0]),
        )
        report = from_nifti({"IMG": src}, tmp_path / "out.medh5", sample_id="s")
        assert not [n for n in report.guesses if n.kind == "geometry"]


class TestNifti:
    def test_S3_1_RAS_becomes_LPS_by_flipping_the_affine_not_the_voxels(
        self, volumes, tmp_path
    ):
        from_nifti({"CT": volumes["ct"]}, tmp_path / "lps.medh5")
        with medh5.open(tmp_path / "lps.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.coord_system == "LPS"
            # RAS origin (-10, -20, 5) -> LPS (10, 20, 5), then z-first ordering.
            assert np.allclose(grid.origin, [10.0, 20.0, 5.0])
            assert np.array_equal(
                sample.images["CT"].read(), np.transpose(volumes["_ct"], (2, 1, 0))
            )

    def test_keeping_RAS_is_available_and_recorded(self, volumes, tmp_path):
        report = from_nifti(
            {"CT": volumes["ct"]}, tmp_path / "ras.medh5", coord_system="RAS"
        )
        with medh5.open(tmp_path / "ras.medh5") as sample:
            assert sample.grids["ref"].coord_system == "RAS"
            # The origin is a world point: reordering the *axes* permutes
            # spacing and direction columns, never the origin's components.
            assert np.allclose(sample.grids["ref"].origin, [-10.0, -20.0, 5.0])
        assert not report.of_kind("coord_system")

    def test_S3_1_axes_are_reordered_and_spacing_follows(self, volumes, tmp_path):
        from_nifti({"CT": volumes["ct"]}, tmp_path / "s.medh5")
        with medh5.open(tmp_path / "s.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == tuple(reversed(SHAPE_XYZ))
            assert np.allclose(grid.spacing, [2.0, 0.9, 0.8])

    def test_S3_1_a_4D_NIfTI_keeps_its_spatial_axes(self, tmp_path):
        """NIfTI puts time *after* i, j, k, so the spatial block leads.

        Reversing the trailing three axes instead moves t into a spatial slot
        and gives the grid a spacing belonging to a different axis --- silently,
        for every cine, DCE and 4-D CT series.
        """
        series = np.zeros((*SHAPE_XYZ, 5), dtype=np.int16)
        series[1, 2, 3, 4] = 7
        path = tmp_path / "cine.nii.gz"
        nib.save(nib.Nifti1Image(series, AFFINE), str(path))

        data, geometry = read_nifti(path)

        assert data.shape == (5, *reversed(SHAPE_XYZ))
        assert data[4, 3, 2, 1] == 7, "the voxel kept its (t, z, y, x) home"
        assert np.allclose(geometry["spacing"], [2.0, 0.9, 0.8])
        assert geometry["axis_order"] == (3, 2, 1, 0)

    def test_S3_1_a_4D_series_converts_and_exports_unchanged(self, tmp_path):
        """The 4-D path has to work end to end, not just in `read_nifti`.

        `from_nifti` passed a 4-D shape with no `axis_kinds`, and `add_grid`
        defaults only cover 2-D and 3-D, so a cine, DCE or 4-D CT conversion
        raised before writing anything.  `to_nifti` then reversed only the
        trailing three axes on the way out, sending time to a spatial slot
        while the affine still described (x, y, z).
        """
        series = np.zeros((*SHAPE_XYZ, 5), dtype=np.int16)
        series[1, 2, 3, 4] = 7
        source = tmp_path / "cine.nii.gz"
        nib.save(nib.Nifti1Image(series, AFFINE), str(source))

        from_nifti({"CINE": source}, tmp_path / "cine.medh5")
        with medh5.open(tmp_path / "cine.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == (5, *reversed(SHAPE_XYZ))
            assert grid.axis_kinds == ("time", "spatial", "spatial", "spatial")
            assert grid.axis_names == ("t", "z", "y", "x")
            assert np.allclose(grid.spacing, [2.0, 0.9, 0.8]), "spatial axes only"
            assert sample.images["CINE"].read()[4, 3, 2, 1] == 7

        back = to_nifti(tmp_path / "cine.medh5", "CINE", tmp_path / "back.nii.gz")
        restored = nib.load(str(back))
        assert restored.shape == (*SHAPE_XYZ, 5), "NIfTI puts time after x, y, z"
        assert np.allclose(restored.header.get_zooms()[:3], [0.8, 0.9, 2.0])
        assert np.array_equal(np.asanyarray(restored.dataobj), series)

    def test_a_NIfTI_with_axes_past_time_is_refused(self, tmp_path):
        """dim[5] carries components whose MEDH5 kind depends on the producer."""
        source = tmp_path / "tensor.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((*SHAPE_XYZ, 2, 3), np.int16), AFFINE), str(source)
        )
        with pytest.raises(MEDH5ValidationError, match="beyond"):
            from_nifti({"DTI": source}, tmp_path / "dti.medh5")

    def test_transpose_can_be_turned_off(self, volumes, tmp_path):
        """And the declared axes have to describe the array that was written.

        The axis names came from the reordered layout regardless, so a file
        left in NIfTI order was labelled `(z, y, x)` over an `(x, y, z)` array.
        """
        from_nifti({"CT": volumes["ct"]}, tmp_path / "t.medh5", transpose=False)
        with medh5.open(tmp_path / "t.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == SHAPE_XYZ
            assert grid.axis_names == ("x", "y", "z")
            assert np.allclose(grid.spacing, [0.8, 0.9, 2.0]), "and follow the axes"

    def test_S3_2_a_time_axis_carries_the_frame_times(self, tmp_path):
        """§3.2 requires `time_values` wherever there is a time axis.

        The NIfTI states a temporal zoom and a `toffset`; the converter was
        declaring the time axis and discarding both, so a cine series arrived
        with no frame timing at all.
        """
        image = nib.Nifti1Image(np.zeros((*SHAPE_XYZ, 5), np.int16), AFFINE)
        image.header.set_xyzt_units("mm", "sec")
        image.header["pixdim"][4] = 2.5
        source = tmp_path / "cine.nii.gz"
        nib.save(image, str(source))

        report = from_nifti({"CINE": source}, tmp_path / "cine.medh5")
        with medh5.open(tmp_path / "cine.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.time_units == "s"
            assert grid.time_values == (0.0, 2.5, 5.0, 7.5, 10.0)
        assert [n.severity for n in report.of_kind("time_values")] == ["decision"]

    def test_frame_times_the_source_never_stated_are_a_guess(self, tmp_path):
        """A grid still needs `time_values`, so the fallback is recorded as one."""
        image = nib.Nifti1Image(np.zeros((*SHAPE_XYZ, 3), np.int16), AFFINE)
        image.header["pixdim"][4] = 0.0
        source = tmp_path / "notr.nii.gz"
        nib.save(image, str(source))

        report = from_nifti({"CINE": source}, tmp_path / "notr.medh5")
        with medh5.open(tmp_path / "notr.medh5") as sample:
            assert sample.grids["ref"].time_values == (0.0, 1.0, 2.0)
        assert [n.severity for n in report.of_kind("time_values")] == ["guess"]

    def test_S3_6_a_2D_radiograph_converts(self, tmp_path):
        """§3.6 gives a 2-D grid S = 2, and nibabel hands over a 3-D affine.

        The converter passed the unreduced spacing and 3x3 direction straight
        through, so `add_grid` raised E109 and the 2-D case the spec explicitly
        supports could not be imported at all.
        """
        source = tmp_path / "xray.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((32, 40), np.int16), AFFINE), str(source))

        from_nifti({"XR": source}, tmp_path / "xray.medh5")
        with medh5.open(tmp_path / "xray.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == (32, 40)
            assert grid.axis_kinds == ("spatial", "spatial")
            assert np.allclose(grid.spacing, [0.8, 0.9])
            assert np.asarray(grid.direction).shape == (2, 2)

    def test_S3_6_a_plane_tilted_in_3D_is_refused(self, tmp_path):
        """Flattening it would move every pixel to somewhere it is not."""
        tilted = np.array(AFFINE, dtype=float, copy=True)
        tilted[:3, :2] = [[0.8, 0.0], [0.0, 0.6], [0.0, 0.67]]
        source = tmp_path / "tilt.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((32, 40), np.int16), tilted), str(source))
        with pytest.raises(MEDH5ValidationError) as exc:
            from_nifti({"XR": source}, tmp_path / "tilt.medh5")
        # Uncoded: E102 is `direction` not orthonormal, and this direction is
        # perfectly orthonormal -- it just cannot be reduced to a 2x2.
        assert exc.value.code is None

    def test_a_4D_series_cannot_keep_its_NIfTI_axis_order(self, tmp_path):
        """§3.1 wants the spatial axes trailing; NIfTI puts time there.

        Declaring the reordered axes over an untransposed 4-D array marked the
        x axis `time` and handed every spatial axis a spacing belonging to a
        different one.  There is no valid grid for this array, so it is refused
        rather than described wrongly.
        """
        source = tmp_path / "cine.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((*SHAPE_XYZ, 5), np.int16), AFFINE), str(source)
        )
        with pytest.raises(MEDH5ValidationError, match="trailing"):
            from_nifti({"CINE": source}, tmp_path / "x.medh5", transpose=False)

    def test_masks_become_an_annotation_with_a_minted_label_set(
        self, volumes, tmp_path
    ):
        report = from_nifti(
            {"CT": volumes["ct"]},
            tmp_path / "seg.medh5",
            masks={"liver": volumes["liver"], "lesion": volumes["lesion"]},
        )
        with medh5.open(tmp_path / "seg.medh5") as sample:
            assert {c.key for c in sample.label_set} == {"liver", "lesion"}
            seg = sample.annotations["seg"]
            assert np.array_equal(
                seg.dense(["liver"])[0],
                np.transpose(volumes["_liver"], (2, 1, 0)).astype(bool),
            )
        assert report.of_kind("label_set")
        assert report.guesses, "coverage inferred from the masks is a guess"

    def test_a_disagreeing_grid_is_refused_rather_than_resampled(
        self, volumes, tmp_path
    ):
        odd = tmp_path / "odd.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), AFFINE), str(odd))
        with pytest.raises(MEDH5ValidationError, match="resample"):
            from_nifti({"CT": volumes["ct"], "PET": odd}, tmp_path / "x.medh5")
        shifted = np.array(AFFINE)
        shifted[0, 3] += 3.0
        other = tmp_path / "shift.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(SHAPE_XYZ, np.int16), shifted), str(other))
        with pytest.raises(MEDH5ValidationError, match="origin"):
            from_nifti({"CT": volumes["ct"], "PET": other}, tmp_path / "y.medh5")

    def test_a_grid_disagreement_does_not_borrow_a_format_code(self, volumes, tmp_path):
        """§15.2's codes describe a MEDH5 file; these are two NIfTI volumes.

        The shape mismatch is not `E202` (an image disagreeing with its grid --
        neither of these is a grid) and the geometry mismatch is not `E101` (a
        reference to a grid that does not exist -- nothing here is referenced).
        """
        odd = tmp_path / "odd.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), AFFINE), str(odd))
        with pytest.raises(MEDH5ValidationError) as shape_error:
            from_nifti({"CT": volumes["ct"], "PET": odd}, tmp_path / "x.medh5")
        assert shape_error.value.code is None

        shifted = np.array(AFFINE)
        shifted[0, 3] += 3.0
        other = tmp_path / "shift.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(SHAPE_XYZ, np.int16), shifted), str(other))
        with pytest.raises(MEDH5ValidationError) as geometry_error:
            from_nifti({"CT": volumes["ct"], "PET": other}, tmp_path / "y.medh5")
        assert geometry_error.value.code is None

    def test_an_empty_image_set_is_refused(self, tmp_path):
        with pytest.raises(MEDH5ValidationError):
            from_nifti({}, tmp_path / "x.medh5")

    def test_S3_3_the_round_trip_preserves_affine_and_voxels(self, volumes, tmp_path):
        from_nifti({"CT": volumes["ct"]}, tmp_path / "r.medh5")
        back = to_nifti(tmp_path / "r.medh5", "CT", tmp_path / "back.nii.gz")
        loaded = nib.load(str(back))
        assert np.allclose(loaded.affine, AFFINE)
        assert np.array_equal(np.asanyarray(loaded.dataobj), volumes["_ct"])

    def test_exporting_one_class_and_the_labelmap(self, volumes, tmp_path):
        from_nifti(
            {"CT": volumes["ct"]},
            tmp_path / "e.medh5",
            masks={"liver": volumes["liver"]},
        )
        one = to_nifti(
            tmp_path / "e.medh5",
            "CT",
            tmp_path / "liver.nii.gz",
            annotation="seg",
            class_key="liver",
        )
        assert np.array_equal(
            np.asanyarray(nib.load(str(one)).dataobj).astype(bool),
            volumes["_liver"].astype(bool),
        )
        whole = to_nifti(
            tmp_path / "e.medh5", "CT", tmp_path / "lm.nii.gz", annotation="seg"
        )
        assert np.asanyarray(nib.load(str(whole)).dataobj).max() == 1

    def test_import_seg_into_an_existing_sample(self, volumes, tmp_path):
        from_nifti({"CT": volumes["ct"]}, tmp_path / "i.medh5")
        report = import_seg_nifti(
            tmp_path / "i.medh5", {"liver": volumes["liver"]}, ann_id="late"
        )
        with medh5.open(tmp_path / "i.medh5") as sample:
            assert "late" in sample.annotations
            assert np.array_equal(
                sample.annotations["late"].dense(["liver"])[0],
                np.transpose(volumes["_liver"], (2, 1, 0)).astype(bool),
            )
        assert report.outputs

    def test_a_mask_of_the_wrong_shape_is_refused(self, volumes, tmp_path):
        from_nifti({"CT": volumes["ct"]}, tmp_path / "i.medh5")
        odd = tmp_path / "odd.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), AFFINE), str(odd))
        with pytest.raises(MEDH5ValidationError) as exc:
            import_seg_nifti(tmp_path / "i.medh5", {"liver": odd})
        assert exc.value.code == "E405"

    def test_convert_world_is_its_own_inverse_and_checks_its_inputs(self):
        there = convert_world(AFFINE, source="RAS", target="LPS")
        assert np.allclose(convert_world(there, source="LPS", target="RAS"), AFFINE)
        assert np.allclose(convert_world(AFFINE, source="RAS", target="RAS"), AFFINE)
        with pytest.raises(MEDH5ValidationError, match="coordinate system"):
            convert_world(AFFINE, source="RAS", target="quaternionic")
        with pytest.raises(MEDH5ValidationError, match="2-D and 3-D"):
            convert_world(np.eye(5), source="RAS", target="LPS")

    def test_S3_6_convert_world_handles_a_2D_affine(self):
        """§3.6 gives a 2-D grid a 3x3 affine, and both exporters convert one.

        Refusing it here is why no 2-D sample could be exported at all: the
        importers take a radiograph and the exporters stopped on its affine.
        """
        plane = np.array([[0.8, 0.0, -1.0], [0.0, 0.9, -2.0], [0.0, 0.0, 1.0]])
        there = convert_world(plane, source="RAS", target="LPS")
        assert np.allclose(there[:2, 2], [1.0, 2.0])
        assert np.allclose(convert_world(there, source="LPS", target="RAS"), plane)

    def test_read_nifti_reports_units_and_header(self, volumes):
        _, geometry = read_nifti(volumes["ct"])
        assert geometry["units"] == "mm"
        assert set(geometry["header"]) == {"descrip", "scl_slope", "scl_inter"}

    def test_converted_files_validate(self, volumes, tmp_path):
        from_nifti(
            {"CT": volumes["ct"]},
            tmp_path / "v.medh5",
            masks={"liver": volumes["liver"]},
        )
        assert not validate_file(tmp_path / "v.medh5", level="integrity").errors


class TestDimensionality:
    """§3.6's table, walked row by row.

    The rows are the specification: 2-D radiographs, 3-D volumes, 4-D
    cine/DCE/CT under `time`, and multi-b-value DWI and multi-echo under
    `channel`.  NIfTI puts the last two in the same `dim[4]` as the first, so
    reading every 4-D series as time labelled every DWI gradient axis a time
    axis and handed it invented frame timings.
    """

    def _write(
        self,
        tmp_path,
        name,
        shape,
        *,
        intent=None,
        tr=None,
        units=None,
        toffset=None,
        bvals=None,
        sidecar=None,
    ):
        image = nib.Nifti1Image(np.zeros(shape, np.int16), AFFINE)
        if intent is not None:
            image.header.set_intent(intent)
        if units is not None:
            image.header.set_xyzt_units("mm", units)
        if tr is not None:
            image.header["pixdim"][4] = tr
        if toffset is not None:
            image.header["toffset"] = toffset
        source = tmp_path / f"{name}.nii.gz"
        nib.save(image, str(source))
        if bvals is not None:
            (tmp_path / f"{name}.bval").write_text(
                " ".join(str(b) for b in bvals), encoding="utf-8"
            )
        if sidecar is not None:
            (tmp_path / f"{name}.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
        return source

    @pytest.mark.parametrize(
        ("name", "shape", "options", "kinds"),
        [
            ("radiograph", (32, 40), {}, ("spatial", "spatial")),
            ("volume", (24, 20, 12), {}, ("spatial",) * 3),
            (
                "cine",
                (24, 20, 12, 5),
                {"tr": 2.5, "units": "sec"},
                ("time", "spatial", "spatial", "spatial"),
            ),
            (
                "dwi",
                (24, 20, 12, 3),
                {"bvals": [0, 1000, 2000]},
                ("channel", "spatial", "spatial", "spatial"),
            ),
            (
                "vector",
                (24, 20, 12, 3),
                {"intent": "vector"},
                ("channel", "spatial", "spatial", "spatial"),
            ),
            ("singleton", (24, 20, 12, 1), {"tr": 1.0}, ("spatial",) * 3),
            (
                "multiecho",
                (24, 20, 12, 4),
                {"sidecar": {"EchoTime": [0.005, 0.01, 0.015, 0.02]}},
                ("channel", "spatial", "spatial", "spatial"),
            ),
        ],
    )
    def test_S3_6_each_row_of_the_table(self, tmp_path, name, shape, options, kinds):
        source = self._write(tmp_path, name, shape, **options)
        from_nifti({"IM": source}, tmp_path / f"{name}.medh5")
        with medh5.open(tmp_path / f"{name}.medh5") as sample:
            assert sample.grids["ref"].axis_kinds == kinds

    def test_S3_6_a_DWI_carries_its_b_values(self, tmp_path):
        """§3.6 puts them in `acquisition` (§4.5); they are what the axis means."""
        source = self._write(tmp_path, "dwi", (24, 20, 12, 3), bvals=[0, 1000, 2000])
        report = from_nifti({"DWI": source}, tmp_path / "dwi.medh5")
        with medh5.open(tmp_path / "dwi.medh5") as sample:
            assert sample.images["DWI"].channel_names == ("b=0", "b=1000", "b=2000")
            assert sample.document.acquisition["DWI"]["b_values"] == [
                0.0,
                1000.0,
                2000.0,
            ]
        assert [n.severity for n in report.of_kind("axis_kinds")] == ["decision"]

    def test_S3_6_b_values_stay_with_the_file_they_came_from(self, tmp_path):
        """`_same_grid` compares the grid, which these volumes share by design.

        The b-values and the axis kind belong to the *file*, and reading them
        off the first geometry handed every DWI in the set the first one's
        gradients --- a silent corruption of what the channel axis means, with
        the conversion reporting success.
        """
        first = self._write(tmp_path, "dwiA", (24, 20, 12, 3), bvals=[0, 500, 1000])
        second = self._write(tmp_path, "dwiB", (24, 20, 12, 3), bvals=[0, 1500, 3000])

        from_nifti({"A": first, "B": second}, tmp_path / "two.medh5")
        with medh5.open(tmp_path / "two.medh5") as sample:
            assert sample.images["A"].channel_names == ("b=0", "b=500", "b=1000")
            assert sample.images["B"].channel_names == ("b=0", "b=1500", "b=3000")
            acquisition = sample.document.acquisition
            assert acquisition["A"]["b_values"] == [0.0, 500.0, 1000.0]
            assert acquisition["B"]["b_values"] == [0.0, 1500.0, 3000.0]

    def test_S3_6_volumes_that_disagree_about_their_axis_are_refused(self, tmp_path):
        """One grid states one set of `axis_kinds`, so they cannot both be right."""
        dwi = self._write(tmp_path, "dwi", (24, 20, 12, 3), bvals=[0, 500, 1000])
        cine = self._write(tmp_path, "cine", (24, 20, 12, 3), tr=2.0, units="sec")
        with pytest.raises(MEDH5ValidationError) as exc:
            from_nifti({"D": dwi, "C": cine}, tmp_path / "mix.medh5")
        # Uncoded: E110 is an invalid `axis_kinds` in a file, and nothing here
        # has one -- two inputs disagree about what the axis is.
        assert exc.value.code is None

    def test_S3_2_toffset_is_scaled_with_the_zoom(self, tmp_path):
        """`toffset` is in the header's own temporal unit, like `pixdim[4]`.

        Converting the zoom to milliseconds and leaving the offset in
        microseconds started the series a thousand frames from where it does.
        """
        source = self._write(
            tmp_path, "usec", (24, 20, 12, 3), units="usec", tr=500, toffset=1000
        )
        _, geometry = read_nifti(source)
        assert geometry["time_units"] == "ms"
        assert geometry["time_values"] == [1.0, 1.5, 2.0]

    def test_S3_6_an_RGB_vector_is_a_channel_axis(self, tmp_path):
        """Intents 2003/2004 state the answer as plainly as the numeric ones."""
        image = nib.Nifti1Image(np.zeros((24, 20, 12, 1, 3), np.int16), AFFINE)
        image.header["intent_code"] = 2003
        source = tmp_path / "rgb.nii.gz"
        nib.save(image, str(source))

        report = from_nifti({"IM": source}, tmp_path / "rgb.medh5")
        with medh5.open(tmp_path / "rgb.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == (3, 12, 20, 24)
            assert grid.axis_kinds == ("channel", "spatial", "spatial", "spatial")
        assert [n.severity for n in report.of_kind("axis_kinds")] == ["decision"]

    def test_S3_6_a_multi_echo_sidecar_states_the_axis(self, tmp_path):
        """The multi-echo row of §3.6, which no header field distinguishes.

        A multi-echo series carries no intent code and no temporal unit, so it
        fell through to the time guess and was imported as a time series with
        invented per-frame timings.  The BIDS sidecar is what the converters
        that write these files already emit, and it states the answer.
        """
        source = self._write(
            tmp_path,
            "megre",
            (24, 20, 12, 4),
            sidecar={"EchoTime": [0.005, 0.01, 0.015, 0.02], "RepetitionTime": 0.05},
        )
        report = from_nifti({"ME": source}, tmp_path / "megre.medh5")
        with medh5.open(tmp_path / "megre.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.axis_kinds == ("channel", "spatial", "spatial", "spatial")
            assert grid.time_values is None
            assert sample.images["ME"].channel_names == (
                "TE=0.005",
                "TE=0.01",
                "TE=0.015",
                "TE=0.02",
            )
            # §4.5 wants the DICOM keyword, and the echo times are what the
            # channel axis *means* --- exactly as b-values are for a DWI.
            assert sample.document.acquisition["ME"]["EchoTime"] == [
                0.005,
                0.01,
                0.015,
                0.02,
            ]
        assert [n.severity for n in report.of_kind("axis_kinds")] == ["decision"]

    def test_S3_6_a_scalar_echo_time_states_nothing_about_the_axis(self, tmp_path):
        """Per-volume is the whole test.

        Every MRI sidecar ever written carries a scalar `EchoTime`.  Reading
        the field's *presence* rather than its length would turn every cine and
        DCE series into a channel axis --- the same bug in the other direction.
        """
        source = self._write(
            tmp_path,
            "dce",
            (24, 20, 12, 4),
            sidecar={"EchoTime": 0.03, "RepetitionTime": 2.0},
        )
        _, geometry = read_nifti(source)
        assert geometry["leading_kind"] == "time"
        assert geometry["leading_stated"] is False

    def test_S3_6_a_list_that_is_not_per_volume_is_not_evidence(self, tmp_path):
        """Two echo times beside four frames do not describe those four frames."""
        source = self._write(
            tmp_path, "short", (24, 20, 12, 4), sidecar={"EchoTime": [0.005, 0.01]}
        )
        _, geometry = read_nifti(source)
        assert geometry["leading_kind"] == "time"
        assert geometry["leading_stated"] is False

    def test_S3_2_volume_timing_beats_a_ramp_rebuilt_from_pixdim(self, tmp_path):
        """BIDS states each volume's acquisition time; `pixdim[4]` assumes evenly
        spaced frames, which is the assumption sparse-sampled fMRI breaks."""
        source = self._write(
            tmp_path,
            "sparse",
            (24, 20, 12, 4),
            tr=2.0,
            units="sec",
            sidecar={"VolumeTiming": [0.0, 2.5, 6.0, 9.0]},
        )
        _, geometry = read_nifti(source)
        assert geometry["leading_kind"] == "time"
        assert geometry["time_values"] == [0.0, 2.5, 6.0, 9.0]
        assert geometry["time_measured"] is True

    def test_S3_6_a_sidecar_claiming_both_kinds_is_refused(self, tmp_path):
        """It says the axis is a channel axis and a time axis at once."""
        source = self._write(
            tmp_path,
            "both",
            (24, 20, 12, 4),
            sidecar={"EchoTime": [1, 2, 3, 4], "VolumeTiming": [0, 1, 2, 3]},
        )
        with pytest.raises(MEDH5ValidationError, match="at once"):
            read_nifti(source)
        # And the advice the refusal gives has to actually work.
        _, geometry = read_nifti(source, fourth_axis="time")
        assert geometry["leading_kind"] == "time"

    def test_S3_6_a_broken_sidecar_is_not_a_broken_nifti(self, tmp_path):
        source = self._write(tmp_path, "bad", (24, 20, 12, 4))
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        _, geometry = read_nifti(source)
        assert geometry["leading_kind"] == "time"

    def test_S3_6_an_unmarked_fourth_axis_is_a_guess(self, tmp_path):
        """`pixdim[4]` is 1.0 in a fresh header, so it states nothing on its own."""
        source = self._write(tmp_path, "plain", (24, 20, 12, 3))
        report = from_nifti({"IM": source}, tmp_path / "plain.medh5")
        with medh5.open(tmp_path / "plain.medh5") as sample:
            assert sample.grids["ref"].axis_kinds[0] == "time"
        assert [n.severity for n in report.of_kind("axis_kinds")] == ["guess"]

    def test_S3_6_the_caller_can_settle_it(self, tmp_path):
        source = self._write(tmp_path, "echo", (24, 20, 12, 1, 4))
        from_nifti({"IM": source}, tmp_path / "echo.medh5", fourth_axis="channel")
        with medh5.open(tmp_path / "echo.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.shape == (4, 12, 20, 24), "the singleton axis carried nothing"
            assert grid.axis_kinds == ("channel", "spatial", "spatial", "spatial")

    def test_an_unknown_fourth_axis_is_refused(self, tmp_path):
        source = self._write(tmp_path, "bad", (24, 20, 12, 3))
        with pytest.raises(MEDH5ValidationError, match="fourth_axis"):
            from_nifti({"IM": source}, tmp_path / "bad.medh5", fourth_axis="vibes")


class TestGrouping:
    def test_S3_7_studies_of_one_subject_become_one_sample(self):
        groups = group_by_subject(
            [
                Occasion("s2", "p1", "20260401"),
                Occasion("s1", "p1", "20260101"),
                Occasion("s3", "p2", "20260201"),
            ]
        )
        assert [g.subject_id for g in groups] == ["p1", "p2"]
        assert [o.key for o in groups[0].occasions] == ["s1", "s2"]
        assert groups[0].days_from_baseline() == [0, 90]
        assert groups[0].ordered_by == "date"
        assert groups[0].is_longitudinal

    def test_identity_is_never_inferred(self):
        report = ConversionReport()
        groups = group_by_subject([Occasion("s1"), Occasion("s2")], report=report)
        assert len(groups) == 2
        assert all(g.subject_id.startswith("study:") for g in groups)
        assert report.warnings

    def test_mtime_ordering_is_reported_as_a_guess(self):
        report = ConversionReport()
        groups = group_by_subject(
            [
                Occasion("b", "p1", order_hint=2.0),
                Occasion("a", "p1", order_hint=1.0),
            ],
            report=report,
        )
        assert [o.key for o in groups[0].occasions] == ["a", "b"]
        assert groups[0].ordered_by == "order_hint"
        assert report.guesses

    def test_without_dates_or_hints_the_order_is_kept_and_flagged(self):
        report = ConversionReport()
        groups = group_by_subject(
            [Occasion("b", "p1"), Occasion("a", "p1")], report=report
        )
        assert [o.key for o in groups[0].occasions] == ["b", "a"]
        assert report.guesses

    def test_study_mode_keeps_every_occasion_apart(self):
        groups = group_by_subject(
            [Occasion("s1", "p1"), Occasion("s2", "p1")], mode="study"
        )
        assert len(groups) == 2
        assert not groups[0].is_longitudinal

    def test_missing_dates_give_no_intervals(self):
        group = SubjectGroup("p", [Occasion("a"), Occasion("b")])
        assert group.days_from_baseline() == [None, None]
        assert group.timepoint_ids() == ["tp0", "tp1"]
        assert group.to_json()["subject_id"] == "p"

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="grouping mode"):
            group_by_subject([], mode="vibes")


class TestNnunet:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> Path:
        root = tmp_path / "Dataset001_Test"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (12, 10, 6)
        for case in ("CASE_001", "CASE_002"):
            rng = np.random.default_rng(len(case))
            for channel in range(2):
                nib.save(
                    nib.Nifti1Image(
                        rng.integers(0, 500, shape).astype(np.int16), np.eye(4)
                    ),
                    str(root / "imagesTr" / f"{case}_{channel:04d}.nii.gz"),
                )
            labels = np.zeros(shape, np.uint8)
            labels[2:8, 2:6, 1:4] = 1
            labels[4:6, 3:5, 2:3] = 2
            nib.save(
                nib.Nifti1Image(labels, np.eye(4)),
                str(root / "labelsTr" / f"{case}.nii.gz"),
            )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "T1", "1": "FLAIR"},
                    "labels": {
                        "background": 0,
                        "edema": 1,
                        "enhancing": 2,
                        "whole_tumor": [1, 2],
                    },
                    "numTraining": 2,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_S5_2_nnunet_ids_are_kept(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        report = from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            ids = {c.key: c.id for c in sample.label_set}
            assert ids["edema"] == 1
            assert ids["enhancing"] == 2
        assert report.of_kind("label_ids")
        assert report.of_kind("background"), "background 0 is reserved (§5.3)"

    def test_S7_regions_become_classes_over_their_components(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            seg = sample.annotations["seg"]
            edema = seg.dense(["edema"])[0]
            enhancing = seg.dense(["enhancing"])[0]
            whole = seg.dense(["whole_tumor"])[0]
            assert np.array_equal(whole, edema | enhancing)
            assert sample.label_set["edema"].parents == (
                sample.label_set["whole_tumor"].id,
            )

    def test_S11_3_an_nnunet_label_volume_is_exhaustive(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            assert sample.annotations["seg"].is_fully_covered

    def test_the_round_trip_reproduces_the_dataset(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2, read_dataset_json, to_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        cases = sorted((tmp_path / "out").glob("*.medh5"))
        to_nnunetv2(cases, tmp_path / "back")
        exported = read_dataset_json(tmp_path / "back" / "Dataset001_medh5")
        assert exported["labels"] == {
            "background": 0,
            "edema": 1,
            "enhancing": 2,
            "whole_tumor": [1, 2],
        }
        assert exported["channel_names"] == {"0": "T1", "1": "FLAIR"}
        for name in ("labelsTr/CASE_001.nii.gz", "imagesTr/CASE_001_0000.nii.gz"):
            original = nib.load(str(dataset / name))
            back = nib.load(str(tmp_path / "back" / "Dataset001_medh5" / name))
            assert np.array_equal(
                np.asanyarray(original.dataobj), np.asanyarray(back.dataobj)
            ), name
            assert np.allclose(original.affine, back.affine), name

    def test_dataset_json_is_stashed_verbatim(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            stashed = sample.document.extra["nnunetv2"]
            assert stashed == json.loads(
                (dataset / "dataset.json").read_text(encoding="utf-8")
            )

    def test_a_malformed_dataset_json_is_named(self, tmp_path):
        from medh5.io.nnunetv2 import read_dataset_json

        (tmp_path / "dataset.json").write_text(
            json.dumps({"labels": {}}), encoding="utf-8"
        )
        with pytest.raises(MEDH5ValidationError, match="missing"):
            read_dataset_json(tmp_path)
        (tmp_path / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"1": "T1"},
                    "labels": {"a": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MEDH5ValidationError, match="0\\.\\.0"):
            read_dataset_json(tmp_path)
        with pytest.raises(MEDH5ValidationError, match="not found"):
            read_dataset_json(tmp_path / "nope")

    def test_S3_2_a_channel_on_another_grid_is_refused(self, dataset, tmp_path):
        """A second channel is written onto the first one's grid, so it has to
        share it.  Same shape, different spacing: nothing in the array reveals
        that these are different volumes of the patient."""
        from medh5.io.nnunetv2 import from_nnunetv2

        odd = np.diag([2.0, 2.0, 2.0, 1.0])
        odd[:3, 3] = [50.0, 0.0, 0.0]
        nib.save(
            nib.Nifti1Image(np.zeros((12, 10, 6), np.int16), odd),
            str(dataset / "imagesTr" / "CASE_001_0001.nii.gz"),
        )
        with pytest.raises(MEDH5ValidationError, match="spacing"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])

    def test_S3_2_a_label_volume_on_another_grid_is_refused(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        shifted = np.eye(4)
        shifted[:3, 3] = [0.0, 0.0, 9.0]
        nib.save(
            nib.Nifti1Image(np.zeros((12, 10, 6), np.uint8), shifted),
            str(dataset / "labelsTr" / "CASE_001.nii.gz"),
        )
        with pytest.raises(MEDH5ValidationError, match="origin"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])

    def test_a_label_named_with_spaces_survives_the_round_trip(self, tmp_path):
        """`dataset.json` names are free text; the label set key is sanitised
        from them.  Matching classes back by name therefore finds nothing for
        any dataset that capitalises, and the export silently wrote an
        all-background volume.  Classes are matched by id instead."""
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        root = tmp_path / "Dataset002_Named"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (8, 8, 4)
        nib.save(
            nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
            str(root / "imagesTr" / "CASE_0000.nii.gz"),
        )
        volume = np.zeros(shape, np.uint8)
        volume[1:4, 1:4, 1:3] = 1
        volume[5:7, 5:7, 1:3] = 2
        nib.save(
            nib.Nifti1Image(volume, np.eye(4)),
            str(root / "labelsTr" / "CASE.nii.gz"),
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "Tumour Core": 1, "GTV": 2},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        imported = tmp_path / "imported"
        from_nnunetv2(root, imported)
        to_nnunetv2([imported / "CASE.medh5"], tmp_path / "back")
        written = np.asarray(
            nib.load(
                str(tmp_path / "back" / "Dataset001_medh5" / "labelsTr" / "CASE.nii.gz")
            ).dataobj
        )
        assert int((written == 1).sum()) == int((volume == 1).sum())
        assert int((written == 2).sum()) == int((volume == 2).sum())

    def test_a_class_the_sample_lacks_is_refused_not_dropped(self, tmp_path):
        from medh5.io.nnunetv2 import _labelmap_for

        class _Stub:
            ann_id = "seg"
            class_ids = (1,)
            spatial_shape = (2, 2, 2)

            def dense(self, ids):
                return np.zeros((1, 2, 2, 2), bool)

            def resolve_class(self, key):
                raise KeyError(key)

        with pytest.raises(MEDH5ValidationError, match="carries no class"):
            _labelmap_for(_Stub(), {"background": 0, "kidney": 1, "spleen": 7})

    def test_a_missing_channel_is_named(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        (dataset / "imagesTr" / "CASE_001_0001.nii.gz").unlink()
        with pytest.raises(MEDH5ValidationError, match="channel 1"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])


class TestMigrate:
    @pytest.fixture
    def legacy(self, tmp_path: Path) -> list[Path]:
        shape = (8, 12, 16)
        rng = np.random.default_rng(3)
        liver = np.zeros(shape, bool)
        liver[1:6, 2:9, 3:12] = True
        lesion = np.zeros(shape, bool)
        lesion[2:4, 4:6, 5:8] = True
        paths = []
        for index, patient in enumerate(("PAT-A", "PAT-A", "PAT-B")):
            path = tmp_path / f"old_{index}.medh5"
            write_legacy_sample(
                path,
                images={"CT": rng.integers(-1000, 1500, shape).astype(np.int16)},
                seg={"liver": liver, "lesion": lesion},
                bboxes=np.array([[[2, 6], [4, 9], [5, 11]]], dtype=np.int32),
                bbox_labels=["lesion"],
                bbox_scores=np.array([0.9], np.float32),
                spacing=[2.0, 0.8, 0.9],
                origin=[1.0, 2.0, 3.0],
                coord_system="LPS",
                label=1,
                label_name="lesion",
                extra={
                    "patient_id": patient,
                    "study_date": f"2026-0{index + 1}-15",
                    "review": {
                        "reviewer": "RAD-07",
                        "status": "approved",
                        "date": "2026-03-01",
                    },
                },
            )
            paths.append(path)
        return paths

    def test_appendix_B_one_file_becomes_one_sample(self, legacy, tmp_path):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.timepoints.ids == ("tp0",)
            assert sample.grids["ref"].timepoint == "tp0"
            assert {"core", "seg", "det", "cls"} <= sample.profiles
            assert sample.annotations["seg"].kind in ("layers", "labelmap", "bitmask")
        assert report.of_kind("encoding")
        assert not validate_file(tmp_path / "new.medh5").errors

    def test_S8_1_boxes_shift_by_half_a_voxel_and_round_trip_to_slices(
        self, legacy, tmp_path
    ):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            boxes = sample.annotations["boxes"]
            assert np.allclose(boxes.boxes[0], [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]])
            assert boxes.as_slices()[0] == (
                slice(2, 6),
                slice(4, 9),
                slice(5, 11),
            )
        note = report.of_kind("box_convention")[0]
        assert note.detail["shift"] == -0.5

    def test_masks_survive_and_coverage_is_flagged(self, legacy, tmp_path):
        from medh5.io.legacy import migrate, read_legacy

        migrate(legacy[0], tmp_path / "new.medh5")
        original = read_legacy(legacy[0])
        with medh5.open(tmp_path / "new.medh5") as sample:
            seg = sample.annotations["seg"]
            for name, mask in original.seg.items():
                assert np.array_equal(seg.dense([name])[0], mask), name

    def test_review_becomes_provenance_and_quality(self, legacy, tmp_path):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.document.quality["seg"].status == "approved"
            types = [a.type for a in sample.document.provenance.activities]
            assert "review" in types
            assert sample.document.extra["legacy"]["patient_id"] == "PAT-A"
        assert report.of_kind("review")

    def test_S3_7_subject_grouping_merges_a_patients_files(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        report = migrate_paths(
            legacy,
            tmp_path / "out",
            group_by="subject",
            subject_key="extra.patient_id",
        )
        written = sorted((tmp_path / "out").glob("*.medh5"))
        assert [p.stem for p in written] == ["pat-a", "pat-b"]
        with medh5.open(tmp_path / "out" / "pat-a.medh5") as sample:
            assert sample.timepoints.ids == ("tp0", "tp1")
            assert sample.timepoints["tp1"].days_from_baseline == 31
            assert sorted(sample.images) == ["CT_tp0", "CT_tp1"]
        assert report.of_kind("instance_ids"), "merging must say it did not join ids"

    def test_the_default_is_one_sample_per_file(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        migrate_paths(legacy, tmp_path / "out")
        assert len(list((tmp_path / "out").glob("*.medh5"))) == 3

    def test_subject_grouping_without_a_key_warns(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        report = migrate_paths(legacy, tmp_path / "out", group_by="subject")
        assert report.warnings
        assert any("subject-key" in w.message for w in report.warnings)

    def test_the_label_set_is_minted_once_for_the_cohort(self, legacy, tmp_path):
        from medh5.io.legacy import build_label_set, load_sidecar, write_sidecar

        report = ConversionReport()
        label_set = build_label_set(legacy, report=report)
        assert {c.key for c in label_set} == {"liver", "lesion"}
        sidecar = write_sidecar(label_set, tmp_path / "labels.json")
        assert {c.key for c in load_sidecar(sidecar)} == {"liver", "lesion"}
        assert report.of_kind("label_set")

    def test_an_unreadable_file_is_reported_not_fatal(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        report = migrate_paths([*legacy, broken], tmp_path / "out")
        assert report.warnings
        assert len(list((tmp_path / "out").glob("*.medh5"))) == 3


class TestLegacyReader:
    """1.0 ships a reader for the 0.x layout, not an implementation of it."""

    def test_the_0x_package_is_gone(self):
        with pytest.raises(ImportError):
            import medh5.legacy  # noqa: F401

    def test_a_1_0_file_is_refused_as_0_x(self, sample_path):
        from medh5.io._legacy_reader import is_legacy, read_sample

        with pytest.raises(MEDH5SchemaError, match="1.0 file"):
            read_sample(sample_path)
        assert not is_legacy(sample_path)

    def test_a_non_hdf5_file_is_refused(self, tmp_path):
        from medh5.io._legacy_reader import is_legacy, read_sample

        path = tmp_path / "junk.medh5"
        path.write_bytes(b"not hdf5")
        with pytest.raises(MEDH5FileError):
            read_sample(path)
        assert not is_legacy(path)

    def test_a_future_0_x_schema_version_is_refused(self, tmp_path):
        from medh5.io._legacy_reader import read_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["schema_version"] = "2"
        with pytest.raises(MEDH5SchemaError, match="schema version"):
            read_meta(path)

    def test_the_file_beats_its_own_flags(self, tmp_path):
        """0.x denormalised `has_seg`/`seg_names`, and they could drift."""
        from medh5.io._legacy_reader import read_sample

        mask = np.zeros((2, 3, 4), bool)
        mask[0, 1, 2] = True
        path = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((2, 3, 4), np.int16)},
            seg={"liver": mask},
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["has_seg"] = False
            del handle.attrs["seg_names"]
        sample = read_sample(path)
        assert list(sample.seg) == ["liver"]
        assert sample.meta.seg_names == ["liver"]

    def test_geometry_and_extra_round_trip(self, tmp_path):
        from medh5.io._legacy_reader import read_sample

        direction = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        path = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((2, 3, 4), np.int16)},
            spacing=[2.0, 0.8, 0.9],
            origin=[1.0, 2.0, 3.0],
            direction=direction,
            coord_system="LPS",
            patch_size=[2, 2, 2],
            label=1,
            label_name="lesion",
            extra={"patient_id": "PAT-A"},
        )
        sample = read_sample(path)
        assert sample.meta.spatial.direction == direction
        assert sample.meta.spatial.spacing == [2.0, 0.8, 0.9]
        assert sample.meta.patch_size == [2, 2, 2]
        assert sample.meta.label == 1
        assert sample.meta.extra["patient_id"] == "PAT-A"

    def test_a_malformed_direction_is_named(self, tmp_path):
        from medh5.io._legacy_reader import read_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle["images"].attrs["direction"] = np.zeros(4, np.float64)
        with pytest.raises(MEDH5SchemaError, match="4 element"):
            read_meta(path)

    def test_malformed_extra_is_named(self, tmp_path):
        from medh5.io._legacy_reader import read_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["extra"] = "{not json"
        with pytest.raises(MEDH5SchemaError, match="not JSON"):
            read_meta(path)


class TestDicom:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> dict[str, Any]:
        """One patient, two studies; the first holds a CT and a co-registered PT."""
        from pydicom.uid import generate_uid

        from tests.v1.conftest import write_dicom_series

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
        from tests.v1.conftest import write_dicom_series

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

        from medh5.errors import MEDH5Error
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

        from medh5.errors import MEDH5Error
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

        from medh5.errors import MEDH5Error
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
        from tests.v1.conftest import write_dicom_series

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


class TestDicomSeg:
    @pytest.fixture
    def prepared(self, tmp_path: Path) -> dict[str, Any]:
        """A one-series sample plus two overlapping masks on its grid."""
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.labels.labelset import LabelClass, LabelSet
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root,
            patient_id="PSEUDO-002",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        path = tmp_path / "case.medh5"
        from_dicom(root, path, group_by="study")
        with medh5.open(path) as sample:
            grid_id = sorted(sample.grids)[0]
            shape = sample.grids[grid_id].spatial_shape
        liver = np.zeros(shape, bool)
        liver[1:5, 2:12, 3:16] = True
        lesion = np.zeros(shape, bool)
        lesion[2:4, 5:8, 6:10] = True  # entirely inside the liver
        with medh5.amend(path) as writer:
            writer.label_set(
                LabelSet(
                    "d",
                    version="1.0.0",
                    classes=[
                        LabelClass(1, "liver", "Liver"),
                        LabelClass(3, "lesion", "Lesion"),
                    ],
                )
            )
            writer.add_segmentation("organs", grid=grid_id, masks={1: liver, 3: lesion})
        return {
            "path": path,
            "series": series,
            "liver": liver,
            "lesion": lesion,
            "grid": grid_id,
        }

    def test_S7_overlapping_segments_survive_the_round_trip(self, prepared, tmp_path):
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        report = from_dicom_seg(out, prepared["path"], ann_id="imported")
        with medh5.open(prepared["path"]) as sample:
            imported = sample.annotations["imported"]
            assert np.array_equal(imported.dense(["liver"])[0], prepared["liver"])
            assert np.array_equal(imported.dense(["lesion"])[0], prepared["lesion"])
            overlap = imported.dense(["liver"])[0] & imported.dense(["lesion"])[0]
            assert overlap.any(), "the lesion is inside the liver and must stay there"
        assert report.of_kind("overlap")

    def test_S5_segments_are_matched_by_label_not_by_number(self, prepared, tmp_path):
        """DICOM numbers segments 1..N; the sample's ids are 1 and 3."""
        from medh5.io.dicom_seg import from_dicom_seg, read_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        _, geometry = read_dicom_seg(out)
        assert sorted(geometry["segments"]) == [1, 2]
        report = from_dicom_seg(out, prepared["path"], ann_id="matched")
        with medh5.open(prepared["path"]) as sample:
            assert sample.annotations["matched"].class_ids == (1, 3)
        assert report.of_kind("segment_mapping")

    def test_frames_are_placed_by_geometry(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        volumes, geometry = read_dicom_seg(out)
        assert geometry["shape"] == prepared["liver"].shape
        assert np.isclose(geometry["spacing"][0], 2.5)
        assert np.array_equal(volumes[1], prepared["liver"])

    def test_a_segment_the_label_set_lacks_is_refused(self, prepared, tmp_path):
        import pydicom

        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        dataset = pydicom.dcmread(str(out))
        dataset.SegmentSequence[0].SegmentLabel = "pancreas"
        dataset.save_as(str(out), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="pancreas"):
            from_dicom_seg(out, prepared["path"], ann_id="x")

    def test_a_non_seg_file_is_refused(self, prepared):
        from medh5.io.dicom_seg import read_dicom_seg

        with pytest.raises(MEDH5ValidationError, match="not 'SEG'"):
            read_dicom_seg(prepared["series"]["paths"][0])

    def test_a_seg_from_another_reconstruction_is_refused(self, prepared, tmp_path):
        """A SEG only means anything against the grid it was drawn on."""
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        other = tmp_path / "other.medh5"
        with medh5.create(other, sample_id="other", codec="portable") as writer:
            writer.add_grid("g", shape=(4, 8, 10), spacing=(1.0, 1.0, 1.0))
            writer.add_image(
                "CT", np.zeros((4, 8, 10), dtype=np.int16), grid="g", modality="CT"
            )
        with pytest.raises(MEDH5ValidationError, match="different reconstruction"):
            from_dicom_seg(out, other, ann_id="x", grid="g")


class TestRtstruct:
    @pytest.fixture
    def prepared(self, tmp_path: Path) -> dict[str, Any]:
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root,
            patient_id="PSEUDO-003",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        path = tmp_path / "case.medh5"
        from_dicom(root, path, group_by="study")
        return {"path": path, "series": series, "root": root}

    def _rtstruct(
        self, prepared, tmp_path, *, hole: bool = True, hole_first: bool = False
    ) -> Path:
        """A square ROI on two slices, optionally with a square hole.

        ``hole_first`` lists the inner contour before its outer, which DICOM
        permits and no ordering rule forbids.
        """
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        reference = pydicom.dcmread(prepared["series"]["paths"][0])
        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[sorted(sample.grids)[0]]

        def square(centre_y: int, centre_x: int, radius: int, plane: int):
            corners = [
                (centre_y - radius, centre_x - radius),
                (centre_y - radius, centre_x + radius),
                (centre_y + radius, centre_x + radius),
                (centre_y + radius, centre_x - radius),
            ]
            return grid.index_to_world(
                np.array([[plane, y, x] for y, x in corners], dtype=float)
            )

        structure = Dataset()
        structure.file_meta = FileMetaDataset()
        structure.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        structure.SOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
        structure.Modality = "RTSTRUCT"
        structure.StructureSetLabel = "test"
        structure.PatientID = reference.PatientID
        structure.PatientName = reference.PatientName
        structure.StudyInstanceUID = reference.StudyInstanceUID
        structure.SeriesInstanceUID = generate_uid()
        frame = Dataset()
        frame.FrameOfReferenceUID = reference.FrameOfReferenceUID
        structure.ReferencedFrameOfReferenceSequence = [frame]
        roi = Dataset()
        roi.ROINumber = 1
        roi.ROIName = "liver"
        roi.ReferencedFrameOfReferenceUID = frame.FrameOfReferenceUID
        structure.StructureSetROISequence = [roi]
        item = Dataset()
        item.ReferencedROINumber = 1
        item.ROIDisplayColor = [200, 90, 70]
        item.ContourSequence = []
        for plane in (2, 3):
            shapes = [square(8, 10, 5, plane)]
            if hole:
                inner = square(8, 10, 2, plane)
                shapes = [inner, *shapes] if hole_first else [*shapes, inner]
            for points in shapes:
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = len(points)
                contour.ContourData = [float(v) for v in points.reshape(-1)]
                item.ContourSequence.append(contour)
        structure.ROIContourSequence = [item]
        out = tmp_path / "rt.dcm"
        structure.save_as(str(out), enforce_file_format=True)
        return out

    def test_S8_6_contours_are_stored_as_contours(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois")
        with medh5.open(prepared["path"]) as sample:
            annotation = sample.annotations["rois"]
            assert annotation.kind == "contours"
            assert annotation.space == "world"
            assert len(list(annotation.polygons())) == 4
            assert {c.key for c in sample.label_set} == {"liver"}
        assert report.of_kind("contours")

    def test_S8_6_a_contour_inside_another_is_a_hole(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=True)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            roles = [p.role for p in sample.annotations["rois"].polygons()]
            assert roles.count("hole") == 2
            mask = sample.annotations["rois_mask"].dense(["liver"])[0]
            assert mask[2, 5, 6], "the ring is filled"
            assert not mask[2, 8, 10], "the hole is not"
        assert report.of_kind("holes")

    def test_S8_6_a_hole_listed_before_its_outer_is_still_a_hole(
        self, prepared, tmp_path
    ):
        """DICOM does not order outer contours before the holes they enclose.

        Subtracting a hole from a mask whose outer has not been drawn yet does
        nothing, and the outer then fills the cavity back in --- a conversion
        that turns holes into foreground without saying so.
        """
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=True, hole_first=True)
        from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            roles = [p.role for p in sample.annotations["rois"].polygons()]
            assert roles.count("hole") == 2, "containment, not order, makes a hole"
            mask = sample.annotations["rois_mask"].dense(["liver"])[0]
            assert mask[2, 5, 6], "the ring is filled"
            assert not mask[2, 8, 10], "and the hole was not filled back in"

    def test_rasterization_is_opt_in_and_recorded(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=False)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois")
        with medh5.open(prepared["path"]) as sample:
            assert "rois_mask" not in sample.annotations
        assert not report.of_kind("rasterization")

        report = from_rtstruct(rt, prepared["path"], ann_id="raster", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            derived = sample.annotations["raster_mask"]
            # §6.2: `derived_from` holds annotation ids, not paths.
            assert derived.header.derived_from == ("raster",)
            activity = sample.document.provenance.activity(derived.prov)
            assert "even-odd" in activity.params["rule"]
        assert report.guesses

    def test_the_round_trip_preserves_the_contours(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct, read_rtstruct, to_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=False)
        original, _ = read_rtstruct(rt)
        from_rtstruct(rt, prepared["path"], ann_id="rois")
        back = to_rtstruct(
            prepared["path"],
            "rois",
            prepared["series"]["paths"],
            tmp_path / "back.dcm",
        )
        recovered, meta = read_rtstruct(back)
        assert meta["names"] == {1: "liver"}
        assert len(recovered[1]) == len(original[1])
        for first, second in zip(
            sorted(original[1], key=lambda a: a[0, 0].item()),
            sorted(recovered[1], key=lambda a: a[0, 0].item()),
            strict=True,
        ):
            assert np.allclose(first, second, atol=1e-3)

    def test_exporting_a_mask_is_refused(self, prepared, tmp_path):
        from medh5.io.rtstruct import to_rtstruct

        rt = self._rtstruct(prepared, tmp_path)
        from medh5.io.rtstruct import from_rtstruct

        from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with pytest.raises(MEDH5ValidationError, match="polygons"):
            to_rtstruct(
                prepared["path"],
                "rois_mask",
                prepared["series"]["paths"],
                tmp_path / "x.dcm",
            )

    def test_a_non_rtstruct_is_refused(self, prepared):
        from medh5.io.rtstruct import read_rtstruct

        with pytest.raises(MEDH5ValidationError, match="not 'RTSTRUCT'"):
            read_rtstruct(prepared["series"]["paths"][0])


class TestConverterDiagnosticCodes:
    """A converter refusal about its *input* carries no diagnostic code.

    §15.2's table describes conditions found in a MEDH5 file. A NIfTI volume or
    a DICOM series is not one yet, so a code applied to it tells anything
    branching on `exc.code` an untrue story --- an irregular DICOM stack read as
    a non-positive grid spacing, a tilted 2-D plane as a non-orthonormal
    `direction`, a modality-LUT disagreement as malformed `channel_names`.

    A refusal about the sample being written or targeted is different and keeps
    its code: a SEG naming a grid the sample does not have really is `E101`, and
    a class absent from the sample's label set really is `E402`.

    This mistake reached six separate sites before it was found, one at a time,
    so the allow-list below is exhaustive: a new coded refusal in `medh5.io` has
    to be added here deliberately, with the reason it is about the sample rather
    than the input.
    """

    ALLOWED = {
        ("dicom_seg.py", "E101"),  # SEG names no grid the sample has
        ("dicom_seg.py", "E402"),  # segment absent from the sample's label set
        ("dicom_seg.py", "E405"),  # SEG shape vs. the target grid's
        ("nifti.py", "E402"),  # mask name absent from the sample's label set
        ("nifti.py", "E405"),  # mask shape vs. the target grid's
        ("nnunetv2.py", "E402"),  # class absent from the annotation
        ("rtstruct.py", "E101"),  # RTSTRUCT names no grid the sample has
        ("rtstruct.py", "E402"),  # ROI absent from the sample's label set
        ("rtstruct.py", "E401"),  # the sample's annotation is the wrong kind
        ("rtstruct.py", "E414"),  # the sample's annotation has no usable space
    }

    def test_no_converter_refusal_borrows_a_format_code(self):
        import re

        import medh5.io

        root = Path(medh5.io.__file__).parent
        found = {
            (path.name, code)
            for path in sorted(root.glob("*.py"))
            for code in re.findall(r'code="(E\d{3})"', path.read_text(encoding="utf-8"))
        }
        assert found <= self.ALLOWED, (
            "new coded refusal(s) in medh5.io: "
            f"{sorted(found - self.ALLOWED)}. If the refusal describes the "
            "MEDH5 sample, add it to ALLOWED with a reason; if it describes the "
            "converter's input, leave it uncoded."
        )


class TestLazyImports:
    def test_converters_resolve_without_importing_medh5_io_eagerly(self):
        import medh5.io as io

        assert callable(io.from_nifti)
        assert callable(io.migrate)
        assert "from_dicom" in dir(io)
        with pytest.raises(AttributeError, match="from_parquet"):
            _ = io.from_parquet

    def test_importing_medh5_does_not_import_the_optional_stacks(self):
        """`import medh5` must not need nibabel, pydicom or highdicom."""
        import subprocess
        import sys

        script = (
            "import sys; import medh5; "
            "assert 'nibabel' not in sys.modules; "
            "assert 'pydicom' not in sys.modules; "
            "assert 'highdicom' not in sys.modules; print('clean')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert "clean" in result.stdout


class TestDicomSegExtras:
    def test_a_seg_into_a_sample_without_a_label_set_mints_one(self, tmp_path):
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg
        from medh5.labels.labelset import LabelClass, LabelSet
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root, patient_id="p", study_uid=generate_uid(), study_date="20260101"
        )
        source = tmp_path / "src.medh5"
        from_dicom(root, source, group_by="study")
        with medh5.open(source) as sample:
            grid_id = sorted(sample.grids)[0]
            shape = sample.grids[grid_id].spatial_shape
        mask = np.zeros(shape, bool)
        mask[1:4, 2:8, 3:9] = True
        with medh5.amend(source) as writer:
            writer.label_set(
                LabelSet(
                    "d", version="1.0.0", classes=[LabelClass(7, "liver", "Liver")]
                )
            )
            writer.add_segmentation("organs", grid=grid_id, masks={7: mask})
        seg = to_dicom_seg(source, "organs", series["paths"], tmp_path / "s.dcm")

        blank = tmp_path / "blank.medh5"
        from_dicom(root, blank, group_by="study")
        report = from_dicom_seg(seg, blank, ann_id="imported")
        with medh5.open(blank) as sample:
            minted = sample.label_set
            assert [c.key for c in minted] == ["liver"]
            assert np.array_equal(
                sample.annotations["imported"].dense(["liver"])[0], mask
            )
        note = report.of_kind("label_set")[0]
        assert note.detail["bound"] == 1, "the SEG's coded concept came through"

    def test_a_fractional_seg_becomes_a_probmap(self, tmp_path):
        pytest.importorskip("highdicom")
        import highdicom as hd
        import pydicom
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg, read_dicom_seg
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root, patient_id="p", study_uid=generate_uid(), study_date="20260101"
        )
        sample_path = tmp_path / "case.medh5"
        from_dicom(root, sample_path, group_by="study")
        with medh5.open(sample_path) as sample:
            shape = sample.grids[sorted(sample.grids)[0]].spatial_shape

        datasets = [pydicom.dcmread(p) for p in series["paths"]]
        normal = np.cross([0, 0, 1], [0, 1, 0])
        datasets.sort(
            key=lambda d: float(
                np.dot([float(v) for v in d.ImagePositionPatient], normal)
            )
        )
        probabilities = np.zeros((*shape, 1), dtype=np.float32)
        probabilities[1:4, 2:8, 3:9, 0] = 0.6
        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label="liver",
            segmented_property_category=hd.sr.CodedConcept(
                "91723000", "SCT", "Anatomical Structure"
            ),
            segmented_property_type=hd.sr.CodedConcept("10200004", "SCT", "Liver"),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="test",
                version="1",
                family=hd.sr.CodedConcept("123037004", "SCT", "Body Structure"),
            ),
        )
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=probabilities,
            segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
            segment_descriptions=[description],
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="test",
            manufacturer_model_name="test",
            software_versions="1",
            device_serial_number="0",
            omit_empty_frames=False,
        )
        out = tmp_path / "frac.dcm"
        segmentation.save_as(str(out))

        volumes, geometry = read_dicom_seg(out)
        assert geometry["fractional"]
        assert 0.0 < float(volumes[1].max()) <= 1.0
        report = from_dicom_seg(out, sample_path, ann_id="prob")
        with medh5.open(sample_path) as sample:
            annotation = sample.annotations["prob"]
            assert annotation.kind == "probmap"
            assert [c.key for c in sample.label_set] == ["liver"]
            assert sample.label_set["liver"].codes[0].code == "10200004"
        assert report.of_kind("fractional")
