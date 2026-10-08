"""NIfTI import and export (``medh5.io.nifti``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.validate import validate_file
from tests.kits import Organs

nib = pytest.importorskip("nibabel")

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


class TestF05NiftiRescale:
    """§4.2: `scl_slope`/`scl_inter` are the file's rescale, stored, not applied."""

    SLOPE, INTERCEPT = 2.0, -1024.0

    def _scaled_nifti(
        self, path: Path, stored: Any, slope: float, inter: float
    ) -> None:
        """A NIfTI whose header scales its `int16` voxels.

        nibabel consumes the header fields on load --- they move onto the
        array proxy and `scl_slope` reads back NaN --- so the check that the
        file carries the scale asks the proxy, as the converter does.
        """
        nib = pytest.importorskip("nibabel")
        image = nib.Nifti1Image(stored, np.eye(4))
        image.header.set_slope_inter(slope, inter)
        nib.save(image, str(path))
        loaded = nib.load(str(path))
        assert float(loaded.dataobj.slope) == slope
        assert float(loaded.dataobj.inter) == inter

    def test_S4_2_the_scale_is_recorded_and_the_dtype_kept(
        self, tmp_path: Path
    ) -> None:
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti

        stored = (
            np.random.default_rng(2).integers(0, 2000, (12, 10, 8)).astype(np.int16)
        )
        source = tmp_path / "ct.nii"
        self._scaled_nifti(source, stored, self.SLOPE, self.INTERCEPT)
        report = from_nifti({"CT": source}, tmp_path / "ct.medh5")
        assert report.of_kind("value_scale"), "the decision is recorded"
        expected = nib.load(str(source)).get_fdata()
        with medh5.open(tmp_path / "ct.medh5") as sample:
            image = sample.images["CT"]
            assert image.dtype == np.int16
            assert image.rescale == (self.SLOPE, self.INTERCEPT)
            physical = image.read(physical=True)
        # Back to NIfTI (x, y, z) order for the comparison.
        assert np.allclose(np.transpose(physical, (2, 1, 0)), expected)

    def test_S4_2_an_unscaled_file_records_no_rescale(self, tmp_path: Path) -> None:
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti

        stored = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        source = tmp_path / "plain.nii.gz"
        nib.save(nib.Nifti1Image(stored, np.eye(4)), str(source))
        report = from_nifti({"CT": source}, tmp_path / "plain.medh5")
        assert not report.of_kind("value_scale")
        with medh5.open(tmp_path / "plain.medh5") as sample:
            assert sample.images["CT"].rescale == (1.0, 0.0)
            assert not sample.images["CT"].is_rescaled

    def test_S4_2_a_scaled_mask_is_thresholded_after_the_scale(
        self, tmp_path: Path
    ) -> None:
        from medh5.io.nifti import from_nifti

        stored = np.zeros((6, 6, 6), dtype=np.int16)
        stored[:3] = 1  # physical 0 under intercept -1: not in the mask
        stored[3:] = 2  # physical 1: in the mask
        image = tmp_path / "ct.nii"
        mask = tmp_path / "mask.nii"
        self._scaled_nifti(image, np.ones((6, 6, 6), dtype=np.int16), 1.0, 0.0)
        self._scaled_nifti(mask, stored, 1.0, -1.0)
        from_nifti({"CT": image}, tmp_path / "m.medh5", masks={"liver": mask})
        with medh5.open(tmp_path / "m.medh5") as sample:
            liver = sample.annotations["seg"].dense(["liver"])[0]
        # NIfTI x becomes the trailing medh5 axis, so the split is along x.
        assert not liver[..., :3].any()
        assert liver[..., 3:].all()


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

    def test_F10_S4_2_a_rescaled_image_exports_with_its_scale(self, tmp_path: Path):
        """`image.read()` returns stored values; the exporters wrote them bare.

        A CT imported from DICOM with intercept −1024 exported every voxel 1024
        HU too high --- in the nnU-Net exporter's only path, and in
        `to_nifti --stored` --- with nothing in `dataset.json` or the report to
        say so.
        """
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import to_nifti

        path = tmp_path / "ct.medh5"
        with medh5.create(path, sample_id="ct", codec="portable") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(2.0, 0.8, 0.9))
            w.add_image(
                "CT",
                np.full(Organs.SHAPE, 1000, np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
            )
        out = to_nifti(path, "CT", tmp_path / "stored.nii.gz", physical=False)
        image = nib.load(str(out))
        assert image.get_data_dtype() == np.int16
        assert np.asanyarray(image.dataobj.get_unscaled()).flat[0] == 1000
        assert image.get_fdata().flat[0] == pytest.approx(-24.0)

    def test_F11_S3_6_a_2D_sample_round_trips_through_NIfTI(self, tmp_path: Path):
        """Both importers take 2-D input; both exporters stopped on a 3x3 affine.

        `convert_world` was defined for 4x4 only, so an imported radiograph
        could never be written back out --- the round trip the converters page
        promises was broken for every 2-D sample.
        """
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti, to_nifti

        affine = np.array(
            [[0.8, 0, 0, -1.0], [0, 0.9, 0, -2.0], [0, 0, 3.0, 5.0], [0, 0, 0, 1]]
        )
        data = np.arange(32 * 40, dtype=np.int16).reshape(32, 40)
        source = tmp_path / "xray.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(source))

        from_nifti({"XR": source}, tmp_path / "xray.medh5")
        with medh5.open(tmp_path / "xray.medh5") as sample:
            first_affine = np.asarray(sample.grids["ref"].affine)
            first_voxels = sample.images["XR"].read()
        assert first_affine.shape == (3, 3)

        exported = to_nifti(tmp_path / "xray.medh5", "XR", tmp_path / "back.nii.gz")
        from_nifti({"XR": exported}, tmp_path / "again.medh5")
        with medh5.open(tmp_path / "again.medh5") as sample:
            assert np.array_equal(sample.images["XR"].read(), first_voxels)
            assert np.allclose(np.asarray(sample.grids["ref"].affine), first_affine)
            assert sample.grids["ref"].shape == (32, 40)


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
