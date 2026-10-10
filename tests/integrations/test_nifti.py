"""NIfTI import and export (``medh5.io.nifti``)."""

from __future__ import annotations

import json
import os
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


class TestImportedMasksSitOnTheGrid:
    """A mask's geometry is the grid's, not only its shape (L01 of the 2.0
    audit): a translated, rotated, mirrored or rescaled mask of the right
    shape was imported onto voxels nobody drew on, with a warning at most."""

    @staticmethod
    def _mask(tmp_path: Path, affine: Any, name: str = "moved") -> Path:
        data = np.zeros(SHAPE_XYZ, np.uint8)
        data[4:8, 4:8, 2:4] = 1
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return path

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            ("translated", "origin"),
            ("rotated", "direction"),
            ("mirrored", "direction"),
            ("rescaled", "spacing"),
        ],
    )
    def test_L01_a_mask_from_another_grid_is_refused(
        self, volumes, tmp_path, change, field
    ):
        sample = tmp_path / "i.medh5"
        from_nifti({"CT": volumes["ct"]}, sample)
        affine = AFFINE.copy()
        if change == "translated":
            affine[:3, 3] += [100.0, 0.0, 0.0]
        elif change == "rotated":
            quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            affine[:3, :3] = quarter @ AFFINE[:3, :3]
        elif change == "mirrored":
            affine[:3, 0] *= -1
        else:
            affine[:3, :3] *= 1.5
        with pytest.raises(MEDH5ValidationError, match=f"on {field}: it was drawn"):
            import_seg_nifti(sample, {"liver": self._mask(tmp_path, affine)})
        with medh5.open(sample) as opened:
            assert "seg" not in opened.annotations, "nothing was written"

    def test_L01_assume_aligned_takes_the_voxels_and_says_so(self, volumes, tmp_path):
        sample = tmp_path / "i.medh5"
        from_nifti({"CT": volumes["ct"]}, sample)
        affine = AFFINE.copy()
        affine[:3, 3] += [100.0, 0.0, 0.0]
        report = import_seg_nifti(
            sample, {"liver": self._mask(tmp_path, affine)}, assume_aligned=True
        )
        (guess,) = [g for g in report.guesses if g.kind == "geometry"]
        assert guess.detail["field"] == "origin"
        with medh5.open(sample) as opened:
            assert opened.annotations["seg"].dense(["liver"])[0].any()

    def test_L01_the_same_grid_in_metres_is_the_same_grid(self, volumes, tmp_path):
        """Lengths are compared in millimetres, whatever the header's unit."""
        sample = tmp_path / "i.medh5"
        from_nifti({"CT": volumes["ct"]}, sample)
        metres = AFFINE.copy()
        metres[:3, :] /= 1000.0
        image = nib.Nifti1Image(np.asarray(nib.load(volumes["liver"]).dataobj), metres)
        image.header.set_xyzt_units(xyz="meter")
        path = tmp_path / "liver_m.nii.gz"
        nib.save(image, str(path))
        report = import_seg_nifti(sample, {"liver": path})
        assert not [g for g in report.guesses if g.kind == "geometry"]

    def test_L01_masks_are_compared_in_the_grid_s_coordinate_system(
        self, volumes, tmp_path
    ):
        sample = tmp_path / "i.medh5"
        from_nifti({"CT": volumes["ct"]}, sample)
        with pytest.raises(MEDH5ValidationError, match="cannot apply"):
            import_seg_nifti(sample, {"liver": volumes["liver"]}, coord_system="RAS")
        import_seg_nifti(sample, {"liver": volumes["liver"]}, coord_system="LPS")


class TestGridsAgreeInTimeAndUnits:
    """Volumes share a grid in units and frame times too (L06), and an export
    says what its numbers are in (L07, L04)."""

    @staticmethod
    def _series(
        path: Path, *, step: float | None, offset: float = 0.0, unit: str = "sec"
    ) -> Path:
        image = nib.Nifti1Image(np.zeros((*SHAPE_XYZ, 3), np.int16), AFFINE)
        if step is not None:
            zooms = list(image.header.get_zooms())
            zooms[3] = step
            image.header.set_zooms(zooms)
            image.header["toffset"] = offset
            image.header.set_xyzt_units(xyz="mm", t=unit)
        nib.save(image, str(path))
        return path

    def test_L06_the_same_grid_in_metres_converts(self, volumes, tmp_path):
        metres = AFFINE.copy()
        metres[:3, :] /= 1000.0
        image = nib.Nifti1Image(np.asarray(nib.load(volumes["ct"]).dataobj), metres)
        image.header.set_xyzt_units(xyz="meter")
        path = tmp_path / "ct_m.nii.gz"
        nib.save(image, str(path))
        from_nifti({"CT": volumes["ct"], "CT_m": path}, tmp_path / "s.medh5")
        with medh5.open(tmp_path / "s.medh5") as sample:
            assert sample.grids["ref"].units == "mm"

    def test_L06_stated_frame_times_that_disagree_are_refused(self, tmp_path):
        a = self._series(tmp_path / "a.nii.gz", step=2.0, offset=5.0)
        b = self._series(tmp_path / "b.nii.gz", step=10.0, offset=20.0)
        with pytest.raises(MEDH5ValidationError, match="states frame times"):
            from_nifti({"A": a, "B": b}, tmp_path / "s.medh5")

    def test_L06_frame_times_compare_in_seconds(self, tmp_path):
        a = self._series(tmp_path / "a.nii.gz", step=2.0, offset=5.0)
        b = self._series(tmp_path / "b.nii.gz", step=2000.0, offset=5000.0, unit="msec")
        from_nifti({"A": a, "B": b}, tmp_path / "s.medh5")

    def test_L06_a_volume_stating_times_lends_them_to_one_that_does_not(self, tmp_path):
        a = self._series(tmp_path / "a.nii.gz", step=None)
        b = self._series(tmp_path / "b.nii.gz", step=2.0, offset=5.0)
        report = from_nifti({"A": a, "B": b}, tmp_path / "s.medh5")
        with medh5.open(tmp_path / "s.medh5") as sample:
            assert sample.grids["ref"].time_values == (5.0, 7.0, 9.0)
        (taken,) = [n for n in report.of_kind("time_values") if "takes them" in str(n)]
        assert taken.detail["from"] == "B"

    def test_L07_an_export_states_its_units_and_frame_times(self, tmp_path):
        source = self._series(tmp_path / "a.nii.gz", step=2.5, offset=10.0)
        from_nifti({"A": source}, tmp_path / "s.medh5")
        out = to_nifti(tmp_path / "s.medh5", "A", tmp_path / "out.nii.gz")
        header = nib.load(str(out)).header
        assert header.get_xyzt_units() == ("mm", "sec")
        assert header.get_zooms()[3] == pytest.approx(2.5)
        assert float(header["toffset"]) == pytest.approx(10.0)
        _, geometry = read_nifti(out)
        assert geometry["time_values"] == pytest.approx([10.0, 12.5, 15.0])
        assert geometry["time_measured"]

    def test_L07_uneven_frames_go_to_a_sidecar_and_come_back(self, tmp_path):
        path = tmp_path / "s.medh5"
        with medh5.create(path, sample_id="s") as w:
            w.add_grid(
                "g",
                shape=(4, 6, 5, 4),
                spacing=(2.0, 1.0, 1.0),
                axis_names=("t", "z", "y", "x"),
                axis_kinds=("time", "spatial", "spatial", "spatial"),
                time_values=[0.0, 1.0, 3.0, 7.0],
                time_units="s",
                timepoint="tp0",
            )
            w.add_image(
                "DCE", np.zeros((4, 6, 5, 4), np.int16), grid="g", modality="MR"
            )
        from medh5.io.report import ConversionReport

        report = ConversionReport(converter="to-nifti")
        out = to_nifti(path, "DCE", tmp_path / "dce.nii.gz", report=report)
        header = nib.load(str(out)).header
        assert header.get_zooms()[3] == 0.0 and header.get_xyzt_units()[1] == "unknown"
        sidecar = json.loads((tmp_path / "dce.json").read_text(encoding="utf-8"))
        assert sidecar == {"VolumeTiming": [0.0, 1.0, 3.0, 7.0]}
        assert report.of_kind("time_values")
        _, geometry = read_nifti(out)
        assert geometry["time_values"] == [0.0, 1.0, 3.0, 7.0]

    def test_L07_a_grid_in_metres_exports_in_millimetres(self, tmp_path):
        path = tmp_path / "s.medh5"
        with medh5.create(path, sample_id="s") as w:
            w.add_grid(
                "g", shape=(4, 5, 6), spacing=(0.002, 0.001, 0.001), units="m",
                origin=(0.01, 0.02, 0.03), timepoint="tp0",
            )  # fmt: skip
            w.add_image("CT", np.zeros((4, 5, 6), np.int16), grid="g", modality="CT")
        out = to_nifti(path, "CT", tmp_path / "ct.nii.gz")
        restored = nib.load(str(out))
        assert restored.header.get_xyzt_units()[0] == "mm"
        assert np.allclose(restored.header.get_zooms(), [1.0, 1.0, 2.0])
        assert np.allclose(np.abs(restored.affine[:3, 3]), [10.0, 20.0, 30.0])

    def test_L04_an_annotation_exports_on_its_own_grid(self, tmp_path):
        """A PET-grid mask exported beside the CT took the CT's affine."""
        path = tmp_path / "s.medh5"
        with Organs.writer(path) as w:
            w.add_grid(
                "pet", shape=(4, 6, 6), spacing=(4.0, 1.6, 1.6),
                origin=(30.0, 40.0, 50.0), timepoint="tp0",
            )  # fmt: skip
            w.add_image(
                "PET", np.zeros((4, 6, 6), np.float32), grid="pet", modality="PT"
            )
            mask = np.zeros((4, 6, 6), bool)
            mask[1:3, 2:4, 2:4] = True
            w.add_segmentation("petseg", grid="pet", masks={1: mask})
        mask_out = to_nifti(path, "CT", tmp_path / "m.nii.gz", annotation="petseg")
        pet_out = to_nifti(path, "PET", tmp_path / "pet.nii.gz")
        exported = nib.load(str(mask_out))
        assert np.allclose(exported.affine, nib.load(str(pet_out)).affine)
        assert exported.shape == (6, 6, 4)


class TestExportsOwnTheirVolumeStatements:
    """What an export says about its volumes beside the file --- a sidecar's
    per-volume fields, a ``.bval`` --- is replaced with the file (N03 of the
    2.0 re-audit), and a channel axis is written as one (N04)."""

    @staticmethod
    def _series(path: Path, times: list[float]) -> Path:
        n = len(times)
        with medh5.create(path, sample_id=path.stem) as w:
            w.add_grid(
                "g",
                shape=(n, 6, 5, 4),
                spacing=(2.0, 1.0, 1.0),
                axis_names=("t", "z", "y", "x"),
                axis_kinds=("time", "spatial", "spatial", "spatial"),
                time_values=times,
                time_units="s",
                timepoint="tp0",
            )
            w.add_image(
                "DCE", np.zeros((n, 6, 5, 4), np.int16), grid="g", modality="MR"
            )
        return path

    @staticmethod
    def _channels(path: Path, **acquisition: Any) -> Path:
        volume = np.stack(
            [np.full((6, 5, 4), 10 * (c + 1), np.int16) for c in range(2)]
        )
        with medh5.create(path, sample_id=path.stem) as w:
            w.add_grid(
                "g",
                shape=(2, 6, 5, 4),
                spacing=(2.0, 1.0, 1.0),
                axis_names=("c", "z", "y", "x"),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
                timepoint="tp0",
            )
            w.add_image("MR", volume, grid="g", modality="MR")
            if acquisition:
                w.acquisition("MR", **acquisition)
        return path

    def test_N03_an_overwrite_replaces_the_timing_beside_the_file(self, tmp_path):
        """Uneven frames wrote `VolumeTiming`; an even series written over the
        file left it, and the reader, which prefers it, read the old timeline
        for the new image.  Fields of no volume's are kept."""
        out, sidecar = tmp_path / "dce.nii.gz", tmp_path / "dce.json"
        to_nifti(self._series(tmp_path / "a.medh5", [0.0, 1.0, 3.0]), "DCE", out)
        fields = json.loads(sidecar.read_text(encoding="utf-8"))
        assert fields == {"VolumeTiming": [0.0, 1.0, 3.0]}
        sidecar.write_text(json.dumps({**fields, "Manufacturer": "x"}), "utf-8")
        to_nifti(self._series(tmp_path / "b.medh5", [10.0, 12.0, 14.0]), "DCE", out)
        assert json.loads(sidecar.read_text(encoding="utf-8")) == {"Manufacturer": "x"}
        assert read_nifti(out)[1]["time_values"] == [10.0, 12.0, 14.0]
        # And back: another frame count, uneven again.
        to_nifti(self._series(tmp_path / "c.medh5", [0.0, 2.0, 3.0, 9.0]), "DCE", out)
        fields = json.loads(sidecar.read_text(encoding="utf-8"))
        assert fields == {"Manufacturer": "x", "VolumeTiming": [0.0, 2.0, 3.0, 9.0]}
        assert read_nifti(out)[1]["time_values"] == [0.0, 2.0, 3.0, 9.0]
        assert (
            sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []
        )

    def test_N03_an_interrupted_overwrite_never_reads_another_timeline(
        self, tmp_path, monkeypatch
    ):
        """The old timing is withdrawn before the image is replaced: an export
        that fails between them leaves the old image with its timing
        unmeasured, not the new one with the old image's."""
        out = tmp_path / "dce.nii.gz"
        to_nifti(self._series(tmp_path / "a.medh5", [0.0, 1.0, 3.0]), "DCE", out)
        replace = os.replace

        def interrupted(source: Any, target: Any) -> None:
            if Path(target) == out:
                raise OSError("interrupted")
            replace(source, target)

        monkeypatch.setattr(os, "replace", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            to_nifti(self._series(tmp_path / "b.medh5", [0.0, 5.0, 6.0]), "DCE", out)
        monkeypatch.undo()
        _, geometry = read_nifti(out)
        assert not geometry["time_measured"]
        assert not (tmp_path / "dce.json").exists()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    @pytest.mark.parametrize("old", ["series", "dwi"])
    def test_F24_an_export_that_cannot_write_its_image_changes_nothing(
        self, tmp_path, monkeypatch, old
    ):
        """The old statements were withdrawn before the new image was
        serialised, so an export that failed to write it --- a full disk, a
        type NIfTI cannot hold --- left the old image with its timing or its
        b-values gone (F24 of the round-4 audit).  The image is written first;
        until it is whole, nothing beside the file is touched."""
        import nibabel

        out = tmp_path / "old.nii.gz"
        if old == "series":
            source = self._series(tmp_path / "a.medh5", [0.0, 1.0, 3.0])
            to_nifti(source, "DCE", out)
        else:
            source = self._channels(tmp_path / "a.medh5", b_values=[0.0, 900.0])
            to_nifti(source, "MR", out)
        beside = sorted(p for p in tmp_path.iterdir() if p.name.startswith("old."))
        assert len(beside) == 2, "the image and its timing or b-values"
        before = {p.name: p.read_bytes() for p in beside}
        stated = ("leading_kind", "time_values", "time_measured")
        expected = {k: read_nifti(out)[1][k] for k in stated}

        def full(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(nibabel, "save", full)
        with pytest.raises(OSError, match="disk full"):
            to_nifti(self._series(tmp_path / "b.medh5", [0.0, 5.0, 6.0]), "DCE", out)
        monkeypatch.undo()
        assert {p.name: p.read_bytes() for p in beside} == before
        assert {k: read_nifti(out)[1][k] for k in stated} == expected
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    def test_N04_a_channel_axis_exports_as_one(self, tmp_path):
        """Written as a plain fourth axis, two MR channels read back as two
        frames of time, one second apart, unmeasured.  NIfTI-1 keeps a
        voxel's components on dim[5] under a vector intent."""
        out = to_nifti(
            self._channels(tmp_path / "mr.medh5"), "MR", tmp_path / "mr.nii.gz"
        )
        image = nib.load(str(out))
        assert image.shape == (4, 5, 6, 1, 2)
        assert image.header.get_intent()[0] == "vector"
        _, geometry = read_nifti(out)
        assert geometry["leading_kind"] == "channel" and geometry["time_values"] is None
        from_nifti({"MR": out}, tmp_path / "back.medh5")
        with medh5.open(tmp_path / "back.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.axis_kinds == ("channel", "spatial", "spatial", "spatial")
            assert grid.time_values is None
            channels = sample.images["MR"].read()
            assert channels[0].max() == 10 and channels[1].max() == 20
        # Told outright, the reader agrees.
        assert read_nifti(out, fourth_axis="channel")[1]["leading_kind"] == "channel"

    def test_N04_what_the_channels_mean_comes_back(self, tmp_path):
        """b-values in a .bval beside a four-dimensional file, the layout every
        diffusion tool reads; echo times in the sidecar."""
        dwi = self._channels(tmp_path / "dwi.medh5", b_values=[0.0, 1000.0])
        to_nifti(dwi, "MR", tmp_path / "dwi.nii.gz")
        assert nib.load(str(tmp_path / "dwi.nii.gz")).shape == (4, 5, 6, 2)
        assert (tmp_path / "dwi.bval").read_text().split() == ["0", "1000"]
        echo = self._channels(
            tmp_path / "echo.medh5",
            EchoTime=[0.005, 0.01],
            ImageType=["ORIGINAL", "PRIMARY"],  # one per channel, and no numbers
        )
        to_nifti(echo, "MR", tmp_path / "echo.nii.gz")
        from_nifti(
            {"DWI": tmp_path / "dwi.nii.gz", "ME": tmp_path / "echo.nii.gz"},
            tmp_path / "back.medh5",
        )
        with medh5.open(tmp_path / "back.medh5") as sample:
            acquisition = sample.document.acquisition
            assert acquisition["DWI"]["b_values"] == [0.0, 1000.0]
            assert acquisition["ME"]["EchoTime"] == [0.005, 0.01]
            assert sample.images["ME"].channel_names == ("TE=0.005", "TE=0.01")
        # A time series written over the DWI takes its b-values with it.
        to_nifti(
            self._series(tmp_path / "t.medh5", [0.0, 1.0]),
            "DCE",
            tmp_path / "dwi.nii.gz",
        )
        assert not (tmp_path / "dwi.bval").exists()
        assert read_nifti(tmp_path / "dwi.nii.gz")[1]["leading_kind"] == "time"

    @pytest.mark.parametrize("sidecar_text", ["{not json", "[]"])
    def test_N16_a_refused_overwrite_leaves_the_old_files(self, tmp_path, sidecar_text):
        """The `.bval` was withdrawn before the sidecar the new statements go
        to was read, so an export refused for that sidecar had already taken
        the old image's b-values: it reread as an unmeasured time axis (N16 of
        the round-3 audit).  Everything is checked before anything goes."""
        out = tmp_path / "dwi.nii.gz"
        dwi = self._channels(tmp_path / "dwi.medh5", b_values=[0.0, 1200.0])
        to_nifti(dwi, "MR", out)
        sidecar, bval = tmp_path / "dwi.json", tmp_path / "dwi.bval"
        sidecar.write_text(sidecar_text, encoding="utf-8")
        before = {p.name: p.read_bytes() for p in (out, sidecar, bval)}
        echo = self._channels(tmp_path / "echo.medh5", EchoTime=[0.005, 0.01])
        with pytest.raises(MEDH5ValidationError, match="not a JSON object"):
            to_nifti(echo, "MR", out)
        assert {p.name: p.read_bytes() for p in (out, sidecar, bval)} == before
        assert read_nifti(out)[1]["leading_kind"] == "channel"
        from_nifti({"DWI": out}, tmp_path / "back.medh5")
        with medh5.open(tmp_path / "back.medh5") as sample:
            assert sample.document.acquisition["DWI"]["b_values"] == [0.0, 1200.0]
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    def test_N16_an_object_sidecar_takes_the_new_statements(self, tmp_path):
        """The control: a sidecar that is a JSON object keeps its other
        fields and takes the export's."""
        out = tmp_path / "dwi.nii.gz"
        to_nifti(
            self._channels(tmp_path / "dwi.medh5", b_values=[0.0, 1200.0]), "MR", out
        )
        sidecar = tmp_path / "dwi.json"
        sidecar.write_text(json.dumps({"Manufacturer": "x"}), encoding="utf-8")
        echo = self._channels(tmp_path / "echo.medh5", EchoTime=[0.005, 0.01])
        to_nifti(echo, "MR", out)
        fields = json.loads(sidecar.read_text(encoding="utf-8"))
        assert fields == {"Manufacturer": "x", "EchoTime": [0.005, 0.01]}
        assert not (tmp_path / "dwi.bval").exists()
