"""Resampling, a recorded derivation (spec §10.2, §4.4, §11.1; 2.2.0 E11).

A resample adds objects and changes none: the sources stay as they were, and
the geometry it uses is the file's own --- two grids' affines when they share
a world (§3.3 rule 4), otherwise the transform path between their frames,
evaluated at the target's voxel centres as §10.2 directs.  Without a path
it refuses; alignment is never assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.transforms import resample
from medh5.validate import validate_file

# The source of the SciPy comparison: a smooth, steep volume whose every
# voxel differs.
SOURCE_SHAPE = (5, 6, 7)
_FLAT = np.arange(np.prod(SOURCE_SHAPE), dtype=np.float64)
SOURCE = (_FLAT * _FLAT * 0.37 + _FLAT / 3.0).reshape(SOURCE_SHAPE)


def _rotation_about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0, 0.3], [0, c, -s, -0.2], [0, s, c, 0.4], [0, 0, 0, 1.0]])


def registered_pair(path: Path, *, transform: bool = True) -> Path:
    """A source volume in frame F0, a small target grid in frame F1, and
    (unless *transform* is false) the affine from F1 to F0: what maps the
    target's points into the source, the direction §10.2 resamples by."""
    with medh5.create(path, codec="portable") as w:
        w.add_grid("src", shape=SOURCE_SHAPE, spacing=(2.0, 1.0, 1.5), frame_uid="F0")
        w.add_image("img", SOURCE, grid="src", modality="CT")
        w.add_grid(
            "tgt",
            shape=(2, 2, 3),
            spacing=(1.7, 1.3, 1.1),
            origin=(1.1, 0.8, 1.2),
            frame_uid="F1",
        )
        w.add_image("ref", np.zeros((2, 2, 3), np.int16), grid="tgt", modality="CT")
        if transform:
            w.add_transform(
                "f1_to_f0",
                kind="affine",
                from_frame="F1",
                to_frame="F0",
                matrix=_rotation_about_z(0.1),
            )
    return path


def one_grid(path: Path, image: np.ndarray, **grid: object) -> Path:
    """One image on one grid in frame F0."""
    spacing = grid.pop("spacing", (1.0,) * image.ndim)
    with medh5.create(path, codec="portable") as w:
        w.add_grid("g", shape=image.shape, spacing=spacing, frame_uid="F0", **grid)
        w.add_image("img", image, grid="g", modality="CT")
    return path


def read(path: Path, image_id: str) -> np.ndarray:
    with medh5.open(path) as s:
        return np.asarray(s.images[image_id].read())


class TestThroughATransform:
    # scipy.ndimage.map_coordinates(SOURCE, coords, order=k), with coords the
    # composed index mapping inv(A_src) @ M @ A_tgt at the target's 12 voxel
    # centres (every one inside the source, so the boundary mode is moot);
    # mode="nearest" for orders 0 and 1, "constant" for 3.
    RECORDED = {
        "nearest": [
            698.4633333333334, 730.9866666666666, 764.25, 1221.1299999999999,
            1264.0133333333333, 1307.6366666666668, 2701.5833333333335,
            2765.1866666666665, 2829.5299999999997, 3659.37, 3733.3333333333335,
            3808.0366666666664,
        ],
        "linear": [
            576.8437830997229, 575.5502416217719, 573.8522416311124,
            838.4908216545243, 838.0551546326654, 837.2790555959742,
            1979.0123709639279, 1976.6872703332265, 1973.9577111898168,
            2482.1479378415734, 2480.6807116669647, 2478.873053477523,
        ],
        "cubic": [
            383.8732534868754, 381.06031925302165, 381.12083790524423,
            641.6379888986115, 641.031845663016, 640.7878866784969,
            1825.5690637724429, 1819.5229221440943, 1820.7466651377097,
            2408.960043270993, 2408.3281503460635, 2408.2043672514615,
        ],
    }  # fmt: skip

    @pytest.mark.parametrize("interpolation", ["nearest", "linear", "cubic"])
    def test_S10_2_resample_through_an_affine_matches_scipy(
        self, tmp_path, interpolation
    ):
        path = registered_pair(tmp_path / "s.medh5")
        report = resample(
            path, "img", onto="tgt", interpolation=interpolation, dtype=np.float64
        )
        assert report.transforms == {"src": ["f1_to_f0"]}
        got = read(path, "img_on_tgt").reshape(-1)
        np.testing.assert_allclose(got, self.RECORDED[interpolation], rtol=1e-12)

    def test_S10_2_a_resample_without_a_transform_path_is_refused(self, tmp_path):
        path = registered_pair(tmp_path / "s.medh5", transform=False)
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="no transform path") as caught:
            resample(path, "img", onto="tgt")
        assert "'F1'" in str(caught.value) and "'F0'" in str(caught.value)
        assert path.read_bytes() == before

    def test_S10_2_a_grid_without_a_frame_is_related_to_nothing(self, tmp_path):
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("a", shape=(2, 3, 3), spacing=(1.0, 1.0, 1.0))
            w.add_grid("b", shape=(2, 3, 3), spacing=(1.0, 1.0, 1.0))
            w.add_image("img", np.zeros((2, 3, 3), np.int16), grid="a", modality="CT")
        with pytest.raises(MEDH5ValidationError, match="names no frame"):
            resample(path, "img", onto="b")

    def test_S10_2_a_displacement_path_is_evaluated_at_the_target_points(
        self, tmp_path
    ):
        """A constant displacement field is a shift: the same values as the
        affine that shifts by as much."""
        shift = np.array([0.5, -0.25, 0.75])
        results = []
        for kind in ("affine", "displacement"):
            path = tmp_path / f"{kind}.medh5"
            with medh5.create(path, codec="portable") as w:
                w.add_grid(
                    "src", shape=SOURCE_SHAPE, spacing=(1.0,) * 3, frame_uid="F0"
                )
                w.add_image("img", SOURCE, grid="src", modality="CT")
                w.add_grid("tgt", shape=(3, 4, 5), spacing=(1.0,) * 3, frame_uid="F1")
                if kind == "affine":
                    matrix = np.eye(4)
                    matrix[:3, 3] = shift
                    w.add_transform(
                        "t",
                        kind="affine",
                        from_frame="F1",
                        to_frame="F0",
                        matrix=matrix,
                    )
                else:
                    field = np.broadcast_to(shift[:, None, None, None], (3, 3, 4, 5))
                    w.add_transform(
                        "t",
                        kind="displacement",
                        from_frame="F1",
                        to_frame="F0",
                        field=np.ascontiguousarray(field, dtype=np.float32),
                        field_grid="tgt",
                    )
            resample(path, "img", onto="tgt", dtype=np.float64)
            results.append(read(path, "img_on_tgt"))
        np.testing.assert_allclose(results[0], results[1], rtol=1e-12)


class TestSharedFrame:
    def test_S3_3_grids_sharing_a_frame_but_not_comparable_are_refused(self, tmp_path):
        """W917: one frame named, another coord_system.  `transform_between`
        answers "the same frame" for them, and no transform relates them."""
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("lps", shape=(2, 3, 3), spacing=(1.0,) * 3, frame_uid="F0")
            w.add_grid(
                "ras",
                shape=(2, 3, 3),
                spacing=(1.0,) * 3,
                frame_uid="F0",
                coord_system="RAS",
            )
            w.add_image("img", np.zeros((2, 3, 3), np.int16), grid="lps", modality="CT")
        with pytest.raises(MEDH5ValidationError, match="not comparable"):
            resample(path, "img", onto="ras")

    def test_S3_3_a_new_spacing_keeps_the_source_edges(self, tmp_path):
        """Two 1 mm voxels at 2 mm are one voxel over the same 2 mm: the
        first centre moves to the middle of the pair."""
        image = np.array([[[10, 30]] * 2] * 2, dtype=np.int16)  # (2, 2, 2)
        path = one_grid(
            tmp_path / "s.medh5", image, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 4.0)
        )
        report = resample(path, "img", spacing=(1.0, 1.0, 2.0), grid_id="coarse")
        assert report.new_grid
        with medh5.open(path) as s:
            grid = s.grids["coarse"]
            assert grid.shape == (2, 2, 1)
            assert grid.origin == pytest.approx((0.0, 0.0, 4.5))
            source = s.grids["g"]
            for k in range(3):
                assert grid.origin[k] - grid.spacing[k] / 2 == pytest.approx(
                    source.origin[k] - source.spacing[k] / 2
                )
            assert grid.frame_uid == source.frame_uid
            assert np.asarray(s.images["img_on_coarse"].read()).tolist() == [
                [[20.0], [20.0]],
                [[20.0], [20.0]],
            ]

    def test_S3_3_a_spacing_that_does_not_divide_the_extent_covers_it(self, tmp_path):
        image = np.zeros((1, 3, 5), np.uint8)
        path = one_grid(tmp_path / "s.medh5", image)
        resample(path, "img", spacing=(1.0, 2.0, 2.0), grid_id="g2")
        with medh5.open(path) as s:
            assert s.grids["g2"].shape == (1, 2, 3)

    def test_S3_3_spacing_must_give_one_finite_value_per_axis(self, tmp_path):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        with pytest.raises(ValueError, match="one per axis"):
            resample(path, "img", spacing=(1.0, 1.0), grid_id="n")
        with pytest.raises(ValueError, match="finite and positive"):
            resample(path, "img", spacing=(1.0, 0.0, 1.0), grid_id="n")

    def test_resampling_onto_the_source_grid_is_the_identity(self, tmp_path):
        rng = np.random.default_rng(7)
        image = rng.integers(-1000, 1500, (6, 7, 8)).astype(np.int16)
        path = one_grid(tmp_path / "s.medh5", image, spacing=(0.7, 1.3, 0.9))
        for interpolation in ("nearest", "linear", "cubic"):
            resample(
                path,
                "img",
                onto="g",
                interpolation=interpolation,
                suffix=f"_{interpolation}",
            )
        with medh5.open(path) as s:
            same = s.images["img_nearest"]
            assert same.dtype == np.int16
            assert np.asarray(same.read()).tobytes() == image.tobytes()
            for interpolation in ("linear", "cubic"):
                got = np.asarray(s.images[f"img_{interpolation}"].read())
                assert got.dtype == np.float32
                np.testing.assert_allclose(got, image, rtol=1e-6, atol=1e-6)
            # Nothing was outside the source, so there is no mask to name.
            assert s.images["img_linear"].valid_mask is None

    def test_integer_output_rounds_half_to_even_and_clips(self, tmp_path):
        """Pairs averaged by a 2x spacing land on halves: 0.5, 1.5 and 2.5
        round to 0, 2 and 2; -299.5 and 300.5 clip to uint8's 0 and 255."""
        row = [0, 1, 1, 2, 2, 3, -300, -299, 300, 301]
        image = np.array([row], dtype=np.float32)  # a 2-D grid, (1, 10)
        path = one_grid(tmp_path / "s.medh5", image)
        resample(path, "img", spacing=(1.0, 2.0), grid_id="half", dtype=np.uint8)
        got = read(path, "img_on_half")
        assert got.dtype == np.uint8
        assert got.tolist() == [[0, 2, 2, 0, 255]]

    def test_only_spatial_axes_are_resampled_channel_by_channel(self, tmp_path):
        image = np.stack([np.full((2, 4), 1.0), np.full((2, 4), 5.0)]).astype(
            np.float32
        )  # (c, y, x)
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=image.shape,
                spacing=(1.0, 1.0),
                axis_kinds=("channel", "spatial", "spatial"),
                frame_uid="F0",
            )
            w.add_image("img", image, grid="g", modality="MR", channel_names=["a", "b"])
        resample(path, "img", spacing=(2.0, 2.0), grid_id="g2")
        with medh5.open(path) as s:
            assert s.grids["g2"].shape == (2, 1, 2)
            assert s.images["img_on_g2"].channel_names == ("a", "b")
            assert np.asarray(s.images["img_on_g2"].read()).tolist() == [
                [[1.0, 1.0]],
                [[5.0, 5.0]],
            ]

    def test_a_grid_without_the_sources_channel_axis_is_refused(self, tmp_path):
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=(2, 3, 3),
                spacing=(1.0, 1.0),
                axis_kinds=("channel", "spatial", "spatial"),
                frame_uid="F0",
            )
            w.add_grid("plain", shape=(3, 3), spacing=(1.0, 1.0), frame_uid="F0")
            w.add_image("img", np.zeros((2, 3, 3), np.float32), grid="g", modality="MR")
        with pytest.raises(MEDH5ValidationError, match="non-spatial axes"):
            resample(path, "img", onto="plain")

    @pytest.mark.parametrize(
        ("units", "values", "same"),
        [
            ("s", (0.0, 1.0), True),
            ("ms", (0.0, 1.0), False),
            ("ms", (0.0, 1000.0), True),
        ],
    )
    def test_S3_2_time_values_are_compared_in_seconds(
        self, tmp_path, units, values, same
    ):
        """A time axis is carried as it is, so the target must hold the same
        acquisition times: ``[0, 1]`` in ms is not ``[0, 1]`` in s."""
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            for grid_id, time_units, time_values in (
                ("g", None, (0.0, 1.0)),  # "s", the default
                ("h", units, values),
            ):
                w.add_grid(
                    grid_id,
                    shape=(2, 3, 3),
                    spacing=(1.0, 1.0),
                    axis_kinds=("time", "spatial", "spatial"),
                    time_values=time_values,
                    time_units=time_units,
                    frame_uid="F0",
                )
            w.add_image("img", np.zeros((2, 3, 3), np.float32), grid="g", modality="MR")
        if same:
            resample(path, "img", onto="h")
            assert read(path, "img_on_h").shape == (2, 3, 3)
        else:
            with pytest.raises(MEDH5ValidationError, match="acquisition times"):
                resample(path, "img", onto="h")

    def test_the_value_attributes_carry_over(self, tmp_path):
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=(2, 3, 3), spacing=(1.0,) * 3, frame_uid="F0")
            w.add_image(
                "img",
                np.zeros((2, 3, 3), np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=2.0,
                rescale_intercept=-1024.0,
                window_center=[40.0],
                window_width=[400.0],
            )
        resample(path, "img", spacing=(2.0, 3.0, 3.0), grid_id="c")
        with medh5.open(path) as s:
            out = s.images["img_on_c"]
            assert (out.modality, out.value_type, out.value_units) == (
                "CT",
                "quantitative",
                "HU",
            )
            assert out.rescale == (2.0, -1024.0)
            assert out.window == ((40.0,), (400.0,))

    def test_a_multiscale_image_resamples_its_level_0(self, tmp_path):
        base = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("l0", shape=(4, 4, 4), spacing=(1.0,) * 3, frame_uid="F0")
            w.add_grid(
                "l1",
                shape=(2, 2, 2),
                spacing=(2.0,) * 3,
                origin=(0.5,) * 3,
                frame_uid="F0",
            )
            w.add_pyramid(
                "img",
                [base, base[::2, ::2, ::2]],
                grid_levels=["l0", "l1"],
                modality="CT",
                downsample_method="nearest",
            )
        resample(path, "img", onto="l0", interpolation="nearest", suffix="_copy")
        with medh5.open(path) as s:
            copy = s.images["img_copy"]
            assert not copy.is_multiscale
            assert np.array_equal(np.asarray(copy.read()), base)


class TestValidity:
    def test_S4_4_voxels_outside_the_source_are_filled_and_masked(self, tmp_path):
        image = np.full((2, 3, 4), 7, np.int16)
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0,) * 3, frame_uid="F0")
            w.add_image("img", image, grid="g", modality="CT")
            # Two voxels further along x than the source reaches.
            w.add_grid("wide", shape=(2, 3, 6), spacing=(1.0,) * 3, frame_uid="F0")
        report = resample(path, "img", onto="wide", interpolation="nearest", fill=-5)
        (written,) = report.images
        assert written.valid_mask == "img_on_wide_valid"
        assert written.filled_voxels == report.filled_voxels == 2 * 3 * 2
        with medh5.open(path) as s:
            out = s.images["img_on_wide"]
            assert out.valid_mask == "img_on_wide_valid"
            data = np.asarray(out.read())
            assert (data[..., :4] == 7).all() and (data[..., 4:] == -5).all()
            valid = s.valid_region("img_on_wide")
            assert valid[..., :4].all() and not valid[..., 4:].any()
        assert validate_file(path, level="integrity").ok

    def test_S4_4_the_half_voxel_margin_is_inside_the_source(self, tmp_path):
        """The source covers [-0.5, n - 0.5] (§3.3): a target centre in the
        margin past the outermost sample takes that sample, not the fill."""
        image = np.array([[[1, 2, 3, 4]]], np.int16)
        path = one_grid(tmp_path / "s.medh5", image)
        with medh5.amend(path) as w:
            w.add_grid(
                "shifted",
                shape=(1, 1, 4),
                spacing=(1.0,) * 3,
                origin=(0.0, 0.0, 0.4),
                frame_uid="F0",
            )
        report = resample(path, "img", onto="shifted", interpolation="nearest")
        assert report.filled_voxels == 0
        assert read(path, "img_on_shifted").tolist() == [[[1, 2, 3, 4]]]

    def test_S4_4_the_source_valid_mask_and_the_extent_are_intersected(self, tmp_path):
        image = np.ones((1, 4, 4), np.float32)
        acquired = np.zeros((1, 4, 4), bool)
        acquired[:, :, :2] = True  # the left half was acquired
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0,) * 3, frame_uid="F0")
            w.add_mask("fov", acquired, grid="g")
            w.add_image("img", image, grid="g", modality="CT", valid_mask="fov")
            # One voxel up along y: its last row lies outside the source.
            w.add_grid(
                "up",
                shape=(1, 4, 4),
                spacing=(1.0,) * 3,
                origin=(0.0, 1.0, 0.0),
                frame_uid="F0",
            )
        report = resample(path, "img", onto="up")
        assert report.filled_voxels == 4
        with medh5.open(path) as s:
            valid = s.valid_region("img_on_up")
        expected = np.zeros((1, 4, 4), bool)
        expected[:, :3, :2] = True
        assert valid.tolist() == expected.tolist()

    @pytest.mark.parametrize(
        ("fill", "dtype", "match"),
        [
            (0.5, np.int16, "fill 0.5"),
            (70000, np.int16, "fill 70000"),
            (np.nan, np.uint8, "fill NaN"),
        ],
    )
    def test_a_fill_the_dtype_cannot_hold_is_refused(
        self, tmp_path, fill, dtype, match
    ):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        with pytest.raises(ValueError, match=match):
            resample(path, "img", onto="g", fill=fill, dtype=dtype)


class TestRecord:
    def test_S11_1_a_resample_is_one_activity_with_every_output(self, tmp_path):
        image = np.ones((2, 3, 4), np.int16)
        path = tmp_path / "s.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0,) * 3, frame_uid="F0")
            w.add_image("CT", image, grid="g", modality="CT")
            w.add_image("PT", image.astype(np.float32), grid="g", modality="PT")
        report = resample(
            path, ["CT", "PT"], spacing=(1.0, 1.0, 3.0), grid_id="iso", fill=-1
        )
        assert report.written == (
            "grids/iso",
            "images/CT_on_iso",
            "images/PT_on_iso",
            "annotations/CT_on_iso_valid",
            "annotations/PT_on_iso_valid",
        )
        with medh5.open(path) as s:
            (activity,) = s.document.provenance.activities_by_type("resample")
            assert activity.id == report.activity_id
            assert activity.inputs == ("images/CT", "images/PT")
            assert activity.outputs == report.written
            assert activity.params["spacing"] == [1.0, 1.0, 3.0]
            assert activity.params["grid_id"] == "iso"
            assert activity.params["interpolation"] == "linear"
            assert activity.params["fill"] == -1.0
            for image_id in ("CT_on_iso", "PT_on_iso"):
                assert s.images[image_id].prov == activity.id
                assert s.annotations[f"{image_id}_valid"].prov == activity.id
            # The sources are untouched.
            assert np.array_equal(np.asarray(s.images["CT"].read()), image)
        report = validate_file(path, level="integrity")
        assert report.ok and not report.errors

    @pytest.mark.parametrize(
        ("fill", "recorded"),
        [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf"), (-2.5, -2.5)],
    )
    def test_S11_1_a_non_finite_fill_is_recorded_as_given(
        self, tmp_path, fill, recorded
    ):
        """JSON has no NaN, and a provenance document refuses one; a fill
        recorded as ``null`` would claim a derivation nobody ran.  A non-finite
        fill is recorded as Python spells the float, which ``float()`` reads
        back."""
        path = one_grid(tmp_path / "s.medh5", np.ones((2, 2, 2), np.float32))
        with medh5.amend(path) as w:
            w.add_grid("wide", shape=(2, 2, 3), spacing=(1.0,) * 3, frame_uid="F0")
        resample(path, "img", onto="wide", interpolation="nearest", fill=fill)
        with medh5.open(path) as s:
            (activity,) = s.document.provenance.activities_by_type("resample")
            assert activity.params["fill"] == recorded
            outside = np.asarray(s.images["img_on_wide"].read())[..., 2]
        assert np.array_equal(outside, np.full((2, 2), fill), equal_nan=True)
        assert float(recorded) == fill or np.isnan(fill)

    def test_S11_1_the_transform_path_and_the_target_are_inputs(self, tmp_path):
        path = registered_pair(tmp_path / "s.medh5")
        report = resample(path, "img", onto="tgt")
        with medh5.open(path) as s:
            activity = s.document.provenance.activity(report.activity_id)
        assert activity.inputs == ("images/img", "grids/tgt", "transforms/f1_to_f0")
        assert activity.params["onto"] == "tgt"
        assert activity.params["transforms"] == {"src": ["f1_to_f0"]}

    def test_an_existing_id_is_refused_before_writing(self, tmp_path):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        resample(path, "img", spacing=(2.0, 3.0, 3.0), grid_id="c")
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="another suffix"):
            resample(path, "img", onto="c")
        with pytest.raises(MEDH5ValidationError, match="already in the sample"):
            resample(path, "img", spacing=(2.0, 3.0, 3.0), grid_id="c")
        assert path.read_bytes() == before
        resample(path, "img", onto="c", suffix="_again")
        with medh5.open(path) as s:
            assert {"img_on_c", "img_again"} <= set(s.images)

    def test_an_invalid_output_id_is_E003(self, tmp_path):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        with pytest.raises(MEDH5ValidationError) as caught:
            resample(path, "img", onto="g", suffix="/x")
        assert caught.value.code == "E003"

    def test_out_writes_a_new_file_and_leaves_the_source(self, tmp_path):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        before = path.read_bytes()
        out = tmp_path / "out.medh5"
        report = resample(path, "img", spacing=(2.0, 3.0, 3.0), grid_id="c", out=out)
        assert Path(report.path) == out
        assert path.read_bytes() == before
        with medh5.open(out) as s:
            assert "img_on_c" in s.images
        with pytest.raises(MEDH5ValidationError, match="does not overwrite"):
            resample(path, "img", spacing=(2.0, 3.0, 3.0), grid_id="d", out=out)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "give onto"),
            ({"onto": "g", "spacing": (1, 1, 1), "grid_id": "n"}, "not both"),
            ({"spacing": (1, 1, 1)}, "grid_id must name it"),
            ({"onto": "g", "grid_id": "n"}, "goes with spacing"),
            ({"onto": "g", "interpolation": "lanczos"}, "interpolation must be"),
        ],
    )
    def test_the_target_is_given_exactly_once(self, tmp_path, kwargs, match):
        path = one_grid(tmp_path / "s.medh5", np.zeros((2, 3, 3), np.int16))
        with pytest.raises(ValueError, match=match):
            resample(path, "img", **kwargs)

    def test_images_on_two_grids_take_no_one_new_spacing(self, tmp_path):
        path = registered_pair(tmp_path / "s.medh5")
        with pytest.raises(MEDH5ValidationError, match="different grids"):
            resample(path, ["img", "ref"], spacing=(1.0, 1.0, 1.0), grid_id="n")
