"""Registration transforms (spec §10).

The direction convention is the thing these tests exist to pin: a transform with
``from_frame = F`` and ``to_frame = M`` maps F points to M points, and there is
no flag to reverse it.
"""

from __future__ import annotations

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.transforms.affine import encode_affine
from medh5.transforms.apply import (
    folding_fraction,
    linear_sample,
    target_registration_error,
)
from medh5.transforms.bspline import basis, encode_bspline
from medh5.transforms.composite import encode_composite
from medh5.transforms.displacement import encode_displacement
from medh5.transforms.resolve import ChainTransform, InverseTransform

SHAPE = (12, 16, 16)
SHIFT = np.array([2.0, -1.0, 0.5])


def registered(
    path, *, inverse=False, displacement=False, composite=False, bspline=False
):
    matrix = np.eye(4)
    matrix[:3, 3] = SHIFT
    with medh5.create(path, codec="portable") as w:
        w.add_timepoint("tp0", days_from_baseline=0)
        w.add_timepoint("tp1", days_from_baseline=92)
        for tp, frame in (("tp0", "F0"), ("tp1", "F1")):
            w.add_grid(
                f"ct_{tp}",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                origin=(0.0, 0.0, 0.0),
                timepoint=tp,
                frame_uid=frame,
            )
            w.add_image(
                f"CT_{tp}",
                np.zeros(SHAPE, dtype=np.int16),
                grid=f"ct_{tp}",
                modality="CT",
            )
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            from_frame="F0",
            to_frame="F1",
            matrix=matrix,
            from_grid="ct_tp0",
            to_grid="ct_tp1",
            invertible=True,
            inverse_id="tp1_to_tp0" if inverse else None,
        )
        if inverse:
            back = np.eye(4)
            back[:3, 3] = -SHIFT
            w.add_transform(
                "tp1_to_tp0",
                kind="affine",
                from_frame="F1",
                to_frame="F0",
                matrix=back,
                invertible=True,
                inverse_id="tp0_to_tp1",
            )
        if displacement or composite:
            field = np.zeros((3, *SHAPE), dtype=np.float32)
            field[0] = 0.75
            w.add_transform(
                "refine",
                kind="displacement",
                from_frame="F1",
                to_frame="F2",
                field=field,
                field_grid="ct_tp1",
            )
        if composite:
            w.add_transform(
                "chain",
                kind="composite",
                from_frame="F0",
                to_frame="F2",
                components=["tp0_to_tp1", "refine"],
            )
        if bspline:
            control = np.zeros((3, 6, 6, 6), dtype=np.float64)
            control[1] = 0.5
            w.add_grid(
                "cp",
                shape=(6, 6, 6),
                spacing=(3.0, 3.2, 3.2),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp1",
                frame_uid="F1",
            )
            w.add_transform(
                "ffd",
                kind="bspline",
                from_frame="F1",
                to_frame="F3",
                control_points=control,
                cp_grid="cp",
            )
        w.deidentification(method="dicom-psi-profile")
    return path


class TestDirectionConvention:
    def test_S10_2_maps_from_frame_points_to_to_frame(self, tmp_path):
        """§10.2: x_M = T(x_F). The one convention, with no flag to reverse it."""
        path = registered(tmp_path / "reg.medh5")
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            assert transform.from_frame == "F0"
            assert transform.to_frame == "F1"
            assert np.allclose(transform.transform_points(points), points + SHIFT)

    def test_S10_a_transform_may_not_map_a_frame_to_itself(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="F0")
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_transform(
                "t", kind="affine", from_frame="F0", to_frame="F0", matrix=np.eye(4)
            )
        assert exc.value.code == "E502"

    def test_S10_timepoints_come_from_the_frames(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with medh5.open(path) as sample:
            assert sample.transforms["tp0_to_tp1"].timepoints == ("tp0", "tp1")
            assert "reg" in sample.profiles


class TestAffine:
    def test_S10_3_round_trip_and_inverse(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        points = np.array([[1.0, 2.0, 3.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            forward = transform.transform_points(points)
            assert np.allclose(transform.inverse_points(forward), points)
            assert transform.jacobian_determinant_value == pytest.approx(1.0)
            assert transform.is_invertible
            assert transform.n_spatial == 3

    def test_S10_3_last_row_is_enforced(self):
        bad = np.eye(4)
        bad[-1, 0] = 0.5
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_affine(bad)
        assert exc.value.code == "E504"

    def test_S10_3_singular_and_non_square_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_affine(np.zeros((3, 4)))
        assert exc.value.code == "E504"
        singular = np.eye(4)
        singular[0, 0] = 0.0
        with pytest.raises(MEDH5ValidationError):
            encode_affine(singular)

    def test_a_rotation_scales_volume_by_its_determinant(self, tmp_path):
        matrix = np.diag([2.0, 3.0, 4.0, 1.0])
        path = tmp_path / "scaled.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="F0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_transform(
                "t", kind="affine", from_frame="F0", to_frame="F1", matrix=matrix
            )
        with medh5.open(path) as sample:
            assert sample.transforms["t"].jacobian_determinant_value == pytest.approx(
                24.0
            )


class TestIdentity:
    def test_identity_is_a_no_op_and_always_invertible(self, tmp_path):
        path = tmp_path / "id.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="F0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_transform("t", kind="identity", from_frame="F0", to_frame="F1")
        points = np.array([[1.0, 2.0, 3.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["t"]
            assert np.array_equal(transform.transform_points(points), points)
            assert transform.is_invertible


class TestDisplacement:
    def test_S10_4_field_adds_its_displacement(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        inside = np.array([[6.0, 6.0, 6.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["refine"]
            assert transform.vector_space == "world"
            assert transform.interpolation == "linear"
            assert np.allclose(
                transform.transform_points(inside), inside + [0.75, 0.0, 0.0]
            )
            assert transform.max_magnitude == pytest.approx(0.75)

    def test_S10_4_outside_the_field_extrapolation_decides(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        outside = np.array([[-99.0, -99.0, -99.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["refine"]
            assert np.allclose(transform.transform_points(outside), outside)

    def test_S10_4_jacobian_of_a_constant_field_is_one(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.open(path) as sample:
            determinants = sample.transforms["refine"].jacobian_determinant()
            assert determinants.shape == SHAPE
            assert np.allclose(determinants, 1.0)
            assert sample.transforms["refine"].folding_fraction() == 0.0

    def test_S10_4_folding_is_detected(self, tmp_path):
        """A field that reverses an axis folds, and the fraction says so."""
        grid_shape = (8, 8, 8)
        coords = np.arange(grid_shape[0], dtype=np.float32)
        field = np.zeros((3, *grid_shape), dtype=np.float32)
        field[0] = -3.0 * coords[:, None, None]
        path = tmp_path / "fold.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=grid_shape,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                frame_uid="F0",
            )
            w.add_image(
                "CT", np.zeros(grid_shape, dtype=np.int16), grid="g", modality="CT"
            )
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="F0",
                to_frame="F1",
                field=field,
                field_grid="g",
            )
        with medh5.open(path) as sample:
            assert sample.transforms["warp"].folding_fraction() > 0.5

    def test_S10_4_one_component_reads_without_the_others(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.open(path) as sample:
            component = sample.transforms["refine"].read_field(component=0)
            assert component.shape == SHAPE
            assert np.allclose(component, 0.75)
            roi = sample.transforms["refine"].read_field(
                roi=np.s_[0:2, 0:2, 0:2], component=1
            )
            assert roi.shape == (2, 2, 2)

    def test_S10_4_index_vector_space_converts_through_the_grid(self, tmp_path):
        field = np.zeros((3, *SHAPE), dtype=np.float32)
        field[0] = 2.0  # two voxels along axis 0, which is 1.5 mm each
        path = tmp_path / "idx.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                origin=(0.0, 0.0, 0.0),
                frame_uid="F0",
            )
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="F0",
                to_frame="F1",
                field=field,
                field_grid="g",
                vector_space="index",
            )
        with medh5.open(path) as sample:
            moved = sample.transforms["warp"].transform_points(
                np.array([[3.0, 3.0, 3.0]])
            )
            assert moved[0, 0] == pytest.approx(3.0 + 2.0 * 1.5)

    def test_field_shape_and_options_are_checked(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_displacement(np.zeros((2, 4, 4, 4)), field_grid="g")
        assert exc.value.code == "E503"
        with pytest.raises(MEDH5ValidationError):
            encode_displacement(
                np.zeros((3, 4, 4, 4)), field_grid="g", vector_space="galactic"
            )
        with pytest.raises(MEDH5ValidationError):
            encode_displacement(
                np.zeros((3, 4, 4, 4)), field_grid="g", interpolation="magic"
            )
        with pytest.raises(MEDH5ValidationError):
            encode_displacement(
                np.zeros((3, 4, 4, 4)), field_grid="g", extrapolation="guess"
            )

    def test_S10_4_field_must_be_in_the_source_frame(self, tmp_path):
        field = np.zeros((3, *SHAPE), dtype=np.float32)
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="F1")
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="F0",
                to_frame="F1",
                field=field,
                field_grid="g",
            )
        assert exc.value.code == "E503"


class TestBSpline:
    def test_S10_5_basis_is_a_partition_of_unity(self):
        t = np.linspace(0.0, 1.0, 25)
        for order in (1, 3):
            assert np.allclose(basis(order, t).sum(axis=0), 1.0)
        assert np.allclose(basis(3, np.array([0.0])).ravel(), [1 / 6, 4 / 6, 1 / 6, 0])

    def test_S10_5_a_constant_lattice_gives_a_constant_displacement(self, tmp_path):
        """Partition of unity means uniform coefficients reproduce themselves."""
        path = registered(tmp_path / "reg.medh5", bspline=True)
        interior = np.array([[6.0, 6.0, 6.0], [7.5, 8.0, 8.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["ffd"]
            assert transform.order == 3
            displacement = transform.displacement_at(interior)
            assert np.allclose(displacement, [0.0, 0.5, 0.0])

    def test_S10_5_sampling_to_a_field_agrees_with_direct_evaluation(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", bspline=True)
        with medh5.open(path) as sample:
            transform = sample.transforms["ffd"]
            grid = sample.grids["ct_tp1"]
            field = transform.to_displacement_field(grid)
            assert field.shape == (3, *SHAPE)
            centre = np.array([[grid.spatial_shape[0] // 2 * 1.5, 5.0, 5.0]])
            direct = transform.displacement_at(centre)[0]
            index = grid.world_to_index(centre)[0].round().astype(int)
            assert np.allclose(field[(slice(None), *index)], direct, atol=1e-3)

    def test_unsupported_orders_and_shapes_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_bspline(np.zeros((3, 6, 6, 6)), cp_grid="cp", order=5)
        assert exc.value.code == "E502"
        with pytest.raises(MEDH5ValidationError):
            encode_bspline(np.zeros((2, 6, 6, 6)), cp_grid="cp")
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_bspline(np.zeros((3, 2, 2, 2)), cp_grid="cp")
        assert exc.value.code == "E503"
        with pytest.raises(MEDH5ValidationError):
            basis(7, np.array([0.5]))


class TestComposite:
    def test_S10_5_components_apply_left_to_right(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", composite=True)
        points = np.array([[3.0, 4.0, 5.0]])
        with medh5.open(path) as sample:
            chain = sample.transforms["chain"]
            assert chain.component_ids == ("tp0_to_tp1", "refine")
            assert chain.check_chain() == []
            expected = sample.transforms["refine"].transform_points(
                sample.transforms["tp0_to_tp1"].transform_points(points)
            )
            assert np.allclose(chain.transform_points(points), expected)

    def test_S10_5_a_broken_chain_refuses_to_evaluate(self, tmp_path):
        import h5py

        from medh5._hdf5 import encode_attr

        path = registered(tmp_path / "reg.medh5", composite=True)
        with h5py.File(path, "r+") as handle:
            handle["transforms/chain"].attrs["to_frame"] = encode_attr("F9")
        with medh5.open(path) as sample:
            chain = sample.transforms["chain"]
            assert chain.check_chain()
            with pytest.raises(MEDH5ValidationError) as exc:
                chain.transform_points(np.zeros((1, 3)))
            assert exc.value.code == "E501"

    def test_composite_needs_at_least_two_declared_components(self, tmp_path):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_composite(["only-one"])
        assert exc.value.code == "E501"
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="F0")
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_transform(
                "c",
                kind="composite",
                from_frame="F0",
                to_frame="F2",
                components=["nope", "also-nope"],
            )
        assert exc.value.code == "E501"


class TestAmend:
    """An amend inherits the transforms, so it must inherit what it knows of them.

    The writer's caches are what ``commit`` hashes and what ``infer_profiles``
    reads; a cache that disagrees with the copied file produces a ``content_id``
    over fewer objects than the reader later finds.
    """

    def test_S14_4_amend_keeps_the_file_verifiable(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with medh5.amend(path, codec="portable") as w:
            w.add_image(
                "CT2_tp0", np.zeros(SHAPE, dtype=np.int16), grid="ct_tp0", modality="CT"
            )
        with medh5.open(path) as sample:
            assert sample.verify().ok
            assert "reg" in sample.profiles

    def test_S10_5_amend_can_compose_an_inherited_transform(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.amend(path, codec="portable") as w:
            w.add_transform(
                "chain",
                kind="composite",
                from_frame="F0",
                to_frame="F2",
                components=["tp0_to_tp1", "refine"],
            )
        with medh5.open(path) as sample:
            assert sample.transforms["chain"].check_chain() == []
            assert sample.verify().ok


class TestResolution:
    def test_S10_resolves_between_timepoints_not_by_name(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with medh5.open(path) as sample:
            found = sample.transform_between("tp0", "tp1")
            assert found is not None
            assert found.transform_id == "tp0_to_tp1"

    def test_resolution_uses_an_inverse_when_it_must(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        points = np.array([[1.0, 2.0, 3.0]])
        with medh5.open(path) as sample:
            back = sample.transform_between("tp1", "tp0")
            assert isinstance(back, InverseTransform)
            forward = sample.transforms["tp0_to_tp1"].transform_points(points)
            assert np.allclose(back.transform_points(forward), points)

    def test_resolution_chains_several_hops(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", composite=True)
        with medh5.open(path) as sample:
            found = sample.transform_between("F0", "F2")
            assert found is not None
            assert np.allclose(
                found.transform_points(np.array([[3.0, 4.0, 5.0]])),
                sample.transforms["chain"].transform_points(
                    np.array([[3.0, 4.0, 5.0]])
                ),
            )

    def test_same_frame_needs_no_transform(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with medh5.open(path) as sample:
            assert sample.transform_between("tp0", "tp0") is None
            assert sample.transform_between("F0", "F9") is None

    def test_a_chain_reports_its_steps(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.open(path) as sample:
            found = sample.transform_between("F0", "F2")
            assert isinstance(found, ChainTransform)
            assert [s.transform_id for s in found.steps] == ["tp0_to_tp1", "refine"]
            assert found.summary()["steps"] == ["tp0_to_tp1", "refine"]

    def test_a_deformable_transform_is_not_traversed_backwards(self, tmp_path):
        """Approximating a dense inverse would report accuracy nobody measured."""
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.open(path) as sample:
            assert sample.transform_between("F2", "F1") is None

    def test_a_stored_inverse_is_used(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", inverse=True)
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            stored = transform.inverse()
            assert stored is not None
            assert stored.transform_id == "tp1_to_tp0"


class TestLandmarksAndMetrics:
    def test_S10_6_tre_is_zero_for_a_perfect_transform(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        fixed = np.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            result = target_registration_error(transform, fixed, fixed + SHIFT)
            assert result["mean"] == pytest.approx(0.0, abs=1e-9)
            assert result["n"] == 2

    def test_S10_6_tre_measures_the_error_it_is_given(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        fixed = np.array([[0.0, 0.0, 0.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            moving = fixed + SHIFT + np.array([3.0, 4.0, 0.0])
            result = target_registration_error(transform, fixed, moving)
            assert result["mean"] == pytest.approx(5.0)

    def test_S10_6_mismatched_landmark_sets_are_refused(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with (
            medh5.open(path) as sample,
            pytest.raises(MEDH5ValidationError, match="row order"),
        ):
            target_registration_error(
                sample.transforms["tp0_to_tp1"], np.zeros((2, 3)), np.zeros((3, 3))
            )

    def test_weights_shift_the_mean(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        moving = fixed + SHIFT + np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            unweighted = target_registration_error(transform, fixed, moving)["mean"]
            weighted = target_registration_error(
                transform, fixed, moving, weights=[9.0, 1.0]
            )["mean"]
            assert weighted < unweighted


class TestInterpolation:
    def test_linear_sample_reproduces_grid_values(self):
        field = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
        at_nodes = linear_sample(field, np.array([[1.0, 2.0], [0.0, 0.0]]))
        assert at_nodes[0, 0] == pytest.approx(field[0, 1, 2])
        assert at_nodes[1, 1] == pytest.approx(field[1, 0, 0])

    def test_linear_sample_interpolates_midpoints(self):
        field = np.zeros((1, 2, 2))
        field[0, 0, 0], field[0, 0, 1] = 0.0, 10.0
        mid = linear_sample(field, np.array([[0.0, 0.5]]))
        assert mid[0, 0] == pytest.approx(5.0)

    def test_extrapolation_modes(self):
        field = np.ones((1, 4, 4))
        outside = np.array([[-5.0, -5.0]])
        assert linear_sample(field, outside)[0, 0] == 0.0
        assert linear_sample(field, outside, extrapolation="nearest")[0, 0] == 1.0
        with pytest.raises(MEDH5ValidationError, match="outside"):
            linear_sample(field, outside, extrapolation="error")
        with pytest.raises(MEDH5ValidationError):
            linear_sample(field, outside, extrapolation="wing-it")

    def test_S10_4_cubic_honours_error_extrapolation_like_linear(self):
        """SciPy has no raising mode, so `error` must not become constant-zero.

        Mapping it onto ``mode="constant"`` answers an out-of-domain query with
        "no displacement" --- the silence the declared contract exists to break,
        and only for cubic fields.
        """
        pytest.importorskip("scipy")
        from medh5.transforms.apply import cubic_sample

        field = np.ones((1, 4, 4))
        outside = np.array([[-5.0, -5.0]])
        assert cubic_sample(field, outside)[0, 0] == 0.0
        # approx: the cubic spline prefilter is not exact on a constant field
        assert cubic_sample(field, outside, extrapolation="nearest")[0, 0] == (
            pytest.approx(1.0)
        )
        with pytest.raises(MEDH5ValidationError, match="outside"):
            cubic_sample(field, outside, extrapolation="error")
        with pytest.raises(MEDH5ValidationError):
            cubic_sample(field, outside, extrapolation="wing-it")

    def test_folding_fraction_of_an_empty_field(self):
        assert folding_fraction(np.zeros(0)) == 0.0
