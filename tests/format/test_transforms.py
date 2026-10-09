"""Registration transforms (spec §10).

The direction convention is the thing these tests exist to pin: a transform with
``from_frame = F`` and ``to_frame = M`` maps F points to M points, and there is
no flag to reverse it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.transforms import (
    ChainTransform,
    InverseTransform,
    basis,
    encode_affine,
    encode_bspline,
    encode_composite,
    encode_displacement,
    folding_fraction,
    linear_sample,
    target_registration_error,
)
from medh5.validate import validate_file
from tests.helpers import ROOT, encode_attr
from tests.kits import Flat

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

    @pytest.mark.parametrize("extrapolation", ["zero", "nearest", "error"])
    def test_S10_4_windowed_sampling_equals_the_whole_field(
        self, tmp_path: Path, extrapolation: str
    ):
        from medh5.transforms import sample_field

        rng = np.random.default_rng(7)
        field = rng.normal(0.0, 2.0, (3, 12, 14, 16)).astype(np.float32)
        path = tmp_path / "field.medh5"
        with medh5.create(path, sample_id="f") as w:
            w.add_grid("g", shape=(12, 14, 16), spacing=(1.0, 1.0, 1.0), frame_uid="f0")
            w.add_image("CT", np.zeros((12, 14, 16), np.int16), grid="g", modality="CT")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="f0",
                to_frame="f1",
                field=field,
                field_grid="g",
                extrapolation=extrapolation,
            )
        inside = rng.uniform([0, 0, 0], [11, 13, 15], size=(40, 3))
        edges = np.array(
            [
                [-0.5, 0.0, 0.0],
                [11.5, 13.5, 15.5],
                [0.0, -0.49, 15.4],
                [11.0, 13.0, 0.0],
            ]
        )
        outside = np.array([[-3.0, 2.0, 2.0], [14.0, 2.0, 2.0], [2.0, 2.0, 40.0]])
        with medh5.open(path) as sample:
            transform: Any = sample.transforms["warp"]
            for points in (inside, edges, np.vstack([inside, edges])):
                expected = sample_field(field, points, extrapolation=extrapolation)
                assert np.array_equal(transform.sample_indices(points), expected)
            if extrapolation == "error":
                with pytest.raises(MEDH5ValidationError, match="outside"):
                    transform.sample_indices(outside)
            else:
                expected = sample_field(field, outside, extrapolation=extrapolation)
                assert np.array_equal(transform.sample_indices(outside), expected)
                mixed = np.vstack([inside[:3], outside])
                assert np.array_equal(
                    transform.sample_indices(mixed),
                    sample_field(field, mixed, extrapolation=extrapolation),
                )
            # And a single far-away point along one axis only.
            lone = np.array([[2.0, 30.0, 2.0]])
            if extrapolation != "error":
                assert np.array_equal(
                    transform.sample_indices(lone),
                    sample_field(field, lone, extrapolation=extrapolation),
                )


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

    def test_S13_2_the_content_id_does_not_depend_on_the_writer_caches(self, tmp_path):
        """The attribute map is derived from the file, not from bookkeeping.

        While the writer built it from ``self._transform_frames``, an amend that
        failed to repopulate that cache hashed a ``content_id`` over fewer
        objects than a reader would find --- so the file verified on the way out
        and reported E702 on the way back in.  In 2.0 there is no cache to
        clear: the writer's `commit` and the reader hash over one map,
        `attr_name_map_of`, derived from the file itself, so they agree by
        construction --- and an amended file with a displacement field is
        stamped with the identity a reader recomputes.
        """
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.amend(path, codec="portable") as w:
            w.add_image(
                "CT2_tp0", np.zeros(SHAPE, dtype=np.int16), grid="ct_tp0", modality="CT"
            )
        with medh5.open(path) as sample:
            assert sample.verify().ok
            assert sample.compute_content_id() == sample.content_id

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

    def test_S10_the_transform_table_is_read_once_per_handle(self, tmp_path):
        """`transform_between` re-read and re-opened every transform per call.

        That is once per pair per training item in `PairedPatchDataset`, and the
        resolver now asks each transform whether its inverse can be evaluated,
        which opens the stored inverse.  A `Sample` is a read-only view and
        `amend` replaces the inode, so an open handle cannot see an edit.
        """
        path = registered(tmp_path / "reg.medh5", displacement=True)
        with medh5.open(path) as sample:
            first = sample.transforms
            assert sample.transforms is first
            assert sample.index is sample.index
            assert sample.transform_between("tp0", "tp1") is not None

    def test_same_frame_needs_no_transform(self, tmp_path):
        path = registered(tmp_path / "reg.medh5")
        with medh5.open(path) as sample:
            assert sample.transform_between("tp0", "tp0") is None
            # A frame nothing in the file declares is a mistyped key, not "no
            # registration exists" (L-35).
            with pytest.raises(KeyError, match="F9"):
                sample.transform_between("F0", "F9")

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

    def _claimed(self, tmp_path, *, deformable: bool):
        """A composite of two invertible steps that carries no `inverse_id`.

        With *deformable* the second step is a dense field declaring
        `invertible: true` --- a claim with nothing behind it.
        """
        path = tmp_path / f"claimed-{deformable}.medh5"
        with medh5.create(
            path, sample_id="s", subject_id="subj", codec="portable"
        ) as w:
            for index, frame in enumerate(("F0", "F1", "F2")):
                w.add_grid(
                    f"g{index}", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid=frame
                )
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g0", modality="CT")
            first = np.eye(4)
            first[0, 3] = 2.0
            w.add_transform(
                "A",
                kind="affine",
                matrix=first,
                from_frame="F0",
                to_frame="F1",
                invertible=True,
            )
            if deformable:
                field = np.zeros((3, *SHAPE), dtype=np.float32)
                field[0] = 0.75
                w.add_transform(
                    "B",
                    kind="displacement",
                    field=field,
                    from_frame="F1",
                    to_frame="F2",
                    field_grid="g1",
                    invertible=True,
                )
            else:
                second = np.eye(4)
                second[1, 3] = 3.0
                w.add_transform(
                    "B",
                    kind="affine",
                    matrix=second,
                    from_frame="F1",
                    to_frame="F2",
                    invertible=True,
                )
            w.add_transform(
                "C",
                kind="composite",
                from_frame="F0",
                to_frame="F2",
                components=["A", "B"],
            )
        return path

    def test_S10_a_resolved_path_is_one_that_can_be_evaluated(self, tmp_path):
        """A declared inverse is not always an evaluable one.

        A composite of invertible affines reports `is_invertible` and carries no
        `inverse_id`, so walking it backwards produced a path that raised on
        first use --- from a call documented to answer None when no path exists.
        Resolution now routes around it through the components, which *are*
        invertible, and takes the same points back to where they started.
        """
        path = self._claimed(tmp_path, deformable=False)
        points = np.array([[3.0, 4.0, 5.0]])
        with medh5.open(path) as sample:
            assert sample.transforms["C"].is_invertible, "the file claims it"
            assert sample.transforms["C"].inverse() is None, "and stores nothing"
            back = sample.transform_between("F2", "F0")
            assert back is not None
            forward = sample.transforms["C"].transform_points(points)
            assert np.allclose(back.transform_points(forward), points)

    def test_S10_an_inverse_that_cannot_be_evaluated_is_not_a_path(self, tmp_path):
        """`invertible: true` on a dense field without `inverse_id` is a claim only.

        There is no analytic inverse of a displacement field here, and
        approximating one would report an accuracy nobody measured, so the edge
        does not exist --- and neither does any route that depended on it.
        """
        path = self._claimed(tmp_path, deformable=True)
        with medh5.open(path) as sample:
            assert sample.transform_between("F2", "F1") is None
            assert sample.transform_between("F2", "F0") is None
            assert sample.transform_between("F0", "F2") is not None, "forward is fine"

    def test_a_stored_inverse_is_used(self, tmp_path):
        path = registered(tmp_path / "reg.medh5", inverse=True)
        with medh5.open(path) as sample:
            transform = sample.transforms["tp0_to_tp1"]
            stored = transform.inverse()
            assert stored is not None
            assert stored.transform_id == "tp1_to_tp0"

    def test_S10_frame_graph_draws_what_the_resolver_walks(self, tmp_path: Path):
        from medh5.transforms import frame_graph

        path = tmp_path / "graph.medh5"
        with Flat.open_writer(path, frames=True) as w:
            w.add_grid(
                "h",
                shape=Flat.SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="f1",
            )
            # Declared invertible, no stored inverse: not traversable backwards.
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="f0",
                to_frame="f1",
                field=np.zeros((3, *Flat.SHAPE), dtype=np.float32),
                field_grid="g",
                invertible=True,
            )
        with medh5.open(path) as sample:
            graph = frame_graph(dict(sample.transforms))
            assert graph["f0"] == ["f1"]
            assert graph["f1"] == []
            assert sample.resolve_frames("f1", "f0") is None

    def test_S10_resolve_frames_is_memoised_per_handle(self, tmp_path: Path):
        path = tmp_path / "memo.medh5"
        with Flat.open_writer(path, frames=True) as w:
            w.add_transform(
                "t", kind="affine", from_frame="f0", to_frame="f1", matrix=np.eye(4)
            )
        with medh5.open(path) as sample:
            first = sample.resolve_frames("f0", "f1")
            assert first is sample.resolve_frames("f0", "f1")
            assert first is not None and first.transform_id == "t"
            assert sample.resolve_frames("f0", "f0") is None
            # The reverse is an analytic inverse, resolved and memoised too.
            back = sample.resolve_frames("f1", "f0")
            assert back is sample.resolve_frames("f1", "f0")


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

    @pytest.mark.parametrize("weights", [[1.0], [1.0, 1.0, 1.0]])
    def test_S10_6_tre_takes_one_weight_per_landmark(self, tmp_path, weights):
        """Errors of 0 and 10 mm: one weight reported a mean of 0.0 --- the
        second landmark dropped --- and three a mean of 3.3."""
        path = registered(tmp_path / "reg.medh5")
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        moving = fixed + SHIFT + np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        with (
            medh5.open(path) as sample,
            pytest.raises(MEDH5ValidationError, match="one weight per landmark"),
        ):
            target_registration_error(
                sample.transforms["tp0_to_tp1"], fixed, moving, weights=weights
            )


class TestInterpolation:
    @pytest.mark.parametrize("columns", [1, 3])
    def test_points_of_another_dimensionality_are_refused(self, columns):
        """A point with fewer coordinates than the field has axes panicked --- a
        ``PanicException``, which ``except Exception`` does not catch --- and one
        with more had the rest ignored.  1.x's SciPy refused the shape."""
        from medh5.transforms import cubic_sample, inside_extent, sample_field

        field = np.ones((1, 4, 4))
        points = np.zeros((2, columns))
        calls = [
            lambda: linear_sample(field, points),
            lambda: cubic_sample(field, points),
            lambda: cubic_sample(field, points, extrapolation="nearest"),
            lambda: sample_field(field, points, interpolation="cubic"),
            lambda: inside_extent([4, 4], points),
        ]
        for call in calls:
            with pytest.raises(ValueError, match=f"{columns} coordinate"):
                call()

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
        and only for cubic fields.  (The engine computes cubic itself, so this
        needs no SciPy; it was skipped wherever SciPy was absent.)
        """
        from medh5.transforms import cubic_sample

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

    def test_D5_S10_4_the_margin_takes_the_outermost_samples_value(self):
        """Half a voxel inside the extent, `error` admitted a point and cubic ---
        SciPy's constant mode --- gave it no displacement, where linear gave the
        outermost sample's value."""
        from medh5.transforms import cubic_sample, linear_sample

        field = (np.arange(12, dtype=np.float64) ** 2).reshape(1, 3, 4)
        margin = np.array([[-0.5, 1.0], [2.0, 3.5]])
        for mode in ("zero", "error"):
            cubic = cubic_sample(field, margin, extrapolation=mode)[:, 0]
            linear = linear_sample(field, margin, extrapolation=mode)[:, 0]
            assert cubic == pytest.approx(linear)
            assert cubic == pytest.approx([field[0, 0, 1], field[0, 2, 3]])
        beyond = np.array([[-0.6, 1.0]])
        assert cubic_sample(field, beyond)[0, 0] == 0.0
        with pytest.raises(MEDH5ValidationError, match="outside"):
            cubic_sample(field, beyond, extrapolation="error")

    def test_folding_fraction_of_an_empty_field(self):
        assert folding_fraction(np.zeros(0)) == 0.0


class TestAmbiguousResolution:
    """§10.2 exists because ambiguity here mirrors registrations silently."""

    def _pair(self, path, *, second: bool):
        shape = (4, 8, 8)
        first = np.eye(4)
        first[:3, 3] = [10.0, 0.0, 0.0]
        other = np.eye(4)
        other[:3, 3] = [-99.0, 0.0, 0.0]
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1", index=1)
            w.add_grid(
                "g0",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="F0",
            )
            w.add_grid(
                "g1",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp1",
                frame_uid="F1",
            )
            w.add_image("CT0", np.zeros(shape, np.int16), grid="g0", modality="CT")
            w.add_image("CT1", np.zeros(shape, np.int16), grid="g1", modality="CT")
            w.add_transform(
                "aaa", kind="affine", matrix=first, from_frame="F0", to_frame="F1"
            )
            if second:
                w.add_transform(
                    "zzz", kind="affine", matrix=other, from_frame="F0", to_frame="F1"
                )
        return path

    def test_one_route_still_resolves(self, tmp_path):
        path = self._pair(tmp_path / "one.medh5", second=False)
        with medh5.open(path) as sample:
            transform = sample.transform_between("tp0", "tp1")
            assert transform is not None
            assert np.allclose(
                transform.transform_points([[0.0, 0.0, 0.0]])[0], [10.0, 0.0, 0.0]
            )

    def test_S10_2_two_equally_short_routes_are_refused_not_picked(self, tmp_path):
        """Two registrations between one frame pair is legal; choosing is not.

        The pick fell out of dict iteration order -- lexicographic transform id
        -- so two registrations 109 mm apart resolved to whichever was named
        first, with nothing in the file saying that one was authoritative.
        """
        path = self._pair(tmp_path / "two.medh5", second=True)
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.transform_between("tp0", "tp1")
            assert exc.value.code == "E501"
            assert "aaa" in str(exc.value) and "zzz" in str(exc.value)
            # Both remain reachable by id -- the file is not malformed.
            assert sorted(sample.transforms) == ["aaa", "zzz"]

    def test_S10_1_a_chain_in_mixed_units_is_refused(self, tmp_path):
        """§10.1 makes `units` a MUST; only the frames were ever checked."""
        shape = (4, 8, 8)
        step = np.eye(4)
        step[:3, 3] = [1.0, 0.0, 0.0]
        path = tmp_path / "units.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1", index=1)
            for gid, frame, tp in (
                ("g0", "F0", "tp0"),
                ("gA", "FA", "tp0"),
                ("g1", "F1", "tp1"),
            ):
                w.add_grid(
                    gid,
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                    units="mm",
                )
                w.add_image(
                    f"CT_{gid}", np.zeros(shape, np.int16), grid=gid, modality="CT"
                )
            w.add_transform(
                "t1",
                kind="affine",
                matrix=step,
                from_frame="F0",
                to_frame="FA",
                units="mm",
            )
            w.add_transform(
                "t2",
                kind="affine",
                matrix=step,
                from_frame="FA",
                to_frame="F1",
                units="mm",
            )
            w.add_transform(
                "comp",
                kind="composite",
                components=["t1", "t2"],
                from_frame="F0",
                to_frame="F1",
                units="mm",
            )
        # The writer refuses the chain (it runs the validator's E501), so a file
        # holding one comes from somewhere else --- which is who the reader's
        # refusal is for.
        import h5py

        with h5py.File(path, "r+") as handle:
            handle["transforms/t2"].attrs["units"] = "um"
        with medh5.open(path) as sample:
            problems = sample.transforms["comp"].check_chain()
            assert any("units" in p for p in problems)
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.transforms["comp"].transform_points([[0.0, 0.0, 0.0]])
            assert exc.value.code == "E501"

    def test_S10_2_equal_routes_that_converge_are_still_ambiguous(self, tmp_path):
        """`A→B→D→T` and `A→C→D→T` are two routes, not one.

        Marking `D` seen as soon as the first predecessor reached it discarded
        the second path, so only one arrival was ever collected and the tie went
        undetected -- the walker returned x=3 where the other route gives x=300.
        """
        shape = (4, 8, 8)

        def shift(dx):
            matrix = np.eye(4)
            matrix[:3, 3] = [dx, 0.0, 0.0]
            return matrix

        path = tmp_path / "converge.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1", index=1)
            w.add_grid(
                "gA",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="A",
            )
            w.add_grid(
                "gT",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp1",
                frame_uid="T",
            )
            for frame in ("B", "C", "D"):
                w.add_grid(
                    f"g{frame}",
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint="tp0",
                    frame_uid=frame,
                )
            for gid in ("gA", "gT", "gB", "gC", "gD"):
                w.add_image(
                    f"CT_{gid}", np.zeros(shape, np.int16), grid=gid, modality="CT"
                )
            w.add_transform(
                "ab", kind="affine", matrix=shift(1.0), from_frame="A", to_frame="B"
            )
            w.add_transform(
                "bd", kind="affine", matrix=shift(2.0), from_frame="B", to_frame="D"
            )
            w.add_transform(
                "ac", kind="affine", matrix=shift(100.0), from_frame="A", to_frame="C"
            )
            w.add_transform(
                "cd", kind="affine", matrix=shift(200.0), from_frame="C", to_frame="D"
            )
            w.add_transform(
                "dt", kind="affine", matrix=shift(0.0), from_frame="D", to_frame="T"
            )

        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.transform_between("tp0", "tp1")
            assert exc.value.code == "E501"

    def test_a_longer_unambiguous_route_still_resolves(self, tmp_path):
        """Only *equal-length* routes are a tie; a single chain must still work."""
        shape = (4, 8, 8)
        step = np.eye(4)
        step[:3, 3] = [1.0, 0.0, 0.0]
        path = tmp_path / "chain.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1", index=1)
            w.add_grid(
                "gA",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="A",
            )
            w.add_grid(
                "gB",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="B",
            )
            w.add_grid(
                "gT",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp1",
                frame_uid="T",
            )
            for gid in ("gA", "gB", "gT"):
                w.add_image(
                    f"CT_{gid}", np.zeros(shape, np.int16), grid=gid, modality="CT"
                )
            w.add_transform(
                "ab", kind="affine", matrix=step, from_frame="A", to_frame="B"
            )
            w.add_transform(
                "bt", kind="affine", matrix=step, from_frame="B", to_frame="T"
            )
        with medh5.open(path) as sample:
            transform = sample.transform_between("tp0", "tp1")
            assert transform is not None
            assert np.allclose(
                transform.transform_points([[0.0, 0.0, 0.0]])[0], [2.0, 0.0, 0.0]
            )


class TestL35FrameGraph:
    """L-35, S-12: inverses are one route, composites end, and typos are errors."""

    def _visits(self, path: Path) -> Any:
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_timepoint("tp0", index=0)
        w.add_timepoint("tp1", index=1)
        for tp, frame in (("tp0", "F"), ("tp1", "M")):
            w.add_grid(
                f"g_{tp}",
                shape=(8, 8, 8),
                spacing=(1, 1, 1),
                frame_uid=frame,
                timepoint=tp,
            )
            w.add_image(
                f"CT_{tp}", np.zeros((8, 8, 8), np.int16), grid=f"g_{tp}", modality="CT"
            )
        return w

    def test_L35_S12_mutually_declared_inverses_resolve_both_ways(self, tmp_path: Path):
        """What E505 checks as "mutually consistent" raised E501 in both directions."""
        path = tmp_path / "mutual.medh5"
        field = np.ones((3, 8, 8, 8), np.float32)
        with self._visits(path) as w:
            for name, source, target, grid, sign, inverse in (
                ("fwd", "F", "M", "g_tp0", 1.0, "bwd"),
                ("bwd", "M", "F", "g_tp1", -1.0, "fwd"),
            ):
                w.add_transform(
                    name,
                    kind="displacement",
                    from_frame=source,
                    to_frame=target,
                    field=sign * field,
                    field_grid=grid,
                    vector_space="world",
                    inverse_id=inverse,
                )
        assert not validate_file(path).errors
        with medh5.open(path) as s:
            forward = s.transform_between("tp0", "tp1")
            backward = s.transform_between("tp1", "tp0")
            assert forward.transform_id == "fwd"
            assert backward.transform_id == "bwd"
            point = [[1.0, 1.0, 1.0]]
            assert forward.transform_points(point).tolist() == [[2.0, 2.0, 2.0]]

    def test_L35_S12_a_one_sided_declaration_is_one_route(self, tmp_path: Path):
        """An affine's analytic inverse and the stored transform naming it are one."""
        path = tmp_path / "one-sided.medh5"
        shift = np.eye(4)
        shift[:3, 3] = 2.0
        back = np.eye(4)
        back[:3, 3] = -2.0
        with self._visits(path) as w:
            w.add_transform(
                "fwd", kind="affine", from_frame="F", to_frame="M", matrix=shift,
                inverse_id="bwd",
            )  # fmt: skip
            w.add_transform(
                "bwd", kind="affine", from_frame="M", to_frame="F", matrix=back
            )
        with medh5.open(path) as s:
            assert s.transform_between("tp0", "tp1").transform_id == "fwd"
            assert s.transform_between("tp1", "tp0").transform_id == "bwd"

    def test_L35_a_composite_that_contains_itself_is_E501(self, tmp_path: Path):
        """It validated clean and evaluating it raised RecursionError."""
        path = tmp_path / "cycle.medh5"
        with self._visits(path) as w:
            w.add_transform(
                "a", kind="affine", from_frame="F", to_frame="X", matrix=np.eye(4)
            )
            w.add_transform(
                "b", kind="affine", from_frame="X", to_frame="M", matrix=np.eye(4)
            )
            w.add_transform(
                "b2", kind="affine", from_frame="X", to_frame="F", matrix=np.eye(4)
            )
            w.add_transform(
                "c1",
                kind="composite",
                from_frame="F",
                to_frame="M",
                components=["a", "b"],
            )
            w.add_transform(
                "c2", kind="composite", from_frame="X", to_frame="M",
                components=["b2", "c1"],
            )  # fmt: skip
        with h5py.File(path, "r+") as handle:
            handle["transforms/c1"].attrs["components"] = np.array(
                ["a", "c2"], dtype=h5py.string_dtype()
            )
        assert "E501" in validate_file(path).codes
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            s.transforms["c1"].transform_points([[0.0, 0.0, 0.0]])
        assert caught.value.code == "E501"
        assert "contains itself" in str(caught.value)

    def test_L35_a_mistyped_key_is_a_KeyError_not_no_registration(self, tmp_path: Path):
        path = tmp_path / "typo.medh5"
        with self._visits(path) as w:
            w.add_transform(
                "t", kind="affine", from_frame="F", to_frame="M", matrix=np.eye(4)
            )
        with medh5.open(path) as s:
            assert s.transform_between("tp0", "tp1").transform_id == "t"
            assert s.transform_between("F", "M").transform_id == "t"
            with pytest.raises(KeyError, match="TP1"):
                s.transform_between("tp0", "TP1")

    def test_L35_the_registration_guide_no_longer_warns_against_linking(self):
        guide = (ROOT / "docs/guides/registration.md").read_text(encoding="utf-8")
        assert "Do not link them with `inverse_id`" not in guide
