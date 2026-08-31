"""Geometric annotations: boxes, OBB, keypoints, points, contours, mesh (spec §8)."""

from __future__ import annotations

import numpy as np
import pytest

import medh5
from medh5.annotations.geometric import (
    Polygon,
    check_space,
    encode_boxes,
    encode_contours,
    encode_keypoints,
    encode_mesh,
    encode_obb,
    encode_points,
)
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import box_to_slices, slices_to_box
from medh5.labels.labelset import LabelSet, Skeleton
from tests.v1.conftest import SHAPE

BOXES = np.array(
    [
        [[1.5, 7.5], [1.5, 9.5], [1.5, 9.5]],
        [[6.5, 11.5], [10.5, 18.5], [4.5, 12.5]],
    ],
    dtype=np.float32,
)


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


@pytest.fixture
def skeleton_label_set(label_set):
    return LabelSet(
        label_set.id,
        version=label_set.version,
        classes=list(label_set.classes),
        skeletons=[Skeleton("pair", (1, 2), ((1, 2),))],
    )


def detection_sample(path, label_set, *, space="index", units="mm", **kwargs):
    with medh5.create(path, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(label_set)
        w.add_grid(
            "ct",
            shape=SHAPE,
            spacing=(1.5, 0.8, 0.8),
            origin=(-12.0, -9.6, -9.6),
            units=units,
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
        w.add_boxes(
            "lesions",
            BOXES,
            ["lesion", "lesion"],
            grid="ct",
            space=space,
            instance_ids=[1, 2],
            scores=[0.91, 0.62],
            attributes=[{"reader": "r1"}, {"reader": "r2"}],
            **kwargs,
        )
    return path


class TestBoxes:
    def test_S8_2_round_trip(self, tmp_path, label_set):
        path = detection_sample(tmp_path / "det.medh5", label_set)
        with medh5.open(path) as sample:
            boxes = sample.annotations["lesions"]
            assert boxes.kind == "boxes"
            assert boxes.task == "detection"
            assert len(boxes) == 2
            assert np.array_equal(boxes.boxes, BOXES)
            assert boxes.instance_ids.tolist() == [1, 2]
            assert boxes.scores[0] == pytest.approx(0.91)
            assert boxes.attributes[1] == {"reader": "r2"}
            assert boxes.space == "index"
            assert boxes.frame_uid == "pseudo:frame-100"
            assert "det" in sample.profiles

    def test_S8_1_boxes_convert_to_slices_without_rounding(self, tmp_path, label_set):
        path = detection_sample(tmp_path / "det.medh5", label_set)
        with medh5.open(path) as sample:
            slices = sample.annotations["lesions"].as_slices()
        assert slices[0] == (slice(2, 8), slice(2, 10), slice(2, 10))
        assert slices[1] == (slice(7, 12), slice(11, 19), slice(5, 13))

    @pytest.mark.parametrize("space", ["index", "world"])
    def test_S8_2_an_annotation_with_no_objects_reads(self, tmp_path, label_set, space):
        """The verified negative is the case the coverage contract exists for.

        A detection that searched for a class and found nothing has zero boxes
        and names the class in `annotated_class_ids` (§9).  `np.stack([])`
        raises, so the one shape the model is designed to express crashed on
        every world-space read.
        """
        path = tmp_path / f"empty-{space}.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid(
                "ct",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                timepoint="tp0",
                frame_uid="pseudo:frame-100",
            )
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_boxes(
                "lesions",
                np.zeros((0, 3, 2), dtype=np.float32),
                [],
                grid="ct",
                space=space,
                annotated_classes=["lesion"],
            )
        with medh5.open(path) as sample:
            boxes = sample.annotations["lesions"]
            assert len(boxes) == 0
            assert boxes.annotated_class_ids == (3,), "searched for and not found"
            assert boxes.as_world().shape == (0, 3, 2)
            assert boxes.world_corners().shape == (0, 8, 3)
            assert boxes.as_slices() == []
            assert list(boxes) == []

    def test_S8_1_box_slice_box_is_the_identity(self):
        """The property the whole voxel-edge convention exists for."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            start = rng.integers(0, 40, 3)
            stop = start + rng.integers(1, 20, 3)
            slices = tuple(
                slice(int(a), int(b)) for a, b in zip(start, stop, strict=True)
            )
            box = slices_to_box(slices)
            assert box_to_slices(box) == slices
            assert np.allclose(box[:, 1] - box[:, 0], stop - start)

    def test_S8_1_world_conversion_uses_corners(self, tmp_path, label_set):
        path = detection_sample(tmp_path / "det.medh5", label_set)
        with medh5.open(path) as sample:
            boxes = sample.annotations["lesions"]
            corners = boxes.world_corners()
            bounds = boxes.as_world()
            assert corners.shape == (2, 8, 3)
            assert np.allclose(bounds[:, :, 0], corners.min(axis=1))
            assert np.allclose(bounds[:, :, 1], corners.max(axis=1))
            grid = sample.grids["ct"]
            assert np.allclose(corners[0, 0], grid.index_to_world(BOXES[0][:, 0]))

    def test_S8_1_world_space_boxes_need_a_frame(self, tmp_path, label_set):
        path = detection_sample(tmp_path / "w.medh5", label_set, space="world")
        with medh5.open(path) as sample:
            boxes = sample.annotations["lesions"]
            assert boxes.space == "world"
            assert boxes.frame_uid == "pseudo:frame-100"
            assert np.array_equal(boxes.as_world(), BOXES.astype(np.float64))
            assert boxes.as_slices()

    def test_S8_1_index_space_requires_a_grid(self, tmp_path, label_set):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_boxes("b", BOXES, [1, 1], space="index")
        assert exc.value.code == "E412"

    def test_S8_1_world_space_requires_a_frame(self, tmp_path, label_set):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_boxes("b", BOXES, [1, 1], grid="g", space="world")
        assert exc.value.code == "E412"

    def test_S3_5_uncalibrated_grids_forbid_world_space(self, tmp_path, label_set):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.label_set(label_set)
            w.add_grid(
                "g",
                shape=(24, 24),
                spacing=(1.0, 1.0),
                units="px",
                frame_uid="f0",
            )
            w.add_image("XR", np.zeros((24, 24)), grid="g", modality="DX")
            w.add_boxes("b", BOXES[:, :2], [1, 1], grid="g", space="world")
        assert exc.value.code == "E414"

    def test_S8_1_lo_gt_hi_is_refused_at_write(self):
        bad = BOXES.copy()
        bad[0, 0] = bad[0, 0][::-1]
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_boxes(bad, [1, 1])
        assert exc.value.code == "E406"

    def test_column_lengths_must_agree(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_boxes(BOXES, [1])
        assert exc.value.code == "E405"
        with pytest.raises(MEDH5ValidationError):
            encode_boxes(BOXES, [1, 1], scores=[0.5])

    def test_shape_must_be_n_s_2(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_boxes(np.zeros((3, 3)), [1, 1, 1])
        assert exc.value.code == "E405"

    def test_iterating_yields_instances(self, tmp_path, label_set):
        path = detection_sample(tmp_path / "det.medh5", label_set)
        with medh5.open(path) as sample:
            objects = list(sample.annotations["lesions"])
            assert [o.instance_id for o in objects] == [1, 2]
            assert objects[0].slices == (slice(2, 8), slice(2, 10), slice(2, 10))

    def test_S8_2_slice_index_expresses_a_2d_box_on_a_slice(self, tmp_path, label_set):
        flat = np.array([[[4.5, 4.5], [1.5, 9.5], [1.5, 9.5]]], dtype=np.float32)
        path = tmp_path / "slice.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_boxes("finding", flat, ["lesion"], grid="ct", slice_index=[5])
        with medh5.open(path) as sample:
            boxes = sample.annotations["finding"]
            assert boxes.slice_index.tolist() == [5]
            assert boxes.boxes[0, 0, 0] == boxes.boxes[0, 0, 1]


class TestObb:
    def test_S8_3_centre_size_rotation_recover_from_the_corners(
        self, tmp_path, label_set
    ):
        """The parameterisation is recoverable, which is what makes it lossless."""
        centers = np.array([[6.0, 8.0, 8.0]], dtype=np.float32)
        sizes = np.array([[4.0, 6.0, 2.0]], dtype=np.float32)
        rot = rotation_x(np.pi / 5)
        path = tmp_path / "obb_rt.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_obb("box", centers, sizes, rot[None], ["lesion"], grid="ct")
        with medh5.open(path) as sample:
            corners = sample.annotations["box"].corners()[0]
        # Corner 0 is the all-minus corner; corners 4, 2, 1 step one full edge
        # along local axes 0, 1, 2 respectively (odometer order).
        assert np.allclose(corners.mean(axis=0), centers[0], atol=1e-5)
        for axis, opposite in enumerate((4, 2, 1)):
            edge = corners[opposite] - corners[0]
            assert np.linalg.norm(edge) == pytest.approx(sizes[0, axis], abs=1e-4)
            assert np.allclose(edge / np.linalg.norm(edge), rot[:, axis], atol=1e-5)

    def test_S8_3_reader_corners_and_aabb(self, tmp_path, label_set):
        centers = np.array([[6.0, 8.0, 8.0]], dtype=np.float32)
        sizes = np.array([[4.0, 6.0, 2.0]], dtype=np.float32)
        rot = rotation_x(np.pi / 5)
        path = tmp_path / "obb.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_obb("box", centers, sizes, rot[None], ["lesion"], grid="ct")
        with medh5.open(path) as sample:
            obb = sample.annotations["box"]
            corners = obb.corners()
            assert corners.shape == (1, 8, 3)
            assert np.allclose(corners.mean(axis=1), centers, atol=1e-5)
            recovered = np.linalg.norm(corners[0][4] - corners[0][0])
            assert recovered == pytest.approx(sizes[0, 0], abs=1e-4)
            aabb = obb.as_aabb()
            assert aabb.shape == (1, 3, 2)
            assert np.all(aabb[:, :, 1] >= aabb[:, :, 0])
            assert obb.volumes[0] == pytest.approx(4.0 * 6.0 * 2.0)
            assert len(obb) == 1

    def test_S8_3_an_axis_aligned_obb_matches_its_aabb(self, tmp_path, label_set):
        """With R = I the oriented box and its enclosing box coincide."""
        centers = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        sizes = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)
        path = tmp_path / "aligned.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_obb(
                "box",
                centers,
                sizes,
                np.eye(3, dtype=np.float32)[None],
                ["liver"],
                grid="ct",
            )
        with medh5.open(path) as sample:
            aabb = sample.annotations["box"].as_aabb()[0]
        assert np.allclose(aabb[:, 0], centers[0] - sizes[0] / 2)
        assert np.allclose(aabb[:, 1], centers[0] + sizes[0] / 2)

    def test_S8_3_improper_rotations_are_refused(self):
        reflection = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_obb(
                np.zeros((1, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
                reflection[None],
                [1],
            )
        assert exc.value.code == "E407"

    def test_negative_sizes_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_obb(
                np.zeros((1, 3), dtype=np.float32),
                np.full((1, 3), -1.0, dtype=np.float32),
                np.eye(3, dtype=np.float32)[None],
                [1],
            )
        assert exc.value.code == "E406"

    def test_shape_disagreement_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_obb(
                np.zeros((2, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
                np.eye(3, dtype=np.float32)[None],
                [1, 1],
            )
        assert exc.value.code == "E405"


class TestKeypoints:
    def test_S8_4_round_trip(self, tmp_path, skeleton_label_set):
        points = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
            dtype=np.float32,
        )
        path = tmp_path / "kp.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(skeleton_label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_keypoints(
                "spine",
                points,
                ["liver", "spleen"],
                ["liver", "liver"],
                grid="ct",
                visibility=np.array([[2, 1], [2, 0]], dtype=np.uint8),
                skeleton="pair",
            )
        with medh5.open(path) as sample:
            kp = sample.annotations["spine"]
            assert kp.points.shape == (2, 2, 3)
            assert kp.keypoint_class_ids.tolist() == [1, 2]
            assert kp.visibility.tolist() == [[2, 1], [2, 0]]
            assert kp.labelled().tolist() == [[True, True], [True, False]]
            assert kp.skeleton_id == "pair"
            assert kp.skeleton().edges == ((1, 2),)
            assert len(kp) == 2

    def test_S8_4_an_unknown_skeleton_is_refused(self, tmp_path, label_set):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="ct", modality="CT")
            w.add_keypoints(
                "kp",
                np.zeros((1, 2, 3), dtype=np.float32),
                [1, 2],
                [1],
                grid="ct",
                skeleton="nope",
            )
        assert exc.value.code == "E413"

    def test_visibility_and_slot_counts_are_checked(self):
        points = np.zeros((2, 3, 3), dtype=np.float32)
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_keypoints(points, [1, 2], [1, 1])
        assert exc.value.code == "E405"
        with pytest.raises(MEDH5ValidationError):
            encode_keypoints(points, [1, 2, 3], [1, 1], visibility=np.zeros((2, 2)))
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_keypoints(
                points, [1, 2, 3], [1, 1], visibility=np.full((2, 3), 9, dtype=np.uint8)
            )
        assert exc.value.code == "E411"
        with pytest.raises(MEDH5ValidationError):
            encode_keypoints(np.zeros((2, 3)), [1], [1, 1])

    def test_visibility_defaults_to_visible(self):
        payload = encode_keypoints(np.zeros((1, 2, 3), dtype=np.float32), [1, 2], [1])
        assert payload.datasets["visibility"].tolist() == [[2, 2]]


class TestPoints:
    def test_S8_5_landmarks_round_trip(self, tmp_path, label_set):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = tmp_path / "pts.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid(
                "ct",
                shape=SHAPE,
                spacing=(2.0, 1.0, 1.0),
                origin=(1.0, 1.0, 1.0),
                frame_uid="f0",
            )
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_points(
                "fixed",
                points,
                grid="ct",
                names=["apex", "carina"],
                weights=[1.0, 0.5],
                correspondence="moving",
            )
            w.add_points(
                "moving",
                points + 1.0,
                grid="ct",
                names=["apex", "carina"],
                correspondence="fixed",
            )
        with medh5.open(path) as sample:
            pts = sample.annotations["fixed"]
            assert pts.names == ("apex", "carina")
            assert pts.weights.tolist() == [1.0, 0.5]
            assert pts.correspondence == "moving"
            assert np.allclose(pts.named()["apex"], points[0])
            assert len(pts) == 2
            world = pts.world_points()
            assert np.allclose(world[0], sample.grids["ct"].index_to_world(points[0]))
            assert pts.object_class_ids.tolist() == [0, 0]

    def test_shape_and_length_checks(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_points(np.zeros((2, 2, 2)))
        assert exc.value.code == "E405"
        with pytest.raises(MEDH5ValidationError):
            encode_points(np.zeros((2, 3)), names=["only-one"])


class TestContours:
    def test_S8_6_polygons_and_holes(self, tmp_path, label_set):
        outer = np.array(
            [[4.0, 4.0, 4.0], [4.0, 4.0, 9.0], [4.0, 9.0, 9.0]], dtype=np.float32
        )
        hole = np.array(
            [[4.0, 6.0, 6.0], [4.0, 6.0, 7.0], [4.0, 7.0, 7.0]], dtype=np.float32
        )
        path = tmp_path / "rt.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_contours(
                "rtstruct",
                [
                    Polygon(outer, class_id=1, plane=(0, 4), role="outer"),
                    Polygon(hole, class_id=1, plane=(0, 4), role="hole"),
                ],
                grid="ct",
            )
        with medh5.open(path) as sample:
            contours = sample.annotations["rtstruct"]
            assert len(contours) == 2
            assert np.array_equal(contours.polygon(0), outer)
            assert contours.roles == ("outer", "hole")
            assert contours.by_plane() == {(0, 4): [0, 1]}
            assert [p.class_id for p in contours.polygons()] == [1, 1]
            assert contours.offsets.tolist() == [0, 3, 6]

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            Polygon(np.zeros((3, 3)), class_id=1, role="inner")

    def test_empty_and_ragged_input_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_contours([])
        assert exc.value.code == "E410"
        with pytest.raises(MEDH5ValidationError):
            encode_contours(
                [Polygon(np.zeros((3, 3)), 1), Polygon(np.zeros((3, 2)), 1)]
            )


class TestMesh:
    def test_S8_7_surface_round_trip(self, tmp_path, label_set):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        path = tmp_path / "mesh.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="f0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_mesh(
                "liver_surface",
                vertices,
                faces,
                grid="ct",
                space="world",
                normals=vertices,
                mesh_class_ids=["liver"],
            )
        with medh5.open(path) as sample:
            mesh = sample.annotations["liver_surface"]
            assert np.array_equal(mesh.vertices, vertices)
            assert np.array_equal(mesh.faces, faces)
            assert mesh.normals is not None
            assert mesh.object_class_ids.tolist() == [1]
            assert mesh.n_submeshes == 1
            assert mesh.bounds().shape == (3, 2)
            assert len(mesh) == 4

    def test_S8_7_a_mesh_alone_does_not_satisfy_seg(self, tmp_path, label_set):
        vertices = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        path = tmp_path / "mesh_only.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="f0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_mesh(
                "surface",
                vertices,
                faces,
                grid="ct",
                space="world",
                mesh_class_ids=["liver"],
            )
        with medh5.open(path) as sample:
            assert "seg" not in sample.profiles

    def test_out_of_range_faces_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_mesh(np.zeros((3, 3)), np.array([[0, 1, 9]]))
        assert exc.value.code == "E405"

    def test_shape_checks(self):
        with pytest.raises(MEDH5ValidationError):
            encode_mesh(np.zeros((3, 2)), np.array([[0, 1, 2]]))
        with pytest.raises(MEDH5ValidationError):
            encode_mesh(np.zeros((3, 3)), np.array([[0, 1]]))
        with pytest.raises(MEDH5ValidationError):
            encode_mesh(
                np.zeros((3, 3)), np.array([[0, 1, 2]]), normals=np.zeros((2, 3))
            )


class TestSpace:
    def test_unknown_space_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            check_space("hyperbolic")
        assert exc.value.code == "E412"

    def test_summary_is_json_safe(self, tmp_path, label_set):
        import json

        path = detection_sample(tmp_path / "det.medh5", label_set)
        with medh5.open(path) as sample:
            json.dumps(sample.annotations["lesions"].summary())
            json.dumps(sample.summary())


class TestSliceIndexBoxes:
    def test_S8_2_a_2d_box_on_a_slice_selects_that_slice(self, tmp_path, label_set):
        """§8.2's canonical form selected no voxels at all.

        `slice_index` with a degenerate axis expresses "a 2D box on slice k",
        the common radiology annotation -- and `as_slices` never read
        `slice_index`, so the degenerate axis became a zero-thickness slice.
        """
        shape = (8, 16, 16)
        path = tmp_path / "sliced.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_boxes(
                "det",
                grid="g",
                boxes=[[[3.0, 3.0], [1.5, 5.5], [1.5, 5.5]]],
                class_ids=[3],
                space="index",
                slice_index=[3],
            )
        with medh5.open(path) as sample:
            slices = sample.annotations["det"].as_slices()[0]
        assert slices[0] == slice(3, 4), "the named slice, one voxel thick"
        assert int(np.prod([s.stop - s.start for s in slices])) == 16

    def test_S8_2_a_short_slice_index_is_refused_at_write(self, tmp_path, label_set):
        """One entry per box, or the boxes past the end vanish.

        `slice_index` is appended after `_object_columns`, which validates the
        per-box columns it builds and never sees this one -- so a short one was
        written without complaint, and every box it did not reach stayed a
        degenerate zero-thickness slice selecting no voxels.
        """
        shape = (8, 16, 16)
        with (
            pytest.raises(MEDH5ValidationError, match="one value for each") as caught,
            medh5.create(tmp_path / "short.medh5", codec="portable") as w,
        ):
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_boxes(
                "det",
                grid="g",
                boxes=[
                    [[3.0, 3.0], [1.5, 5.5], [1.5, 5.5]],
                    [[6.0, 6.0], [8.5, 12.5], [8.5, 12.5]],
                ],
                class_ids=[3, 3],
                space="index",
                slice_index=[3],
            )
        assert caught.value.code == "E405"

    @pytest.mark.parametrize(
        ("kind", "kwargs", "column"),
        [
            ("points", {"class_ids": [3]}, "class_ids"),
            ("points", {"class_ids": [3, 3, 3], "weights": [0.5]}, "weights"),
            ("points", {"class_ids": [3, 3, 3], "names": ["a"]}, "names"),
            ("mesh", {"vertex_class_ids": [3]}, "vertex_class_ids"),
            (
                "mesh",
                {"mesh_offsets": [0, 1, 2], "mesh_class_ids": [3]},
                "mesh_class_ids",
            ),
        ],
    )
    def test_S8_every_per_element_column_is_checked(
        self, tmp_path, label_set, kind, kwargs, column
    ):
        """`slice_index` was unchecked because it is written after
        `_object_columns`, which validates only the columns it builds itself.

        Five more columns were added the same way and went the same way: a short
        one wrote cleanly and validated clean, leaving every element past its end
        silently unlabelled. They all route through one helper now, because the
        checks that existed were exactly the ones somebody remembered to write.
        """
        shape = (8, 16, 16)
        with (
            pytest.raises(MEDH5ValidationError, match=column) as caught,
            medh5.create(tmp_path / f"{column}.medh5", codec="portable") as w,
        ):
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            if kind == "points":
                w.add_points(
                    "p",
                    grid="g",
                    space="index",
                    points=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                    **kwargs,
                )
            else:
                w.add_mesh(
                    "m",
                    grid="g",
                    space="index",
                    vertices=[
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    faces=[[0, 1, 2], [1, 2, 3]],
                    **kwargs,
                )
        assert caught.value.code == "E405"

    def test_S8_2_a_short_slice_index_already_in_a_file_is_refused(
        self, tmp_path, label_set
    ):
        """Files predating the writer check exist, so reading refuses too.

        Skipping the boxes `slice_index` does not reach -- the bounds guard the
        reader used to carry -- returns each of them as a zero-thickness slice,
        so the annotation quietly yields fewer objects than it holds.
        """
        import h5py

        from medh5.validate import validate_file

        shape = (8, 16, 16)
        path = tmp_path / "legacy.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_boxes(
                "det",
                grid="g",
                boxes=[
                    [[3.0, 3.0], [1.5, 5.5], [1.5, 5.5]],
                    [[6.0, 6.0], [8.5, 12.5], [8.5, 12.5]],
                ],
                class_ids=[3, 3],
                space="index",
                slice_index=[3, 6],
            )
        with h5py.File(path, "a") as handle:
            group = handle["annotations/det"]
            del group["slice_index"]
            group.create_dataset("slice_index", data=np.array([3], np.int32))

        report = validate_file(path)
        assert not report.ok
        assert [e.code for e in report.errors] == ["E405"]
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as caught:
            sample.annotations["det"].as_slices()
        assert caught.value.code == "E405"

    def test_S8_2_slice_index_does_not_touch_a_box_with_real_extent(
        self, tmp_path, label_set
    ):
        """Only the 2D-on-a-slice form is reinterpreted, never a 3D box."""
        shape = (8, 16, 16)
        path = tmp_path / "solid.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_boxes(
                "det",
                grid="g",
                boxes=[[[1.5, 4.5], [1.5, 5.5], [1.5, 5.5]]],
                class_ids=[3],
                space="index",
                slice_index=[3],
            )
        with medh5.open(path) as sample:
            slices = sample.annotations["det"].as_slices()[0]
        assert slices[0] == slice(2, 5)
