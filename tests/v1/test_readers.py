"""The reader API each encoding exposes (spec §6, §7).

Every voxel encoding answers the same questions; these tests ask them of all of
them, so a divergence in behaviour between encodings fails here rather than in a
consumer's training loop.
"""

from __future__ import annotations

import numpy as np
import pytest

import medh5
from medh5.annotations.base import AnnotationHeader
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError
from tests.v1.conftest import SHAPE, write_sample

ENCODINGS = ("layers", "bitmask", "labelmap", "instances", "probmap")


def disjoint_masks() -> dict[int, np.ndarray]:
    """Classes that never overlap, so every encoding can represent them."""
    masks = {}
    for i, class_id in enumerate((1, 2, 3)):
        mask = np.zeros(SHAPE, dtype=bool)
        mask[2:8, i * 6 : i * 6 + 5, 2:8] = True
        masks[class_id] = mask
    return masks


@pytest.fixture(params=ENCODINGS)
def encoded(request, tmp_path, label_set):
    """One sample per encoding, all carrying identical ground truth."""
    masks = disjoint_masks()
    if request.param == "probmap":
        path = tmp_path / f"{request.param}.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "organs",
                grid="g",
                probabilities={k: v.astype(np.float32) for k, v in masks.items()},
            )
    else:
        path = write_sample(
            tmp_path / f"{request.param}.medh5",
            label_set=label_set,
            masks=masks,
            encoding=request.param,
        )
    return request.param, path, masks


class TestUniformContract:
    def test_S7_6_contains_agrees_across_encodings(self, encoded):
        """§7.6: `contains` MUST behave identically regardless of `kind`."""
        kind, path, masks = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        with medh5.open(path) as sample:
            seg = sample.annotations[name]
            assert seg.kind == kind
            for class_id, mask in masks.items():
                inside = tuple(int(v) for v in np.argwhere(mask)[0])
                outside = tuple(int(v) for v in np.argwhere(~mask)[0])
                assert seg.contains(class_id, inside)
                assert not seg.contains(class_id, outside)

    def test_dense_agrees_across_encodings(self, encoded):
        kind, path, masks = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        with medh5.open(path) as sample:
            seg = sample.annotations[name]
            dense = seg.dense(sorted(masks))
            for i, class_id in enumerate(sorted(masks)):
                assert np.array_equal(dense[i], masks[class_id])

    def test_roi_slicing_agrees_across_encodings(self, encoded):
        kind, path, masks = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        roi = np.s_[2:8, 0:12, 2:8]
        with medh5.open(path) as sample:
            seg = sample.annotations[name]
            windowed = seg.dense([1], roi=roi)[0]
            assert np.array_equal(windowed, masks[1][roi])

    def test_voxel_counts_and_bboxes_agree(self, encoded):
        kind, path, masks = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        with medh5.open(path) as sample:
            seg = sample.annotations[name]
            counts = seg.voxel_counts()
            for class_id, mask in masks.items():
                assert counts[class_id] == int(mask.sum())
            boxes = seg.class_bboxes([1])
            assert boxes[1] is not None

    def test_labelmap_flattening_agrees(self, encoded):
        kind, path, masks = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        with medh5.open(path) as sample:
            flat = sample.annotations[name].labelmap()
            for class_id, mask in masks.items():
                assert set(np.unique(flat[mask])) == {class_id}

    def test_summary_is_json_safe(self, encoded):
        import json

        kind, path, _ = encoded
        name = "organs" if kind == "probmap" else "organs_tp0"
        with medh5.open(path) as sample:
            json.dumps(sample.annotations[name].summary())


class TestPriorityFlattening:
    def test_S7_2_overlap_ties_are_broken_explicitly(self, sample_path):
        """Flattening is lossy, so which class survives is the caller's call."""
        with medh5.open(sample_path) as sample:
            seg = sample.annotations["organs_tp0"]
            liver_first = seg.labelmap(priority=["liver"])
            lesion_first = seg.labelmap(priority=["lesion"])
            overlap = seg.dense([1])[0] & seg.dense([3])[0]
            assert overlap.any()
            assert set(np.unique(liver_first[overlap])) == {1}
            assert set(np.unique(lesion_first[overlap])) == {3}


class TestLayers:
    def test_layer_table_round_trips(self, sample_path):
        with medh5.open(sample_path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.n_layers >= 1
            assert sum(len(bucket) for bucket in seg.layer_classes()) == len(
                seg.class_ids
            )
            assert set(seg.layer_of) == set(seg.class_ids)
            assert seg.read_layer(0).shape == SHAPE
            assert seg.summary()["layers"] == seg.n_layers

    def test_a_class_in_two_layers_is_refused_on_read(self, sample_path):
        import h5py

        with h5py.File(sample_path, "r+") as handle:
            table = np.asarray(handle["annotations/organs_tp0/layer_class_ids"][...])
            table[1, 0] = table[0, 0]
            handle["annotations/organs_tp0/layer_class_ids"][...] = table
        with (
            medh5.open(sample_path) as sample,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            _ = sample.annotations["organs_tp0"].layer_of
        assert exc.value.code == "E404"

    def test_an_unknown_class_reads_as_empty(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert not sample.annotations["organs_tp0"].dense([4])[0].any()


class TestBitmask:
    def test_S7_3_classes_at_a_voxel_is_one_pass(self, tmp_path, label_set):
        masks = disjoint_masks()
        path = write_sample(
            tmp_path / "b.medh5", label_set=label_set, masks=masks, encoding="bitmask"
        )
        with medh5.open(path) as sample:
            seg = sample.annotations["organs_tp0"]
            voxel = tuple(int(v) for v in np.argwhere(masks[2])[0])
            assert seg.classes_at(voxel) == (2,)
            empty = tuple(
                int(v)
                for v in np.argwhere(~np.logical_or.reduce(list(masks.values())))[0]
            )
            assert seg.classes_at(empty) == ()
            assert seg.n_planes == 1
            assert seg.summary()["planes"] == 1

    def test_unknown_class_reads_as_empty(self, tmp_path, label_set):
        path = write_sample(
            tmp_path / "b.medh5",
            label_set=label_set,
            masks=disjoint_masks(),
            encoding="bitmask",
        )
        with medh5.open(path) as sample:
            assert not sample.annotations["organs_tp0"].dense([4])[0].any()


class TestProbmap:
    def test_probabilities_are_readable_and_thresholded(self, tmp_path, label_set):
        rng = np.random.default_rng(0)
        path = tmp_path / "p.medh5"
        soft = {1: rng.random(SHAPE).astype(np.float32)}
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation("soft", grid="g", probabilities=soft)
        with medh5.open(path) as sample:
            seg = sample.annotations["soft"]
            values = seg.probabilities([1])
            assert values.shape == (1, *SHAPE)
            assert np.array_equal(seg.dense([1])[0], values[0] >= seg.threshold)
            assert not seg.normalized
            assert not seg.probabilities([4]).any()
            assert seg.summary()["threshold"] == 0.5


class TestInstances:
    def test_objects_decode_with_boxes_and_crops(self, tmp_path, label_set):
        def lesion(origin):
            mask = np.zeros(SHAPE, dtype=bool)
            mask[tuple(slice(o, o + 4) for o in origin)] = True
            return mask

        path = tmp_path / "i.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "lesions",
                grid="g",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1)), score=0.8),
                    InstanceInput(3, 2, mask=lesion((8, 8, 8))),
                ],
            )
        with medh5.open(path) as sample:
            seg = sample.annotations["lesions"]
            assert seg.n_objects == 2
            assert seg.has_masks
            objects = list(seg.instances())
            assert objects[0].score == pytest.approx(0.8)
            assert objects[1].score is None
            assert objects[0].voxel_count == 64
            assert objects[0].slices == (slice(1, 5), slice(1, 5), slice(1, 5))
            assert "Instance(id=1" in repr(objects[0])
            assert seg.tracking() == {1: 3, 2: 3}
            assert seg.summary()["objects"] == 2
            with pytest.raises(KeyError):
                seg.instance(99)

    def test_box_only_instances_paint_the_whole_box(self, tmp_path, label_set):
        from medh5.annotations.voxel import encode_instances
        from medh5.geometry.affine import slices_to_box

        box = slices_to_box([slice(1, 4), slice(1, 4), slice(1, 4)])
        payload = encode_instances(
            [InstanceInput(3, 1, box=box)], SHAPE, store_masks=False
        )
        assert "mask_data" not in payload.datasets
        from medh5.annotations.voxel.transcode import payload_to_masks

        decoded = payload_to_masks(payload, spatial_shape=SHAPE)
        assert decoded[3][1:4, 1:4, 1:4].all()
        assert decoded[3].sum() == 27


class TestMaskAnnotation:
    def test_a_mask_carries_no_classes(self, tmp_path, label_set):
        fov = np.zeros(SHAPE, dtype=bool)
        fov[2:10] = True
        path = tmp_path / "m.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_mask("fov", fov, grid="g")
        with medh5.open(path) as sample:
            mask = sample.annotations["fov"]
            assert mask.kind == "mask"
            assert mask.class_ids == ()
            assert np.array_equal(mask.read(), fov)
            assert mask.dense().shape == (1, *SHAPE)
            assert mask.summary()["true_voxels"] == int(fov.sum())

    def test_mask_shape_must_match_the_grid(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
            w.add_mask("fov", np.zeros((2, 2, 2), dtype=bool), grid="g")
        assert exc.value.code == "E405"


class TestErrorPaths:
    def test_a_reserved_kind_is_refused_by_the_header(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            AnnotationHeader(kind="rle", task="segmentation")
        assert exc.value.code == "E401"

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            AnnotationHeader(kind="runes", task="segmentation")
        assert exc.value.code == "E401"

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            AnnotationHeader(kind="layers", task="divination")
        assert exc.value.code == "E412"

    def test_a_reserved_class_id_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            AnnotationHeader(kind="layers", task="segmentation", class_ids=(0,))
        assert exc.value.code == "E303"

    def test_class_names_need_a_label_set(self, tmp_path, label_set):
        path = tmp_path / "n.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation("s", grid="g", masks=disjoint_masks())
        with medh5.open(path) as sample:
            seg = sample.annotations["s"]
            assert seg.classes == ()
            assert seg.class_key(1) == "1"
            with pytest.raises(MEDH5ValidationError, match="label set"):
                seg.contains("liver", (0, 0, 0))

    def test_a_roi_of_the_wrong_rank_is_refused(self, sample_path):
        with (
            medh5.open(sample_path) as sample,
            pytest.raises(MEDH5ValidationError, match="roi"),
        ):
            sample.annotations["organs_tp0"].dense([1], roi=[slice(0, 2)])

    def test_instances_are_the_only_kind_with_object_identity(self, sample_path):
        with (
            medh5.open(sample_path) as sample,
            pytest.raises(MEDH5ValidationError, match="instance identity"),
        ):
            list(sample.annotations["organs_tp0"].instances())

    def test_an_annotation_without_a_grid_says_so(self):
        header = AnnotationHeader(kind="classification", task="classification")
        assert header.grid is None
