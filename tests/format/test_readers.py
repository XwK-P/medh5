"""The reader API each encoding exposes (spec §6, §7).

Every voxel encoding answers the same questions; these tests ask them of all of
them, so a divergence in behaviour between encodings fails here rather than in a
consumer's training loop.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.base import AnnotationHeader
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError
from tests.helpers import SHAPE, write_sample
from tests.kits import Flat, Framed, Organs

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

    def test_S7_5_the_threshold_is_written_read_and_addressed(self, tmp_path: Path):
        path = tmp_path / "soft.medh5"
        soft = np.zeros(Flat.SHAPE, dtype=np.float32)
        soft[Flat.mask()] = 0.4
        with Flat.open_writer(path) as w:
            w.add_segmentation("soft", grid="g", probabilities={3: soft}, threshold=0.3)
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["soft"]
            assert ann.threshold == pytest.approx(0.3)
            assert float(ann.group.attrs["threshold"]) == pytest.approx(0.3)
            # 0.4 is above 0.3 and below the 0.5 default: the declaration decides.
            assert ann.contains(3, (3, 3, 3))
            assert ann.dense([3])[0].sum() == int(Flat.mask().sum())
            with_threshold = sample.content_id
        plain = tmp_path / "plain.medh5"
        with Flat.open_writer(plain) as w:
            w.add_segmentation("soft", grid="g", probabilities={3: soft})
        with medh5.open(plain) as sample:
            assert not sample.annotations["soft"].contains(3, (3, 3, 3))
            assert "threshold" not in sample.annotations["soft"].group.attrs
            # It is a spec attribute, so it is part of the address (§13.2).
            assert sample.content_id != with_threshold

    def test_S7_5_threshold_applies_to_probabilities_only(self, tmp_path: Path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Flat.open_writer(tmp_path / "x.medh5") as w,
        ):
            w.add_segmentation("s", grid="g", masks={3: Flat.mask()}, threshold=0.3)
        assert exc.value.code == "E404"
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Flat.open_writer(tmp_path / "y.medh5") as w,
        ):
            w.add_segmentation(
                "s", grid="g", probabilities={3: np.zeros(Flat.SHAPE)}, threshold=1.5
            )
        assert exc.value.code == "E404"


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
        from medh5.geometry import slices_to_box

        box = slices_to_box([slice(1, 4), slice(1, 4), slice(1, 4)])
        payload = encode_instances(
            [InstanceInput(3, 1, box=box)], SHAPE, store_masks=False
        )
        assert "mask_data" not in payload.datasets
        from medh5.annotations.voxel import payload_to_masks

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


class TestLabelmapWarnsOnOverlap:
    def test_S7_0_flattening_an_overlapping_annotation_says_so(
        self, tmp_path, label_set
    ):
        """`layers` exists because classes overlap; one integer volume cannot.

        Three exits flatten on the way out -- NIfTI export, the torch loader's
        `labelmap` format, and the MONAI bridge -- and each was deleting the
        overlap region silently: a lesion inside a liver stopped being liver.
        """
        shape = (8, 16, 16)
        liver = np.zeros(shape, bool)
        liver[2:6, 4:12, 4:12] = True
        lesion = np.zeros(shape, bool)
        lesion[3:5, 6:10, 6:10] = True
        path = tmp_path / "overlap.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "seg", grid="g", masks={1: liver, 3: lesion}, annotated_classes=[1, 3]
            )

        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            dense = annotation.dense([1, 3])
            overlap = int((dense[0] & dense[1]).sum())
            assert overlap > 0, "the fixture must actually overlap"

            with pytest.warns(UserWarning, match="overlapping voxel"):
                flat = annotation.labelmap()
            assert int((flat == 1).sum()) == int(dense[0].sum()) - overlap

            # An explicit priority is the caller making the decision, so it is
            # not a surprise and does not warn.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                annotation.labelmap(priority=[3])


class TestBatchedDense:
    """Reading the planes of many classes at once returns what per-class reads do."""

    @pytest.mark.parametrize("encoding", ["layers", "bitmask"])
    def test_batched_reads_agree_with_per_class_reads(
        self, tmp_path, label_set, masks, encoding
    ):
        path = write_sample(
            tmp_path / f"{encoding}.medh5",
            label_set=label_set,
            masks=masks,
            encoding=encoding,
        )
        with medh5.open(path) as sample:
            ann = sample.annotations["organs_tp0"]
            assert ann.kind == encoding
            wanted = list(ann.class_ids)
            roi = [slice(2, 10), slice(2, 12), slice(2, 12)]
            batched = ann.dense(wanted, roi=roi)
            one_at_a_time = np.stack(
                [ann.dense([class_id], roi=roi)[0] for class_id in wanted]
            )
            assert np.array_equal(batched, one_at_a_time)
            assert np.array_equal(ann.dense(wanted)[0], masks[wanted[0]])

    def test_an_unstored_class_reads_as_empty(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "layers.medh5",
            label_set=label_set,
            masks=masks,
            encoding="layers",
        )
        with medh5.open(path) as sample:
            ann = sample.annotations["organs_tp0"]
            assert not ann.dense([4])[0].any()


class TestFastPathsAreExact:
    """A read optimisation returns what the slow path did, within its budget."""

    def test_S7_4_instance_columns_are_read_once(self, tmp_path: Path):
        path = tmp_path / "inst.medh5"
        with Flat.open_writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(
                        class_id=3, instance_id=7, mask=Flat.mask(), score=0.5
                    ),
                    InstanceInput(
                        class_id=3, instance_id=8, mask=Flat.mask((5, 5, 5), 2)
                    ),
                ],
            )
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["les"]
            first = ann.boxes
            assert ann.instance_ids is ann.instance_ids
            assert ann.boxes is first
            assert [o.instance_id for o in ann.instances()] == [7, 8]
            assert ann.dense([3])[0].sum() == int(Flat.mask().sum() + 8)
            assert ann.scores is not None and np.isnan(ann.scores[1])
            crop = ann.crop(1)
            assert crop is not None and crop.shape == (2, 2, 2) and crop.all()

    def test_S7_7_labelmap_and_mask_scan_in_slabs(self, tmp_path: Path):
        """The slab budget is the engine's, held to one row per slab by
        `annotations::payload::tests::s7_7_scans_reach_the_last_slab`."""
        ignore = np.zeros(Flat.SHAPE, dtype=bool)
        ignore[-1] = True
        path = tmp_path / "slabs.medh5"
        with Flat.open_writer(path) as w:
            w.add_segmentation(
                "lm",
                grid="g",
                masks={3: Flat.mask()},
                encoding="labelmap",
                ignore=ignore,
            )
            w.add_mask("fov", Flat.mask((0, 0, 0), 4), grid="g")
        with medh5.open(path) as sample:
            assert sample.annotations["lm"].has_ignore_region
            assert sample.annotations["fov"].summary()["true_voxels"] == 64

    def test_P05_the_class_table_is_read_once_per_annotation(self, tmp_path: Path):
        """`dense()` asked the table once per class: 63 reads for 63 classes."""

        masks = {c: np.zeros(Organs.SHAPE, dtype=bool) for c in range(1, 4)}
        for c, mask in masks.items():
            mask[c, c:, c:] = True
        for encoding, table in (
            ("layers", "layer_class_ids"),
            ("bitmask", "bit_class_ids"),
        ):
            path = tmp_path / f"{encoding}.medh5"
            with Organs.writer(path) as w:
                w.add_segmentation("seg", grid="g", masks=masks, encoding=encoding)

            reads = 0
            original = h5py.Dataset.__getitem__

            def counting(
                self: Any, key: Any, _name: str = table, _original: Any = original
            ) -> Any:
                nonlocal reads
                if self.name.endswith(_name):
                    reads += 1
                return _original(self, key)

            with medh5.open(path) as sample:
                annotation = sample.annotations["seg"]
                h5py.Dataset.__getitem__ = counting  # type: ignore[method-assign]
                try:
                    annotation.dense(list(annotation.class_ids))
                finally:
                    h5py.Dataset.__getitem__ = original  # type: ignore[method-assign]
            assert reads <= 1, f"{encoding}: {reads} reads of {table}"

    @pytest.mark.parametrize(
        "encoding", ["labelmap", "layers", "bitmask", "instances", "probmap"]
    )
    def test_P07_voxel_counts_agree_with_the_generic_path(
        self, tmp_path: Path, encoding: str
    ):
        path = tmp_path / f"{encoding}.medh5"
        blocks = Organs.blocks()
        with Organs.writer(path) as w:
            if encoding == "probmap":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    probabilities={c: m.astype(np.float32) for c, m in blocks.items()},
                )
            elif encoding == "instances":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    instances=[
                        InstanceInput(class_id=c, instance_id=i + 1, mask=m)
                        for i, (c, m) in enumerate(blocks.items())
                    ],
                )
            elif encoding == "labelmap":
                w.add_segmentation(
                    "seg", grid="g", masks={1: blocks[1]}, encoding="labelmap"
                )
            else:
                w.add_segmentation("seg", grid="g", masks=blocks, encoding=encoding)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            fast = annotation.voxel_counts()
            slow = {
                c: int(annotation.dense([c])[0].sum()) for c in annotation.class_ids
            }
            assert fast == slow

    def test_P07_an_empty_class_still_counts_zero(self, tmp_path: Path):
        """The coverage contract: examined and absent is not the same as missing."""
        path = tmp_path / "empty.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks=Organs.blocks(),
                encoding="layers",
                annotated_classes=[1, 2, 3],
            )
        with medh5.open(path) as sample:
            counts = sample.annotations["seg"].voxel_counts()
            assert counts[2] == 0
            assert set(counts) == {1, 2, 3}

    @pytest.mark.parametrize("encoding", ["layers", "bitmask"])
    def test_P09_S14_plane_counts_read_within_the_slab_budget(
        self, tmp_path: Path, encoding: str
    ):
        """`data[layer]` read the whole plane before the slab loop bounded nothing.

        The read budget is the engine's: its Rust tests
        (`annotations::payload::tests::p09_s14_*`) hold every planned read to a
        4 KiB budget, tiling the dataset exactly, and the counts to the
        full-budget ones.  Here: the counts a reader gets are exact."""
        shape = (16, 32, 32)
        first = np.zeros(shape, bool)
        first[2:10, 2:20, 2:20] = True
        second = first.copy()
        second[:, :, 10:30] = True  # overlaps: two layers, or two bits
        path = tmp_path / f"{encoding}.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1))
        w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        w.label_set(Framed.label_set(2))
        w.add_segmentation(
            "seg", grid="g", masks={1: first, 2: second}, encoding=encoding
        )
        w.commit()

        with medh5.open(path) as s:
            annotation = s.annotations["seg"]
            assert annotation.kind == encoding
            counts = annotation.voxel_counts()
        assert counts[1] == int(first.sum()) and counts[2] == int(second.sum())
