"""Tests for in-place update operations."""

from pathlib import Path

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "update.medh5"
    images = {"CT": np.zeros((8, 16, 16), dtype=np.float32)}
    MEDH5File.write(path, images=images, label=1, extra={"key": "old"})
    return path


class TestUpdateMeta:
    def test_update_label(self, sample_file):
        MEDH5File.update_meta(sample_file, label=42)
        meta = MEDH5File.read_meta(sample_file)
        assert meta.label == 42
        assert meta.extra == {"key": "old"}

    def test_update_extra(self, sample_file):
        MEDH5File.update_meta(sample_file, extra={"key": "new", "added": True})
        meta = MEDH5File.read_meta(sample_file)
        assert meta.extra == {"key": "new", "added": True}
        assert meta.label == 1

    def test_clear_label(self, sample_file):
        MEDH5File.update_meta(sample_file, label=None)
        meta = MEDH5File.read_meta(sample_file)
        assert meta.label is None

    def test_clear_extra(self, sample_file):
        MEDH5File.update_meta(sample_file, extra=None)
        meta = MEDH5File.read_meta(sample_file)
        assert meta.extra is None

    def test_update_label_name(self, sample_file):
        MEDH5File.update_meta(sample_file, label_name="benign")
        meta = MEDH5File.read_meta(sample_file)
        assert meta.label_name == "benign"

    def test_unified_update_meta_and_spatial(self, sample_file):
        MEDH5File.update(
            sample_file,
            meta={
                "label": 7,
                "label_name": "reviewed",
                "extra": {"key": "new"},
                "spacing": [1.0, 2.0, 3.0],
                "coord_system": "RAS",
                "patch_size": [4, 8, 8],
            },
        )
        meta = MEDH5File.read_meta(sample_file)
        assert meta.label == 7
        assert meta.label_name == "reviewed"
        assert meta.extra == {"key": "new"}
        assert meta.spatial.spacing == [1.0, 2.0, 3.0]
        assert meta.spatial.coord_system == "RAS"
        assert meta.patch_size == [4, 8, 8]


class TestAddSeg:
    def test_add_seg_to_file_without_seg(self, sample_file):
        mask = np.zeros((8, 16, 16), dtype=bool)
        mask[2:6, 4:12, 4:12] = True

        MEDH5File.add_seg(sample_file, "tumor", mask)

        sample = MEDH5File.read(sample_file)
        assert sample.seg is not None
        assert "tumor" in sample.seg
        np.testing.assert_array_equal(sample.seg["tumor"], mask)
        assert sample.meta.has_seg is True
        assert sample.meta.seg_names == ["tumor"]

    def test_add_second_seg(self, sample_file):
        mask1 = np.zeros((8, 16, 16), dtype=bool)
        mask2 = np.ones((8, 16, 16), dtype=bool)

        MEDH5File.add_seg(sample_file, "organ_a", mask1)
        MEDH5File.add_seg(sample_file, "organ_b", mask2)

        sample = MEDH5File.read(sample_file)
        assert set(sample.seg.keys()) == {"organ_a", "organ_b"}
        assert sample.meta.seg_names == ["organ_a", "organ_b"]

    def test_add_seg_shape_mismatch(self, sample_file):
        bad_mask = np.zeros((8, 16, 32), dtype=bool)
        with pytest.raises(MEDH5ValidationError, match="shape"):
            MEDH5File.add_seg(sample_file, "bad", bad_mask)

    def test_add_seg_duplicate_name(self, sample_file):
        mask = np.zeros((8, 16, 16), dtype=bool)
        MEDH5File.add_seg(sample_file, "tumor", mask)
        with pytest.raises(MEDH5ValidationError, match="already exists"):
            MEDH5File.add_seg(sample_file, "tumor", mask)

    def test_update_replace_and_remove_seg(self, sample_file):
        mask = np.zeros((8, 16, 16), dtype=bool)
        mask[0, 0, 0] = True
        MEDH5File.update(sample_file, seg_ops={"add": {"tumor": mask}})

        replacement = np.ones((8, 16, 16), dtype=bool)
        MEDH5File.update(sample_file, seg_ops={"replace": {"tumor": replacement}})
        sample = MEDH5File.read(sample_file)
        assert sample.seg is not None
        np.testing.assert_array_equal(sample.seg["tumor"], replacement)

        MEDH5File.update(sample_file, seg_ops={"remove": ["tumor"]})
        sample = MEDH5File.read(sample_file)
        assert sample.seg is None
        assert sample.meta.has_seg is False
        assert sample.meta.seg_names is None

    def test_failed_replace_does_not_create_empty_seg_group(self, sample_file):
        with pytest.raises(MEDH5ValidationError, match="does not exist"):
            MEDH5File.update(
                sample_file,
                seg_ops={"replace": {"tumor": np.zeros((8, 16, 16), dtype=bool)}},
            )

        sample = MEDH5File.read(sample_file)
        assert sample.seg is None
        assert sample.meta.has_seg is False
        assert sample.meta.seg_names is None

    def test_failed_remove_does_not_partially_delete_masks(self, sample_file):
        mask = np.zeros((8, 16, 16), dtype=bool)
        MEDH5File.update(
            sample_file,
            seg_ops={"add": {"tumor": mask, "organ": mask}},
        )

        with pytest.raises(MEDH5ValidationError, match="does not exist"):
            MEDH5File.update(sample_file, seg_ops={"remove": ["tumor", "missing"]})

        sample = MEDH5File.read(sample_file)
        assert sample.seg is not None
        assert set(sample.seg) == {"tumor", "organ"}

    def test_update_bbox_payload(self, sample_file):
        bboxes = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.int64)
        scores = np.array([0.9], dtype=np.float32)
        labels = ["tumor"]
        MEDH5File.update(
            sample_file,
            bbox_ops={
                "bboxes": bboxes,
                "bbox_scores": scores,
                "bbox_labels": labels,
            },
        )
        sample = MEDH5File.read(sample_file)
        np.testing.assert_array_equal(sample.bboxes, bboxes)
        np.testing.assert_array_equal(sample.bbox_scores, scores)
        assert sample.bbox_labels == labels

        MEDH5File.update(sample_file, bbox_ops={"clear": True})
        sample = MEDH5File.read(sample_file)
        assert sample.bboxes is None
        assert sample.meta.has_bbox is False


class TestOnReopenedCallback:
    def test_update_fires_callback_on_success(self, sample_file):
        seen: list[Path] = []
        MEDH5File.update(sample_file, meta={"label": 7}, on_reopened=seen.append)
        assert seen == [sample_file]

    def test_update_meta_forwards_callback(self, sample_file):
        seen: list[Path] = []
        MEDH5File.update_meta(sample_file, label=9, on_reopened=seen.append)
        assert seen == [sample_file]

    def test_add_seg_forwards_callback(self, sample_file):
        seen: list[Path] = []
        mask = np.zeros((8, 16, 16), dtype=bool)
        MEDH5File.add_seg(sample_file, "tumor", mask, on_reopened=seen.append)
        assert seen == [sample_file]

    def test_callback_not_fired_on_failure(self, sample_file):
        seen: list[Path] = []
        # Trigger MEDH5ValidationError during update via unknown meta key.
        with pytest.raises(MEDH5ValidationError):
            MEDH5File.update(sample_file, meta={"bogus": 1}, on_reopened=seen.append)
        assert seen == []

    def test_set_review_status_fires_callback(self, sample_file):
        from medh5 import ReviewStatus

        seen: list[Path] = []
        result = MEDH5File.set_review_status(
            sample_file,
            status="reviewed",
            annotator="qa",
            on_reopened=seen.append,
        )
        assert seen == [sample_file]
        assert isinstance(result, ReviewStatus)
