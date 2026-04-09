"""Tests for in-place update operations."""

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
