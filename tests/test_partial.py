"""Test lazy / partial reads via MEDH5File.open() and context manager."""

import numpy as np
import pytest

from medh5 import MEDH5File


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "partial.medh5"
    rng = np.random.default_rng(99)
    images = {
        "CT": rng.random((32, 64, 64), dtype=np.float32),
        "MRI_T1": rng.random((32, 64, 64), dtype=np.float32),
    }
    seg = {
        "organ": rng.random((32, 64, 64)) > 0.6,
        "lesion": rng.random((32, 64, 64)) > 0.9,
    }
    MEDH5File.write(
        path,
        images=images,
        seg=seg,
        label=1,
        spacing=[1.0, 1.0, 2.0],
        patch_size=32,
    )
    return path, images, seg


class TestPartialRead:
    def test_patch_read(self, sample_file):
        path, images, _ = sample_file
        with MEDH5File.open(path) as f:
            patch = f["images/CT"][5:15, 10:30, 20:50]
        np.testing.assert_array_equal(patch, images["CT"][5:15, 10:30, 20:50])

    def test_patch_read_second_modality(self, sample_file):
        path, images, _ = sample_file
        with MEDH5File.open(path) as f:
            patch = f["images/MRI_T1"][5:15, 10:30, 20:50]
        np.testing.assert_array_equal(patch, images["MRI_T1"][5:15, 10:30, 20:50])

    def test_seg_patch_read(self, sample_file):
        path, _, seg = sample_file
        with MEDH5File.open(path) as f:
            patch = f["seg/organ"][0:8, 0:16, 0:16]
        np.testing.assert_array_equal(patch, seg["organ"][0:8, 0:16, 0:16])

    def test_meta_from_open(self, sample_file):
        path, _, _ = sample_file
        from medh5.meta import read_meta

        with MEDH5File.open(path) as f:
            meta = read_meta(f)
        assert meta.label == 1
        assert meta.spatial.spacing == [1.0, 1.0, 2.0]
        assert meta.image_names == ["CT", "MRI_T1"]

    def test_datasets_listed(self, sample_file):
        path, _, _ = sample_file
        with MEDH5File.open(path) as f:
            keys = set(f.keys())
            image_keys = set(f["images"].keys())
            seg_keys = set(f["seg"].keys())
        assert "images" in keys
        assert "seg" in keys
        assert image_keys == {"CT", "MRI_T1"}
        assert seg_keys == {"organ", "lesion"}


class TestContextManager:
    def test_context_manager_images(self, sample_file):
        path, images, _ = sample_file
        with MEDH5File(path) as f:
            patch = f.images["CT"][5:15, 10:30, 20:50]
        np.testing.assert_array_equal(patch, images["CT"][5:15, 10:30, 20:50])

    def test_context_manager_meta(self, sample_file):
        path, _, _ = sample_file
        with MEDH5File(path) as f:
            meta = f.meta
        assert meta.label == 1
        assert meta.spatial.spacing == [1.0, 1.0, 2.0]
        assert meta.image_names == ["CT", "MRI_T1"]

    def test_context_manager_seg(self, sample_file):
        path, _, seg = sample_file
        with MEDH5File(path) as f:
            assert f.seg is not None
            patch = f.seg["organ"][0:8, 0:16, 0:16]
        np.testing.assert_array_equal(patch, seg["organ"][0:8, 0:16, 0:16])

    def test_context_manager_no_seg(self, tmp_path):
        path = tmp_path / "noseg.medh5"
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        MEDH5File.write(path, images=images)
        with MEDH5File(path) as f:
            assert f.seg is None

    def test_context_manager_h5_property(self, sample_file):
        path, _, _ = sample_file
        with MEDH5File(path) as f:
            assert isinstance(f.h5, type(f.h5))
            assert "images" in f.h5


class TestRepr:
    def test_sample_repr(self, sample_file):
        path, _, _ = sample_file
        sample = MEDH5File.read(path)
        r = repr(sample)
        assert "MEDH5Sample" in r
        assert "CT" in r
        assert "MRI_T1" in r
        assert "organ" in r or "lesion" in r

    def test_meta_repr(self, sample_file):
        path, _, _ = sample_file
        meta = MEDH5File.read_meta(path)
        r = repr(meta)
        assert "SampleMeta" in r
        assert "CT" in r
        assert "spacing" in r

    def test_meta_repr_with_bbox_and_extra(self, tmp_path):
        path = tmp_path / "bbox_extra.medh5"
        rng = np.random.default_rng(0)
        MEDH5File.write(
            path,
            images={"CT": rng.random((4, 8, 8), dtype=np.float32)},
            bboxes=np.array([[[0, 3], [0, 3], [0, 3]]]),
            extra={"study_id": "S42"},
        )
        meta = MEDH5File.read_meta(path)
        r = repr(meta)
        assert "has_bbox=True" in r
        assert "study_id" in r
