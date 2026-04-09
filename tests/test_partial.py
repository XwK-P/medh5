"""Test lazy / partial reads via MEDH5File.open()."""

import numpy as np
import pytest

from medh5 import MEDH5File


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "partial.medh5"
    rng = np.random.default_rng(99)
    image = rng.random((32, 64, 64), dtype=np.float32)
    seg = rng.integers(0, 3, size=(32, 64, 64), dtype=np.uint8)
    MEDH5File.write(
        path,
        image=image,
        seg=seg,
        label=1,
        spacing=[1.0, 1.0, 2.0],
        patch_size=32,
    )
    return path, image, seg


class TestPartialRead:
    def test_patch_read(self, sample_file):
        path, image, _ = sample_file
        with MEDH5File.open(path) as f:
            patch = f["image"][5:15, 10:30, 20:50]
        np.testing.assert_array_equal(patch, image[5:15, 10:30, 20:50])

    def test_seg_patch_read(self, sample_file):
        path, _, seg = sample_file
        with MEDH5File.open(path) as f:
            patch = f["seg"][0:8, 0:16, 0:16]
        np.testing.assert_array_equal(patch, seg[0:8, 0:16, 0:16])

    def test_meta_from_open(self, sample_file):
        path, _, _ = sample_file
        from medh5.meta import read_meta

        with MEDH5File.open(path) as f:
            meta = read_meta(f)
        assert meta.label == 1
        assert meta.spatial.spacing == [1.0, 1.0, 2.0]

    def test_datasets_listed(self, sample_file):
        path, _, _ = sample_file
        with MEDH5File.open(path) as f:
            keys = set(f.keys())
        assert "image" in keys
        assert "seg" in keys
