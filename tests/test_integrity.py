"""Tests for checksum/integrity features."""

import h5py
import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5FileError


@pytest.fixture
def sample_with_checksum(tmp_path):
    path = tmp_path / "checksummed.medh5"
    images = {"CT": np.random.default_rng(0).random((8, 8, 8), dtype=np.float32)}
    MEDH5File.write(path, images=images, checksum=True)
    return path


@pytest.fixture
def sample_without_checksum(tmp_path):
    path = tmp_path / "nochecksum.medh5"
    images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
    MEDH5File.write(path, images=images)
    return path


class TestIntegrity:
    def test_verify_valid(self, sample_with_checksum):
        assert MEDH5File.verify(sample_with_checksum) is True

    def test_verify_no_checksum_returns_true(self, sample_without_checksum):
        assert MEDH5File.verify(sample_without_checksum) is True

    def test_verify_corrupted(self, sample_with_checksum):
        with h5py.File(str(sample_with_checksum), "a") as f:
            data = f["images/CT"][...]
            data[0, 0, 0] = 999.0
            f["images/CT"][...] = data

        assert MEDH5File.verify(sample_with_checksum) is False

    def test_checksum_stored_as_attribute(self, sample_with_checksum):
        with h5py.File(str(sample_with_checksum), "r") as f:
            assert "checksum_sha256" in f.attrs
            digest = f.attrs["checksum_sha256"]
            assert len(digest) == 64

    def test_verify_detects_metadata_change(self, sample_with_checksum):
        MEDH5File.update_meta(sample_with_checksum, extra={"changed": True})
        assert MEDH5File.verify(sample_with_checksum) is True

        with h5py.File(str(sample_with_checksum), "a") as f:
            f.attrs["label"] = 99

        assert MEDH5File.verify(sample_with_checksum) is False

    def test_verify_after_review_update(self, sample_with_checksum):
        MEDH5File.set_review_status(
            sample_with_checksum,
            status="reviewed",
            annotator="qa",
        )
        assert MEDH5File.verify(sample_with_checksum) is True

    def test_verify_detects_seg_change(self, tmp_path):
        path = tmp_path / "with_seg.medh5"
        MEDH5File.write(
            path,
            images={"CT": np.zeros((4, 4, 4), dtype=np.float32)},
            seg={"tumor": np.zeros((4, 4, 4), dtype=bool)},
            checksum=True,
        )
        with h5py.File(str(path), "a") as f:
            data = f["seg/tumor"][...]
            data[0, 0, 0] = True
            f["seg/tumor"][...] = data
        assert MEDH5File.verify(path) is False


class TestUpdateVerifiesChecksum:
    def test_update_meta_refuses_on_corrupted_data(self, sample_with_checksum):
        with h5py.File(str(sample_with_checksum), "a") as f:
            data = f["images/CT"][...]
            data[0, 0, 0] = 999.0
            f["images/CT"][...] = data

        with pytest.raises(MEDH5FileError, match="checksum"):
            MEDH5File.update_meta(sample_with_checksum, label=1)

        assert MEDH5File.verify(sample_with_checksum) is False

    def test_update_with_force_bypasses_verify(self, sample_with_checksum):
        with h5py.File(str(sample_with_checksum), "a") as f:
            data = f["images/CT"][...]
            data[0, 0, 0] = 999.0
            f["images/CT"][...] = data

        MEDH5File.update(sample_with_checksum, meta={"label": 1}, force=True)
        assert MEDH5File.verify(sample_with_checksum) is True

    def test_update_no_checksum_is_unchanged(self, sample_without_checksum):
        MEDH5File.update_meta(sample_without_checksum, label=7)
        sample = MEDH5File.read(sample_without_checksum)
        assert sample.meta.label == 7
