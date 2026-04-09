"""Tests for checksum/integrity features."""

import h5py
import numpy as np
import pytest

from medh5 import MEDH5File


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
