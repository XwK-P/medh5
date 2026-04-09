"""Unit tests for the chunk optimizer."""

import pytest

from medh5.chunks import optimize_chunks


class TestOptimizeChunks:
    def test_3d_basic(self):
        chunks = optimize_chunks(
            image_shape=(128, 256, 256),
            patch_size=192,
            bytes_per_element=4,
        )
        assert len(chunks) == 3
        assert all(c > 0 for c in chunks)
        assert all(c <= s for c, s in zip(chunks, (128, 256, 256)))

    def test_2d_basic(self):
        chunks = optimize_chunks(
            image_shape=(512, 512),
            patch_size=128,
            bytes_per_element=4,
        )
        assert len(chunks) == 2
        assert all(c > 0 for c in chunks)

    def test_chunk_le_image(self):
        shape = (64, 64, 64)
        chunks = optimize_chunks(shape, patch_size=192, bytes_per_element=4)
        for c, s in zip(chunks, shape):
            assert c <= s

    def test_small_image_equals_image(self):
        shape = (4, 4, 4)
        chunks = optimize_chunks(shape, patch_size=192, bytes_per_element=1)
        assert chunks == shape

    def test_anisotropic_patch(self):
        chunks = optimize_chunks(
            image_shape=(128, 256, 256),
            patch_size=(32, 128, 128),
            bytes_per_element=4,
        )
        assert len(chunks) == 3

    def test_with_channel_axis(self):
        chunks = optimize_chunks(
            image_shape=(3, 128, 256, 256),
            patch_size=(128, 192, 192),
            bytes_per_element=4,
            spatial_axis_mask=[False, True, True, True],
        )
        assert len(chunks) == 4
        assert chunks[0] == 3  # non-spatial axis = full extent

    def test_rejects_4d_spatial(self):
        with pytest.raises(NotImplementedError):
            optimize_chunks(
                image_shape=(10, 10, 10, 10),
                patch_size=(5, 5, 5, 5),
                bytes_per_element=4,
            )

    def test_rejects_mismatched_mask(self):
        with pytest.raises(ValueError, match="spatial_axis_mask"):
            optimize_chunks(
                image_shape=(128, 256, 256),
                patch_size=64,
                spatial_axis_mask=[True, True],
            )

    def test_int_patch_broadcast(self):
        c1 = optimize_chunks((128, 128, 128), patch_size=64)
        c2 = optimize_chunks((128, 128, 128), patch_size=(64, 64, 64))
        assert c1 == c2
