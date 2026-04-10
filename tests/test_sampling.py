"""Tests for medh5.sampling.PatchSampler."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError
from medh5.sampling import PatchSampler


def _make_volume(path, shape=(8, 32, 32), seg_shape=None):
    rng = np.random.default_rng(0)
    img = rng.standard_normal(shape).astype(np.float32)
    seg = None
    if seg_shape is not None:
        mask = np.zeros(shape, dtype=bool)
        # Single foreground voxel at the center
        mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] = True
        seg = {"tumor": mask}
    MEDH5File.write(path, images={"CT": img}, seg=seg)
    return img


class TestUniformSampler:
    def test_shape(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_volume(p, shape=(8, 32, 32))
        sampler = PatchSampler(patch_size=(4, 16, 16), seed=0)
        with MEDH5File(p) as f:
            sample = sampler.sample(f)
        assert sample["images"]["CT"].shape == (4, 16, 16)
        assert "patch_origin" in sample

    def test_scalar_patch_size(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_volume(p, shape=(8, 32, 32))
        sampler = PatchSampler(patch_size=8, seed=0)
        with MEDH5File(p) as f:
            sample = sampler.sample(f)
        assert sample["images"]["CT"].shape == (8, 8, 8)

    def test_pad_when_patch_larger_than_volume(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_volume(p, shape=(4, 4, 4))
        sampler = PatchSampler(patch_size=(8, 8, 8), pad_value=-1.0, seed=0)
        with MEDH5File(p) as f:
            sample = sampler.sample(f)
        assert sample["images"]["CT"].shape == (8, 8, 8)
        # Padded region must contain the pad value
        assert (sample["images"]["CT"] == -1.0).any()

    def test_invalid_strategy(self):
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(patch_size=4, strategy="bogus")

    def test_foreground_requires_mask_name(self):
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(patch_size=4, strategy="foreground")


class TestForegroundSampler:
    def test_patch_contains_foreground(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_volume(p, shape=(8, 32, 32), seg_shape=(8, 32, 32))
        sampler = PatchSampler(
            patch_size=(4, 16, 16),
            strategy="foreground",
            foreground_seg="tumor",
            seed=0,
        )
        # Many draws — every one should land on the (single) foreground voxel.
        with MEDH5File(p) as f:
            for _ in range(10):
                sample = sampler.sample(f)
                assert sample["seg"]["tumor"].any()

    def test_balanced_falls_back_when_no_mask(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_volume(p, shape=(8, 16, 16))  # no seg
        sampler = PatchSampler(
            patch_size=(4, 8, 8),
            strategy="balanced",
            foreground_seg="tumor",
            foreground_prob=1.0,
            seed=0,
        )
        with MEDH5File(p) as f:
            sample = sampler.sample(f)
        # No crash; just behaves like uniform
        assert sample["images"]["CT"].shape == (4, 8, 8)
