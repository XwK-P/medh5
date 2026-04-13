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


class TestIncludeBboxes:
    def _write_volume_with_bboxes(self, path):
        img = np.zeros((8, 32, 32), dtype=np.float32)
        bboxes = np.array(
            [
                # Box entirely inside patch [2:6, 8:24, 8:24]
                [[3, 5], [10, 15], [12, 20]],
                # Box entirely outside that patch (far corner)
                [[6, 7], [28, 30], [28, 30]],
                # Box that straddles the patch lower edge on axis 0
                [[1, 4], [10, 14], [10, 14]],
            ],
            dtype=np.float64,
        )
        MEDH5File.write(
            path,
            images={"CT": img},
            bboxes=bboxes,
            bbox_scores=np.array([0.9, 0.5, 0.7], dtype=np.float64),
            bbox_labels=["tumor", "cyst", "edge"],
        )
        return bboxes

    def test_patch_local_bboxes_filter_and_translate(self, tmp_path):
        path = tmp_path / "boxes.medh5"
        boxes = self._write_volume_with_bboxes(path)
        sampler = PatchSampler(patch_size=(4, 16, 16), include_bboxes=True, seed=0)

        # Pin starts to (2, 8, 8) so we can predict which boxes survive.
        sampler._uniform_start = lambda shape, patch: (2, 8, 8)  # type: ignore[method-assign]

        with MEDH5File(path) as f:
            sample = sampler.sample(f)

        # Boxes 0 and 2 intersect the patch; box 1 does not.
        assert sample["bboxes"].shape == (2, 3, 2)
        assert sample["bbox_labels"] == ["tumor", "edge"]
        np.testing.assert_allclose(sample["bbox_scores"], [0.9, 0.7])

        # Patch-local coordinates: subtract (2, 8, 8).
        expected0 = boxes[0] - np.array([[2, 2], [8, 8], [8, 8]])
        np.testing.assert_allclose(sample["bboxes"][0], expected0)
        expected2 = boxes[2] - np.array([[2, 2], [8, 8], [8, 8]])
        np.testing.assert_allclose(sample["bboxes"][1], expected2)

    def test_include_bboxes_default_off(self, tmp_path):
        path = tmp_path / "boxes.medh5"
        self._write_volume_with_bboxes(path)
        sampler = PatchSampler(patch_size=(4, 16, 16), seed=0)
        with MEDH5File(path) as f:
            sample = sampler.sample(f)
        assert "bboxes" not in sample

    def test_include_bboxes_no_bboxes_in_file(self, tmp_path):
        path = tmp_path / "plain.medh5"
        MEDH5File.write(
            path,
            images={"CT": np.zeros((8, 16, 16), dtype=np.float32)},
        )
        sampler = PatchSampler(patch_size=(4, 8, 8), include_bboxes=True, seed=0)
        with MEDH5File(path) as f:
            sample = sampler.sample(f)
        assert "bboxes" not in sample
