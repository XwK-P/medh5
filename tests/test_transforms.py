"""Tests for medh5.transforms."""

from __future__ import annotations

import numpy as np

from medh5.meta import SampleMeta, SpatialMeta
from medh5.transforms import Clip, Compose, Normalize, RandomFlip, ZScore


def _sample(ct=None, pet=None, seg=None):
    s = {"images": {}}
    if ct is not None:
        s["images"]["CT"] = ct
    if pet is not None:
        s["images"]["PET"] = pet
    if seg is not None:
        s["seg"] = seg
    return s


class TestClip:
    def test_basic(self):
        ct = np.array([-50.0, 0.0, 50.0, 100.0], dtype=np.float32)
        out = Clip(min=0.0, max=10.0)(_sample(ct=ct))
        np.testing.assert_array_equal(out["images"]["CT"], [0.0, 0.0, 10.0, 10.0])

    def test_modality_filter(self):
        ct = np.array([-5.0, 5.0], dtype=np.float32)
        pet = np.array([-5.0, 5.0], dtype=np.float32)
        out = Clip(min=0.0, max=2.0, modalities=["CT"])(_sample(ct=ct, pet=pet))
        np.testing.assert_array_equal(out["images"]["CT"], [0.0, 2.0])
        np.testing.assert_array_equal(out["images"]["PET"], [-5.0, 5.0])  # untouched


class TestNormalize:
    def test_dict_per_modality(self):
        ct = np.full((4,), 10.0, dtype=np.float32)
        pet = np.full((4,), 5.0, dtype=np.float32)
        out = Normalize(mean={"CT": 10.0, "PET": 5.0}, std={"CT": 2.0, "PET": 1.0})(
            _sample(ct=ct, pet=pet)
        )
        np.testing.assert_allclose(out["images"]["CT"], 0.0)
        np.testing.assert_allclose(out["images"]["PET"], 0.0)

    def test_scalar(self):
        ct = np.full((4,), 10.0, dtype=np.float32)
        out = Normalize(mean=4.0, std=2.0)(_sample(ct=ct))
        np.testing.assert_allclose(out["images"]["CT"], 3.0)

    def test_zero_std_skipped(self):
        ct = np.full((4,), 10.0, dtype=np.float32)
        out = Normalize(mean=0.0, std=0.0)(_sample(ct=ct))
        # Untouched (zero-std skipped)
        np.testing.assert_allclose(out["images"]["CT"], 10.0)


class TestZScore:
    def test_basic(self):
        ct = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = ZScore()(_sample(ct=ct))
        z = out["images"]["CT"]
        assert abs(float(z.mean())) < 1e-5
        assert abs(float(z.std()) - 1.0) < 1e-3


class TestRandomFlip:
    def test_always_flip(self):
        ct = np.arange(8, dtype=np.float32).reshape(2, 4)
        original_mask = (ct > 4).astype(bool)
        seg = {"tumor": original_mask.copy()}
        out = RandomFlip(axes=(0, 1), p=1.0, seed=0)(_sample(ct=ct, seg=seg))
        np.testing.assert_array_equal(out["images"]["CT"], np.flip(ct, axis=(0, 1)))
        np.testing.assert_array_equal(
            out["seg"]["tumor"], np.flip(original_mask, axis=(0, 1))
        )

    def test_never_flip(self):
        ct = np.arange(8, dtype=np.float32).reshape(2, 4)
        out = RandomFlip(axes=(0,), p=0.0, seed=0)(_sample(ct=ct))
        np.testing.assert_array_equal(out["images"]["CT"], ct)

    def test_direction_column_negated_on_flip(self):
        ct = np.zeros((4, 4, 4), dtype=np.float32)
        meta = SampleMeta(
            spatial=SpatialMeta(direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            shape=[4, 4, 4],
        )
        sample = _sample(ct=ct)
        sample["meta"] = meta
        out = RandomFlip(axes=(0,), p=1.0, seed=0)(sample)
        assert out["meta"].spatial.direction == [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        # Original meta must not have been mutated.
        assert meta.spatial.direction == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_origin_shifted_on_flip(self):
        ct = np.zeros((4, 5, 6), dtype=np.float32)
        meta = SampleMeta(
            spatial=SpatialMeta(
                spacing=[2.0, 3.0, 4.0],
                origin=[10.0, 20.0, 30.0],
                direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
            shape=[4, 5, 6],
        )
        sample = _sample(ct=ct)
        sample["meta"] = meta
        out = RandomFlip(axes=(0, 2), p=1.0, seed=0)(sample)
        assert out["meta"].spatial.origin == [16.0, 20.0, 50.0]
        assert out["meta"].spatial.direction == [
            [-1.0, 0.0, -0.0],
            [0.0, 1.0, -0.0],
            [0.0, 0.0, -1.0],
        ]
        assert meta.spatial.origin == [10.0, 20.0, 30.0]

    def test_flip_without_meta_ok(self):
        ct = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
        sample = _sample(ct=ct)
        out = RandomFlip(axes=(0,), p=1.0, seed=0)(sample)
        np.testing.assert_array_equal(out["images"]["CT"], np.flip(ct, axis=0))
        assert "meta" not in out

    def test_bbox_mirrored(self):
        ct = np.zeros((32, 64, 64), dtype=np.float32)
        bboxes = np.array(
            [
                [[2, 10], [5, 20], [5, 20]],
                [[12, 28], [40, 55], [30, 50]],
            ],
            dtype=np.float64,
        )
        sample = _sample(ct=ct)
        sample["bboxes"] = bboxes.copy()
        out = RandomFlip(axes=(0,), p=1.0, seed=0)(sample)
        expected = bboxes.copy()
        expected[0, 0, 0] = 32 - 1 - 10  # 21
        expected[0, 0, 1] = 32 - 1 - 2  # 29
        expected[1, 0, 0] = 32 - 1 - 28  # 3
        expected[1, 0, 1] = 32 - 1 - 12  # 19
        np.testing.assert_array_equal(out["bboxes"], expected)
        # Non-flipped axes untouched.
        np.testing.assert_array_equal(out["bboxes"][:, 1, :], bboxes[:, 1, :])
        np.testing.assert_array_equal(out["bboxes"][:, 2, :], bboxes[:, 2, :])

    def test_bbox_flip_multiple_axes(self):
        ct = np.zeros((10, 20), dtype=np.float32)
        bboxes = np.array([[[1, 3], [4, 9]]], dtype=np.float64)
        sample = _sample(ct=ct)
        sample["bboxes"] = bboxes.copy()
        out = RandomFlip(axes=(0, 1), p=1.0, seed=0)(sample)
        # axis 0: [1,3] -> [10-1-3, 10-1-1] = [6, 8]
        # axis 1: [4,9] -> [20-1-9, 20-1-4] = [10, 15]
        expected = np.array([[[6, 8], [10, 15]]], dtype=np.float64)
        np.testing.assert_array_equal(out["bboxes"], expected)


class TestCompose:
    def test_pipeline(self):
        ct = np.array([-50.0, 0.0, 50.0], dtype=np.float32)
        t = Compose(
            [
                Clip(min=0.0, max=100.0),
                Normalize(mean=0.0, std=10.0),
            ]
        )
        out = t(_sample(ct=ct))
        np.testing.assert_allclose(out["images"]["CT"], [0.0, 0.0, 5.0])
