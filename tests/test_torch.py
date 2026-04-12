"""Tests for the PyTorch dataset wrapper."""

import numpy as np
import pytest

from medh5 import MEDH5File

torch = pytest.importorskip("torch")

from medh5.sampling import PatchSampler  # noqa: E402
from medh5.torch import (  # noqa: E402
    _HANDLE_CACHE,
    MEDH5PatchDataset,
    MEDH5TorchDataset,
    _HandleCache,
)


@pytest.fixture
def sample_files(tmp_path):
    paths = []
    rng = np.random.default_rng(7)
    for i in range(3):
        path = tmp_path / f"sample_{i}.medh5"
        images = {
            "CT": rng.random((8, 16, 16), dtype=np.float32),
            "PET": rng.random((8, 16, 16), dtype=np.float32),
        }
        seg = {"tumor": rng.random((8, 16, 16)) > 0.8}
        MEDH5File.write(
            path,
            images=images,
            seg=seg,
            label=i,
            spacing=[1.0, 1.0, 1.0],
        )
        paths.append(path)
    return paths


class TestMEDH5TorchDataset:
    def test_len(self, sample_files):
        ds = MEDH5TorchDataset(sample_files)
        assert len(ds) == 3

    def test_getitem_images(self, sample_files):
        ds = MEDH5TorchDataset(sample_files)
        out = ds[0]
        assert "images" in out
        assert isinstance(out["images"]["CT"], torch.Tensor)
        assert isinstance(out["images"]["PET"], torch.Tensor)
        assert out["images"]["CT"].shape == (8, 16, 16)

    def test_getitem_seg(self, sample_files):
        ds = MEDH5TorchDataset(sample_files)
        out = ds[0]
        assert "seg" in out
        assert isinstance(out["seg"]["tumor"], torch.Tensor)

    def test_getitem_label(self, sample_files):
        ds = MEDH5TorchDataset(sample_files)
        for i in range(3):
            assert ds[i]["label"] == i

    def test_transform(self, sample_files):
        def add_flag(sample):
            sample["transformed"] = True
            return sample

        ds = MEDH5TorchDataset(sample_files, transform=add_flag)
        assert ds[0]["transformed"] is True

    def test_numpy_transforms_run_before_tensor_conversion(self, sample_files):
        # Regression: the numpy transforms in medh5.transforms call
        # ``arr.astype(np.float32)``, so the transform must see numpy
        # arrays, not tensors. MEDH5TorchDataset used to convert to
        # tensors first and broke this contract.
        from medh5.transforms import Clip, Compose, ZScore

        pipe = Compose([Clip(min=0.0, max=1.0), ZScore()])
        ds = MEDH5TorchDataset(sample_files, transform=pipe)
        out = ds[0]
        ct = out["images"]["CT"]
        assert isinstance(ct, torch.Tensor)
        # ZScore should have centered the per-volume mean on ~0.
        assert abs(ct.float().mean().item()) < 1e-4

    def test_getitem_bboxes(self, tmp_path):
        rng = np.random.default_rng(3)
        path = tmp_path / "bbox_sample.medh5"
        MEDH5File.write(
            path,
            images={"CT": rng.random((4, 8, 8), dtype=np.float32)},
            bboxes=np.array([[[0, 3], [0, 3], [0, 3]], [[1, 4], [2, 5], [3, 6]]]),
            bbox_scores=np.array([0.9, 0.75]),
            bbox_labels=["tumor", "lesion"],
        )
        ds = MEDH5TorchDataset([path])
        out = ds[0]
        assert isinstance(out["bboxes"], torch.Tensor)
        assert out["bboxes"].shape == (2, 3, 2)
        assert isinstance(out["bbox_scores"], torch.Tensor)
        assert out["bbox_labels"] == ["tumor", "lesion"]


class TestMEDH5PatchDataset:
    def test_basic(self, sample_files):
        sampler = PatchSampler(patch_size=(4, 8, 8), seed=0)
        ds = MEDH5PatchDataset(sample_files, sampler=sampler, samples_per_volume=2)
        assert len(ds) == 6
        out = ds[0]
        assert out["images"]["CT"].shape == (4, 8, 8)
        assert isinstance(out["images"]["CT"], torch.Tensor)
        assert isinstance(out["seg"]["tumor"], torch.Tensor)

    def test_foreground_strategy(self, tmp_path):
        ct = np.zeros((4, 16, 16), dtype=np.float32)
        mask = np.zeros((4, 16, 16), dtype=bool)
        mask[2, 8, 8] = True
        path = tmp_path / "f.medh5"
        MEDH5File.write(path, images={"CT": ct}, seg={"tumor": mask})
        sampler = PatchSampler(
            patch_size=(2, 6, 6),
            strategy="foreground",
            foreground_seg="tumor",
            seed=0,
        )
        ds = MEDH5PatchDataset([path], sampler=sampler, samples_per_volume=4)
        for i in range(len(ds)):
            out = ds[i]
            assert out["seg"]["tumor"].any()

    def test_transform_applied(self, sample_files):
        sampler = PatchSampler(patch_size=(4, 8, 8), seed=0)

        def add_flag(s):
            s["transformed"] = True
            return s

        ds = MEDH5PatchDataset(sample_files, sampler=sampler, transform=add_flag)
        assert ds[0]["transformed"] is True


class TestHandleCache:
    def test_handle_reuse(self, sample_files):
        cache = _HandleCache(maxsize=4)
        h1 = cache.get(sample_files[0])
        h2 = cache.get(sample_files[0])
        assert h1 is h2
        assert cache.opens == 1
        cache.close_all()

    def test_lru_eviction(self, sample_files):
        cache = _HandleCache(maxsize=2)
        cache.get(sample_files[0])
        cache.get(sample_files[1])
        cache.get(sample_files[2])  # evicts the first
        assert cache.opens == 3
        # Re-fetching the evicted one re-opens it
        cache.get(sample_files[0])
        assert cache.opens == 4
        cache.close_all()

    def test_patch_dataset_uses_module_cache(self, sample_files, monkeypatch):
        # Reset the module-level cache to a fresh known state
        _HANDLE_CACHE.close_all()
        _HANDLE_CACHE.opens = 0
        sampler = PatchSampler(patch_size=(4, 8, 8), seed=0)
        ds = MEDH5PatchDataset(sample_files[:1], sampler=sampler, samples_per_volume=10)
        for i in range(len(ds)):
            ds[i]
        assert _HANDLE_CACHE.opens == 1  # one open across all 10 reads
        _HANDLE_CACHE.close_all()
