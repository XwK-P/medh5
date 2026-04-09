"""Tests for the PyTorch dataset wrapper."""

import numpy as np
import pytest

from medh5 import MEDH5File

torch = pytest.importorskip("torch")

from medh5.torch import MEDH5TorchDataset  # noqa: E402


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
