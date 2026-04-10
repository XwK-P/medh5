"""Tests for medh5.stats.compute_stats."""

from __future__ import annotations

import numpy as np

from medh5 import MEDH5File
from medh5.dataset import Dataset
from medh5.stats import DatasetStats, compute_stats


def _write(path, ct, label=None, seg=None):
    MEDH5File.write(
        path,
        images={"CT": ct},
        seg=seg,
        label=label,
    )


class TestComputeStats:
    def test_known_mean_std(self, tmp_path):
        rng = np.random.default_rng(0)
        # Two volumes with controlled mean/std
        for i in range(4):
            ct = rng.normal(loc=10.0, scale=2.0, size=(8, 16, 16)).astype(np.float32)
            _write(tmp_path / f"s{i}.medh5", ct, label=i % 2)

        ds = Dataset.from_directory(tmp_path)
        stats = compute_stats(ds)
        assert stats.n_files == 4
        ct_stats = stats["CT"]
        assert abs(ct_stats.mean - 10.0) < 0.1
        assert abs(ct_stats.std - 2.0) < 0.1
        assert ct_stats.n_voxels == 4 * 8 * 16 * 16
        assert stats.label_counts == {"0": 2, "1": 2}
        assert "(8, 16, 16)" in stats.shape_histogram

    def test_foreground_only(self, tmp_path):
        ct = np.ones((4, 8, 8), dtype=np.float32) * 5.0
        # Make a mask that selects only voxels with value 5
        mask = np.zeros((4, 8, 8), dtype=bool)
        mask[1:3, 2:6, 2:6] = True
        ct[~mask] = 100.0  # background = 100
        _write(tmp_path / "s.medh5", ct, seg={"liver": mask})

        ds = Dataset.from_directory(tmp_path)
        stats = compute_stats(ds, foreground_mask="liver")
        assert abs(stats["CT"].mean - 5.0) < 1e-6  # background ignored
        assert stats.seg_coverage["liver"] > 0.0

    def test_percentiles(self, tmp_path):
        # Volume of values 0..999 → p01 ≈ 9, p99 ≈ 989
        ct = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
        _write(tmp_path / "s.medh5", ct)
        stats = compute_stats([tmp_path / "s.medh5"])
        assert 5 <= stats["CT"].p01 <= 15
        assert 985 <= stats["CT"].p99 <= 995

    def test_save_load_roundtrip(self, tmp_path):
        ct = np.zeros((2, 4, 4), dtype=np.float32)
        _write(tmp_path / "s.medh5", ct, label="benign")
        stats = compute_stats([tmp_path / "s.medh5"])
        out = tmp_path / "stats.json"
        stats.save(out)
        loaded = DatasetStats.load(out)
        assert loaded.n_files == 1
        assert loaded["CT"].mean == stats["CT"].mean

    def test_modality_filter(self, tmp_path):
        MEDH5File.write(
            tmp_path / "s.medh5",
            images={
                "CT": np.ones((2, 4, 4), dtype=np.float32) * 1.0,
                "PET": np.ones((2, 4, 4), dtype=np.float32) * 5.0,
            },
        )
        stats = compute_stats([tmp_path / "s.medh5"], modalities=["PET"])
        assert "PET" in stats.modalities
        assert "CT" not in stats.modalities
        assert stats["PET"].mean == 5.0
