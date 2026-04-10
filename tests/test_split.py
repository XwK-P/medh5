"""Tests for medh5.dataset.split (ratio + k-fold + stratify + group)."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError
from medh5.dataset import Dataset, make_splits


def _make_files(tmp_path, n=20):
    """Create n files with alternating labels and 2-sample patient groups."""
    for i in range(n):
        MEDH5File.write(
            tmp_path / f"s{i:02d}.medh5",
            images={"CT": np.zeros((2, 2, 2), dtype=np.float32)},
            label=i % 2,
            extra={"patient_id": f"P{i // 2:02d}"},
        )
    return Dataset.from_directory(tmp_path)


class TestRatioSplit:
    def test_basic(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        splits = make_splits(
            ds, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
        )
        total = sum(len(s) for s in splits.values())
        assert total == 20
        assert len(splits["train"]) == 14
        assert len(splits["val"]) + len(splits["test"]) == 6

    def test_stratify_by_label(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        splits = make_splits(
            ds, ratios={"train": 0.5, "val": 0.5}, stratify_by="label", seed=0
        )
        for s in splits.values():
            labels = [r.label for r in s]
            assert labels.count(0) == labels.count(1)

    def test_group_by_patient(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        splits = make_splits(
            ds,
            ratios={"train": 0.7, "val": 0.3},
            group_by="extra.patient_id",
            seed=7,
        )
        seen: dict[str, str] = {}
        for name, s in splits.items():
            for r in s:
                pid = r.extra["patient_id"]
                assert pid not in seen or seen[pid] == name, (
                    f"Patient {pid} leaked across {seen[pid]} and {name}"
                )
                seen[pid] = name

    def test_seeded_reproducibility(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        a = make_splits(ds, ratios={"train": 0.5, "val": 0.5}, seed=123)
        b = make_splits(ds, ratios={"train": 0.5, "val": 0.5}, seed=123)
        assert a["train"].paths == b["train"].paths

    def test_bad_ratios_raises(self, tmp_path):
        ds = _make_files(tmp_path, n=4)
        with pytest.raises(MEDH5ValidationError, match="sum to 1.0"):
            make_splits(ds, ratios={"train": 0.6, "val": 0.6})

    def test_neither_ratios_nor_kfolds(self, tmp_path):
        ds = _make_files(tmp_path, n=4)
        with pytest.raises(MEDH5ValidationError):
            make_splits(ds)
        with pytest.raises(MEDH5ValidationError):
            make_splits(ds, ratios={"a": 1.0}, k_folds=5)


class TestKFoldSplit:
    def test_basic(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        folds = make_splits(ds, k_folds=5, seed=0)
        assert isinstance(folds, list) and len(folds) == 5
        for fm in folds:
            assert len(fm["train"]) + len(fm["val"]) == 20
        # Every record appears in exactly one val fold
        all_val = []
        for fm in folds:
            all_val.extend(fm["val"].paths)
        assert sorted(all_val) == sorted(ds.paths)

    def test_kfold_too_small(self, tmp_path):
        ds = _make_files(tmp_path, n=4)
        with pytest.raises(MEDH5ValidationError):
            make_splits(ds, k_folds=1)

    def test_kfold_with_stratify(self, tmp_path):
        ds = _make_files(tmp_path, n=20)
        folds = make_splits(ds, k_folds=4, stratify_by="label", seed=42)
        for fm in folds:
            val_labels = [r.label for r in fm["val"]]
            # roughly balanced
            assert abs(val_labels.count(0) - val_labels.count(1)) <= 1
