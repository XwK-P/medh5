"""Tests for medh5.dataset.index (Dataset + manifest + filter)."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5File
from medh5.dataset import Dataset
from medh5.exceptions import MEDH5Error


def _make_file(path, label=None, label_name=None, extra=None, seg=False):
    img = np.zeros((2, 4, 4), dtype=np.float32)
    seg_d = {"tumor": np.zeros((2, 4, 4), dtype=bool)} if seg else None
    MEDH5File.write(
        path,
        images={"CT": img},
        seg=seg_d,
        label=label,
        label_name=label_name,
        extra=extra,
    )


class TestDatasetFromDirectory:
    def test_scan_and_len(self, tmp_path):
        for i in range(3):
            _make_file(tmp_path / f"s{i}.medh5", label=i % 2)
        ds = Dataset.from_directory(tmp_path)
        assert len(ds) == 3
        labels = sorted(r.label for r in ds)
        assert labels == [0, 0, 1]

    def test_recursive_toggle(self, tmp_path):
        (tmp_path / "sub").mkdir()
        _make_file(tmp_path / "a.medh5")
        _make_file(tmp_path / "sub" / "b.medh5")
        flat = Dataset.from_directory(tmp_path, recursive=False)
        assert len(flat) == 1
        deep = Dataset.from_directory(tmp_path, recursive=True)
        assert len(deep) == 2

    def test_skip_invalid(self, tmp_path):
        _make_file(tmp_path / "ok.medh5")
        # Create a corrupted file with the right extension
        (tmp_path / "bad.medh5").write_bytes(b"not an h5 file")
        with pytest.raises(MEDH5Error):
            Dataset.from_directory(tmp_path, skip_invalid=False)
        ds = Dataset.from_directory(tmp_path, skip_invalid=True)
        assert len(ds) == 1

    def test_filter_and_paths(self, tmp_path):
        _make_file(tmp_path / "a.medh5", label=0)
        _make_file(tmp_path / "b.medh5", label=1, seg=True)
        ds = Dataset.from_directory(tmp_path)
        with_seg = ds.filter(lambda r: r.has_seg)
        assert len(with_seg) == 1
        assert with_seg.paths[0].endswith("b.medh5")

    def test_manifest_roundtrip(self, tmp_path):
        _make_file(tmp_path / "a.medh5", label=0, extra={"site": "A"})
        _make_file(tmp_path / "b.medh5", label=1, extra={"site": "B"})
        MEDH5File.set_review_status(tmp_path / "a.medh5", status="reviewed")
        ds = Dataset.from_directory(tmp_path)
        manifest = tmp_path / "manifest.json"
        ds.save(manifest)
        reloaded = Dataset.load(manifest)
        assert len(reloaded) == len(ds)
        assert reloaded.paths == ds.paths
        assert reloaded.records[0].extra is not None
        assert reloaded.records[0].extra["site"] == "A"
        assert reloaded.records[0].shape == [2, 4, 4]
        assert reloaded.records[0].review_status == "reviewed"

    def test_staleness_detection(self, tmp_path):
        path = tmp_path / "a.medh5"
        _make_file(path, label=0)
        ds = Dataset.from_directory(tmp_path)
        assert ds.stale() == []
        # Touch file: rewrite changes size/mtime
        _make_file(path, label=1)
        assert len(ds.stale()) == 1

    def test_iter_and_getitem(self, tmp_path):
        _make_file(tmp_path / "a.medh5", label=0)
        _make_file(tmp_path / "b.medh5", label=1)
        ds = Dataset.from_directory(tmp_path)
        assert ds[0].label == 0
        for r in ds:
            assert r.path.endswith(".medh5")

    def test_from_paths(self, tmp_path):
        p1 = tmp_path / "a.medh5"
        p2 = tmp_path / "b.medh5"
        _make_file(p1, label=0)
        _make_file(p2, label=1)
        ds = Dataset.from_paths([p1, p2])
        assert len(ds) == 2

    def test_record_spatial_fields(self, tmp_path):
        path = tmp_path / "a.medh5"
        MEDH5File.write(
            path,
            images={"CT": np.zeros((2, 4, 4), dtype=np.float32)},
            spacing=[1.0, 0.5, 0.5],
            coord_system="RAS",
            patch_size=[2, 4, 4],
        )
        ds = Dataset.from_directory(tmp_path)
        record = ds[0]
        assert record.spacing == [1.0, 0.5, 0.5]
        assert record.coord_system == "RAS"
        assert record.patch_size == [2, 4, 4]
