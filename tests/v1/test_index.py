"""The sampling index and chunking policy (spec §14.1, §14.3)."""

from __future__ import annotations

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.storage.chunking import (
    MAX_CHUNK_BYTES,
    chunk_report,
    optimize_chunks,
    spatial_chunk_for,
)
from medh5.storage.codecs import (
    COMPRESS_MIN_BYTES,
    PROFILES,
    dataset_kwargs,
    is_bulk,
    resolve_profile,
)
from medh5.storage.index import build_index, read_indices


class TestChunking:
    def test_S14_1_chunk_starts_from_the_patch_hint(self):
        chunk = spatial_chunk_for((256, 256, 256), (96, 96, 96), itemsize=2)
        assert all(c >= 96 for c in chunk)

    def test_S14_1_chunk_never_exceeds_the_extent(self):
        assert spatial_chunk_for((10, 12, 14), (96, 96, 96)) == (10, 12, 14)

    def test_S14_1_chunk_stays_within_the_cache_budget(self):
        chunk = spatial_chunk_for((512, 512, 512), (64, 64, 64), itemsize=4)
        assert int(np.prod(chunk)) * 4 <= MAX_CHUNK_BYTES

    def test_S14_1_non_spatial_axes_get_extent_one(self):
        chunk = optimize_chunks(
            (4, 32, 64, 64), ("time", "spatial", "spatial", "spatial"), (16, 16, 16)
        )
        assert chunk[0] == 1

    def test_S14_1_stacked_encodings_read_one_plane(self):
        """§14.1: layers/bitmask/probmap MUST chunk as (1, *spatial_chunk)."""
        chunk = optimize_chunks(
            (5, 32, 64, 64), ("spatial", "spatial", "spatial"), (16, 16, 16), leading=1
        )
        assert chunk[0] == 1
        assert len(chunk) == 4

    def test_S14_1_l3_detection_spawns_nothing_where_sysfs_answers(self, monkeypatch):
        """The probe used to fork `/bin/sh` on every platform, before the read.

        On Linux that is a wasted fork+exec for a sysctl key that does not
        exist, in a package whose handle cache exists because HDF5 state must
        not cross a fork --- and `os.popen` signalled failure by returning empty
        output rather than by raising the `OSError` the handler caught.
        """
        import os
        import subprocess

        from medh5.storage import chunking

        def refuse(*args, **kwargs):
            raise AssertionError("L3 detection must not spawn a process here")

        monkeypatch.setattr(chunking.sys, "platform", "linux")
        monkeypatch.setattr(os, "popen", refuse)
        monkeypatch.setattr(subprocess, "run", refuse)
        chunking.detect_l3_bytes.cache_clear()
        try:
            assert chunking.detect_l3_bytes() > 0
        finally:
            chunking.detect_l3_bytes.cache_clear()

    def test_axis_kinds_must_describe_the_shape(self):
        with pytest.raises(MEDH5ValidationError):
            optimize_chunks((4, 4), ("spatial",), (2,))

    def test_patch_length_must_match(self):
        with pytest.raises(MEDH5ValidationError):
            spatial_chunk_for((8, 8, 8), (4, 4))

    def test_degenerate_shape_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            spatial_chunk_for((0, 8, 8))

    def test_chunk_report(self):
        report = chunk_report((64, 64, 64), (32, 32, 32), 2)
        assert report["n_chunks"] == 8
        assert report["chunk_bytes"] == 32 * 32 * 32 * 2


class TestCodecs:
    def test_every_profile_builds_kwargs(self):
        for name in PROFILES:
            kwargs = dataset_kwargs(
                (256, 256, 8), np.dtype(np.int16), profile=name, role="image"
            )
            assert "chunks" in kwargs

    def test_small_datasets_stay_contiguous(self):
        assert dataset_kwargs((4,), np.dtype(np.uint16)) == {}
        big = (COMPRESS_MIN_BYTES // 2 + 16,)
        assert dataset_kwargs(big, np.dtype(np.uint16)) != {}

    def test_empty_datasets_stay_contiguous(self):
        assert dataset_kwargs((0, 3), np.dtype(np.float32)) == {}

    def test_unknown_profile_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            resolve_profile("turbo")

    def test_profile_objects_pass_through(self):
        assert resolve_profile(PROFILES["archive"]) is PROFILES["archive"]
        assert resolve_profile(None).name == "balanced"

    def test_S14_2_portable_needs_no_plugin(self, tmp_path):
        """A `portable` file must open with stock h5py filters only."""
        import h5py

        import medh5

        shape = (64, 64, 64)
        path = tmp_path / "p.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
        with h5py.File(path) as handle:
            assert handle["images/CT"].compression == "gzip"
            assert np.asarray(handle["images/CT"][0, 0, :4]).size == 4

    def test_is_bulk(self, sample_path):
        import h5py

        with h5py.File(sample_path) as handle:
            assert not is_bulk(handle["annotations/organs_tp0/layer_class_ids"])


class TestSamplingIndex:
    def test_S14_3_index_answers_foreground_sampling(self, longitudinal_path, masks):
        with medh5.open(longitudinal_path) as sample:
            index = sample.index["organs_tp0"]
            assert index.voxel_counts[1] == int(masks[1].sum())
            centres = index.sample_foreground(1, n=8, rng=np.random.default_rng(0))
            assert centres.shape == (8, 3)
            for centre in centres:
                assert masks[1][tuple(centre)]

    def test_S14_3_coordinates_are_capped(self, tmp_path, label_set):
        big = {1: np.ones((16, 24, 24), dtype=bool)}
        from tests.v1.conftest import write_sample

        path = write_sample(
            tmp_path / "cap.medh5", label_set=label_set, masks=big, index=True
        )
        with medh5.open(path) as sample:
            index = sample.index["organs_tp0"]
            assert index.coords(1).shape[0] == 64
            assert index.voxel_counts[1] == 16 * 24 * 24

    def test_S14_3_bboxes_are_tight(self, longitudinal_path, masks):
        with medh5.open(longitudinal_path) as sample:
            box = sample.index["organs_tp0"].bbox(1)
            assert box is not None
            from medh5.geometry.affine import box_to_slices

            assert np.array_equal(
                masks[1][box_to_slices(box)],
                masks[1][masks[1].any(axis=(1, 2))][:, masks[1].any(axis=(0, 2))][
                    :, :, masks[1].any(axis=(0, 1))
                ],
            )

    def test_empty_class_has_no_bbox_and_no_samples(self, tmp_path, label_set):
        from tests.v1.conftest import write_sample

        masks = {1: np.zeros((16, 24, 24), dtype=bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(
            tmp_path / "e.medh5", label_set=label_set, masks=masks, index=True
        )
        with medh5.open(path) as sample:
            index = sample.index["organs_tp0"]
            assert index.voxel_counts[1] == 1
            with pytest.raises(KeyError):
                index.coords(99)

    def test_class_weights(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            index = sample.index["organs_tp0"]
            weights = index.class_weights()
            assert abs(sum(weights.values()) - 1.0) < 1e-9
            assert set(index.class_weights("uniform").values()) == {1.0}
            with pytest.raises(MEDH5ValidationError):
                index.class_weights("magic")

    def test_occupancy_is_optional(self, sample_path, masks):
        with medh5.open(sample_path) as sample:
            annotation = sample.annotations["organs_tp0"]
            payload = build_index(annotation, occupancy=None, max_coords=16)
            assert payload.occupancy is None
            payload = build_index(annotation, occupancy=8, max_coords=16)
            assert payload.occupancy is not None
            assert payload.occupancy.shape[0] == len(annotation.class_ids)

    def test_summary_and_reading(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            indices = read_indices(sample.root)
            summary = indices["organs_tp0"].summary()
            assert summary["max_coords"] == 64
            assert summary["source_digest"].startswith("sha256:")

    def test_sampling_an_empty_class_is_an_error(self, tmp_path, label_set):
        from tests.v1.conftest import write_sample

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(
            tmp_path / "z.medh5", label_set=label_set, masks=masks, index=True
        )
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError):
            sample.index["organs_tp0"].sample_foreground(2, 1)

    def test_S11_3_a_class_found_empty_stays_in_the_contract(self, tmp_path, label_set):
        """A class searched for and not found is `verified absent`, not `unknown`."""
        from tests.v1.conftest import write_sample

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(tmp_path / "c.medh5", label_set=label_set, masks=masks)
        with medh5.open(path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "instances"
            assert set(seg.class_ids) == {1, 2}
            assert seg.is_annotated(2)
            assert not seg.dense([2])[0].any()
