"""The storage layer: chunking, codecs, windowed reads, sampling indices and
recompression (spec §14)."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.storage import (
    COMPRESS_MIN_BYTES,
    MAX_CHUNK_BYTES,
    PROFILES,
    build_index,
    chunk_report,
    dataset_kwargs,
    is_bulk,
    optimize_chunks,
    read_indices,
    resolve_profile,
    spatial_chunk_for,
)
from medh5.validate import validate_file
from tests.helpers import write_sample
from tests.kits import Framed


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
        not cross a fork.  The probe is the engine's now: it reads sysfs, and
        its one process (`sysctl`, for macOS) is compiled in only on macOS.
        What stays observable here: no Python process-spawning entry point is
        reached, and the answer is positive and stable.
        """
        import subprocess

        from medh5.storage import detect_l3_bytes

        def refuse(*args, **kwargs):
            raise AssertionError("L3 detection must not spawn a process here")

        monkeypatch.setattr(os, "popen", refuse)
        monkeypatch.setattr(subprocess, "run", refuse)
        monkeypatch.setattr(subprocess, "Popen", refuse)
        found = detect_l3_bytes()
        assert found > 0
        assert detect_l3_bytes() == found

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
        with medh5.open(sample_path) as sample:
            stored = sample.root["annotations/organs_tp0/layer_class_ids"]
            assert not is_bulk(stored)


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
            from medh5.geometry import box_to_slices

            assert np.array_equal(
                masks[1][box_to_slices(box)],
                masks[1][masks[1].any(axis=(1, 2))][:, masks[1].any(axis=(0, 2))][
                    :, :, masks[1].any(axis=(0, 1))
                ],
            )

    def test_empty_class_has_no_bbox_and_no_samples(self, tmp_path, label_set):

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

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(
            tmp_path / "z.medh5", label_set=label_set, masks=masks, index=True
        )
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError):
            sample.index["organs_tp0"].sample_foreground(2, 1)

    def test_S11_3_a_class_found_empty_stays_in_the_contract(self, tmp_path, label_set):
        """A class searched for and not found is `verified absent`, not `unknown`."""

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(tmp_path / "c.medh5", label_set=label_set, masks=masks)
        with medh5.open(path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "instances"
            assert set(seg.class_ids) == {1, 2}
            assert seg.is_annotated(2)
            assert not seg.dense([2])[0].any()

    def test_P06_the_occupancy_map_equals_the_loop_it_replaced(self):
        from medh5.storage import occupancy as _occupancy

        rng = np.random.default_rng(0)
        for shape in [(9, 7, 5), (16, 16, 16), (33, 17, 8)]:
            mask = rng.random(shape) > 0.9
            coarse = tuple(max(1, -(-n // 8)) for n in shape)
            expected = np.zeros(coarse, dtype=bool)
            for block in np.ndindex(*coarse):
                window = tuple(
                    slice(i * 8, min((i + 1) * 8, n))
                    for i, n in zip(block, shape, strict=True)
                )
                expected[block] = bool(mask[window].any())
            assert np.array_equal(_occupancy(mask, 8), expected), shape

    def test_P12_S14_3_the_occupancy_map_is_compressed_one_class_per_chunk(
        self, tmp_path: Path
    ):
        """52 MB of mostly `False` at 200 classes and 512³, stored contiguous."""
        # 40 classes x (64/8, 128/8, 128/8): 80 KiB, over the compression floor.
        path = Framed.many_classes(
            tmp_path / "occ.medh5", classes=40, shape=(64, 128, 128)
        )
        with h5py.File(path, "r") as handle:
            occupancy = handle["index/organs/occupancy"]
            assert occupancy.chunks == (1, *occupancy.shape[1:])
            assert occupancy.compression == "gzip"
            assert occupancy.id.get_storage_size() < occupancy.nbytes / 10
        with medh5.open(path) as s:
            assert s.verify().ok and "organs" in s.fresh_indices


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_S14_2_a_window_reads_what_hdf5plugin_wrote(
    byteorder: str, tmp_path: Path
) -> None:
    """A window decompresses only the Blosc2 blocks it covers, from the stored
    chunk.  What another tool wrote must read the same as HDF5 reads it:
    `hdf5plugin`'s B2ND chunks through that path, a big-endian dataset --- whose
    bytes HDF5 converts on the way out --- through HDF5's."""
    hdf5plugin = pytest.importorskip("hdf5plugin")
    path = tmp_path / "other.medh5"
    w = medh5.create(path, sample_id="o", subject_id="s")
    w.add_grid("g", shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    w.add_image("CT", np.zeros((8, 8, 8), np.int16), grid="g", modality="CT")
    w.commit()
    rng = np.random.default_rng(7)
    values = rng.integers(-1000, 1500, (37, 45, 53)).astype(f"{byteorder}i2")
    values[:, :20] = 3  # runs, so blocks compress unevenly
    with h5py.File(path, "r+") as f:
        f.create_dataset(
            "x_other/ct",
            data=values,
            chunks=(16, 16, 32),
            **hdf5plugin.Blosc2(
                cname="zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE
            ),
        )
    with medh5.open(path) as s:
        ds = s.root["x_other/ct"]
        for _ in range(40):
            lo = [int(rng.integers(0, n)) for n in values.shape]
            hi = [
                int(rng.integers(a + 1, n + 1))
                for a, n in zip(lo, values.shape, strict=True)
            ]
            window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
            np.testing.assert_array_equal(ds[window], values[window])
        np.testing.assert_array_equal(ds[5, 3:40, 7], values[5, 3:40, 7])


class TestRecompress:
    def test_S13_1_recompression_preserves_the_content_id(self, tmp_path, label_set):
        from medh5.storage import recompress

        path = tmp_path / "big.medh5"
        shape = (48, 64, 64)
        rng = np.random.default_rng(5)
        mask = np.zeros(shape, dtype=bool)
        mask[4:20, 8:40, 8:40] = True
        with medh5.create(path, codec="training") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid="g",
                modality="CT",
            )
            w.add_segmentation("organs", grid="g", masks={1: mask})
        with medh5.open(path) as sample:
            before = sample.content_id
            values = sample.images["CT"].read()

        result = recompress(path, "archive")
        assert result.content_id_preserved
        assert result.content_id == before
        assert result.datasets >= 1
        assert any("zstd" in after for _, _, after in result.changed)

        with medh5.open(path) as sample:
            assert sample.content_id == before
            assert np.array_equal(sample.images["CT"].read(), values)
            assert np.array_equal(sample.annotations["organs"].dense([1])[0], mask)
            assert sample.verify().ok

    def test_out_writes_beside_the_source(self, tmp_path, indexed_cohort):
        from medh5.storage import recompress

        target = tmp_path / "copy.medh5"
        result = recompress(indexed_cohort[0], "portable", out=target)
        assert target.exists()
        assert Path(indexed_cohort[0]).exists()
        assert result.path == str(target)
        assert "portable" in str(result)

    def test_an_unknown_profile_is_refused(self, indexed_cohort):
        from medh5.storage import recompress

        with pytest.raises(MEDH5ValidationError, match="unknown codec profile"):
            recompress(indexed_cohort[0], "maximum-effort")

    def test_recompress_paths_and_json(self, indexed_cohort):
        from medh5.storage import recompress_paths

        results = recompress_paths(indexed_cohort[:2], "portable")
        assert len(results) == 2
        assert set(results[0].to_json()) >= {"path", "profile", "ratio", "content_id"}

    def test_L28_S14_1_rechunk_chunks_the_way_the_writer_does(self, tmp_path: Path):
        """A `layers` dataset went to h5py's (2, 16, 24, 48), and drew W902."""
        from medh5.storage import fit_chunks, grid_chunks, recompress

        shape = (32, 48, 48)
        rng = np.random.default_rng(0)
        masks = {}
        for c in range(1, 5):
            mask = np.zeros(shape, bool)
            mask[int(rng.integers(0, 16)) :, int(rng.integers(0, 24)) :, :] = True
            masks[c] = mask
        path = tmp_path / "rc.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(16, 16, 16))
        w.add_image(
            "CT", rng.integers(0, 1000, shape).astype(np.int16), grid="g", modality="CT"
        )
        w.label_set(Framed.label_set(4))
        w.add_segmentation("organs", grid="g", masks=masks, encoding="layers")
        w.commit()
        # Re-chunk it the way some other tool might have, then ask for the rule.
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs/data"][...]
            attrs = dict(handle["annotations/organs/data"].attrs)
            del handle["annotations/organs/data"]
            node = handle["annotations/organs"].create_dataset(
                "data", data=data, chunks=(2, 8, 12, 24), compression="gzip"
            )
            node.attrs.update(attrs)
        result = recompress(path, "portable", rechunk=True)
        assert result.ok
        with medh5.open(path) as s:
            grid = s.grids["g"]
        with h5py.File(path, "r") as handle:
            data = handle["annotations/organs/data"]
            assert data.chunks == fit_chunks(
                grid_chunks(grid, data.dtype.itemsize, leading=1), data.shape
            )
            assert data.chunks[0] == 1
            image = handle["images/CT"]
            assert image.chunks == grid_chunks(grid, image.dtype.itemsize)
        assert "W902" not in validate_file(path, level="strict").codes

    def test_L28_rechunk_finds_each_sample_root_in_a_collection(self, tmp_path: Path):
        from medh5.storage import recompress

        shape = (32, 64, 64)  # 256 KiB of layers: chunked
        paths = []
        for i in range(2):
            path = tmp_path / f"m{i}.medh5"
            first = np.zeros(shape, bool)
            first[2:20] = True
            second = np.zeros(shape, bool)
            second[10:28] = True
            w = medh5.create(
                path, sample_id=f"m{i}", subject_id=f"p{i}", codec="portable"
            )
            w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(8, 8, 8))
            w.add_image("CT", np.ones(shape, np.int16), grid="g", modality="CT")
            w.label_set(Framed.label_set(2))
            w.add_segmentation(
                "seg", grid="g", masks={1: first, 2: second}, encoding="layers"
            )
            w.commit()
            paths.append(path)
        shard = tmp_path / "shard.medh5c"
        medh5.pack(paths, shard)
        # Verifying the output read the collection root as a sample, and raised
        # a KeyError for its `/meta` after the file had been replaced.
        result = recompress(shard, "balanced", rechunk=True)
        assert result.ok and result.verified and result.content_id_preserved
        with h5py.File(shard, "r") as handle:
            for key in ("m0", "m1"):
                data = handle[f"samples/{key}/annotations/seg/data"]
                assert data.chunks is not None and data.chunks[0] == 1
        with medh5.open_collection(shard) as collection:
            assert all(collection[key].verify().ok for key in collection)
