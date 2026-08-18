"""PyTorch datasets, handle safety, MONAI and recompression (plan §2.3, §4.3)."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.sampling import PatchSampler, TimepointPairSampler
from tests.v1.conftest import SHAPE, block, write_sample

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from medh5.torch import (  # noqa: E402
    CACHE,
    GridPatchDataset,
    HandleCache,
    PairedPatchDataset,
    PatchDataset,
    VolumeDataset,
    collate,
    open_cached,
    stack_images,
    worker_init_fn,
)


@pytest.fixture
def cohort(tmp_path: Path, label_set, masks) -> list[Path]:
    return [
        write_sample(
            tmp_path / f"case{i}.medh5",
            label_set=label_set,
            masks=masks,
            sample_id=f"case{i}",
            index=True,
        )
        for i in range(3)
    ]


class TestVolumeDataset:
    def test_one_item_per_file(self, cohort):
        ds = VolumeDataset(
            cohort, images=["CT_tp0"], annotations={"organs_tp0": [1, 2]}
        )
        assert len(ds) == len(cohort)
        item = ds[1]
        assert tuple(item["images"]["CT_tp0"].shape) == SHAPE
        assert tuple(item["label"]["organs_tp0"].shape) == (2, *SHAPE)
        assert item["meta"]["sample_id"] == "case1"

    def test_labelmap_format_is_one_plane(self, cohort):
        ds = VolumeDataset(
            cohort, annotations={"organs_tp0": [1, 2, 3]}, label_format="labelmap"
        )
        assert tuple(ds[0]["label"]["organs_tp0"].shape) == SHAPE

    def test_label_format_none_returns_images_only(self, cohort):
        ds = VolumeDataset(cohort, annotations={"organs_tp0": []}, label_format="none")
        assert "label" not in ds[0]

    def test_physical_values_are_rescaled(self, tmp_path, label_set):
        path = tmp_path / "rescaled.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                np.full(SHAPE, 10, dtype=np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=2.0,
                rescale_intercept=-5.0,
            )
        assert VolumeDataset([path])[0]["images"]["CT"][0, 0, 0].item() == 15.0
        raw = VolumeDataset([path], physical=False)[0]["images"]["CT"]
        assert raw[0, 0, 0].item() == 10.0

    def test_missing_objects_are_named(self, cohort):
        with pytest.raises(MEDH5ValidationError, match="no image"):
            VolumeDataset(cohort, images=["nope"])[0]
        with pytest.raises(MEDH5ValidationError, match="no annotation"):
            VolumeDataset(cohort, annotations={"nope": []})[0]

    def test_an_empty_dataset_and_bad_format_are_refused(self, cohort):
        with pytest.raises(MEDH5ValidationError):
            VolumeDataset([])
        with pytest.raises(MEDH5ValidationError):
            VolumeDataset(cohort, label_format="protobuf")


class TestPatchDataset:
    def test_length_is_files_times_samples(self, cohort):
        ds = PatchDataset(cohort, PatchSampler(8), samples_per_volume=4)
        assert len(ds) == 12

    def test_patches_have_the_requested_shape(self, cohort):
        ds = PatchDataset(
            cohort,
            PatchSampler((8, 8, 8), strategy="foreground"),
            annotations={"organs_tp0": [1, 2]},
            samples_per_volume=2,
        )
        item = ds[0]
        assert tuple(item["images"]["CT_tp0"].shape) == (8, 8, 8)
        assert tuple(item["label"]["organs_tp0"].shape) == (2, 8, 8, 8)
        assert item["meta"]["patch"]["strategy"] == "foreground"

    def test_a_patch_larger_than_the_volume_is_padded(self, cohort):
        ds = PatchDataset(cohort, PatchSampler(64, strategy="uniform"))
        assert tuple(ds[0]["images"]["CT_tp0"].shape) == (64, 64, 64)

    def test_draws_are_reproducible_and_epoch_dependent(self, cohort):
        ds = PatchDataset(cohort, PatchSampler(8), seed=11)
        first = ds[0]["meta"]["patch"]["center"]
        assert ds[0]["meta"]["patch"]["center"] == first
        ds.set_epoch(1)
        assert ds[0]["meta"]["patch"]["center"] != first

    def test_samples_per_volume_must_be_positive(self, cohort):
        with pytest.raises(MEDH5ValidationError):
            PatchDataset(cohort, PatchSampler(8), samples_per_volume=0)

    def test_instances_label_format_returns_objects(self, tmp_path, label_set):
        from medh5.annotations.voxel import InstanceInput

        path = tmp_path / "inst.medh5"
        mask = np.zeros(SHAPE, dtype=bool)
        mask[4:8, 4:8, 4:8] = True
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "les",
                grid="g",
                instances=[InstanceInput(class_id=3, instance_id=1, mask=mask)],
            )
        ds = PatchDataset(
            [path],
            PatchSampler(16, strategy="uniform"),
            annotations={"les": []},
            label_format="instances",
        )
        objects = ds[0]["label"]["les"]
        assert isinstance(objects, list)
        for obj in objects:
            assert set(obj) == {"instance_id", "class_id", "box", "score"}


class TestPatchGrids:
    def test_S14_3_a_patch_is_not_read_out_of_two_grids(
        self, tmp_path, label_set, masks
    ):
        """A patch is a window in one grid's index space, not a shared coordinate.

        Reading a second grid with those same slices assumes both index the same
        anatomy at the same spacing, and a smaller one quietly returns a
        truncated tensor because the padding was computed for the first.
        """
        path = write_sample(
            tmp_path / "two.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        spanning = PatchDataset(
            [path], PatchSampler(8), annotations={"organs_tp0": [1]}
        )
        with pytest.raises(MEDH5ValidationError, match="one grid"):
            spanning[0]

        scoped = PatchDataset(
            [path],
            PatchSampler(8),
            images=["CT_tp0"],
            annotations={"organs_tp0": [1]},
        )
        assert tuple(scoped[0]["images"]["CT_tp0"].shape) == (8, 8, 8)

    def test_S14_3_the_window_grid_is_part_of_the_check(self, tmp_path, label_set):
        """A single selected image agrees with itself; that is not the question.

        A window drawn on the annotation's grid and applied to an image on a
        different one looks like one grid to a check that only compares the
        objects being read --- and comes back silently misregistered, truncated
        wherever the target grid is the smaller of the two.  The window is a
        member of the comparison now, so it carries the grid it was measured in.
        """
        path = tmp_path / "two.medh5"
        big, small = (16, 16, 16), (10, 10, 10)
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1")
            w.label_set(label_set)
            for tp, shape, frame in (("tp0", big, "F0"), ("tp1", small, "F1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
            w.add_segmentation(
                "ann_A", grid="g_tp0", masks={1: block(big, (2, 2, 2), 6)}
            )

        sampler = PatchSampler(8, strategy="foreground")
        crossed = PatchDataset(
            [path], sampler, annotation="ann_A", images=["CT_tp1"], label_format="none"
        )
        with pytest.raises(MEDH5ValidationError, match="the patch window"):
            crossed[0]

        aligned = PatchDataset(
            [path], sampler, annotation="ann_A", images=["CT_tp0"], label_format="none"
        )
        assert tuple(aligned[0]["images"]["CT_tp0"].shape) == (8, 8, 8)

    def test_S14_3_a_grid_cover_refuses_an_image_off_the_reference_grid(
        self, tmp_path, label_set
    ):
        """`grid_patches` measures its windows in the reference grid."""
        path = tmp_path / "cover.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1")
            w.label_set(label_set)
            for tp, shape in (("tp0", (16, 16, 16)), ("tp1", (10, 10, 10))):
                w.add_grid(
                    f"g_{tp}", shape=shape, spacing=(1.0, 1.0, 1.0), timepoint=tp
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
        reference = GridPatchDataset([path], 8, images=["CT_tp0"])
        assert tuple(reference[0]["images"]["CT_tp0"].shape) == (8, 8, 8)
        with pytest.raises(MEDH5ValidationError, match="the patch window"):
            GridPatchDataset([path], 8, images=["CT_tp1"])[0]

    def test_a_whole_volume_read_still_spans_every_grid(
        self, tmp_path, label_set, masks
    ):
        """The refusal is about the *window*: without one there is nothing to align."""
        path = write_sample(
            tmp_path / "two.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        item = VolumeDataset([path])[0]
        assert set(item["images"]) == {"CT_tp0", "CT_tp1"}


class TestGridPatchDataset:
    def test_the_plan_covers_every_file(self, cohort):
        ds = GridPatchDataset(cohort, 8, images=["CT_tp0"])
        per_file = len(ds) // len(cohort)
        assert per_file > 1
        assert tuple(ds[0]["images"]["CT_tp0"].shape) == (8, 8, 8)
        assert ds[0]["meta"]["patch"]["strategy"] == "grid"


class TestPairedPatchDataset:
    @pytest.fixture
    def registered(self, tmp_path, label_set, masks) -> Path:
        """Two visits whose frames differ by a known +4-voxel shift."""
        return self._visits(tmp_path, label_set, masks, transform=True)

    @pytest.fixture
    def unregistered(self, tmp_path, label_set, masks) -> Path:
        """The same two visits, with nothing in the file relating their frames."""
        return self._visits(tmp_path, label_set, masks, transform=False)

    def _visits(self, tmp_path, label_set, masks, *, transform: bool) -> Path:
        path = tmp_path / ("pair.medh5" if transform else "unrelated.medh5")
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp in ("tp0", "tp1"):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(SHAPE, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
                w.add_segmentation(f"organs_{tp}", grid=f"g_{tp}", masks=masks)
            if transform:
                shift = np.eye(4)
                shift[2, 3] = 4.0
                w.add_transform(
                    "tp0_to_tp1",
                    kind="affine",
                    matrix=shift,
                    from_frame="pseudo:tp0",
                    to_frame="pseudo:tp1",
                )
        return path

    def test_align_transform_moves_the_window_by_the_transform(self, registered):
        """The hand-checked fixture: a +4-voxel shift must move the patch by 4."""
        ds = PairedPatchDataset(
            [registered], PatchSampler(8, strategy="foreground"), align="transform"
        )
        meta = ds[0]["meta"]
        first = meta["patches"]["tp0"]["center"]
        second = meta["patches"]["tp1"]["center"]
        assert [b - a for a, b in zip(first, second, strict=True)] == [0, 0, 4]
        assert meta["interval_days"] == 90

    def test_align_none_reads_the_same_index_window(self, registered):
        ds = PairedPatchDataset(
            [registered], PatchSampler(8, strategy="foreground"), align="none"
        )
        meta = ds[0]["meta"]
        assert meta["patches"]["tp0"]["center"] == meta["patches"]["tp1"]["center"]

    def test_both_visits_are_read(self, registered):
        item = PairedPatchDataset([registered], PatchSampler(8))[0]
        assert set(item["images"]) == {"tp0", "tp1"}
        assert tuple(item["images"]["tp1"]["CT_tp1"].shape) == (8, 8, 8)

    def test_a_cross_sectional_file_is_counted_not_dropped(self, registered, cohort):
        ds = PairedPatchDataset([registered, *cohort], PatchSampler(8))
        assert ds.report.files == 1 + len(cohort)
        assert ds.report.pairs == 1
        assert len(ds.report.skipped) == len(cohort)
        assert "cross-sectional" in str(ds.report)
        assert ds.report.summary()["skipped_cross_sectional"] == len(cohort)

    def test_a_change_label_travels_with_the_pair(self, registered):
        with medh5.amend(registered) as w:
            w.add_classification(
                "response", labels={3: 1.0}, scope="sample", timepoints=["tp0", "tp1"]
            )
        item = PairedPatchDataset([registered], PatchSampler(8))[0]
        assert item["label"]["response"] == {"lesion": 1.0}
        assert item["meta"]["label_annotation"] == "response"

    def test_S10_2_align_transform_refuses_frames_it_cannot_relate(self, unregistered):
        """`transform_between` returns None for "one frame" and "no path" alike.

        Only the first makes the two index spaces comparable.  Reading the
        second as "nothing to apply" feeds source-frame world coordinates into
        an unrelated grid and pairs patches from different anatomy.
        """
        ds = PairedPatchDataset(
            [unregistered], PatchSampler(8, strategy="foreground"), align="transform"
        )
        with pytest.raises(MEDH5ValidationError, match="needs a transform"):
            ds[0]
        # the same file pairs fine when the caller does not claim alignment
        same = PairedPatchDataset([unregistered], PatchSampler(8), align="none")[0]
        assert set(same["images"]) == {"tp0", "tp1"}

    def test_S3_7_a_padded_first_visit_keeps_the_pair_one_shape(
        self, tmp_path, label_set
    ):
        """The second visit is asked for the requested size, not the clipped one.

        Where the first visit is smaller than the patch its window is short and
        padded back up to the patch shape; asking the second for the clipped
        extent returns a full, unpadded array of that smaller size, and the
        paired tensors no longer stack.
        """
        path = tmp_path / "uneven.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, shape in (("tp0", (8, 8, 8)), ("tp1", (24, 24, 24))):
                w.add_grid(
                    f"g_{tp}",
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
                w.add_segmentation(
                    f"organs_{tp}",
                    grid=f"g_{tp}",
                    masks={1: block(shape, (1, 1, 1), 4)},
                )

        item = PairedPatchDataset([path], PatchSampler(12), align="none")[0]
        assert tuple(item["images"]["tp0"]["CT_tp0"].shape) == (12, 12, 12)
        assert tuple(item["images"]["tp1"]["CT_tp1"].shape) == (12, 12, 12)

    def test_S3_7_a_pair_annotated_at_only_one_visit_still_loads(
        self, tmp_path, label_set
    ):
        """`annotation=None` told the sampler to find one *anywhere* in the file.

        For a longitudinal sample that is another visit's, whose coordinates
        are in another visit's grid --- so the baseline window was drawn in the
        follow-up's space.  That used to be a silent misread and, once the
        window carried its grid, a refusal of a perfectly good pair.  The
        timepoint's grid is named explicitly now.
        """
        path = tmp_path / "partial.medh5"
        shape = (12, 12, 12)
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, frame in (("tp0", "F0"), ("tp1", "F1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
            # Only the follow-up is annotated.
            w.add_segmentation(
                "ann_tp1", grid="g_tp1", masks={1: block(shape, (2, 2, 2), 4)}
            )

        item = PairedPatchDataset([path], PatchSampler(8), align="none")[0]
        assert set(item["images"]) == {"tp0", "tp1"}
        assert tuple(item["images"]["tp0"]["CT_tp0"].shape) == (8, 8, 8)
        grids = {tp: p["grid_id"] for tp, p in item["meta"]["patches"].items()}
        assert grids == {"tp0": "g_tp0", "tp1": "g_tp1"}, "each in its own visit"

    def test_unknown_alignment_is_refused(self, registered):
        with pytest.raises(MEDH5ValidationError):
            PairedPatchDataset([registered], PatchSampler(8), align="telepathy")

    def test_pair_modes_change_the_plan(self, registered):
        ds = PairedPatchDataset(
            [registered],
            PatchSampler(8),
            pair_sampler=TimepointPairSampler("all_pairs"),
            samples_per_pair=2,
        )
        assert len(ds) == 2


class TestCollate:
    def test_stacks_tensors_and_keeps_metadata(self, cohort):
        ds = PatchDataset(
            cohort, PatchSampler(8), annotations={"organs_tp0": [1, 2]}, seed=3
        )
        batch = collate([ds[0], ds[1]])
        assert tuple(batch["images"]["CT_tp0"].shape) == (2, 8, 8, 8)
        assert tuple(batch["label"]["organs_tp0"].shape) == (2, 2, 8, 8, 8)
        assert len(batch["meta"]["sample_id"]) == 2

    def test_a_shape_mismatch_names_the_key(self, cohort):
        small = PatchDataset(cohort, PatchSampler(8))[0]
        large = PatchDataset(cohort, PatchSampler(16))[0]
        with pytest.raises(MEDH5ValidationError, match="images.CT_tp0"):
            collate([small, large])

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            collate([])

    def test_stack_images_builds_a_channel_axis(self, cohort):
        ds = PatchDataset(cohort, PatchSampler(8))
        stacked = stack_images([ds[0], ds[1]], ["CT_tp0"])
        assert tuple(stacked.shape) == (2, 1, 8, 8, 8)
        with pytest.raises(MEDH5ValidationError, match="missing image"):
            stack_images([ds[0]], ["PET"])


class TestDataLoader:
    def test_a_real_dataloader_round_trips(self, cohort):
        ds = PatchDataset(
            cohort,
            PatchSampler(8, strategy="balanced"),
            annotations={"organs_tp0": [1, 2]},
            samples_per_volume=2,
        )
        loader = DataLoader(
            ds,
            batch_size=2,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        batches = list(loader)
        assert len(batches) == 3
        assert tuple(batches[0]["images"]["CT_tp0"].shape) == (2, 8, 8, 8)

    def test_S14_4_a_soak_does_not_grow_the_handle_cache(self, cohort):
        """10 epochs must not leak handles or file descriptors (plan §4.1)."""
        CACHE.clear()
        ds = PatchDataset(cohort, PatchSampler(8), samples_per_volume=4)
        loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate)
        counts = []
        for epoch in range(10):
            ds.set_epoch(epoch)
            for _ in loader:
                pass
            gc.collect()
            counts.append((len(CACHE), _open_fds()))
        assert len({c[0] for c in counts}) == 1, f"handle cache grew: {counts}"
        assert counts[-1][1] <= counts[0][1] + 2, f"file descriptors grew: {counts}"
        assert len(CACHE) <= len(cohort)


def _open_fds() -> int:
    """Open file descriptors for this process, or 0 where /dev/fd is absent."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:  # pragma: no cover - platform without /dev/fd
        return 0


class TestHandleCache:
    def test_S14_4_handles_are_reused_within_a_process(self, cohort):
        cache = HandleCache(maxsize=4)
        first = cache.get(cohort[0])
        assert cache.get(cohort[0]) is first
        assert cache.opens == 1

    def test_the_lru_bound_is_honoured(self, cohort):
        cache = HandleCache(maxsize=2)
        for path in cohort:
            cache.get(path)
        assert len(cache) == 2
        cache.close_all()
        assert len(cache) == 0

    def test_S14_4_a_new_pid_abandons_inherited_handles(self, cohort, monkeypatch):
        """A forked child must never touch the parent's HDF5 descriptors."""
        cache = HandleCache()
        cache.get(cohort[0])
        assert len(cache) == 1
        monkeypatch.setattr(os, "getpid", lambda: cache.owner_pid + 1)
        cache.get(cohort[1])
        assert len(cache) == 1, "the inherited handle must be dropped, not reused"

    def test_S14_4_close_all_abandons_handles_it_did_not_open(
        self, cohort, monkeypatch
    ):
        """The `atexit` hook runs in forked children too.

        A child that exits through normal interpreter shutdown would otherwise
        call into HDF5 to close descriptors belonging to its parent --- the one
        thing this module exists to prevent, and the one place the PID check was
        missing.  `worker_init_fn` covers only callers who pass it.
        """
        cache = HandleCache()
        sample = cache.get(cohort[0])
        monkeypatch.setattr(os, "getpid", lambda: cache.owner_pid + 1)

        cache.close_all()

        assert len(cache) == 0, "the child drops what it inherited"
        assert sample.identity.sample_id == "case0", "and leaves it open for the parent"

    def test_S14_4_building_a_plan_does_not_evict_the_shared_cache(self, cohort):
        """`CACHE` is process-global; a constructor must not spend it on a scan.

        Walking a cohort through a 32-entry LRU to build the plan evicts what
        training was about to reuse and leaves the cache holding the tail of the
        cohort --- chosen by plan order, not by what is being read.  The
        metadata pass itself is unavoidable: `__len__` needs one read per file.
        """
        CACHE.clear()
        warm = open_cached(cohort[0])
        opens = CACHE.opens

        GridPatchDataset(cohort, 8)
        PairedPatchDataset(cohort, PatchSampler(8))

        assert len(CACHE) == 1, "the scan left the shared cache alone"
        assert CACHE.opens == opens, "and opened nothing through it"
        assert open_cached(cohort[0]) is warm

    def test_worker_init_clears_the_module_cache(self, cohort):
        open_cached(cohort[0])
        assert len(CACHE) >= 1
        worker_init_fn(0)
        assert len(CACHE) == 0

    def test_set_cache_size(self):
        from medh5.torch.handles import set_cache_size

        before = CACHE.maxsize
        set_cache_size(4)
        assert CACHE.maxsize == 4
        set_cache_size(before)


class TestMonai:
    def test_the_affine_is_the_grid_affine(self, cohort):
        from medh5.monai import affine_for, meta_dict

        with medh5.open(cohort[0]) as sample:
            affine = affine_for(sample, "CT_tp0")
            assert np.allclose(affine, sample.grids["ct_tp0"].affine)
            meta = meta_dict(sample, "CT_tp0")
            assert meta["space"] == "LPS"
            assert meta["medh5"]["modality"] == "CT"
            assert list(meta["spatial_shape"]) == list(SHAPE)

    def test_S3_1_LPS_to_RAS_flips_only_the_first_two_world_axes(self, cohort):
        from medh5.monai import affine_for, convert_affine

        with medh5.open(cohort[0]) as sample:
            lps = affine_for(sample, "CT_tp0")
            ras = affine_for(sample, "CT_tp0", space="RAS")
        assert np.allclose(ras[:2], -lps[:2])
        assert np.allclose(ras[2:], lps[2:])
        assert np.allclose(convert_affine(lps, source="LPS", target="LPS"), lps)

    def test_a_conversion_it_cannot_justify_is_refused(self):
        from medh5.monai import convert_affine

        with pytest.raises(MEDH5ValidationError, match="3-D"):
            convert_affine(np.eye(3), source="LPS", target="RAS")
        with pytest.raises(MEDH5ValidationError, match="world convention"):
            convert_affine(np.eye(4), source="LPS", target="talairach")

    def test_an_roi_moves_the_origin(self, cohort):
        from medh5.monai import _shift_origin, affine_for

        with medh5.open(cohort[0]) as sample:
            grid = sample.grids["ct_tp0"]
            affine = affine_for(sample, "CT_tp0")
            roi = (slice(2, 6), slice(4, 8), slice(0, 4))
            shifted = _shift_origin(affine, roi)
        assert np.allclose(shifted[:3, 3], grid.index_to_world([[2, 4, 0]])[0])
        assert np.allclose(shifted[:3, :3], affine[:3, :3])

    def test_S4_3_a_MetaTensor_reads_the_level_its_affine_describes(self, tmp_path):
        """Level-0 voxels under a level-1 affine misplace every saved prediction.

        `meta_dict` already selects the requested level's grid and affine, so the
        array has to come from that level too; nothing about the MetaTensor
        would say the two disagree.
        """
        pytest.importorskip("monai")
        from medh5.geometry.multiscale import derive_level_grid
        from medh5.monai import to_metatensor

        shape = (16, 32, 32)
        path = tmp_path / "pyr.medh5"
        with medh5.create(path, codec="portable") as w:
            base = w.add_grid(
                "l0", shape=shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
            )
            level1 = derive_level_grid(base, (2, 2, 2), "l1")
            w.add_grid(
                "l1",
                shape=level1.shape,
                spacing=level1.spacing,
                origin=level1.origin,
                direction=level1.direction,
            )
            w.add_pyramid(
                "CT",
                [
                    np.zeros(shape, dtype=np.int16),
                    np.full((8, 16, 16), 7, dtype=np.int16),
                ],
                grid_levels=["l0", "l1"],
                modality="CT",
            )

        with medh5.open(path) as sample:
            tensor = to_metatensor(sample, "CT", level=1)
            assert tuple(tensor.shape) == (8, 16, 16), "the level-1 array, not level 0"
            assert float(np.asarray(tensor).flat[0]) == 7.0
            assert list(tensor.meta["spatial_shape"]) == [8, 16, 16]
            assert np.allclose(
                np.asarray(tensor.meta["affine"]),
                sample.images["CT"].level(1).grid.affine,
            )

    def test_medh5_metadata_is_json_safe(self, cohort):
        from medh5.monai import meta_dict

        with medh5.open(cohort[0]) as sample:
            json.dumps(meta_dict(sample, "CT_tp0")["medh5"])


class TestRecompress:
    def test_S13_1_recompression_preserves_the_content_id(self, tmp_path, label_set):
        from medh5.storage.recompress import recompress

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

    def test_out_writes_beside_the_source(self, tmp_path, cohort):
        from medh5.storage.recompress import recompress

        target = tmp_path / "copy.medh5"
        result = recompress(cohort[0], "portable", out=target)
        assert target.exists()
        assert Path(cohort[0]).exists()
        assert result.path == str(target)
        assert "portable" in str(result)

    def test_an_unknown_profile_is_refused(self, cohort):
        from medh5.storage.recompress import recompress

        with pytest.raises(MEDH5ValidationError, match="unknown codec profile"):
            recompress(cohort[0], "maximum-effort")

    def test_recompress_paths_and_json(self, cohort):
        from medh5.storage.recompress import recompress_paths

        results = recompress_paths(cohort[:2], "portable")
        assert len(results) == 2
        assert set(results[0].to_json()) >= {"path", "profile", "ratio", "content_id"}


class TestBench:
    def test_the_metrics_run_and_are_reported(self, cohort):
        from medh5.bench import TARGETS, benchmark_file, report

        measurements = benchmark_file(cohort[0], patch=8, repeats=2)
        names = {m.name for m in measurements}
        assert set(TARGETS) <= names
        assert all(m.value >= 0 for m in measurements)
        assert "target" in report(measurements) or "all targets met" in report(
            measurements
        )

    def test_a_measurement_knows_whether_it_met_its_target(self):
        from medh5.bench import Measurement

        assert Measurement("x", 1.0, target=2.0).ok
        assert not Measurement("x", 3.0, target=2.0).ok
        assert Measurement("x", 3.0).ok
        assert "!" in str(Measurement("x", 3.0, target=2.0))
        assert Measurement("x", 1.0).to_json()["ok"] is True

    def test_timed_returns_a_median(self):
        from medh5.bench import timed

        assert timed(lambda: None, repeats=3, warmup=1) >= 0.0

    def test_the_throughput_run_measures_steady_state(self, cohort):
        from medh5.bench import throughput

        measured = throughput(cohort, patch=8, batches=2, batch_size=1, workers=0)
        assert measured.unit == "patches/s"
        assert measured.value > 0
        assert measured.detail["workers"] == 0


class TestWithoutMonai:
    """The parts of the adapter that do not need MONAI installed."""

    def test_from_metatensor_recovers_the_geometry(self):
        from medh5.monai import from_metatensor

        class Duck:
            """Anything with `.meta` and array semantics --- MONAI's own shape."""

            def __init__(self, array, meta):
                self._array = array
                self.meta = meta

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._array, dtype=dtype)

        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [5.0, 6.0, 7.0]
        array, geometry = from_metatensor(
            Duck(np.zeros((4, 4, 4)), {"affine": affine, "space": "RAS"})
        )
        assert array.shape == (4, 4, 4)
        assert geometry["spacing"] == [2.0, 1.0, 1.0]
        assert geometry["origin"] == [5.0, 6.0, 7.0]
        assert geometry["coord_system"] == "RAS"

    def test_a_tensor_without_metadata_still_decomposes(self):
        from medh5.monai import from_metatensor

        array, geometry = from_metatensor(np.zeros((2, 2)))
        assert geometry["spacing"] == [1.0, 1.0]
        assert geometry["coord_system"] == "LPS"


class TestBatchedDense:
    """The phase-6 read optimisation must not change what `dense` returns."""

    @pytest.mark.parametrize("encoding", ["layers", "bitmask"])
    def test_batched_reads_agree_with_per_class_reads(
        self, tmp_path, label_set, masks, encoding
    ):
        path = write_sample(
            tmp_path / f"{encoding}.medh5",
            label_set=label_set,
            masks=masks,
            encoding=encoding,
        )
        with medh5.open(path) as sample:
            ann = sample.annotations["organs_tp0"]
            assert ann.kind == encoding
            wanted = list(ann.class_ids)
            roi = [slice(2, 10), slice(2, 12), slice(2, 12)]
            batched = ann.dense(wanted, roi=roi)
            one_at_a_time = np.stack(
                [ann.dense([class_id], roi=roi)[0] for class_id in wanted]
            )
            assert np.array_equal(batched, one_at_a_time)
            assert np.array_equal(ann.dense(wanted)[0], masks[wanted[0]])

    def test_an_unstored_class_reads_as_empty(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "layers.medh5",
            label_set=label_set,
            masks=masks,
            encoding="layers",
        )
        with medh5.open(path) as sample:
            ann = sample.annotations["organs_tp0"]
            assert not ann.dense([4])[0].any()


class TestSyntheticBench:
    def test_the_synthetic_sample_is_valid_and_indexed(self, tmp_path):
        from medh5.bench import synthetic_sample
        from medh5.validate import validate_file

        path = synthetic_sample(
            tmp_path, shape=(8, 16, 16), classes=2, codec="portable"
        )
        assert not validate_file(path).errors
        with medh5.open(path) as sample:
            assert "training" in sample.profiles
            assert len(sample.annotations["organs"].class_ids) == 2
