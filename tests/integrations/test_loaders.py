"""The PyTorch datasets, collation and the handle cache (``medh5.torch``)."""

from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError
from medh5.sampling import PatchSampler, TimepointPairSampler
from tests.helpers import SHAPE, block, write_sample
from tests.kits import Framed, Numbered

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


def _open_fds() -> int:
    """Open file descriptors for this process, or 0 where /dev/fd is absent."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:  # pragma: no cover - platform without /dev/fd
        return 0


class TestVolumeDataset:
    def test_one_item_per_file(self, indexed_cohort):
        ds = VolumeDataset(
            indexed_cohort, images=["CT_tp0"], annotations={"organs_tp0": [1, 2]}
        )
        assert len(ds) == len(indexed_cohort)
        item = ds[1]
        assert tuple(item["images"]["CT_tp0"].shape) == SHAPE
        assert tuple(item["label"]["organs_tp0"].shape) == (2, *SHAPE)
        assert item["meta"]["sample_id"] == "case1"

    def test_labelmap_format_is_one_plane(self, indexed_cohort):
        ds = VolumeDataset(
            indexed_cohort,
            annotations={"organs_tp0": [1, 2, 3]},
            label_format="labelmap",
        )
        assert tuple(ds[0]["label"]["organs_tp0"].shape) == SHAPE

    def test_label_format_none_returns_images_only(self, indexed_cohort):
        ds = VolumeDataset(
            indexed_cohort, annotations={"organs_tp0": []}, label_format="none"
        )
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

    def test_missing_objects_are_named(self, indexed_cohort):
        with pytest.raises(MEDH5ValidationError, match="no image"):
            VolumeDataset(indexed_cohort, images=["nope"])[0]
        with pytest.raises(MEDH5ValidationError, match="no annotation"):
            VolumeDataset(indexed_cohort, annotations={"nope": []})[0]

    def test_an_empty_dataset_and_bad_format_are_refused(self, indexed_cohort):
        with pytest.raises(MEDH5ValidationError):
            VolumeDataset([])
        with pytest.raises(MEDH5ValidationError):
            VolumeDataset(indexed_cohort, label_format="protobuf")


class TestPatchDataset:
    def test_length_is_files_times_samples(self, indexed_cohort):
        ds = PatchDataset(indexed_cohort, PatchSampler(8), samples_per_volume=4)
        assert len(ds) == 12

    def test_patches_have_the_requested_shape(self, indexed_cohort):
        ds = PatchDataset(
            indexed_cohort,
            PatchSampler((8, 8, 8), strategy="foreground"),
            annotations={"organs_tp0": [1, 2]},
            samples_per_volume=2,
        )
        item = ds[0]
        assert tuple(item["images"]["CT_tp0"].shape) == (8, 8, 8)
        assert tuple(item["label"]["organs_tp0"].shape) == (2, 8, 8, 8)
        assert item["meta"]["patch"]["strategy"] == "foreground"

    def test_a_patch_larger_than_the_volume_is_padded(self, indexed_cohort):
        ds = PatchDataset(indexed_cohort, PatchSampler(64, strategy="uniform"))
        assert tuple(ds[0]["images"]["CT_tp0"].shape) == (64, 64, 64)

    def test_draws_are_reproducible_and_epoch_dependent(self, indexed_cohort):
        ds = PatchDataset(indexed_cohort, PatchSampler(8), seed=11)
        first = ds[0]["meta"]["patch"]["center"]
        assert ds[0]["meta"]["patch"]["center"] == first
        ds.set_epoch(1)
        assert ds[0]["meta"]["patch"]["center"] != first

    def test_samples_per_volume_must_be_positive(self, indexed_cohort):
        with pytest.raises(MEDH5ValidationError):
            PatchDataset(indexed_cohort, PatchSampler(8), samples_per_volume=0)

    def test_instances_label_format_returns_objects(self, tmp_path, label_set):

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

    @staticmethod
    def _instances(path: Path, label_set: Any, shape: tuple[int, ...]) -> None:
        liver = np.zeros(shape, dtype=bool)
        liver[0:3, 0:3, 0:3] = True
        lesion = np.zeros(shape, dtype=bool)
        lesion[1:3, 1:3, 1:3] = True
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(class_id=1, instance_id=1, mask=liver),
                    InstanceInput(class_id=3, instance_id=2, mask=lesion),
                ],
            )

    def test_L05_a_padded_patch_shifts_boxes_by_its_padding(self, tmp_path, label_set):
        """A 4^3 volume in an 8^3 patch is padded before as well as after; the
        boxes kept the unpadded crop's coordinates, two voxels off."""
        path = tmp_path / "small.medh5"
        self._instances(path, label_set, (4, 4, 4))
        ds = PatchDataset(
            [path],
            PatchSampler(8, strategy="uniform"),
            annotations={"les": []},
            label_format="instances",
        )
        item = ds[0]
        patch = item["meta"]["patch"]
        before = np.asarray([pad[0] for pad in patch["pad"]], dtype=np.float64)
        start = np.asarray(patch["start"], dtype=np.float64)
        assert before.any(), "the patch is padded before"
        with medh5.open(path) as sample:
            boxes = {
                o.instance_id: np.asarray(o.box)
                for o in sample.annotations["les"].instances()
            }
        for obj in item["label"]["les"]:
            expected = boxes[obj["instance_id"]] - start[:, None] + before[:, None]
            assert np.allclose(obj["box"], expected)
        # The lesion's voxels in the padded tensor are where its box says.
        dense = PatchDataset(
            [path],
            PatchSampler(8, strategy="uniform"),
            annotations={"les": ["lesion"]},
            label_format="onehot",
        )[0]["label"]["les"][0]
        (lesion,) = [o for o in item["label"]["les"] if o["class_id"] == 3]
        found = np.argwhere(np.asarray(dense))
        assert np.allclose(lesion["box"][:, 0] + 0.5, found.min(axis=0))
        assert np.allclose(lesion["box"][:, 1] - 0.5, found.max(axis=0))

    def test_L09_instances_are_the_requested_classes(self, tmp_path, label_set):
        path = tmp_path / "two.medh5"
        self._instances(path, label_set, (4, 4, 4))
        every = PatchDataset(
            [path],
            PatchSampler(8, strategy="uniform"),
            annotations={"les": []},
            label_format="instances",
        )
        assert sorted(o["class_id"] for o in every[0]["label"]["les"]) == [1, 3]
        ds = PatchDataset(
            [path],
            PatchSampler(8, strategy="uniform"),
            annotations={"les": ["lesion"]},
            label_format="instances",
        )
        assert [o["class_id"] for o in ds[0]["label"]["les"]] == [3]

    def test_L05_a_box_touching_the_patch_edge_does_not_overlap_it(self):
        """Edges sit half a voxel out: a box ending at ``start - 0.5`` covers
        no voxel of the patch, and one starting at ``stop - 0.5`` none either."""
        from types import SimpleNamespace

        from medh5.sampling import Patch

        def ann(*boxes: Any) -> Any:
            objects = [
                SimpleNamespace(instance_id=i, class_id=1, box=box, score=None)
                for i, box in enumerate(boxes)
            ]
            return SimpleNamespace(instances=lambda: iter(objects))

        patch = Patch(slices=(slice(4, 8),))
        touching = ann([[1.5, 3.5]], [[7.5, 9.5]])
        assert PatchDataset._instances_in(touching, patch) == []
        inside = ann([[2.5, 4.0]], [[7.0, 9.5]])
        boxes = [o["box"].tolist() for o in PatchDataset._instances_in(inside, patch)]
        assert boxes == [[[-1.5, 0.0]], [[3.0, 5.5]]]


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
    def test_the_plan_covers_every_file(self, indexed_cohort):
        ds = GridPatchDataset(indexed_cohort, 8, images=["CT_tp0"])
        per_file = len(ds) // len(indexed_cohort)
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

    def test_F06_S10_1_a_transform_in_other_units_is_not_applied(self, registered):
        """A transform maps coordinates in its own units, and none maps
        between two: one declaring metres between two millimetre grids moved
        every point a thousandth of its displacement (F06 of the round-4
        audit).  The writer refuses to write it, so it is planted."""
        import h5py

        from tests.helpers import encode_attr

        with h5py.File(registered, "r+") as handle:
            handle["transforms/tp0_to_tp1"].attrs["units"] = encode_attr("m")
        ds = PairedPatchDataset(
            [registered], PatchSampler(8, strategy="foreground"), align="transform"
        )
        with pytest.raises(MEDH5ValidationError, match="'g_tp0' is in 'mm'") as exc:
            ds[0]
        assert exc.value.code == "E506"

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

    def test_a_cross_sectional_file_is_counted_not_dropped(
        self, registered, indexed_cohort
    ):
        ds = PairedPatchDataset([registered, *indexed_cohort], PatchSampler(8))
        assert ds.report.files == 1 + len(indexed_cohort)
        assert ds.report.pairs == 1
        assert len(ds.report.skipped) == len(indexed_cohort)
        assert "cross-sectional" in str(ds.report)
        assert ds.report.summary()["skipped_cross_sectional"] == len(indexed_cohort)

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

    @staticmethod
    def _one_frame(path: Path, **second: Any) -> Path:
        """Two visits whose grids share a frame, the second as *second* says."""
        with medh5.create(path, codec="portable") as w:
            for index, tp in enumerate(("tp0", "tp1")):
                w.add_timepoint(tp, index=index, days_from_baseline=index * 90)
                options: dict[str, Any] = {"spacing": (1.0, 1.0, 1.0)}
                if tp == "tp1":
                    options.update(second)
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    timepoint=tp,
                    frame_uid="pseudo:one",
                    **options,
                )
                w.add_image(
                    f"CT_{tp}", np.zeros(SHAPE, np.int16), grid=f"g_{tp}", modality="CT"
                )
        return path

    def test_N10_S3_5_one_frame_in_two_units_pairs_one_place(self, tmp_path):
        """The second visit's grid in metres: the first's centre, in
        millimetres, was read on it as it was --- a thousandfold off (N10 of
        the round-3 audit)."""
        path = self._one_frame(
            tmp_path / "units.medh5", spacing=(0.001, 0.001, 0.001), units="m"
        )
        for _ in range(4):
            meta = PairedPatchDataset([path], PatchSampler(8), align="transform")[0][
                "meta"
            ]
            assert meta["patches"]["tp0"]["center"] == meta["patches"]["tp1"]["center"]

    def test_N10_S3_3_rule_4_one_frame_in_two_conventions_is_refused(self, tmp_path):
        path = self._one_frame(tmp_path / "ras.medh5", coord_system="RAS")
        ds = PairedPatchDataset([path], PatchSampler(8), align="transform")
        with pytest.raises(MEDH5ValidationError, match="coord_system") as caught:
            ds[0]
        assert caught.value.code == "E414"
        same = PairedPatchDataset([path], PatchSampler(8), align="none")[0]
        assert set(same["images"]) == {"tp0", "tp1"}

    def test_S10_2_a_registration_of_another_grid_is_not_this_pair_s(self, tmp_path):
        """A transform must relate the grids being read, not merely the visits.

        A visit may hold a CT grid and a PET grid on different frames.  Asking
        `transform_between` about the two *timepoints* searches every frame of
        one against every frame of the other and returns the first path it
        finds, so a CT registration answered for a PET pair that had none --- and
        its displacement was applied to PET coordinates.  Nothing raised: the
        patches came back the right shape from the wrong place, which is the
        failure the guard beside it exists to prevent.
        """
        shape = (32, 32, 32)
        shift = np.eye(4)
        shift[0, 3] = 10.0
        path = tmp_path / "two_modalities.medh5"
        with medh5.create(path, sample_id="s", subject_id="S") as writer:
            for index, timepoint in enumerate(("tp0", "tp1")):
                writer.add_timepoint(
                    timepoint,
                    index=index,
                    label=timepoint,
                    days_from_baseline=index * 90,
                )
                for modality in ("ct", "pet"):
                    writer.add_grid(
                        f"{modality}_{timepoint}",
                        shape=shape,
                        spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0),
                        timepoint=timepoint,
                        frame_uid=f"pseudo:{modality}-{timepoint}",
                    )
                    writer.add_image(
                        f"{modality.upper()}_{timepoint}",
                        np.zeros(shape, np.int16),
                        grid=f"{modality}_{timepoint}",
                        modality=modality.upper(),
                    )
            # only the CT frames are registered
            writer.add_transform(
                "ct_reg",
                kind="affine",
                from_frame="pseudo:ct-tp0",
                to_frame="pseudo:ct-tp1",
                matrix=shift,
            )

        def dataset(modality):
            return PairedPatchDataset(
                [str(path)],
                PatchSampler(8),
                align="transform",
                images=[f"{modality}_tp0", f"{modality}_tp1"],
            )

        patches = dataset("CT")[0]["meta"]["patches"]
        moved = patches["tp1"]["center"][0] - patches["tp0"]["center"][0]
        assert moved == 10, "the CT pair is aligned by its own registration"

        with pytest.raises(MEDH5ValidationError, match="pet_tp0"):
            dataset("PET")[0]

    def test_S2_3_a_grid_named_like_a_timepoint_still_resolves_its_own_frame(
        self, tmp_path
    ):
        """Grid ids and timepoint ids are separate namespaces (§2.3).

        Uniqueness is scoped to the group, so a grid MAY be named `tp0` beside a
        timepoint `tp0`, and such a file conforms.  `Sample._frames_for` reads a
        key as a timepoint first, so asking it about the *grid* `tp0` answers
        for the whole visit --- every frame of it --- and a CT registration was
        again returned for a PET pair that had none.  Naming a grid after its
        visit is a common enough habit that this is not an exotic file.
        """
        shape = (32, 32, 32)
        shift = np.eye(4)
        shift[0, 3] = 10.0
        path = tmp_path / "colliding_ids.medh5"
        with medh5.create(path, sample_id="s", subject_id="S") as writer:
            for index, timepoint in enumerate(("tp0", "tp1")):
                writer.add_timepoint(
                    timepoint,
                    index=index,
                    label=timepoint,
                    days_from_baseline=index * 90,
                )
                # the PET grid is *named* for the visit, colliding with it
                for modality, grid_id in (
                    ("ct", f"ct_{timepoint}"),
                    ("pet", timepoint),
                ):
                    writer.add_grid(
                        grid_id,
                        shape=shape,
                        spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0),
                        timepoint=timepoint,
                        frame_uid=f"pseudo:{modality}-{timepoint}",
                    )
                    writer.add_image(
                        f"{modality.upper()}_{timepoint}",
                        np.zeros(shape, np.int16),
                        grid=grid_id,
                        modality=modality.upper(),
                    )
            writer.add_transform(
                "ct_reg",
                kind="affine",
                from_frame="pseudo:ct-tp0",
                to_frame="pseudo:ct-tp1",
                matrix=shift,
            )

        with medh5.open(path) as sample:
            # the collision is real: the key names a grid on one frame, and
            # reads back as the visit's two
            assert sample.grids["tp0"].frame_uid == "pseudo:pet-tp0"
            assert len(sample._frames_for("tp0")) == 2

        def dataset(modality):
            return PairedPatchDataset(
                [str(path)],
                PatchSampler(8),
                align="transform",
                images=[f"{modality}_tp0", f"{modality}_tp1"],
            )

        patches = dataset("CT")[0]["meta"]["patches"]
        moved = patches["tp1"]["center"][0] - patches["tp0"]["center"][0]
        assert moved == 10, "the CT pair is still aligned by its own registration"

        # the PET grids are unregistered, and being named after the visits does
        # not lend them the CT registration
        with pytest.raises(MEDH5ValidationError, match="needs a transform"):
            dataset("PET")[0]

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

    def test_S3_7_a_visit_with_two_grids_samples_the_one_that_was_asked_for(
        self, tmp_path, label_set
    ):
        """A visit may hold a CT grid and a PET grid; neither is "the" grid.

        The window was pinned to whichever sorted first, so selecting the PET
        images drew in the CT grid and the pair was refused --- the guard from
        the previous commit firing on a case it should have sampled.  The grid
        comes from the data being read now.
        """
        shape = (12, 12, 12)
        path = tmp_path / "multi.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp in ("tp0", "tp1"):
                for prefix, modality in (("a_ct", "CT"), ("z_pet", "PT")):
                    w.add_grid(
                        f"{prefix}_{tp}",
                        shape=shape,
                        spacing=(1.0, 1.0, 1.0),
                        timepoint=tp,
                        frame_uid=f"F_{tp}",
                    )
                    w.add_image(
                        f"{prefix.upper()}_{tp}",
                        np.zeros(shape, dtype=np.int16),
                        grid=f"{prefix}_{tp}",
                        modality=modality,
                    )
                w.add_segmentation(
                    f"pet_ann_{tp}",
                    grid=f"z_pet_{tp}",
                    masks={1: block(shape, (2, 2, 2), 4)},
                )

        for prefix in ("Z_PET", "A_CT"):
            images = [f"{prefix}_tp0", f"{prefix}_tp1"]
            item = PairedPatchDataset(
                [path], PatchSampler(8), align="none", images=images
            )[0]
            grids = {tp: p["grid_id"] for tp, p in item["meta"]["patches"].items()}
            assert grids == {
                "tp0": f"{prefix.lower()}_tp0",
                "tp1": f"{prefix.lower()}_tp1",
            }
            assert tuple(item["images"]["tp0"][images[0]].shape) == (8, 8, 8)

        # Both modalities at once is still one window over two grids.
        with pytest.raises(MEDH5ValidationError, match="one grid"):
            PairedPatchDataset([path], PatchSampler(8), align="none")[0]

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
    def test_stacks_tensors_and_keeps_metadata(self, indexed_cohort):
        ds = PatchDataset(
            indexed_cohort, PatchSampler(8), annotations={"organs_tp0": [1, 2]}, seed=3
        )
        batch = collate([ds[0], ds[1]])
        assert tuple(batch["images"]["CT_tp0"].shape) == (2, 8, 8, 8)
        assert tuple(batch["label"]["organs_tp0"].shape) == (2, 2, 8, 8, 8)
        assert len(batch["meta"]["sample_id"]) == 2

    def test_a_shape_mismatch_names_the_key(self, indexed_cohort):
        small = PatchDataset(indexed_cohort, PatchSampler(8))[0]
        large = PatchDataset(indexed_cohort, PatchSampler(16))[0]
        with pytest.raises(MEDH5ValidationError, match="images.CT_tp0"):
            collate([small, large])

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            collate([])

    def test_stack_images_builds_a_channel_axis(self, indexed_cohort):
        ds = PatchDataset(indexed_cohort, PatchSampler(8))
        stacked = stack_images([ds[0], ds[1]], ["CT_tp0"])
        assert tuple(stacked.shape) == (2, 1, 8, 8, 8)
        with pytest.raises(MEDH5ValidationError, match="missing image"):
            stack_images([ds[0]], ["PET"])


class TestDataLoader:
    def test_a_real_dataloader_round_trips(self, indexed_cohort):
        ds = PatchDataset(
            indexed_cohort,
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

    def test_S14_4_a_soak_does_not_grow_the_handle_cache(self, indexed_cohort):
        """10 epochs must not leak handles or file descriptors."""
        CACHE.clear()
        ds = PatchDataset(indexed_cohort, PatchSampler(8), samples_per_volume=4)
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
        assert len(CACHE) <= len(indexed_cohort)


class TestHandleCache:
    def test_S14_4_handles_are_reused_within_a_process(self, indexed_cohort):
        cache = HandleCache(maxsize=4)
        first = cache.get(indexed_cohort[0])
        assert cache.get(indexed_cohort[0]) is first
        assert cache.opens == 1

    def test_the_lru_bound_is_honoured(self, indexed_cohort):
        cache = HandleCache(maxsize=2)
        for path in indexed_cohort:
            cache.get(path)
        assert len(cache) == 2
        cache.close_all()
        assert len(cache) == 0

    def test_S14_4_a_new_pid_abandons_inherited_handles(
        self, indexed_cohort, monkeypatch
    ):
        """A forked child must never touch the parent's HDF5 descriptors."""
        cache = HandleCache()
        cache.get(indexed_cohort[0])
        assert len(cache) == 1
        monkeypatch.setattr(os, "getpid", lambda: cache.owner_pid + 1)
        cache.get(indexed_cohort[1])
        assert len(cache) == 1, "the inherited handle must be dropped, not reused"

    def test_S14_4_close_all_abandons_handles_it_did_not_open(
        self, indexed_cohort, monkeypatch
    ):
        """The `atexit` hook runs in forked children too.

        A child that exits through normal interpreter shutdown would otherwise
        call into HDF5 to close descriptors belonging to its parent --- the one
        thing this module exists to prevent, and the one place the PID check was
        missing.  `worker_init_fn` covers only callers who pass it.
        """
        cache = HandleCache()
        sample = cache.get(indexed_cohort[0])
        monkeypatch.setattr(os, "getpid", lambda: cache.owner_pid + 1)

        cache.close_all()

        assert len(cache) == 0, "the child drops what it inherited"
        assert sample.identity.sample_id == "case0", "and leaves it open for the parent"

    def test_S14_4_building_a_plan_does_not_evict_the_shared_cache(
        self, indexed_cohort
    ):
        """`CACHE` is process-global; a constructor must not spend it on a scan.

        Walking a cohort through a 32-entry LRU to build the plan evicts what
        training was about to reuse and leaves the cache holding the tail of the
        cohort --- chosen by plan order, not by what is being read.  The
        metadata pass itself is unavoidable: `__len__` needs one read per file.
        """
        CACHE.clear()
        warm = open_cached(indexed_cohort[0])
        opens = CACHE.opens

        GridPatchDataset(indexed_cohort, 8)
        PairedPatchDataset(indexed_cohort, PatchSampler(8))

        assert len(CACHE) == 1, "the scan left the shared cache alone"
        assert CACHE.opens == opens, "and opened nothing through it"
        assert open_cached(indexed_cohort[0]) is warm

    def test_worker_init_clears_the_module_cache(self, indexed_cohort):
        open_cached(indexed_cohort[0])
        assert len(CACHE) >= 1
        worker_init_fn(0)
        assert len(CACHE) == 0

    def test_set_cache_size(self):
        from medh5.torch.handles import set_cache_size

        before = CACHE.maxsize
        set_cache_size(4)
        assert CACHE.maxsize == 4
        set_cache_size(before)

    def test_P11_one_open_per_file_per_epoch(self, tmp_path: Path):
        """`shuffle=True` re-opened 76 % of items' files at 100 files."""
        pytest.importorskip("torch")
        from medh5.sampling import PatchSampler
        from medh5.torch import FileGroupedSampler, PatchDataset
        from medh5.torch.handles import CACHE

        paths = []
        for i in range(40):
            path = tmp_path / f"c{i:02d}.medh5"
            with Framed.writer(path, shape=(8, 16, 16)):
                pass
            paths.append(path)
        dataset = PatchDataset(
            paths, PatchSampler(4, strategy="uniform"), images=["CT"],
            samples_per_volume=3, seed=0,
        )  # fmt: skip
        sampler = FileGroupedSampler(dataset, seed=0)
        orders = []
        for epoch in range(2):
            dataset.set_epoch(epoch)
            CACHE.close_all()
            CACHE.opens = 0
            order = list(sampler)
            for index in order:
                dataset[index]
            assert CACHE.opens == len(paths)
            assert sorted(order) == list(range(len(dataset)))
            orders.append(order)
        assert orders[0] != orders[1]
        assert len(sampler) == len(dataset)

    def test_P11_every_dataset_says_which_items_read_which_file(self, tmp_path: Path):
        pytest.importorskip("torch")
        from medh5.sampling import PatchSampler
        from medh5.torch import (
            FileGroupedSampler,
            GridPatchDataset,
            PatchDataset,
            VolumeDataset,
        )

        paths = []
        for i in range(3):
            path = tmp_path / f"f{i}.medh5"
            with Framed.writer(path):
                pass
            paths.append(path)
        volume = VolumeDataset(paths, images=["CT"])
        patches = PatchDataset(
            paths, PatchSampler(4), images=["CT"], samples_per_volume=2
        )
        grid = GridPatchDataset(paths, patch_size=(8, 8, 8), images=["CT"])
        assert [list(g) for g in volume.file_groups()] == [[0], [1], [2]]
        assert [list(g) for g in patches.file_groups()] == [[0, 1], [2, 3], [4, 5]]
        per_file = len(grid) // 3
        assert [len(g) for g in grid.file_groups()] == [per_file] * 3
        unshuffled = list(FileGroupedSampler(grid, shuffle=False))
        assert unshuffled == list(range(len(grid)))
        with pytest.raises(MEDH5ValidationError, match="file_groups"):
            FileGroupedSampler(list(range(4)))

    def test_P11_T10_spawned_persistent_workers_follow_the_grouped_order(
        self, tmp_path: Path
    ):
        """The spawn start method, explicitly, on every platform CI runs."""
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from medh5.sampling import PatchSampler
        from medh5.torch import FileGroupedSampler, PatchDataset, collate

        paths = []
        for i in range(2):
            path = tmp_path / f"w{i}.medh5"
            with Framed.writer(path, shape=(16, 32, 32)):
                pass
            paths.append(path)
        dataset = PatchDataset(
            paths, PatchSampler((8, 8, 8), strategy="uniform"), images=["CT"],
            samples_per_volume=4,
        )  # fmt: skip
        loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=FileGroupedSampler(dataset),
            num_workers=2,
            collate_fn=collate,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

        def epoch(n: int) -> list[tuple[str, tuple[int, ...]]]:
            dataset.set_epoch(n)
            seen = []
            for batch in loader:
                assert len(set(batch["meta"]["path"])) == 1  # one file per batch
                seen += [
                    (p, tuple(s))
                    for p, s in zip(
                        batch["meta"]["path"],
                        batch["meta"]["patch"]["start"],
                        strict=True,
                    )
                ]
            return seen

        first, second = epoch(0), epoch(1)
        assert len(first) == len(second) == len(dataset)
        assert sorted(first) != sorted(second)

    def test_P11_set_cache_size_is_public_and_shrinks_the_cache(self, tmp_path: Path):
        from medh5.torch import CACHE, set_cache_size
        from medh5.torch.handles import DEFAULT_MAXSIZE

        paths = []
        for i in range(4):
            path = tmp_path / f"s{i}.medh5"
            with Framed.writer(path):
                pass
            paths.append(path)
        try:
            CACHE.close_all()
            for path in paths:
                CACHE.get(path)
            assert len(CACHE) == 4
            set_cache_size(2)
            assert len(CACHE) == 2
            with pytest.raises(ValueError):
                set_cache_size(0)
        finally:
            set_cache_size(DEFAULT_MAXSIZE)
            CACHE.close_all()

    def test_Q16_a_leased_handle_outlives_eviction(self, tmp_path: Path):
        from medh5.torch.handles import HandleCache

        paths = []
        for i in range(3):
            path = tmp_path / f"h{i}.medh5"
            with Framed.writer(path):
                pass
            paths.append(path)
        cache = HandleCache(maxsize=1)
        with cache.lease(paths[0]) as held:
            cache.get(paths[1])
            cache.get(paths[2])
            assert held.is_open  # still open: it is in use
            assert held.images["CT"].read().shape == Framed.SHAPE
        assert not held.is_open  # released, then evicted and closed
        assert len(cache) == 1
        cache.close_all()

    def test_Q16_threads_share_the_cache_without_closing_each_others_files(
        self, tmp_path: Path
    ):
        from medh5.torch.handles import HandleCache

        paths = []
        for i in range(12):
            path = tmp_path / f"t{i}.medh5"
            with Framed.writer(path) as w:
                w.add_segmentation(
                    "seg",
                    grid="g",
                    masks={1: np.full(Framed.SHAPE, True)},
                    encoding="labelmap",
                )
            paths.append(path)
        cache = HandleCache(maxsize=2)
        failures: list[BaseException] = []

        def work(seed: int) -> None:
            rng = np.random.default_rng(seed)
            try:
                for _ in range(40):
                    path = paths[int(rng.integers(0, len(paths)))]
                    with cache.lease(path) as sample:
                        assert sample.annotations["seg"].dense([1])[0].all()
            except BaseException as exc:  # pragma: no cover - the failure itself
                failures.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        cache.close_all()
        assert not failures, failures[:1]


class TestW16Loaders:
    """F-16, F-17, F-18, F-24, L-34: the file's contracts reach the model."""

    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.mark.parametrize("encoding", Numbered.ENCODINGS)
    def test_F16_S7_7_the_ignore_region_reaches_the_item(
        self, tmp_path: Path, encoding: str
    ):
        """Every loader item dropped it; `layers` & co. then trained it as 0."""
        from medh5.torch import VolumeDataset

        path = tmp_path / f"{encoding}.medh5"
        region = Numbered.encoded(path, encoding)
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].kind == encoding
            np.testing.assert_array_equal(sample.ignore_region("seg"), region)
        for label_format in ("onehot", "labelmap"):
            item = VolumeDataset(
                [path],
                images=["CT"],
                annotations={"seg": [1, 2]},
                label_format=label_format,
            )[0]
            np.testing.assert_array_equal(item["ignore"]["seg"].numpy(), region)
            assert item["meta"]["annotated"]["seg"].tolist() == [True, True]
            assert item["valid"]["CT"].numpy().all()
            if label_format == "labelmap":
                label = item["label"]["seg"].numpy()
                assert set(np.unique(label[region]).tolist()) == {65535}
                assert set(np.unique(label[~region]).tolist()) == {0, 1, 2}

    @pytest.mark.parametrize("encoding", Numbered.ENCODINGS)
    def test_F16_padding_is_ignored_and_never_valid(
        self, tmp_path: Path, encoding: str
    ):
        """Padding was labelled 0 --- background --- in every format."""
        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset

        path = tmp_path / f"pad-{encoding}.medh5"
        Numbered.encoded(path, encoding)
        size = (10, 16, 16)  # larger than the grid on every axis
        item = PatchDataset(
            [path],
            PatchSampler(size, strategy="uniform"),
            images=["CT"],
            annotations={"seg": [1, 2]},
            label_format="labelmap",
        )[0]
        pads = item["meta"]["patch"]["pad"]
        outside = np.ones(size, bool)
        outside[
            tuple(
                slice(before, n - after)
                for (before, after), n in zip(pads, size, strict=True)
            )
        ] = False
        assert outside.any()
        assert item["ignore"]["seg"].numpy()[outside].all()
        assert not item["valid"]["CT"].numpy()[outside].any()
        assert (item["label"]["seg"].numpy()[outside] == 65535).all()
        assert item["valid"]["CT"].numpy()[~outside].all()

    def test_F16_S4_4_the_valid_mask_reaches_the_item(self, tmp_path: Path):
        from medh5.torch import VolumeDataset

        fov = np.zeros(Numbered.SHAPE, bool)
        fov[:, 2:10, 2:10] = True
        path = tmp_path / "fov.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_mask("fov", fov, grid="g")
            w.add_image(
                "CT",
                np.zeros(Numbered.SHAPE, np.int16),
                grid="g",
                modality="CT",
                valid_mask="fov",
            )
        item = VolumeDataset([path], images=["CT"])[0]
        np.testing.assert_array_equal(item["valid"]["CT"].numpy(), fov)

    def test_F16_S11_3_coverage_is_per_channel(self, tmp_path: Path):
        """A 0 in a class nobody looked for is not a negative."""
        from medh5.torch import VolumeDataset

        liver, spleen, _ = Numbered.two_classes()
        path = tmp_path / "partial.medh5"
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks={1: liver, 2: spleen}, annotated_classes=[1]
            )
        item = VolumeDataset([path], images=["CT"], annotations={"seg": [2, 1]})[0]
        assert item["meta"]["annotated"]["seg"].tolist() == [False, True]

    def test_F16_the_batch_collates(self, tmp_path: Path):
        from torch.utils.data import DataLoader

        from medh5.torch import VolumeDataset, collate

        paths = []
        for n in range(2):
            path = tmp_path / f"b{n}.medh5"
            Numbered.encoded(path, "layers")
            paths.append(path)
        dataset = VolumeDataset(paths, images=["CT"], annotations={"seg": [1, 2]})
        for collate_fn in (collate, None):
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))
            assert tuple(batch["ignore"]["seg"].shape) == (2, *Numbered.SHAPE)
            assert tuple(batch["valid"]["CT"].shape) == (2, *Numbered.SHAPE)
            assert tuple(batch["meta"]["annotated"]["seg"].shape) == (2, 2)

    def test_F16_the_monai_bridge_writes_65535_under_every_encoding(
        self, tmp_path: Path
    ):
        pytest.importorskip("monai")
        import medh5.monai as bridge

        for encoding in Numbered.ENCODINGS:
            path = tmp_path / f"monai-{encoding}.medh5"
            region = Numbered.encoded(path, encoding)
            with medh5.open(path) as sample:
                label = bridge.to_dict(sample, images=["CT"], annotations=["seg"])
                values = np.asarray(label["seg"])[region]
            assert set(np.unique(values).tolist()) == {65535}, encoding

    def test_F17_persistent_workers_see_the_epoch(self, tmp_path: Path):
        """The documented setup drew epoch 0's patches for the whole run."""
        from torch.utils.data import DataLoader

        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset, collate, worker_init_fn

        path = Numbered.plain(tmp_path / "big.medh5", shape=(32, 64, 64))
        dataset = PatchDataset(
            [path],
            PatchSampler((8, 8, 8), strategy="uniform"),
            images=["CT"],
            samples_per_volume=6,
        )
        loader = DataLoader(
            dataset,
            batch_size=6,
            num_workers=2,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

        def draw(epoch: int) -> list[tuple[int, ...]]:
            dataset.set_epoch(epoch)
            batch = next(iter(loader))
            return sorted(tuple(s) for s in batch["meta"]["patch"]["start"])

        first, second, again = draw(0), draw(1), draw(0)
        assert first != second
        assert first == again
        assert dataset.epoch == 0

    def test_F17_the_epoch_is_an_attribute_as_before(self, tmp_path: Path):
        from medh5.sampling import PatchSampler
        from medh5.torch import PatchDataset

        dataset = PatchDataset(
            [Numbered.plain(tmp_path / "e.medh5")], PatchSampler(4), images=["CT"]
        )
        dataset.epoch = 3
        assert dataset.epoch == 3

    def _visits(self, path: Path, *, frames: tuple[Any, Any], register: bool) -> Path:
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_timepoint("tp0", index=0, days_from_baseline=0)
            w.add_timepoint("tp1", index=1, days_from_baseline=90)
            for n, (origin, frame) in enumerate(
                zip(((0, 0, 0), (-40, 15, 7)), frames, strict=True)
            ):
                w.add_grid(
                    f"g{n}",
                    shape=(16, 32, 32),
                    spacing=(2.0, 1.0, 1.0),
                    origin=origin,
                    timepoint=f"tp{n}",
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT{n}",
                    np.zeros((16, 32, 32), np.int16),
                    grid=f"g{n}",
                    modality="CT",
                )
            if register:
                w.add_transform(
                    "t", kind="affine", from_frame="F0", to_frame="F1", matrix=np.eye(4)
                )
        return path

    def test_F18_S3_3_two_frameless_grids_are_not_registered(self, tmp_path: Path):
        """`None == None` let unregistered visits through as aligned."""
        from medh5.sampling import PatchSampler
        from medh5.torch import PairedPatchDataset

        path = self._visits(tmp_path / "nif.medh5", frames=(None, None), register=False)
        dataset = PairedPatchDataset(
            [path], PatchSampler(8, strategy="uniform"), align="transform"
        )
        with pytest.raises(MEDH5ValidationError, match="frame of reference"):
            dataset[0]
        loose = PairedPatchDataset([path], PatchSampler(8), align="none")
        assert loose[0]["meta"]["aligned"] == "none"

    def test_F18_registered_and_shared_frames_still_map(self, tmp_path: Path):
        from medh5.sampling import PatchSampler
        from medh5.torch import PairedPatchDataset

        for name, frames, register in (
            ("reg", ("F0", "F1"), True),
            ("shared", ("F", "F"), False),
        ):
            path = self._visits(
                tmp_path / f"{name}.medh5", frames=frames, register=register
            )
            item = PairedPatchDataset([path], PatchSampler(8), align="transform")[0]
            assert item["meta"]["aligned"] == "transform"
            assert set(item["valid"]) == {"tp0", "tp1"}

    def test_F24_class_weights_come_from_voxel_classes(self, tmp_path: Path):
        """A sample-level diagnosis became the heaviest segmentation class."""
        from medh5.dataset.stats import compute_stats

        path = tmp_path / "dx.medh5"
        liver = np.zeros(Numbered.SHAPE, bool)
        liver[:4, :8, :8] = True  # 256 voxels
        spleen = np.zeros(Numbered.SHAPE, bool)
        spleen[4:5, :8, :8] = True  # 64 voxels
        with Numbered.writer(path) as w:
            w.add_segmentation("organs", grid="g", masks={1: liver, 2: spleen})
            w.add_classification("dx", {3: 1.0})
        stats = compute_stats([path])
        assert set(stats.classes) == {1, 2}
        weights = stats.class_weights()
        assert set(weights) == {1, 2}
        assert weights[2] / weights[1] == pytest.approx(256 / 64)

    def test_F24_an_absent_class_gets_no_weight_and_a_warning(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        liver, _, _ = Numbered.two_classes()
        path = tmp_path / "absent.medh5"
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks={1: liver}, annotated_classes=[1, 2]
            )
        stats = compute_stats([path])
        assert stats.classes[2].examined_in == 1 and stats.classes[2].voxels == 0
        with pytest.warns(UserWarning, match=r"\[2\]"):
            assert set(stats.class_weights()) == {1}

    def test_F24_present_and_examined_count_samples(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        liver, _, _ = Numbered.two_classes()
        path = tmp_path / "long.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(Numbered.label_set())
            for n in range(2):
                w.add_timepoint(f"tp{n}", index=n)
                w.add_grid(
                    f"g{n}",
                    shape=Numbered.SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=f"tp{n}",
                )
                w.add_image(
                    f"CT{n}",
                    np.zeros(Numbered.SHAPE, np.int16),
                    grid=f"g{n}",
                    modality="CT",
                )
                w.add_segmentation(f"seg{n}", grid=f"g{n}", masks={1: liver})
        stats = compute_stats([path])
        assert (stats.classes[1].present_in, stats.classes[1].examined_in) == (1, 1)
        assert stats.classes[1].voxels == 2 * int(liver.sum())

    def test_L34_S4_4_moments_honour_the_valid_mask(self, tmp_path: Path):
        from medh5.dataset.stats import compute_stats

        shape = (8, 32, 32)
        image = np.full(shape, -3024, np.int16)
        yy, xx = np.mgrid[:32, :32]
        circle = (yy - 15.5) ** 2 + (xx - 15.5) ** 2 < 14**2
        image[:, circle] = 40
        path = tmp_path / "fov.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_mask("fov", np.broadcast_to(circle, shape).copy(), grid="g")
            w.add_image("CT", image, grid="g", modality="CT", valid_mask="fov")
            w.add_grid("small", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            w.add_mask("other", np.ones((2, 2, 2), bool), grid="small")
        stats = compute_stats([path])
        moments = stats.images["CT"]
        assert moments.mean == pytest.approx(40.0)
        assert moments.minimum == 40
        assert moments.count == int(circle.sum()) * shape[0]
        assert stats.total_voxels == int(np.prod(shape))
