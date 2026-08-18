"""Patch and pair sampling (spec §14.3, plan §2.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.sampling import (
    Patch,
    PatchSampler,
    TimepointPairSampler,
    coerce_patch_size,
    grid_patches,
    iter_patches,
    window_around,
)
from tests.v1.conftest import SHAPE, block, write_sample


@pytest.fixture
def indexed(tmp_path: Path, label_set, masks) -> Path:
    return write_sample(
        tmp_path / "idx.medh5", label_set=label_set, masks=masks, index=True
    )


@pytest.fixture
def unindexed(tmp_path: Path, label_set, masks) -> Path:
    return write_sample(tmp_path / "raw.medh5", label_set=label_set, masks=masks)


class TestWindow:
    def test_a_centre_near_the_edge_shifts_inwards(self):
        slices, pad = window_around((0, 0, 0), (8, 8, 8), (16, 16, 16))
        assert [s.start for s in slices] == [0, 0, 0]
        assert [s.stop for s in slices] == [8, 8, 8]
        assert pad == ((0, 0),) * 3

        slices, _ = window_around((15, 15, 15), (8, 8, 8), (16, 16, 16))
        assert [s.stop for s in slices] == [16, 16, 16]

    def test_a_patch_larger_than_the_volume_pads_symmetrically(self):
        slices, pad = window_around((4,), (10,), (4,))
        assert slices == (slice(0, 4),)
        assert pad == ((3, 3),)

    def test_the_patch_keeps_its_requested_size(self):
        patch = Patch(*window_around((2, 2), (6, 6), (4, 10)))
        assert patch.shape == (6, 6)
        assert patch.needs_padding
        padded = patch.apply_padding(np.ones((4, 6)))
        assert padded.shape == (6, 6)
        assert padded[0, 0] == 0.0

    def test_padding_leaves_leading_axes_alone(self):
        patch = Patch(*window_around((2,), (6,), (4,)))
        assert patch.apply_padding(np.ones((3, 4))).shape == (3, 6)

    def test_patch_size_is_broadcast_and_checked(self):
        assert coerce_patch_size(8, 3) == (8, 8, 8)
        assert coerce_patch_size((4, 8), 2) == (4, 8)
        with pytest.raises(MEDH5ValidationError):
            coerce_patch_size((4, 8), 3)
        with pytest.raises(MEDH5ValidationError):
            coerce_patch_size(0, 2)


class TestPatchSampler:
    def test_S14_3_foreground_centres_land_in_the_class(self, indexed, masks):
        sampler = PatchSampler(8, strategy="foreground", foreground_classes=["spleen"])
        rng = np.random.default_rng(7)
        with medh5.open(indexed) as sample:
            for _ in range(20):
                patch = sampler.draw(sample, "organs_tp0", rng)
                assert patch.strategy == "foreground"
                assert patch.used_index
                assert masks[2][patch.center]

    def test_S14_3_the_index_is_used_when_present(self, indexed, unindexed):
        sampler = PatchSampler(8, strategy="foreground")
        rng = np.random.default_rng(0)
        with medh5.open(indexed) as sample:
            assert sampler.draw(sample, "organs_tp0", rng).used_index
        with medh5.open(unindexed) as sample:
            drawn = sampler.draw(sample, "organs_tp0", rng)
            assert drawn.strategy == "foreground"
            assert not drawn.used_index, (
                "an unindexed file must say it fell back to scanning"
            )

    def test_uniform_draws_cover_the_volume(self, indexed):
        sampler = PatchSampler(4, strategy="uniform")
        rng = np.random.default_rng(1)
        with medh5.open(indexed) as sample:
            centers = {
                sampler.draw(sample, "organs_tp0", rng).center for _ in range(40)
            }
        assert len(centers) > 20

    def test_balanced_mixes_both(self, indexed):
        rng = np.random.default_rng(2)
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(8, strategy="balanced", foreground_prob=0.5)
            kinds = {
                sampler.draw(sample, "organs_tp0", rng).strategy for _ in range(40)
            }
        assert kinds == {"foreground", "uniform"}

    def test_foreground_prob_zero_never_uses_the_index(self, indexed):
        rng = np.random.default_rng(3)
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(8, strategy="balanced", foreground_prob=0.0)
            kinds = {
                sampler.draw(sample, "organs_tp0", rng).strategy for _ in range(20)
            }
        assert kinds == {"uniform"}

    def test_class_weights_bias_the_draw(self, indexed):
        """Inverse-frequency sampling must favour the rarer class."""
        rng = np.random.default_rng(4)
        with medh5.open(indexed) as sample:
            counts = sample.index["organs_tp0"].voxel_counts
            rare = min(counts, key=lambda c: counts[c])
            sampler = PatchSampler(
                4, strategy="foreground", class_weights="inverse_frequency"
            )
            picks = [
                sampler.draw(sample, "organs_tp0", rng).class_id for _ in range(60)
            ]
            frequency = PatchSampler(
                4, strategy="foreground", class_weights="frequency"
            )
            common = [
                frequency.draw(sample, "organs_tp0", rng).class_id for _ in range(60)
            ]
        assert picks.count(rare) > common.count(rare)

    def test_explicit_weights_select_one_class(self, indexed):
        rng = np.random.default_rng(5)
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(4, strategy="foreground", class_weights={2: 1.0})
            assert {
                sampler.draw(sample, "organs_tp0", rng).class_id for _ in range(10)
            } == {2}

    def test_unknown_strategy_and_weights_are_refused(self, indexed):
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(8, strategy="vibes")
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(8, foreground_prob=1.5)
        with medh5.open(indexed) as sample, pytest.raises(MEDH5ValidationError):
            PatchSampler(8, strategy="foreground", class_weights="popularity").draw(
                sample, "organs_tp0", np.random.default_rng(0)
            )

    def test_an_empty_class_falls_back_to_uniform(self, tmp_path, label_set):
        path = tmp_path / "empty.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "organs",
                grid="g",
                masks={1: block(SHAPE, (2, 2, 2), 4), 4: np.zeros(SHAPE, dtype=bool)},
            )
        with medh5.open(path) as sample:
            sampler = PatchSampler(4, strategy="foreground", foreground_classes=[4])
            assert sampler.draw(
                sample, "organs", np.random.default_rng(0)
            ).strategy == ("uniform")

    def test_draws_and_iter_patches(self, indexed):
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(4, strategy="uniform")
            assert len(sampler.draws(sample, "organs_tp0", 5)) == 5
            assert len(list(iter_patches(sample, sampler, n=3))) == 3
            assert "uniform" in repr(sampler)

    def test_the_annotation_defaults_to_the_first_voxel_one(self, indexed):
        with medh5.open(indexed) as sample:
            assert PatchSampler(4).draw(sample) is not None

    def test_json_round_trip(self, indexed):
        with medh5.open(indexed) as sample:
            payload = PatchSampler(4).draw(sample, "organs_tp0").to_json()
        assert set(payload) == {
            "start",
            "stop",
            "pad",
            "center",
            "strategy",
            "class_id",
            "used_index",
        }

    def test_S14_3_a_stale_index_is_not_trusted_for_foreground(self, indexed):
        """§13.3: a stale index is ignored and rebuilt, never read.

        Its coordinates describe the annotation as it was before somebody
        edited it, so a centre drawn from one is silently not foreground --- the
        training distribution shifts and the file says nothing about it.
        """
        import h5py

        from medh5._hdf5 import encode_attr

        sampler = PatchSampler(8, strategy="foreground")
        with medh5.open(indexed) as sample:
            assert sampler.draw(
                sample, "organs_tp0", np.random.default_rng(0)
            ).used_index

        with h5py.File(indexed, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )

        with medh5.open(indexed) as sample:
            drawn = sampler.draw(sample, "organs_tp0", np.random.default_rng(0))
            assert not drawn.used_index, "a stale index is scanned past, not read"
            mask = sample.annotations["organs_tp0"].dense([drawn.class_id])[0]
            assert mask[drawn.center], "and the centre it fell back to is foreground"


class TestGridPatches:
    def test_the_cover_is_complete(self):
        patches = grid_patches((10, 10), 4)
        covered = np.zeros((10, 10), dtype=bool)
        for patch in patches:
            covered[patch.slices] = True
        assert covered.all()

    def test_the_last_window_shifts_in_rather_than_padding(self):
        patches = grid_patches((10,), 4)
        assert [p.slices[0].start for p in patches] == [0, 4, 6]
        assert not any(p.needs_padding for p in patches)

    def test_overlap_produces_more_windows(self):
        assert len(grid_patches((16,), 4, overlap=2)) > len(grid_patches((16,), 4))

    def test_a_patch_larger_than_the_volume_is_one_padded_window(self):
        patches = grid_patches((3,), 8)
        assert len(patches) == 1
        assert patches[0].needs_padding

    def test_bad_overlap_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            grid_patches((16,), 4, overlap=4)
        with pytest.raises(MEDH5ValidationError):
            grid_patches((16,), 4, overlap=-1)


class TestPairSampler:
    def test_modes_enumerate_the_right_pairs(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "three.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1", "tp2"),
        )
        with medh5.open(path) as sample:
            consecutive = TimepointPairSampler("consecutive").pairs(sample)
            assert [(p.first, p.second) for p in consecutive] == [
                ("tp0", "tp1"),
                ("tp1", "tp2"),
            ]
            baseline = TimepointPairSampler("baseline_vs_all")(sample)
            assert [(p.first, p.second) for p in baseline] == [
                ("tp0", "tp1"),
                ("tp0", "tp2"),
            ]
            assert len(TimepointPairSampler("all_pairs").pairs(sample)) == 3

    def test_intervals_come_from_the_timeline(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            pair = TimepointPairSampler().pairs(sample)[0]
            assert pair.interval_days == 90
            assert "tp0 -> tp1" in repr(pair)

    def test_a_cross_sectional_sample_yields_no_pairs(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert TimepointPairSampler().pairs(sample) == []

    def test_unknown_mode_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            TimepointPairSampler("every-other-tuesday")

    def test_S9_a_reversed_change_label_does_not_match_the_forward_pair(
        self, tmp_path, label_set, masks
    ):
        """Timepoint order is what tells a change label from its opposite.

        "grew 40 %" and "shrank 40 %" span the same two visits and differ only
        in which is the baseline, so a label written ``(tp1, tp0)`` must not be
        hung on the forward pair ``(tp0, tp1)`` --- that trains the example with
        the label for the comparison nobody made.
        """
        path = write_sample(
            tmp_path / "reversed.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        with medh5.amend(path) as w:
            w.add_classification(
                "regression",
                labels={3: 1.0},
                scope="sample",
                timepoints=["tp1", "tp0"],
            )
        with medh5.open(path) as sample:
            assert sample.annotations["regression"].timepoints == ("tp1", "tp0")
            assert TimepointPairSampler().pairs(sample)[0].label is None

    def test_a_change_label_is_attached_to_its_pair(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "change.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        with medh5.amend(path) as w:
            w.add_classification(
                "response",
                labels={3: 1.0},
                scope="sample",
                timepoints=["tp0", "tp1"],
            )
        with medh5.open(path) as sample:
            assert TimepointPairSampler().pairs(sample)[0].label == "response"
