"""Patch and pair sampling (spec §14.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
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
from tests.helpers import SHAPE, block, encode_attr, write_sample
from tests.kits import Flat, Framed


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

    @pytest.mark.parametrize("ndim", [33, 2**31, 2**63 - 1])
    def test_S14_3_an_axis_count_no_dataset_has_is_refused(self, ndim):
        """Refused before the size is broadcast: allocating ``ndim`` sizes
        first aborted the process (or panicked) rather than raising."""
        with pytest.raises(MEDH5ValidationError, match="at most 32 axes"):
            coerce_patch_size(4, ndim)
        assert len(coerce_patch_size(4, 32)) == 32

    def test_S14_3_a_window_never_runs_backwards(self):
        """A negative extent would give ``slice(3, -1)`` and read nothing."""
        with pytest.raises(MEDH5ValidationError, match="positive"):
            window_around((1, 1), (-4, -4), (4, 4))
        with pytest.raises(MEDH5ValidationError, match="non-negative"):
            window_around((1, 1), (4, 4), (-4, 4))
        with pytest.raises(MEDH5ValidationError, match="one entry per axis"):
            window_around((1,), (4, 4), (8, 8))
        with pytest.raises(MEDH5ValidationError, match="non-negative"):
            grid_patches((-5, 4, 4), 2)


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

    def test_unknown_strategy_and_weights_are_refused(self):
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(8, strategy="vibes")
        with pytest.raises(MEDH5ValidationError):
            PatchSampler(8, foreground_prob=1.5)

    @pytest.mark.parametrize(
        "weights",
        ["popularity", {1: -1.0}, {1: float("nan")}, {1: 1.0, 2: float("inf")}],
    )
    def test_S14_3_a_weighting_no_draw_could_apply_is_refused_when_configured(
        self, weights
    ):
        """An unknown name, or a weight below zero or not a number, was taken
        and failed at the first draw --- in a DataLoader worker, an epoch in
        --- or drew with probabilities nobody asked for."""
        with pytest.raises(MEDH5ValidationError, match="class_weights"):
            PatchSampler(8, strategy="foreground", class_weights=weights)
        PatchSampler(8, strategy="foreground", class_weights={1: 0.0, 2: 2.5})

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

    def test_S14_3_iter_patches_yields_what_draws_returns(self, indexed):
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(4)
            drawn = sampler.draws(sample, "organs_tp0", 12, 7)
            lazy = iter_patches(sample, sampler, annotation="organs_tp0", n=12, rng=7)
            assert list(lazy) == drawn
            numpy = list(
                iter_patches(sample, sampler, n=5, rng=np.random.default_rng(3))
            )
            assert numpy == sampler.draws(sample, None, 5, np.random.default_rng(3))

    def test_S14_3_iter_patches_draws_one_patch_per_next(self, tmp_path: Path):
        """``iter_patches`` was ``yield from draws(...)``: every patch was drawn
        before the first was yielded.  Each draw here reads its index row, so
        the engine's read count shows how many have been drawn."""
        from medh5 import _core

        path = Framed.many_classes(tmp_path / "many.medh5")
        sampler = PatchSampler(8, strategy="foreground")
        with medh5.open(path) as s:
            sampler.draw(s, "organs", 0)
            _core._read_stats()
            patches = iter_patches(s, sampler, annotation="organs", n=10_000, rng=0)
            assert sum(_core._read_stats().values()) == 0
            first = next(patches)
            one = sum(_core._read_stats().values())
            assert 1 <= one <= 2
            next(patches)
            assert sum(_core._read_stats().values()) <= 2
        assert first.used_index is True

    def test_S14_3_draws_pins_the_window_grid(self, tmp_path: Path, label_set, masks):
        """``draw`` took ``grid=`` and ``draws`` did not, so a run of draws
        could not be pinned to a visit."""
        path = write_sample(
            tmp_path / "two.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        with medh5.open(path) as sample:
            sampler = PatchSampler(4, strategy="uniform")
            for grid in ("ct_tp0", "ct_tp1"):
                patches = sampler.draws(sample, None, 6, 1, grid=grid)
                assert {p.grid_id for p in patches} == {grid}
                lazy = iter_patches(sample, sampler, n=6, rng=1, grid=grid)
                assert list(lazy) == patches

    def test_S14_3_a_misspelt_annotation_is_refused_before_the_first_patch(
        self, indexed
    ):
        """A generator checks nothing until its first ``next()``: a misspelt
        name surfaced inside the training loop, not where it was written."""
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(4)
            with pytest.raises(KeyError, match="organs_tpx"):
                iter_patches(sample, sampler, annotation="organs_tpx", n=3)
            with pytest.raises(KeyError, match="nowhere"):
                iter_patches(sample, sampler, n=3, grid="nowhere")
            assert list(iter_patches(sample, sampler, n=0)) == []

    @pytest.mark.parametrize("seed", [-1, 2**64, 2**200, [1, -2], [1, 2**64]])
    def test_S14_3_a_seed_is_a_64_bit_word_never_wrapped(self, indexed, seed):
        """``2**64`` wrapped onto ``0`` and ``-1`` onto ``2**64 - 1``, drawing
        another seed's patches; NumPy refuses a negative seed too."""
        with medh5.open(indexed) as sample:
            sampler = PatchSampler(4, strategy="uniform")
            with pytest.raises(ValueError, match=r"from 0 to 2\*\*64 - 1"):
                sampler.draw(sample, "organs_tp0", seed)
            last = sampler.draw(sample, "organs_tp0", 2**64 - 1)
            assert sampler.draw(sample, "organs_tp0", 2**64 - 1) == last

    def test_the_annotation_defaults_to_the_first_voxel_one(self, indexed):
        with medh5.open(indexed) as sample:
            assert PatchSampler(4).draw(sample) is not None

    def test_json_round_trip(self, indexed):
        with medh5.open(indexed) as sample:
            payload = PatchSampler(4).draw(sample, "organs_tp0").to_json()
        assert payload["grid_id"] == "ct_tp0", "the window names the grid it is in"
        assert set(payload) == {
            "start",
            "stop",
            "pad",
            "center",
            "grid_id",
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

    def test_S14_3_a_class_the_annotation_cannot_express_keeps_the_index(
        self, indexed, masks
    ):
        """A sampler is configured once for a cohort, and a class this file
        does not hold has no voxels in it (§6.2).  It counts zero, and the
        index answers for the classes the file holds --- the draw scanned the
        labels instead, for every file missing one of the cohort's classes."""
        sampler = PatchSampler(
            8, strategy="foreground", foreground_classes=["spleen", "vessel"]
        )
        rng = np.random.default_rng(8)
        with medh5.open(indexed) as sample:
            assert 4 not in sample.annotations["organs_tp0"].class_ids
            draws = [sampler.draw(sample, "organs_tp0", rng) for _ in range(10)]
        assert {(d.strategy, d.class_id, d.used_index) for d in draws} == {
            ("foreground", 2, True)
        }
        assert all(masks[2][d.center] for d in draws)

    def test_S14_3_an_index_caching_no_coordinates_is_refused_and_scanned_past(
        self, indexed, masks
    ):
        """``max_coords=0`` wrote an index with every class counted and no
        coordinate to draw, and every foreground draw from it failed."""
        with (
            pytest.raises(MEDH5ValidationError, match="max_coords"),
            medh5.amend(indexed) as w,
        ):
            w.build_index(max_coords=0)
        with h5py.File(indexed, "r+") as handle:
            entry = handle["index/organs_tp0"]
            entry.attrs["max_coords"] = 0
            for name in list(entry["fg_coords"]):
                del entry["fg_coords"][name]
                entry["fg_coords"].create_dataset(name, shape=(0, 3), dtype=np.int32)
        sampler = PatchSampler(8, strategy="foreground", foreground_classes=[2])
        with medh5.open(indexed) as sample:
            drawn = sampler.draw(sample, "organs_tp0", np.random.default_rng(0))
        assert (drawn.class_id, drawn.used_index) == (2, False)
        assert masks[2][drawn.center]

    def test_a_mask_is_never_the_foreground_annotation(self, tmp_path: Path):
        from medh5.sampling import PatchSampler

        path = tmp_path / "fov.medh5"
        with Flat.open_writer(path) as w:
            w.add_mask("fov", np.ones(Flat.SHAPE, dtype=bool), grid="g")
        with medh5.open(path) as sample:
            sampler = PatchSampler(4, strategy="foreground")
            assert sampler._annotation(sample, None) is None
            patch = sampler.draw(sample, None, np.random.default_rng(0))
            assert patch.strategy == "uniform"
            assert patch.used_index is None

    def test_S14_3_an_annotation_named_that_is_not_there_is_refused(
        self, indexed, tmp_path: Path
    ):
        """A misspelt name drew `uniform` windows without a word, and
        `balanced` failed only on the draws whose coin chose foreground; boxes
        named for a foreground draw did the same."""
        with medh5.open(indexed) as sample:
            for strategy in ("uniform", "balanced", "foreground"):
                with pytest.raises(KeyError, match="organs_tpX"):
                    PatchSampler(4, strategy=strategy).draw(sample, "organs_tpX", 0)
        path = tmp_path / "boxes.medh5"
        with Flat.open_writer(path) as w:
            w.add_boxes("boxes", np.array([[[0, 2]] * 3], np.float32), [1], grid="g")
        with medh5.open(path) as sample:
            for strategy in ("balanced", "foreground"):
                with pytest.raises(MEDH5ValidationError, match="voxel annotation"):
                    PatchSampler(4, strategy=strategy).draw(sample, "boxes", 0)
            # No foreground is ever drawn: the window is measured in its grid.
            for sampler in (
                PatchSampler(4, strategy="uniform"),
                PatchSampler(4, strategy="balanced", foreground_prob=0.0),
            ):
                assert sampler.draw(sample, "boxes", 0).strategy == "uniform"

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("strategy", "vibes"),
            ("foreground_prob", 1.5),
            ("class_weights", "popularity"),
        ],
    )
    def test_S14_3_a_setting_changed_later_is_checked_as_the_constructor_checks_it(
        self, indexed, name, value
    ):
        """Set after construction, a bad setting was taken and refused at the
        next draw --- in a DataLoader worker, an epoch in."""
        sampler = PatchSampler(4, strategy="foreground")
        before = getattr(sampler, name)
        with pytest.raises(MEDH5ValidationError):
            setattr(sampler, name, value)
        assert getattr(sampler, name) == before
        with medh5.open(indexed) as sample:
            assert sampler.draw(sample, "organs_tp0", 0).strategy == "foreground"
            sampler.strategy = "uniform"
            assert sampler.draw(sample, "organs_tp0", 0).strategy == "uniform"

    def test_S14_3_an_indexed_draw_reads_at_most_two_datasets_at_63_classes(
        self, tmp_path: Path
    ):
        """190 HDF5 reads and 7.6 ms per draw: the class table, re-read per class.

        The reads are the engine's, so the engine counts them: a count of
        `h5py` calls, which the package no longer makes, could not fail."""
        from medh5 import _core
        from medh5.sampling import PatchSampler

        path = Framed.many_classes(tmp_path / "many.medh5")
        sampler = PatchSampler(8, strategy="foreground")
        draws = 20
        with medh5.open(path) as s:
            sampler.draw(s, "organs", np.random.default_rng(0))
            _core._read_stats()
            for i in range(draws):
                patch = sampler.draw(s, "organs", np.random.default_rng(i))
            reads = _core._read_stats()
        assert patch.used_index is True
        # One row of a class's pool per draw, at least: the count sees them.
        assert draws <= sum(reads.values()) <= 2 * draws, reads

    def test_S14_3_a_draw_picks_what_it_picked_before(self, tmp_path: Path):
        """One row is read instead of the pool, and the same row: runs reproduce.

        A seed --- an int, or a NumPy generator in a given state --- picks the
        same row every time; the row is the pool's.  (1.x picked the row NumPy
        would; 2.0 draws from the engine's generator, so the rows differ.)"""
        path = Framed.many_classes(tmp_path / "many.medh5", classes=4)
        with medh5.open(path) as s:
            index = s.index["organs"]
            for class_id in index.class_ids:
                pool = index.coords(class_id).tolist()
                picked = set()
                for seed in range(5):
                    row = index.sample_foreground(class_id, 1, seed).tolist()
                    assert row == index.sample_foreground(class_id, 1, seed).tolist()
                    assert row[0] in pool
                    picked.add(tuple(row[0]))
                    rng = np.random.default_rng(seed)
                    again = np.random.default_rng(seed)
                    assert (
                        index.sample_foreground(class_id, 1, rng).tolist()
                        == index.sample_foreground(class_id, 1, again).tolist()
                    )
                assert len(pool) == 1 or len(picked) > 1
            assert index.has_class(1) and not index.has_class(99)
            assert index.voxel_counts == dict(index.voxel_counts)


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


class TestUniformIsUniform:
    def test_S14_3_window_placement_is_uniform_not_just_the_centre(self, tmp_path):
        """Drawing a centre over every voxel and clamping is not uniform.

        Every centre in the leading half-patch collapses onto window start 0 and
        every centre in the trailing half onto the last start, so on a 24-voxel
        axis with an 8-voxel patch the first window took 3.6x its share and the
        last 2.8x -- border-heavy training data from a strategy named `uniform`.
        The skew grows with patch size.
        """
        shape = (24, 24, 24)
        path = tmp_path / "uniform.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")

        sampler = PatchSampler((8, 8, 8), strategy="uniform")
        counts: dict[int, int] = {}
        draws = 20_000
        with medh5.open(path) as sample:
            rng = np.random.default_rng(0)
            for _ in range(draws):
                start = sampler.draw(sample, rng=rng).slices[0].start
                counts[start] = counts.get(start, 0) + 1

        starts = sorted(counts)
        assert starts == list(range(0, 24 - 8 + 1)), "every valid start must occur"
        expected = draws / len(starts)
        worst = max(abs(counts[s] - expected) for s in starts) / expected
        assert worst < 0.25, f"window placement is {worst:.1%} off uniform"

    def test_a_patch_larger_than_the_volume_still_pads(self, tmp_path):
        """The oversized-patch path is unchanged: whole axis plus padding."""
        shape = (24, 24, 24)
        path = tmp_path / "small.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        with medh5.open(path) as sample:
            patch = PatchSampler((64, 64, 64), strategy="uniform").draw(
                sample, rng=np.random.default_rng(0)
            )
        assert all(s == slice(0, 24) for s in patch.slices)
        assert all(pad == (20, 20) for pad in patch.pad)


class TestWeightedPicks:
    """Every foreground draw picks its class by ``class_weights`` from every
    candidate's voxel count, with an index or without.  The scan that stands
    in for a missing or stale index drew classes uniformly, so
    ``inverse_frequency`` meant nothing there and a class weighted zero was
    drawn; and with an index, an empty weighted pool fell back to that scan."""

    DRAWS = 300
    # `masks`: 512, 216 and 27 voxels.
    COUNTS = {1: 512, 2: 216, 3: 27}

    @staticmethod
    def _variant(path: Path, which: str) -> Path:
        """The indexed sample as written, without its index, or with it stale."""
        if which != "fresh":
            with h5py.File(path, "r+") as handle:
                if which == "absent":
                    del handle["index"]
                else:
                    handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                        "sha256:" + "0" * 64
                    )
        return path

    def _draws(self, path: Path, weights: Any) -> list[Patch]:
        sampler = PatchSampler(4, strategy="foreground", class_weights=weights)
        rng = np.random.default_rng(11)
        with medh5.open(path) as sample:
            return [sampler.draw(sample, "organs_tp0", rng) for _ in range(self.DRAWS)]

    @pytest.mark.parametrize("which", ["absent", "fresh", "stale"])
    def test_S14_3_a_class_weighted_zero_is_never_drawn(self, indexed, which):
        draws = self._draws(self._variant(indexed, which), {1: 0.0, 2: 1.0, 3: 1.0})
        assert {d.class_id for d in draws} == {2, 3}
        assert {d.used_index for d in draws} == {which == "fresh"}

    @pytest.mark.parametrize("which", ["absent", "fresh", "stale"])
    @pytest.mark.parametrize("policy", ["frequency", "inverse_frequency"])
    def test_S14_3_the_policies_hold_with_or_without_an_index(
        self, indexed, which, policy
    ):
        draws = self._draws(self._variant(indexed, which), policy)
        weights = {
            c: float(n) if policy == "frequency" else 1.0 / n
            for c, n in self.COUNTS.items()
        }
        total = sum(weights.values())
        for class_id, weight in weights.items():
            seen = sum(d.class_id == class_id for d in draws) / self.DRAWS
            assert seen == pytest.approx(weight / total, abs=0.08), class_id

    @pytest.mark.parametrize("which", ["absent", "fresh", "stale"])
    def test_S14_3_an_empty_weighted_pool_draws_no_foreground(self, indexed, which):
        """No candidate with foreground and weight: a uniform window, not a
        foreground one picked without the weights."""
        draws = self._draws(self._variant(indexed, which), {1: 0.0, 2: 0.0})
        assert {(d.strategy, d.class_id) for d in draws} == {("uniform", None)}


class TestPatchAggregator:
    """Patch predictions stitched back into one volume."""

    @staticmethod
    def _volume(shape: tuple[int, ...] = (9, 14, 11)) -> Any:
        rng = np.random.default_rng(3)
        return rng.integers(-50, 50, shape).astype(np.float32)

    def _stitch(self, volume: Any, patch: Any, overlap: int, **kwargs: Any) -> Any:
        from medh5.sampling import PatchAggregator

        aggregator = PatchAggregator(volume.shape, grid_id="g", **kwargs)
        patches = grid_patches(volume.shape, patch, overlap=overlap, grid_id="g")
        for p in patches:
            aggregator.add(p.apply_padding(volume[p.slices]), p)
        assert aggregator.count == len(patches)
        assert aggregator.coverage.all()
        return aggregator.result()

    def test_constant_blend_reassembles_the_volume_exactly(self):
        volume = self._volume()
        for patch, overlap in (((4, 6, 5), 0), ((4, 6, 5), 2), ((5, 5, 5), 3)):
            out = self._stitch(volume, patch, overlap, mode="constant")
            assert np.array_equal(out, volume), (patch, overlap)

    def test_gaussian_blend_of_a_constant_is_that_constant(self):
        volume = np.full((9, 14, 11), 7.0, dtype=np.float32)
        out = self._stitch(volume, (4, 6, 5), 2)
        assert np.allclose(out, 7.0, rtol=1e-6)

    def test_padding_is_cut_off(self):
        """A patch larger than the volume is padded; the padding is not
        stitched in."""
        from medh5.sampling import PatchAggregator

        volume = self._volume((3, 4, 5))
        (patch,) = grid_patches(volume.shape, (6, 6, 6), grid_id="g")
        assert patch.needs_padding
        aggregator = PatchAggregator(volume.shape, channels=2, mode="constant")
        padded = patch.apply_padding(volume[patch.slices], value=1e9)
        aggregator.add(np.stack([padded, -padded]), patch)
        out = aggregator.result()
        assert out.shape == (2, 3, 4, 5)
        assert np.array_equal(out[0], volume) and np.array_equal(out[1], -volume)
        with pytest.raises(MEDH5ValidationError, match=r"shaped \(2, 6, 6, 6\)"):
            aggregator.add(padded, patch)

    def test_weights_are_never_zero(self):
        from medh5.sampling import BLEND_MODES, blend_weights

        assert BLEND_MODES == ("constant", "gaussian")
        weights = blend_weights((32, 32, 32), sigma_scale=0.01)
        assert weights.dtype == np.float32
        assert weights.max() == 1.0
        assert weights.min() >= np.float32(1e-6) * weights.max()
        assert (blend_weights((3, 4), mode="constant") == 1.0).all()
        with pytest.raises(MEDH5ValidationError, match="blend mode"):
            blend_weights((3, 4), mode="linear")
        with pytest.raises(MEDH5ValidationError, match="sigma_scale"):
            blend_weights((3, 4), sigma_scale=0.0)

    def test_an_uncovered_voxel_is_said_so(self):
        from medh5.sampling import PatchAggregator

        aggregator = PatchAggregator((4, 4))
        aggregator.add(np.ones((2, 2)), Patch((slice(0, 2), slice(0, 2))))
        out = aggregator.result()
        assert np.isnan(out[3, 3]) and out[0, 0] == 1.0
        assert aggregator.result(uncovered=-1.0)[3, 3] == -1.0
        with pytest.raises(MEDH5ValidationError, match="outside the volume"):
            aggregator.add(np.ones((2, 2)), Patch((slice(3, 5), slice(0, 2))))

    def test_another_grid_is_refused(self):
        from medh5.sampling import PatchAggregator

        aggregator = PatchAggregator((4, 4))
        aggregator.add(np.ones((2, 2)), Patch((slice(0, 2), slice(0, 2)), grid_id="a"))
        assert aggregator.grid_id == "a"
        with pytest.raises(MEDH5ValidationError, match="on grid 'b'"):
            aggregator.add(
                np.ones((2, 2)), Patch((slice(2, 4), slice(0, 2)), grid_id="b")
            )
        assert aggregator.count == 1
