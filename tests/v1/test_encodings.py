"""Voxel encodings: the uniform contract and lossless transcoding (spec §7)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from medh5.annotations.voxel import (
    InstanceInput,
    analyse,
    cost_model,
    encode_bitmask,
    encode_instances,
    encode_labelmap,
    encode_layers,
    encode_mask,
    encode_masks,
    encode_probmap,
    greedy_colour,
    payload_to_masks,
    select_encoding,
    transcode_payload,
)
from medh5.annotations.voxel.select import layers_from_colouring
from medh5.annotations.voxel.transcode import TRANSCODABLE, masks_equal
from medh5.errors import MEDH5ValidationError

SHAPE = (12, 16, 16)
DENSE = ("labelmap", "layers", "bitmask", "instances", "probmap")


def random_masks(seed: int, n_classes: int = 6, overlap: bool = True):
    rng = np.random.default_rng(seed)
    masks = {}
    for i in range(n_classes):
        mask = np.zeros(SHAPE, dtype=bool)
        origin = rng.integers(0, 6, 3) if overlap else np.array([0, 0, i * 2])
        size = rng.integers(2, 5, 3) if overlap else np.array([3, 3, 2])
        mask[tuple(slice(o, o + s) for o, s in zip(origin, size, strict=True))] = True
        masks[i + 1] = mask
    return masks


def disjoint_masks():
    masks = {}
    for i in range(4):
        mask = np.zeros(SHAPE, dtype=bool)
        mask[:, :, i * 4 : i * 4 + 3] = True
        masks[i + 1] = mask
    return masks


class TestSelection:
    def test_S7_6_edgeless_graph_selects_labelmap(self):
        kind, stats = select_encoding(disjoint_masks(), SHAPE)
        assert stats.is_edgeless
        assert kind == "labelmap"

    def test_S7_6_overlap_selects_layers(self):
        masks = random_masks(3)
        kind, stats = select_encoding(masks, SHAPE)
        assert stats.edges
        assert kind == "layers"

    def test_S7_6_sparse_localized_selects_instances(self):
        masks = {}
        for i in range(3):
            mask = np.zeros((64, 64, 64), dtype=bool)
            mask[i, i, i] = True
            masks[i + 1] = mask
        kind, stats = select_encoding(masks, (64, 64, 64))
        assert stats.fill < 1e-3
        assert kind == "instances"

    def test_S7_6_soft_values_select_probmap(self):
        kind, _ = select_encoding(disjoint_masks(), SHAPE, soft=True)
        assert kind == "probmap"

    def test_explicit_preference_wins(self):
        kind, _ = select_encoding(disjoint_masks(), SHAPE, prefer="bitmask")
        assert kind == "bitmask"

    def test_S7_3_layers_bitmask_crossover(self):
        """§7.3: uint16 layers beat bitmask while L < 4*ceil(C/64)."""
        stats = analyse(random_masks(5, n_classes=6), SHAPE)
        costs = cost_model(stats)
        assert costs.detail["crossover_layers"] == 8 * stats.n_planes
        assert (costs.layers < costs.bitmask) == (
            stats.n_layers < costs.detail["crossover_layers"]
        )

    def test_greedy_colouring_is_a_valid_colouring(self):
        masks = random_masks(11, n_classes=10)
        stats = analyse(masks, SHAPE)
        for a, b in stats.edges:
            assert stats.colouring[a] != stats.colouring[b]

    def test_colouring_of_a_clique_needs_one_colour_each(self):
        ids = [1, 2, 3, 4]
        edges = set(itertools.combinations(ids, 2))
        colouring = greedy_colour(ids, edges)
        assert len(set(colouring.values())) == 4

    def test_layers_from_colouring_groups_ascending(self):
        assert layers_from_colouring({1: 0, 2: 1, 3: 0}) == ((1, 3), (2,))
        assert layers_from_colouring({}) == ()


class TestEncoders:
    def test_S7_1_labelmap_rejects_overlap(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_labelmap(random_masks(2), SHAPE)
        assert exc.value.code == "E404"

    def test_S7_1_dtype_narrows_when_ids_fit(self):
        assert encode_labelmap(disjoint_masks(), SHAPE).data.dtype == np.uint8
        wide = {300: np.ones(SHAPE, dtype=bool)}
        assert encode_labelmap(wide, SHAPE).data.dtype == np.uint16

    def test_S7_2_every_class_lands_in_exactly_one_layer(self):
        payload = encode_layers(random_masks(7, n_classes=8), SHAPE)
        table = payload.datasets["layer_class_ids"]
        seen = [int(v) for v in table.reshape(-1) if int(v)]
        assert sorted(seen) == sorted(payload.class_ids)
        assert len(seen) == len(set(seen))

    def test_S7_2_invalid_colouring_is_refused(self):
        overlapping = {
            1: np.zeros(SHAPE, dtype=bool),
            2: np.zeros(SHAPE, dtype=bool),
        }
        overlapping[1][0:4, 0:4, 0:4] = True
        overlapping[2][2:6, 2:6, 2:6] = True
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_layers(overlapping, SHAPE, colouring=dict.fromkeys(overlapping, 0))
        assert exc.value.code == "E404"

    def test_S7_2_incomplete_colouring_is_refused(self):
        masks = random_masks(2, n_classes=3)
        with pytest.raises(MEDH5ValidationError):
            encode_layers(masks, SHAPE, colouring={1: 0})

    def test_S7_3_bitplane_count(self):
        masks = {i: np.zeros(SHAPE, dtype=bool) for i in range(1, 70)}
        payload = encode_bitmask(masks, SHAPE)
        assert payload.data.shape[0] == 2
        assert payload.data.dtype == np.uint64

    def test_S7_4_instances_need_a_mask_or_a_box(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_instances([InstanceInput(class_id=1, instance_id=1)], SHAPE)
        assert exc.value.code == "E410"

    def test_S7_4_empty_mask_is_refused(self):
        obj = InstanceInput(class_id=1, instance_id=1, mask=np.zeros(SHAPE, dtype=bool))
        with pytest.raises(MEDH5ValidationError):
            encode_instances([obj], SHAPE)

    def test_S7_4_offsets_are_monotonic(self):
        masks = random_masks(4, n_classes=3)
        objects = [
            InstanceInput(class_id=c, instance_id=c, mask=m) for c, m in masks.items()
        ]
        payload = encode_instances(objects, SHAPE)
        offsets = payload.datasets["mask_offsets"]
        assert np.all(np.diff(offsets.astype(np.int64)) > 0)
        assert offsets[-1] == payload.datasets["mask_data"].size

    def test_S7_5_probabilities_must_be_in_range(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_probmap({1: np.full(SHAPE, 1.5)}, SHAPE)
        assert exc.value.code == "E411"

    def test_mask_encoding_carries_no_classes(self):
        payload = encode_mask(np.ones(SHAPE, dtype=bool))
        assert payload.class_ids == ()

    def test_shape_disagreement_is_reported(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_bitmask({1: np.zeros((2, 2, 2), dtype=bool)}, SHAPE)
        assert exc.value.code == "E405"


class TestTranscoding:
    @pytest.mark.parametrize("target", [k for k in TRANSCODABLE if k != "labelmap"])
    def test_S7_6_every_pair_round_trips(self, target):
        """§7.6: transcoding MUST preserve contains() for every class and voxel."""
        masks = random_masks(13, n_classes=5)
        original = encode_masks(masks, "layers", SHAPE)
        converted = transcode_payload(original, target, spatial_shape=SHAPE)
        decoded = payload_to_masks(converted, spatial_shape=SHAPE)
        assert masks_equal(masks, decoded)
        back = transcode_payload(converted, "layers", spatial_shape=SHAPE)
        assert masks_equal(masks, payload_to_masks(back, spatial_shape=SHAPE))

    @pytest.mark.parametrize("source", ["layers", "bitmask", "instances", "probmap"])
    @pytest.mark.parametrize("target", ["layers", "bitmask", "instances", "probmap"])
    def test_S7_6_full_matrix(self, source, target):
        masks = random_masks(21, n_classes=4)
        start = encode_masks(masks, source, SHAPE)
        end = transcode_payload(start, target, spatial_shape=SHAPE)
        assert masks_equal(masks, payload_to_masks(end, spatial_shape=SHAPE))

    def test_S7_1_labelmap_is_only_a_valid_target_without_overlap(self):
        """A lossy conversion fails loudly rather than dropping the loser."""
        payload = encode_masks(random_masks(13, n_classes=5), "layers", SHAPE)
        with pytest.raises(MEDH5ValidationError) as exc:
            transcode_payload(payload, "labelmap", spatial_shape=SHAPE)
        assert exc.value.code == "E404"

    def test_labelmap_participates_when_classes_are_disjoint(self):
        masks = disjoint_masks()
        payload = encode_masks(masks, "labelmap", SHAPE)
        for target in TRANSCODABLE:
            converted = transcode_payload(payload, target, spatial_shape=SHAPE)
            assert masks_equal(masks, payload_to_masks(converted, spatial_shape=SHAPE))

    def test_transcode_to_same_kind_is_identity(self):
        payload = encode_masks(random_masks(1), "layers", SHAPE)
        assert transcode_payload(payload, "layers") is payload

    def test_unknown_target_is_refused(self):
        payload = encode_masks(random_masks(1), "layers", SHAPE)
        with pytest.raises(MEDH5ValidationError):
            transcode_payload(payload, "runes", spatial_shape=SHAPE)

    def test_instances_need_a_shape_to_decode(self):
        masks = random_masks(9, n_classes=2)
        payload = encode_masks(masks, "instances", SHAPE)
        with pytest.raises(MEDH5ValidationError) as exc:
            payload_to_masks(payload)
        assert exc.value.code == "E405"


class TestCostModel:
    def test_instances_beat_dense_for_sparse_data(self):
        masks = {1: np.zeros(SHAPE, dtype=bool)}
        masks[1][0, 0, 0] = True
        costs = cost_model(analyse(masks, SHAPE))
        assert costs.instances < costs.layers
        assert costs.best() == "instances"

    def test_labelmap_absent_when_classes_overlap(self):
        assert cost_model(analyse(random_masks(2), SHAPE)).labelmap is None

    def test_empty_stats_are_well_defined(self):
        stats = analyse({1: np.zeros(SHAPE, dtype=bool)}, SHAPE)
        assert stats.fill == 0.0
        assert stats.depth == 0.0
        assert stats.mean_degree == 0.0
