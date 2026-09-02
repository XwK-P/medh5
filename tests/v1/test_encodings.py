"""Voxel encodings: the uniform contract and lossless transcoding (spec §7)."""

from __future__ import annotations

import itertools

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.geometric import encode_obb
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
from medh5.labels.labelset import LabelClass, LabelSet
from tests.v1.conftest import block

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

    @pytest.mark.parametrize(
        ("box", "extent", "fill", "expected"),
        [
            ([[-2.5, 9.5]], 12, slice(2, 5), [0, 1, 2]),
            ([[5.5, 14.5]], 9, slice(0, 3), [6, 7, 8]),
            ([[-1.5, 11.5]], 14, slice(1, 4), [0, 1, 2]),
            ([[2.5, 5.5]], 3, slice(0, 3), [3, 4, 5]),
        ],
    )
    def test_S7_4_a_crop_is_read_in_the_frame_it_was_cut_in(
        self, tmp_path, box, extent, fill, expected
    ):
        """A box a resample pushed out of bounds still decodes where it belongs.

        The crop's coordinates are its *unclipped* box.  Deriving the window
        with the grid clip applied moved `start` for a box hanging off the near
        edge, so the crop was read from its own element 0 rather than from the
        first in-bounds one and the mask landed shifted by the overhang --- with
        nothing raised and the right dtype and shape.
        """
        shape = (10, 10, 10)
        full_box = np.array([box[0], [1.5, 4.5], [1.5, 4.5]], dtype=np.float32)
        crop = np.zeros((extent, 3, 3), dtype=bool)
        crop[fill] = True
        path = tmp_path / "inst.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
            w.label_set(
                LabelSet(
                    "t", version="1.0.0", classes=[LabelClass(3, "lesion", "Lesion")]
                )
            )
            w.add_segmentation(
                "objs",
                grid="g",
                encoding="instances",
                instances=[InstanceInput(3, 1, box=full_box, crop=crop)],
                annotated_classes=[3],
            )
        with medh5.open(path) as sample:
            dense = sample.annotations["objs"].dense([3])[0]
            rows = sorted(int(v) for v in np.unique(np.argwhere(dense)[:, 0]))
            assert rows == expected
            windowed = sample.annotations["objs"].dense(
                [3], roi=[slice(0, 10), slice(0, 10), slice(0, 10)]
            )[0]
            assert np.array_equal(windowed, dense)

    @pytest.mark.parametrize("plane", [None, 0, 15])
    def test_S7_2_has_ignore_region_reads_in_slabs(self, tmp_path, monkeypatch, plane):
        """A property that reads like a header lookup must not read the volume.

        It materialised every layer to answer, which is ~1.3 GiB for five uint16
        layers over a 512³ grid --- enough to end a per-sample loop on a question
        nobody expected to touch the data.  Correct whether the ignore region
        sits in the first slab, the last, or nowhere.
        """
        from medh5.annotations.voxel import layers as layers_module

        shape = (16, 32, 32)
        masks = {
            1: block(shape, (1, 1, 1), 4),
            2: block(shape, (1, 1, 1), 4),
            3: block(shape, (8, 8, 8), 4),
        }
        ignore = None
        if plane is not None:
            ignore = np.zeros(shape, dtype=bool)
            ignore[plane, 20:24, 20:24] = True

        path = tmp_path / f"ign-{plane}.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
            w.label_set(
                LabelSet(
                    "t",
                    version="1.0.0",
                    classes=[LabelClass(i, f"c{i}", f"C{i}") for i in (1, 2, 3)],
                )
            )
            w.add_segmentation(
                "objs", grid="g", masks=masks, encoding="layers", ignore=ignore
            )

        # One row per slab, so the scan cannot be reading the dataset whole.
        monkeypatch.setattr(layers_module, "_SCAN_BYTES", 1)
        with medh5.open(path) as sample:
            annotation = sample.annotations["objs"]
            assert annotation.kind == "layers"
            assert annotation.has_ignore_region is (plane is not None)

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


class TestTheIgnoreIdIsNotAClass:
    """§5.2 keeps `65535` out of `classes`, so no `dense` may answer for it.

    Every encoding *could* return an all-zero plane for it, and that shape is
    indistinguishable from a class examined and found absent --- which is how
    `dense([65535])` came to be documented as the way to read the ignore region.
    The guard sits in `resolve_classes` so one placement covers every encoding,
    and this asks all six whether it actually did.  It did not: `mask` overrode
    `dense` and dropped the argument, so five refused and the sixth returned the
    whole mask, reading as "ignored everywhere".
    """

    SHAPE = (6, 12, 12)

    def _labels(self):
        return LabelSet(
            "v",
            version="1.0.0",
            classes=[
                LabelClass(1, "liver", "Liver"),
                LabelClass(2, "spleen", "Spleen"),
            ],
        )

    def _annotation(self, tmp_path, encoding):
        """One sample per encoding, `mask` included --- it has no classes at all."""
        first = np.zeros(self.SHAPE, bool)
        first[:3] = True
        second = np.zeros(self.SHAPE, bool)
        # overlapping, so the multi-class encodings are all writable
        second[2:5] = True
        path = tmp_path / f"{encoding}.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(self._labels())
            w.add_grid("g", shape=self.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(self.SHAPE, np.int16), grid="g", modality="CT")
            if encoding == "mask":
                w.add_mask("seg", np.ones(self.SHAPE, bool), grid="g")
            else:
                w.add_segmentation(
                    "seg",
                    grid="g",
                    masks={"liver": first, "spleen": second},
                    annotated_classes=["liver", "spleen"],
                    encoding=encoding,
                )
        return path

    @pytest.mark.parametrize("encoding", [*DENSE, "mask"])
    def test_S5_2_no_encoding_returns_a_plane_for_the_ignore_id(
        self, tmp_path, encoding
    ):
        if encoding == "labelmap":
            pytest.skip("labelmap cannot hold the overlapping masks this uses")
        path = self._annotation(tmp_path, encoding)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            with pytest.raises(MEDH5ValidationError, match="reserved ignore id") as exc:
                annotation.dense([65535])
            assert exc.value.code == "E404"

    @pytest.mark.parametrize("encoding", [*DENSE, "mask"])
    def test_S7_6_refusing_the_ignore_id_leaves_ordinary_reads_alone(
        self, tmp_path, encoding
    ):
        """The guard must cost the uniform contract nothing.

        `mask` is the case worth stating: it has no classes, so `dense()` takes
        no argument and must keep working untouched, and a named class still
        selects nothing rather than starting to raise.
        """
        if encoding == "labelmap":
            pytest.skip("labelmap cannot hold the overlapping masks this uses")
        path = self._annotation(tmp_path, encoding)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.dense().shape[1:] == self.SHAPE
            if encoding == "mask":
                assert int(annotation.dense().sum()) == int(np.prod(self.SHAPE))
                assert annotation.dense(["liver"]).shape[1:] == self.SHAPE
            else:
                assert int(annotation.dense(["liver"]).sum()) > 0


class TestTranscodingRefusesWhatItCannotCarry:
    """§7.6 calls transcoding lossless, so anything it cannot carry must stop it."""

    SHAPE = (6, 12, 12)

    def _sample(self, tmp_path, **kwargs):
        labels = LabelSet(
            "v",
            version="1.0.0",
            classes=[
                LabelClass(1, "liver", "Liver"),
                LabelClass(3, "lesion", "Lesion"),
            ],
        )
        path = tmp_path / "t.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(labels)
            w.add_grid("g", shape=self.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(self.SHAPE, np.int16), grid="g", modality="CT")
            w.add_segmentation("seg", grid="g", annotated_classes=[1, 3], **kwargs)
        return path

    def test_S7_7_an_in_band_ignore_region_blocks_a_target_that_cannot_hold_it(
        self, tmp_path
    ):
        """`0` means "verified absent", not "unknown" -- §7.7's central point.

        `bitmask` and `probmap` express ignore as a separate `mask` annotation,
        which a payload-returning function cannot create, so the region was
        simply dropped: unexamined voxels silently became verified background
        for every annotated class, with W904 unable to fire because
        `annotated_class_ids == class_ids`.
        """
        liver = np.zeros(self.SHAPE, bool)
        liver[1:3, 1:5, 1:5] = True
        lesion = np.zeros(self.SHAPE, bool)
        lesion[4:5, 6:9, 6:9] = True
        ignore = np.zeros(self.SHAPE, bool)
        ignore[5:6, :, :] = True
        path = self._sample(tmp_path, masks={1: liver, 3: lesion}, ignore=ignore)

        with medh5.open(path) as sample:
            assert sample.annotations["seg"].has_ignore_region

        for target in ("bitmask", "probmap"):
            with (
                medh5.amend(path) as writer,
                pytest.raises(MEDH5ValidationError, match="ignore region"),
            ):
                writer.transcode_annotation("seg", target)

        # `layers` can hold it in band, so it is carried rather than refused.
        with medh5.amend(path) as writer:
            writer.transcode_annotation("seg", "layers")
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].has_ignore_region

    def test_S7_7_a_custom_ignore_id_survives_the_transcode(self, tmp_path):
        """The header carries `ignore_id` forward; the data has to agree with it.

        The target encoder was told the ignore *mask* and not the *id*, so it
        wrote the global default 65535 while the header still named the
        source's value. The header then pointed at a value the data did not
        contain, `_encodes_ignore` went False, and the region read as ordinary
        background -- the exact loss the refusal above exists to prevent.
        """
        liver = np.zeros(self.SHAPE, bool)
        liver[1:3, 1:5, 1:5] = True
        ignore = np.zeros(self.SHAPE, bool)
        ignore[5:6, :, :] = True
        path = self._sample(tmp_path, masks={1: liver}, ignore=ignore)

        with h5py.File(path, "r+") as handle:
            group = handle["annotations/seg"]
            data = np.asarray(group["data"])
            data[data == 65535] = 200
            group["data"][...] = data
            group.attrs["ignore_id"] = np.int64(200)

        with medh5.amend(path) as writer:
            writer.transcode_annotation("seg", "layers")

        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.header.ignore_id == 200
            assert annotation.has_ignore_region
            assert int(annotation.ignore_mask().sum()) == int(ignore.sum())
        with h5py.File(path) as handle:
            values = np.asarray(handle["annotations/seg"]["data"])
            assert 65535 not in np.unique(values), "the default id leaked back in"

    def test_S7_7_layers_exposes_the_ignore_region_it_reports_holding(self, tmp_path):
        """`labelmap` had `ignore_mask()`; `layers` reported one and offered none.

        A caller written as `getattr(annotation, "ignore_mask", None)` then read
        "no ignore region" from an annotation that had one, so the refusal above
        never fired for a `layers` source and the region was dropped anyway.
        """
        liver = np.zeros(self.SHAPE, bool)
        liver[1:3, 1:5, 1:5] = True
        lesion = np.zeros(self.SHAPE, bool)
        lesion[3:4, 6:9, 6:9] = True
        ignore = np.zeros(self.SHAPE, bool)
        ignore[5:6, :, :] = True
        path = self._sample(
            tmp_path, masks={1: liver, 3: lesion}, ignore=ignore, encoding="layers"
        )
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert type(annotation).__name__ == "LayersAnnotation"
            assert annotation.has_ignore_region
            assert int(annotation.ignore_mask().sum()) == int(ignore.sum())

        with (
            medh5.amend(path) as writer,
            pytest.raises(MEDH5ValidationError, match="ignore region"),
        ):
            writer.transcode_annotation("seg", "bitmask")

    def test_S7_4_a_dense_encoding_will_not_be_transcoded_to_instances(self, tmp_path):
        """A dense encoding knows which voxels; it never knew which object.

        Going to `instances` merged every object of a class into one mask and
        minted a fresh `instance_id` for it, so two lesions came back as one
        object carrying an id neither of them had -- in the field §7.4 makes
        the entire longitudinal join.
        """
        first = np.zeros(self.SHAPE, bool)
        first[0:2, 0:2, 0:2] = True
        second = np.zeros(self.SHAPE, bool)
        second[5:6, 8:10, 8:10] = True
        path = self._sample(
            tmp_path,
            instances=[
                InstanceInput(class_id=3, instance_id=101, mask=first),
                InstanceInput(class_id=3, instance_id=202, mask=second),
            ],
        )
        with medh5.amend(path) as writer:
            writer.transcode_annotation("seg", "layers")
        with (
            medh5.amend(path) as writer,
            pytest.raises(MEDH5ValidationError, match="no object identity"),
        ):
            writer.transcode_annotation("seg", "instances")


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


class TestPayloadEncoderGuards:
    """The encoders are exported, so a third-party converter calls them direct."""

    SHAPE = (4, 6, 6)

    def test_S5_3_reserved_and_out_of_range_class_ids_are_refused(self):
        """Unchecked, each of these was cast into the label dtype and wrapped.

        0 became background, -1 became 255, 65535 became the ignore value and
        70000 became 4464 -- every one of them decoding as a different class
        than the caller asked for, with nothing raised. The public writer
        already caught these; the encoders under it did not.
        """
        mask = np.zeros(self.SHAPE, bool)
        mask[1] = True
        for bad in (0, -1, 65535, 70000):
            with pytest.raises(MEDH5ValidationError) as exc:
                encode_labelmap({bad: mask}, self.SHAPE)
            assert exc.value.code == "E303"
        assert encode_labelmap({1: mask}, self.SHAPE).class_ids == (1,)
        assert encode_labelmap({65534: mask}, self.SHAPE).class_ids == (65534,)

    def test_S7_4_an_instance_id_beyond_uint32_keeps_its_value(self):
        """§7.4 permits uint64; hard-casting silently gave one object another's id."""
        mask = np.zeros(self.SHAPE, bool)
        mask[1] = True
        big = 2**32 + 7
        payload = encode_instances(
            [InstanceInput(class_id=1, instance_id=big, mask=mask)], self.SHAPE
        )
        stored = np.asarray(payload.datasets["instance_ids"])
        assert int(stored[0]) == big
        # Narrow ids keep the narrow dtype; the width follows the data.
        small = encode_instances(
            [InstanceInput(class_id=1, instance_id=7, mask=mask)], self.SHAPE
        )
        assert small.datasets["instance_ids"].dtype == np.uint32

    def test_S11_3_a_class_examined_and_absent_survives_the_instances_decode(self):
        """`payload_to_masks` keyed off the objects present, not the declared set.

        A class searched for and not found has no object, so it vanished on
        decode -- turning "verified absent" into "never looked for". The same
        path is what `check_roundtrip` uses to decode the original, so the
        module's own losslessness check could never see the loss.
        """
        present = np.zeros(self.SHAPE, bool)
        present[1] = True
        payload = encode_masks(
            {1: present, 5: np.zeros(self.SHAPE, bool)}, "instances", self.SHAPE
        )
        assert payload.class_ids == (1, 5)
        decoded = payload_to_masks(payload, spatial_shape=self.SHAPE)
        assert sorted(decoded) == [1, 5]
        assert not decoded[5].any()
        onward = transcode_payload(payload, "layers", spatial_shape=self.SHAPE)
        assert onward.class_ids == (1, 5)

    def test_an_empty_obb_collection_is_refused_with_a_coded_error(self):
        """`boxes`, `mesh` and `instances` all raise E405 here; `obb` did not."""
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_obb([], [], [], [])
        assert exc.value.code == "E405"
        empty = encode_obb(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3)), [])
        assert empty.class_ids == ()
