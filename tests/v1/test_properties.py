"""Property-based sweep over the annotation encoders.

Four of the last findings in this release were one shape: a column or tag whose
*length* or *rank* did not match the elements it described.  Each was found by
inspection, one at a time, in whichever encoder somebody happened to be reading
--- and each time the sibling columns beside it turned out to have the same
hole.  Enumerating the mismatches instead covers every encoder at once, and
keeps covering them.

The contract under test is the one the corpus smoke test asserts for readers,
applied where these defects actually live:

* an encoder either **succeeds** or raises **MEDH5Error** --- never an
  ``IndexError``, ``TypeError`` or ``ValueError`` that a caller cannot catch by
  the documented type, and never a silent success that drops data;
* when it succeeds, every per-element column really is as long as the elements
  it labels, so nothing was quietly truncated on the way in.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from medh5.annotations.geometric import (
    encode_boxes,
    encode_keypoints,
    encode_mesh,
    encode_obb,
    encode_points,
)
from medh5.errors import MEDH5Error

# Small counts: the defects are at the boundaries between lengths, not at scale.
COUNT = st.integers(min_value=0, max_value=4)
DIM = st.integers(min_value=2, max_value=3)
SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Columns that hold one value per element, and how to build one of length n.
PER_ELEMENT = {
    "class_ids": lambda n: [1] * n,
    "instance_ids": lambda n: list(range(n)),
    "scores": lambda n: [0.5] * n,
    "attributes": lambda n: [{"a": 1}] * n,
}


def _consistent(payload, n: int, columns: tuple[str, ...]) -> None:
    """Every per-element column the payload kept is as long as the elements."""
    for name in columns:
        if name in payload.datasets:
            got = len(payload.datasets[name])
            assert got == n, (
                f"{payload.kind}: {name} kept {got} entries for {n} elements; "
                "a column shorter than what it labels leaves the rest unlabelled"
            )


def _guard(call, n: int, columns: tuple[str, ...]) -> None:
    """Succeed consistently, or refuse as `MEDH5Error`. Nothing else."""
    try:
        payload = call()
    except MEDH5Error:
        return
    except Exception as exc:
        raise AssertionError(
            f"raised {type(exc).__name__} rather than MEDH5Error: {exc}"
        ) from exc
    _consistent(payload, n, columns)


class TestEncoderColumns:
    @given(n=COUNT, dim=DIM, k=COUNT, column=st.sampled_from(sorted(PER_ELEMENT)))
    @SETTINGS
    def test_boxes(self, n, dim, k, column):
        kwargs = {c: PER_ELEMENT[c](n) for c in ("class_ids",)}
        kwargs[column] = PER_ELEMENT[column](k)
        boxes = np.zeros((n, dim, 2), np.float32)
        _guard(
            lambda: encode_boxes(boxes, **kwargs),
            n,
            tuple(PER_ELEMENT),
        )

    @given(n=COUNT, dim=DIM, k=COUNT)
    @SETTINGS
    def test_boxes_slice_index(self, n, dim, k):
        boxes = np.zeros((n, dim, 2), np.float32)
        _guard(
            lambda: encode_boxes(boxes, [1] * n, slice_index=[0] * k),
            n,
            ("class_ids", "slice_index"),
        )

    @given(n=COUNT, dim=DIM, k=COUNT, column=st.sampled_from(sorted(PER_ELEMENT)))
    @SETTINGS
    def test_obb(self, n, dim, k, column):
        kwargs = {"class_ids": [1] * n}
        kwargs[column] = PER_ELEMENT[column](k)
        _guard(
            lambda: encode_obb(
                np.zeros((n, dim), np.float32),
                np.zeros((n, dim), np.float32),
                np.broadcast_to(np.eye(dim, dtype=np.float32), (n, dim, dim)),
                **kwargs,
            ),
            n,
            tuple(PER_ELEMENT),
        )

    @given(n=COUNT, dim=DIM, k=COUNT, column=st.sampled_from(["class_ids", "weights"]))
    @SETTINGS
    def test_points(self, n, dim, k, column):
        kwargs = {"class_ids": [1] * n}
        kwargs[column] = ([1] * k) if column == "class_ids" else ([0.5] * k)
        _guard(
            lambda: encode_points(np.zeros((n, dim), np.float32), **kwargs),
            n,
            ("class_ids", "names", "weights"),
        )

    @given(n=COUNT, dim=DIM, k=COUNT, slots=st.integers(min_value=1, max_value=3))
    @SETTINGS
    def test_keypoints(self, n, dim, k, slots):
        _guard(
            lambda: encode_keypoints(
                np.zeros((n, slots, dim), np.float32),
                [1] * slots,
                [1] * n,
                visibility=np.ones((n, slots), np.uint8),
                scores=[0.5] * k,
            ),
            n,
            ("class_ids", "scores"),
        )

    @given(v=st.integers(min_value=3, max_value=6), k=COUNT, m=COUNT)
    @SETTINGS
    def test_mesh(self, v, k, m):
        vertices = np.zeros((v, 3), np.float32)
        faces = np.zeros((1, 3), np.int32)
        _guard(
            lambda: encode_mesh(
                vertices,
                faces,
                vertex_class_ids=[1] * k,
                mesh_offsets=[0, 1],
                mesh_class_ids=[1] * m,
            ),
            v,
            ("vertex_class_ids",),
        )


class TestEncoderRanks:
    """Rank, not just length: a column of the right length can be the wrong shape."""

    @given(n=st.integers(min_value=1, max_value=3), extra=st.integers(1, 3))
    @SETTINGS
    def test_a_two_dimensional_slice_index_is_refused(self, n, extra):
        boxes = np.zeros((n, 3, 2), np.float32)
        with pytest.raises(MEDH5Error):
            encode_boxes(boxes, [1] * n, slice_index=np.zeros((n, extra), np.int32))

    @given(n=st.integers(min_value=1, max_value=3))
    @SETTINGS
    def test_a_scalar_slice_index_is_refused(self, n):
        boxes = np.zeros((n, 3, 2), np.float32)
        with pytest.raises(MEDH5Error):
            encode_boxes(boxes, [1] * n, slice_index=np.int32(0))


class TestVoxelRoundTrip:
    """What every voxel encoding promises: the masks come back.

    Five encodings exist because one integer volume cannot hold overlapping
    classes (§7.0), so the interesting inputs are the overlapping ones --- and
    which encoding is legal depends on the overlap. Generating the overlap
    rather than fixing it exercises the selection logic and each encoding's
    packing against the same invariant: `dense()` returns what was written.
    """

    @staticmethod
    def _masks(rng, shape, n_classes, overlap):
        """Random masks, including the degenerate ones.

        An all-empty class is not a degenerate input to skip: it is the verified
        negative the coverage contract exists to record (§11.3), and an all-true
        one is the boundary where every class claims every voxel.
        """
        out = {}
        for index in range(n_classes):
            draw = rng.integers(0, 6)
            if draw == 0:
                mask = np.zeros(shape, bool)
            elif draw == 1:
                mask = np.ones(shape, bool)
            elif overlap:
                mask = rng.random(shape) < 0.5
            else:
                # Disjoint by construction: each class owns one residue class of
                # a flat index, so no two can claim the same voxel.
                flat = np.arange(int(np.prod(shape))).reshape(shape)
                mask = (flat % n_classes) == index
            out[index + 1] = mask
        return out

    @given(
        n_classes=st.integers(min_value=1, max_value=3),
        overlap=st.booleans(),
        encoding=st.sampled_from(["auto", "labelmap", "layers", "bitmask"]),
        seed=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=120, deadline=None)
    def test_masks_survive_every_encoding(
        self, tmp_path_factory, n_classes, overlap, encoding, seed
    ):
        import medh5
        from medh5.labels.labelset import LabelClass, LabelSet

        shape = (6, 8, 8)
        rng = np.random.default_rng(seed)
        masks = self._masks(rng, shape, n_classes, overlap)
        path = tmp_path_factory.mktemp("prop") / "s.medh5"
        label_set = LabelSet(
            "p",
            version="1.0.0",
            classes=[LabelClass(i, f"c{i}", f"C{i}") for i in masks],
        )
        try:
            with medh5.create(path, sample_id="S", subject_id="P") as writer:
                writer.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                writer.add_image(
                    "CT", np.zeros(shape, np.int16), grid="g", modality="CT"
                )
                writer.label_set(label_set)
                writer.add_segmentation("seg", grid="g", masks=masks, encoding=encoding)
        except MEDH5Error:
            # `labelmap` cannot hold an overlap and says so (E404); refusing is
            # a correct outcome, silently dropping the overlap would not be.
            return

        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            for class_id, expected in masks.items():
                got = annotation.dense([class_id])[0]
                assert got.shape == expected.shape
                assert np.array_equal(got, expected), (
                    f"{encoding}: class {class_id} did not survive the round trip"
                )

    @given(
        n_classes=st.integers(min_value=2, max_value=3),
        odd=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=60, deadline=None)
    def test_masks_of_different_shapes_are_refused(
        self, tmp_path_factory, n_classes, odd
    ):
        """A mask that is not the grid's shape has no defined placement."""
        import medh5
        from medh5.labels.labelset import LabelClass, LabelSet

        shape = (6, 8, 8)
        masks = {i + 1: np.zeros(shape, bool) for i in range(n_classes)}
        masks[odd + 1] = np.zeros((6, 8, 7), bool)
        path = tmp_path_factory.mktemp("prop") / "bad.medh5"
        label_set = LabelSet(
            "p",
            version="1.0.0",
            classes=[LabelClass(i, f"c{i}", f"C{i}") for i in masks],
        )
        with (
            pytest.raises(MEDH5Error),
            medh5.create(path, sample_id="S", subject_id="P") as writer,
        ):
            writer.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            writer.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            writer.label_set(label_set)
            writer.add_segmentation("seg", grid="g", masks=masks)


class TestTranscodeRoundTrip:
    """Transcoding preserves what an annotation *contains* (§7).

    Two earlier findings in this release were transcode losing information ---
    an in-band ignore region with nowhere to go, and a dense encoding converted
    to `instances` it could not represent. Both were found by inspection of one
    pair. Enumerating the pairs holds every one of them to the same rule: the
    masks that come out are the masks that went in, or the conversion refuses.
    """

    DENSE = ("labelmap", "layers", "bitmask")

    @given(
        source=st.sampled_from(DENSE),
        target=st.sampled_from([*DENSE, "instances", "probmap"]),
        n_classes=st.integers(min_value=1, max_value=3),
        overlap=st.booleans(),
        seed=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=150, deadline=None)
    def test_every_pair_preserves_the_masks_or_refuses(
        self, source, target, n_classes, overlap, seed
    ):
        from medh5.annotations.voxel import encode_voxels
        from medh5.annotations.voxel.transcode import (
            payload_to_masks,
            transcode_payload,
        )

        shape = (4, 6, 6)
        rng = np.random.default_rng(seed)
        masks = TestVoxelRoundTrip._masks(rng, shape, n_classes, overlap)
        try:
            payload, _ = encode_voxels(masks, shape, encoding=source)
        except MEDH5Error:
            return  # labelmap refusing an overlap is correct

        try:
            converted = transcode_payload(payload, target, spatial_shape=shape)
        except MEDH5Error:
            return  # a target that cannot hold the content must refuse
        except Exception as exc:
            raise AssertionError(
                f"{source}->{target} raised {type(exc).__name__}: {exc}"
            ) from exc

        back = payload_to_masks(converted, spatial_shape=shape)
        for class_id, expected in masks.items():
            assert class_id in back, (
                f"{source}->{target} dropped class {class_id} entirely"
            )
            assert np.array_equal(back[class_id], expected), (
                f"{source}->{target} changed class {class_id}'s voxels"
            )
