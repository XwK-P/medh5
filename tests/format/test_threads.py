"""Engine calls that leave other Python threads running.

A call whose work grows with its data runs with the GIL released, so a
loader thread, a progress bar or a web server keeps going beside it.  Each
test runs a call while another thread stamps the time every half a
millisecond; a gap as long as the call is the GIL held for all of it.  Work
that walks Python objects, and constant-size work, keeps the GIL.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.geometric import (
    Polygon,
    encode_boxes,
    encode_contours,
    encode_points,
)
from medh5.annotations.voxel import normalize_masks
from medh5.curation.agreement import dice, iou
from medh5.transforms import refuse_outside


def watched(call: Callable[[], Any]) -> tuple[float, float]:
    """The call's duration and the longest another thread went unscheduled."""
    stamps: list[float] = []
    done = threading.Event()

    def tick() -> None:
        while not done.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.0005)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.01)
        t0 = time.perf_counter()
        call()
        t1 = time.perf_counter()
    finally:
        done.set()
        ticker.join()
    inside = [t for t in stamps if t0 < t < t1]
    return t1 - t0, float(np.diff([t0, *inside, t1]).max())


def released(call: Callable[[], Any], share: float = 0.25) -> None:
    """Assert no other thread waited *share* of the call or longer, taking
    the best of three runs: one scheduler hiccup is not the GIL."""
    runs = [watched(call) for _ in range(3)]
    took, longest = min(runs, key=lambda run: run[1] / run[0])
    assert longest < share * took, f"no tick for {longest:.3f} of {took:.3f} s"


@pytest.fixture(scope="module")
def masks() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = (160, 256, 256)
    return rng.random(shape) < 0.3, rng.random(shape) < 0.3


class TestMasks:
    def test_normalize_masks_does_not_copy_contiguous_bool(self):
        mask = np.zeros((4, 5, 6), dtype=bool)
        resolved, shape = normalize_masks({1: mask})
        assert shape == (4, 5, 6)
        assert np.shares_memory(resolved[1], mask)

    def test_normalize_masks_converts_what_is_not(self):
        fortran = np.asfortranarray(np.ones((4, 5, 6), dtype=bool))
        resolved, _ = normalize_masks({2: fortran, 1: np.ones((4, 5, 6), np.uint8)})
        assert list(resolved) == [1, 2]
        for found in resolved.values():
            assert found.dtype == np.bool_ and found.flags.c_contiguous
            assert found.all()
        assert not np.shares_memory(resolved[2], fortran)

    def test_normalize_masks_still_checks_shapes(self):
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({1: np.zeros((4, 5), bool), 2: np.zeros((4, 6), bool)})
        assert exc.value.code == "E405"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({1: np.zeros((4, 5), bool)}, (4, 6))
        assert exc.value.code == "E405"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({})
        assert exc.value.code == "E410"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({0: np.zeros((4, 5), bool)})
        assert exc.value.code == "E303"

    def test_agreement_releases_the_gil(self, masks):
        a, b = masks
        released(lambda: dice(a, b))
        released(lambda: iou(a, b))

    def test_agreement_broadcasts_without_copying(self, masks):
        """A plane against a volume counts as NumPy's broadcast would, with
        no broadcast copy made, and with the GIL released."""
        a, b = masks
        plane = b[:1]
        both = np.broadcast_to(plane, a.shape)
        expect = 2 * (a & both).sum() / (a.sum() + both.sum())
        assert dice(a, plane) == pytest.approx(expect)
        assert iou(plane, a) == pytest.approx((a & both).sum() / (a | both).sum())
        released(lambda: dice(a, plane))

    def test_large_array_arguments_copy_without_the_gil(self, tmp_path: Path, masks):
        """An argument in another layout than C is copied into one, which for
        a large array is a pass over all of it: done with the GIL released."""
        mask = np.asfortranarray(masks[0])
        assert not mask.flags.c_contiguous
        path = tmp_path / "s.medh5"

        def write() -> None:
            with medh5.create(path, sample_id="s") as w:
                w.add_grid("g", shape=mask.shape, spacing=(1.0, 1.0, 1.0))
                ct = np.zeros(mask.shape, np.int8)
                w.add_image("CT", ct, grid="g", modality="CT")
                w.add_segmentation("seg", grid="g", masks={1: mask}, encoding="bitmask")

        took, longest = watched(write)
        start = time.perf_counter()
        np.ascontiguousarray(mask)
        copy = time.perf_counter() - start
        assert longest < max(copy / 2, 0.02), f"no tick for {longest:.3f} of {took:.3f}"
        with medh5.open(path) as s:
            assert np.array_equal(s.annotations["seg"].dense(), mask[np.newaxis])


class TestGeometric:
    def test_encode_contours_releases_while_packing(self):
        rng = np.random.default_rng(1)
        polygons = [
            Polygon(
                vertices=np.c_[np.full(400_000, z), rng.random((400_000, 2)) * 500],
                class_id=1,
                plane=(0, z),
            )
            for z in range(8)
        ]
        released(lambda: encode_contours(polygons, ndim=3))

    @pytest.mark.parametrize("rows", [1_000_000])
    def test_row_encoders_release_the_gil(self, rows: int):
        rng = np.random.default_rng(2)
        boxes = np.sort(rng.random((rows, 3, 2)) * 100, axis=2)
        class_ids = np.ones(rows, np.int64)
        released(lambda: encode_boxes(boxes, class_ids))
        released(lambda: encode_points(boxes[:, :, 0]))

    def test_refuse_outside_releases_the_gil(self):
        inside = np.ones(40_000_000, dtype=bool)
        released(lambda: refuse_outside(inside))

    def test_to_world_releases_the_gil(self, tmp_path: Path):
        path = tmp_path / "p.medh5"
        with medh5.create(path, sample_id="p") as w:
            w.add_grid("g", shape=(10, 10, 10), spacing=(0.5, 0.7, 1.1))
            w.add_image("CT", np.zeros((10, 10, 10), np.int16), grid="g", modality="CT")
            w.add_points("pts", np.zeros((1, 3)), grid="g")
        coords = np.random.default_rng(3).random((2_000_000, 3)) * 10
        with medh5.open(path) as s:
            ann = s.annotations["pts"]
            # Index space on an axis-aligned grid at the origin.
            assert np.allclose(ann.to_world(coords), coords * (0.5, 0.7, 1.1))
            assert np.allclose(ann.to_index(coords), coords)
            released(lambda: ann.to_world(coords))
            released(lambda: ann.to_index(coords))
