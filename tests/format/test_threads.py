"""Engine calls that leave other Python threads running.

A call whose work grows with its data runs with the GIL released, so a
loader thread, a progress bar or a web server keeps going beside it.  Each
test runs a call while another thread stamps the time every half a
millisecond; a gap as long as the call is the GIL held for all of it.  Work
that walks Python objects, and constant-size work, keeps the GIL.
"""

from __future__ import annotations

import itertools
import os
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


def watched(call: Callable[[], Any]) -> tuple[float, float, float]:
    """The call's duration, the longest another thread went unscheduled during
    it, and the longest it went unscheduled before it, with nothing running:
    the ticker's own jitter, which a slow timer (macOS) makes milliseconds."""
    stamps: list[float] = []
    done = threading.Event()

    def tick() -> None:
        while not done.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.0005)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.02)
        t0 = time.perf_counter()
        call()
        t1 = time.perf_counter()
    finally:
        done.set()
        ticker.join()
    before = [t for t in stamps if t < t0]
    inside = [t for t in stamps if t0 < t < t1]
    idle = float(np.diff([*before, t0]).max()) if before else 0.0
    return t1 - t0, float(np.diff([t0, *inside, t1]).max()), idle


def released(call: Callable[[], Any], share: float = 0.25) -> None:
    """Assert no other thread waited *share* of the call or longer, beyond the
    ticker's own jitter, taking the best of three runs: one scheduler hiccup
    is not the GIL.  A call holding the GIL throughout still fails, since the
    calls here run many times longer than that jitter."""
    runs = [watched(call) for _ in range(3)]
    took, longest, idle = min(runs, key=lambda run: (run[1] - run[2]) / run[0])
    assert longest < share * took + idle, (
        f"no tick for {longest:.3f} of {took:.3f} s (idle jitter {idle:.3f} s)"
    )


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
        runs = []
        for k in range(3):
            path = tmp_path / f"s{k}.medh5"
            with medh5.create(path, sample_id="s") as w:
                w.add_grid("g", shape=mask.shape, spacing=(1.0, 1.0, 1.0))
                ct = np.zeros(mask.shape, np.int8)
                w.add_image("CT", ct, grid="g", modality="CT")
                # The call alone: the file's creation and commit are other
                # work, and a longer window only catches more scheduler noise,
                # which a slow runner shows at tens of milliseconds.
                runs.append(
                    watched(
                        lambda w=w: w.add_segmentation(
                            "seg", grid="g", masks={1: mask}, encoding="bitmask"
                        )
                    )
                )
        # The best of three: one scheduler hiccup is not the GIL.
        took, longest, idle = min(runs, key=lambda r: r[1] - r[2])
        start = time.perf_counter()
        np.ascontiguousarray(mask)
        copy = time.perf_counter() - start
        assert longest < max(copy / 2, 0.02) + idle, (
            f"no tick for {longest:.3f} of {took:.3f}"
        )
        with medh5.open(path) as s:
            assert np.array_equal(s.annotations["seg"].dense(), mask[np.newaxis])


class TestHandles:
    def test_closing_a_written_dataset_releases_the_gil(self, tmp_path: Path):
        """The last handle to a dataset just written flushes its chunk cache
        when it goes: compression and writes, with the GIL released."""
        shape = (192, 256, 256)
        ct = np.random.default_rng(4).integers(-1000, 1000, shape, dtype=np.int16)
        runs = []
        for k in range(3):
            with medh5.create(tmp_path / f"c{k}.medh5", sample_id="c") as w:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                held = [w.add_image("CT", ct, grid="g", modality="CT")]
                runs.append(watched(held.clear))
        took, longest, idle = min(runs, key=lambda r: (r[1] - r[2]) / r[0])
        assert longest < 0.25 * took + idle, (
            f"no tick for {longest:.3f} of {took:.3f} s"
        )


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
        # The class ids too: an integer array is not walked as Python ints.
        released(lambda: encode_points(boxes[:, :, 0], class_ids=class_ids))

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


# -- one thread at a time (E4) ------------------------------------------------------

BUSY = "is used by one thread at a time"


def refused_while(
    hold: Callable[[int], Any],
    call: Callable[[], Any],
    *,
    attempts: int = 20,
    passed: str | None = None,
    pause: float = 0.0,
) -> medh5.MEDH5Error:
    """The refusal *call* meets from another thread while *hold* runs, tried
    until one call lands inside one of *attempts* runs of *hold*.  Whatever
    else the other thread meets fails the test: a refusal is the only way a
    call on an object in use may end --- but an error naming *passed*, which
    the call raises once it is let through, and which means try again, after
    *pause* (as long before the first call): a call let through holds the
    object too, and one that never pauses keeps *hold* from starting."""
    for attempt in range(attempts):
        started, done = threading.Event(), threading.Event()
        refused: list[medh5.MEDH5Error] = []
        failed: list[BaseException] = []

        def other(
            started: threading.Event,
            done: threading.Event,
            refused: list[medh5.MEDH5Error],
            failed: list[BaseException],
        ) -> None:
            started.set()
            time.sleep(pause)
            while not done.is_set():
                try:
                    call()
                except medh5.MEDH5Error as e:
                    if passed is not None and passed in str(e):
                        time.sleep(pause)
                        continue
                    (refused if BUSY in str(e) else failed).append(e)
                    return
                except BaseException as e:  # pragma: no cover - the failure
                    failed.append(e)
                    return

        thread = threading.Thread(target=other, args=(started, done, refused, failed))
        thread.start()
        started.wait()
        try:
            hold(attempt)
        except medh5.MEDH5Error as e:
            # The other thread's call got there first: this run held nothing.
            if BUSY not in str(e):
                raise
        finally:
            done.set()
            thread.join()
        assert not failed, failed
        if refused:
            return refused[0]
    pytest.fail("no call of the other thread's landed while the first's ran")


def hammered(calls: list[Callable[[], Any]], *, rounds: int = 200) -> int:
    """Run each of *calls* *rounds* times, all at once from as many threads;
    the number refused.  Anything else a call raises fails the test."""
    barrier = threading.Barrier(len(calls))
    refused: list[medh5.MEDH5Error] = []
    failed: list[BaseException] = []

    def run(call: Callable[[], Any]) -> None:
        barrier.wait()
        for _ in range(rounds):
            try:
                call()
            except medh5.MEDH5Error as e:
                if BUSY not in str(e):  # pragma: no cover - the failure
                    failed.append(e)
                refused.append(e)
            except BaseException as e:  # pragma: no cover - the failure
                failed.append(e)

    threads = [threading.Thread(target=run, args=(c,)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failed, failed[:3]
    return len(refused)


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    """Enough voxels that writing them takes a while, and compresses slowly."""
    return np.random.default_rng(5).integers(-1000, 1000, (96, 256, 256), np.int16)


class TestOneThreadAtATime:
    """A mutable object of the extension is used by one thread at a time: a
    call that finds it in use is refused with a `MEDH5Error`, never made to
    wait (which could deadlock a free-threaded interpreter) and never raced
    (`RuntimeError: Already borrowed` was the borrow checker's refusal)."""

    def test_two_threads_on_one_writer_get_a_medh5_error(
        self, tmp_path: Path, volume: np.ndarray
    ):
        with medh5.create(tmp_path / "w.medh5", sample_id="w") as w:
            w.add_grid("g", shape=volume.shape, spacing=(1.0, 1.0, 1.0))
            images = itertools.count()

            def hold(_: int) -> None:
                w.add_image(f"CT{next(images)}", volume, grid="g", modality="CT")

            error = refused_while(hold, lambda: w.identity(sex="F"))
            assert f"a SampleWriter {BUSY}" in str(error)
            assert not isinstance(error, RuntimeError)
            # Readers are refused too while the writer is changed under them,
            # its document as much as the writer itself.
            assert BUSY in str(refused_while(hold, lambda: w.closed))
            assert BUSY in str(refused_while(hold, lambda: w.document.identity))
        with medh5.open(tmp_path / "w.medh5") as s:
            # A hold whose call the other thread's got ahead of wrote nothing.
            assert s.images and set(s.images) <= {f"CT{k}" for k in range(next(images))}

    def test_a_refused_call_changes_nothing(self, tmp_path: Path, volume: np.ndarray):
        with medh5.create(tmp_path / "w.medh5", sample_id="w") as w:
            w.add_grid("g", shape=volume.shape, spacing=(1.0, 1.0, 1.0))
            refused_while(
                lambda k: w.add_image(f"CT{k}", volume, grid="g", modality="CT"),
                lambda: w.extra("org.example", {"seen": True}),
            )
            w.extra("org.example", {"seen": False})
        with medh5.open(tmp_path / "w.medh5") as s:
            assert s.document.extra["org.example"] == {"seen": False}

    def test_two_threads_on_one_cache_writer_get_a_medh5_error(self, tmp_path: Path):
        from medh5 import _core
        from medh5.task import SourceRef
        from tests.helpers import write_sample

        source = SourceRef.pin(write_sample(tmp_path / "s.medh5")).to_json()

        handle = _core.cache_create(
            os.fspath(tmp_path / "c.medh5cache"),
            {
                "level": "event",
                "encoder": {"name": "fixture", "revision": "1"},
                "output": {"dtype": "float32", "shape": [4_000_000]},
            },
        )
        values = np.random.default_rng(6).random(4_000_000).astype(np.float32)

        def entry(k: str) -> dict[str, Any]:
            return {"entry_id": k, "sources": [source], "event_id": k}

        error = refused_while(
            lambda k: handle.add(entry(f"e{k}"), values),
            # Let through, the cache refuses its shape (T402): try again.
            lambda: handle.add(entry("other"), values[:4]),
            passed="T402",
            pause=0.01,
        )
        assert f"a CacheWriterHandle {BUSY}" in str(error)
        handle.abort()

    def test_a_write_resolves_its_classes_against_one_label_set(self, tmp_path: Path):
        """A label set another thread swaps in lands before or after a
        write's class names are resolved, never between two of them: every
        polygon of one call has the one class."""
        from medh5.labels import LabelClass, LabelSet
        from tests.helpers import SHAPE

        swapped = [
            LabelSet(
                "t",
                version="1.0.0",
                classes=[
                    LabelClass(liver, "liver", "Liver"),
                    LabelClass(3 - liver, "spleen", "Spleen"),
                ],
            )
            for liver in (1, 2)
        ]
        square = np.array([[4.0, 2, 2], [4, 2, 9], [4, 9, 9], [4, 9, 2]])
        polygons = [Polygon(square, class_id="liver", plane=(0, 4))] * 2_000  # type: ignore[arg-type]
        done = threading.Event()
        failed: list[BaseException] = []
        written: list[str] = []

        with medh5.create(tmp_path / "w.medh5", codec="portable") as w:
            w.label_set(swapped[0])
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="ct", modality="CT")

            def swap() -> None:
                for k in itertools.count():
                    if done.is_set():
                        return
                    try:
                        w.label_set(swapped[k % 2])
                    except medh5.MEDH5Error as e:
                        if BUSY not in str(e):  # pragma: no cover - the failure
                            failed.append(e)
                            return

            thread = threading.Thread(target=swap)
            thread.start()
            try:
                for k in range(10):
                    try:
                        w.add_contours(f"rt{k}", polygons, grid="ct")
                    except medh5.MEDH5Error as e:
                        assert BUSY in str(e)
                    else:
                        written.append(f"rt{k}")
            finally:
                done.set()
                thread.join()
            assert not failed, failed
        assert written
        with medh5.open(tmp_path / "w.medh5") as s:
            for ann_id in written:
                classes = {p.class_id for p in s.annotations[ann_id].polygons()}
                assert len(classes) == 1, (ann_id, classes)

    def test_two_threads_on_one_patch_draw_get_a_medh5_error(
        self, tmp_path: Path, label_set: Any
    ):
        from medh5.sampling import PatchSampler
        from tests.helpers import SHAPE, block, write_sample

        path = write_sample(
            tmp_path / "s.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2))},
        )
        sampler = PatchSampler(patch_size=(16, 24, 24))
        with medh5.open(path) as s:
            draws = sampler._engine().iter_draws(s, None, 10_000, 0)
            assert hammered([lambda: next(draws)] * 2, rounds=2_000) >= 0
            assert draws.__length_hint__() >= 0

    @pytest.mark.parametrize("threads", [8])
    def test_shared_records_are_refused_or_kept_whole(self, threads: int):
        """Short calls on one document, payload or provenance from many
        threads: each ends whole or refused, and what the object holds after
        is every edit that was not refused."""
        from medh5.annotations.base import AnnotationPayload
        from medh5.curation import Agent, Provenance
        from medh5.document import new_document

        doc = new_document("s")
        payload = AnnotationPayload("points")
        prov = Provenance()
        kept: list[str] = []
        lock = threading.Lock()

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def call() -> None:
                key = f"t{k}.{next(count)}"
                doc.extra[key] = {"k": k}
                with lock:
                    kept.append(key)
                payload.class_ids = (k,)
                assert payload.kind == "points"
                prov.add_agent(Agent(key, "software", key))
                _ = doc.to_json(), len(prov), prov.to_json()

            return call

        hammered([edit(k) for k in range(threads)], rounds=50)
        assert set(kept) <= set(doc.extra)
        assert len(payload.class_ids) == 1
        assert len(prov.agents) <= threads * 50

    @pytest.mark.parametrize("threads", [8])
    def test_a_documents_provenance_agrees_with_its_document(self, threads: int):
        """Threads replacing one agent of a document's provenance: the graph
        and the document end with the same agent, the last one not refused,
        because the provenance is held across both writes."""
        from medh5.curation import Agent
        from medh5.document import new_document

        doc = new_document("s")
        prov = doc.provenance

        def replace(k: int) -> Callable[[], None]:
            count = iter(range(10**9))
            return lambda: prov.add_agent(
                Agent("a", "software", f"t{k}.{next(count)}"), replace=True
            )

        hammered([replace(k) for k in range(threads)], rounds=200)
        assert prov.to_json()["agents"] == doc.provenance.to_json()["agents"]


class TestShared:
    """What threads may share: an open sample, read from many at once, and the
    process-wide registries."""

    @pytest.mark.parametrize("threads", [8])
    def test_one_sample_read_from_eight_threads(
        self, tmp_path: Path, label_set: Any, threads: int
    ):
        """Every thread reads what one thread alone reads: the image whole and
        by region, every annotation's planes, and the digests verified."""
        from tests.helpers import SHAPE, block, write_sample

        path = write_sample(
            tmp_path / "s.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2)), 2: block(SHAPE, (1, 1, 1))},
            index=True,
        )
        with medh5.open(path) as s:
            image = s.images["CT_tp0"]
            names = list(s.annotations)
            regions = [
                tuple(slice(i, i + 4) for _ in SHAPE) for i in range(min(SHAPE) - 4)
            ]
            whole = image.read()
            planes = {n: s.annotations[n].dense() for n in names}
            barrier = threading.Barrier(threads)
            failed: list[BaseException] = []

            def read(k: int) -> None:
                barrier.wait()
                try:
                    for i in range(50):
                        roi = regions[(k + i) % len(regions)]
                        assert np.array_equal(image.read(roi), whole[roi])
                        name = names[(k + i) % len(names)]
                        assert np.array_equal(s.annotations[name].dense(), planes[name])
                        if i % 10 == k % 10:
                            assert np.array_equal(s.images["CT_tp0"].read(), whole)
                            assert not s.verify().mismatched
                except BaseException as e:  # pragma: no cover - the failure
                    failed.append(e)

            workers = [threading.Thread(target=read, args=(k,)) for k in range(threads)]
            for t in workers:
                t.start()
            for t in workers:
                t.join()
            assert not failed, failed[:3]

    @pytest.mark.parametrize("threads", [8])
    def test_registries_are_safe_across_threads(self, label_set: Any, threads: int):
        """Threads registering, loading and dropping vocabularies meet only
        what one of them registered under a name, or no name at all."""
        from medh5 import labels

        barrier = threading.Barrier(threads)
        failed: list[BaseException] = []

        def churn(k: int) -> None:
            name = f"org.example.threads-{k % 2}"
            barrier.wait()
            try:
                for _ in range(200):
                    labels.register(name, label_set)
                    try:
                        assert labels.load(name) == label_set
                    except medh5.MEDH5ValidationError as e:
                        # Another thread dropped it between the two (E305).
                        assert "E305" in str(e), e
                    assert labels.load("binary-foreground") is not None
                    labels.unregister(name)
            except BaseException as e:  # pragma: no cover - the failure
                failed.append(e)

        workers = [threading.Thread(target=churn, args=(k,)) for k in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        assert not failed, failed[:3]
        assert not any(n.startswith("org.example.threads-") for n in labels.available())
