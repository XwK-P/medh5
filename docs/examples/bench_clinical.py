"""Measure the clinical (format 1.1) paths: write, select, read, batch, cache, amend.

A synthetic cohort, built here: ``--subjects`` subjects, each with a baseline
and a follow-up CT (``--shape``, ``int16``), an MR at follow-up for every
second subject, ``--events`` coded laboratory values spread over two years
before and after the baseline, a report and its revision (about 2 KiB of text
each), two responses and a lesion-assessment, all on one subject clock.  The
same images are also written as imaging-only 1.0 files, which is the baseline
the imaging path is compared with.

What is measured, each as the median of ``--repeats`` runs where it repeats:

- writing a sample, 1.1 with its history against 1.0 without, and the bytes;
- cold metadata selection: open a sample, read its clinical tables, select at
  a cutoff --- and the whole task's preflight;
- a slot window read from the 1.1 file and from its 1.0 imaging projection,
  and one document's text;
- complete training batches through ``ClinicalTaskDataset`` and a
  ``DataLoader``, with ``--workers`` processes: items per second;
- building and validating an event-level cache;
- amending a sample by one event (copy-on-write).

Synthetic measurements validate the path; they are not clinical-scale
performance, and nothing here promises a speedup.  Run it::

    python docs/examples/bench_clinical.py [--subjects 24] [--events 500] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.cache import HashingTextEncoder, build_document_cache, validate_cache
from medh5.clinical import DAY, HOUR, Clock, Document, Event, Link
from medh5.clinical import strip as strip_clinical
from medh5.task import Slot, SourceRef, Target, TaskManifest

try:  # not on Windows; the peak resident size is then not reported
    import resource
except ImportError:  # pragma: no cover - platform-dependent
    resource = None  # type: ignore[assignment]

RECIST = "org.example.recist"
LOINC = ("2160-0", "718-7", "6690-2", "777-3", "2345-7", "1742-6", "1920-8", "2951-2")
WORDS = (
    "nodule lobe right left lower upper stable increased decreased margin spiculated "
    "ground glass opacity lymph node mediastinal pleural effusion no evidence of "
    "metastatic disease comparison prior study impression findings technique"
).split()


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return 1000.0 * statistics.median(times)


def report_text(rng: np.random.Generator, n_words: int = 300) -> str:
    return " ".join(rng.choice(WORDS, n_words)) + "."


def imaging(w: Any, rng: np.random.Generator, shape: tuple[int, ...], mr: bool) -> None:
    patch = tuple(min(n, 32) for n in shape)
    for tp, days in (("tp0", 0), ("tp1", 90)):
        w.add_timepoint(tp, days_from_baseline=days)
        w.add_grid(f"ct_{tp}", shape=shape, spacing=(2.0, 0.8, 0.8), timepoint=tp,
                   patch_hint=patch)
        volume = np.rint(rng.normal(40.0, 25.0, shape)).astype(np.int16)
        w.add_image(f"CT_{tp}", volume, grid=f"ct_{tp}", modality="CT",
                    value_type="quantitative", value_units="HU")
    if mr:
        small = tuple(max(8, n // 2) for n in shape)
        w.add_grid("mr_tp1", shape=small, spacing=(4.0, 1.6, 1.6), timepoint="tp1")
        w.add_image("MR_tp1", rng.normal(300, 40, small).astype(np.float32),
                    grid="mr_tp1", modality="MR")


def history(w: Any, rng: np.random.Generator, n_events: int, mr: bool) -> None:
    w.set_clock(Clock.relative("clock", "acquisition start of the baseline CT"))
    times = np.sort(rng.integers(-730 * DAY, 730 * DAY, n_events))
    for i, t in enumerate(times):
        w.add_event(Event(
            f"lab{i}", f"lab{i}", "observation", "point", "final",
            effective_start_us=int(t), available_us=int(t) + 6 * HOUR,
            code_system="http://loinc.org", code=LOINC[i % len(LOINC)],
            value_num=float(rng.normal(1.0, 0.2)), unit="1",
        ))
    for tp, t in (("tp0", 0), ("tp1", 90 * DAY)):
        event_id = f"ct_{tp}"
        w.add_event(Event(event_id, event_id, "imaging", "point", "final",
                          effective_start_us=t, available_us=t + HOUR, timepoint_id=tp))
        w.add_link(Link.between(("event", event_id), "describes", ("image", f"CT_{tp}")))
    if mr:
        w.add_event(Event("mr_tp1", "mr_tp1", "imaging", "point", "final",
                          effective_start_us=90 * DAY, available_us=90 * DAY + 2 * HOUR,
                          timepoint_id="tp1"))
        w.add_link(Link.between(("event", "mr_tp1"), "describes", ("image", "MR_tp1")))
    for version, hours, status in (("v1", 4, "preliminary"), ("v2", 48, "amended")):
        w.add_event(Event(f"report_{version}", "report", "document", "point", status,
                          effective_start_us=0, available_us=hours * HOUR))
        w.add_document(Document(f"report_text_{version}", report_text(rng)))
        w.add_link(Link.between(("event", f"report_{version}"), "describes",
                                ("document", f"report_text_{version}")))
    w.add_link(Link.between(("event", "report_v2"), "supersedes", ("event", "report_v1")))
    for event_id, day, value in (("recist1", 90, "SD"), ("recist2", 200, "PD")):
        w.add_event(Event(event_id, event_id, "assessment", "point", "final",
                          effective_start_us=day * DAY, available_us=(day + 1) * DAY,
                          code_system=RECIST, code="overall_response", value_text=value))


def build(out: Path, subjects: int, n_events: int, shape: tuple[int, ...]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    with_history, without = [], []
    sizes = {"1.1": 0, "1.0": 0}
    for i in range(subjects):
        sid = f"S{i:03d}"
        for clinical, store in ((True, with_history), (False, without)):
            rng = np.random.default_rng(i)
            path = out / (f"{sid}.medh5" if clinical else f"{sid}-imaging.medh5")
            t = time.perf_counter()
            with medh5.create(path, sample_id=sid, subject_id=sid, codec="training") as w:
                imaging(w, rng, shape, mr=i % 2 == 0)
                if clinical:
                    history(w, rng, n_events, mr=i % 2 == 0)
            store.append(time.perf_counter() - t)
            sizes["1.1" if clinical else "1.0"] += path.stat().st_size
    return {
        "write_s_per_sample": {
            "1.1": statistics.median(with_history),
            "1.0": statistics.median(without),
        },
        "bytes_per_sample": {k: v / subjects for k, v in sizes.items()},
    }


def task_over(out: Path, subjects: int, shape: tuple[int, ...]) -> TaskManifest:
    patch = tuple(min(n, 32) for n in shape)
    task = TaskManifest.new(
        "bench-progression", "1", identity_namespace="bench",
        slots=[Slot("ct", "CT", required=True, patch=patch),
               Slot("mr", "MR", patch=tuple(max(8, n // 2) for n in patch))],
        target=Target("progression", "1", RECIST, "overall_response", positive=("PD",),
                      negative=("SD", "PR", "CR"), horizon_us=365 * DAY,
                      min_follow_up_us=60 * DAY),
        policy={"context_us": 365 * DAY},
        split=("fold-0", ["train", "val"]), base=out,
    )
    for i in range(subjects):
        sid = f"S{i:03d}"
        source = SourceRef.pin(out / f"{sid}.medh5", uri=f"{sid}.medh5", source_id=sid)
        task.add_subject(sid, [source], partition="train" if i % 5 else "val")
        task.add_row(f"{sid}@24h", sid, 24 * HOUR)
        task.add_row(f"{sid}@d95", sid, 95 * DAY)
    return task


def measure(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from medh5.torch import CACHE, ClinicalTaskDataset, collate_clinical, worker_init_fn

    out = Path(args.out or tempfile.mkdtemp(prefix="bench-clinical-"))
    shape = tuple(args.shape)
    results: dict[str, Any] = {"build": build(out, args.subjects, args.events, shape)}
    first = out / "S000.medh5"

    def select_cold() -> None:
        with medh5.open(first) as s:
            assert s.clinical is not None
            s.clinical.select(95 * DAY)

    results["select_cold_ms"] = median_ms(select_cold, args.repeats)
    with medh5.open(first) as s:
        assert s.clinical is not None
        clinical = s.clinical
        clinical.events  # noqa: B018 - read once, then time selection alone
        results["select_warm_ms"] = median_ms(lambda: clinical.select(95 * DAY), args.repeats)
        results["text_read_ms"] = median_ms(lambda: clinical.text("report_text_v2"), args.repeats)
        image = s.images["CT_tp1"]
        chunks = image.chunks
        window = tuple(slice(0, min(n, 32)) for n in shape)
        results["window_read_ms"] = {"1.1": median_ms(lambda: image.read(window), args.repeats)}
    projection = out / "S000-projection.medh5"
    strip_clinical(first, projection)
    with medh5.open(projection) as s:
        image = s.images["CT_tp1"]
        results["window_read_ms"]["1.0 projection"] = median_ms(
            lambda: image.read(window), args.repeats
        )

    task = task_over(out, args.subjects, shape)
    t = time.perf_counter()
    report = task.preflight()
    results["preflight_s"] = time.perf_counter() - t
    results["rows"] = report.counts

    encoder = HashingTextEncoder(dim=64)
    t = time.perf_counter()
    cache = build_document_cache(task, out / "reports.medh5cache", encoder)
    results["cache_build_s"] = time.perf_counter() - t
    t = time.perf_counter()
    assert validate_cache(out / "reports.medh5cache").ok
    results["cache_validate_s"] = time.perf_counter() - t
    results["cache_entries"] = cache.entries

    batching: dict[str, Any] = {}
    for label, documents in (("encoder", encoder), ("cache", out / "reports.medh5cache")):
        dataset = ClinicalTaskDataset(task, preflight=report, documents=documents)
        for workers in sorted({0, args.workers}):
            loader = DataLoader(dataset, batch_size=8, num_workers=workers,
                                worker_init_fn=worker_init_fn if workers else None,
                                collate_fn=collate_clinical)
            t = time.perf_counter()
            items = sum(len(batch["meta"]) for batch in loader)
            elapsed = time.perf_counter() - t
            batching[f"{label}, {workers} workers"] = {
                "items": items, "seconds": elapsed, "items_per_s": items / elapsed,
            }
    results["batches"] = batching

    def amend_one() -> None:
        with medh5.amend(first) as w:
            n = len(w.clinical()["events"])  # type: ignore[index]
            w.add_event(Event(f"extra{n}", f"extra{n}", "other", "point", "final",
                              effective_start_us=800 * DAY, available_us=800 * DAY))

    # The batches' handles, cached in this process: Windows cannot replace a
    # file that is still open.
    CACHE.close_all()
    results["amend_one_event_ms"] = median_ms(amend_one, max(3, args.repeats // 4))
    if resource is not None:
        # Linux reports KiB; this is the whole process, workers excluded.
        results["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results["conditions"] = {
        "subjects": args.subjects,
        "events_per_subject": args.events,
        "ct_shape": shape,
        "ct_chunks": chunks,
        "codec": "training (Blosc2 lz4)",
        "workers": args.workers,
        "repeats": args.repeats,
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "medh5": medh5.__version__,
        "numpy": np.__version__,
        "out": str(out),
    }
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 128, 128))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", help="where to write the cohort (default: a temporary directory)")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    results = measure(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
