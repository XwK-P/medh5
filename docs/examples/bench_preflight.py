"""Measure a task's preflight on a large longitudinal cohort, and what it costs.

A synthetic multi-site cohort, built here once and reused: ``--subjects``
subjects, each split across ``--fragments`` files (one per site, all on the
subject's clock).  Every subject has

- ``--events`` events over four years: coded laboratory values (one in ten
  revised by a later version, one in a hundred with unknown availability),
  medication orders and administered courses (intervals), day-precision
  diagnoses, two static demographics held by every fragment (so preflight
  reconciles them), and an imaging event and a response assessment per visit;
- ``--documents`` notes of about ``--note-bytes`` of text each, a document
  event owning each, every tenth note revised;
- a small CT per visit (``--visits``) at the first site, a small MR at the
  others --- enough for slots to be filled; voxels are not the subject here;
- ``--cutoffs`` rows, monthly from the first visit.

Each measurement runs in a fresh interpreter, so its peak resident size and
the bytes it read through ``read(2)`` (``rchar`` of ``/proc/self/io``, Linux
only; page-cache hits included) are its own:

- ``preflight`` --- ``TaskManifest.preflight()``: open, pin-check and
  validate every source, reconcile fragments, select at every cutoff, fill
  slots, label;
- ``dataset`` --- that, then ``ClinicalTaskDataset`` (vocabulary fitted on
  the training partition) and its pickled size, which spawn-started
  ``DataLoader`` workers each receive;
- ``items`` --- reading ``--items`` items with document text encoded.

Run it::

    python docs/examples/bench_preflight.py [--subjects 60] [--events 2000] [--json out.json]

The cohort is written under ``--out`` (a temporary directory by default) and
reused when its parameters match.  Synthetic measurements validate the path
on one machine; they are not a promise about clinical-scale performance.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.clinical import DAY, HOUR
from medh5.task import Slot, SourceRef, Target, TaskManifest

try:  # not on Windows; the peak resident size is then not reported
    import resource
except ImportError:  # pragma: no cover - platform-dependent
    resource = None  # type: ignore[assignment]

RECIST = "org.example.recist"
LOINC = ("2160-0", "718-7", "6690-2", "777-3", "2345-7", "1742-6", "1920-8", "2951-2",
         "2823-3", "2075-0", "17861-6", "2028-9", "3094-0", "1751-7", "1975-2", "2885-2")
DRUGS = ("cisplatin", "pemetrexed", "pembrolizumab", "dexamethasone", "ondansetron")
ICD = ("C34.1", "J18.9", "I10", "E11.9", "N17.9", "D64.9")
WORDS = (
    "nodule lobe right left lower upper stable increased decreased margin spiculated "
    "ground glass opacity lymph node mediastinal pleural effusion no evidence of "
    "metastatic disease comparison prior study impression findings technique patient "
    "tolerated cycle dose held renal function nausea fatigue grade follow up plan"
).split()


def event(event_id: str, kind: str, start: tuple[int, int] | None, available: tuple[int, int] | None,
          *, record: str | None = None, temporal: str = "point", status: str = "final",
          end: tuple[int, int] | None = None, **fields: Any) -> dict[str, Any]:
    return {
        "event_id": event_id, "record_id": record or event_id, "kind": kind,
        "temporal_type": temporal, "status": status,
        "effective_start_us": None if start is None else list(start),
        "effective_end_us": None if end is None else list(end),
        "available_us": None if available is None else list(available), **fields,
    }


def link(source: tuple[str, str], relation: str, target: tuple[str, str]) -> dict[str, Any]:
    return {"source_type": source[0], "source_id": source[1], "relation": relation,
            "target_type": target[0], "target_id": target[1]}


def subject_history(rng: np.random.Generator, args: argparse.Namespace, n_sites: int
                    ) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    """Per site: (events, documents, links), without the imaging."""
    sites: list[tuple[list[Any], list[Any], list[Any]]] = [([], [], []) for _ in range(n_sites)]
    span = 4 * 365 * DAY
    origin = -2 * 365 * DAY

    def site_of(t: int) -> tuple[list[Any], list[Any], list[Any]]:
        # Earlier history at the first site, later at the others, in bands.
        band = min(n_sites - 1, max(0, int((t - origin) * n_sites // span)))
        return sites[band]

    for s in sites:  # demographics, held by every fragment
        s[0].append(event("sex", "other", None, (0, 0), temporal="static",
                          code_system="http://loinc.org", code="76689-9", value_text="female"))
        s[0].append(event("birth_year", "other", None, (0, 0), temporal="static",
                          code_system="org.example.demo", code="birth_year", value_num=1956.0,
                          unit="a"))
    n_docs = args.documents
    n_rest = max(0, args.events - 2)
    n_labs = int(n_rest * 0.70)
    n_meds = int(n_rest * 0.15)
    n_dx = n_rest - n_labs - n_meds
    for i, t in enumerate(np.sort(rng.integers(origin, origin + span, n_labs))):
        t = int(t)
        site = site_of(t)
        lag = int(rng.integers(1, 12)) * HOUR
        known = rng.random() >= 0.01
        code = LOINC[i % len(LOINC)]
        site[0].append(event(f"lab{i}", "observation", (t, t), (t + lag, t + lag) if known else None,
                             code_system="http://loinc.org", code=code,
                             value_num=float(rng.normal(1.0, 0.3)), unit="mg/dL"))
        if i % 10 == 3:  # a corrected result, learnt days later
            later = t + lag + int(rng.integers(1, 20)) * DAY
            site[0].append(event(f"lab{i}_v2", "observation", (t, t), (later, later), record=f"lab{i}",
                                 status="amended", code_system="http://loinc.org", code=code,
                                 value_num=float(rng.normal(1.0, 0.3)), unit="mg/dL"))
            site[2].append(link(("event", f"lab{i}_v2"), "supersedes", ("event", f"lab{i}")))
    for i, t in enumerate(np.sort(rng.integers(origin, origin + span, n_meds))):
        t = int(t)
        site = site_of(t)
        drug = DRUGS[i % len(DRUGS)]
        if i % 2 == 0:
            site[0].append(event(f"rx{i}", "medication_order", (t, t), (t, t), status="completed",
                                 code_system="org.example.rx", code=drug))
        else:
            length = int(rng.integers(1, 21)) * DAY
            site[0].append(event(f"rx{i}", "medication_administration", (t, t + HOUR),
                                 (t + 2 * HOUR, t + 2 * HOUR), temporal="interval",
                                 end=(t + length, t + length + DAY), status="completed",
                                 code_system="org.example.rx", code=drug))
    for i, t in enumerate(np.sort(rng.integers(origin, origin + span, n_dx))):
        day = int(t) // DAY * DAY
        site = site_of(day)
        site[0].append(event(f"dx{i}", "diagnosis", (day, day + DAY - 1), (day + DAY, day + 2 * DAY),
                             code_system="http://hl7.org/fhir/sid/icd-10", code=ICD[i % len(ICD)]))
    for i, t in enumerate(np.sort(rng.integers(origin, origin + span, n_docs))):
        t = int(t)
        site = site_of(t)
        words = max(1, args.note_bytes // 7)
        versions = 2 if i % 10 == 0 else 1
        for v in range(versions):
            eid = f"note{i}" if v == 0 else f"note{i}_v{v + 1}"
            when = t + (2 + 72 * v) * HOUR
            site[0].append(event(eid, "document", (t, t), (when, when), record=f"note{i}",
                                 status="final" if v == 0 else "amended"))
            text = " ".join(rng.choice(WORDS, words)) + "."
            site[1].append({"document_id": f"{eid}_text", "media_type": "text/plain", "text": text})
            site[2].append(link(("event", eid), "describes", ("document", f"{eid}_text")))
            if v:
                site[2].append(link(("event", eid), "supersedes", ("event", f"note{i}")))
    return sites


def write_fragment(path: Path, sid: str, site: int, n_visits: int, history: tuple[Any, Any, Any],
                   rng: np.random.Generator) -> None:
    events, documents, links = history
    shape = (8, 16, 16)
    with medh5.create(path, sample_id=f"{sid}-{site}", subject_id=sid, codec="training") as w:
        visits = range(n_visits) if site == 0 else range(1)
        for v in visits:
            tp = f"tp{v}"
            day = 90 * v
            w.add_timepoint(tp, days_from_baseline=day)
            modality, image = ("CT", f"CT_{tp}") if site == 0 else ("MR", f"MR_{tp}")
            w.add_grid(f"g_{tp}", shape=shape, spacing=(2.0, 1.0, 1.0), timepoint=tp)
            w.add_image(image, rng.normal(40, 20, shape).astype(np.float32), grid=f"g_{tp}",
                        modality=modality)
            t = day * DAY + site * HOUR
            events = [*events, event(f"img_{site}_{tp}", "imaging", (t, t), (t + HOUR, t + HOUR),
                                     timepoint_id=tp)]
            links = [*links, link(("event", f"img_{site}_{tp}"), "describes", ("image", image))]
            if site == 0 and v:
                response = ("SD", "PR", "PD")[int(rng.integers(0, 3))]
                events = [*events, event(f"recist_{tp}", "assessment", (t, t + DAY), (t + 2 * DAY, t + 2 * DAY),
                                         code_system=RECIST, code="overall_response", value_text=response)]
        w.add_records({
            "clinical": {"schema": "medh5.clinical/1",
                         "clock": {"id": f"{sid}-clock", "unit": "us", "reference": "relative",
                                   "origin_description": "acquisition of the baseline CT"}},
            "events": events, "documents": documents, "links": links,
        })


def build(out: Path, args: argparse.Namespace) -> Path:
    """Write the cohort and its task, unless an identical one is there."""
    params = {k: getattr(args, k) for k in ("subjects", "fragments", "events", "documents",
                                            "note_bytes", "visits", "cutoffs")}
    marker = out / "cohort.json"
    if marker.exists() and json.loads(marker.read_text()) == params:
        return out / "task.json"
    out.mkdir(parents=True, exist_ok=True)
    task = TaskManifest.new(
        "bench-preflight", "1", identity_namespace="bench",
        slots=[Slot("ct", "CT", required=True, patch=(8, 16, 16)), Slot("mr", "MR", patch=(8, 16, 16))],
        target=Target("progression", "1", RECIST, "overall_response", positive=("PD",),
                      negative=("SD", "PR", "CR"), horizon_us=180 * DAY, min_follow_up_us=60 * DAY),
        policy={"context_us": 365 * DAY},
        split=("fold-0", ["train", "val"]), base=out,
    )
    t = time.perf_counter()
    for i in range(args.subjects):
        sid = f"S{i:04d}"
        rng = np.random.default_rng(i)
        sources = []
        for site, history in enumerate(subject_history(rng, args, args.fragments)):
            name = f"{sid}-site{site}.medh5"
            write_fragment(out / name, sid, site, args.visits, history, rng)
            sources.append(SourceRef.pin(out / name, uri=name, source_id=f"{sid}-site{site}"))
        task.add_subject(sid, sources, partition="train" if i % 5 else "val")
        for k in range(args.cutoffs):
            task.add_row(f"{sid}@m{k}", sid, k * 30 * DAY + 3 * DAY)
    task.reconcile()
    path = task.save(out / "task.json")
    marker.write_text(json.dumps(params))
    print(f"built {args.subjects} subjects x {args.fragments} fragments in "
          f"{time.perf_counter() - t:.1f} s under {out}", file=sys.stderr)
    return path


# -- measurements, each in its own interpreter -----------------------------------------------


def _io() -> int | None:
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("rchar:"):
                return int(line.split()[1])
    except OSError:  # pragma: no cover - not Linux
        return None
    return None


def _rss_mib() -> float | None:
    if resource is None:  # pragma: no cover - platform-dependent
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def run_one(what: str, task_path: Path, items: int) -> dict[str, Any]:
    from medh5 import _core
    from medh5.task import Preflight

    out: dict[str, Any] = {}
    if what in ("dataset", "items"):
        import torch  # noqa: F401 - its own resident size is not the dataset's

        out["rss_after_torch_mib"] = _rss_mib()
    task = TaskManifest.load(task_path)
    before_io, rss0 = _io(), _rss_mib()
    t = time.perf_counter()
    columns = _core.task_preflight(task.to_json(), os.fspath(task_path.parent), False)
    out["engine_s"] = time.perf_counter() - t
    report = Preflight.from_core(columns)
    del columns
    out["preflight_s"] = time.perf_counter() - t
    out.update(rows=len(report.rows), counts=report.counts, ok=report.ok)
    if what in ("dataset", "items"):
        from medh5.cache import HashingTextEncoder
        from medh5.torch import ClinicalTaskDataset

        t = time.perf_counter()
        encoder = HashingTextEncoder(dim=64) if what == "items" else None
        ds = ClinicalTaskDataset(task, preflight=report, documents=encoder)
        out["dataset_s"] = time.perf_counter() - t
        out["dataset_rows"] = len(ds)
        t = time.perf_counter()
        out["pickled_mib"] = len(pickle.dumps(ds)) / 2**20
        out["pickle_s"] = time.perf_counter() - t
        if what == "items":
            n = min(items, len(ds))
            t = time.perf_counter()
            events = 0
            for i in range(n):
                item = ds[(i * 7919) % len(ds)]
                events += int(item["events"]["concept"].shape[0])
            out["items_per_s"] = n / (time.perf_counter() - t)
            out["events_per_item"] = events / max(n, 1)
    after_io = _io()
    out["read_mib"] = None if before_io is None or after_io is None else (after_io - before_io) / 2**20
    out["rss_start_mib"] = rss0
    out["rss_peak_mib"] = _rss_mib()
    return out


def measure(what: str, task_path: Path, items: int) -> dict[str, Any]:
    found = subprocess.run(
        [sys.executable, __file__, "--measure", what, "--task", os.fspath(task_path), "--items", str(items)],
        check=True, capture_output=True, text=True,
    )
    return dict(json.loads(found.stdout))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--subjects", type=int, default=60)
    p.add_argument("--fragments", type=int, default=2)
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--documents", type=int, default=100)
    p.add_argument("--note-bytes", type=int, default=2048)
    p.add_argument("--visits", type=int, default=6)
    p.add_argument("--cutoffs", type=int, default=12)
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--only", choices=("preflight", "dataset", "items"), action="append")
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--measure", help=argparse.SUPPRESS)
    p.add_argument("--task", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.measure:
        return run_one(args.measure, args.task, args.items)
    out = args.out or Path(tempfile.mkdtemp(prefix="bench-preflight-"))
    task_path = build(out, args)
    results: dict[str, Any] = {
        "conditions": {
            "subjects": args.subjects, "fragments": args.fragments, "events": args.events,
            "documents": args.documents, "note_bytes": args.note_bytes, "visits": args.visits,
            "cutoffs": args.cutoffs, "rows": args.subjects * args.cutoffs,
            "bytes_on_disk_mib": sum(f.stat().st_size for f in out.glob("*.medh5")) / 2**20,
            "platform": platform.platform(), "python": platform.python_version(),
            "cpus": os.cpu_count(), "medh5": medh5.__version__, "numpy": np.__version__,
        },
    }
    for what in args.only or ("preflight", "dataset", "items"):
        results[what] = measure(what, task_path, args.items)
        print(f"{what}: {json.dumps(results[what])}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
