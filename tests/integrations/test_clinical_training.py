"""Cutoff-aware multimodal longitudinal training: real batches (contract §6).

The runnable examples are executed here --- the worked cohort, the collection
members, the cache invalidations and the benchmark --- and the batches they
produce are held to the contract: different visits, missing modalities,
histories of different lengths, and masks that stay distinct.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

import medh5  # noqa: E402
from medh5.cache import (  # noqa: E402
    CacheWriter,
    HashingTextEncoder,
    build_document_cache,
    validate_cache,
)
from medh5.clinical import DAY, HOUR  # noqa: E402
from medh5.errors import MEDH5ValidationError  # noqa: E402
from medh5.task import TaskManifest  # noqa: E402
from medh5.torch import (  # noqa: E402
    CACHE,
    ClinicalTaskDataset,
    ConceptVocabulary,
    collate_clinical,
    open_cached,
    worker_init_fn,
)
from medh5.torch.clinical import UNKNOWN, concept_of  # noqa: E402
from tests.helpers import ROOT  # noqa: E402

EXAMPLES = ROOT / "docs" / "examples"


def example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The worked example's cohort and task, built once."""
    module = example("clinical_longitudinal")
    out = tmp_path_factory.mktemp("worked")
    summary = module.main(out)
    return {"out": out, "summary": summary, "task": out / "progression.task.json"}


class TestWorkedExample:
    def test_S6_a_real_batch(self, worked: dict[str, Any]):
        s = worked["summary"]
        assert s["preflight"] == {"eligible": 10, "excluded": 2}
        # Different visits fill one slot: the baseline at 24 h, the follow-up
        # at day 95.
        assert set(s["ct visit"]) == {"tp0", "tp1"}
        # A missing modality is a mask, not a dropped key.
        assert set(s["mr present"]) == {True, False}
        # Histories of different lengths, padded.
        assert len(set(s["events"])) > 1
        assert s["event batch"][1] == max(s["events"])
        # A censored target is unobserved; an unannotated class is uncovered.
        assert False in s["observed"] and False in s["coverage"]
        assert s["ct batch"] == (7, 1, 8, 16, 16)

    def test_S6_masks_stay_distinct(self, worked: dict[str, Any]):
        ds = ClinicalTaskDataset(worked["task"], partition="train")
        batch = collate_clinical([ds[i] for i in range(len(ds))])
        rows = [m["row_id"] for m in batch["meta"]]
        mr = batch["present"]["mr"]
        missing = [i for i, r in enumerate(rows) if not mr[i]]
        assert missing
        for i in missing:
            assert not batch["images"]["mr"][i].any()
            assert not batch["valid"]["mr"][i].any()
            assert batch["ignore"]["mr"][i].all()
        lengths = batch["events"]["length"]
        mask = batch["events"]["mask"]
        assert torch.equal(mask.sum(dim=1), lengths)
        assert (batch["events"]["concept"][~mask] == 0).all()
        assert batch["target"]["value"].shape == (len(rows),)
        assert batch["target"]["observed"].dtype == torch.bool
        censored = rows.index("P-04@d95")
        assert not batch["target"]["observed"][censored]
        assert not batch["annotated"]["ct"][censored].any()
        ct = batch["image_time"]["ct"]
        later = rows.index("P-02@d95")
        assert ct["start_known"][later] and (ct["start_age_h"][later] > 0).all()
        assert ct["available_known"][later]

    def test_S9_1_no_future_reaches_an_input(self, worked: dict[str, Any]):
        ds = ClinicalTaskDataset(
            worked["task"], partition="train", documents=HashingTextEncoder(dim=8)
        )
        early = ds.rows.index(next(r for r in ds.rows if r.row_id == "P-02@24h"))
        item = ds[early]
        assert item["meta"]["event_ids"] == ["lab0", "ct0", "report0_v1"]
        assert item["meta"]["document_events"] == ["report0_v1"]
        assert item["meta"]["visits"]["ct"]["image_id"] == "CT_tp0"
        # The lesion masks were drawn on day 92: no 24 h crop is centred on
        # them, though they supervise the slot as labels.
        assert item["meta"]["visits"]["ct"]["roi"] == "center_fallback"
        assert item["annotated"]["ct"].all()
        events = item["events"]
        assert events["available_known"].all()
        assert (events["available_age_h"] >= 0).all()
        assert (events["start_age_h"][events["start_known"]] >= 0).all()
        late = ds[ds.rows.index(next(r for r in ds.rows if r.row_id == "P-02@d95"))]
        assert late["meta"]["visits"]["ct"]["roi"] == "eligible_instances"
        assert "report0_v2" in late["meta"]["document_events"]

    def test_S7_cached_and_encoded_documents_agree(
        self, worked: dict[str, Any], tmp_path
    ):
        task = TaskManifest.load(worked["task"])
        encoder = HashingTextEncoder(dim=8)
        cache = worked["out"] / "docs.medh5cache"
        assert build_document_cache(task, cache, encoder).ok
        live = ClinicalTaskDataset(task, partition="train", documents=encoder)
        cached = ClinicalTaskDataset(task, partition="train", documents=cache)
        for i in range(len(live)):
            a, b = live[i]["documents"]["features"], cached[i]["documents"]["features"]
            assert torch.allclose(a, b)

    def test_S7_a_patient_cache_must_be_admissible(self, worked: dict[str, Any]):
        task = TaskManifest.load(worked["task"])
        report = task.preflight()
        path = worked["out"] / "rows.medh5cache"
        with CacheWriter(
            path,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [3]},
            task=task,
        ) as w:
            for row in report.eligible():
                versions = [e.event_id for e in row.events]
                if row.row_id == "P-02@24h":
                    versions.append("report0_v2")  # a whole-history embedding
                w.add(
                    row.row_id.replace("@", "_"),
                    np.full(3, row.cutoff_us / DAY, np.float32),
                    sources=list(row.sources),
                    row_id=row.row_id,
                    row_fingerprint=row.fingerprint,
                    cutoff_us=row.cutoff_us,
                    event_versions=versions,
                )
        with pytest.raises(MEDH5ValidationError) as caught:
            ClinicalTaskDataset(task, partition="train", row_features=path)
        assert caught.value.code == "T406"

    def test_N11_a_cache_replaced_after_validation_is_refused(
        self, worked: dict[str, Any], tmp_path: Path
    ):
        """Construction validated a cache and kept only its path; the first
        read, a worker and an unpickled copy opened whatever the path named by
        then, and a replacement --- valid on its own --- served its features
        with this row's image (N11 of the round-3 audit).  Every handle is held
        to the validated manifest's checksum; the same manifest rebuilt is the
        same cache."""
        import pickle

        task = TaskManifest.load(worked["task"])
        report = task.preflight()

        def build(path: Path, scale: float) -> None:
            with CacheWriter(
                path,
                level="patient",
                encoder={"name": "fixture", "revision": "1"},
                output={"dtype": "float32", "shape": [3]},
                task=task,
            ) as w:
                for row in report.eligible():
                    w.add(
                        row.row_id.replace("@", "_"),
                        np.full(3, scale * row.cutoff_us / DAY, np.float32),
                        sources=list(row.sources),
                        row_id=row.row_id,
                        row_fingerprint=row.fingerprint,
                        cutoff_us=row.cutoff_us,
                        event_versions=[e.event_id for e in row.events],
                    )

        path, staged = tmp_path / "rows.medh5cache", tmp_path / "staged.medh5cache"
        build(path, 1.0)
        # A rebuild that writes the same manifest is the cache validated.
        same = ClinicalTaskDataset(task, partition="train", row_features=path)
        build(staged, 1.0)
        os.replace(staged, path)
        first = same[0]["row_feature"]
        assert torch.equal(pickle.loads(pickle.dumps(same))[0]["row_feature"], first)
        del same  # its handle, which Windows would not let a replace pass
        # A cache valid for this task, but not the one validated.
        replaced = ClinicalTaskDataset(task, partition="train", row_features=path)
        shipped = pickle.dumps(replaced)
        build(staged, 7.0)
        assert validate_cache(staged, base=task.base, task=task).ok
        os.replace(staged, path)
        for ds in (replaced, pickle.loads(shipped)):
            with pytest.raises(MEDH5ValidationError, match="not the cache") as caught:
                ds[0]
            assert caught.value.code == "T404"
        if os.name != "nt":  # Windows cannot replace a file another handle holds
            build(staged, 1.0)
            os.replace(staged, path)
            reading = ClinicalTaskDataset(task, partition="train", row_features=path)
            held = reading[0]["row_feature"]
            build(staged, 7.0)
            os.replace(staged, path)
            # The handle that was checked keeps serving what was checked.
            assert torch.equal(reading[0]["row_feature"], held)

    def test_S7_3_vocabularies_are_fitted_on_the_training_partition(
        self, worked: dict[str, Any], tmp_path: Path
    ):
        task = TaskManifest.load(worked["task"])
        train = ClinicalTaskDataset(task, partition="train")
        vocab = train.concepts
        assert vocab.fitted_on is not None and vocab.fitted_on["partition"] == "train"
        assert vocab.index("no|such|concept") == UNKNOWN
        lab = concept_of(train.rows[0].events[0])
        assert vocab.index(lab) >= 2 and lab in vocab.stats
        assert len(vocab) == len(vocab.concepts) + 2
        saved = vocab.save(tmp_path / "vocab.json")
        assert ConceptVocabulary.load(saved) == vocab and vocab.digest.startswith(
            "sha256:"
        )
        val = ClinicalTaskDataset(task, partition="val", concepts=vocab)
        assert len(val) == 3
        wrong = ConceptVocabulary.fit(task, partition="val")
        with pytest.raises(MEDH5ValidationError) as caught:
            ClinicalTaskDataset(task, partition="train", concepts=wrong)
        assert caught.value.code == "T405"
        with pytest.raises(MEDH5ValidationError):
            ClinicalTaskDataset(task, partition="test")

    def test_S4_a_task_with_findings_is_refused_unless_asked(
        self, worked: dict[str, Any], tmp_path: Path
    ):
        import shutil

        root = tmp_path / "copy"
        shutil.copytree(worked["out"], root)
        with medh5.amend(root / "P-03.medh5") as w:
            w.add_event(
                medh5.clinical.Event(
                    "late",
                    "late",
                    "other",
                    "point",
                    "final",
                    effective_start_us=400 * DAY,
                    available_us=400 * DAY,
                )
            )
        with pytest.raises(MEDH5ValidationError) as caught:
            ClinicalTaskDataset(root / "progression.task.json", partition="train")
        assert caught.value.code == "T302"
        partial = ClinicalTaskDataset(
            root / "progression.task.json", partition="train", strict=False
        )
        assert {r.subject_id for r in partial.rows} == {"P-01", "P-02", "P-04"}

    def test_S6_workers_produce_the_same_batches(self, worked: dict[str, Any]):
        ds = ClinicalTaskDataset(worked["task"], partition="train")
        alone = next(iter(DataLoader(ds, batch_size=8, collate_fn=collate_clinical)))
        workers = DataLoader(
            ds,
            batch_size=8,
            num_workers=2,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_clinical,
        )
        together = next(iter(workers))
        for key in ("ct", "mr"):
            assert torch.equal(alone["images"][key], together["images"][key])
        assert torch.equal(alone["events"]["concept"], together["events"]["concept"])


def timeline(path: Path) -> None:
    """The kit's history and the time shapes 1.1 §5.2 distinguishes: a
    day-precision diagnosis, a course with a known end and one ongoing, a
    static fact, an event whose time is unknown, a value known only as a bound,
    a result missing for a reason, two events whose times overlap, and a plan."""
    from medh5.clinical import Event
    from tests.kits import History

    def coded(event_id: str, kind: str, temporal: str, **f: Any) -> Event:
        status = f.pop("status", "final")
        return Event(
            event_id,
            event_id,
            kind,
            temporal,
            status,
            code_system="org.example",
            code=event_id,
            **f,
        )

    History.write(
        path,
        events=[
            coded(
                "dx",
                "diagnosis",
                "point",
                effective_start_us=(10 * DAY, 11 * DAY - 1),
                available_us=11 * DAY,
            ),
            coded(
                "course",
                "medication_administration",
                "interval",
                effective_start_us=(2 * DAY, 2 * DAY + HOUR),
                effective_end_us=(12 * DAY, 13 * DAY),
                available_us=13 * DAY,
            ),
            coded(
                "course_open",
                "medication_administration",
                "interval",
                effective_start_us=15 * DAY,
                available_us=15 * DAY + HOUR,
            ),
            coded("sex", "other", "static", available_us=0, value_text="female"),
            coded(
                "unknown_time",
                "other",
                "unknown",
                available_us=5 * DAY,
                value_text="smoker",
            ),
            coded(
                "below",
                "observation",
                "point",
                effective_start_us=8 * DAY,
                available_us=8 * DAY + HOUR,
                value_num=5.0,
                value_comparator="lt",
                unit="mg/L",
            ),
            coded(
                "missing",
                "observation",
                "point",
                effective_start_us=9 * DAY,
                available_us=9 * DAY + HOUR,
                missing_reason="not_done",
            ),
            coded(
                "tie_a",
                "procedure",
                "point",
                effective_start_us=(16 * DAY, 17 * DAY),
                available_us=17 * DAY,
            ),
            coded(
                "tie_b",
                "procedure",
                "point",
                effective_start_us=16 * DAY + 12 * HOUR,
                available_us=17 * DAY,
            ),
            coded(
                "plan",
                "medication_order",
                "point",
                status="planned",
                effective_start_us=30 * DAY,
                available_us=14 * DAY,
            ),
        ],
    )


def timeline_task(tmp_path: Path, **policy: Any) -> TaskManifest:
    from medh5.task import Slot, SourceRef

    path = tmp_path / "timeline.medh5"
    if not path.exists():
        timeline(path)
    task = TaskManifest.new(
        "timeline",
        "1",
        identity_namespace="site",
        slots=[Slot("ct", "CT")],
        policy=policy or None,
        base=tmp_path,
    )
    task.add_subject("P-01", [SourceRef.pin(path, uri=path.name)])
    task.add_row("d20", "P-01", 20 * DAY)
    task.add_row("d3", "P-01", 3 * DAY)
    return task


class TestTimeInBatches:
    """Contract §6 and 1.1 §5.2: a batch keeps every time's bounds, says which
    times are unknown, and tells a static fact, a plan and a tie apart from an
    observed order --- and nothing after the cutoff reaches it."""

    def item(
        self, task: TaskManifest, row_id: str = "d20", **options: Any
    ) -> dict[str, Any]:
        ds = ClinicalTaskDataset(task, **options)
        index = [r.row_id for r in ds.rows].index(row_id)
        return ds[index]

    def test_S5_2_uncertain_times_keep_their_bounds(self, tmp_path: Path):
        item = self.item(timeline_task(tmp_path))
        ev, at = item["events"], item["meta"]["event_ids"].index
        lo, hi = ev["start_age_h"][at("dx")].tolist()
        assert lo == pytest.approx((20 * DAY - (11 * DAY - 1)) / HOUR, abs=1e-3)
        assert hi == pytest.approx(10 * 24.0, abs=1e-3)
        assert hi - lo == pytest.approx(24.0, abs=1e-3), "a day stays a day"
        assert ev["start_age_h"][at("ct0")].tolist() == [480.0, 480.0], (
            "an instant is exact"
        )
        # A course with a recorded end, and one still running.
        assert ev["end_known"][at("course")]
        assert ev["end_age_h"][at("course")].tolist() == [7 * 24.0, 8 * 24.0]
        assert not ev["end_known"][at("course_open")]
        assert ev["end_age_h"][at("course_open")].tolist() == [0.0, 0.0]
        assert not ev["end_known"][at("dx")], "a point has no end"
        # Availability is a time of its own, and every input's is known.
        assert ev["available_age_h"][at("dx")].tolist() == [9 * 24.0, 9 * 24.0]
        assert ev["available_known"].all()

    def test_S5_2_static_unknown_and_missing_are_not_times_or_values(
        self, tmp_path: Path
    ):
        from medh5.clinical import COMPARATORS, TEMPORAL_TYPES

        item = self.item(timeline_task(tmp_path))
        ev, ids = item["events"], item["meta"]["event_ids"]
        at = ids.index
        assert ev["temporal_type"][at("sex")] == TEMPORAL_TYPES.index("static") + 1
        assert not ev["start_known"][at("sex")] and ev["available_known"][at("sex")]
        assert "unknown_time" not in ids, (
            "strict selection never orders an unknown time"
        )
        assert ev["comparator"][at("below")] == COMPARATORS.index("lt") + 1
        assert ev["comparator"][at("lab0")] == COMPARATORS.index("eq") + 1
        assert ev["has_value"][at("below")] and ev["has_value"][at("lab0")]
        assert ev["missing"][at("missing")] and not ev["has_value"][at("missing")]
        assert ev["comparator"][at("missing")] == 0 and ev["value"][at("missing")] == 0
        assert not ev["missing"][at("lab0")]
        # Ordered by availability, the unknown-time event is read --- as
        # unknown, which is not static.
        by_availability = self.item(timeline_task(tmp_path, order_by="available"))
        ev, at = by_availability["events"], by_availability["meta"]["event_ids"].index
        assert (
            ev["temporal_type"][at("unknown_time")]
            == TEMPORAL_TYPES.index("unknown") + 1
        )
        assert not ev["start_known"][at("unknown_time")]
        assert ev["temporal_type"][at("sex")] == TEMPORAL_TYPES.index("static") + 1

    def test_S9_1_ties_and_plans_are_said_not_invented(self, tmp_path: Path):
        item = self.item(timeline_task(tmp_path))
        ev, ids = item["events"], item["meta"]["event_ids"]
        at = ids.index
        assert ev["tie_group"][at("tie_a")] == ev["tie_group"][at("tie_b")]
        assert ev["tie_group"][at("tie_a")] != ev["tie_group"][at("course_open")]
        assert "plan" not in ids and not ev["plan"].any()
        assert (ev["start_age_h"] >= 0).all()
        planned = self.item(timeline_task(tmp_path, plans=True))
        ev, at = planned["events"], planned["meta"]["event_ids"].index
        assert ev["plan"][at("plan")] and ev["plan"].sum() == 1
        assert ev["start_age_h"][at("plan")].tolist() == [-240.0, -240.0], (
            "a plan's start is ahead"
        )
        assert ev["available_age_h"][at("plan")].tolist() == [144.0, 144.0], (
            "but it was known"
        )

    def test_S6_padding_is_never_a_time(self, tmp_path: Path):
        ds = ClinicalTaskDataset(timeline_task(tmp_path))
        batch = collate_clinical([ds[i] for i in range(len(ds))])
        ev = batch["events"]
        assert ev["start_age_h"].shape == (*ev["mask"].shape, 2)
        assert ev["mask"][0].sum() != ev["mask"][1].sum(), (
            "two histories of different lengths"
        )
        for key in ("start_known", "end_known", "available_known", "plan", "has_value"):
            assert not ev[key][~ev["mask"]].any()
        for key in ("start_age_h", "end_age_h", "available_age_h"):
            assert (ev[key][~ev["mask"]] == 0).all()
        assert batch["image_time"]["ct"]["start_age_h"].shape == (2, 2)

    def test_S9_1_nothing_after_the_cutoff_changes_an_input(self, tmp_path: Path):
        from medh5.clinical import Document, Event, Link

        task = timeline_task(tmp_path)
        encoder = HashingTextEncoder(dim=8)
        before = self.item(task, "d20", documents=encoder)
        path = tmp_path / "timeline.medh5"
        with medh5.amend(path) as w:
            # A later lab, a correction learnt after the cutoff, and a report
            # written after it: all definitely after day 20.
            w.add_event(
                Event(
                    "late_lab",
                    "late_lab",
                    "observation",
                    "point",
                    "final",
                    effective_start_us=25 * DAY,
                    available_us=25 * DAY,
                    code_system="org.example",
                    code="late_lab",
                    value_num=9.0,
                    unit="1",
                )
            )
            w.add_event(
                Event(
                    "below_v2",
                    "below",
                    "observation",
                    "point",
                    "amended",
                    effective_start_us=8 * DAY,
                    available_us=21 * DAY,
                    code_system="org.example",
                    code="below",
                    value_num=4.0,
                    unit="mg/L",
                )
            )
            w.add_link(
                Link.between(("event", "below_v2"), "supersedes", ("event", "below"))
            )
            w.add_event(
                Event(
                    "late_note",
                    "late_note",
                    "document",
                    "point",
                    "final",
                    effective_start_us=19 * DAY,
                    available_us=22 * DAY,
                )
            )
            w.add_document(Document("late_note_text", "Written after the cutoff."))
            w.add_link(
                Link.between(
                    ("event", "late_note"), "describes", ("document", "late_note_text")
                )
            )
        stale = task.preflight()
        assert "T302" in {f.code for f in stale.findings}
        repinned = timeline_task(tmp_path)  # pins the amended sample
        after = self.item(repinned, "d20", documents=encoder)
        assert after["meta"]["event_ids"] == before["meta"]["event_ids"]
        for key, value in before["events"].items():
            assert torch.equal(after["events"][key], value), key
        for key, value in before["documents"].items():
            assert torch.equal(after["documents"][key], value), key
        assert torch.equal(after["images"]["ct"], before["images"]["ct"])
        # At a later cutoff the same additions are inputs.
        repinned.add_row("d26", "P-01", 26 * DAY)
        late = self.item(repinned, "d26", documents=encoder)
        assert {"late_lab", "below_v2", "late_note"} <= set(late["meta"]["event_ids"])
        assert "below" not in late["meta"]["event_ids"], (
            "the revision replaces the version"
        )


class TestHandles:
    def test_S2_members_are_cached_by_path_and_key(self, tmp_path: Path):
        from medh5.collection import pack
        from tests.kits import History

        a, b = tmp_path / "a.medh5", tmp_path / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-02")
        shard = pack([a, b], tmp_path / "s.medh5c", keys=["a", "b"])
        CACHE.clear()
        first = open_cached(shard, "a")
        assert open_cached(shard, "a") is first
        second = open_cached(shard, "b")
        assert second is not first
        assert first.identity.subject_id == "P-01"
        assert second.identity.subject_id == "P-02"
        with CACHE.lease(shard, "b") as held:
            assert held is second
        CACHE.clear()

    def test_S2_a_lease_reads_the_pinned_version(self, tmp_path: Path):
        from medh5.clinical import Event
        from tests.kits import History

        path = tmp_path / "a.medh5"
        old = History.write(path)
        CACHE.clear()
        with CACHE.lease(path, content_id=old) as held:
            assert held.content_id == old
        with medh5.amend(path) as w:  # writes a new file over the old
            w.add_event(
                Event(
                    "x",
                    "x",
                    "other",
                    "static",
                    "final",
                    available_us=0,
                    code_system="org.example",
                    code="x",
                    value_text="y",
                )
            )
        with medh5.open(path) as s:
            new = s.content_id
        assert new != old
        opens = CACHE.opens
        # The cached handle still reads the replaced inode: it is reopened.
        with CACHE.lease(path, content_id=new) as held:
            assert held.content_id == new
            assert held.clinical is not None
            assert any(e.event_id == "x" for e in held.clinical.events)
        assert CACHE.opens == opens + 1
        with (
            pytest.raises(MEDH5ValidationError) as found,
            CACHE.lease(path, content_id=old),
        ):
            pass  # pragma: no cover - refused before the block
        assert found.value.code == "T302"
        CACHE.clear()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Windows cannot replace an open file"
    )
    def test_S2_a_stale_handle_in_use_elsewhere_is_not_a_changed_source(
        self, tmp_path: Path
    ):
        from medh5.clinical import Event
        from tests.kits import History

        path = tmp_path / "a.medh5"
        old = History.write(path)
        CACHE.clear()
        with CACHE.lease(path, content_id=old) as reading:
            with medh5.amend(path) as w:  # another thread's reader keeps the old inode
                w.add_event(Event("x", "x", "other", "static", "final", available_us=0))
            with medh5.open(path) as s:
                new = s.content_id
            # The file is the version the row pins: this lease reads it, on a
            # handle of its own, while the other keeps reading the old one.
            with CACHE.lease(path, content_id=new) as held:
                assert held is not reading and held.content_id == new
                assert reading.content_id == old and reading.clinical is not None
            assert not held.is_open
        # Once nothing reads the old handle, the cache replaces it.
        with CACHE.lease(path, content_id=new) as held:
            assert held.content_id == new
        with CACHE.lease(path, content_id=new) as again:
            assert again is held
        CACHE.clear()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX")
    def test_S2_a_forked_worker_abandons_its_parents_cache(self, tmp_path: Path):
        from medh5.cache import FeatureCache
        from medh5.task import SourceRef
        from medh5.torch.clinical import _LazyCache
        from tests.kits import History

        a = tmp_path / "a.medh5"
        History.write(a)
        task = TaskManifest.new("t", "1", identity_namespace="n", base=tmp_path)
        task.add_subject("P-01", [SourceRef.pin(a, uri="a.medh5")])
        path = tmp_path / "c.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        with FeatureCache.open(path) as cache:
            digest = cache.manifest_digest
        lazy = _LazyCache(path, digest=digest, level="event")
        assert isinstance(lazy.get(), FeatureCache)
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child's exit code is the assertion
            try:
                ok = lazy.get().path == str(path) and lazy._pid == os.getpid()
            except Exception:
                ok = False
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert lazy.get().path == str(path)  # the parent's handle still works


def test_OBS01_ages_do_not_wrap_at_the_ends_of_the_clock():
    """An age was an int64 difference, which wrapped for timestamps near the
    ends of the clock (OBS-01 of the round-3 audit)."""
    from medh5.torch.clinical import _hours_before

    ages = _hours_before(2**62, np.array([-(2**62), 2**62 - HOUR], dtype=np.int64))
    assert ages[0] == pytest.approx(2.0**63 / HOUR) and ages[0] > 0
    assert ages[1] == pytest.approx(1.0)


class TestExamples:
    def test_the_collection_member_example(self, tmp_path: Path):
        found = example("clinical_collection").main(tmp_path, workers=2)
        assert found["before reconciling"] == ["T305"]
        assert found["preflight"] == {"eligible": 3, "excluded": 1}
        assert found["ct image"] == ["CT_tp0", "CT_tp1", "CT_tp0"]
        assert found["ct source"][0] != found["ct source"][1]
        assert found["observed"] == [True, True, True]

    def test_the_cache_invalidation_example(self, tmp_path: Path):
        found = example("clinical_cache").main(tmp_path)
        assert found["fresh"] == ([], [])
        assert found["whole history"] == ["T406"]
        assert found["fitted on val"] == ["T405"]
        assert found["corrupted bytes"] == (["T402"], 1, 0)
        assert found["amended source"][0] == ["T403"]
        assert found["task after amend"] == ["T302"]
        assert found["repinned task"] == ([], True, False, False, True)
        assert found["rebuilt"] == []

    def test_the_benchmark_runs(self, tmp_path: Path):
        results = example("bench_clinical").main(
            [
                "--subjects",
                "2",
                "--events",
                "10",
                "--shape",
                "8",
                "16",
                "16",
                "--repeats",
                "2",
                "--workers",
                "0",
                "--out",
                str(tmp_path / "bench"),
                "--json",
                str(tmp_path / "bench.json"),
            ]
        )
        assert results["rows"] == {"eligible": 4}
        assert results["conditions"]["subjects"] == 2
        assert (tmp_path / "bench.json").exists()
        assert set(results["window_read_ms"]) == {"1.1", "1.0 projection"}
        assert results["cache_entries"] == 4


def test_the_preflight_benchmark_runs(tmp_path: Path):
    """The large-cohort benchmark, at a size a test can afford."""
    results = example("bench_preflight").main(
        [
            "--subjects",
            "2",
            "--events",
            "60",
            "--documents",
            "4",
            "--cutoffs",
            "3",
            "--visits",
            "2",
            "--items",
            "4",
            "--out",
            str(tmp_path / "cohort"),
            "--json",
            str(tmp_path / "bench.json"),
        ]
    )
    assert results["conditions"]["rows"] == 6
    for what in ("preflight", "dataset", "items"):
        assert results[what]["ok"] and results[what]["rows"] == 6
    assert results["items"]["items_per_s"] > 0
    assert (tmp_path / "bench.json").exists()


def test_the_cutoff_in_hours_reads_the_preliminary_report(tmp_path: Path):
    """1.1 §9.3, from the package's front door."""
    from tests.kits import History

    path = tmp_path / "h.medh5"
    History.write(path)
    with medh5.open(path) as s:
        assert s.clinical is not None
        assert s.clinical.select(24 * HOUR).event_ids == ["lab0", "ct0", "rep_v1"]


class TestAdmissibility:
    """What a dataset is built from: this task's own preflight, and feature
    caches of the level its role needs, fitted on this task's training split
    (audit B05, B06)."""

    @staticmethod
    def _tasks(tmp_path: Path) -> tuple[TaskManifest, TaskManifest]:
        """One cohort as two instances of one task definition, the split
        swapped: A trains and B validates, then the other way round."""
        from medh5.clinical import Event
        from medh5.task import Slot, SourceRef
        from tests.kits import History

        paths = {}
        for subject in ("P-A", "P-B"):
            paths[subject] = tmp_path / f"{subject}.medh5"
            only = Event(
                f"x{subject}",
                f"x{subject}",
                "observation",
                "point",
                "final",
                effective_start_us=-30 * HOUR,
                available_us=-29 * HOUR,
                code_system="http://loinc.org",
                code=f"only-{subject}",
                value_num=5.0,
                unit="mg/dL",
            )
            History.write(paths[subject], subject_id=subject, events=[only])

        def task(train: str, val: str) -> TaskManifest:
            t = TaskManifest.new(
                "t",
                "1",
                identity_namespace="site",
                slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8))],
                split=("fold-0", ["train", "val"]),
                base=tmp_path,
            )
            for subject, part in ((train, "train"), (val, "val")):
                source = SourceRef.pin(paths[subject], uri=paths[subject].name)
                t.add_subject(subject, [source], partition=part)
                t.add_row(f"{subject}@24h", subject, 24 * HOUR)
            return t

        return task("P-A", "P-B"), task("P-B", "P-A")

    def test_B05_a_preflight_of_another_split_is_refused(self, tmp_path: Path):
        """The definition fingerprint is shared by both instances; a preflight
        of the other one put this task's validation subject in its training
        rows, and the vocabulary fitted on it recorded this task's split."""
        current, swapped = self._tasks(tmp_path)
        assert current.task_fingerprint == swapped.task_fingerprint
        assert current.manifest_fingerprint != swapped.manifest_fingerprint
        other = swapped.preflight()
        for build in (
            lambda: ClinicalTaskDataset(current, partition="train", preflight=other),
            lambda: ConceptVocabulary.fit(current, other),
        ):
            with pytest.raises(
                MEDH5ValidationError, match="another instance"
            ) as caught:
                build()
            assert caught.value.code == "T404"
        own = ClinicalTaskDataset(
            current, partition="train", preflight=current.preflight()
        )
        assert [r.subject_id for r in own.rows] == ["P-A"]
        assert "observation|http://loinc.org|only-P-B" not in own.concepts.concepts
        assert "observation|http://loinc.org|only-P-A" in own.concepts.concepts

    def test_N05_a_vocabulary_is_held_to_the_cache_comparison(self, tmp_path: Path):
        """A vocabulary compared its task, partition and membership but not the
        split, so one fitted under another `set_id` of the same subjects passed
        (N05 of the 2.0 re-audit).  It is held to the comparison a cache is;
        another membership or the held-out partition stays refused."""
        current, swapped = self._tasks(tmp_path)
        vocabulary = ConceptVocabulary.fit(current, current.preflight())
        vocabulary.check_fitted_on(current)
        doc = current.to_json()
        doc["split"]["set_id"] = "fold-1"
        renamed = TaskManifest(doc, base=tmp_path)
        assert renamed.task_fingerprint == current.task_fingerprint
        for other, why in ((renamed, "set_id"), (swapped, "subjects_digest")):
            with pytest.raises(MEDH5ValidationError, match=why) as caught:
                vocabulary.check_fitted_on(other)
            assert caught.value.code == "T405"
        with pytest.raises(MEDH5ValidationError, match="set_id"):
            ClinicalTaskDataset(renamed, partition="train", concepts=vocabulary)

    def test_B06_a_document_cache_is_event_level_and_fitted_on_training(
        self, tmp_path: Path
    ):
        """`documents=` was validated without the task --- so a cache fitted on
        the validation split was accepted --- and served any entry naming an
        event, a patient-level one included: the 24 h row read a feature of a
        later row's whole history."""
        from medh5.cache import fitted_on

        task, _ = self._tasks(tmp_path)
        report = task.preflight()
        header = {
            "encoder": {"name": "enc", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }

        def documents(path: Path, partition: str) -> Path:
            with CacheWriter(
                path, level="event", fitted_on=fitted_on(task, partition), **header
            ) as w:
                for subject in task.subjects:
                    for source in subject.sources:
                        for version in ("v1", "v2"):
                            w.add_event(
                                source,
                                f"rep_{version}",
                                np.ones(1, np.float32),
                                document_id=f"rep_text_{version}",
                            )
            return path

        val_fitted = documents(tmp_path / "val.medh5cache", "val")
        with pytest.raises(MEDH5ValidationError) as caught:
            ClinicalTaskDataset(task, partition="train", documents=val_fitted)
        assert caught.value.code == "T405"

        row = report.row("P-A@24h")
        rows = tmp_path / "rows.medh5cache"
        with CacheWriter(rows, level="patient", task=task, **header) as w:
            w.add(
                row.row_id,
                np.full(1, 999.0, np.float32),
                sources=list(row.sources),
                row_id=row.row_id,
                row_fingerprint=row.fingerprint,
                cutoff_us=row.cutoff_us,
                event_versions=[e.event_id for e in row.events],
                event_id="rep_v1",
            )
        with pytest.raises(MEDH5ValidationError, match="level 'event'") as caught:
            ClinicalTaskDataset(task, partition="train", documents=rows)
        assert caught.value.code == "T404"
        with pytest.raises(MEDH5ValidationError, match="level 'patient'"):
            ClinicalTaskDataset(task, partition="train", row_features=val_fitted)

        # Event-level features fitted on the training split serve every row.
        train_fitted = documents(tmp_path / "train.medh5cache", "train")
        dataset = ClinicalTaskDataset(task, partition="train", documents=train_fitted)
        assert dataset[0]["documents"]["features"].tolist() == [[1.0]]


class TestValuesKeepTheirMeaning:
    """A value's category and unit reach the batch (audit B07): `female` and
    `male` were the same input, and 5 mg/dL and 5 mmol/L were normalised with
    one pooled mean."""

    SEX = "observation|http://loinc.org|76689-9"
    GLUCOSE = "observation|http://loinc.org|2345-7"

    @staticmethod
    def _task(
        tmp_path: Path,
        histories: dict[str, list[Any]],
        partitions: dict[str, str] | None = None,
    ) -> TaskManifest:
        from medh5.task import Slot, SourceRef
        from tests.kits import History

        task = TaskManifest.new(
            "c",
            "1",
            identity_namespace="site",
            slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8))],
            split=None if partitions is None else ("fold-0", ["train", "val"]),
            base=tmp_path,
        )
        for subject, events in histories.items():
            path = tmp_path / f"{subject}.medh5"
            History.write(path, subject_id=subject, events=events)
            partition = None if partitions is None else partitions[subject]
            task.add_subject(
                subject, [SourceRef.pin(path, uri=path.name)], partition=partition
            )
            task.add_row(f"{subject}@24h", subject, 24 * HOUR)
        return task

    @staticmethod
    def _sex(value: str) -> Any:
        from medh5.clinical import Event

        return Event(
            "sex",
            "sex",
            "observation",
            "static",
            "final",
            available_us=-10 * HOUR,
            code_system="http://loinc.org",
            code="76689-9",
            value_text=value,
        )

    @staticmethod
    def _glucose(value: float, unit: str) -> Any:
        from medh5.clinical import Event

        return Event(
            "glu",
            "glu",
            "observation",
            "point",
            "final",
            effective_start_us=-30 * HOUR,
            available_us=-29 * HOUR,
            code_system="http://loinc.org",
            code="2345-7",
            value_num=value,
            unit=unit,
        )

    @staticmethod
    def _field(item: dict[str, Any], event_id: str, name: str) -> Any:
        return item["events"][name][item["meta"]["event_ids"].index(event_id)]

    def test_B07_categorical_values_are_inputs(self, tmp_path: Path):
        task = self._task(
            tmp_path, {"F": [self._sex("female")], "M": [self._sex("male")]}
        )
        ds = ClinicalTaskDataset(task)
        assert ds.concepts.categories[self.SEX] == ("female", "male")
        index = {
            ds.rows[i].subject_id: int(self._field(ds[i], "sex", "value_index"))
            for i in range(len(ds))
        }
        assert index["F"] != index["M"] and min(index.values()) >= 2
        assert ds.concepts.value_index(self.SEX, "not-seen") == UNKNOWN
        assert ds.concepts.value_index(self.SEX, None) == 0

    def test_B07_values_in_two_units_are_refused_at_fit(self, tmp_path: Path):
        task = self._task(
            tmp_path,
            {"A": [self._glucose(5.0, "mg/dL")], "B": [self._glucose(5.0, "mmol/L")]},
        )
        with pytest.raises(MEDH5ValidationError, match="more than one unit"):
            ConceptVocabulary.fit(task)

    def test_B07_a_value_in_another_unit_is_present_and_not_normalised(
        self, tmp_path: Path
    ):
        task = self._task(
            tmp_path,
            {
                "A": [self._glucose(90.0, "mg/dL")],
                "B": [self._glucose(110.0, "mg/dL")],
                "C": [self._glucose(5.5, "mmol/L")],
            },
            partitions={"A": "train", "B": "train", "C": "val"},
        )
        train = ClinicalTaskDataset(task, partition="train")
        assert train.concepts.units[self.GLUCOSE] == "mg/dL"
        fitted = train[0]
        assert int(self._field(fitted, "glu", "unit")) == 2
        assert float(self._field(fitted, "glu", "value")) == pytest.approx(-1.0)
        other = ClinicalTaskDataset(task, partition="val")[0]
        assert bool(self._field(other, "glu", "has_value"))
        assert int(self._field(other, "glu", "unit")) == 1
        assert float(self._field(other, "glu", "value")) == 0.0

    def test_B07_the_vocabulary_records_units_and_categories(self, tmp_path: Path):
        task = self._task(
            tmp_path,
            {
                "F": [self._sex("female"), self._glucose(5.0, "mg/dL")],
                "M": [self._sex("male"), self._glucose(7.0, "mg/dL")],
            },
        )
        vocab = ConceptVocabulary.fit(task)
        assert vocab.units[self.GLUCOSE] == "mg/dL"
        assert vocab.n_values >= 4
        again = ConceptVocabulary.from_json(vocab.to_json())
        assert again == vocab and again.digest == vocab.digest
        batch = collate_clinical(
            [ClinicalTaskDataset(task, concepts=vocab)[i] for i in range(2)]
        )
        assert batch["events"]["unit"].dtype == torch.int64
        assert batch["events"]["value_index"].shape == batch["events"]["concept"].shape


class TestF03DocumentOwnership:
    """F03 of the round-4 audit: an event-level entry was looked up by its
    event alone, so a feature filed under ``report0_v1`` that encoded the
    amended report --- written two days later --- validated clean and was
    served to every row admitting ``report0_v1``.  An entry encodes a version
    of its source and the document that version owns (task-cache-1 §7.2,
    T407); the dataset checks it before every read."""

    @staticmethod
    def _cache(path: Path, entries: list[tuple[Any, str, str | None]]) -> Path:
        with CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [8]},
        ) as w:
            for source, event_id, document_id in entries:
                w.add_event(
                    source, event_id, np.ones(8, np.float32), document_id=document_id
                )
        return path

    def test_F03_a_feature_joined_to_another_version_is_refused(
        self, worked: dict[str, Any]
    ):
        task = TaskManifest.load(worked["task"])
        source = task.subjects[0].sources[0]
        out = worked["out"]
        wrong = self._cache(
            out / "wrong.medh5cache", [(source, "report0_v1", "report0_text_v2")]
        )
        report = validate_cache(wrong, base=task.base)
        assert [f.code for f in report.findings] == ["T407"]
        assert "report0_text_v1" in str(report.findings[0])
        foreign = self._cache(out / "foreign.medh5cache", [(source, "nobody_v9", None)])
        (finding,) = validate_cache(foreign, base=task.base).findings
        assert finding.code == "T407" and "does not hold" in str(finding)
        for path in (wrong, foreign):
            with pytest.raises(MEDH5ValidationError) as caught:
                ClinicalTaskDataset(task, partition="train", documents=path)
            assert caught.value.code == "T407"
        right = self._cache(
            out / "right.medh5cache", [(source, "report0_v1", "report0_text_v1")]
        )
        assert validate_cache(right, base=task.base).ok

    def test_F03_the_dataset_checks_the_document_before_it_reads(
        self, worked: dict[str, Any]
    ):
        """An entry naming no document validates --- it may be an event's own
        feature --- but is not a document's: the dataset refuses it as one,
        on the handle it reopens lazily and after a pickle round trip."""
        import pickle

        task = TaskManifest.load(worked["task"])
        entries = []
        for subject in task.subjects:
            for source in subject.sources:
                for event_id in ("report0_v1", "report0_v2"):
                    entries.append((source, event_id, None))
        path = self._cache(worked["out"] / "unnamed.medh5cache", entries)
        assert validate_cache(path, base=task.base).ok
        dataset = ClinicalTaskDataset(task, partition="train", documents=path)
        index = next(
            i for i, row in enumerate(dataset.rows) if row.cutoff_us >= 24 * HOUR
        )
        for ds in (dataset, pickle.loads(pickle.dumps(dataset))):
            with pytest.raises(MEDH5ValidationError) as caught:
                ds[index]
            assert caught.value.code == "T407"


class TestF16FeatureShapes:
    """F16 of the round-4 audit: a row without documents took a feature of
    shape ``(0, dim[0])`` while a row with them took the encoder's whole
    shape, so a batch collated in one order and raised in the other."""

    class Matrix:
        """An encoder whose feature is a matrix."""

        dim = (2, 3)

        def encode(self, text: str) -> Any:
            return np.full((2, 3), len(text), np.float32)

    def test_F16_rows_with_and_without_documents_collate_in_either_order(
        self, tmp_path: Path
    ):
        task = timeline_task(tmp_path)
        task.add_row("h1", "P-01", HOUR)  # before the first report
        dataset = ClinicalTaskDataset(task, documents=self.Matrix())
        ids = [r.row_id for r in dataset.rows]
        empty, full = dataset[ids.index("h1")], dataset[ids.index("d20")]
        assert tuple(empty["documents"]["features"].shape) == (0, 2, 3)
        assert tuple(full["documents"]["features"].shape) == (1, 2, 3)
        for batch in ([empty, full], [full, empty], [empty, empty], [full, full]):
            docs = collate_clinical(batch)["documents"]
            assert tuple(docs["features"].shape[2:]) == (2, 3)
            lengths = [len(item["documents"]["features"]) for item in batch]
            assert docs["length"].tolist() == lengths
            assert docs["mask"].sum(dim=1).tolist() == lengths
        # An int `dim` is a vector.
        vectors = ClinicalTaskDataset(task, documents=HashingTextEncoder(dim=4))
        assert tuple(vectors[ids.index("h1")]["documents"]["features"].shape) == (0, 4)

    def test_F16_rows_that_disagree_are_refused(self, tmp_path: Path):
        task = timeline_task(tmp_path)
        dataset = ClinicalTaskDataset(task, documents=self.Matrix())
        full = dataset[0]
        other = dict(full, documents=dict(full["documents"]))
        other["documents"]["features"] = torch.ones((1, 4))
        with pytest.raises(ValueError, match="cannot collate 'features'"):
            collate_clinical([full, other])


class TestF17ConceptTokens:
    """F17 of the round-4 audit: a concept token joined its fields with `|`
    unescaped, so system ``alpha|beta`` with code ``gamma`` and system
    ``alpha`` with code ``beta|gamma`` were one input."""

    def test_F17_delimiters_in_codes_keep_concepts_apart(self, tmp_path: Path):
        from medh5.clinical import Event
        from medh5.task import SourceRef
        from tests.kits import History

        def coded(event_id: str, system: str, code: str) -> Event:
            return Event(
                event_id,
                event_id,
                "observation",
                "point",
                "final",
                effective_start_us=-HOUR,
                available_us=-HOUR,
                code_system=system,
                code=code,
                value_text="seen",
            )

        path = tmp_path / "codes.medh5"
        History.write(
            path,
            events=[
                coded("a", "alpha|beta", "gamma"),
                coded("b", "alpha", "beta|gamma"),
                coded("c", "système\\", "|código"),
            ],
        )
        task = TaskManifest.new("codes", "1", identity_namespace="site", base=tmp_path)
        task.add_subject("P-01", [SourceRef.pin(path, uri=path.name)])
        task.add_row("d1", "P-01", DAY)
        dataset = ClinicalTaskDataset(task)
        item = dataset[0]
        at = item["meta"]["event_ids"].index
        concept = item["events"]["concept"]
        assert len({int(concept[at(e)]) for e in ("a", "b", "c")}) == 3
        tokens = {concept_of(e) for e in dataset.rows[0].events}
        assert {
            "observation|alpha\\|beta|gamma",
            "observation|alpha|beta\\|gamma",
            "observation|système\\\\|\\|código",
        } <= tokens
        saved = dataset.concepts.save(tmp_path / "vocab.json")
        assert ConceptVocabulary.load(saved) == dataset.concepts
