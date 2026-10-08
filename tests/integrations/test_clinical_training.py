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
        assert batch["image_age_h"]["ct"][rows.index("P-02@d95")] > 0

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
        assert (item["events"]["age_h"] >= 0).all()
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
                    cutoff_us=row.cutoff_us,
                    event_versions=versions,
                )
        with pytest.raises(MEDH5ValidationError) as caught:
            ClinicalTaskDataset(task, partition="train", row_features=path)
        assert caught.value.code == "T406"

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
        lazy = _LazyCache(path)
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


def test_the_cutoff_in_hours_reads_the_preliminary_report(tmp_path: Path):
    """1.1 §9.3, from the package's front door."""
    from tests.kits import History

    path = tmp_path / "h.medh5"
    History.write(path)
    with medh5.open(path) as s:
        assert s.clinical is not None
        assert s.clinical.select(24 * HOUR).event_ids == ["lab0", "ct0", "rep_v1"]
