"""The task-and-cache contract (``docs/spec/task-cache-1.md``), through Python and
the command line.

Fixtures are written by the public writer (``tests.kits.History``); the
companion files by the public task and cache APIs.  Test names cite the
contract's clause.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.cache import (
    CacheWriter,
    FeatureCache,
    HashingTextEncoder,
    build_document_cache,
    event_entry_id,
    fitted_on,
    open_cache,
    validate_cache,
)
from medh5.cli import EXIT_ERROR, EXIT_OK, main
from medh5.clinical import DAY, HOUR, Clock, Event, Link
from medh5.collection import pack
from medh5.errors import MEDH5ValidationError
from medh5.task import (
    CODES,
    Finding,
    Slot,
    SourceRef,
    Target,
    TaskManifest,
    preflight,
    schema_text,
)
from tests.kits import History

TARGET = Target(
    "progression",
    "1",
    History.RECIST,
    "overall_response",
    positive=("PD",),
    negative=("SD", "PR", "CR"),
    horizon_us=180 * DAY,
    min_follow_up_us=60 * DAY,
)


def cohort(root: Path, responses: dict[str, str] | None = None) -> dict[str, Path]:
    responses = responses or {"P-01": "PD", "P-02": "SD", "P-03": "SD"}
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for subject, response in responses.items():
        paths[subject] = root / f"{subject}.medh5"
        History.write(paths[subject], subject_id=subject, response=response)
    return paths


def task_for(root: Path, paths: dict[str, Path], **options: Any) -> TaskManifest:
    task = TaskManifest.new(
        "progression",
        "1",
        identity_namespace="site",
        slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8), classes=(3,))],
        target=TARGET,
        split=("fold-0", ["train", "val"]),
        base=root,
        **options,
    )
    for i, (subject, path) in enumerate(sorted(paths.items())):
        task.add_subject(
            subject,
            [SourceRef.pin(path, uri=path.name)],
            partition="val" if i == len(paths) - 1 else "train",
        )
        task.add_row(f"{subject}@24h", subject, 24 * HOUR)
        task.add_row(f"{subject}@d95", subject, 95 * DAY)
    return task


@pytest.fixture
def setup(tmp_path: Path) -> tuple[TaskManifest, dict[str, Path]]:
    paths = cohort(tmp_path / "cohort")
    return task_for(tmp_path / "cohort", paths), paths


def found(findings: Any) -> list[str]:
    return sorted({f.code for f in findings})


class TestManifest:
    def test_S3_1_build_save_load(self, setup, tmp_path: Path):
        task, _ = setup
        path = task.save(tmp_path / "cohort" / "t.task.json")
        loaded = TaskManifest.load(path)
        assert loaded.declared_fingerprint == task.manifest_fingerprint
        assert loaded.to_json()["subjects"] == task.to_json()["subjects"]
        assert loaded.task_id == "progression" and loaded.task_version == "1"
        assert loaded.identity_namespace == "site"
        assert loaded.policy.selection == "strict_prospective"
        assert loaded.slots[0] == Slot("ct", "CT", True, (4, 8, 8), "center", (3,))
        assert loaded.target == TARGET
        assert loaded.split == ("fold-0", ("train", "val"))
        assert loaded.training_partition == "train"
        assert [s.subject_id for s in loaded.subjects] == ["P-01", "P-02", "P-03"]
        assert len(loaded.rows) == 6 and loaded.partition_of("P-03") == "val"
        assert loaded.validate() == []
        assert "TaskManifest('progression'" in repr(loaded)
        assert '"medh5.task/1"' in schema_text()
        assert set(CODES) >= {"T101", "T302", "T406"}

    def test_S3_2_the_task_fingerprint_is_the_definition(self, setup):
        task, _ = setup
        spelled = task.to_json()
        # Defaults written out are the same task as defaults left out.
        terse = {k: v for k, v in spelled.items() if k != "policy"}
        assert TaskManifest(terse).task_fingerprint == task.task_fingerprint
        moved = dict(spelled, rows=spelled["rows"][:2])
        assert TaskManifest(moved).task_fingerprint == task.task_fingerprint
        assert TaskManifest(moved).manifest_fingerprint != task.manifest_fingerprint
        other = dict(spelled, policy={"selection": "latest_provable"})
        assert TaskManifest(other).task_fingerprint != task.task_fingerprint
        assert task.row_fingerprint("P-01@24h") != task.row_fingerprint("P-01@d95")
        assert task.subjects_digest("train") != task.subjects_digest("val")

    def test_S3_2_a_wrong_declared_fingerprint_is_T103(self, setup):
        task, _ = setup
        doc = dict(task.to_json(), fingerprint="sha256:" + "0" * 64)
        assert found(TaskManifest(doc).validate()) == ["T103"]

    def test_S3_3_splits_are_by_subject(self, setup):
        task, paths = setup
        doc = task.to_json()
        doc["subjects"][0]["partition"] = "test"
        doc["subjects"][1].pop("partition")
        doc["subjects"].append(dict(doc["subjects"][2]))
        doc["rows"].append({"row_id": "x", "subject_id": "P-99", "cutoff_us": 0})
        doc["rows"].append(dict(doc["rows"][0]))
        codes = found(TaskManifest(doc).validate())
        assert {"T201", "T202", "T204"} <= set(codes)

    def test_S3_3_two_subjects_never_pin_one_sample(self, setup, tmp_path: Path):
        task, paths = setup
        copy = tmp_path / "cohort" / "copy.medh5"
        shutil.copyfile(paths["P-01"], copy)
        task.add_subject(
            "P-04", [SourceRef.pin(copy, uri=copy.name)], partition="train"
        )
        assert "T203" in found(task.validate())

    def test_S3_4_an_inconsistent_definition_is_T102(self, tmp_path: Path):
        bad = Target(
            "t", "1", "s", "c", positive=("PD",), negative=("PD",), horizon_us=1
        )
        task = TaskManifest.new(
            "t",
            "1",
            identity_namespace="n",
            slots=[Slot("ct", "CT"), Slot("ct", "MR")],
            target=bad,
        )
        assert found(task.validate()) == ["T102"]
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest.new(
                "t", "1", identity_namespace="n", policy={"keep": "middle"}
            )
        assert caught.value.code in {"T101", "T102"}

    def test_S3_1_not_json_is_T101(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest.load(path)
        assert caught.value.code == "T101"
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest({"schema": "medh5.task/1"})
        assert caught.value.code == "T101"

    def test_a_manifest_pickles(self, setup):
        task, _ = setup
        back = pickle.loads(pickle.dumps(task))
        assert back.manifest_fingerprint == task.manifest_fingerprint
        assert back.base == task.base


class TestSources:
    def test_S2_a_pin_holds_until_the_sample_changes(self, setup):
        task, paths = setup
        source = task.subjects[0].sources[0]
        assert source.check(task.base) == []
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        assert found(source.check(task.base)) == ["T302"]
        repinned = SourceRef.pin(paths["P-01"], uri=source.uri)
        assert repinned.content_id != source.content_id
        assert repinned.check(task.base) == []

    def test_S2_edited_clinical_bytes_fail_under_an_unchanged_root(self, setup):
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            data = f["clinical/documents/text/data"]
            data[1] = data[1] ^ 1
        source = task.subjects[1].sources[0]
        assert found(source.check(task.base)) == ["T302"]

    def test_S2_a_missing_source_is_T301(self, setup, tmp_path: Path):
        task, paths = setup
        paths["P-03"].unlink()
        report = task.preflight()
        assert "T301" in found(report.findings)
        assert {r.status for r in report.rows if r.subject_id == "P-03"} == {"error"}

    def test_S2_members_are_named_by_shard_and_key(self, setup, tmp_path: Path):
        task, paths = setup
        shard = pack([paths["P-01"]], tmp_path / "cohort" / "s.medh5c", keys=["P-01"])
        member = SourceRef.pin(shard, sample_key="P-01", uri=shard.name)
        assert member.locator == "s.medh5c::P-01"
        assert member.content_id == task.subjects[0].sources[0].content_id
        with member.open(task.base) as sample:
            assert sample.identity.subject_id == "P-01"
        with pytest.raises(MEDH5ValidationError, match="collection"):
            SourceRef(shard.name, member.content_id).open(task.base)
        assert SourceRef.from_json(member.to_json()) == member
        assert member.resolve() == Path(shard.name)


class TestPreflight:
    def test_S4_rows_admit_what_was_available(self, setup):
        task, _ = setup
        report = preflight(task)
        assert report.ok
        assert report.counts == {"eligible": 5, "excluded": 1}
        early = report.row("P-02@24h")
        assert [e.event_id for e in early.events] == ["lab0", "ct0", "rep_v1"]
        assert early.slots["ct"].image_id == "CT_tp0"
        assert early.slots["ct"].annotations == ()  # nothing attested them
        assert early.slots["ct"].label_annotations == ("lesions_tp0",)
        late = report.row("P-02@d95")
        assert late.slots["ct"].image_id == "CT_tp1"
        assert late.selection is not None and late.selection.certified
        assert report.eligible("val") == (
            report.row("P-03@24h"),
            report.row("P-03@d95"),
        )
        with pytest.raises(KeyError):
            report.row("nope")

    def test_S5_targets(self, setup):
        task, _ = setup
        report = task.preflight()
        assert report.row("P-01@24h").target.status == "positive"
        assert report.row("P-01@24h").target.value == 1.0
        assert report.row("P-02@24h").target.status == "negative"
        prevalent = report.row("P-01@d95")
        assert (prevalent.status, prevalent.reasons) == (
            "excluded",
            ("prevalent_target",),
        )
        censored = report.row("P-02@d95")
        assert censored.target.status == "censored" and not censored.target.observed
        assert censored.status == "eligible"

    def test_S5_censored_rows_can_be_excluded(self, tmp_path: Path):
        paths = cohort(tmp_path / "c")
        doc = task_for(tmp_path / "c", paths).to_json()
        doc["target"]["censoring"] = "exclude"
        report = TaskManifest(doc, base=tmp_path / "c").preflight()
        assert report.row("P-02@d95").reasons == ("censored",)

    def test_S4_an_uncertain_revision_is_uncertifiable(self, tmp_path: Path):
        root = tmp_path / "u"
        root.mkdir()
        path = root / "P-01.medh5"
        History.write(
            path,
            events=[
                Event(
                    "rep_v3",
                    "rep",
                    "document",
                    "point",
                    "amended",
                    effective_start_us=0,
                )
            ],
            documents=[],
            links=[
                Link.between(("event", "rep_v3"), "supersedes", ("event", "rep_v2"))
            ],
        )
        report = task_for(root, {"P-01": path}).preflight()
        row = report.row("P-01@24h")
        assert row.status == "uncertifiable"
        assert row.reasons == ("uncertain_revision:rep",)

    def test_S4_a_missing_required_slot_excludes(self, tmp_path: Path):
        paths = cohort(tmp_path / "m", {"P-01": "SD"})
        doc = task_for(tmp_path / "m", paths).to_json()
        doc["slots"][0]["modality"] = "MR"
        report = TaskManifest(doc, base=tmp_path / "m").preflight()
        assert report.row("P-01@24h").reasons == ("missing_required_slot:ct",)

    def test_S3_3_fragments_reconcile(self, tmp_path: Path):
        root = tmp_path / "f"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-01")  # the same history, exported twice
        task = TaskManifest.new(
            "t", "1", identity_namespace="n", target=TARGET, base=root
        )
        task.add_subject(
            "P-01", [SourceRef.pin(a, uri="a.medh5"), SourceRef.pin(b, uri="b.medh5")]
        )
        task.add_row("r", "P-01", 24 * HOUR)
        assert found(task.preflight().findings) == ["T305"]
        task.reconcile()
        assert len(task.subjects[0].reconciled) == 6
        assert task.preflight().ok

    def test_S3_3_identity_and_clocks(self, tmp_path: Path):
        root = tmp_path / "i"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-02", clock=Clock.relative("other", "elsewhere"))
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        # A recorded cross-site mapping is legitimate: site B calls the subject
        # P-02, and the manifest says so.  A recorded id the sample does not
        # carry is not.
        pinned_b = SourceRef.pin(b, uri="b.medh5")
        assert pinned_b.local_subject_id == "P-02"
        task.add_subject("P-01", [SourceRef.pin(a, uri="a.medh5"), pinned_b])
        # Two clocks (T304), and the shared events not yet reconciled (T305).
        assert found(task.preflight().findings) == ["T304", "T305"]
        wrong = SourceRef(
            "a.medh5", pinned_b.content_id, source_id="x", local_subject_id="P-9"
        )
        other = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        other.add_subject("P-01", [wrong])
        assert "T303" in found(other.preflight().findings)


class TestCaches:
    def test_S7_event_level_caches(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "docs.medh5cache"
        encoder = HashingTextEncoder(dim=8)
        report = build_document_cache(task, path, encoder)
        assert report.ok and report.entries == 6
        source = task.subjects[0].sources[0]
        with open_cache(path) as cache:
            assert cache.level == "event" and len(cache) == 6
            feature = cache.event_feature(source.content_id, "rep_v1")
            assert feature is not None and feature.shape == (8,)
            assert cache.event_feature(source.content_id, "lab0") is None
            entry_id = event_entry_id(source.content_id, "rep_v1")
            assert np.array_equal(cache.get(entry_id), feature)
            assert cache.header["encoder"]["name"] == HashingTextEncoder.NAME
            assert cache.row_feature("P-01@24h") is None
            assert "FeatureCache(" in repr(cache)
        assert np.array_equal(encoder.encode("a b"), encoder.encode("A, b!"))

    def test_S8_stale_corrupt_and_inadmissible_are_told_apart(
        self, setup, tmp_path: Path
    ):
        task, paths = setup
        report = task.preflight()
        path = tmp_path / "cohort" / "rows.medh5cache"
        record = fitted_on(task)
        assert record["partition"] == "train"
        with CacheWriter(
            path,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
            fitted_on=record,
        ) as w:
            for row in report.eligible("train"):
                w.add(
                    row.row_id,
                    np.zeros(2, np.float32),
                    sources=list(row.sources),
                    row_id=row.row_id,
                    cutoff_us=row.cutoff_us,
                    event_versions=[e.event_id for e in row.events],
                )
        assert validate_cache(path, task=task).ok

        # Inadmissible: an entry encoding a version its row cannot have read.
        leaky = tmp_path / "cohort" / "leaky.medh5cache"
        with CacheWriter(
            leaky,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        ) as w:
            row = report.row("P-02@24h")
            w.add(
                row.row_id,
                np.zeros(2, np.float32),
                sources=list(row.sources),
                row_id=row.row_id,
                cutoff_us=row.cutoff_us,
                event_versions=["lab0", "ct0", "rep_v2"],
            )
        assert found(validate_cache(leaky, task=task).findings) == ["T406"]

        # Another task.
        other = TaskManifest(
            dict(task.to_json(), policy={"context_us": DAY}), base=task.base
        )
        assert "T404" in found(validate_cache(path, task=other).findings)

        # Corrupt: the cache's own bytes.
        with h5py.File(path, "r+") as f:
            f["entries"][sorted(f["entries"])[0]][0] = 5.0
        corrupt = validate_cache(path)
        assert corrupt.corrupt and not corrupt.stale

        # Stale: a source changed.
        shutil.copyfile(paths["P-01"], tmp_path / "keep.medh5")
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        stale = validate_cache(path)
        assert stale.stale and "T403" in found(stale.findings)

    def test_S8_a_manifest_that_fails_its_checksum_is_T401(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "docs.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        with h5py.File(path, "r+") as f:
            f.attrs["manifest_digest"] = "sha256:" + "0" * 64
        report = validate_cache(path)
        assert report.corrupt == ("manifest",) and found(report.findings) == ["T401"]
        with pytest.raises(MEDH5ValidationError) as caught:
            FeatureCache.open(path)
        assert caught.value.code == "T401"

    def test_S7_entries_match_the_declared_output(self, tmp_path: Path, setup):
        task, _ = setup
        source = task.subjects[0].sources[0]
        path = tmp_path / "x.medh5cache"
        writer = CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            writer.add_event(source, "rep_v1", np.zeros(3, np.float32))
        assert caught.value.code == "T402"
        with pytest.raises(MEDH5ValidationError) as caught:
            writer.add("x", np.zeros(2, np.float32), sources=[])
        assert caught.value.code == "T403"
        writer.abort()
        assert not path.exists()
        with pytest.raises(MEDH5ValidationError) as caught:
            CacheWriter(
                path,
                level="patient",
                encoder={"name": "f", "revision": "1"},
                output={"dtype": "float32", "shape": [2]},
            )
        assert caught.value.code == "T404"

    def test_S7_learned_preprocessing_needs_a_split(self):
        task = TaskManifest.new("t", "1", identity_namespace="n")
        with pytest.raises(MEDH5ValidationError) as caught:
            fitted_on(task)
        assert caught.value.code == "T405"

    def test_a_closed_cache_says_so(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "d.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        cache = FeatureCache.open(path)
        cache.close()
        with pytest.raises(medh5.MEDH5FileError):
            cache.path  # noqa: B018
        cache.close()
        abandoned = FeatureCache.open(path)
        abandoned.abandon()

    def test_findings_print_their_code(self):
        finding = Finding("T302", "P-01", "changed")
        assert str(finding) == "T302 P-01: changed"
        assert "pinned" in finding.summary


class TestCommandLine:
    def test_task_and_cache_commands(self, setup, tmp_path: Path, capsys):
        task, paths = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        assert main(["task", "validate", str(manifest)]) == EXIT_OK
        assert "OK" in capsys.readouterr().out
        assert main(["task", "validate", str(manifest), "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["ok"] is True
        assert main(["task", "preflight", str(manifest)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "eligible" in out and "prevalent_target" in out
        assert main(["task", "preflight", str(manifest), "--json", "--deep"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["ok"] is True
        reconciled = tmp_path / "cohort" / "r.task.json"
        assert main(["task", "reconcile", str(manifest), "--out", str(reconciled)]) == 0
        assert "recorded 0" in capsys.readouterr().out
        cache = tmp_path / "cohort" / "d.medh5cache"
        build_document_cache(task, cache, HashingTextEncoder(dim=4))
        assert (
            main(["cache", "validate", str(cache), "--task", str(manifest)]) == EXIT_OK
        )
        assert "OK" in capsys.readouterr().out
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        assert main(["cache", "validate", str(cache), "--json"]) == EXIT_ERROR
        assert json.loads(capsys.readouterr().out)["stale"]
        assert main(["task", "preflight", str(manifest)]) == EXIT_ERROR
        assert "T302" in capsys.readouterr().out
        broken = tmp_path / "broken.json"
        broken.write_text("[]", encoding="utf-8")
        assert main(["task", "validate", str(broken)]) == EXIT_ERROR
        assert "T101" in capsys.readouterr().out

    def test_clinical_commands(self, setup, tmp_path: Path, capsys):
        _, paths = setup
        path = str(paths["P-01"])
        assert main(["clinical", "show", path]) == EXIT_OK
        out = capsys.readouterr().out
        assert "rep_v1" in out and "clock" in out
        assert main(["clinical", "show", path, "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["events"] == 6
        assert main(["clinical", "select", path, "--cutoff-hours", "24"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "certified" in out and "rep_v2" not in out
        assert (
            main(["clinical", "select", path, "--cutoff-us", "0", "--json"]) == EXIT_OK
        )
        assert json.loads(capsys.readouterr().out)["status"] == "certified"
        assert main(["clinical", "select", path]) == EXIT_ERROR
        capsys.readouterr()
        records = tmp_path / "records.json"
        assert main(["clinical", "export", path, "--out", str(records)]) == EXIT_OK
        capsys.readouterr()
        stripped = tmp_path / "imaging.medh5"
        assert main(["clinical", "strip", path, "--out", str(stripped)]) == EXIT_OK
        assert "removed 6 events" in capsys.readouterr().out
        back = tmp_path / "back.medh5"
        args = ["clinical", "augment", str(stripped), str(records), "--out", str(back)]
        assert main(args) == EXIT_OK
        assert "1.0 -> 1.1" in capsys.readouterr().out
        with medh5.open(back) as s:
            assert s.clinical is not None and len(s.clinical.events) == 6
        assert main(["clinical", "show", str(stripped)]) == EXIT_ERROR
        assert "does not declare the clinical profile" in capsys.readouterr().err
