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
from medh5.clinical import DAY, HOUR, Clock, Document, Event, Link
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
from medh5.validate import validate_file
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

    def test_S2_a_repin_keeps_the_view_of_a_post_cutoff_addition(self, setup):
        """An old pin fails against a changed source; once repinned on purpose,
        an addition definitely after a row's cutoff leaves its view unchanged."""
        task, paths = setup
        before = task.preflight().row("P-01@24h")
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event(
                    "late_lab",
                    "late_lab",
                    "observation",
                    "point",
                    "final",
                    effective_start_us=300 * DAY,
                    available_us=300 * DAY,
                    code_system="local",
                    code="x",
                    value_text="high",
                )
            )
        assert found(task.preflight().findings) == ["T302"]
        doc = task.to_json()
        source = doc["subjects"][0]["sources"][0]
        source["content_id"] = SourceRef.pin(paths["P-01"]).content_id
        repinned = TaskManifest(doc, base=task.base)
        after = repinned.preflight().row("P-01@24h")
        assert repinned.preflight().ok
        assert after.events == before.events and after.slots == before.slots
        assert after.target == before.target
        assert after.fingerprint != before.fingerprint  # it pins other bytes

    def test_S2_edited_clinical_bytes_fail_under_an_unchanged_root(self, setup):
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            data = f["clinical/documents/text/data"]
            data[1] = data[1] ^ 1
        source = task.subjects[1].sources[0]
        assert found(source.check(task.base)) == ["T302"]

    def test_B03_a_dataset_the_root_does_not_cover_breaks_the_pin(self, setup):
        """`content_id` is a root over *stored* digests, so a dataset added
        without one changes no root: a planted column passed the shallow and
        the deep check and a fresh preflight while the rows read its values
        (1.1 §8, E818).  Every dataset of an attested object must carry its
        digest (task-cache-1 §2, step 2)."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                events.create_dataset(name, data=np.zeros(n, dtype="<i8"))
                events["valid"].create_dataset(name, data=np.zeros(n, dtype="u1"))
        with h5py.File(paths["P-03"], "r+") as f:
            f["images"].create_dataset("extra", data=np.zeros(4, dtype="u1"))
        for i in (1, 2):
            source = task.subjects[i].sources[0]
            assert found(source.check(task.base)) == ["T302"]
            assert found(source.check(task.base, deep=True)) == ["T302"]
        assert "carry no digest" in str(task.subjects[1].sources[0].check(task.base)[0])
        report = task.preflight()
        assert found(report.findings) == ["T302"]
        assert {r.status for r in report.rows if r.subject_id != "P-01"} == {"error"}
        assert "E818" in validate_file(paths["P-02"], level="integrity").codes

    @pytest.mark.parametrize("alias", ["aaa", "zzz"])
    def test_B03_a_root_alias_does_not_hide_an_undigested_column(self, setup, alias):
        """A walk of the root visits an object once, at its first path, and the
        pin decided what an object belongs to by that path: an undigested
        column linked first at the root, before `clinical` in name order, was
        no clinical dataset, and the shallow and deep checks and a preflight
        passed it while the rows read it (B03 of the 2.0 re-audit).  Membership
        is reached through each attested group, whatever else links to it."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                events.create_dataset(name, data=np.zeros(n, dtype="<i8"))
                events["valid"].create_dataset(name, data=np.zeros(n, dtype="u1"))
                f[f"{alias}_{name}"] = events[name]
                f[f"{alias}_valid_{name}"] = events[f"valid/{name}"]
        source = task.subjects[1].sources[0]
        for deep in (False, True):
            (finding,) = source.check(task.base, deep=deep)
            assert finding.code == "T302"
            assert "clinical/events/valid/effective_end_lo_us" in str(finding)
        report = task.preflight()
        assert found(report.findings) == ["T302"]
        assert {r.status for r in report.rows if r.subject_id == "P-02"} == {"error"}

    def test_B03_an_aliased_clinical_column_is_judged_by_its_bytes(self, setup):
        """The controls.  A later alias of a digested column changes no root and
        breaks no pin.  An earlier one is the column's first path, so its digest
        and the root are stamped there --- and the shallow check, which read
        only digests keyed under `clinical/`, passed its edited bytes.  It reads
        every object reachable through `clinical/`, by identity."""
        from medh5.integrity import dataset_digest

        task, paths = setup
        path = paths["P-02"]
        source = task.subjects[1].sources[0]
        with h5py.File(path, "r+") as f:
            f["zzz_value"] = f["clinical/events/value_num"]
        assert source.check(task.base) == []
        assert source.check(task.base, deep=True) == []
        with h5py.File(path, "r+") as f:
            del f["zzz_value"]
            f["aaa_value"] = f["clinical/events/value_num"]
        with medh5.open(path) as sample:
            digest = dataset_digest(sample.root["aaa_value"], "aaa_value")
        with h5py.File(path, "r+") as f:
            f["aaa_value"].attrs["digest"] = digest
        with medh5.open(path) as sample:
            content_id = sample.compute_content_id()
        with h5py.File(path, "r+") as f:
            f.attrs["content_id"] = content_id
        repinned = SourceRef.pin(path, uri=source.uri)
        assert repinned.check(task.base) == []
        assert repinned.check(task.base, deep=True) == []
        with h5py.File(path, "r+") as f:
            values = f["clinical/events/value_num"]
            values[0] = values[0] + 1.0
        for deep in (False, True):
            (finding,) = repinned.check(task.base, deep=deep)
            assert finding.code == "T302" and "aaa_value" in str(finding)

    def test_B03_a_column_soft_linked_elsewhere_is_outside_the_pin(self, setup):
        """Readers follow a soft link as HDF5 does; the walk that decides what
        a pin covers followed hard links only.  Optional columns linked softly
        to storage under `index/`, which the root excludes, were read by every
        selection while the pin, the deep check and a preflight passed (B03 of
        the round-3 audit).  Attestation walks what readers read."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            store = f.require_group("index").create_group("x")
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                store.create_dataset(name, data=np.full(n, 518, dtype="<i8"))
                store.create_dataset(f"valid_{name}", data=np.ones(n, dtype="u1"))
                events[name] = h5py.SoftLink(f"/index/x/{name}")
                events["valid"][name] = h5py.SoftLink(f"/index/x/valid_{name}")
        source = task.subjects[1].sources[0]
        for deep in (False, True):
            (finding,) = source.check(task.base, deep=deep)
            assert finding.code == "T302"
            assert "clinical/events/effective_end_lo_us" in str(finding)
        # 518 us ends some events before they start: the tables are invalid
        # too (T306), and that is the second finding, not the first.
        assert "T302" in found(task.preflight().findings)
        assert "E818" in validate_file(paths["P-02"], level="integrity").codes

    @staticmethod
    def _registered(path: Path) -> Path:
        shape = (6, 8, 8)
        matrix = np.eye(4)
        matrix[0, 3] = 2.0
        with medh5.create(path, codec="portable") as w:
            for tp, frame in (("tp0", "F0"), ("tp1", "F1")):
                w.add_timepoint(tp, days_from_baseline=0 if tp == "tp0" else 92)
                w.add_grid(
                    f"ct_{tp}", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint=tp,
                    frame_uid=frame,
                )  # fmt: skip
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, np.int16),
                    grid=f"ct_{tp}",
                    modality="CT",
                )
            w.add_transform(
                "t", kind="affine", from_frame="F0", to_frame="F1", matrix=matrix,
                from_grid="ct_tp0", to_grid="ct_tp1", invertible=True,
            )  # fmt: skip
        return path

    @staticmethod
    def _translate(path: Path, by: float) -> None:
        with h5py.File(path, "r+") as f:
            matrix = f["transforms/t/matrix"]
            values = matrix[...]
            values[0, 3] = by
            matrix[...] = values

    def test_B03_a_transform_aliased_under_index_is_outside_the_pin(
        self, tmp_path: Path
    ):
        """An object is listed once, under the first path that reaches it, and
        `index/` is outside the root: a transform also linked under `index/`
        before its sample was restamped and pinned was in no dataset line, and
        its translation moved from 2 to 102 mm under a matching root and an
        unchanged pin (B03 of the round-3 audit).  Covered is by identity."""
        path = self._registered(tmp_path / "reg.medh5")
        with h5py.File(path, "r+") as f:
            f.require_group("index")["alias"] = f["transforms/t/matrix"]
        with medh5.open(path) as sample:
            content_id = sample.compute_content_id()
        with h5py.File(path, "r+") as f:
            f.attrs["content_id"] = content_id
        pin = SourceRef.pin(path, uri=path.name)
        for deep in (False, True):
            (finding,) = pin.check(tmp_path, deep=deep)
            assert finding.code == "T302" and "transforms/t/matrix" in str(finding)
        self._translate(path, 102.0)
        assert found(pin.check(tmp_path, deep=True)) == ["T302"]
        report = validate_file(path, level="integrity")
        assert "/transforms/t/matrix" in {
            d.location for d in report.diagnostics if d.code == "E702"
        }

    def test_B03_an_unaliased_transform_is_held_by_its_digest(self, tmp_path: Path):
        """The control: without the alias the pin holds, and an edit fails the
        deep check and the validator."""
        path = self._registered(tmp_path / "reg.medh5")
        pin = SourceRef.pin(path, uri=path.name)
        assert pin.check(tmp_path) == [] and pin.check(tmp_path, deep=True) == []
        self._translate(path, 102.0)
        assert found(pin.check(tmp_path, deep=True)) == ["T302"]
        assert "E701" in validate_file(path, level="integrity").codes

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

    @staticmethod
    def _observation(event_id: str, record_id: str, available_us: int) -> Event:
        return Event(
            event_id,
            record_id,
            "observation",
            "point",
            "final",
            effective_start_us=-10 * HOUR,
            available_us=available_us,
            code_system="http://loinc.org",
            code="1-1",
            value_num=1.0,
        )

    def test_C07_fragments_that_contradict_are_their_subjects_finding(
        self, tmp_path: Path
    ):
        """Each fragment's revision chain was sound, but merged v0 was
        superseded twice: the whole preflight raised E816, and the cohort's
        other subject got no report."""
        root = tmp_path / "c7"
        root.mkdir()
        obs = self._observation
        for name, newer in (("f1", "v1"), ("f2", "v2")):
            History.write(
                root / f"{name}.medh5",
                subject_id="P-01",
                events=[obs("v0", "r", -9 * HOUR), obs(newer, "r", -8 * HOUR)],
                links=[Link.between(("event", newer), "supersedes", ("event", "v0"))],
            )
        History.write(root / "good.medh5", subject_id="P-02")
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        task.add_subject(
            "P-01",
            [
                SourceRef.pin(root / "f1.medh5", uri="f1.medh5"),
                SourceRef.pin(root / "f2.medh5", uri="f2.medh5"),
            ],
        )
        task.add_subject("P-02", [SourceRef.pin(root / "good.medh5", uri="good.medh5")])
        task.add_row("bad", "P-01", 24 * HOUR)
        task.add_row("good", "P-02", 24 * HOUR)
        task.reconcile()
        report = task.preflight()
        assert found(report.findings) == ["T305"]
        assert "contradict" in str(report.findings[0])
        assert report.row("bad").status == "error"
        assert report.row("good").status == "eligible"

    def test_B04_one_row_identity_reads_one_input(self, tmp_path: Path):
        """Two fragments of one subject hold the same reconciled imaging event,
        each describing its own CT --- every voxel 11 in one, 93 in the other.
        The slot kept the first fragment the manifest listed while the row
        fingerprint sorts the pins, so reordering two sources changed the
        input under one row identity (B04 of the 2.0 re-audit).  A tie goes to
        the smallest pinned `content_id`, whatever the order."""
        root = tmp_path / "b04"
        root.mkdir()
        refs = {}
        for name, value in (("a", 11), ("b", 93)):
            History.write(root / f"{name}.medh5", subject_id="P-01", fill=value)
            refs[name] = SourceRef.pin(root / f"{name}.medh5", uri=f"{name}.medh5")
        canonical = min(refs.values(), key=lambda r: r.content_id)

        def read(*names: str) -> tuple[str, float, str]:
            task = TaskManifest.new(
                "t", "1", identity_namespace="n", slots=[Slot("ct", "CT")], base=root
            )
            task.add_subject("P-01", [refs[n] for n in names])
            task.add_row("r", "P-01", 24 * HOUR)
            report = task.reconcile().preflight()
            assert report.ok, report.findings
            row = report.row("r")
            fill = row.slots["ct"]
            assert fill.fragment is not None and fill.image_id == "CT_tp0"
            source = row.sources[fill.fragment]
            with source.open(root) as sample:
                value = float(np.mean(sample.images[fill.image_id].read()))
            return row.fingerprint, value, source.content_id

        forward, backward = read("a", "b"), read("b", "a")
        assert forward == backward
        assert forward[2] == canonical.content_id
        assert forward[1] == (11.0 if canonical is refs["a"] else 93.0)

    def test_C13_a_reconciled_document_event_owns_one_text(self, tmp_path: Path):
        """The event JSON agreed while each fragment's event owned another
        text --- "NO metastasis" and "Metastasis CONFIRMED" --- and the merged
        version owned both.  A copy under another id is the same text."""
        root = tmp_path / "c13"
        root.mkdir()
        report_event = Event(
            "rep_x",
            "rep_x",
            "document",
            "point",
            "final",
            effective_start_us=0,
            available_us=2 * HOUR,
        )

        def fragment(name: str, document_id: str, text: str) -> SourceRef:
            path = root / f"{name}.medh5"
            History.write(
                path,
                subject_id="P-01",
                events=[report_event],
                documents=[Document(document_id, text)],
                links=[
                    Link.between(
                        ("event", "rep_x"), "describes", ("document", document_id)
                    )
                ],
            )
            return SourceRef.pin(path, uri=path.name, source_id=name)

        def task(*sources: SourceRef) -> TaskManifest:
            t = TaskManifest.new("t", "1", identity_namespace="n", base=root)
            t.add_subject("P-01", list(sources))
            t.add_row("r", "P-01", 24 * HOUR)
            return t.reconcile()

        contradicting = task(
            fragment("A", "doc_a", "NO metastasis."),
            fragment("B", "doc_b", "Metastasis CONFIRMED."),
        )
        report = contradicting.preflight()
        assert found(report.findings) == ["T305"]
        assert "owns another text" in str(report.findings[0])

        aliased = task(
            fragment("C", "doc_c", "NO metastasis."),
            fragment("D", "doc_d", "NO metastasis."),
        )
        report = aliased.preflight()
        assert report.ok
        row = report.row("r")
        owned = row.subject.owned_documents(row.subject.events.index("rep_x"))
        assert len(owned) == 1, owned

        # The record names who holds the event; a record of no shared event,
        # or a second record of one, is not a reconciliation.
        doc = aliased.to_json()
        records = doc["subjects"][0]["reconciled"]
        bogus = [dict(r, sources=["nowhere-1", "nowhere-2"]) for r in records]
        stale = [*records, dict(records[0], event_id="no_such_event")]
        twice = [*records, records[0]]
        for reconciled, why in (
            (bogus, "held by"),
            (stale, "no two"),
            (twice, "twice"),
        ):
            doc["subjects"][0]["reconciled"] = reconciled
            findings = TaskManifest(doc, base=root).preflight().findings
            assert "T305" in found(findings), why
            assert any(why in str(f) for f in findings), (why, findings)


class TestPreflightAtScale:
    """§4 at cohort scale: rows index their subject's history rather than
    copying it, a damaged source is a finding, and the result pickles small."""

    def many_cutoffs(self, tmp_path: Path, rows: int) -> TaskManifest:
        paths = cohort(tmp_path / f"c{rows}", {"P-01": "PD"})
        task = TaskManifest.new(
            "progression",
            "1",
            identity_namespace="site",
            slots=[Slot("ct", "CT")],
            target=TARGET,
            base=tmp_path / f"c{rows}",
        )
        task.add_subject("P-01", [SourceRef.pin(paths["P-01"], uri="P-01.medh5")])
        for k in range(rows):
            task.add_row(f"r{k}", "P-01", (k + 1) * DAY)
        return task

    def test_S4_rows_index_one_shared_history(self, tmp_path: Path):
        report = self.many_cutoffs(tmp_path, 40).preflight()
        assert report.ok, report.findings
        (subject,) = report.subjects
        assert len(subject.events) == len(History.events())
        for row in report.rows:
            assert row.subject is subject
            assert row.events == tuple(subject.events[int(i)] for i in row.selected)
            assert row.event_fragments == (0,) * len(row.selected)
            selection = row.selection
            assert selection is not None
            assert [e.event_id for e in selection.events] == [
                e.event_id for e in row.events
            ]
        late = report.row("r39")
        assert [e.event_id for e in late.events] == ["lab0", "ct0", "rep_v2"]
        assert subject.owned_documents(subject.events.index("rep_v2")) == [
            (0, "rep_text_v2")
        ]
        # Each event version crosses once however many rows admit it: `lab0`
        # is its own record, so its id is in the pickle twice (event id and
        # record id), not once per row.
        blob = pickle.dumps(report)
        assert blob.count(b"lab0") == 2
        again = pickle.loads(blob)
        assert again.counts == report.counts
        assert again.row("r39").events == late.events
        assert again.row("r39").selection == late.selection

    def test_S4_a_damaged_source_is_a_finding_not_a_failure(self, tmp_path: Path):
        from medh5.clinical import Document

        path = tmp_path / "notes.medh5"
        long = Document("note_text", "a long note " * 20_000)  # chunked, compressed
        History.write(
            path,
            events=[
                Event(
                    "note",
                    "note",
                    "document",
                    "point",
                    "final",
                    effective_start_us=DAY,
                    available_us=DAY + HOUR,
                )
            ],
            documents=[long],
            links=[
                Link.between(("event", "note"), "describes", ("document", "note_text"))
            ],
        )
        task = TaskManifest.new("t", "1", identity_namespace="site", base=tmp_path)
        task.add_subject("P-01", [SourceRef.pin(path, uri=path.name)])
        task.add_row("r", "P-01", 2 * DAY)
        with h5py.File(path, "r+") as f:
            data = f["clinical/documents/text/data"]
            assert data.chunks is not None
            # A chunk that no longer decompresses, under unchanged digests:
            # verifying it and checking its text both fail to read it.
            data.id.write_direct_chunk((0,), b"\x00" * 16)
        report = task.preflight()
        assert not report.ok
        assert {f.code for f in report.findings} == {"T302", "T306"}
        assert report.row("r").status == "error"

    def test_S4_the_manifest_findings_leave_no_history_read(self, tmp_path: Path):
        task = self.many_cutoffs(tmp_path, 2)
        task.add_row("ghost", "P-99", DAY)
        report = task.preflight()
        assert {f.code for f in report.findings} == {"T201"}
        assert report.subjects == ()
        assert all(r.subject is None and r.events == () for r in report.rows)
        assert report.row("r0").reasons[0].startswith("manifest_invalid")


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
                    row_fingerprint=row.fingerprint,
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
                row_fingerprint=row.fingerprint,
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

    def test_B04_a_row_feature_holds_for_the_row_it_was_built_from(
        self, tmp_path: Path
    ):
        """An honest patient-level feature --- the mean of the CT the row's slot
        reads --- built while the subject had one fragment still validated
        after a second fragment of the same visit was added and the slot read
        that one instead: the cutoff and the admitted versions were the same,
        the image and so the feature were not (B04 of the 2.0 re-audit).  The
        entry pins the fingerprint of the row it was built from; a relocated
        copy of unchanged content is the same row."""
        root = tmp_path / "b04"
        root.mkdir()
        refs = {}
        for name, value in (("a", 11), ("b", 93)):
            History.write(root / f"{name}.medh5", subject_id="P-01", fill=value)
            refs[name] = SourceRef.pin(root / f"{name}.medh5", uri=f"{name}.medh5")
        # With both listed the slot reads the smaller pin; the feature is
        # first built from the other.
        later = min(refs, key=lambda n: refs[n].content_id)
        first = "b" if later == "a" else "a"

        def task_of(*sources: SourceRef) -> TaskManifest:
            task = TaskManifest.new(
                "t", "1", identity_namespace="n", slots=[Slot("ct", "CT")], base=root
            )
            task.add_subject("P-01", list(sources))
            task.add_row("r", "P-01", 24 * HOUR)
            return task.reconcile()

        def honest(task: TaskManifest, path: Path) -> float:
            row = task.preflight().row("r")
            fill = row.slots["ct"]
            assert fill.fragment is not None and fill.image_id is not None
            read = row.sources[fill.fragment]
            with read.open(root) as sample:
                mean = float(np.mean(sample.images[fill.image_id].read()))
            with CacheWriter(
                path,
                level="patient",
                encoder={"name": "mean-ct", "revision": "1"},
                output={"dtype": "float32", "shape": [1]},
                task=task,
            ) as w:
                w.add_row(row, np.full(1, mean, np.float32), sources=[read])
            return mean

        alone = task_of(refs[first])
        cache = root / "rows.medh5cache"
        assert honest(alone, cache) == (11.0 if first == "a" else 93.0)
        assert validate_cache(cache, task=alone).ok
        both = task_of(refs[first], refs[later])
        assert honest(both, root / "now.medh5cache") != honest(alone, cache)
        findings = validate_cache(cache, task=both).findings
        assert found(findings) == ["T404"]
        assert "built for row 'r'" in str(findings[0])
        # The same content elsewhere is the same row.
        moved = root / "moved"
        moved.mkdir()
        shutil.copyfile(root / f"{first}.medh5", moved / "copy.medh5")
        relocated = task_of(SourceRef.pin(moved / "copy.medh5", uri="moved/copy.medh5"))
        assert relocated.preflight().row("r").fingerprint == (
            alone.preflight().row("r").fingerprint
        )
        assert validate_cache(cache, task=relocated).ok

    def test_N06_an_event_feature_names_no_row(self, setup, tmp_path: Path):
        """An event-level entry declaring a later row, its cutoff and its
        history validated against that row, while a lookup by event version
        --- which `documents=` makes --- served it to every row (N06 of the
        2.0 re-audit).  An event feature names no row: the writer refuses
        one, and a manifest declaring one fails its schema."""
        task, _ = setup
        later = task.preflight().row("P-01@d95")
        source = task.subjects[0].sources[0]
        path = tmp_path / "cohort" / "events.medh5cache"
        header: dict[str, Any] = {
            "level": "event",
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }
        row_fields: dict[str, Any] = {
            "row_id": later.row_id,
            "cutoff_us": later.cutoff_us,
            "event_versions": [e.event_id for e in later.events],
        }
        with (
            pytest.raises(MEDH5ValidationError, match="no row") as caught,
            CacheWriter(path, **header) as w,
        ):
            w.add(
                "e1",
                np.ones(1, np.float32),
                sources=[source],
                event_id="rep_v1",
                **row_fields,
            )
        assert caught.value.code == "T404" and not path.exists()
        with CacheWriter(path, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        assert validate_cache(path, task=task).ok  # stateless reuse
        self._rewrite_manifest(path, lambda doc: doc["entries"][0].update(row_fields))
        report = validate_cache(path, task=task)
        assert found(report.findings) == ["T401"] and report.corrupt == ("manifest",)

    @staticmethod
    def _rewrite_manifest(path: Path, change: Any) -> None:
        """Edit a cache's manifest and re-checksum it, as a writer would."""
        import hashlib

        from medh5._core import canonical_json

        with h5py.File(path, "r+") as f:
            doc = json.loads(f["manifest"][()].decode())
            change(doc)
            text = canonical_json(doc)
            del f["manifest"]
            f.create_dataset(
                "manifest", data=text.decode(), dtype=h5py.string_dtype("utf-8")
            )
            f.attrs["manifest_digest"] = "sha256:" + hashlib.sha256(text).hexdigest()

    def test_B04_a_row_feature_reads_only_its_subjects_sources(
        self, setup, tmp_path: Path
    ):
        """Event ids are local to a sample, so two patients' rows admit the same
        ids at one cutoff: an entry for one row pinned to the other patient's
        source validated against the task, and served that patient's feature."""
        task, paths = setup
        report = task.preflight()
        mine, theirs = report.row("P-01@24h"), report.row("P-02@24h")
        assert [e.event_id for e in mine.events] == [e.event_id for e in theirs.events]
        header = {
            "level": "patient",
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [2]},
            "task": task,
        }

        def write(path: Path, sources: list[SourceRef]) -> Path:
            with CacheWriter(path, **header) as w:
                w.add(
                    mine.row_id,
                    np.zeros(2, np.float32),
                    sources=sources,
                    row_id=mine.row_id,
                    row_fingerprint=mine.fingerprint,
                    cutoff_us=mine.cutoff_us,
                    event_versions=[e.event_id for e in mine.events],
                )
            return path

        crossed = write(
            tmp_path / "cohort" / "crossed.medh5cache", list(theirs.sources)
        )
        findings = validate_cache(crossed, task=task).findings
        assert found(findings) == ["T404"]
        assert "does not pin" in str(findings[0])
        # A relocated copy of the row's own source is the same version.
        moved = tmp_path / "elsewhere" / "P-01.medh5"
        moved.parent.mkdir()
        shutil.copyfile(paths["P-01"], moved)
        relocated = SourceRef.pin(moved, uri=str(moved))
        assert relocated.content_id == mine.sources[0].content_id
        own = write(tmp_path / "cohort" / "own.medh5cache", [relocated])
        assert validate_cache(own, task=task).ok

    def test_C03_an_entry_is_the_layout_its_cache_declares(self, setup, tmp_path: Path):
        """The checksum vouches for the bytes, not for the declaration: a header
        relabelled `float64[999]` validated and served `float32[2]`."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        path = tmp_path / "cohort" / "events.medh5cache"
        with CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        ) as w:
            w.add_event(source, "rep_v1", np.ones(2, np.float32))
        for output in (
            {"dtype": "float64", "shape": [2]},
            {"dtype": "float32", "shape": [3]},
        ):
            self._rewrite_manifest(path, lambda doc, o=output: doc.update(output=o))
            report = validate_cache(path)
            assert found(report.findings) == ["T402"] and report.corrupt
            with (
                FeatureCache.open(path) as cache,
                pytest.raises(MEDH5ValidationError) as caught,
            ):
                cache.event_feature(source.content_id, "rep_v1")
            assert caught.value.code == "T402"

    def test_C04_a_cache_holds_what_its_level_requires(self, setup, tmp_path: Path):
        """The writer required a patient cache to name its task and every entry
        its row, cutoff and selection, and nothing read it back: a cache
        relabelled `patient` with none of it validated against the task."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        header = {
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }
        path = tmp_path / "cohort" / "relabelled.medh5cache"
        with CacheWriter(path, level="event", **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        self._rewrite_manifest(path, lambda doc: doc.update(level="patient"))
        assert found(validate_cache(path, task=task).findings) == ["T401"]
        writer = CacheWriter(
            tmp_path / "cohort" / "x.medh5cache", level="event", **header
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            writer.add("x", np.ones(1, np.float32), sources=[source])
        writer.abort()
        assert caught.value.code == "T404", "an event-level entry names its event"
        # Fitted on another split with the same training subjects is still
        # another split: the record names it.
        foreign = tmp_path / "cohort" / "foreign.medh5cache"
        record = dict(fitted_on(task), set_id="another-split")
        with CacheWriter(foreign, level="event", fitted_on=record, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        findings = validate_cache(foreign, task=task).findings
        assert found(findings) == ["T405"] and "set_id" in str(findings[0])

    @pytest.mark.parametrize("cutoff", [2**63, -(2**63) - 1, 3600000000.0])
    def test_C12_a_cutoff_is_a_64_bit_integer(self, setup, cutoff):
        """JSON Schema's `integer` admits `3600000000.0`, which serialisers
        write for integers, and values past int64; both read as cutoff 0 and
        validated --- moving the row, and what it may read."""
        task, _ = setup
        doc = task.to_json()
        bad = dict(doc, rows=[dict(doc["rows"][0], cutoff_us=cutoff)])
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest(bad, base=task.base)
        assert caught.value.code == "T101"

    @pytest.mark.parametrize("cutoff", [2**63 - 1, -(2**63)])
    def test_C12_both_int64_endpoints_are_cutoffs(self, setup, cutoff):
        task, _ = setup
        doc = task.to_json()
        fine = TaskManifest(dict(doc, rows=[dict(doc["rows"][0], cutoff_us=cutoff)]))
        assert fine.validate() == [] and fine.rows[0].cutoff_us == cutoff

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
