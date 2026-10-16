"""The clinical profile (format 1.1), through the public Python API.

Every fixture is written by the public writer (``tests.kits.History``); defects
are planted afterwards with ``h5py``.  Test names cite the 1.1 clause they hold.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
import medh5.clinical as clinical
from medh5.clinical import (
    DAY,
    HOUR,
    ClinicalRecords,
    Clock,
    Document,
    Event,
    Link,
    SelectionPolicy,
)
from medh5.collection import open_collection, pack, unpack
from medh5.errors import MEDH5ValidationError, MEDH5VersionError
from medh5.storage import recompress
from medh5.validate import validate_file
from tests.helpers import LATER_MINOR, write_sample
from tests.kits import History


@pytest.fixture
def history(tmp_path: Path) -> Path:
    path = tmp_path / "history.medh5"
    History.write(path)
    return path


def codes(path: Path, level: str = "semantic") -> list[str]:
    """Every code but the 1.0 warnings the kit's samples carry by design (no
    de-identification, no registration, no ontology binding)."""
    found = validate_file(path, level=level).codes
    return sorted(c for c in found if c not in {"W903", "W911", "W912"})


class TestVersions:
    def test_S2_1_imaging_alone_is_written_as_1_0(self, sample_path: Path):
        with medh5.open(sample_path) as s:
            assert s.version == "1.0"
            assert s.support == "full"
            assert "clinical" not in s.profiles
            assert s.clinical is None

    def test_S2_1_a_history_makes_the_sample_1_1(self, history: Path):
        with medh5.open(history) as s:
            assert s.version == "1.1"
            assert {"core", "clinical", "longitudinal"} <= s.profiles
            assert s.clinical is not None
            assert s.clinical is s.clinical  # read once
        assert codes(history, "integrity") == []

    def test_S2_1_the_lowest_version_a_content_needs(self):
        assert medh5.FORMAT_VERSION == "1.2"
        from medh5._core import written_version

        assert written_version(None, ["core"]) == "1.0"
        assert written_version(None, ["core", "clinical"]) == "1.1"
        assert written_version("1.1", ["core"]) == "1.1"  # never downgraded

    def test_S2_2_a_higher_minor_is_a_projection(self, history: Path, tmp_path: Path):
        future = tmp_path / "future.medh5"
        shutil.copyfile(history, future)
        with h5py.File(future, "r+") as f:
            f.attrs["medh5_version"] = LATER_MINOR
            f["clinical/events"].attrs["defined_by"] = f"MEDH5 {LATER_MINOR}"
        with medh5.open(future) as s:
            assert s.support == "projection"
            assert s.clinical is not None and s.clinical.projection
            assert len(s.clinical.events) == 6
        report = validate_file(future, level="integrity")
        assert codes(future, "integrity") == ["W913"], report.codes
        where = {d.location for d in report.diagnostics if d.code == "W913"}
        # The version, the attribute the later minor defined, and the root this engine
        # cannot recompute (a later minor may cover more): reported, not E702.
        assert {"/", "/clinical/events@defined_by"} <= where, where
        assert report.ok
        with pytest.raises(MEDH5VersionError), medh5.amend(future):
            pass
        with pytest.raises(MEDH5VersionError):
            recompress(future, profile="portable")
        with pytest.raises(MEDH5VersionError):
            pack([future], tmp_path / "shard.medh5c")

    @pytest.mark.parametrize("policy", ["strict_prospective", "latest_provable"])
    def test_S2_2_a_projection_is_never_selected_from(
        self, history: Path, tmp_path: Path, capsys, policy: str
    ):
        """What a cutoff admits from a later minor is not this engine's to
        claim: ``select`` reported such a file ``certified``, while a task's
        preflight refused it (T306)."""
        from medh5.cli import EXIT_ERROR, EXIT_OK, main

        future = tmp_path / "future.medh5"
        shutil.copyfile(history, future)
        with h5py.File(future, "r+") as f:
            f.attrs["medh5_version"] = LATER_MINOR
        argv = ["--cutoff-hours", "24", "--policy", policy, "--json"]
        with medh5.open(history) as s:
            assert s.clinical.select(24 * HOUR, policy).status != "uncertifiable"
        assert main(["clinical", "select", str(history), *argv]) == EXIT_OK
        capsys.readouterr()
        with medh5.open(future) as s:
            assert s.clinical is not None and s.clinical.projection
            with pytest.raises(MEDH5VersionError, match="projection"):
                s.clinical.select(24 * HOUR, policy)
            events, links = s.clinical.events, s.clinical.links
        assert main(["clinical", "select", str(future), *argv]) == EXIT_ERROR
        assert "projection" in capsys.readouterr().err
        # Records taken out of the file are the caller's to select from.
        assert clinical.select(events, links, 24 * HOUR, policy).event_ids

    def test_S2_2_a_member_of_a_text_column_follows_the_version(
        self, history: Path, tmp_path: Path
    ):
        """A member a UTF-8 column does not define is a later minor's under a
        projection (W913), as an unknown column is; it was E804 at any version."""
        for version, expected in (("1.1", "E804"), (LATER_MINOR, "W913")):
            path = tmp_path / f"v{version}.medh5"
            shutil.copyfile(history, path)
            with h5py.File(path, "r+") as f:
                f.attrs["medh5_version"] = version
                f["clinical/events/kind"].create_dataset(
                    "index", data=np.zeros(3, np.uint8)
                )
            found = {
                d.code
                for d in validate_file(path).diagnostics
                if d.location == "/clinical/events/kind/index"
            }
            assert found == {expected}, (version, found)

    def test_S4_a_big_endian_numeric_column_is_E805(
        self, history: Path, tmp_path: Path
    ):
        """1.1 §4 stores numeric columns little-endian; HDF5 converts on read,
        so a big-endian column read correctly and validated clean."""
        path = tmp_path / "big.medh5"
        shutil.copyfile(history, path)
        with h5py.File(path, "r+") as f:
            table = f["clinical/events"]
            values = table["value_num"][...]
            del table["value_num"]
            table.create_dataset("value_num", data=values.astype(">f8"))
        report = validate_file(path)
        (found,) = [d for d in report.diagnostics if d.code == "E805"]
        assert found.location == "/clinical/events/value_num"
        assert "big-endian" in found.message

    @pytest.mark.parametrize(
        "column",
        ["events/event_id", "events/value_text", "documents/text", "links/relation"],
    )
    def test_S4_big_endian_offsets_are_E805(
        self, history: Path, tmp_path: Path, column: str
    ):
        """A UTF-8 column's offsets are little-endian too: the packed branch
        checked their dtype and rank and not their byte order, so big-endian
        offsets validated --- required, nullable
        (empty in every row here) and deferred columns alike."""
        assert "E805" not in validate_file(history).codes
        path = tmp_path / "big.medh5"
        shutil.copyfile(history, path)
        with h5py.File(path, "r+") as f:
            group = f[f"clinical/{column}"]
            offsets = group["offsets"][...]
            del group["offsets"]
            group.create_dataset("offsets", data=offsets.astype(">u8"))
        report = validate_file(path)
        (found,) = [d for d in report.diagnostics if d.code == "E805"]
        assert found.location == f"/clinical/{column}/offsets"
        assert "big-endian" in found.message

    def test_S2_2_another_major_is_refused(self, history: Path, tmp_path: Path):
        future = tmp_path / "major.medh5"
        shutil.copyfile(history, future)
        with h5py.File(future, "r+") as f:
            f.attrs["medh5_version"] = "2.0"
        assert "E002" in codes(future)

    def test_S2_3_an_unknown_profile_refuses_the_amendment(
        self, history: Path, tmp_path: Path
    ):
        odd = tmp_path / "odd.medh5"
        shutil.copyfile(history, odd)
        with h5py.File(odd, "r+") as f:
            f.attrs["medh5_profiles"] = ["clinical", "core", "x_unknown"]
        before = odd.read_bytes()
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(odd):
            pass
        assert caught.value.code == "E007"
        assert odd.read_bytes() == before


class TestRecords:
    def test_S5_records_round_trip(self, history: Path):
        with medh5.open(history) as s:
            c = s.clinical
            assert c is not None
            assert c.clock == Clock.relative("clock", "baseline CT acquisition")
            assert [e.event_id for e in c.events] == [
                "lab0",
                "ct0",
                "rep_v1",
                "rep_v2",
                "ct1",
                "recist1",
            ]
            lab = c.event("lab0")
            assert lab.effective_start_us == (-48 * HOUR, -48 * HOUR)
            assert (lab.value_num, lab.unit, lab.code) == (1.1, "mg/dL", "2160-0")
            assert c.text("rep_text_v1").endswith("Preliminary.")
            info = {d.document_id: d for d in c.documents}
            assert info["rep_text_v1"].n_bytes == len(c.text("rep_text_v1").encode())
            assert c.document("rep_text_v2").text.endswith("Final.")
            assert len(c.links) == 5
            records = c.records()
            assert ClinicalRecords.from_json(records.to_json()) == records
            assert "events" in c.summary() and "Clinical(" in repr(c)
            with pytest.raises(KeyError):
                c.event("nope")

    def test_S5_1_a_record_is_checked_as_it_is_added(self, tmp_path: Path):
        with History.writer_for(tmp_path / "w.medh5") as w:
            with pytest.raises(MEDH5ValidationError, match="clock first"):
                w.add_event(Event("z", "z", "other", "static", "final"))
            w.set_clock(Clock.relative("clock", "baseline CT acquisition"))
            with pytest.raises(MEDH5ValidationError) as caught:
                w.add_event(Event("x", "x", "observation", "point", "final"))
            assert caught.value.code == "E811"
            with pytest.raises(MEDH5ValidationError) as caught:
                w.add_event(
                    Event(
                        "y",
                        "y",
                        "observation",
                        "point",
                        "final",
                        effective_start_us=0,
                        value_num=2.0,
                    )
                )
            assert caught.value.code == "E812"
            w.add_event(Event("z", "z", "other", "static", "final"))
            with pytest.raises(MEDH5ValidationError) as caught:
                w.add_event(Event("z", "z", "other", "static", "final"))
            assert caught.value.code == "E809"

    def test_S5_2_a_numpy_integer_is_a_time(self, tmp_path: Path):
        """An instant of ``numpy.int64`` could not even make an Event, though
        the same value in a dict was written.  A float or a bool is no time
        either way: in a pair, an Event truncated ``1.5`` to ``1``."""
        t = np.int64(3 * HOUR)
        event = Event(
            "e",
            "e",
            "observation",
            "point",
            "final",
            effective_start_us=t,
            available_us=(np.int32(-60), t),
        )
        assert event.effective_start_us == (3 * HOUR, 3 * HOUR)
        assert event.available_us == (-60, 3 * HOUR)
        assert {type(x) for x in (*event.effective_start_us, *event.available_us)} == {
            int
        }
        with History.writer_for(tmp_path / "w.medh5") as w:
            w.set_clock(Clock.relative("clock", "baseline CT acquisition"))
            w.add_event(event)
            for bad in [(1.5, 2), (np.float64(1), 2), (True, 2), (1, np.bool_(True))]:
                as_event = Event("x", "x", "other", "static", "final", available_us=bad)
                as_dict = {**as_event.to_json(), "available_us": list(bad)}
                for given in (as_event, as_dict):
                    with pytest.raises(MEDH5ValidationError) as caught:
                        w.add_event(given)
                    assert caught.value.code == "E811", (bad, given)
            for scalar in (2.0, True, np.float32(2)):
                with pytest.raises(TypeError, match="a clinical time"):
                    Event("x", "x", "other", "static", "final", available_us=scalar)
            w.commit()
        with medh5.open(tmp_path / "w.medh5") as s:
            assert s.clinical is not None
            assert s.clinical.events == (event,)
            w.abort()

    def test_S3_hours_and_days_are_the_engines(self):
        """One rounding of hours and days, half to even: the command line and
        augment rounded half away from zero, so ``1.25e-09`` hours was 4 us
        here and 5 there, and admission is a closed bound."""
        from medh5 import _core

        ties = [k + 0.5 for k in range(-40, 40)]
        for x in [*(t / HOUR for t in ties), *(t / DAY for t in ties), 1.25e-09]:
            assert clinical.hours(x) == _core.clinical_round_micros(x, HOUR), x
            assert clinical.days(x) == _core.clinical_round_micros(x, DAY), x
        assert clinical.hours(1.25e-09) == 4
        with pytest.raises(ValueError, match="no time on the subject clock"):
            _core.clinical_round_micros(float("nan"), HOUR)

    def test_S3_the_clock_is_fixed_once(self, tmp_path: Path):
        with History.writer_for(tmp_path / "w.medh5") as w:
            w.set_clock(Clock.relative("other", "a different origin"))
            with pytest.raises(MEDH5ValidationError) as caught:
                w.set_clock(Clock.relative("clock", "baseline CT acquisition"))
            assert caught.value.code == "E802"
            w.abort()

    def test_S7_a_dangling_link_is_refused_at_commit(self, tmp_path: Path):
        path = tmp_path / "dangling.medh5"
        with pytest.raises(MEDH5ValidationError) as caught:
            History.write(
                path,
                links=[Link.between(("event", "ct0"), "describes", ("image", "MR"))],
            )
        assert "E813" in str(caught.value)
        assert not path.exists()

    def test_S6_a_document_event_owns_at_most_one_document(self, tmp_path: Path):
        """One text per event version: two would share one availability and
        one revision chain, and an event-level feature would name neither."""
        path = tmp_path / "two.medh5"
        note = Event(
            "note",
            "note",
            "document",
            "point",
            "final",
            effective_start_us=DAY,
            available_us=DAY + HOUR,
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            History.write(
                path,
                events=[note],
                documents=[Document("note_body", "body"), Document("note_add", "add")],
                links=[
                    Link.between(("event", "note"), "describes", ("document", d))
                    for d in ("note_body", "note_add")
                ],
            )
        assert "E815" in str(caught.value) and "owns 2 documents" in str(caught.value)
        assert not path.exists()

    def test_S5_keyword_and_dict_forms(self, tmp_path: Path):
        path = tmp_path / "kw.medh5"
        with History.writer_for(path) as w:
            w.set_clock(
                id="clock", unit="us", reference="relative", origin_description="scan"
            )
            w.add_event(
                event_id="ct0",
                record_id="ct0",
                kind="imaging",
                temporal_type="point",
                status="final",
                effective_start_us=0,
                available_us=HOUR,
                timepoint_id="tp0",
            )
            w.add_link(
                {
                    "source_type": "event",
                    "source_id": "ct0",
                    "relation": "describes",
                    "target_type": "image",
                    "target_id": "CT_tp0",
                }
            )
            w.add_document(document_id="note", text="")
            w.add_event(Event("note_ev", "note_ev", "document", "static", "final"))
            w.add_link(
                Link.between(("event", "note_ev"), "describes", ("document", "note"))
            )
            assert w.has_clinical
            assert len(w.clinical()["events"]) == 2
        with medh5.open(path) as s:
            assert s.clinical is not None and s.clinical.text("note") == ""

    def test_S5_add_records_takes_a_bundle(self, history: Path, tmp_path: Path):
        with medh5.open(history) as s:
            assert s.clinical is not None
            bundle = s.clinical.records()
        path = tmp_path / "bundle.medh5"
        with History.writer_for(path) as w:
            w.add_records(bundle.to_json())
        with medh5.open(path) as s:
            assert s.clinical is not None and s.clinical.records() == bundle

    def test_S5_1_clinical_vocabularies(self):
        assert "medication_administration" in clinical.EVENT_KINDS
        assert clinical.PROFILE == "clinical" and clinical.SCHEMA == "medh5.clinical/1"
        assert "resolved" in clinical.LESION_VALUES
        assert clinical.hours(1.5) == 90 * 60 * 1_000_000 and clinical.days(1) == DAY
        assert '"descriptor"' in clinical.schema_text()
        assert set(clinical.SELECTION_POLICIES) == {
            "strict_prospective",
            "latest_provable",
        }


class TestAmendment:
    def test_S8_a_no_op_amend_keeps_the_address(self, history: Path):
        with medh5.open(history) as s:
            before = s.content_id
        with medh5.amend(history):
            pass
        with medh5.open(history) as s:
            assert s.content_id == before and s.version == "1.1"

    def test_S8_a_revision_changes_the_address_not_the_payloads(self, history: Path):
        with medh5.open(history) as s:
            before = s.content_id
            image = s.images["CT_tp0"].digest
        with medh5.amend(history) as w:
            w.add_event(
                Event(
                    "rep_v3",
                    "rep",
                    "document",
                    "point",
                    "amended",
                    effective_start_us=0,
                    available_us=72 * HOUR,
                )
            )
            w.add_document(Document("rep_text_v3", "Addendum."))
            w.add_link(
                Link.between(
                    ("event", "rep_v3"), "describes", ("document", "rep_text_v3")
                )
            )
            w.add_link(
                Link.between(("event", "rep_v3"), "supersedes", ("event", "rep_v2"))
            )
        with medh5.open(history) as s:
            assert s.content_id != before
            assert s.images["CT_tp0"].digest == image
            assert s.clinical is not None
            assert s.clinical.select(80 * HOUR).event_ids[-1] == "rep_v3"

    def test_S10_strip_and_augment(self, history: Path, tmp_path: Path):
        projection = tmp_path / "imaging.medh5"
        report = clinical.strip(history, projection)
        assert report["version_after"] == "1.0"
        assert report["lost"]["events"] == 6
        with medh5.open(projection) as s:
            assert s.version == "1.0" and s.clinical is None
        with medh5.open(history) as s:
            assert s.clinical is not None
            records = s.clinical.records()
        back = tmp_path / "back.medh5"
        augmented = clinical.augment(projection, records, out=back)
        assert augmented["version_before"] == "1.0"
        assert augmented["version_after"] == "1.1"
        assert augmented["unchanged_digests"] > 0
        with medh5.open(back) as a, medh5.open(history) as b:
            assert a.clinical is not None and b.clinical is not None
            assert a.clinical.records() == b.clinical.records()
        with pytest.raises(MEDH5ValidationError):
            clinical.strip(history, history)

    def test_S10_augment_in_place_reports_what_is_unknown(
        self, sample_path: Path, tmp_path: Path
    ):
        events, links, notes = clinical.imaging_events_from_timepoints(sample_path)
        assert events and all(e.available_us is None for e in events)
        assert notes
        records = clinical.records_from(
            clinical.baseline_day_clock("clock"), events, (), links
        )
        report = clinical.augment(sample_path, records)
        assert report["assumptions"]
        with medh5.open(sample_path) as s:
            assert s.version == "1.1"
            assert s.clinical is not None
            # Unknown availability: strict selection never uses them.
            assert s.clinical.select(365 * DAY).event_ids == []

    def test_S10_imaging_events_keep_the_day_their_interval_names(self, tmp_path: Path):
        """`days_from_baseline` counts days after the baseline visit, whose time
        of day is unknown.  1.75 days is `[1.75, 2.75)` days, where it was
        floored to `[1, 2)`, which may not hold the visit; an interval the clock
        cannot hold makes no event, where `2e14` days wrapped; and an image id
        too long to prefix makes a valid event id, which the helper's own
        output was not."""
        path = tmp_path / "visits.medh5"
        long_id = "CT_" + "x" * 122
        visits = (("tp0", 0, "CT0"), ("tp1", 1.75, long_id), ("tp2", 2e14, "CT2"))
        with medh5.create(path, sample_id="visits", subject_id="P-01") as w:
            for tp, days, image_id in visits:
                w.add_timepoint(tp, days_from_baseline=days)
                w.add_grid(
                    f"g_{tp}", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0), timepoint=tp
                )
                w.add_image(
                    image_id,
                    np.zeros((2, 2, 2), np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
        events, links, notes = clinical.imaging_events_from_timepoints(path)
        start = {e.timepoint_id: e.effective_start_us for e in events}
        assert start == {
            "tp0": (0, DAY - 1),
            "tp1": (DAY * 7 // 4, DAY * 7 // 4 + DAY - 1),
        }
        assert any("tp2" in n and "cannot hold" in n for n in notes)
        (shortened,) = [e.event_id for e in events if e.timepoint_id == "tp1"]
        assert len(shortened) == 128 and shortened.startswith("imaging.CT_xxx")
        assert [link.target_id for link in links] == ["CT0", long_id]
        records = clinical.records_from(
            clinical.baseline_day_clock("clock"), events, (), links
        )
        report = clinical.augment(path, records)
        assert report["events"] == 2

    def test_S10_a_foreign_clinical_group_is_refused_not_reinterpreted(
        self, sample_path: Path
    ):
        with h5py.File(sample_path, "r+") as f:
            f.create_group("clinical").create_dataset("ours", data=np.arange(3))
        with medh5.open(sample_path) as s:
            assert s.clinical is None
        with medh5.amend(sample_path):
            pass  # copied through, untouched
        with h5py.File(sample_path) as f:
            assert list(f["clinical/ours"][()]) == [0, 1, 2]
        records = ClinicalRecords(Clock.relative("c", "origin"))
        with pytest.raises(MEDH5ValidationError):
            clinical.augment(sample_path, records)


class TestImmutableVersions:
    """An amendment could attach a payload to an
    event version the file already held.  Text linked to an inherited
    text-less document event available at 1 h was owned by that version, and
    selection at 24 h admitted it, though it was written later; an image could
    be added to an inherited imaging event the same way.  An event version is
    immutable, and so is what it owns (1.1 §7.3)."""

    NOTE = Event(
        "note_v1",
        "note",
        "document",
        "point",
        "final",
        effective_start_us=0,
        available_us=HOUR,
    )

    def test_S7_3_text_attached_to_an_inherited_version_is_refused(
        self, tmp_path: Path
    ):
        path = tmp_path / "note.medh5"
        History.write(path, events=[self.NOTE])
        with medh5.open(path) as s:
            before = s.content_id
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.add_document(Document("note_text", "Written a week later."))
            w.add_link(
                Link.between(
                    ("event", "note_v1"), "describes", ("document", "note_text")
                )
            )
        assert caught.value.code == "E809"
        assert "supersedes" in str(caught.value)
        with medh5.open(path) as s:
            assert s.content_id == before

    def test_S7_3_an_image_attached_to_an_inherited_version_is_refused(
        self, history: Path
    ):
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(history) as w:
            w.add_image(
                "CT2_tp0",
                np.zeros(History.SHAPE, np.int16),
                grid="ct_tp0",
                modality="CT",
            )
            w.add_link(
                Link.between(("event", "ct0"), "describes", ("image", "CT2_tp0"))
            )
        assert caught.value.code == "E809"

    def test_S7_3_a_grounding_of_an_inherited_version_stays_allowed(
        self, history: Path
    ):
        """A link from an inherited version that is not its structural link
        --- asserted by a new version, available when it was drawn, or one no
        selection reads as a payload --- is still added."""
        with medh5.amend(history) as w:
            w.add_event(
                Event(
                    "ground1",
                    "ground1",
                    "procedure",
                    "point",
                    "completed",
                    effective_start_us=72 * HOUR,
                    available_us=72 * HOUR,
                )
            )
            w.add_link(
                Link.between(
                    ("event", "ct0"),
                    "describes",
                    ("annotation", "lesions_tp0"),
                    asserted_by="ground1",
                )
            )
            w.add_link(
                Link.between(("event", "rep_v1"), "compares_with", ("event", "ct0"))
            )
        with medh5.open(history) as s:
            assert s.clinical is not None and len(s.clinical.links) == 7
            assert not s.clinical.select(48 * HOUR).admits("annotation", "lesions_tp0")
            assert s.clinical.select(72 * HOUR).admits("annotation", "lesions_tp0")

    def test_S7_2_a_superseding_version_owns_the_text_from_its_availability(
        self, tmp_path: Path
    ):
        path = tmp_path / "note.medh5"
        History.write(path, events=[self.NOTE])
        with medh5.amend(path) as w:
            w.add_event(
                Event(
                    "note_v2",
                    "note",
                    "document",
                    "point",
                    "amended",
                    effective_start_us=0,
                    available_us=48 * HOUR,
                )
            )
            w.add_document(Document("note_text", "Written two days later."))
            w.add_link(
                Link.between(
                    ("event", "note_v2"), "describes", ("document", "note_text")
                )
            )
            w.add_link(
                Link.between(("event", "note_v2"), "supersedes", ("event", "note_v1"))
            )
        with medh5.open(path) as s:
            assert s.clinical is not None
            early = s.clinical.select(24 * HOUR)
            late = s.clinical.select(72 * HOUR)
        assert "note_v1" in early.event_ids
        assert not early.admits("document", "note_text")
        assert "note_v2" in late.event_ids and late.admits("document", "note_text")


class TestHeldVersions:
    """What an event version the amended file held attests is fixed with it
    (1.1 §7.3): its structural link, the links it asserts, and the payloads
    they reach.  Attached later --- a grounding asserted by the original
    report, a mask replaced under the id a procedure attests, a link added to a
    dropped and re-added history --- each was an input from the old version's
    availability, before it existed."""

    SEG = Event(
        "seg0",
        "seg0",
        "procedure",
        "point",
        "completed",
        effective_start_us=92 * DAY,
        available_us=92 * DAY,
    )
    NOTE = Event(
        "note1",
        "note1",
        "document",
        "point",
        "final",
        effective_start_us=DAY,
        available_us=DAY,
    )

    def _segmented(self, path: Path) -> Path:
        History.write(
            path,
            events=[self.SEG],
            links=[
                Link.between(
                    ("event", "seg0"),
                    "describes",
                    ("annotation", "lesions_tp0"),
                    asserted_by="seg0",
                )
            ],
        )
        return path

    def test_S7_3_a_link_asserted_by_a_held_version_is_refused(self, tmp_path: Path):
        path = self._segmented(tmp_path / "seg.medh5")
        with medh5.open(path) as s:
            before = s.content_id
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.add_link(
                Link.between(
                    ("event", "rep_v1"),
                    "describes",
                    ("annotation", "lesions_tp0"),
                    asserted_by="rep_v1",
                )
            )
        assert caught.value.code == "E809"
        assert "asserted by event 'rep_v1'" in str(caught.value)
        with medh5.open(path) as s:
            assert s.content_id == before
            assert s.clinical is not None
            assert not s.clinical.select(24 * HOUR).admits("annotation", "lesions_tp0")

    @pytest.mark.parametrize("readd", [True, False], ids=["replaced", "removed"])
    def test_S7_3_a_payload_a_held_version_attests_is_fixed(
        self, tmp_path: Path, readd: bool
    ):
        from medh5.annotations.voxel import InstanceInput
        from tests.kits import block

        path = self._segmented(tmp_path / "seg.medh5")
        with medh5.open(path) as s:
            before = s.content_id
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.remove_annotation("lesions_tp0")
            if readd:
                w.add_segmentation(
                    "lesions_tp0",
                    grid="ct_tp0",
                    instances=[
                        InstanceInput(3, 1, mask=block(History.SHAPE, (5, 8, 2), 2))
                    ],
                    annotated_classes=[3],
                )
        assert caught.value.code == "E809"
        assert "annotation 'lesions_tp0', which event 'seg0' attests" in str(
            caught.value
        )
        with medh5.open(path) as s:
            assert s.content_id == before

    def test_S7_3_the_same_payload_written_again_is_unchanged(self, tmp_path: Path):
        from medh5.annotations.voxel import InstanceInput
        from tests.kits import block

        path = self._segmented(tmp_path / "seg.medh5")
        with medh5.open(path) as s:
            before = s.content_id
        with medh5.amend(path) as w:
            w.remove_annotation("lesions_tp0")
            w.add_segmentation(
                "lesions_tp0",
                grid="ct_tp0",
                instances=[
                    InstanceInput(3, 1, mask=block(History.SHAPE, (2, 1, 6), 3))
                ],
                annotated_classes=[3],
            )
        with medh5.open(path) as s:
            assert s.content_id == before

    @pytest.mark.parametrize("damage", ["digest", "bytes"])
    def test_S7_3_a_digest_restamped_over_the_same_bytes_changes_nothing(
        self, tmp_path: Path, damage: str
    ):
        """Re-attesting the bytes a payload has --- `fix --rewrite-digests` over
        a wrong digest, or over bytes edited outside medh5 --- is no change to
        what a version attests: the amendment changes no content.  Comparing
        stored digests alone refused it (E809)."""
        from medh5.integrity import fix

        path = tmp_path / "ct.medh5"
        History.write(path)
        with h5py.File(path, "r+") as f:
            image = f["images/CT_tp0"]
            if damage == "digest":
                image.attrs["digest"] = "sha256:" + "0" * 64
            else:
                volume = image[...]
                volume[0, 0, 0] += 1
                image[...] = volume
        fix(path, rewrite_digests=True, reason="re-attested in a test")
        assert not validate_file(path, level="integrity").errors

    def test_S7_3_a_transform_a_held_version_attests_is_fixed(self, tmp_path: Path):
        path = tmp_path / "reg.medh5"
        History.write(path)
        with medh5.amend(path) as w:
            w.add_transform(
                "tp1_to_tp0",
                kind="affine",
                from_frame="P-01-tp1",
                to_frame="P-01-tp0",
                matrix=np.eye(4),
            )
            w.add_event(
                Event(
                    "reg0",
                    "reg0",
                    "procedure",
                    "point",
                    "completed",
                    effective_start_us=91 * DAY,
                    available_us=91 * DAY,
                )
            )
            w.add_link(
                Link.between(
                    ("event", "reg0"),
                    "derived_from",
                    ("transform", "tp1_to_tp0"),
                    asserted_by="reg0",
                )
            )
        shifted = np.eye(4)
        shifted[:3, 3] = 2.0
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.remove_transform("tp1_to_tp0")
            w.add_transform(
                "tp1_to_tp0",
                kind="affine",
                from_frame="P-01-tp1",
                to_frame="P-01-tp0",
                matrix=shifted,
            )
        assert caught.value.code == "E809"
        assert "transform 'tp1_to_tp0'" in str(caught.value)

    @pytest.mark.parametrize("attr", ["rescale_slope", "modality"])
    def test_S7_3_an_attested_image_s_attributes_are_immutable(
        self, tmp_path: Path, attr: str
    ):
        """What 1.0 §13.2 covers of an attested image is held with its pixels:
        a slope rescales every voxel read, and an edit to it passed while the
        pixels were unchanged."""
        path = tmp_path / "ct.medh5"
        History.write(path)
        before = path.read_bytes()
        value = {"rescale_slope": 2.0, "modality": "MR"}[attr]
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.handle["images/CT_tp0"].attrs[attr] = value
        assert caught.value.code == "E809"
        assert "image 'CT_tp0', which event 'ct0' attests" in str(caught.value)
        assert path.read_bytes() == before
        # An image no version attests is the amendment's to change.
        with medh5.amend(path) as w:
            pet = np.ones(History.SHAPE, dtype=np.float32)
            w.add_image("PT_tp0", pet, grid="ct_tp0", modality="PT")
        with medh5.amend(path) as w:
            w.handle["images/PT_tp0"].attrs[attr] = value
        assert not validate_file(path, level="integrity").errors

    def _note(self, path: Path) -> ClinicalRecords:
        History.write(path, events=[self.NOTE])
        with medh5.open(path) as s:
            assert s.clinical is not None
            return s.clinical.records()

    def test_S7_3_a_dropped_history_added_again_attests_what_it_did(
        self, tmp_path: Path
    ):
        path = tmp_path / "note.medh5"
        records = self._note(path)
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.drop_clinical()
            w.add_records(records)
            w.add_document(Document("late_text", "Written a week later."))
            w.add_link(
                Link.between(("event", "note1"), "describes", ("document", "late_text"))
            )
        assert caught.value.code == "E809"
        with medh5.open(path) as s:
            assert s.clinical is not None
            assert not s.clinical.select(2 * DAY).admits("document", "late_text")

    @pytest.mark.parametrize("change", ["row", "document", "clock"])
    def test_S7_3_a_held_version_added_again_is_what_it_was(
        self, tmp_path: Path, change: str
    ):
        import dataclasses

        path = tmp_path / "note.medh5"
        records = self._note(path)
        if change == "row":
            earlier = dataclasses.replace(
                records.events[0], available_us=(-HOUR, -HOUR)
            )
            records = dataclasses.replace(
                records, events=(earlier, *records.events[1:])
            )
        elif change == "document":
            documents = [
                dataclasses.replace(d, text=d.text + " Amended.")
                if d.document_id == "rep_text_v1"
                else d
                for d in records.documents
            ]
            records = dataclasses.replace(records, documents=tuple(documents))
        else:
            records = dataclasses.replace(records, clock=Clock.relative("other", "x"))
        with pytest.raises(MEDH5ValidationError) as caught, medh5.amend(path) as w:
            w.drop_clinical()
            w.add_records(records)
        assert caught.value.code == ("E802" if change == "clock" else "E809")

    def test_S7_3_a_dropped_history_may_start_anew(self, tmp_path: Path):
        path = tmp_path / "note.medh5"
        self._note(path)
        fresh = Event(
            "fresh1",
            "fresh1",
            "document",
            "point",
            "final",
            effective_start_us=DAY,
            available_us=DAY,
        )
        with medh5.amend(path) as w:
            w.drop_clinical()
            w.set_clock(Clock.relative("other", "a new origin"))
            w.add_event(fresh)
            w.add_document(Document("late_text", "Written a week later."))
            w.add_link(
                Link.between(
                    ("event", "fresh1"), "describes", ("document", "late_text")
                )
            )
        with medh5.open(path) as s:
            assert s.clinical is not None
            assert [e.event_id for e in s.clinical.events] == ["fresh1"]


class TestValidation:
    def test_S4_columns_are_checked(self, history: Path):
        with h5py.File(history, "r+") as f:
            del f["clinical/events/available_lo_us"]
            f["clinical/events"].create_dataset(
                "available_lo_us", data=np.zeros(6, dtype=np.int32)
            )
        assert "E805" in codes(history, "structural")

    def test_S7_2_a_chain_is_checked_whole(self, tmp_path: Path):
        """A link given twice is one link, as selection reads it (the commit
        refused it as a branch and a merge); and a version available before
        an ancestor could have been is E816 though the version between them
        has unknown availability (the commit accepted it, and selection
        certified the newest version before its ancestor existed)."""
        twice = tmp_path / "twice.medh5"
        History.write(
            twice,
            links=[
                Link.between(("event", "rep_v2"), "supersedes", ("event", "rep_v1"))
            ],
        )
        assert codes(twice) == []
        with medh5.open(twice) as s:
            assert s.clinical is not None
            assert "rep_v2" in s.clinical.select(72 * HOUR).event_ids

        def version(event_id: str, available_us: int | None) -> Event:
            return Event(
                event_id,
                "x",
                "observation",
                "point",
                "final",
                effective_start_us=0,
                available_us=available_us,
            )

        with pytest.raises(MEDH5ValidationError) as caught:
            History.write(
                tmp_path / "contradicted.medh5",
                events=[
                    version("x1", 10 * HOUR),
                    version("x2", None),
                    version("x3", 5 * HOUR),
                ],
                links=[
                    Link.between(("event", "x2"), "supersedes", ("event", "x1")),
                    Link.between(("event", "x3"), "supersedes", ("event", "x2")),
                ],
            )
        assert caught.value.code == "E816"
        assert "'x3' was available by" in str(caught.value)

    def test_S4_a_null_cell_holds_nothing(self, history: Path):
        with h5py.File(history, "r+") as f:
            valid = f["clinical/events/valid/value_num"][()]
            row = int(np.flatnonzero(valid == 0)[0])
            f["clinical/events/value_num"][row] = 7.0
        assert "E807" in codes(history, "structural")

    def test_S3_undeclared_content_is_E803(self, history: Path):
        with h5py.File(history, "r+") as f:
            f.attrs["medh5_profiles"] = ["core", "longitudinal", "seg"]
        assert "E803" in codes(history, "structural")

    @pytest.mark.parametrize("dropped", ["source_end", "source_start"])
    def test_S7_1_a_span_is_both_endpoints_or_neither(self, tmp_path: Path, dropped):
        """A link record has a span or none, so a stored half-null pair
        reached the span rules as no span, validated, and lost the endpoint
        that was there on export."""
        path = tmp_path / "spans.medh5"
        grounding = Link.between(
            ("document", "rep_text_v1"),
            "describes",
            ("instance", "1"),
            span=(0, 8),
            target_annotation_id="lesions_tp0",
        )
        History.write(path, links=[grounding])
        assert "E814" not in codes(path)
        with h5py.File(path, "r+") as f:
            links = f["clinical/links"]
            # The kit's other links carry no span, so the masks exist.
            row = int(np.flatnonzero(links["valid/source_start"][()] == 1)[0])
            links["valid"][dropped][row] = 0
            links[dropped][row] = 0  # a null cell holds 0
        assert "E814" in codes(path)

    def test_S8_edited_text_is_found_under_an_unchanged_root(self, history: Path):
        with h5py.File(history, "r+") as f:
            data = f["clinical/documents/text/data"]
            data[0] = data[0] ^ 1
        assert "E701" in codes(history, "integrity")
        with medh5.open(history) as s:
            assert not s.verify().ok


def with_notes(path: Path, n: int = 6, size: int = 24_000) -> dict[str, str]:
    """The kit's history plus `n` long notes: a text buffer big enough to be
    chunked and compressed.  Returns each note's text by document id."""
    notes = {
        f"note{i}_text": f"note {i}: " + "lorem ipsum dolor sit amet " * (size // 27)
        for i in range(n)
    }
    History.write(
        path,
        events=[
            Event(
                f"note{i}",
                f"note{i}",
                "document",
                "point",
                "final",
                effective_start_us=i * DAY,
                available_us=i * DAY + HOUR,
            )
            for i in range(n)
        ],
        documents=[Document(k, v) for k, v in notes.items()],
        links=[
            Link.between(
                ("event", f"note{i}"), "describes", ("document", f"note{i}_text")
            )
            for i in range(n)
        ],
    )
    return notes


class TestDocumentText:
    """1.1 §6: a document's text is read when it is asked for --- its own
    bytes, checked as UTF-8 --- and opening or selecting reads none."""

    def test_S6_opening_and_selecting_read_no_text(self, tmp_path: Path):
        path = tmp_path / "notes.medh5"
        notes = with_notes(path)
        with h5py.File(path, "r+") as f:
            data = f["clinical/documents/text/data"]
            assert data.chunks is not None and data.compression == "gzip"
            # Damage the buffer's first compressed chunk: whatever reads it fails.
            data.id.write_direct_chunk((0,), b"\x00" * 64)
        with medh5.open(path) as s:
            c = s.clinical
            assert c is not None
            sizes = {d.document_id: d.n_bytes for d in c.documents}
            assert sizes["note5_text"] == len(notes["note5_text"].encode())
            assert c.select(10 * DAY).certified
            # The first note's bytes are in the damaged chunk; the last
            # report's are not, and read without it.
            with pytest.raises(OSError):
                c.text("note0_text")
            assert c.text("rep_text_v2") == History.documents()[1].text
            assert s.document_text("note5_text") == notes["note5_text"]

    def test_S4_text_is_checked_as_utf8_when_it_is_read(self, history: Path):
        with h5py.File(history, "r+") as f:
            data = f["clinical/documents/text/data"]
            raw = data[...]
            raw[0] = 0xFF  # the first byte of `rep_text_v1`, first by id
            data[...] = raw
        with medh5.open(history) as s:
            c = s.clinical  # nothing of the text was read to open it
            assert c is not None
            assert c.text("rep_text_v2").endswith("Final.")
            with pytest.raises(MEDH5ValidationError) as found:
                c.text("rep_text_v1")
            assert found.value.code == "E806"
        assert "E806" in codes(history)

    def test_S6_one_document_reads_without_the_events(self, history: Path):
        with h5py.File(history, "r+") as f:
            del f["clinical/events/kind"]  # the events table is now malformed
        with medh5.open(history) as s:
            with pytest.raises(MEDH5ValidationError):
                s.clinical  # noqa: B018 - the property reads the events
            assert s.document_text("rep_text_v2").endswith("Final.")
            with pytest.raises(KeyError):
                s.document_text("no-such-document")

    def test_S4_the_validator_streams_long_text(self, tmp_path: Path):
        path = tmp_path / "notes.medh5"
        with_notes(path, n=3, size=1_500_000)  # longer than a scan slab each
        assert codes(path) == []
        with h5py.File(path, "r+") as f:
            data = f["clinical/documents/text/data"]
            offsets = f["clinical/documents/text/offsets"][...]
            # A two-byte character at the very end of the second note, cut in
            # half by the end of its row: invalid however the slabs fall.
            end = int(offsets[2])
            tail = data[end - 1 : end]
            assert tail.tobytes() == b" "
            data[end - 1] = 0xC3
        assert "E806" in codes(path)


class TestClocks:
    def test_S3_utc_clocks(self, tmp_path: Path):
        """Only relative clocks could be built; 1.1 §3 has three."""
        import datetime

        from medh5.clinical import clock_day, utc_us

        assert Clock.utc("c").to_json() == {"id": "c", "unit": "us", "reference": "utc"}
        assert Clock.from_json(Clock.shifted_utc("c").to_json()) == Clock.shifted_utc(
            "c"
        )
        path = tmp_path / "utc.medh5"
        History.write(path, subject_id="P-01", clock=Clock.utc("subject-utc"))
        assert validate_file(path).ok
        moment = datetime.datetime(2024, 3, 1, 9, 30, tzinfo=datetime.timezone.utc)
        assert utc_us(moment) == int(moment.timestamp()) * 1_000_000
        assert (
            utc_us(
                datetime.datetime(
                    1969, 12, 31, 23, 59, 59, 999_999, tzinfo=datetime.timezone.utc
                )
            )
            == -1
        )
        with pytest.raises(ValueError, match="no time zone"):
            utc_us(datetime.datetime(2024, 3, 1, 9, 30))
        assert clock_day(utc_us(moment)) == (
            utc_us(datetime.datetime(2024, 3, 1, tzinfo=datetime.timezone.utc)),
            utc_us(datetime.datetime(2024, 3, 2, tzinfo=datetime.timezone.utc)) - 1,
        )
        assert clock_day(-1) == (-DAY, -1)

    def test_S5_2_day_bounds_span_the_possible_instants(self):
        """Day-only information spans the day's possible instants after clock
        conversion and never implies midnight: with no zone, from the day's
        start at UTC+14 to its end at UTC-12."""
        import datetime
        import zoneinfo

        from medh5.clinical import day_bounds, utc_us

        day = datetime.date(2024, 3, 1)
        utc = datetime.timezone.utc
        midnight = utc_us(datetime.datetime(2024, 3, 1, tzinfo=utc))
        lo, hi = day_bounds(day)
        assert (lo, hi) == (midnight - 14 * HOUR, midnight + DAY + 12 * HOUR - 1)
        assert hi - lo + 1 == 50 * HOUR
        assert day_bounds(day, utc) == (midnight, midnight + DAY - 1)
        plus_two = datetime.timezone(datetime.timedelta(hours=2))
        assert day_bounds(day, plus_two) == (
            midnight - 2 * HOUR,
            midnight + DAY - 2 * HOUR - 1,
        )
        for zone in (utc, plus_two):
            zlo, zhi = day_bounds(day, zone)
            assert lo <= zlo and zhi <= hi, "a known zone narrows the unknown one"
        try:
            spring = day_bounds(datetime.date(2024, 3, 31), "Europe/Berlin")
        except zoneinfo.ZoneInfoNotFoundError:  # pragma: no cover - no tzdata
            pytest.skip("no IANA zone database")
        assert spring[1] - spring[0] + 1 == 23 * HOUR, "the day the clocks go forward"
        with pytest.raises(TypeError):
            day_bounds(datetime.datetime(2024, 3, 1, tzinfo=utc))


class TestSelection:
    def test_S9_1_the_worked_example_at_hour_24(self, history: Path):
        with medh5.open(history) as s:
            assert s.clinical is not None
            chosen = s.clinical.select(24 * HOUR)
        assert chosen.certified and chosen.status == "certified"
        assert chosen.event_ids == ["lab0", "ct0", "rep_v1"]
        assert chosen.admits("document", "rep_text_v1")
        assert not chosen.admits("document", "rep_text_v2")
        assert chosen.admits("image", "CT_tp0") and not chosen.admits("image", "CT_tp1")
        assert chosen.excluded == {"after_cutoff": 2}
        assert chosen.events[1].order_us == (0, 0)

    def test_S9_1_admits_any_looks_in_every_fragment(self):
        """``admits`` answers for fragment 0 unless told otherwise, so a
        payload of a subject's second fragment read as not admitted."""
        from medh5.clinical import Selection

        chosen = Selection.from_json(
            {
                "cutoff_us": 0,
                "policy": "strict_prospective",
                "status": "certified",
                "events": [],
                "links": [],
                "payloads": [[1, "image", "CT_tp1"]],
                "uncertain_records": [],
            }
        )
        assert not chosen.admits("image", "CT_tp1")
        assert chosen.admits("image", "CT_tp1", fragment=1)
        assert chosen.admits_any("image", "CT_tp1")
        assert not chosen.admits_any("image", "CT_tp0")
        assert not chosen.admits_any("annotation", "CT_tp1")

    def test_S9_1_an_uncertain_revision_is_uncertifiable(self, tmp_path: Path):
        path = tmp_path / "uncertain.medh5"
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
            documents=[Document("rep_text_v3", "Unknown when.")],
            links=[
                Link.between(
                    ("event", "rep_v3"), "describes", ("document", "rep_text_v3")
                ),
                Link.between(("event", "rep_v3"), "supersedes", ("event", "rep_v2")),
            ],
        )
        with medh5.open(path) as s:
            assert s.clinical is not None
            strict = s.clinical.select(72 * HOUR)
            provable = s.clinical.select(72 * HOUR, policy="latest_provable")
        assert strict.status == "uncertifiable" and strict.uncertain_records == ("rep",)
        assert not strict.certified
        assert provable.status == "provable"
        assert "rep_v2" in provable.event_ids

    def test_S9_1_policies(self, history: Path):
        with medh5.open(history) as s:
            assert s.clinical is not None
            window = s.clinical.select(24 * HOUR, SelectionPolicy(context_us=24 * HOUR))
            kinds = s.clinical.select(24 * HOUR, {"kinds": ["document"]})
            limited = s.clinical.select(
                95 * DAY, SelectionPolicy(max_events=2, keep="latest")
            )
        assert window.event_ids == ["ct0", "rep_v1"]
        assert window.excluded["outside_context"] == 1
        assert kinds.event_ids == ["rep_v1"]
        assert limited.event_ids == ["ct1", "recist1"]

    def test_S7_3_an_eligible_version_attests_whatever_a_row_reads(
        self, tmp_path: Path
    ):
        """An asserting version need only be known at the cutoff.  The kinds,
        plan, unknown-time, window and limit filters choose what a row reads,
        and withdrew what a version they left out asserts: under the default
        ordering, a segmentation of unknown time never made its mask eligible,
        so no crop ever centred on it."""
        path = tmp_path / "seg.medh5"
        History.write(
            path,
            events=[
                Event(
                    "seg0",
                    "seg0",
                    "procedure",
                    "unknown",
                    "completed",
                    available_us=2 * HOUR,
                )
            ],
            links=[
                Link.between(
                    ("event", "ct0"),
                    "describes",
                    ("annotation", "lesions_tp0"),
                    asserted_by="seg0",
                )
            ],
        )
        with medh5.open(path) as s:
            assert s.clinical is not None
            early = s.clinical.select(HOUR)
            default = s.clinical.select(24 * HOUR)
            imaging = s.clinical.select(24 * HOUR, {"kinds": ["imaging"]})
            documents = s.clinical.select(24 * HOUR, {"kinds": ["document"]})
        assert early.admits("image", "CT_tp0")
        assert not early.admits("annotation", "lesions_tp0"), "not known at 1 h"
        for chosen in (default, imaging):
            assert "seg0" not in chosen.event_ids
            assert chosen.admits("annotation", "lesions_tp0")
        # The relationship still needs its event endpoint read.
        assert not documents.admits("annotation", "lesions_tp0")
        assert SelectionPolicy.from_json(SelectionPolicy(kinds=("imaging",)).to_json())

    @pytest.mark.parametrize("order_by", ["effective", "available"])
    def test_S7_3_an_asserter_is_its_own_endpoint(self, tmp_path: Path, order_by: str):
        """The worked example asserts ``seg0 describes lesions_tp0`` from
        ``seg0`` itself.  Under effective ordering a procedure of unknown time
        is never admitted, and the asserter had to be admitted when it was also
        the endpoint: the mask was not attested, while the same evidence
        phrased from ``ct0`` attested it."""
        seg0 = Event(
            "seg0", "seg0", "procedure", "unknown", "completed", available_us=2 * HOUR
        )
        chosen = {}
        for source in ("seg0", "ct0"):
            path = tmp_path / f"{source}.medh5"
            History.write(
                path,
                events=[seg0],
                links=[
                    Link.between(
                        ("event", source),
                        "describes",
                        ("annotation", "lesions_tp0"),
                        asserted_by="seg0",
                    )
                ],
            )
            with medh5.open(path) as s:
                assert s.clinical is not None
                chosen[source] = s.clinical.select(24 * HOUR, {"order_by": order_by})
        for selection in chosen.values():
            assert selection.admits("annotation", "lesions_tp0")
        assert chosen["seg0"].event_ids == chosen["ct0"].event_ids

    def test_S9_1_a_window_bounds_plans_from_below(self):
        """A seven-day window at day +1 excluded a
        completed event of day -100 and admitted the same event planned.  The
        window's lower edge holds for plans; only the cutoff is not theirs."""

        def event(event_id: str, status: str, t: int, available: int) -> Event:
            return Event(
                event_id,
                event_id,
                "procedure",
                "point",
                status,
                effective_start_us=t,
                available_us=available,
            )

        events = [
            event("done", "completed", -100 * DAY, -100 * DAY),
            event("stale", "planned", -100 * DAY, -100 * DAY),
            event("soon", "planned", -3 * DAY, -3 * DAY),
            event("future", "planned", 30 * DAY, 0),
        ]
        policy = SelectionPolicy(context_us=7 * DAY, plans=True)
        chosen = clinical.select(events, [], DAY, policy)
        assert chosen.event_ids == ["soon", "future"]
        assert chosen.excluded == {"outside_context": 2}
        assert all(e.plan for e in chosen.events)

    def test_S5_2_an_interval_is_read_with_the_fields_of_its_eligible_version(self):
        """A course whose end was learned later is a new version (1.1 §5.2)."""
        course = Event(
            "rx_v1",
            "rx",
            "medication_administration",
            "interval",
            "in_progress",
            effective_start_us=0,
            available_us=HOUR,
        )
        ended = Event(
            "rx_v2",
            "rx",
            "medication_administration",
            "interval",
            "completed",
            effective_start_us=0,
            effective_end_us=(20 * DAY, 21 * DAY),
            available_us=30 * DAY,
        )
        links = [Link.between(("event", "rx_v2"), "supersedes", ("event", "rx_v1"))]
        at_day_ten = clinical.select([course, ended], links, 10 * DAY)
        assert at_day_ten.event_ids == ["rx_v1"] and at_day_ten.certified
        assert clinical.select([course, ended], links, 31 * DAY).event_ids == ["rx_v2"]

    def test_S5_2_day_precision_is_never_narrowed(self):
        """A result known to the day is available somewhere in that day."""
        day = Event(
            "lab",
            "lab",
            "observation",
            "point",
            "final",
            effective_start_us=(3 * DAY, 4 * DAY - 1),
            available_us=(3 * DAY, 4 * DAY - 1),
            code_system="local",
            code="x",
            value_text="high",
        )
        assert clinical.select([day], [], 3 * DAY + 12 * HOUR).event_ids == []
        later = clinical.select([day], [], 4 * DAY)
        assert later.event_ids == ["lab"]
        assert later.events[0].order_us == (3 * DAY, 4 * DAY - 1)

    def test_S5_2_an_order_is_not_an_administration(self):
        order = Event(
            "order",
            "order",
            "medication_order",
            "point",
            "planned",
            effective_start_us=10 * DAY,
            available_us=DAY,
        )
        given = Event(
            "given",
            "given",
            "medication_administration",
            "point",
            "completed",
            effective_start_us=10 * DAY,
            available_us=10 * DAY + HOUR,
        )
        events = [order, given]
        assert clinical.select(events, [], 2 * DAY).event_ids == []
        plans = clinical.select(events, [], 2 * DAY, SelectionPolicy(plans=True))
        assert plans.event_ids == ["order"] and plans.events[0].plan
        assert clinical.select(events, [], 11 * DAY).event_ids == ["given"]

    def test_S9_1_ties_are_kept_or_dropped_whole(self):
        def at(event_id: str, lo: int, hi: int) -> Event:
            return Event(
                event_id,
                event_id,
                "other",
                "point",
                "final",
                effective_start_us=(lo, hi),
                available_us=hi,
            )

        events = [at("a", 0, 0), at("b", DAY, 2 * DAY), at("c", DAY, 2 * DAY)]
        keep = clinical.select(events, [], 3 * DAY, SelectionPolicy(max_events=1))
        drop = clinical.select(
            events, [], 3 * DAY, SelectionPolicy(max_events=1, ties="drop_group")
        )
        assert keep.event_ids == ["b", "c"]
        assert keep.events[0].tie_group == keep.events[1].tie_group
        assert drop.event_ids == []
        assert drop.excluded["event_limit"] == 3

    @pytest.mark.parametrize("keep", ["latest", "earliest"])
    def test_S9_1_a_limit_of_zero_keeps_no_timed_event(self, keep):
        """The schema admits 0, which panicked the engine."""
        timed = Event(
            "t", "t", "other", "point", "final", effective_start_us=0, available_us=HOUR
        )
        fact = Event("s", "s", "other", "static", "final", available_us=0)
        chosen = clinical.select(
            [timed, fact], [], DAY, SelectionPolicy(max_events=0, keep=keep)
        )
        assert chosen.event_ids == ["s"]
        assert chosen.excluded["event_limit"] == 1

    @pytest.mark.parametrize("order_by", ["effective", "available"])
    def test_S9_1_a_static_fact_is_not_a_timed_event(self, order_by):
        """Ordered by availability, a static fact counted against the
        limit on timed events and was dropped."""
        timed = Event(
            "t",
            "t",
            "other",
            "point",
            "final",
            effective_start_us=0,
            available_us=2 * HOUR,
        )
        fact = Event("s", "s", "other", "static", "final", available_us=HOUR)
        chosen = clinical.select(
            [timed, fact], [], DAY, SelectionPolicy(max_events=1, order_by=order_by)
        )
        assert chosen.event_ids == ["s", "t"]

    def test_S9_1_select_from_records_not_in_a_file(self):
        events = History.events()
        chosen = clinical.select(events, History.links(), 24 * HOUR)
        assert chosen.event_ids == ["lab0", "ct0", "rep_v1"]
        as_dicts = clinical.select(
            [e.to_json() for e in events],
            [link.to_json() for link in History.links()],
            0,
        )
        assert as_dicts.event_ids == ["lab0"]


class TestCollections:
    def test_S8_mixed_versions_keep_their_own(
        self, history: Path, sample_path: Path, tmp_path: Path
    ):
        shard = pack([history, sample_path], tmp_path / "mixed.medh5c", keys=["a", "b"])
        with open_collection(shard) as c:
            assert c.version == "1.1"
            assert c["a"].version == "1.1" and c["b"].version == "1.0"
            assert c["a"].clinical is not None and c["b"].clinical is None
        assert codes(shard) == []
        out = unpack(shard, tmp_path / "out")
        with medh5.open(out[0]) as a, medh5.open(history) as b:
            assert a.content_id == b.content_id

    def test_S8_recompression_keeps_the_address(self, history: Path, tmp_path: Path):
        with medh5.open(history) as s:
            before = s.content_id
        out = tmp_path / "re.medh5"
        recompress(history, profile="archive", out=out)
        with medh5.open(out) as s:
            assert s.content_id == before
            assert s.clinical is not None and len(s.clinical.events) == 6


def test_write_sample_helper_stays_1_0(tmp_path: Path, label_set, masks):
    path = write_sample(tmp_path / "plain.medh5", label_set=label_set, masks=masks)
    with medh5.open(path) as s:
        assert s.version == "1.0"


class TestProjection:
    """An imaging projection holds no byte of the history it drops.
    ``clinical strip`` copied the file, then amended the
    copy to unlink ``clinical/``; HDF5 does not reclaim an unlinked group's
    space, so the output verified as 1.0 while every record --- a report's
    text included --- was still in its bytes."""

    MARKER = "SYNTHETIC-MARKER-7f3c9a1e"

    def _history(self, path: Path, codec: str) -> Path:
        History.write(
            path,
            events=[
                Event(
                    "note1",
                    "note1",
                    "document",
                    "point",
                    "final",
                    effective_start_us=DAY,
                    available_us=DAY,
                )
            ],
            documents=[Document("note_text", f"{self.MARKER} in a note.")],
            links=[
                Link.between(("event", "note1"), "describes", ("document", "note_text"))
            ],
        )
        if codec != "portable":
            recompress(path, codec)
        return path

    def _bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    @pytest.mark.parametrize("codec", ["portable", "balanced"])
    def test_S10_strip_leaves_no_record_in_the_bytes(self, tmp_path: Path, codec: str):
        import hashlib

        source = self._history(tmp_path / "history.medh5", codec)
        assert self.MARKER.encode() in self._bytes(source), "the control"
        digest = hashlib.sha256(self._bytes(source)).hexdigest()
        out = tmp_path / "imaging.medh5"
        report = clinical.strip(source, out)
        assert report["version_after"] == "1.0"
        assert self.MARKER.encode() not in self._bytes(out)
        assert b"note_text" not in self._bytes(out)
        assert hashlib.sha256(self._bytes(source)).hexdigest() == digest
        with medh5.open(out) as s:
            assert s.clinical is None and s.verify().ok
        # A read-only scan of the projection treats it as any imaging sample.
        from medh5.curation import scrub as scrubber

        scrubber.scan(out)

    def test_S10_the_command_line_strip_leaves_no_record_either(self, tmp_path: Path):
        from medh5.cli import main

        source = self._history(tmp_path / "history.medh5", "portable")
        out = tmp_path / "cli.medh5"
        assert main(["clinical", "strip", str(source), "--out", str(out)]) == 0
        assert self.MARKER.encode() not in self._bytes(out)

    @pytest.mark.parametrize("readd", [False, True], ids=["dropped", "re-added"])
    def test_S10_drop_clinical_leaves_no_record_in_the_bytes(
        self, tmp_path: Path, readd: bool
    ):
        path = self._history(tmp_path / "history.medh5", "portable")
        with medh5.amend(path) as w:
            w.drop_clinical()
            if readd:
                w.set_clock(Clock.relative("clock", "baseline CT acquisition"))
                w.add_event(History.events()[1])
                w.add_link(History.links()[0])
            w.commit()
        data = self._bytes(path)
        assert self.MARKER.encode() not in data
        assert b"rep_text_v1" not in data
        with medh5.open(path) as s:
            assert s.verify().ok
            if readd:
                assert s.clinical is not None
                assert [e.event_id for e in s.clinical.records().events] == ["ct0"]
            else:
                assert s.clinical is None and s.version == "1.0"

    def test_S10_a_refused_strip_writes_nothing(self, tmp_path: Path, sample_path):
        out = tmp_path / "nothing.medh5"
        with pytest.raises(MEDH5ValidationError):
            clinical.strip(sample_path, out)  # no clinical profile to strip
        assert not out.exists()
        existing = tmp_path / "existing.medh5"
        existing.write_bytes(b"keep me")
        source = self._history(tmp_path / "history.medh5", "portable")
        with pytest.raises(MEDH5ValidationError):
            clinical.strip(source, existing)
        assert existing.read_bytes() == b"keep me"
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


class TestOutputPaths:
    """No clinical command writes over the file it reads (F11).  ``clinical
    export S -o S`` truncated the sample and wrote the JSON describing it in
    its place, with exit status 0."""

    @staticmethod
    def _alias(path: Path, tmp_path: Path, alias: str) -> Path:
        if alias == "exact":
            return path
        if alias == "relative":
            (tmp_path / "sub").mkdir()
            return tmp_path / "sub" / ".." / path.name
        out = tmp_path / "alias.medh5"
        try:
            (out.symlink_to if alias == "symlink" else out.hardlink_to)(path)
        except OSError as exc:  # pragma: no cover - no links on this file system
            pytest.skip(str(exc))
        return out

    @pytest.mark.parametrize("alias", ["exact", "relative", "symlink", "hardlink"])
    def test_S10_no_output_names_the_sample_it_reads(
        self, history: Path, tmp_path: Path, capsys, alias: str
    ):
        from medh5.cli import EXIT_ERROR, EXIT_OK, main

        records = tmp_path / "records.json"
        assert main(["clinical", "export", str(history), "-o", str(records)]) == 0
        out = str(self._alias(history, tmp_path, alias))
        before = history.read_bytes()
        capsys.readouterr()
        assert main(["clinical", "export", str(history), "--out", out]) == EXIT_ERROR
        assert "the file being exported" in capsys.readouterr().err
        assert history.read_bytes() == before
        # The commands that write a sample refuse it too.
        others = [["clinical", "strip", str(history), "--out", out]]
        if alias != "exact":  # `augment S R -o S` is the in-place augmentation
            others.append(
                ["clinical", "augment", str(history), str(records), "-o", out]
            )
        for argv in others:
            assert main(argv) == EXIT_ERROR, argv
            assert history.read_bytes() == before, argv
        with medh5.open(history) as s:
            assert s.clinical is not None and len(s.clinical.events) == 6
        assert main(["clinical", "export", str(history)]) == EXIT_OK

    def test_S10_export_refuses_the_collection_it_reads(
        self, history: Path, tmp_path: Path, capsys
    ):
        from medh5.cli import EXIT_ERROR, main

        shard = pack([history], tmp_path / "one.medh5c", keys=["a"])
        before = shard.read_bytes()
        argv = ["clinical", "export", str(shard), "--key", "a", "-o", str(shard)]
        assert main(argv) == EXIT_ERROR
        assert "the file being exported" in capsys.readouterr().err
        assert shard.read_bytes() == before
