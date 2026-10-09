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
from tests.helpers import write_sample
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
        assert medh5.FORMAT_VERSION == "1.1"
        from medh5._core import written_version

        assert written_version(None, ["core"]) == "1.0"
        assert written_version(None, ["core", "clinical"]) == "1.1"
        assert written_version("1.1", ["core"]) == "1.1"  # never downgraded

    def test_S2_2_a_higher_minor_is_a_projection(self, history: Path, tmp_path: Path):
        future = tmp_path / "future.medh5"
        shutil.copyfile(history, future)
        with h5py.File(future, "r+") as f:
            f.attrs["medh5_version"] = "1.2"
            f["clinical/events"].attrs["defined_by"] = "MEDH5 1.2"
        with medh5.open(future) as s:
            assert s.support == "projection"
            assert s.clinical is not None and s.clinical.projection
            assert len(s.clinical.events) == 6
        report = validate_file(future, level="integrity")
        assert codes(future, "integrity") == ["W913"], report.codes
        where = {d.location for d in report.diagnostics if d.code == "W913"}
        # The version, the attribute 1.2 defined, and the root this engine
        # cannot recompute (a later minor may cover more): reported, not E702.
        assert {"/", "/clinical/events@defined_by"} <= where, where
        assert report.ok
        with pytest.raises(MEDH5VersionError), medh5.amend(future):
            pass
        with pytest.raises(MEDH5VersionError):
            recompress(future, profile="portable")
        with pytest.raises(MEDH5VersionError):
            pack([future], tmp_path / "shard.medh5c")

    def test_C06_a_member_of_a_text_column_follows_the_version(
        self, history: Path, tmp_path: Path
    ):
        """A member a UTF-8 column does not define is a later minor's under a
        projection (W913), as an unknown column is; it was E804 at any version."""
        for version, expected in (("1.1", "E804"), ("1.2", "W913")):
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

    def test_C11_S4_a_big_endian_numeric_column_is_E805(
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
            w.abort()

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


class TestValidation:
    def test_S4_columns_are_checked(self, history: Path):
        with h5py.File(history, "r+") as f:
            del f["clinical/events/available_lo_us"]
            f["clinical/events"].create_dataset(
                "available_lo_us", data=np.zeros(6, dtype=np.int32)
            )
        assert "E805" in codes(history, "structural")

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
        """C05: a link record has a span or none, so a stored half-null pair
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
        assert SelectionPolicy.from_json(SelectionPolicy(kinds=("imaging",)).to_json())

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
        """C01: the schema admits 0, which panicked the engine."""
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
        """C02: ordered by availability, a static fact counted against the
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
