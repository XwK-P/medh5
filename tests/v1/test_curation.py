"""Curation records and the sample document (spec §2.4, §11, §12)."""

from __future__ import annotations

import json

import pytest

import medh5
from medh5._hdf5 import (
    as_bool,
    as_float,
    as_float_tuple,
    as_int,
    as_int_tuple,
    as_matrix,
    as_str,
    as_str_tuple,
    encode_attr,
    validate_id,
    validate_sample_key,
)
from medh5.curation.identity import (
    Cohort,
    Deidentification,
    Identity,
    SplitClaim,
    splits_from_json,
)
from medh5.curation.provenance import (
    Activity,
    Agent,
    Provenance,
    check_timestamp,
)
from medh5.curation.quality import (
    Agreement,
    Issue,
    QualityRecord,
    dice_agreement,
    quality_from_json,
    quality_to_json,
)
from medh5.curation.timeline import Timeline, Timepoint
from medh5.document import SampleDocument, new_document, validate_against_schema
from medh5.errors import MEDH5SchemaError, MEDH5ValidationError


class TestProvenance:
    def test_S11_1_two_node_graph_describes_model_then_human(self):
        """The workflow a review-status field cannot express."""
        graph = Provenance(
            agents=[
                Agent("m1", "model", "nnU-Net", version="2.5.1"),
                Agent("r1", "person", "pseudonym:RAD-07", role="annotator"),
            ],
            activities=[
                Activity("a1", "predict", agent="m1", outputs=("annotations/organs",)),
                Activity(
                    "a2",
                    "annotate",
                    agent="r1",
                    inputs=("annotations/organs",),
                    outputs=("annotations/organs",),
                ),
            ],
        )
        assert graph
        assert len(graph.produced_by("annotations/organs")) == 2
        assert graph.activities_by_type("predict")[0].id == "a1"
        assert graph.agent("r1").role == "annotator"
        assert graph.has_activity("a2")
        assert not graph.dangling_agent_refs()
        assert [a.id for a in graph] == ["a1", "a2"]
        assert "2 activities" in repr(graph)

    def test_dangling_agents_are_reported_not_raised(self):
        graph = Provenance(activities=[Activity("a1", "import", agent="ghost")])
        assert graph.dangling_agent_refs() == (("a1", "ghost"),)

    def test_unknown_lookups_raise_clearly(self):
        graph = Provenance()
        assert not graph
        with pytest.raises(KeyError, match="unknown agent"):
            graph.agent("nope")
        with pytest.raises(KeyError, match="unknown activity"):
            graph.activity("nope")

    def test_unknown_types_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Agent("a", "wizard", "Merlin")
        assert exc.value.code == "E603"
        with pytest.raises(MEDH5ValidationError) as exc:
            Activity("a", "divination")
        assert exc.value.code == "E603"

    def test_S11_1_timestamps_must_be_rfc3339(self):
        Activity("a", "import", ended="2026-02-03T09:11:40Z")
        with pytest.raises(MEDH5ValidationError) as exc:
            Activity("a", "import", ended="yesterday")
        assert exc.value.code == "E604"
        with pytest.raises(MEDH5ValidationError):
            check_timestamp("2026-02-03", where="test")

    def test_json_round_trip_preserves_extras(self):
        graph = Provenance(
            agents=[Agent("a", "software", "medh5", extra={"x_site": "B"})],
            activities=[
                Activity("t", "import", params={"kernel": "B30f"}, extra={"x_run": 3})
            ],
        )
        back = Provenance.from_json(graph.to_json())
        assert back.agent("a").extra == {"x_site": "B"}
        assert back.activity("t").extra == {"x_run": 3}
        assert Provenance.from_json(None).to_json()["agents"] == []


class TestQuality:
    def test_S11_2_status_is_current_state_not_history(self):
        record = QualityRecord(
            status="approved",
            confidence=0.92,
            reviewed_by=("r2",),
            agreement=(Agreement("dice", 0.913, against="annotations/organs_rater2"),),
            issues=(Issue("boundary_uncertain", "info", (12,), "motion"),),
            edit_effort_s=640,
        )
        assert record.is_usable
        assert not QualityRecord(status="draft").is_usable
        payload = record.to_json()
        assert QualityRecord.from_json(payload) == record

    def test_unknown_status_and_severity_are_refused(self):
        with pytest.raises(MEDH5ValidationError):
            QualityRecord(status="vibes")
        with pytest.raises(MEDH5ValidationError):
            Issue("x", "catastrophic")

    def test_dice_agreement_averages_per_class(self):
        agreement = dice_agreement({5: 1.0, 12: 0.5}, against="annotations/other")
        assert agreement.value == pytest.approx(0.75)
        assert agreement.per_class == {"5": 1.0, "12": 0.5}
        assert Agreement.from_json(agreement.to_json()) == agreement
        assert dice_agreement({}).value == 0.0

    def test_mapping_helpers(self):
        records = {"organs": QualityRecord(status="approved")}
        assert quality_from_json(quality_to_json(records)) == records
        assert quality_from_json(None) == {}


class TestIdentity:
    def test_S12_1_identity_requires_both_ids(self):
        with pytest.raises(MEDH5ValidationError):
            Identity(sample_id="", subject_id="s")
        with pytest.raises(MEDH5ValidationError):
            Identity(sample_id="s", subject_id="s", sex="yes")
        with pytest.raises(MEDH5ValidationError):
            Identity(sample_id="s", subject_id="s", laterality="upwards")

    def test_S12_2_group_id_defaults_to_subject(self):
        assert Cohort().grouping_key("subj-A") == "subj-A"
        assert Cohort(group_id="family-3").grouping_key("subj-A") == "family-3"
        assert Cohort.from_json(None).to_json() == {}

    def test_S12_3_split_claims_round_trip(self):
        claim = SplitClaim("cv5", "train", fold=2, manifest_sha256="a" * 64)
        assert SplitClaim.from_json(claim.to_json()) == claim
        with pytest.raises(MEDH5ValidationError):
            SplitClaim("cv5", "everything")
        assert splits_from_json(None) == ()

    def test_S11_4_deidentification_round_trips(self):
        record = Deidentification(
            method="dicom-psi-profile",
            date_shift_days=-117,
            burned_in_annotation_checked=True,
            extra={"x_tool": "ctp"},
        )
        assert record is not None
        back = Deidentification.from_json(record.to_json())
        assert back == record
        assert Deidentification.from_json(None) is None


class TestTimeline:
    def test_sequence_protocol(self):
        timeline = Timeline([Timepoint("tp0", 0), Timepoint("tp1", 1)])
        assert len(timeline) == 2
        assert timeline[1].id == "tp1"
        assert timeline["tp0"].index == 0
        assert "tp1" in timeline
        assert timeline[0] in timeline
        assert timeline.baseline.id == "tp0"
        assert timeline.ids == ("tp0", "tp1")
        assert "tp0" in repr(timeline)

    def test_interval_is_none_without_both_endpoints(self):
        timeline = Timeline(
            [Timepoint("tp0", 0, days_from_baseline=0), Timepoint("tp1", 1)]
        )
        assert timeline.interval_days("tp0", "tp1") is None

    def test_require_reports_the_declared_set(self):
        timeline = Timeline.single()
        assert timeline.require("tp0").index == 0
        with pytest.raises(MEDH5ValidationError) as exc:
            timeline.require("tp9", where="annotations/x")
        assert exc.value.code == "E409"
        assert "annotations/x" in str(exc.value)

    def test_bad_ids_and_dates_are_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Timepoint("bad id", 0)
        assert exc.value.code == "E003"
        with pytest.raises(MEDH5ValidationError) as exc:
            Timepoint("tp0", 0, date="last tuesday")
        assert exc.value.code == "E604"
        with pytest.raises(MEDH5ValidationError):
            Timepoint("tp0", -1)

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(MEDH5ValidationError):
            Timeline([Timepoint("tp0", 0), Timepoint("tp0", 1)])

    def test_json_round_trip_keeps_extras(self):
        timeline = Timeline(
            [Timepoint("tp0", 0, series_uids={"CT": "s1"}, extra={"x_note": "n"})]
        )
        back = Timeline.from_json(timeline.to_json())
        assert back["tp0"].series_uids == {"CT": "s1"}
        assert back["tp0"].extra == {"x_note": "n"}


class TestDocument:
    def test_new_document_defaults_subject_to_sample(self):
        document = new_document("case_1")
        assert document.subject_id == "case_1"
        assert document.timepoints.ids == ("tp0",)
        assert new_document("c", timepoints=["a", "b"]).timepoints.ids == ("a", "b")

    def test_missing_required_members_are_named(self):
        with pytest.raises(MEDH5SchemaError, match="identity"):
            SampleDocument.from_json({"timepoints": [{"id": "tp0", "index": 0}]})

    def test_bad_json_is_reported(self):
        with pytest.raises(MEDH5SchemaError, match="not valid JSON"):
            SampleDocument.loads("{oops")
        with pytest.raises(MEDH5SchemaError, match="JSON object"):
            SampleDocument.loads("[1]")

    def test_round_trip_through_bytes(self):
        document = new_document("c")
        assert SampleDocument.loads(document.dumps().encode()).subject_id == "c"

    def test_schema_errors_are_readable(self):
        errors = validate_against_schema({"identity": {}, "timepoints": []})
        assert errors
        assert all(":" in message for message in errors)

    def test_written_documents_validate(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert sample.document.check_schema() == []
            json.dumps(sample.summary(), default=str)


class TestAttributeCodecs:
    def test_S2_5_types_round_trip(self):
        assert as_str(encode_attr("x")) == "x"
        assert as_str(b"x") == "x"
        assert as_str_tuple(encode_attr(["a", "b"])) == ("a", "b")
        assert as_str_tuple("solo") == ("solo",)
        assert as_int(encode_attr(3)) == 3
        assert as_int_tuple(encode_attr([1, 2])) == (1, 2)
        assert as_float(encode_attr(1.5)) == 1.5
        assert as_float_tuple(encode_attr([1.5, 2.5])) == (1.5, 2.5)
        assert as_bool(encode_attr(True)) is True

    def test_S2_5_matrices_stay_two_dimensional(self):
        import numpy as np

        assert as_matrix(np.eye(3)).shape == (3, 3)
        with pytest.raises(MEDH5ValidationError) as exc:
            as_matrix(np.zeros(9))
        assert exc.value.code == "E109"

    def test_empty_sequences_and_mixed_types(self):
        import numpy as np

        assert encode_attr([]).shape == (0,)
        assert encode_attr([True, False]).dtype == np.bool_
        assert encode_attr([1, 2.5]).dtype == np.float64
        with pytest.raises(MEDH5ValidationError):
            encode_attr(object())

    def test_S2_3_identifier_rules(self):
        assert validate_id("CT_tp0") == "CT_tp0"
        for bad in ("", "a b", "x" * 129, "meta"):
            with pytest.raises(MEDH5ValidationError) as exc:
                validate_id(bad)
            assert exc.value.code == "E003"
        assert validate_sample_key("a.b-c_1") == "a.b-c_1"
        with pytest.raises(MEDH5ValidationError):
            validate_sample_key("x" * 256)
