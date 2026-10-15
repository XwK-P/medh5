"""MEDS import (1.1 §5, §10): one subject's rows as clinical records, with
nothing invented --- no zone, no origin, no unit, no availability, no code."""

from __future__ import annotations

import datetime
import zoneinfo
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from medh5.clinical import DAY, HOUR, Clock, day_bounds, utc_us
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.meds import augment_from_meds, read_meds
from medh5.sample import open_sample
from medh5.validate import validate_file
from tests.kits import Meds

NEW_YORK = "America/New_York"


def _utc(**options):
    return {"subject_id": Meds.SUBJECT, "clock": Clock.utc("ehr"), **options}


def _by_code(records):
    return {e.code: e for e in records.events}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return Meds.write(tmp_path / "meds")


def test_S10_meds_naive_times_need_a_zone(dataset):
    with pytest.raises(MEDH5ValidationError, match="no time zone.*give timezone"):
        read_meds(dataset, **_utc())
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    lab = _by_code(records)["LOINC/2160-0"]
    expected = utc_us(
        datetime.datetime(2020, 3, 1, 8, 15, tzinfo=zoneinfo.ZoneInfo(NEW_YORK))
    )
    assert lab.temporal_type == "point"
    assert lab.effective_start_us == (expected, expected)
    assert _by_code(records)["GENDER//F"].temporal_type == "static"
    assert not report.guesses


def test_S10_a_reading_that_falls_twice_is_bounded_by_both(tmp_path):
    rows = [
        {
            "subject_id": Meds.SUBJECT,
            "time": datetime.datetime(2020, 11, 1, 1, 30),
            "code": "X",
            "numeric_value": None,
            "text_value": None,
        }
    ]
    records, _ = read_meds(Meds.write(tmp_path, rows), **_utc(timezone=NEW_YORK))
    lo, hi = records.events[0].effective_start_us
    assert hi - lo == HOUR
    rows[0]["time"] = datetime.datetime(2020, 3, 8, 2, 30)
    with pytest.raises(MEDH5ValidationError, match="never occurred"):
        read_meds(Meds.write(tmp_path / "gap", rows), **_utc(timezone=NEW_YORK))


def test_S10_a_relative_clock_needs_an_origin(dataset):
    clock = Clock.relative("ehr", "admission, 2020-03-01 00:00 local")
    with pytest.raises(MEDH5ValidationError, match="invents no origin"):
        read_meds(dataset, subject_id=Meds.SUBJECT, clock=clock)
    records, report = read_meds(
        dataset, subject_id=Meds.SUBJECT, clock=clock, origin="2020-03-01T00:00:00"
    )
    lab = _by_code(records)["LOINC/2160-0"]
    assert lab.effective_start_us == (8 * HOUR + 15 * 60 * 10**6,) * 2
    assert report.of_kind("time")  # counted on one wall clock, and said so
    with pytest.raises(MEDH5ValidationError, match="origin is for a relative clock"):
        read_meds(dataset, **_utc(timezone=NEW_YORK, origin="2020-03-01T00:00:00"))


def test_S5_2_meds_availability_stays_unknown(tmp_path, dataset):
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    assert all(e.available_us is None for e in records.events)
    (note,) = report.of_kind("availability")
    assert note.severity == "warning" and "never its inputs" in note.message
    known = [datetime.datetime(2020, 3, 6, 9, 0)] * len(Meds.rows())
    declared = Meds.write(
        tmp_path / "declared",
        extra={"reported": (__import__("pyarrow").timestamp("us"), known)},
    )
    records, report = read_meds(
        declared, **_utc(timezone=NEW_YORK, columns={"available": "reported"})
    )
    when = utc_us(datetime.datetime(2020, 3, 6, 9, tzinfo=zoneinfo.ZoneInfo(NEW_YORK)))
    assert {e.available_us for e in records.events} == {(when, when)}
    assert not report.of_kind("availability")


def test_S5_2_a_day_spans_all_of_its_instants(dataset):
    records, _ = read_meds(dataset, **_utc(day_precision=["DX"], timezone=NEW_YORK))
    dx = _by_code(records)["DX//I10"]
    assert dx.effective_start_us == day_bounds(datetime.date(2020, 3, 1), NEW_YORK)
    relative = Clock.relative("ehr", "admission")
    records, _ = read_meds(
        dataset,
        subject_id=Meds.SUBJECT,
        clock=relative,
        origin="2020-03-01T00:00:00",
        day_precision=["DX"],
    )
    assert _by_code(records)["DX//I10"].effective_start_us == (0, DAY - 1)


def test_S5_1_no_unit_is_invented(tmp_path, dataset):
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    lab = _by_code(records)["LOINC/2160-0"]
    assert lab.value_num == 1.1  # the float32 MEDS stores, written shortest
    assert lab.unit is None and report.of_kind("unit")
    sample = Meds.sample(tmp_path / "s.medh5")
    augment_from_meds(sample, dataset, **_utc(timezone=NEW_YORK))
    assert "W914" in validate_file(sample).codes
    units = ["mg/dL"] * len(Meds.rows())
    declared = Meds.write(
        tmp_path / "units", extra={"unit": (__import__("pyarrow").string(), units)}
    )
    records, _ = read_meds(
        declared, **_utc(timezone=NEW_YORK, columns={"unit": "unit"})
    )
    found = _by_code(records)
    assert found["LOINC/2160-0"].unit == "mg/dL"
    assert found["GENDER//F"].unit is None  # a unit belongs to a value


def test_S5_1_both_values_are_refused(tmp_path):
    rows = Meds.rows()
    rows[2]["text_value"] = "high"
    root = Meds.write(tmp_path, rows)
    with pytest.raises(MEDH5ValidationError, match="both_values"):
        read_meds(root, **_utc(timezone=NEW_YORK))
    for keep, value in (("numeric", (1.1, None)), ("text", (None, "high"))):
        records, report = read_meds(root, **_utc(timezone=NEW_YORK, both_values=keep))
        lab = _by_code(records)["LOINC/2160-0"]
        assert (lab.value_num, lab.value_text) == value
        assert report.of_kind("value")[0].severity == "decision"


def test_S5_1_a_code_system_is_declared_never_guessed(tmp_path, dataset):
    records, _ = read_meds(
        dataset,
        **_utc(
            timezone=NEW_YORK,
            code_systems={"LOINC": "http://loinc.org"},
            kinds={"LOINC": "observation", "DX": "diagnosis"},
        ),
    )
    found = {(e.code_system, e.code): e.kind for e in records.events}
    assert found[("http://loinc.org", "2160-0")] == "observation"
    assert found[("meds:demo-ehr", "LAB//50912")] == "other"
    assert found[("meds:demo-ehr", "DX//I10")] == "diagnosis"
    unnamed = Meds.write(tmp_path / "unnamed", name=None)
    with pytest.raises(MEDH5ValidationError, match="give code_system"):
        read_meds(unnamed, **_utc(timezone=NEW_YORK))
    records, _ = read_meds(unnamed, **_utc(timezone=NEW_YORK, code_system="local"))
    assert {e.code_system for e in records.events} == {"local"}
    with pytest.raises(MEDH5ValidationError, match="event kind"):
        read_meds(dataset, **_utc(timezone=NEW_YORK, kinds={"LAB": "lab"}))


def test_S10_birth_and_text_are_opt_in(dataset):
    options = _utc(timezone=NEW_YORK, document_codes=["NOTE//discharge"])
    records, report = read_meds(dataset, **options)
    assert "MEDS_BIRTH" not in _by_code(records)
    note = _by_code(records)["NOTE//discharge"]
    assert note.kind == "document" and note.value_text is None
    assert not records.documents and not records.links
    assert report.of_kind("excluded") and report.of_kind("text")
    records, report = read_meds(
        dataset, **{**options, "exclude_codes": (), "texts": True}
    )
    assert "MEDS_BIRTH" in _by_code(records)
    (document,) = records.documents
    assert document.text == "Discharged home."
    (link,) = records.links
    assert (link.source_id, link.relation, link.target_id) == (
        _by_code(records)["NOTE//discharge"].event_id,
        "describes",
        document.document_id,
    )
    assert not report.of_kind("text")


def test_S10_ids_survive_reextraction(tmp_path, dataset):
    first, _ = read_meds(dataset, **_utc(timezone=NEW_YORK))
    rows = Meds.rows()[::-1]
    again = Meds.write(tmp_path / "again", rows, shards=3)
    second, _ = read_meds(again, **_utc(timezone=NEW_YORK))
    assert {e.event_id for e in first.events} == {e.event_id for e in second.events}
    assert all(e.event_id.startswith("meds.") for e in first.events)
    assert all(e.event_id == e.record_id for e in first.events)
    doubled = Meds.write(tmp_path / "doubled", [*Meds.rows(), Meds.rows()[2]])
    records, report = read_meds(doubled, **_utc(timezone=NEW_YORK))
    ids = sorted(e.event_id for e in records.events)
    assert len(set(ids)) == len(ids) and any(i.endswith(".2") for i in ids)
    assert report.of_kind("id")


def test_S10_a_reimport_is_refused(tmp_path, dataset):
    sample = Meds.sample(tmp_path / "s.medh5")
    options = _utc(timezone=NEW_YORK)
    result, report = augment_from_meds(sample, dataset, **options)
    assert result["version_after"] == "1.1"
    assert report.outputs == [str(sample)]
    with open_sample(sample) as s:
        before = s.content_id
        assert len(s.clinical.events) == 5
    with pytest.raises(MEDH5Error, match="meds\\."):
        augment_from_meds(sample, dataset, **options)
    with open_sample(sample) as s:
        assert s.content_id == before


def test_S10_the_subject_is_the_callers_mapping(tmp_path, dataset):
    with pytest.raises(MEDH5ValidationError, match="no row of MEDS subject 7"):
        read_meds(dataset, **{**_utc(timezone=NEW_YORK), "subject_id": 7})
    _, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    (note,) = report.of_kind("subject")
    assert note.severity == "decision" and note.detail["meds_subject_id"] == 1001
    old = Meds.write(tmp_path / "old", rename={"subject_id": "patient_id"})
    with pytest.raises(MEDH5ValidationError, match="patient_id"):
        read_meds(old, **_utc(timezone=NEW_YORK))
    records, _ = read_meds(
        old, **_utc(timezone=NEW_YORK, columns={"subject_id": "patient_id"})
    )
    assert len(records.events) == 5
    shard, _ = read_meds(dataset / "data" / "0.parquet", **_utc(timezone=NEW_YORK))
    assert {e.code_system for e in shard.events} == {"meds:demo-ehr"}


def test_clinical_augment_from_meds(tmp_path, dataset, capsys):
    """`medh5 clinical augment --from-meds`: the import, its report kept as
    JSON, and exit 1 for the availability warning MEDS always earns."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_USAGE, main

    sample = Meds.sample(tmp_path / "s.medh5")
    out, report = tmp_path / "out.medh5", tmp_path / "report.json"
    argv = [
        "clinical", "augment", str(sample), "--from-meds", str(dataset),
        "--meds-subject", "1001", "--clock-id", "ehr", "--clock", "utc",
        "--timezone", NEW_YORK, "--code-system", "LOINC=http://loinc.org",
        "--kind", "LOINC=observation", "--day-precision", "DX",
        "--document-code", "NOTE//discharge", "--with-text",
        "--report", str(report), "-o", str(out),
    ]  # fmt: skip
    assert main(argv) == EXIT_ERROR
    printed = capsys.readouterr().out
    assert "MEDH5 1.0 -> 1.1; 5 events, 1 documents, 1 links" in printed
    assert "WARNING  availability" in printed
    kept = json.loads(report.read_text(encoding="utf-8"))
    assert kept["converter"] == "from-meds" and kept["outputs"] == [str(out)]
    with open_sample(out) as s:
        assert s.clinical.clock == Clock.utc("ehr")
        found = {e.code_system for e in s.clinical.events}
        assert "http://loinc.org" in found
    # Again, onto the sample's own clock: the same rows are refused.
    again = ["clinical", "augment", str(out), "--from-meds", str(dataset),
             "--meds-subject", "1001", "--timezone", NEW_YORK, "--json"]  # fmt: skip
    assert main(again) == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert main([*argv[:3], "--from-meds", str(dataset)]) == EXIT_USAGE
    capsys.readouterr()
    plain = Meds.sample(tmp_path / "plain.medh5")
    assert main(["clinical", "augment", str(plain), "--from-meds", str(dataset),
                 "--meds-subject", "1001"]) == EXIT_ERROR  # fmt: skip
    assert "--clock-id and --clock" in capsys.readouterr().err
