"""FHIR R4 import (1.1 §5, §10): precision becomes bounds, ``issued`` is
availability, versions supersede, and nothing FHIR leaves out is invented."""

from __future__ import annotations

import datetime
import zoneinfo
from pathlib import Path

import pytest

from medh5.clinical import HOUR, Clock, day_bounds, utc_us
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.fhir import augment_from_fhir, read_fhir
from medh5.sample import open_sample
from medh5.validate import validate_file
from tests.kits import Fhir

UTC = datetime.timezone.utc


def _zones() -> bool:
    try:
        zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:  # pragma: no cover - Windows, no tzdata
        return False
    return True


NAMED_ZONES = pytest.mark.skipif(
    not _zones(), reason="no IANA time-zone data here (Windows needs tzdata)"
)
"""On the tests that name a zone: Linux and macOS have the data, and run them."""


def _read(resources=None, **options):
    options = {"subject": Fhir.SUBJECT, "clock": Clock.utc("ehr"), **options}
    return read_fhir(Fhir.bundle(resources), **options)


def _events(records):
    return {e.event_id: e for e in records.events}


def _at(*parts, offset=0):
    zone = datetime.timezone(datetime.timedelta(hours=offset))
    return utc_us(datetime.datetime(*parts, tzinfo=zone))


@NAMED_ZONES
def test_S5_2_fhir_precision_becomes_bounds():
    records, report = _read()
    found = _events(records)
    t = _at(2020, 3, 1, 8, 15, offset=-5)
    assert found["obs.creat.v1"].effective_start_us == (t, t)
    # A date with no known zone spans every instant it covers anywhere.
    assert found["obs.k"].effective_start_us == day_bounds(datetime.date(2020, 3, 1))
    a1c = found["obs.a1c"]
    assert a1c.temporal_type == "interval" and a1c.effective_end_us is None
    lo, hi = a1c.effective_start_us
    assert lo == _at(2020, 2, 1) - 14 * HOUR and hi == _at(2020, 3, 1) + 12 * HOUR - 1
    assert found["obs.smoke"].temporal_type == "unknown"
    assert any("effectiveTiming" in n.message for n in report.warnings)
    records, _ = _read(timezone="America/New_York")
    assert _events(records)["obs.k"].effective_start_us == day_bounds(
        datetime.date(2020, 3, 1), "America/New_York"
    )
    year = [{**Fhir.resources()[3], "effectiveDateTime": "2020"}]
    records, _ = _read(year)
    assert records.events[0].effective_start_us == (
        _at(2020, 1, 1) - 14 * HOUR,
        _at(2021, 1, 1) + 12 * HOUR - 1,
    )


@NAMED_ZONES
def test_S10_a_time_without_an_offset_needs_a_zone():
    naive = [{**Fhir.resources()[3], "effectiveDateTime": "2020-03-01T08:00:00"}]
    with pytest.raises(MEDH5ValidationError, match="give timezone"):
        _read(naive)
    records, _ = _read(naive, timezone="Europe/Paris")
    assert records.events[0].effective_start_us == (_at(2020, 3, 1, 7),) * 2
    clock = Clock.relative("ehr", "admission")
    with pytest.raises(MEDH5ValidationError, match="invents no origin"):
        _read(clock=clock)
    records, _ = _read(clock=clock, origin="2020-03-01T00:00:00-05:00")
    assert (
        _events(records)["obs.creat.v1"].effective_start_us
        == (8 * HOUR + 15 * 60 * 10**6,) * 2
    )


def test_S5_2_issued_is_availability():
    records, report = _read()
    found = _events(records)
    issued = _at(2020, 3, 1, 9, 15, offset=-5)
    assert found["obs.creat.v1"].available_us == (issued, issued)
    assert found["obs.k"].available_us is None
    (note,) = report.of_kind("availability")
    assert "never its inputs" in note.message


def test_S5_1_comparators_are_kept():
    records, _ = _read()
    found = _events(records)
    k = found["obs.k"]
    assert (k.value_comparator, k.value_num, k.unit) == ("lt", 5.0, "mmol/L")
    creat = found["obs.creat.v2"]
    assert (creat.value_comparator, creat.value_num, creat.unit) == (None, 1.2, "mg/dL")
    assert found["obs.bp.c0"].unit == "mm[Hg]"  # the UCUM code, not the display
    assert found["obs.bp.c1"].unit == "mmHg"  # no UCUM code: the unit string
    ad = [
        {
            **Fhir.resources()[3],
            "valueQuantity": {"value": 5, "comparator": "ad", "unit": "mmol/L"},
        },
        Fhir.resources()[1],
    ]
    records, report = _read(ad)
    assert "obs.k" not in _events(records)
    assert any("'ad'" in n.message for n in report.warnings)


def test_S5_1_data_absent_reason_is_missing_reason():
    a1c = _events(_read()[0])["obs.a1c"]
    assert a1c.missing_reason == "not-performed"
    assert a1c.value_num is None and a1c.value_text is None


def test_S5_2_versions_supersede():
    records, report = _read()
    found = _events(records)
    assert {found[e].record_id for e in ("obs.creat.v1", "obs.creat.v2")} == {
        "obs.creat"
    }
    assert found["obs.creat.v1"].status == "preliminary"
    assert found["obs.creat.v2"].status == "amended"
    assert any(n.severity == "decision" for n in report.of_kind("status"))
    supersedes = [
        (link.source_id, link.target_id)
        for link in records.links
        if link.relation == "supersedes"
    ]
    assert supersedes == [("obs.creat.v2", "obs.creat.v1")]
    # Unorderable versions are refused, not ordered by guess.
    first, second = Fhir.resources()[1:3]
    first = {**first, "meta": {"versionId": "a"}}
    second = {**second, "meta": {"versionId": "b"}}
    with pytest.raises(MEDH5ValidationError, match="have no order"):
        _read([first, second])
    stamped = [
        {**first, "meta": {"versionId": "a", "lastUpdated": "2020-03-02T00:00:00Z"}},
        {**second, "meta": {"versionId": "b", "lastUpdated": "2020-03-01T00:00:00Z"}},
    ]
    records, _ = _read(stamped)
    assert [(link.source_id, link.target_id) for link in records.links] == [
        ("obs.creat.va", "obs.creat.vb")
    ]


def test_S10_components_are_events_of_their_own():
    records, report = _read()
    found = _events(records)
    panel, systolic = found["obs.bp"], found["obs.bp.c0"]
    assert (systolic.code, systolic.value_num) == ("8480-6", 120.0)
    assert systolic.effective_start_us == panel.effective_start_us
    assert systolic.available_us == panel.available_us
    assert systolic.record_id == "obs.bp.c0"
    (dropped,) = report.of_kind("dropped")
    assert dropped.detail == {"interpretation": 1, "referenceRange": 1}


def test_S10_registered_is_skipped():
    records, report = _read()
    found = _events(records)
    assert "obs.ordered" not in found
    assert any("registered" in n.message for n in report.of_kind("status"))
    # No code is guessed from text, and other subjects stay out.
    assert "obs.free" not in found and "obs.other" not in found
    assert any("no coding" in n.message for n in report.warnings)
    assert report.of_kind("subject")[0].severity == "decision"
    assert found["obs.smoke"].value_text == "8517006"
    with pytest.raises(MEDH5ValidationError, match="no Observation"):
        _read(subject="Patient/p9")


def test_S10_images_link_only_through_the_map(tmp_path):
    records, report = _read()
    found = _events(records)
    assert found["dr.ct-report"].kind == "document"
    assert not records.documents and not records.links[1:]
    notes = {n.message for n in report.of_kind("link")} | {
        n.message for n in report.of_kind("text")
    }
    assert any("image_studies" in m for m in notes)
    assert any("texts=True" in m for m in notes)
    records, _ = _read(texts=True, image_studies={Fhir.STUDY: "CT_tp0"})
    (document,) = records.documents
    assert document.text == Fhir.TEXT
    described = {
        (link.target_type, link.target_id)
        for link in records.links
        if link.relation == "describes"
    }
    assert described == {("document", "dr.ct-report"), ("image", "CT_tp0")}
    sample = Fhir.sample(tmp_path / "s.medh5")
    bundle = Fhir.write(tmp_path / "bundle.json")
    result, report = augment_from_fhir(
        sample,
        bundle,
        subject=Fhir.SUBJECT,
        clock=Clock.utc("ehr"),
        texts=True,
        image_studies={Fhir.STUDY: "CT_tp0"},
    )
    assert result["version_after"] == "1.1" and report.outputs == [str(sample)]
    assert validate_file(sample).ok
    with open_sample(sample) as s:
        assert s.clinical.text("dr.ct-report") == Fhir.TEXT
    with pytest.raises(MEDH5Error):
        augment_from_fhir(sample, bundle, subject=Fhir.SUBJECT, clock=Clock.utc("ehr"))


def test_S10_a_report_holds_no_guess(tmp_path: Path):
    _, report = read_fhir(
        Fhir.write(tmp_path / "b.json"), subject=Fhir.SUBJECT, clock=Clock.utc("c")
    )
    assert report.converter == "from-fhir" and not report.guesses


def test_clinical_augment_from_fhir(tmp_path, capsys):
    """`medh5 clinical augment --from-fhir`, with `--json`; and the MEDS
    options it does not take."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_OK, main

    sample = Fhir.sample(tmp_path / "s.medh5")
    bundle = Fhir.write(tmp_path / "bundle.json")
    argv = [
        "clinical", "augment", str(sample), "--from-fhir", str(bundle),
        "--fhir-subject", Fhir.SUBJECT, "--clock-id", "ehr", "--clock", "utc",
    ]  # fmt: skip
    assert main([*argv, "--both-values", "text"]) == EXIT_ERROR
    assert "--both-values: MEDS options" in capsys.readouterr().err
    # Warnings (a Timing, a code in text only) exit 1, the file written.
    assert main([*argv, "--with-text", "--json"]) == EXIT_ERROR
    printed = json.loads(capsys.readouterr().out)
    assert printed["augment"]["version_after"] == "1.1"
    assert printed["augment"]["documents"] == 1
    assert printed["report"]["converter"] == "from-fhir"
    clean = [r for i, r in enumerate(Fhir.resources()) if i in (1, 2, 3)]
    other = Fhir.sample(tmp_path / "t.medh5")
    relative = [
        "clinical", "augment", str(other),
        "--from-fhir", str(Fhir.write(tmp_path / "clean.json", clean)),
        "--fhir-subject", Fhir.SUBJECT, "--clock-id", "ehr", "--clock", "relative",
        "--origin", "2020-03-01T00:00:00-05:00",
    ]  # fmt: skip
    assert main(relative) == EXIT_OK
    assert "0 warning(s)" in capsys.readouterr().out
    with open_sample(other) as s:
        assert s.clinical.clock.reference == "relative"
