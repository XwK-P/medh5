"""What every converter shares: the conversion report, study grouping, minted
keys, the diagnostic-code policy and lazy imports."""

from __future__ import annotations

from pathlib import Path

import pytest

from medh5.io.grouping import Occasion, SubjectGroup, group_by_subject
from medh5.io.report import ConversionReport, Note, merge_reports


class TestReport:
    def test_a_guess_is_not_a_failure_and_a_warning_is(self):
        report = ConversionReport(converter="test")
        report.decision("encoding", "chose layers", {"kind": "layers"})
        report.guess("order", "ordered by mtime")
        assert report.ok
        assert len(report.guesses) == 1
        report.warn("geometry", "spacing disagreed")
        assert not report.ok
        assert "1 warning" in report.format()
        assert "GUESS" in report.format(verbose=False)

    def test_detail_may_carry_a_key_called_kind(self):
        """The report's own parameter names must not shadow a converter's data."""
        report = ConversionReport()
        note = report.decision("encoding", "m", {"kind": "layers"})
        assert note.detail == {"kind": "layers"}
        assert note.kind == "encoding"

    def test_json_and_merge(self):
        first = ConversionReport(converter="a", outputs=["x"])
        first.warn("k", "m")
        second = ConversionReport(converter="b", outputs=["y"])
        merged = merge_reports([first, second], converter="batch")
        assert merged.outputs == ["x", "y"]
        assert not merged.ok
        payload = merged.to_json()
        assert payload["counts"]["warning"] == 1
        assert Note("k", "m").to_json()["severity"] == "info"
        assert first.of_kind("k")


class TestGrouping:
    def test_S3_7_studies_of_one_subject_become_one_sample(self):
        groups = group_by_subject(
            [
                Occasion("s2", "p1", "20260401"),
                Occasion("s1", "p1", "20260101"),
                Occasion("s3", "p2", "20260201"),
            ]
        )
        assert [g.subject_id for g in groups] == ["p1", "p2"]
        assert [o.key for o in groups[0].occasions] == ["s1", "s2"]
        assert groups[0].days_from_baseline() == [0, 90]
        assert groups[0].ordered_by == "date"
        assert groups[0].is_longitudinal

    def test_identity_is_never_inferred(self):
        report = ConversionReport()
        groups = group_by_subject([Occasion("s1"), Occasion("s2")], report=report)
        assert len(groups) == 2
        assert all(g.subject_id.startswith("study:") for g in groups)
        assert report.warnings

    def test_mtime_ordering_is_reported_as_a_guess(self):
        report = ConversionReport()
        groups = group_by_subject(
            [
                Occasion("b", "p1", order_hint=2.0),
                Occasion("a", "p1", order_hint=1.0),
            ],
            report=report,
        )
        assert [o.key for o in groups[0].occasions] == ["a", "b"]
        assert groups[0].ordered_by == "order_hint"
        assert report.guesses

    def test_without_dates_or_hints_the_order_is_kept_and_flagged(self):
        report = ConversionReport()
        groups = group_by_subject(
            [Occasion("b", "p1"), Occasion("a", "p1")], report=report
        )
        assert [o.key for o in groups[0].occasions] == ["b", "a"]
        assert report.guesses

    def test_study_mode_keeps_every_occasion_apart(self):
        groups = group_by_subject(
            [Occasion("s1", "p1"), Occasion("s2", "p1")], mode="study"
        )
        assert len(groups) == 2
        assert not groups[0].is_longitudinal

    def test_missing_dates_give_no_intervals(self):
        group = SubjectGroup("p", [Occasion("a"), Occasion("b")])
        assert group.days_from_baseline() == [None, None]
        assert group.timepoint_ids() == ["tp0", "tp1"]
        assert group.to_json()["subject_id"] == "p"

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="grouping mode"):
            group_by_subject([], mode="vibes")


class TestConverterDiagnosticCodes:
    """A converter refusal about its *input* carries no diagnostic code.

    §15.2's table describes conditions found in a MEDH5 file. A NIfTI volume or
    a DICOM series is not one yet, so a code applied to it tells anything
    branching on `exc.code` an untrue story --- an irregular DICOM stack read as
    a non-positive grid spacing, a tilted 2-D plane as a non-orthonormal
    `direction`, a modality-LUT disagreement as malformed `channel_names`.

    A refusal about the sample being written or targeted is different and keeps
    its code: a SEG naming a grid the sample does not have really is `E101`, and
    a class absent from the sample's label set really is `E402`.

    This mistake reached six separate sites before it was found, one at a time,
    so the allow-list below is exhaustive: a new coded refusal in `medh5.io` has
    to be added here deliberately, with the reason it is about the sample rather
    than the input.
    """

    ALLOWED = {
        ("dicom_seg.py", "E101"),  # SEG names no grid the sample has
        ("dicom_seg.py", "E402"),  # segment absent from the sample's label set
        ("dicom_seg.py", "E405"),  # SEG shape vs. the target grid's
        ("nifti.py", "E402"),  # mask name absent from the sample's label set
        ("nifti.py", "E405"),  # mask shape vs. the target grid's
        ("nnunetv2.py", "E402"),  # class absent from the annotation
        ("rtstruct.py", "E101"),  # RTSTRUCT names no grid the sample has
        ("rtstruct.py", "E402"),  # ROI absent from the sample's label set
        ("rtstruct.py", "E401"),  # the sample's annotation is the wrong kind
        ("rtstruct.py", "E414"),  # the sample's annotation has no usable space
    }

    def test_no_converter_refusal_borrows_a_format_code(self):
        import re

        import medh5.io

        root = Path(medh5.io.__file__).parent
        found = {
            (path.name, code)
            for path in sorted(root.glob("*.py"))
            for code in re.findall(r'code="(E\d{3})"', path.read_text(encoding="utf-8"))
        }
        assert found <= self.ALLOWED, (
            "new coded refusal(s) in medh5.io: "
            f"{sorted(found - self.ALLOWED)}. If the refusal describes the "
            "MEDH5 sample, add it to ALLOWED with a reason; if it describes the "
            "converter's input, leave it uncoded."
        )


class TestLazyImports:
    def test_converters_resolve_without_importing_medh5_io_eagerly(self):
        import medh5.io as io

        assert callable(io.from_nifti)
        assert callable(io.migrate)
        assert "from_dicom" in dir(io)
        with pytest.raises(AttributeError, match="from_parquet"):
            _ = io.from_parquet

    def test_importing_medh5_does_not_import_the_optional_stacks(self, tmp_path):
        """`import medh5` must not need nibabel, pydicom or highdicom."""
        import subprocess
        import sys

        script = (
            "import sys; import medh5; "
            "assert 'nibabel' not in sys.modules; "
            "assert 'pydicom' not in sys.modules; "
            "assert 'highdicom' not in sys.modules; print('clean')"
        )
        # Away from the checkout, whose `medh5/` would shadow the installed one.
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        assert "clean" in result.stdout


class TestKeys:
    def test_S5_2_keys_are_schema_valid_wherever_they_are_minted(self):
        from medh5.io._common import sanitize_key, sanitize_stem

        assert sanitize_key("GTV-1") == "gtv_1"
        assert sanitize_key("Tumour Core") == "tumour_core"
        assert sanitize_key("Liver.L") == "liver_l"
        assert sanitize_key("_x") == "x"
        assert sanitize_key("", fallback="roi") == "roi"
        assert sanitize_key("é") == "class"
        assert sanitize_stem("a b/c", limit=4) == "a_b_"
