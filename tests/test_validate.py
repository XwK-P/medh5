"""Tests for MEDH5File.validate and ValidationIssue.location."""

from __future__ import annotations

import h5py
import numpy as np

from medh5 import MEDH5File


def _make_file(path):
    MEDH5File.write(
        path,
        images={"CT": np.zeros((4, 4, 4), dtype=np.float32)},
        seg={"tumor": np.zeros((4, 4, 4), dtype=bool)},
    )


class TestValidationLocation:
    def test_clean_file_has_no_errors(self, tmp_path):
        p = tmp_path / "ok.medh5"
        _make_file(p)
        report = MEDH5File.validate(p)
        assert report.is_valid
        # Clean files still warn about missing checksum (opt-in).
        missing = [w for w in report.warnings if w.code == "missing_checksum"]
        assert missing and missing[0].location == "checksum_sha256"

    def test_seg_shape_mismatch_location(self, tmp_path):
        p = tmp_path / "bad_seg.medh5"
        _make_file(p)
        with h5py.File(str(p), "a") as f:
            # Replace seg/tumor with a mismatched-shape dataset.
            del f["seg/tumor"]
            f["seg"].create_dataset("tumor", data=np.zeros((2, 2, 2), dtype=bool))
        report = MEDH5File.validate(p)
        issue = next(e for e in report.errors if e.code == "seg_shape_mismatch")
        assert issue.location == "seg/tumor"

    def test_to_dict_omits_location_when_absent(self):
        from medh5.core import ValidationIssue

        issue = ValidationIssue(code="x", message="y")
        assert issue.to_dict() == {"code": "x", "message": "y"}
        with_loc = ValidationIssue(code="x", message="y", location="images/CT")
        assert with_loc.to_dict() == {
            "code": "x",
            "message": "y",
            "location": "images/CT",
        }

    def test_missing_images_group_location(self, tmp_path):
        p = tmp_path / "empty.medh5"
        with h5py.File(str(p), "w") as f:
            f.attrs["schema_version"] = "1"
        report = MEDH5File.validate(p)
        issue = next(e for e in report.errors if e.code == "missing_images_group")
        assert issue.location == "images"
