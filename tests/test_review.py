"""Tests for the review/QA workflow helpers."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError, ReviewStatus


def _make_file(path):
    MEDH5File.write(
        path,
        images={"CT": np.zeros((2, 4, 4), dtype=np.float32)},
        extra={"patient_id": "P001"},
    )


class TestReviewStatus:
    def test_default_pending(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        st = MEDH5File.get_review_status(p)
        assert isinstance(st, ReviewStatus)
        assert st.status == "pending"
        assert st.annotator is None

    def test_set_and_get(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        MEDH5File.set_review_status(
            p, status="reviewed", annotator="puyang", notes="ok"
        )
        st = MEDH5File.get_review_status(p)
        assert st.status == "reviewed"
        assert st.annotator == "puyang"
        assert st.notes == "ok"
        assert st.timestamp is not None
        # User-supplied keys preserved
        meta = MEDH5File.read_meta(p)
        assert meta.extra["patient_id"] == "P001"

    def test_history_appended(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        MEDH5File.set_review_status(p, status="reviewed", annotator="a")
        MEDH5File.set_review_status(p, status="flagged", annotator="b", notes="recheck")
        st = MEDH5File.get_review_status(p)
        assert st.status == "flagged"
        assert st.history is not None
        assert len(st.history) == 1  # only the prior 'reviewed' state
        assert st.history[0]["status"] == "reviewed"
        assert st.history[0]["annotator"] == "a"

    def test_invalid_status(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        with pytest.raises(MEDH5ValidationError, match="Unknown review status"):
            MEDH5File.set_review_status(p, status="bogus")

    def test_explicit_timestamp(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        MEDH5File.set_review_status(
            p, status="reviewed", timestamp="2026-01-01T00:00:00+00:00"
        )
        st = MEDH5File.get_review_status(p)
        assert st.timestamp == "2026-01-01T00:00:00+00:00"

    def test_works_on_file_without_extra(self, tmp_path):
        p = tmp_path / "s.medh5"
        MEDH5File.write(p, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})
        MEDH5File.set_review_status(p, status="reviewed", annotator="x")
        assert MEDH5File.get_review_status(p).status == "reviewed"
