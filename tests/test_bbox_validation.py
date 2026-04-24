"""Tests for medh5.validate_bboxes."""

from __future__ import annotations

import numpy as np
import pytest

from medh5 import MEDH5ValidationError, validate_bboxes


class TestValidateBboxes:
    def test_in_bounds_returns_identical_boxes(self):
        boxes = np.array([[[0, 4], [0, 4], [0, 4]]], dtype=np.int64)
        clamped, issues = validate_bboxes(boxes, (8, 8, 8))
        assert issues == []
        np.testing.assert_array_equal(clamped, boxes)

    def test_negative_min_clamped(self):
        boxes = np.array([[[-2, 4], [0, 4], [0, 4]]], dtype=np.int64)
        clamped, issues = validate_bboxes(boxes, (8, 8, 8))
        assert (0, 0, "min<0") in issues
        assert clamped[0, 0, 0] == 0

    def test_max_beyond_shape_clamped(self):
        boxes = np.array([[[0, 20], [0, 4], [0, 4]]], dtype=np.int64)
        clamped, issues = validate_bboxes(boxes, (8, 8, 8))
        assert (0, 0, "max>shape") in issues
        assert clamped[0, 0, 1] == 8

    def test_inverted_box_swapped(self):
        boxes = np.array([[[5, 2], [0, 4], [0, 4]]], dtype=np.int64)
        clamped, issues = validate_bboxes(boxes, (8, 8, 8))
        assert (0, 0, "min>max") in issues
        assert clamped[0, 0, 0] == 2
        assert clamped[0, 0, 1] == 5

    def test_multiple_issues_on_one_box(self):
        boxes = np.array([[[-3, 30], [0, 4], [0, 4]]], dtype=np.int64)
        _, issues = validate_bboxes(boxes, (8, 8, 8))
        reasons = {r for (_, _, r) in issues if issues}
        assert "min<0" in reasons
        assert "max>shape" in reasons

    def test_shape_mismatch_raises(self):
        with pytest.raises(MEDH5ValidationError, match="shape"):
            validate_bboxes(np.zeros((2, 2), dtype=np.int64), (8, 8, 8))

    def test_does_not_mutate_input(self):
        boxes = np.array([[[-2, 20], [0, 4], [0, 4]]], dtype=np.int64)
        original = boxes.copy()
        clamped, _ = validate_bboxes(boxes, (8, 8, 8))
        np.testing.assert_array_equal(boxes, original)
        assert clamped is not boxes
