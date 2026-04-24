"""Tests for SampleMeta / SpatialMeta helpers."""

from __future__ import annotations

import json
import warnings

import h5py
import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5ValidationError, SpatialMeta


class TestAsAffine:
    def test_identity_direction_returns_none(self):
        s = SpatialMeta(spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        assert s.as_affine(3) is None

    def test_explicit_identity_direction_returns_none(self):
        s = SpatialMeta(
            spacing=[2.0, 2.0, 2.0],
            origin=[1.0, 2.0, 3.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        assert s.as_affine(3) is None

    def test_rotated_direction_builds_affine(self):
        # 90° rotation about z in 3D
        direction = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        s = SpatialMeta(
            spacing=[2.0, 3.0, 4.0],
            origin=[10.0, 20.0, 30.0],
            direction=direction,
        )
        m = s.as_affine(3)
        assert m is not None
        assert m.shape == (4, 4)
        # Linear block: direction @ diag(spacing)
        expected_linear = np.array(direction, dtype=float) @ np.diag([2.0, 3.0, 4.0])
        np.testing.assert_allclose(m[:3, :3], expected_linear)
        np.testing.assert_allclose(m[:3, 3], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(m[3], [0, 0, 0, 1])

    def test_missing_spacing_defaults_to_ones(self):
        direction = [[0, 1], [1, 0]]  # swap axes
        s = SpatialMeta(direction=direction)
        m = s.as_affine(2)
        assert m is not None
        np.testing.assert_allclose(m[:2, :2], direction)
        np.testing.assert_allclose(m[:2, 2], [0.0, 0.0])

    def test_direction_shape_mismatch_raises(self):
        s = SpatialMeta(direction=[[1, 0], [0, 1]])
        with pytest.raises(MEDH5ValidationError, match="direction must be a 3x3"):
            s.as_affine(3)

    def test_spacing_length_mismatch_raises(self):
        s = SpatialMeta(spacing=[1.0, 2.0], direction=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        with pytest.raises(MEDH5ValidationError, match="spacing length"):
            s.as_affine(3)

    def test_origin_length_mismatch_raises(self):
        s = SpatialMeta(origin=[0.0, 0.0], direction=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        with pytest.raises(MEDH5ValidationError, match="origin length"):
            s.as_affine(3)


def _write_raw_extra(path, extra_payload):
    """Overwrite extra attribute with a raw JSON blob, bypassing validation."""
    with h5py.File(str(path), "a") as f:
        f.attrs["extra"] = json.dumps(extra_payload)


def _make(path):
    MEDH5File.write(path, images={"CT": np.zeros((2, 2, 2), dtype=np.float32)})


class TestExtraWarnings:
    def test_review_not_dict_warns(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        _write_raw_extra(p, {"review": "wrong-type"})
        with pytest.warns(UserWarning, match="Malformed extra.review"):
            meta = MEDH5File.read_meta(p)
        assert meta.extra == {"review": "wrong-type"}

    def test_nnunetv2_labels_bad_shape_warns(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        _write_raw_extra(p, {"nnunetv2": {"labels": {"bad": "value"}}})
        with pytest.warns(UserWarning, match="extra.nnunetv2.labels"):
            MEDH5File.read_meta(p)

    def test_review_status_not_string_warns(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        _write_raw_extra(p, {"review": {"status": 42}})
        with pytest.warns(UserWarning, match="extra.review.status"):
            MEDH5File.read_meta(p)

    def test_newer_schema_version_warns(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        _write_raw_extra(p, {"review": {"status": "reviewed", "schema_version": 99}})
        with pytest.warns(UserWarning, match="schema_version=99"):
            MEDH5File.read_meta(p)

    def test_well_formed_extra_is_silent(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        _write_raw_extra(
            p,
            {
                "review": {"status": "pending", "schema_version": 1},
                "nnunetv2": {"labels": {"background": 0, "tumor": 1}},
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors
            MEDH5File.read_meta(p)


class TestSubsystemSchemaVersion:
    def test_review_write_stamps_schema_version(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make(p)
        MEDH5File.set_review_status(p, status="reviewed", annotator="qa")
        meta = MEDH5File.read_meta(p)
        assert meta.extra["review"]["schema_version"] == 1
