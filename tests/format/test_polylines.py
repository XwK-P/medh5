"""Format 1.2: the version a sample is written at, and the `polylines` kind
(1.2 §2, §3)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations import Polyline, PolylinesAnnotation
from medh5.errors import MEDH5ValidationError
from medh5.validate import Report, validate_file
from tests.helpers import LATER_MINOR, attested_as_written, encode_attr
from tests.kits import Framed, History

PLANAR = [[4.0, 2.0, 2.0], [4.0, 5.5, 3.0], [4.0, 9.0, 6.5]]
THROUGH = [[1.0, 8.0, 8.0], [3.5, 8.5, 8.0], [6.0, 9.0, 8.5], [7.0, 9.0, 9.0]]


def at(report: Report, location: str) -> list[str]:
    return [d.code for d in report.diagnostics if d.location == location]


def curves(path: Path, **options) -> Path:
    """On Framed's grid `g`: the centreline `vessels` of two curves, one on
    plane z = 4 and one through the volume, tracing objects 11 and 12."""
    with Framed.writer(path) as w:
        w.add_polylines(
            "vessels",
            [
                Polyline(np.array(PLANAR), "c1", plane=(0, 4)),
                Polyline(np.array(THROUGH), 2),
            ],
            grid="g",
            instance_ids=[11, 12],
            **options,
        )
    return path


def plain(path: Path) -> Path:
    with Framed.writer(path) as w:
        w.add_points("seeds", [[1.0, 2.0, 3.0]], grid="g")
    return path


class TestVersion:
    """1.2 §2.1: a sample is written at the lowest version its content needs."""

    def test_v1_2_S2_1_a_sample_without_polylines_is_written_at_1_0_or_1_1(
        self, tmp_path
    ):
        with medh5.open(plain(tmp_path / "plain.medh5")) as s:
            assert s.version == "1.0"
        History.write(tmp_path / "history.medh5")
        with medh5.open(tmp_path / "history.medh5") as s:
            assert s.version == "1.1"

    def test_v1_2_S2_1_a_polylines_annotation_raises_the_version_to_1_2(self, tmp_path):
        with medh5.open(curves(tmp_path / "c.medh5")) as s:
            assert s.version == "1.2"
            assert s.support == "full"
        assert validate_file(tmp_path / "c.medh5", level="integrity").ok
        # With the clinical profile too: 1.2, whichever needs more.
        path = tmp_path / "both.medh5"
        History.write(path)
        with medh5.amend(path) as w:
            w.add_polylines("track", [Polyline(np.eye(3), 3)], grid="ct_tp0")
        with medh5.open(path) as s:
            assert s.version == "1.2" and "clinical" in s.profiles
        assert validate_file(path, level="integrity").ok

    def test_v1_2_S2_1_removing_the_last_polylines_lowers_the_version(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        with medh5.amend(path) as w:
            w.add_polylines("second", [Polyline(np.eye(3), 1)], grid="g")
        with medh5.amend(path) as w:
            w.remove_annotation("vessels")
        with medh5.open(path) as s:
            assert s.version == "1.2", "one is left"
            before = s.content_id
        with medh5.amend(path) as w:
            pass
        with medh5.open(path) as s:
            assert (s.version, s.content_id) == ("1.2", before), "a no-op keeps it"
        with medh5.amend(path) as w:
            w.remove_annotation("second")
        with medh5.open(path) as s:
            assert s.version == "1.0"
            assert s.content_id != before
        assert validate_file(path, level="integrity").ok

    def test_v1_2_S2_1_a_1_2_file_without_polylines_keeps_its_version(self, tmp_path):
        """A no-op amendment never lowers the version (1.1 §2.3): only removing
        the last polylines annotation does."""
        path = plain(tmp_path / "p.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("1.2")
        attested_as_written(path)
        with medh5.amend(path):
            pass
        with medh5.open(path) as s:
            assert s.version == "1.2"

    def test_v1_2_S2_1_written_version_reads_kinds(self):
        from medh5._core import written_version

        assert written_version(None, ["core"], kinds=["contours"]) == "1.0"
        assert written_version(None, ["core"], kinds=["polylines"]) == "1.2"
        assert written_version(None, ["core", "clinical"], kinds=["mesh"]) == "1.1"
        assert written_version("1.1", ["core"], kinds=["polylines"]) == "1.2"
        assert written_version("1.2", ["core"]) == "1.2"

    def test_v1_2_S2_2_the_later_minor_is_1_3(self):
        assert medh5.FORMAT_VERSION == "1.2"
        assert LATER_MINOR == "1.3"

    def test_v1_2_S2_1_polylines_in_a_1_1_file_is_E401(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("1.1")
        report = validate_file(path)
        assert at(report, "/annotations/vessels") == ["E401"]
        assert "defined by MEDH5 1.2" in str(report.diagnostics)
        # A later minor reads it as the kind this engine knows.
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr(LATER_MINOR)
        assert at(validate_file(path), "/annotations/vessels") == []


class TestPolylines:
    """1.2 §3: open curves."""

    def test_v1_2_S3_polylines_round_trip(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        with medh5.amend(path) as w:
            w.add_polylines(
                "catheter",
                [Polyline(np.array([[0.0, 1.0, 1.0], [2.0, 3.0, 4.0]]), 3)],
                grid="g",
                space="world",
            )
        with medh5.open(path) as s:
            vessels = s.annotations["vessels"]
            assert isinstance(vessels, PolylinesAnnotation)
            assert (vessels.kind, vessels.task, vessels.space) == (
                "polylines",
                "other",
                "index",
            )
            assert len(vessels) == 2
            assert vessels.offsets.tolist() == [0, 3, 7]
            assert vessels.planes.tolist() == [[0, 4], [-1, 0]]
            assert vessels.instance_ids.tolist() == [11, 12]
            assert vessels.object_class_ids.tolist() == [1, 2]
            assert list(vessels.class_ids) == [1, 2]
            np.testing.assert_array_equal(vessels.polyline(1), np.float32(THROUGH))
            lines = list(vessels.polylines())
            assert [line.plane for line in lines] == [(0, 4), (-1, 0)]
            assert [line.class_id for line in lines] == [1, 2]
            np.testing.assert_array_equal(lines[0].vertices, PLANAR)
            # Index to world on the grid's spacing (2, 1, 1); its own index
            # is what it stores.
            world = vessels.to_world(vessels.polyline(0))
            np.testing.assert_allclose(world[:, 0], np.float32(PLANAR)[:, 0] * 2.0)
            np.testing.assert_allclose(vessels.to_index(vessels.polyline(0)), PLANAR)
            catheter = s.annotations["catheter"]
            assert catheter.space == "world" and catheter.instance_ids is None
            assert catheter.planes.tolist() == [[-1, 0]]
            np.testing.assert_allclose(
                catheter.to_index(catheter.polyline(0))[1], [1.0, 3.0, 4.0]
            )
            summary = {a["id"]: a for a in s.summary()["annotations"]}
            assert summary["vessels"]["objects"] == 2
        assert validate_file(path, level="integrity").ok

    def test_v1_2_S3_an_empty_polylines_annotation_is_a_negative(self, tmp_path):
        with Framed.writer(tmp_path / "e.medh5") as w:
            w.add_polylines("none", [], grid="g", annotated_classes=[1])
        with medh5.open(tmp_path / "e.medh5") as s:
            empty = s.annotations["none"]
            assert len(empty) == 0 and empty.vertices.shape == (0, 3)
            assert list(empty.annotated_class_ids) == [1]
        assert validate_file(tmp_path / "e.medh5").ok

    def test_v1_2_S3_a_polyline_of_one_vertex_is_E404(self, tmp_path):
        with Framed.writer(tmp_path / "w.medh5") as w:
            with pytest.raises(MEDH5ValidationError, match="at least two") as e:
                w.add_polylines(
                    "dot", [Polyline(np.array([[1.0, 1.0, 1.0]]), 1)], grid="g"
                )
            assert e.value.code == "E404"
            w.abort()
        path = curves(tmp_path / "c.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/vessels/polyline_offsets"][...] = [0, 1, 7]
        assert at(validate_file(path), "/annotations/vessels") == ["E404"]

    def test_v1_2_S3_offsets_that_do_not_delimit_are_E408(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        located = "/annotations/vessels/polyline_offsets"
        for offsets in ([0, 3, 6], [1, 3, 7], [0, 5, 3]):
            with h5py.File(path, "r+") as handle:
                handle["annotations/vessels/polyline_offsets"][...] = offsets
            report = validate_file(path)
            assert set(at(report, located)) == {"E408"}, offsets
            assert at(report, "/annotations/vessels") == [], "E404 waits on E408"

    def test_v1_2_S3_polylines_satisfy_no_profile(self, tmp_path):
        path = curves(tmp_path / "c.medh5", task="detection")
        with medh5.open(path) as s:
            assert s.annotations["vessels"].task == "detection"
            assert s.profiles == {"core"}
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_profiles"] = encode_attr(["core", "det"])
        assert "E009" in validate_file(path).codes

    def test_v1_2_S3_polylines_check_their_columns(self, tmp_path):
        with Framed.writer(tmp_path / "w.medh5") as w:
            line = np.array(PLANAR)
            with pytest.raises(MEDH5ValidationError) as e:
                w.add_polylines("x", [Polyline(line, 1)], grid="g", instance_ids=[1, 2])
            assert e.value.code == "E405"
            with pytest.raises(MEDH5ValidationError) as e:
                w.add_polylines("x", [Polyline(line[:, :2], 1)], grid="g")
            assert e.value.code == "E405", "index space has the grid's rank"
            w.abort()
        with (
            pytest.raises(MEDH5ValidationError) as e,
            Framed.writer(tmp_path / "w.medh5") as w,
        ):
            w.add_polylines("x", [Polyline(line, 9)], grid="g")
        assert e.value.code == "E402"


class TestIdentity:
    """1.2 §3: a polyline's `instance_ids` are those of the other kinds, so a
    curve joins its object's track and its class is held to W909 (1.0 §7.4)."""

    def test_v1_2_S3_a_track_joins_polylines_as_presence(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        with medh5.open(path) as s:
            tracking = s.tracks()
            assert tracking.skipped == {}
            assert set(tracking) == {11, 12}
            [seen] = tracking[11].observations
        assert seen.kind == "polylines"
        assert seen.class_id == 1 and seen.volume is None
        # The box enclosing the curve's vertices, as a point's is [p, p].
        assert np.array_equal(seen.box, [[4.0, 4.0], [2.0, 9.0], [2.0, 6.5]])

    def test_v1_2_S3_W909_reads_a_polyline_s_class(self, tmp_path):
        path = curves(tmp_path / "c.medh5")
        with medh5.amend(path) as w:
            w.add_points(
                "tip", [[4.0, 2.0, 2.0]], grid="g", class_ids=["c3"], instance_ids=[11]
            )
        report = validate_file(path)
        [conflict] = [d for d in report.diagnostics if d.code == "W909"]
        assert "instance id 11" in conflict.message
        assert "'vessels'" in conflict.message and "'tip'" in conflict.message
