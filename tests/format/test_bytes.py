"""The bytes two implementations must agree on (spec §2.4, §2.5, §5.1, §13).

Attribute encodings, canonical JSON and the input of every digest are defined to
the byte, so a file and its ``content_id`` do not depend on which implementation
wrote them.  These tests hold the specification's text to what the engine
writes; Appendix C.1 records the clauses 2.0 had to make explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.integrity import array_digest, canonical_attrs, dataset_digest
from medh5.labels import canonical_json
from tests.helpers import ROOT

# Floats either side of every notation boundary §5.1 states, and the values a
# shortest-round-trip printer most often gets wrong.
FLOATS = (
    0.0,
    -0.0,
    1.0,
    2.5,
    -7.25,
    0.1,
    1 / 3,
    1e-4,
    9.999e-5,
    1e-5,
    1.5e-5,
    123456789.125,
    1e15,
    9999999999999998.0,
    1e16,
    1.5e16,
    1e22,
    1.5e300,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
)
STRINGS = (
    "plain",
    "ünïcödé",
    "日本語",
    'quote"back\\slash',
    "\n\r\t\b\f",
    "\x00\x01\x1f",
    "\x7f",
    " ",
)
NON_FINITE = (math.nan, math.inf, -math.inf)


def _python(doc: object) -> bytes:
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _awkward_floats(path: Path) -> Path:
    """A sample whose grid attributes sit either side of float notation boundaries."""
    with medh5.create(path, sample_id="c", subject_id="s") as w:
        w.add_grid(
            "g", shape=(2, 2, 2), spacing=(0.1, 1e-5, 1e16), origin=(-0.0, 2.5, 1 / 3)
        )
        w.add_image("CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT")
    return path


def _image_with_attrs(path: Path, **attrs: object) -> Path:
    """A sample whose image carries *attrs*, set through the writer's views."""
    with medh5.create(path, sample_id="attrs") as w:
        w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
        image = w.add_image(
            "CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT"
        )
        for name, value in attrs.items():
            image.attrs[name] = value
    return path


@pytest.mark.parametrize("value", FLOATS, ids=repr)
def test_S5_1_canonical_json_writes_a_float_as_shortest_round_trip(
    value: float,
) -> None:
    assert canonical_json({"x": value}) == _python({"x": value})


@pytest.mark.parametrize("value", STRINGS, ids=repr)
def test_S5_1_canonical_json_keeps_text_as_utf8_and_escapes_only_controls(
    value: str,
) -> None:
    assert canonical_json({"x": value}) == _python({"x": value})


def test_S5_1_canonical_json_sorts_keys_and_keeps_number_types() -> None:
    doc = {"b": [1, 1.0, True, None], "a": {"z": 0, "é": -0.0}, "A": "x"}
    assert canonical_json(doc) == _python(doc)
    assert canonical_json(doc).startswith(
        b'{"A":"x","a":{"z":0,"\xc3\xa9":-0.0},"b":[1,1.0,true,null]'
    )


def test_S13_2_canonical_attrs_is_the_canonical_json_of_the_attributes(
    tmp_path: Path,
) -> None:
    path = _awkward_floats(tmp_path / "a.medh5")
    names = (
        "axis_kinds",
        "axis_names",
        "direction",
        "origin",
        "shape",
        "spacing",
        "units",
    )
    with h5py.File(path, "r") as handle:
        stored = handle["grids/g"].attrs

        def jsonable(value: object) -> object:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, np.ndarray):
                return (
                    [jsonable(v) for v in value.tolist()]
                    if value.dtype.kind in "OSU"
                    else value.tolist()
                )
            if isinstance(value, np.generic):
                return value.item()
            return value

        expected = _python(
            {n: jsonable(stored[n]) for n in sorted(names) if n in stored}
        ).decode()
    with medh5.open(path) as sample:
        assert canonical_attrs(sample.root["grids/g"], list(names)) == expected


def _covered_attributes() -> dict[str, set[str]]:
    """§13.2's table of covered attributes, by object: ``""`` for the root,
    else the group whose members the row is for."""
    spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
    section = spec[spec.index("### 13.2") : spec.index("### 13.3")]
    table = section[section.index("| Object | Covered attributes |") :]
    covered: dict[str, set[str]] = {}
    for row in table.splitlines()[2:]:
        if not row.startswith("|"):
            break
        target, names = row.strip("|").split("|")
        group = re.search(r"`(\w+)/<id>`", target)
        covered[group.group(1) if group else ""] = set(re.findall(r"`(\w+)`", names))
    return covered


def _content_id_from_the_text(path: Path, covered: dict[str, set[str]]) -> str:
    """§13.2 followed literally, with h5py, json and hashlib: nothing of the
    engine's between the text and the bytes."""

    def hexdigest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def plain(value: object) -> object:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            return [plain(v) for v in value] if value.ndim else plain(value[()])
        if isinstance(value, np.generic):
            return value.item()
        return value

    def attribute_line(path: str, obj: h5py.HLObject, names: set[str]) -> str:
        doc = {n: plain(obj.attrs[n]) for n in sorted(names) if n in obj.attrs}
        return f"@{path}\tsha256:{hexdigest(_python(doc))}\n"

    datasets: list[str] = []
    attributes: list[str] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: h5py.HLObject) -> None:
            if (
                isinstance(obj, h5py.Dataset)
                and not name.startswith("index/")
                and "digest" in obj.attrs
            ):
                datasets.append(f"{name}\t{plain(obj.attrs['digest'])}\n")

        handle.visititems(visit)
        meta = plain(handle["meta"][()])
        assert isinstance(meta, str)
        attributes.append(attribute_line("", handle, covered[""]))
        for group in ("grids", "images", "annotations", "transforms"):
            for member in handle.get(group, {}):
                obj = handle[f"{group}/{member}"]
                attributes.append(
                    attribute_line(f"{group}/{member}", obj, covered[group])
                )
    lines = (
        sorted(datasets)
        + [f"meta\t{hexdigest(meta.encode('utf-8'))}\n"]
        + sorted(attributes)
    )
    return "sha256:" + hexdigest("".join(lines).encode("utf-8"))


def test_S13_2_the_covered_attributes_reproduce_content_id(tmp_path: Path) -> None:
    """An implementation that reads §13.2's table computes the engine's address:
    for every kind of object, and for a landmark pair, whose `correspondence`
    the table leaves out."""
    path = tmp_path / "pair.medh5"
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with medh5.create(path, sample_id="pair", subject_id="s") as w:
        w.add_timepoint("tp0", days_from_baseline=0)
        w.add_timepoint("tp1", days_from_baseline=30)
        for tp in ("tp0", "tp1"):
            w.add_grid(
                f"ct_{tp}",
                shape=(4, 4, 4),
                spacing=(1.0, 1.0, 1.0),
                timepoint=tp,
                frame_uid=f"pseudo:{tp}",
            )
            w.add_image(
                f"CT_{tp}",
                np.zeros((4, 4, 4), np.int16),
                grid=f"ct_{tp}",
                modality="CT",
            )
        w.add_points("fid_tp0", points, grid="ct_tp0", correspondence="fid_tp1")
        w.add_points("fid_tp1", points, grid="ct_tp1", correspondence="fid_tp0")
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            from_frame="pseudo:tp0",
            to_frame="pseudo:tp1",
            matrix=np.eye(4),
        )

    covered = _covered_attributes()
    with medh5.open(path) as sample:
        stored = sample.content_id
        engine = sample.attr_name_map()
    assert {obj.split("/")[0] for obj in engine} == set(covered)
    for obj, names in engine.items():
        assert set(names) == covered[obj.split("/")[0]], obj
    assert "correspondence" not in covered["annotations"]
    with h5py.File(path, "r") as handle:
        assert handle["annotations/fid_tp0"].attrs["correspondence"] == "fid_tp1"
    assert _content_id_from_the_text(path, covered) == stored
    covering = {**covered, "annotations": covered["annotations"] | {"correspondence"}}
    assert _content_id_from_the_text(path, covering) != stored


@pytest.mark.parametrize(
    ("dtype", "dtype_str"),
    [
        (np.bool_, "|b1"),
        (np.int8, "|i1"),
        (np.uint8, "|u1"),
        (np.int16, "<i2"),
        (np.int32, "<i4"),
        (np.int64, "<i8"),
        (np.uint16, "<u2"),
        (np.uint32, "<u4"),
        (np.uint64, "<u8"),
        (np.float16, "<f2"),
        (np.float32, "<f4"),
        (np.float64, "<f8"),
    ],
)
def test_S13_1_dtype_str_for_every_stored_type(dtype: type, dtype_str: str) -> None:
    array = np.arange(6).reshape(2, 3).astype(dtype)
    stream = (
        b"x/y\x00"
        + dtype_str.encode()
        + b"\x002,3\x00"
        + array.astype(array.dtype.newbyteorder("<")).tobytes()
    )
    assert array_digest("x/y", array) == "sha256:" + hashlib.sha256(stream).hexdigest()


def test_S13_1_a_string_dataset_digests_as_O_with_nul_separated_utf8(
    tmp_path: Path,
) -> None:
    path = tmp_path / "points.medh5"
    with medh5.create(path, sample_id="p", subject_id="s") as w:
        w.add_grid("g", shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
        w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
        w.add_points(
            "landmarks",
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            grid="g",
            names=["apex", "nœud"],
        )
    with medh5.open(path) as sample:
        names = sample.root["annotations/landmarks/names"]
        stream = (
            b"annotations/landmarks/names\x00|O\x002\x00"
            + b"apex"
            + b"\x00"
            + "nœud".encode()
            + b"\x00"
        )
        assert dataset_digest(names) == "sha256:" + hashlib.sha256(stream).hexdigest()
        assert names.attrs["digest"] == dataset_digest(names)


def test_S2_5_a_boolean_is_an_int8_enumeration_FALSE_TRUE(tmp_path: Path) -> None:
    path = tmp_path / "b.medh5"
    with medh5.create(path, sample_id="b", subject_id="s") as w:
        w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
        image = w.add_image(
            "CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT"
        )
        image.attrs["x_flag"] = True
    with h5py.File(path, "r") as handle:
        tid = handle["images/CT"].attrs.get_id("x_flag").get_type()
        assert isinstance(tid, h5py.h5t.TypeEnumID)
        assert (
            tid.get_super().get_size() == 1
            and tid.get_super().get_sign() == h5py.h5t.SGN_2
        )
        members = {
            tid.get_member_name(i).decode(): tid.get_member_value(i)
            for i in range(tid.get_nmembers())
        }
        assert members == {"FALSE": 0, "TRUE": 1}


@pytest.mark.parametrize("value", NON_FINITE, ids=repr)
def test_S2_4_a_non_finite_number_is_refused_not_nulled(
    value: float, tmp_path: Path
) -> None:
    """JSON has no NaN or infinity.  1.x wrote Python's tokens for them, making
    `/meta` something other than JSON; writing `null` instead would change the
    data without a word.  The caller decides what a missing value is."""
    with pytest.raises(MEDH5ValidationError, match="JSON has no NaN or infinity"):
        canonical_json({"x": value})
    w = medh5.create(tmp_path / "n.medh5", sample_id="n", subject_id="s")
    try:
        with pytest.raises(MEDH5ValidationError, match="use None for a missing value"):
            w.extra("x_lab", {"missing": value})
        with pytest.raises(MEDH5ValidationError):
            w.add_timepoint("tp0", subject_age_years=value)
    finally:
        w.abort()


def test_S13_2_an_attribute_keeps_its_non_finite_value_in_canonical_attrs(
    tmp_path: Path,
) -> None:
    """An attribute can hold NaN where a JSON document cannot.  1.x hashed it
    as `NaN`/`Infinity`, so a `content_id` stamped then must still verify."""
    path = tmp_path / "w.medh5"
    with medh5.create(path, sample_id="w", subject_id="s") as w:
        w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
        w.add_image(
            "CT",
            np.zeros((2, 2, 2), np.int16),
            grid="g",
            modality="CT",
            window_center=[math.nan],
            window_width=[math.inf],
        )
    with h5py.File(path, "r") as handle:
        stored = handle["images/CT"].attrs
        expected = json.dumps(
            {k: stored[k].tolist() for k in ("window_center", "window_width")},
            sort_keys=True,
            separators=(",", ":"),
        )
    assert expected == '{"window_center":[NaN],"window_width":[Infinity]}'
    with medh5.open(path) as sample:
        got = canonical_attrs(
            sample.images["CT"].dataset, ["window_center", "window_width"]
        )
        assert got == expected
        assert sample.verify().ok


def test_S2_4_a_1x_document_with_NaN_opens_and_validates_as_E004(
    tmp_path: Path,
) -> None:
    """1.x wrote `NaN` into `/meta` when handed one.  The file still opens ---
    its images and annotations are not hostage to a value in `extra` --- the
    tokens read as None, and the validator names the first one."""
    from medh5.validate import validate_file

    path = tmp_path / "old.medh5"
    with medh5.create(path, sample_id="old", subject_id="s") as w:
        w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
        w.add_image("CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT")
        w.extra("x_lab", {"missing": None})
    with h5py.File(path, "r+") as handle:
        text = handle["meta"][()].decode("utf-8")
        assert '"missing": null' in text
        del handle["meta"]
        handle["meta"] = text.replace('"missing": null', '"missing": NaN')
    with medh5.open(path) as sample:
        assert sample.document.extra["x_lab"] == {"missing": None}
        assert sample.images["CT"].read().shape == (2, 2, 2)
    report = validate_file(path)
    e004 = [d for d in report.diagnostics if d.code == "E004"]
    assert e004 and "NaN at line 1 column" in e004[0].message


class TestAttributeCodecs:
    """Spec §2.5 through the public ``Attrs`` view; the engine's codec has its
    own tests (``h5::attrs::tests``)."""

    def test_S2_5_types_round_trip(self, tmp_path):
        path = _image_with_attrs(
            tmp_path / "types.medh5",
            x_str="x",
            x_bytes=b"x",
            x_strs=["a", "b"],
            x_int=3,
            x_ints=[1, 2],
            x_float=1.5,
            x_floats=[1.5, 2.5],
            x_bool=True,
        )
        with medh5.open(path) as sample:
            attrs = sample.images["CT"].dataset.attrs
            assert attrs["x_str"] == "x"
            assert attrs["x_bytes"] == "x"
            assert tuple(attrs["x_strs"]) == ("a", "b")
            assert attrs["x_int"] == 3
            assert tuple(attrs["x_ints"]) == (1, 2)
            assert attrs["x_float"] == 1.5
            assert tuple(attrs["x_floats"]) == (1.5, 2.5)
            assert attrs["x_bool"] is True
        # Stored as the table in §2.5 says, as any HDF5 reader sees it.
        with h5py.File(path, "r") as handle:
            stored = handle["images/CT"].attrs
            text = h5py.check_string_dtype(stored.get_id("x_str").dtype)
            assert text.encoding == "utf-8" and text.length is None
            texts = h5py.check_string_dtype(stored.get_id("x_strs").dtype)
            assert texts.encoding == "utf-8" and texts.length is None
            assert stored["x_strs"].shape == (2,)
            assert stored["x_int"].dtype == np.int64
            assert stored["x_ints"].dtype == np.int64
            assert stored["x_float"].dtype == np.float64
            assert stored["x_floats"].dtype == np.float64
            assert stored["x_bool"].dtype == np.bool_

    def test_S2_5_matrices_stay_two_dimensional(self, tmp_path):
        path = _image_with_attrs(tmp_path / "matrix.medh5")
        with medh5.open(path) as sample:
            assert np.asarray(sample.grids["g"].direction).shape == (3, 3)
        with h5py.File(path, "r+") as handle:
            handle["grids/g"].attrs["direction"] = np.eye(3).reshape(9)
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.grids["g"]
            assert exc.value.code == "E109"
        from medh5.validate import validate_file

        assert "E109" in validate_file(path).codes

    def test_empty_sequences_and_mixed_types(self, tmp_path):
        path = _image_with_attrs(
            tmp_path / "mixed.medh5",
            x_empty=[],
            x_bools=[True, False],
            x_mixed=[1, 2.5],
        )
        with medh5.open(path) as sample:
            attrs = sample.images["CT"].dataset.attrs
            assert attrs["x_empty"].shape == (0,)
            assert attrs["x_empty"].dtype == np.int64
            assert attrs["x_bools"].dtype == np.bool_
            assert attrs["x_mixed"].dtype == np.float64
        with medh5.create(tmp_path / "object.medh5", sample_id="o") as w:
            w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            image = w.add_image(
                "CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT"
            )
            with pytest.raises(MEDH5ValidationError):
                image.attrs["x_object"] = object()

    def test_S2_3_identifier_rules(self, tmp_path):
        from medh5.collection import pack

        source = tmp_path / "ids.medh5"
        with medh5.create(source, sample_id="ids") as w:
            w.add_grid("CT_tp0", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            w.add_grid("x" * 128, shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            for bad in ("", "a b", "x" * 129, "meta"):
                with pytest.raises(MEDH5ValidationError) as exc:
                    w.add_grid(bad, shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
                assert exc.value.code == "E003"
            assert set(w.grids) == {"CT_tp0", "x" * 128}
            w.add_image(
                "CT", np.zeros((2, 2, 2), np.int16), grid="CT_tp0", modality="CT"
            )
        pack([source], tmp_path / "ok.medh5c", keys=["a.b-c_1"])
        with pytest.raises(MEDH5ValidationError) as exc:
            pack([source], tmp_path / "long.medh5c", keys=["x" * 256])
        assert exc.value.code == "E003"
