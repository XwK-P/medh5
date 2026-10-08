"""What 2.0.0 made explicit, held to by the engine that writes it.

2.0 re-implemented the format in Rust.  Three places in the specification
named a Python function where they meant bytes --- `np.bool_`, `repr`, "the
NumPy dtype string" --- and the engine had to read 1.x's code to agree with it
on a single digest.  Appendix C.1 records the corrections; these tests hold the
corrected text to what the engine writes, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.integrity.digest import array_digest, canonical_attrs, dataset_digest
from medh5.labels.labelset import canonical_json

ROOT = Path(__file__).resolve().parents[2]

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
NON_FINITE = (math.nan, math.inf, -math.inf)
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


def _python(doc: object) -> bytes:
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


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


def _attr_sample(path: Path) -> Path:
    with medh5.create(path, sample_id="c", subject_id="s") as w:
        w.add_grid(
            "g", shape=(2, 2, 2), spacing=(0.1, 1e-5, 1e16), origin=(-0.0, 2.5, 1 / 3)
        )
        w.add_image("CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT")
    return path


def test_S13_2_canonical_attrs_is_the_canonical_json_of_the_attributes(
    tmp_path: Path,
) -> None:
    path = _attr_sample(tmp_path / "a.medh5")
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


def test_S12_appendix_C_records_the_four_2_0_corrections() -> None:
    spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
    section = spec[spec.index("### C.1") : spec.index("### C.2")]
    for clause in ("| §2.5 |", "| §5.1, §13.2 |", "| §13.1 |", "| §14.3 |"):
        assert clause in section
    assert "four when the engine was written a second time, in Rust" in section


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


class TestThe1xNamesStillAnswer:
    """1.x read helpers that took an ``h5py`` group take the package's own view
    of one (``sample.root[...]``) and answer as they did."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        path = tmp_path / "a.medh5"
        with medh5.create(path, sample_id="a", subject_id="s") as w:
            w.add_grid("g", shape=(4, 4, 4), spacing=(1.0, 1.0, 2.0))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.add_segmentation("seg", grid="g", masks={1: np.ones((4, 4, 4), bool)})
            w.add_transform("t", kind="identity", from_frame="a", to_frame="b")
            w.build_index()
        return path

    def test_documents_grids_and_digests_read_from_a_view(self, path: Path) -> None:
        from medh5.document import SCHEMA_PATH, read_document, read_document_text
        from medh5.geometry import read_grid, read_grids
        from medh5.integrity import collect_digests

        assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["title"]
        with medh5.open(path) as sample:
            assert read_document(sample.root).identity.subject_id == "s"
            assert (
                json.loads(read_document_text(sample.root))["identity"]["sample_id"]
                == "a"
            )
            stored = read_grid(sample.root["grids/g"])
            resolved = sample.grids["g"]
            assert (stored.grid_id, stored.shape, stored.spacing) == (
                resolved.grid_id,
                resolved.shape,
                resolved.spacing,
            )
            # The stored group, not the reader's interpretation of it: the one
            # timepoint is resolved by `Sample`, never written back (1.2.1).
            assert stored.timepoint is None and resolved.timepoint == "tp0"
            assert list(read_grids(sample.root)) == ["g"]
            digests = collect_digests(sample.root)
            assert digests["images/CT"] == sample.root["images/CT"].attrs["digest"]
            assert not any(k.startswith("index/") for k in digests)

    def test_headers_read_from_a_group_and_the_index_names_its_own(
        self, path: Path
    ) -> None:
        from medh5.annotations.base import AnnotationHeader
        from medh5.transforms.base import TransformHeader

        with medh5.open(path) as sample:
            header = AnnotationHeader.read(sample.root["annotations/seg"])
            assert header == sample.annotations["seg"].header
            assert TransformHeader.read(sample.root["transforms/t"]).kind == "identity"
            assert sample.index["seg"].group.name == "/index/seg"

    def test_a_timeline_is_a_sequence_and_a_record_lists_its_fields(
        self, path: Path
    ) -> None:
        from collections.abc import Sequence

        from medh5.curation.identity import Identity

        with medh5.open(path) as sample:
            timeline = sample.timepoints
            assert isinstance(timeline, Sequence)
            assert timeline.index(timeline[0]) == 0 and timeline.count(timeline[0]) == 1
            with pytest.raises(ValueError):
                timeline.index("not a timepoint")
        assert {"sample_id", "subject_id", "extra"} <= set(dir(Identity("x", "y")))

    def test_a_codec_still_gives_h5py_its_keywords(self) -> None:
        from medh5.annotations.voxel.payload import AnnotationPayload
        from medh5.storage.codecs import PROFILES

        assert AnnotationPayload.__name__ == "AnnotationPayload"
        assert PROFILES["portable"].image.kwargs() == {
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
        }
        assert PROFILES["training"].image.kwargs()["compression"] == 32026


def test_the_rust_example_on_the_docs_page_is_the_tested_one() -> None:
    """`cargo test` compiles and runs the engine crate's README; the docs page
    shows the same block, so it is tested too --- as long as they are one."""
    import re

    block = re.compile(r"```rust\n.*?```\n", re.S)
    readme = (ROOT / "crates/medh5/README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs/reference/rust.md").read_text(encoding="utf-8")
    assert block.findall(page) == block.findall(readme)[:1]
    assert '#![doc = include_str!("../README.md")]' in (
        ROOT / "crates/medh5/src/lib.rs"
    ).read_text(encoding="utf-8")
