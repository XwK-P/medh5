"""The validator: levels, codes and the report model (spec §15)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import CODES
from medh5.validate import (
    Diagnostic,
    Report,
    merge,
    validate_file,
    validate_paths,
    validate_root,
)
from tests.helpers import encode_attr, str_dtype, write_sample
from tests.kits import Numbered


def codes(path, level="semantic"):
    return set(validate_file(path, level=level).codes)


class TestLevels:
    def test_S15_1_levels_are_cumulative(self, sample_path):
        rules = {
            level: set(validate_file(sample_path, level=level).checked["rules"])
            for level in ("structural", "semantic", "integrity")
        }
        assert rules["structural"] < rules["semantic"] < rules["integrity"]

    def test_S15_1_strict_promotes_warnings(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            raw = handle["meta"][()]
            document = json.loads(
                raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            )
            document.pop("deidentification", None)
            del handle["meta"]
            handle.create_dataset("meta", data=json.dumps(document), dtype=str_dtype())
        assert validate_file(sample_path, level="semantic").ok
        strict = validate_file(sample_path, level="strict")
        assert not strict.ok
        assert "W903" in strict.codes

    def test_unknown_level_is_refused(self, sample_path):
        with (
            h5py.File(sample_path) as handle,
            pytest.raises(ValueError, match="unknown validation level"),
        ):
            validate_root(handle, level="paranoid")  # type: ignore[arg-type]

    def test_a_valid_sample_is_clean(self, sample_path):
        report = validate_file(sample_path, level="integrity")
        assert report.ok
        assert not report.errors

    def test_a_missing_file_reports_rather_than_raises(self, tmp_path):
        report = validate_file(tmp_path / "nope.medh5")
        assert not report.ok
        assert "E001" in report.codes

    def test_validate_paths_returns_one_report_each(self, sample_path):
        reports = validate_paths([sample_path, sample_path])
        assert len(reports) == 2


class TestReport:
    def test_json_round_trip(self, sample_path):
        report = validate_file(sample_path)
        payload = json.loads(report.dumps())
        assert payload["ok"] is True
        assert payload["level"] == "semantic"

    def test_format_is_readable(self, sample_path):
        text = validate_file(sample_path).format(verbose=True)
        assert str(sample_path) in text
        assert "OK" in text

    def test_merge_prefixes_locations(self):
        a = Report(path="a.medh5")
        a.add(Diagnostic("E001", "/", "boom"))
        b = Report(path="b.medh5")
        merged = merge([a, b])
        assert merged.diagnostics[0].location == "a.medh5:/"

    def test_diagnostic_carries_the_table_summary(self):
        diagnostic = Diagnostic("E102", "/grids/ct", "bad")
        assert diagnostic.summary == CODES["E102"].summary
        assert "E102" in str(diagnostic)


class TestRules:
    def test_S15_2_every_emitted_code_is_in_the_table(self, sample_path):
        from medh5.conformance import CASES

        for case in CASES:
            for code in (*case.errors, *case.warnings):
                assert code in CODES, code

    def test_missing_meta_is_reported(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            del handle["meta"]
        assert "E004" in codes(sample_path, "structural")

    def test_non_object_meta_is_reported(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            del handle["meta"]
            handle.create_dataset("meta", data="[1,2]", dtype=str_dtype())
        assert "E004" in codes(sample_path, "structural")

    def test_reserved_identifier_is_reported(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle["grids"].move("ct_tp0", "meta")
        assert "E003" in codes(sample_path, "structural")

    def test_S3_4_shared_frame_across_timepoints_warns(self, longitudinal_path):
        with h5py.File(longitudinal_path, "r+") as handle:
            handle["grids/ct_tp1"].attrs["frame_uid"] = encode_attr("pseudo:frame-tp0")
        assert "W910" in codes(longitudinal_path)

    def test_S7_7_partial_coverage_without_ignore_warns(
        self, tmp_path, label_set, masks
    ):

        path = write_sample(
            tmp_path / "p.medh5", label_set=label_set, masks=masks, annotated=[1]
        )
        assert "W904" in codes(path)

    def test_S7_2_class_in_two_layers_is_an_error(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            table = np.asarray(handle["annotations/organs_tp0/layer_class_ids"][...])
            table[1, 0] = table[0, 0]
            handle["annotations/organs_tp0/layer_class_ids"][...] = table
        assert "E404" in codes(sample_path)

    def test_S16_reserved_kind_is_refused(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle["annotations/organs_tp0"].attrs["kind"] = encode_attr("rle")
        assert "E401" in codes(sample_path, "structural")

    def test_profile_claims_are_checked(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle.attrs["medh5_profiles"] = encode_attr(["core", "reg", "cls"])
        found = codes(sample_path)
        assert "E009" in found

    def test_bulk_uncompressed_dataset_warns(self, tmp_path):
        import medh5

        shape = (64, 96, 96)
        path = tmp_path / "bulk.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
        with h5py.File(path, "r+") as handle:
            values = np.asarray(handle["images/CT"][...])
            attrs = dict(handle["images/CT"].attrs)
            del handle["images/CT"]
            node = handle["images"].create_dataset("CT", data=values)
            for key, value in attrs.items():
                node.attrs[key] = value
        assert "W902" in codes(path, "structural")

    def test_W908_fires_on_a_poor_colouring(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            group = handle["annotations/organs_tp0"]
            data = np.asarray(group["data"][...])
            table = np.asarray(group["layer_class_ids"][...])
            classes = sorted({int(v) for v in table.reshape(-1) if int(v)})
            shape = data.shape[1:]
            wide = np.zeros((6, *shape), dtype=data.dtype)
            for position, class_id in enumerate(classes):
                merged = np.zeros(shape, dtype=bool)
                for layer in range(data.shape[0]):
                    merged |= data[layer] == class_id
                wide[position][merged] = class_id
            new_table = np.zeros((6, 1), dtype=np.uint16)
            for position, class_id in enumerate(classes):
                new_table[position, 0] = class_id
            del group["data"], group["layer_class_ids"]
            group.create_dataset("data", data=wide)
            group.create_dataset("layer_class_ids", data=new_table)
        assert "W908" in codes(sample_path)

    def test_S15_2_E603_names_agents_too(self):
        assert medh5.CODES["E603"].summary == "unknown agent or activity type"


class TestStoredProbabilities:
    """§7.5 at the integrity level, on the bytes the digest pass reads: values
    are numbers in [0, 1], and a `normalized` map's classes sum to 1 (C09 of
    the 2.0 audit).  Only `threshold` was checked."""

    @staticmethod
    def _probmap(path: Path) -> Path:
        votes = np.zeros(Numbered.SHAPE)
        votes[2:4] = 0.75
        with Numbered.writer(path) as w:
            w.add_segmentation("soft", grid="g", probabilities={1: votes, 2: 1 - votes})
        return path

    @pytest.mark.parametrize("value", [np.nan, 1.5, -0.25])
    def test_C09_a_stored_value_outside_the_unit_interval_is_E411(
        self, tmp_path, value
    ):
        path = self._probmap(tmp_path / "pm.medh5")
        with h5py.File(path, "r+") as f:
            data = f["annotations/soft/data"]
            stored = data[...]
            stored[1, 3, 2, 1] = value
            data[...] = stored
        report = validate_file(path, level="integrity")
        (found,) = [d for d in report.diagnostics if d.code == "E411"]
        assert found.location == "/annotations/soft/data"
        assert "class 2, voxel (3, 2, 1)" in found.message
        # Values are read where the digests are: not below the integrity level.
        assert "E411" not in validate_file(path, level="semantic").codes

    def test_C09_a_normalized_map_that_does_not_sum_to_one_is_E404(self, tmp_path):
        path = self._probmap(tmp_path / "pm.medh5")
        assert validate_file(path, level="integrity").ok
        with h5py.File(path, "r+") as f:
            f["annotations/soft"].attrs["normalized"] = True
        assert "E404" not in validate_file(path, level="integrity").codes
        with h5py.File(path, "r+") as f:
            data = f["annotations/soft/data"]
            stored = data[...]
            stored[0, 5, 0, 0] = 0.25
            data[...] = stored
        report = validate_file(path, level="integrity")
        (found,) = [d for d in report.diagnostics if d.code == "E404"]
        assert "sum to 1.25 at voxel (5, 0, 0)" in found.message

    @staticmethod
    def _many(path: Path, planes: np.ndarray) -> Path:
        """One float16 class per plane of *planes*, on a 2x2x2 grid, declared
        normalized."""
        from medh5.labels import LabelClass, LabelSet

        n = len(planes)
        labels = LabelSet(
            "many",
            version="1.0.0",
            classes=[LabelClass(i, f"c{i}", f"C{i}") for i in range(1, n + 1)],
        )
        with medh5.create(path, sample_id="s1", subject_id="subj-1") as w:
            w.add_grid("g", shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros((2, 2, 2), np.int16), grid="g", modality="CT")
            w.label_set(labels)
            w.add_segmentation(
                "soft",
                grid="g",
                probabilities={i + 1: plane for i, plane in enumerate(planes)},
            )
        with h5py.File(path, "r+") as f:
            f["annotations/soft"].attrs["normalized"] = True
            assert f["annotations/soft/data"].dtype == np.float16
        return path

    def test_C09_a_map_that_lost_its_mass_is_E404_at_any_class_count(self, tmp_path):
        """The allowance was `classes · epsilon`: 1.000001 at 1,024 float16
        classes, so a normalized map whose every value was zeroed validated
        (C09 of the 2.0 re-audit).  It is the rounding of the values stored,
        so a softmax's rounding passes and a lost, halved or grown mass does
        not."""
        rng = np.random.default_rng(9)
        logits = rng.normal(size=(1024, 2, 2, 2))
        softmax = np.exp(logits) / np.exp(logits).sum(axis=0)
        path = self._many(tmp_path / "many.medh5", softmax)
        assert "E404" not in validate_file(path, level="integrity").codes
        for scale, shown in (
            (0.0, "sum to 0.0 "),
            (0.99, "sum to 0.98"),
            (1.01, "sum to 1.0"),
        ):
            with h5py.File(path, "r+") as f:
                data = f["annotations/soft/data"]
                data[...] = (softmax * scale).astype(np.float16)
            report = validate_file(path, level="integrity")
            (found,) = [d for d in report.diagnostics if d.code == "E404"]
            assert shown in found.message, (scale, found.message)

    @pytest.mark.parametrize(
        "shape",
        [(0, 2**62), (0, 2**62, 1, 1), (2, 0, 2**62, 2**62), (0,), (0, 8, 12, 12)],
    )
    def test_N07_a_malformed_normalized_shape_is_reported(self, tmp_path, shape):
        """The sums were allocated for the declared voxels before the shape was
        checked: `(0, 2**62)` asked for 2**65 bytes and the validator panicked
        --- a PanicException, which `except Exception` does not catch (N07 of
        the 2.0 re-audit).  A map is summed only on its own shape, a row per
        declared class on its grid; any other is the semantic level's E405."""
        path = self._probmap(tmp_path / "pm.medh5")
        with h5py.File(path, "r+") as f:
            group = f["annotations/soft"]
            del group["data"]
            group.create_dataset("data", shape=shape, dtype="f2")
            group.attrs["normalized"] = True
        report = validate_file(path, level="integrity")
        assert "E404" not in report.codes
        assert "E405" in report.codes

    @pytest.mark.parametrize("shape", [(1, 2**62), (2, 8, 12, 12 * 10**9)])
    def test_N07_a_dataset_too_large_to_hold_is_reported(self, tmp_path, shape):
        """A file may declare a dataset of any extent and store none of it: the
        digest pass asked the allocator for a row of 2**63 bytes (a panic) or of
        9 TB (an abort, killing the process) for a 10 KiB file.  A read that
        cannot be held is refused, and the validator reports it."""
        path = self._probmap(tmp_path / "pm.medh5")
        with h5py.File(path, "r+") as f:
            group = f["annotations/soft"]
            del group["data"]
            group.create_dataset(
                "data", shape=shape, dtype="f2", chunks=(1,) * len(shape)
            )
            group["data"].attrs["digest"] = "sha256:" + "0" * 64
        report = validate_file(path, level="integrity")
        (found,) = [d for d in report.diagnostics if d.code == "E001"]
        assert "more than this process can hold" in found.message


class TestCorruptFiles:
    """A validator is pointed at files of unknown provenance; it may not crash."""

    def test_S15_random_corruption_yields_a_diagnostic_not_a_traceback(
        self, sample_path, tmp_path
    ):
        """Bytes damaged past the header raise from inside h5py's traversal.

        Missing, truncated and non-HDF5 files were already handled as E001, but
        corruption that survives the open surfaced from the decompressor or the
        object walk instead -- so the command exited with a traceback, printed
        nothing on stdout, and `--json` produced no JSON for a pipeline to read.
        Failing to read an object is a finding about the file, not a crash.
        """
        import random

        rng = random.Random(11)
        size = sample_path.stat().st_size
        crashed = []
        for i in range(40):
            victim = tmp_path / f"corrupt{i}.medh5"
            victim.write_bytes(sample_path.read_bytes())
            with victim.open("r+b") as handle:
                for _ in range(rng.randint(1, 6)):
                    handle.seek(rng.randrange(size))
                    handle.write(bytes([rng.randrange(256)]))
            try:
                report = validate_file(victim, level="strict")
            except Exception as exc:
                crashed.append(f"{victim.name}: {type(exc).__name__}: {exc}")
                continue
            # Whatever it found, it has to be reportable and serialisable.
            json.loads(report.dumps())
        assert not crashed, "validate raised instead of reporting:\n" + "\n".join(
            crashed
        )

    def test_S15_invalid_utf8_in_a_string_is_read_not_trusted(self, sample_path):
        """A string declared UTF-8 holding bytes that are not --- a damaged file,
        or a writer that lied about the encoding.  Reading it trusted the
        declaration and handed the bytes on as text, and the report that should
        have carried the damage crashed instead.  The bytes are decoded with
        replacement, so the value reads back and the report serialises."""
        with h5py.File(sample_path, "r+") as handle:
            handle["images/CT_tp0"].attrs.create(
                "x_note",
                np.array(b"ok\xff\xfe", dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
        report = validate_file(sample_path, level="strict")
        json.loads(report.dumps())
        with medh5.open(sample_path) as sample:
            note = sample.root["images/CT_tp0"].attrs["x_note"]
        assert note == "ok\ufffd\ufffd"


class TestStrictPromotion:
    def test_S15_1_strict_promotes_warnings_in_the_counts_not_only_the_verdict(
        self, sample_path
    ):
        """§15.1: strict is the other levels "with warnings promoted to errors".

        The promotion reached `ok` and the exit code but not the counts, so the
        report said `FAILED ... (0 errors, 2 warnings)` -- self-contradictory on
        its face, and a CI job gating on `errors == 0` passed a file the same
        payload called not-ok.
        """
        lenient = validate_file(sample_path, level="semantic")
        strict = validate_file(sample_path, level="strict")
        assert lenient.warnings, "the fixture must carry at least one warning"

        assert lenient.ok and not lenient.errors
        assert not strict.ok
        assert len(strict.errors) == len(lenient.warnings) + len(lenient.errors)
        assert strict.warnings == ()
        assert len(strict.promoted) == len(lenient.warnings)

        payload = json.loads(strict.dumps())
        assert payload["ok"] is False
        assert payload["errors"] == len(strict.errors)
        assert payload["warnings"] == 0
        # The measured severity is still on each diagnostic, so what the §15.2
        # table says about a code is not lost.
        assert any(d["severity"] == "warning" for d in payload["diagnostics"])


class TestCompositeUnits:
    def test_S10_1_the_validator_rejects_a_mixed_unit_composite(self, tmp_path):
        """`check_chain()` rejected it while `validate` reported OK.

        The validator has its own composite rule, and it compared components and
        frames but not units -- so the advertised conformance check passed a
        chain the object model refuses, which is the one place the two must not
        disagree.
        """
        shape = (4, 8, 8)
        step = np.eye(4)
        step[:3, 3] = [1.0, 0.0, 0.0]
        path = tmp_path / "units.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1", index=1)
            for gid, frame, tp in (
                ("g0", "F0", "tp0"),
                ("gA", "FA", "tp0"),
                ("g1", "F1", "tp1"),
            ):
                w.add_grid(
                    gid,
                    shape=shape,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                    units="mm",
                )
                w.add_image(
                    f"CT_{gid}", np.zeros(shape, np.int16), grid=gid, modality="CT"
                )
            w.add_transform(
                "t1",
                kind="affine",
                matrix=step,
                from_frame="F0",
                to_frame="FA",
                units="mm",
            )
            w.add_transform(
                "t2",
                kind="affine",
                matrix=step,
                from_frame="FA",
                to_frame="F1",
                units="mm",
            )
            w.add_transform(
                "comp",
                kind="composite",
                components=["t1", "t2"],
                from_frame="F0",
                to_frame="F1",
                units="mm",
            )
        # Since 1.4.1 the writer runs this rule itself and would refuse the
        # chain, so the file is made the way a third-party writer would make it.
        import h5py

        with h5py.File(path, "r+") as handle:
            handle["transforms/t2"].attrs["units"] = "um"

        report = validate_file(path, level="semantic")
        assert not report.ok
        assert "E501" in report.codes
        assert any("units" in d.message for d in report.diagnostics)


class TestAtScale:
    """The checks hold at the sizes real files have."""

    def test_F21_S7_7_the_in_band_ignore_check_has_no_size_cap(self, tmp_path: Path):
        """E411 fired on a correct 512×512×256 labelmap: the check declined to
        look past 64M elements and then reported what it had not found.

        The many-slab path, with the only ignore voxel in the last slab, is the
        engine's Rust test (`annotations::payload::tests::s7_7_scans_reach_the_
        last_slab`, budgets down to one byte); this holds the validator to it.
        """
        path = tmp_path / "ignore.medh5"
        liver = np.zeros(Numbered.SHAPE, bool)
        liver[:2] = True
        region = np.zeros(Numbered.SHAPE, bool)
        region[-1, -1, -1] = True
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver},
                ignore=region,
                encoding="labelmap",
            )
        report = validate_file(path)
        assert not [c for c in report.codes if c in ("E411", "W904")], report.codes

    def test_F21_a_ct_sized_labelmap_with_an_ignore_region_validates(
        self, tmp_path: Path
    ):
        """The reproduction at its real size: 256×512×512, just past 64M elements.

        The datasets are grown in place with HDF5 chunks that are never
        written, which read back as the fill value --- so the file costs
        neither the memory nor the disk of a real CT, and the validator still
        has 67M elements to scan.  1.4.0 reported E411 and W904 here.
        """
        big = (256, 512, 512)
        liver = np.zeros(Numbered.SHAPE, bool)
        liver[:2] = True
        region = np.zeros(Numbered.SHAPE, bool)
        region[-1] = True
        path = tmp_path / "ct.medh5"
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: np.zeros(Numbered.SHAPE, bool)},
                ignore=region,
                annotated_classes=[1],
                encoding="labelmap",
            )
        with h5py.File(path, "r+") as handle:
            handle["grids/g"].attrs["shape"] = np.asarray(big, dtype=np.int64)
            for name, dtype in (
                ("images/CT", np.int16),
                ("annotations/seg/data", np.uint16),
            ):
                attrs = dict(handle[name].attrs)
                del handle[name]
                grown = handle.create_dataset(
                    name,
                    shape=big,
                    dtype=dtype,
                    chunks=(16, 128, 128),
                    compression="gzip",
                    fillvalue=0,
                )
                for key, value in attrs.items():
                    grown.attrs[key] = value
            data = handle["annotations/seg/data"]
            data[:8, :64, :64] = 1
            data[-4:, :64, :64] = 65535
        codes = validate_file(path).codes
        assert "E411" not in codes and "W904" not in codes, codes
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].has_ignore_region

    def test_P10_the_overlap_graph_is_read_in_slabs(self, tmp_path: Path):
        """The slab-bounded read is the engine's (`overlap_edges_within`, held
        at a 64-byte and a 1-byte budget by its Rust test); what W908 judges
        from that graph is checked here: one overlap needs two layers, so two
        layers are not too many."""
        path = tmp_path / "layers.medh5"
        liver = np.zeros(Numbered.SHAPE, bool)
        liver[1:6] = True
        lesion = np.zeros(Numbered.SHAPE, bool)
        lesion[5, 3:5, 3:5] = True
        spleen = np.zeros(Numbered.SHAPE, bool)
        spleen[7] = True
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: lesion, 3: spleen},
                encoding="layers",
            )
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].kind == "layers"
            assert sample.root["annotations/seg/data"].shape[0] == 2
        assert "W908" not in validate_file(path).codes
