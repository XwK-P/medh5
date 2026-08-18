"""The validator: levels, codes and the report model (spec §15)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from medh5._hdf5 import encode_attr, str_dtype
from medh5.errors import CODES
from medh5.validate import validate_file, validate_paths, validate_root
from medh5.validate.report import Diagnostic, Report, merge


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
        from tests.v1.conftest import write_sample

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
