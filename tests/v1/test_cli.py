"""The command line (implementation plan §5)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from medh5._hdf5 import encode_attr
from medh5.cli import main
from medh5.cli._common import EXIT_ERROR, EXIT_OK, EXIT_USAGE, human_bytes, table


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


class TestEntryPoint:
    def test_no_command_prints_help(self, capsys):
        code, out = run(capsys)
        assert code == EXIT_USAGE
        assert "COMMAND" in out.out

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            main(["--version"])
        assert "medh5" in capsys.readouterr().out


class TestInfo:
    def test_text_output_names_every_section(self, capsys, longitudinal_path):
        code, out = run(capsys, "info", str(longitudinal_path))
        assert code == EXIT_OK
        for section in ("timepoints", "grids", "images", "annotations", "label set"):
            assert section in out.out

    def test_json_output_is_machine_readable(self, capsys, sample_path):
        code, out = run(capsys, "info", str(sample_path), "--json")
        assert code == EXIT_OK
        payload = json.loads(out.out)
        assert payload["sample_id"] == "case"
        assert payload["grids"][0]["id"] == "ct_tp0"

    def test_a_missing_file_exits_nonzero(self, capsys, tmp_path):
        code, out = run(capsys, "info", str(tmp_path / "nope.medh5"))
        assert code == EXIT_ERROR
        assert "medh5:" in out.err

    def test_tree_labels_spec_roles(self, capsys, longitudinal_path):
        code, out = run(capsys, "tree", str(longitudinal_path))
        assert code == EXIT_OK
        assert "sample document (§2.4)" in out.out
        assert "kind=" in out.out


class TestValidate:
    def test_a_clean_file_exits_zero(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_a_broken_file_exits_one(self, capsys, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            del handle.attrs["medh5_version"]
        code, out = run(capsys, "validate", str(sample_path), "--level", "structural")
        assert code == EXIT_ERROR
        assert "E001" in out.out

    def test_json_output(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path), "--json")
        payload = json.loads(out.out)
        assert payload[0]["ok"] is True

    def test_profile_override(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path), "--profile", "reg")
        assert code == EXIT_ERROR
        assert "E009" in out.out

    def test_verbose_prints_the_code_summary(self, capsys, longitudinal_path):
        code, out = run(capsys, "validate", str(longitudinal_path), "-v")
        assert "->" in out.out


class TestVerify:
    def test_clean(self, capsys, sample_path):
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_mismatch_is_named(self, capsys, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(0 for _ in block.shape)] = 4
            data[...] = block
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_ERROR
        assert "MISMATCH" in out.out

    def test_partial(self, capsys, sample_path):
        code, out = run(
            capsys, "verify", str(sample_path), "--partial", "images/CT_tp0", "--json"
        )
        assert code == EXIT_OK
        assert json.loads(out.out)[0]["checked"] == 1

    def test_stale_index_is_reported(self, capsys, longitudinal_path):
        with h5py.File(longitudinal_path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        code, out = run(capsys, "verify", str(longitudinal_path))
        assert "STALE" in out.out


class TestLongitudinal:
    def test_timeline(self, capsys, longitudinal_path):
        code, out = run(capsys, "timeline", str(longitudinal_path))
        assert code == EXIT_OK
        assert "tp0" in out.out and "tp1" in out.out

    def test_timeline_json(self, capsys, longitudinal_path):
        code, out = run(capsys, "timeline", str(longitudinal_path), "--json")
        payload = json.loads(out.out)
        assert payload[1]["images"] == ["CT_tp1"]

    def test_track_without_instances(self, capsys, longitudinal_path):
        code, out = run(capsys, "track", str(longitudinal_path))
        assert code == EXIT_OK
        assert "no instance-carrying" in out.out

    def test_track_reports_persisted_resolved_and_new(
        self, capsys, tmp_path, label_set
    ):
        import medh5
        from medh5.annotations.voxel import InstanceInput
        from tests.v1.conftest import SHAPE

        def lesion(origin):
            mask = np.zeros(SHAPE, dtype=bool)
            mask[tuple(slice(o, o + 3) for o in origin)] = True
            return mask

        path = tmp_path / "t.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(SHAPE, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
            w.add_segmentation(
                "les_tp0",
                grid="g_tp0",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 3, mask=lesion((5, 5, 5))),
                ],
            )
            w.add_segmentation(
                "les_tp1",
                grid="g_tp1",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 8, mask=lesion((9, 9, 9))),
                ],
            )
        code, out = run(capsys, "track", str(path), "--json")
        payload = json.loads(out.out)
        assert payload["1"]["state"] == "persisted"
        assert payload["3"]["state"] == "resolved"
        assert payload["8"]["state"] == "new"


class TestLabels:
    def test_show(self, capsys, sample_path):
        code, out = run(capsys, "labels", "show", str(sample_path))
        assert code == EXIT_OK
        assert "liver" in out.out

    def test_show_json(self, capsys, sample_path):
        code, out = run(capsys, "labels", "show", str(sample_path), "--json")
        assert json.loads(out.out)["id"] == "test-v1"

    def test_show_without_a_label_set(self, capsys, tmp_path):
        import medh5
        from tests.v1.conftest import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "labels", "show", str(path))
        assert "no label set" in out.out

    def test_registry_list(self, capsys):
        code, out = run(capsys, "labels", "registry", "list")
        assert code == EXIT_OK
        assert "amos22-organs" in out.out

    def test_check_agrees_with_itself(self, capsys, sample_path):
        code, out = run(capsys, "labels", "check", str(sample_path), str(sample_path))
        assert code == EXIT_OK

    def test_check_detects_drift(self, capsys, tmp_path, label_set, masks):
        from medh5.labels.labelset import LabelClass, LabelSet
        from tests.v1.conftest import write_sample

        a = write_sample(tmp_path / "a.medh5", label_set=label_set, masks=masks)
        other = LabelSet("other-v1", classes=[LabelClass(1, "liver", "Liver")])
        b = write_sample(tmp_path / "b.medh5", label_set=other, masks={1: masks[1]})
        code, out = run(capsys, "labels", "check", str(a), str(b))
        assert code == EXIT_ERROR
        assert "distinct vocabularies" in out.out

    def test_usage_errors(self, capsys, sample_path):
        assert run(capsys, "labels")[0] == EXIT_ERROR
        assert run(capsys, "labels", "registry")[0] == EXIT_ERROR


class TestSegAndIndex:
    def test_stats_reports_the_cost_model(self, capsys, sample_path):
        code, out = run(capsys, "seg", "stats", str(sample_path), "organs_tp0")
        assert code == EXIT_OK
        assert "overlap graph" in out.out
        assert "<- stored" in out.out

    def test_stats_json(self, capsys, sample_path):
        code, out = run(
            capsys, "seg", "stats", str(sample_path), "organs_tp0", "--json"
        )
        payload = json.loads(out.out)
        assert payload["kind"] == "layers"
        assert payload["cost_bytes"]["bitmask"] > 0

    def test_stats_on_a_missing_annotation(self, capsys, sample_path):
        code, out = run(capsys, "seg", "stats", str(sample_path), "nope")
        assert code == EXIT_ERROR

    def test_convert_dry_run_writes_nothing(self, capsys, sample_path):
        before = sample_path.read_bytes()
        code, out = run(
            capsys,
            "seg",
            "convert",
            str(sample_path),
            "organs_tp0",
            "--to",
            "bitmask",
            "--dry-run",
        )
        assert code == EXIT_OK
        assert "would re-encode" in out.out
        assert sample_path.read_bytes() == before

    def test_convert_is_lossless(self, capsys, sample_path, masks):
        import medh5

        code, out = run(
            capsys, "seg", "convert", str(sample_path), "organs_tp0", "--to", "bitmask"
        )
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "bitmask"
            for class_id, mask in masks.items():
                assert np.array_equal(seg.dense([class_id])[0], mask)

    def test_index_build(self, capsys, sample_path):
        import medh5

        code, out = run(
            capsys, "index", "build", str(sample_path), "--max-coords", "32"
        )
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            assert "organs_tp0" in sample.index

    def test_index_build_on_a_sample_without_annotations(self, capsys, tmp_path):
        import medh5
        from tests.v1.conftest import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "index", "build", str(path))
        assert code == EXIT_ERROR

    def test_usage_errors(self, capsys):
        assert run(capsys, "seg")[0] == EXIT_ERROR
        assert run(capsys, "index")[0] == EXIT_ERROR


class TestConformanceCommands:
    def test_list(self, capsys):
        code, out = run(capsys, "conformance", "list")
        assert code == EXIT_OK
        assert "core-minimal" in out.out

    def test_list_json(self, capsys):
        code, out = run(capsys, "conformance", "list", "--json")
        assert any(c["name"] == "seg-layers" for c in json.loads(out.out))

    def test_build_and_run_one_case(self, capsys, tmp_path):
        code, out = run(
            capsys, "conformance", "build", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert (tmp_path / "core-minimal.medh5").exists()
        assert (tmp_path / "expected.json").exists()
        code, out = run(
            capsys, "conformance", "run", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    def test_usage(self, capsys):
        assert run(capsys, "conformance")[0] == EXIT_ERROR


class TestHelpers:
    def test_human_bytes(self):
        assert human_bytes(512) == "512 B"
        assert human_bytes(2048) == "2.0 KiB"
        assert human_bytes(5 * 1024**3) == "5.0 GiB"

    def test_table_pads_columns(self):
        rendered = table([["a", 1], ["bbbb", 22]], ["k", "v"])
        assert rendered.splitlines()[0].startswith("k     v")

    def test_empty_table(self):
        assert table([], ["k", "v"]).splitlines()[0] == "k  v"


class TestPhase3Kinds:
    """`info` and `validate` must handle the §8/§9 kinds, including grid-less ones."""

    @staticmethod
    def _build(tmp_path, name):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[name])
        return tmp_path / f"{name}.medh5"

    def test_info_renders_boxes_and_obb(self, capsys, tmp_path):
        path = self._build(tmp_path, "det-boxes-obb")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "boxes" in out.out and "obb" in out.out
        assert "det" in out.out

    def test_info_renders_a_grid_less_classification(self, capsys, tmp_path):
        path = self._build(tmp_path, "cls-staging-and-change")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "classification" in out.out
        assert "tp0,tp1" in out.out  # the change label names the visits compared

    def test_info_json_covers_every_kind(self, capsys, tmp_path):
        path = self._build(tmp_path, "shapes-contours-mesh")
        code, out = run(capsys, "info", str(path), "--json")
        kinds = {a["kind"] for a in json.loads(out.out)["annotations"]}
        assert kinds == {"contours", "mesh", "points"}

    def test_validate_is_clean_for_every_phase3_case(self, capsys, tmp_path):
        for name in (
            "det-boxes-obb",
            "det-keypoints",
            "det-boxes-world",
            "shapes-contours-mesh",
            "cls-staging-and-change",
        ):
            path = self._build(tmp_path, name)
            code, out = run(capsys, "validate", str(path), "--level", "integrity")
            assert code == EXIT_OK, out.out

    def test_seg_convert_refuses_a_geometric_annotation(self, capsys, tmp_path):
        path = self._build(tmp_path, "det-boxes-obb")
        code, out = run(
            capsys, "seg", "convert", str(path), "lesions", "--to", "layers"
        )
        assert code == EXIT_ERROR
        assert "not a voxel encoding" in out.err

    def test_tree_names_every_dataset(self, capsys, tmp_path):
        path = self._build(tmp_path, "shapes-contours-mesh")
        code, out = run(capsys, "tree", str(path))
        assert code == EXIT_OK
        for dataset in ("contour_offsets", "contour_role", "faces", "vertices"):
            assert dataset in out.out
