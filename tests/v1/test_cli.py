"""The command line (implementation plan §5)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

import medh5
from medh5._hdf5 import encode_attr
from medh5.cli import main
from medh5.cli._common import EXIT_ERROR, EXIT_OK, EXIT_USAGE, human_bytes, table
from tests.v1.conftest import write_legacy_sample


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
        states = {t["instance_id"]: t["states"] for t in payload["tracks"]}
        assert states[1] == {"tp0": "present", "tp1": "present"}
        assert states[3] == {"tp0": "present", "tp1": "resolved"}
        assert states[8] == {"tp0": "resolved", "tp1": "present"}

        code, out = run(capsys, "track", str(path))
        assert code == EXIT_OK
        assert "resolved" in out.out and "new" in out.out


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

    def test_publish_then_score(self, capsys, tmp_path):
        code, out = run(
            capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "SHA256SUMS").exists()

        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}])
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    def test_score_reports_a_wrong_answer(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": ["E101"]}])
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR
        assert "unexpected" in out.out and "E101" in out.out

    def test_score_warns_when_the_files_have_drifted(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        case = tmp_path / "core-minimal.medh5"
        case.write_bytes(case.read_bytes() + b"\x00")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}])
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert "differ" in out.out and "core-minimal.medh5" in out.out

    def test_score_without_a_suite_fails_cleanly(self, capsys, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("[]")
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR

    def test_usage(self, capsys):
        assert run(capsys, "conformance")[0] == EXIT_ERROR


class TestFixAndScrub:
    def test_fix_reports_and_changes_nothing(self, capsys, sample_path):
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_OK
        assert "nothing to fix" in out.out

    def test_fix_refuses_to_restamp_without_a_reason(self, capsys, sample_path):
        code, out = run(capsys, "fix", str(sample_path), "--rewrite-digests")
        assert code == EXIT_ERROR

    def test_fix_restamps_when_told_why(self, capsys, sample_path):
        code, out = run(
            capsys,
            "fix",
            str(sample_path),
            "--rewrite-digests",
            "--reason",
            "rebuilt by an external tool",
        )
        assert code == EXIT_OK
        assert "rewrote digests" in out.out
        assert "asserts nothing" in out.out

    def test_fix_rebuilds_a_stale_index(self, capsys, longitudinal_path):
        import h5py

        from medh5._hdf5 import encode_attr

        with h5py.File(longitudinal_path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        code, out = run(
            capsys, "fix", str(longitudinal_path), "--rebuild-index", "--json"
        )
        assert code == EXIT_OK
        result = json.loads(out.out)[0]
        assert result["changed"]
        assert result["rebuilt_index"] == ["organs_tp0"]

    def test_fix_leaves_a_healthy_file_alone(self, capsys, longitudinal_path):
        """An amend replaces the file, so doing one for nothing is not free."""
        before = longitudinal_path.stat().st_mtime_ns
        code, out = run(
            capsys, "fix", str(longitudinal_path), "--rebuild-index", "--json"
        )
        assert code == EXIT_OK
        result = json.loads(out.out)[0]
        assert not result["changed"]
        assert result["rebuilt_index"] == []
        assert longitudinal_path.stat().st_mtime_ns == before

    def test_scrub_finds_and_exits_non_zero(self, capsys, tmp_path):
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=(4, 4, 4), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.acquisition("CT", PatientName="Doe^Jane")
        code, out = run(capsys, "scrub", str(path))
        assert code == EXIT_ERROR
        assert "PatientName" in out.out
        assert "re-run with --apply" in out.out
        assert "NOT checked" in out.out

    def test_scrub_applies_and_attests(self, capsys, tmp_path):
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0", date="2026-02-03")
            w.add_grid("g", shape=(4, 4, 4), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.acquisition("CT", PatientName="Doe^Jane")
        code, out = run(
            capsys, "scrub", str(path), "--apply", "--date-shift-days", "-30"
        )
        assert code == EXIT_OK
        with medh5.open(path) as sample:
            assert "PatientName" not in sample.document.acquisition["CT"]
            assert sample.document.deidentification.method == "medh5-scrub"
        assert run(capsys, "scrub", str(path))[0] == EXIT_OK

    def test_scrub_on_a_clean_file_passes(self, capsys, sample_path):
        assert run(capsys, "scrub", str(sample_path))[0] == EXIT_OK


class TestDatasetCommands:
    """The cohort commands (plan §5): index, split, stats, check."""

    @pytest.fixture
    def cohort(self, tmp_path, label_set):
        from tests.v1.conftest import SHAPE, block, write_sample

        root = tmp_path / "cohort"
        root.mkdir()
        masks = {1: block(SHAPE, (2, 2, 2), 8)}
        for index, subject in enumerate(("A", "A", "B", "C", "D", "E")):
            path = root / f"case-{index}.medh5"
            write_sample(
                path, label_set=label_set, masks=masks, sample_id=f"case-{index}"
            )
            with medh5.amend(path) as writer:
                writer.identity(subject_id=f"subj-{subject}")
        return root

    def test_index_writes_a_manifest(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        code, out = run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        assert code == EXIT_OK
        assert "6 sample(s), 5 subject(s)" in out.out
        assert json.loads(manifest.read_text())["samples"] == 6

    def test_index_json_carries_the_digest(self, capsys, cohort, tmp_path):
        code, out = run(
            capsys,
            "dataset",
            "index",
            str(cohort),
            "-o",
            str(tmp_path / "m.json"),
            "--json",
        )
        payload = json.loads(out.out)
        assert len(payload["sha256"]) == 64
        assert payload["failed"] == []

    def test_index_reports_what_would_not_open(self, capsys, cohort, tmp_path):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        code, out = run(
            capsys, "dataset", "index", str(cohort), "-o", str(tmp_path / "m.json")
        )
        assert code == EXIT_OK
        assert "unreadable" in out.out

    def test_index_strict_stops(self, capsys, cohort, tmp_path):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        code, out = run(
            capsys,
            "dataset",
            "index",
            str(cohort),
            "-o",
            str(tmp_path / "m.json"),
            "--strict",
        )
        assert code == EXIT_ERROR

    def test_split_then_write_claims(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "-o",
            str(tmp_path / "split.json"),
            "--seed",
            "3",
            "--write-claims",
        )
        assert code == EXIT_OK
        assert "wrote split claims into 6 file(s)" in out.out
        payload = json.loads((tmp_path / "split.json").read_text())
        assert sum(payload["counts"].values()) == 6
        with medh5.open(sorted(cohort.glob("*.medh5"))[0]) as sample:
            assert sample.document.splits[0].manifest_sha256

    def test_split_warns_when_a_partition_gets_nothing(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--ratios",
            "train=0.98,val=0.01,test=0.01",
        )
        assert "got no groups" in out.out

    def test_split_k_folds_needs_a_fold_to_write_claims(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--k-folds",
            "3",
            "--write-claims",
        )
        assert code == EXIT_ERROR
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--k-folds",
            "3",
            "--write-claims",
            "--fold",
            "0",
            "--json",
        )
        assert code == EXIT_OK
        assert len(json.loads(out.out)["claims_written"]) == 6

    def test_split_rejects_bad_ratios(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        assert (
            run(capsys, "dataset", "split", str(manifest), "--ratios", "train")[0]
            == EXIT_ERROR
        )

    def test_stats_reports_intensities_and_classes(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys, "dataset", "stats", str(manifest), "-o", str(tmp_path / "s.json")
        )
        assert code == EXIT_OK
        assert "CT_tp0" in out.out
        payload = json.loads((tmp_path / "s.json").read_text())
        assert payload["samples"] == 6
        assert payload["images"]["CT_tp0"]["std"] > 0

    def test_stats_can_be_restricted_to_a_partition(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        run(capsys, "dataset", "split", str(manifest), "--write-claims", "--seed", "1")
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "stats",
            str(manifest),
            "--partition",
            "train",
            "--json",
        )
        assert code == EXIT_OK
        assert 0 < json.loads(out.out)["samples"] < 6

    def test_stats_says_so_when_no_sample_claims_the_partition(
        self, capsys, cohort, tmp_path
    ):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys, "dataset", "stats", str(manifest), "--partition", "train"
        )
        assert code == EXIT_ERROR

    def test_check_passes_a_clean_cohort(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(capsys, "dataset", "check", str(manifest))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_check_reports_a_missing_file(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        sorted(cohort.glob("*.medh5"))[0].unlink()
        code, out = run(capsys, "dataset", "check", str(manifest), "--json")
        assert code == EXIT_ERROR
        assert "C402" in {f["code"] for f in json.loads(out.out)["findings"]}

    def test_usage(self, capsys):
        assert run(capsys, "dataset")[0] == EXIT_ERROR


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


class TestPhase4Transforms:
    """`info` and `validate` must render and check §10 transforms."""

    @staticmethod
    def _build(tmp_path, name):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[name])
        return tmp_path / f"{name}.medh5"

    def test_info_renders_the_transform_table(self, capsys, tmp_path):
        path = self._build(tmp_path, "reg-displacement-composite")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "transforms" in out.out
        for kind in ("affine", "displacement", "composite"):
            assert kind in out.out
        assert "reg" in out.out

    def test_info_json_carries_frames_and_timepoints(self, capsys, tmp_path):
        path = self._build(tmp_path, "reg-affine-landmarks")
        code, out = run(capsys, "info", str(path), "--json")
        transforms = json.loads(out.out)["transforms"]
        assert transforms[0]["from_frame"] != transforms[0]["to_frame"]
        assert transforms[0]["timepoints"] == ["tp0", "tp1"]

    def test_validate_is_clean_for_every_registration_case(self, capsys, tmp_path):
        for name in (
            "reg-affine-landmarks",
            "reg-inverse-pair",
            "reg-displacement-composite",
            "reg-bspline",
        ):
            path = self._build(tmp_path, name)
            code, out = run(capsys, "validate", str(path), "--level", "strict")
            assert code == EXIT_OK, out.out

    def test_a_registered_pair_no_longer_warns_about_W911(self, capsys, tmp_path):
        """The warning exists to catch exactly the file this case is not."""
        path = self._build(tmp_path, "reg-affine-landmarks")
        code, out = run(capsys, "validate", str(path))
        assert "W911" not in out.out


class TestPhase5Curation:
    """Shards, provenance, agreement and the split audit (§2.2, §11, §12.3)."""

    def _members(self, tmp_path, label_set, masks, n=2):
        from tests.v1.conftest import write_sample

        return [
            write_sample(
                tmp_path / f"m{i}.medh5",
                label_set=label_set,
                masks=masks,
                sample_id=f"m{i}",
            )
            for i in range(n)
        ]

    def test_pack_ls_unpack_round_trip(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "cohort.medh5c"
        code, out = run(capsys, "pack", *map(str, members), "-o", str(shard))
        assert code == EXIT_OK
        assert "2 samples" in out.out

        code, out = run(capsys, "ls", str(shard))
        assert code == EXIT_OK
        assert "m0" in out.out and "content_id" in out.out

        code, out = run(capsys, "unpack", str(shard), "-o", str(tmp_path / "back"))
        assert code == EXIT_OK
        assert (tmp_path / "back" / "m1.medh5").exists()

    def test_pack_json_reports_sizes(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks, n=1)
        shard = tmp_path / "one.medh5c"
        code, out = run(capsys, "pack", str(members[0]), "-o", str(shard), "--json")
        payload = json.loads(out.out)
        assert payload["samples"] == 1
        assert payload["bytes"] > 0

    def test_pack_with_explicit_keys(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "k.medh5c"
        code, out = run(
            capsys,
            "pack",
            *map(str, members),
            "-o",
            str(shard),
            "--key",
            "alpha",
            "--key",
            "beta",
        )
        code, out = run(capsys, "ls", str(shard), "--json")
        assert [e["key"] for e in json.loads(out.out)["samples"]] == ["alpha", "beta"]

    def test_ls_on_a_sample_file_fails_clearly(self, capsys, sample_path):
        code, out = run(capsys, "ls", str(sample_path))
        assert code == EXIT_ERROR
        assert "collection" in out.err

    def test_unpack_selecting_a_missing_key(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks, n=1)
        shard = tmp_path / "s.medh5c"
        run(capsys, "pack", str(members[0]), "-o", str(shard))
        code, out = run(
            capsys, "unpack", str(shard), "-o", str(tmp_path / "x"), "--key", "ghost"
        )
        assert code == EXIT_ERROR

    def test_validate_reads_a_collection(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "v.medh5c"
        run(capsys, "pack", *map(str, members), "-o", str(shard))
        code, out = run(capsys, "validate", str(shard), "--level", "integrity")
        assert code == EXIT_OK, out.out

    def test_prov_prints_the_graph(self, capsys, sample_path):
        code, out = run(capsys, "prov", str(sample_path))
        assert code == EXIT_OK
        assert "agents" in out.out and "activities" in out.out
        assert "quality" in out.out
        assert "dicom-psi-profile" in out.out

    def test_prov_json(self, capsys, sample_path):
        code, out = run(capsys, "prov", str(sample_path), "--json")
        payload = json.loads(out.out)
        assert payload["provenance"]["agents"]
        assert payload["deidentification"]["method"] == "dicom-psi-profile"

    def test_prov_without_a_graph(self, capsys, tmp_path):
        import medh5
        from tests.v1.conftest import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "prov", str(path))
        assert "no provenance graph" in out.out
        assert "ABSENT (W903)" in out.out

    def test_agree_between_two_voxel_annotations(
        self, capsys, tmp_path, label_set, masks
    ):
        import medh5
        from tests.v1.conftest import SHAPE, block

        path = tmp_path / "raters.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation("r1", grid="g", masks=masks)
            w.add_segmentation(
                "r2",
                grid="g",
                masks={**masks, 2: block(SHAPE, (8, 8, 8), 3)},
            )
        code, out = run(capsys, "agree", str(path), "r1", "r2")
        assert code == EXIT_OK
        assert "dice =" in out.out and "liver" in out.out

        code, out = run(capsys, "agree", str(path), "r1", "r2", "--json", "--record")
        assert json.loads(out.out)["quality_agreement"]["metric"] == "dice"

        code, out = run(capsys, "agree", str(path), "r1", "nope")
        assert code == EXIT_ERROR

    def test_agree_between_instance_annotations(self, capsys, tmp_path, label_set):
        from tests.v1.test_tracking import write_series

        path = write_series(tmp_path / "series.medh5", label_set)
        code, out = run(capsys, "agree", str(path), "les_tp0", "les_tp1")
        assert code == EXIT_OK
        assert "matched" in out.out and "instance_id" in out.out

    def test_splits_audit_reports_leakage(self, capsys, tmp_path, label_set, masks):
        import medh5
        from tests.v1.conftest import write_sample

        paths = []
        for i in range(2):
            path = tmp_path / f"visit{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id="subj-shared")
                w.split(set_id="cv5", partition="train" if i == 0 else "test")
            paths.append(path)
        code, out = run(capsys, "splits", *map(str, paths))
        assert code == EXIT_ERROR
        assert "LEAK" in out.out
        assert "re-split rather than re-stamp" in out.out

        code, out = run(capsys, "splits", *map(str, paths), "--json")
        assert len(json.loads(out.out)["leaks"]) == 1

    def test_splits_audit_is_quiet_when_clean(self, capsys, sample_path):
        code, out = run(capsys, "splits", str(sample_path))
        assert code == EXIT_OK
        assert "no split claims" in out.out

    def test_splits_reports_conflicting_manifests(
        self, capsys, tmp_path, label_set, masks
    ):
        import medh5
        from tests.v1.conftest import write_sample

        paths = []
        for i in range(2):
            path = tmp_path / f"c{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id=f"subj-{i}")
                w.split(
                    set_id="cv5",
                    partition="train",
                    manifest_sha256=("a" if i == 0 else "b") * 64,
                )
            paths.append(path)
        code, out = run(capsys, "splits", *map(str, paths))
        assert code == EXIT_ERROR
        assert "W906" in out.out

    def test_splits_lists_unreadable_files(self, capsys, tmp_path):
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        code, out = run(capsys, "splits", str(bad))
        assert code == EXIT_ERROR
        assert "UNREADABLE" in out.out


class TestPhase6Performance:
    """``recompress`` and ``bench`` (§14.2, plan §4.3)."""

    def test_recompress_preserves_the_content_id(self, capsys, sample_path):
        import medh5

        with medh5.open(sample_path) as sample:
            before = sample.content_id
        code, out = run(capsys, "recompress", str(sample_path), "--profile", "archive")
        assert code == EXIT_OK
        assert "archive" in out.out
        assert "content_id" in out.out
        with medh5.open(sample_path) as sample:
            assert sample.content_id == before

    def test_recompress_json(self, capsys, sample_path):
        code, out = run(
            capsys, "recompress", str(sample_path), "--profile", "portable", "--json"
        )
        payload = json.loads(out.out)
        assert payload[0]["content_id_preserved"] is True
        assert payload[0]["profile"] == "portable"

    def test_recompress_out_takes_one_input(self, capsys, sample_path, tmp_path):
        code, out = run(
            capsys,
            "recompress",
            str(sample_path),
            str(sample_path),
            "--profile",
            "archive",
            "-o",
            str(tmp_path / "x.medh5"),
        )
        assert code == EXIT_ERROR
        assert "single input" in out.err

    def test_recompress_rechunks_only_when_asked(self, capsys, tmp_path, label_set):
        import h5py

        import medh5

        path = tmp_path / "chunked.medh5"
        with medh5.create(path, codec="balanced") as w:
            w.add_grid("g", shape=(48, 64, 64), spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                np.zeros((48, 64, 64), dtype=np.int16),
                grid="g",
                modality="CT",
            )
        with h5py.File(path) as handle:
            before = handle["images/CT"].chunks
        run(capsys, "recompress", str(path), "--profile", "archive")
        with h5py.File(path) as handle:
            assert handle["images/CT"].chunks == before

    def test_bench_runs_against_a_file(self, capsys, longitudinal_path):
        code, out = run(
            capsys,
            "bench",
            str(longitudinal_path),
            "--patch",
            "8",
            "--repeats",
            "2",
            "--no-throughput",
            "--json",
        )
        payload = json.loads(out.out)
        names = {m["name"] for m in payload["measurements"]}
        assert "meta_read_ms" in names
        assert "foreground_sample_ms" in names
        assert all("target" in m for m in payload["measurements"])

    def test_bench_text_output_names_the_targets(self, capsys, sample_path):
        code, out = run(
            capsys,
            "bench",
            str(sample_path),
            "--patch",
            "8",
            "--repeats",
            "2",
            "--no-throughput",
        )
        assert "target" in out.out
        assert code in (EXIT_OK, EXIT_ERROR)


class TestPhase7Convert:
    """``convert`` and ``migrate`` (plan §7)."""

    @pytest.fixture
    def nifti(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        shape = (12, 10, 6)
        ct = tmp_path / "ct.nii.gz"
        mask = tmp_path / "liver.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), str(ct))
        volume = np.zeros(shape, np.uint8)
        volume[2:6, 2:6, 1:3] = 1
        nib.save(nib.Nifti1Image(volume, affine), str(mask))
        return {"ct": ct, "mask": mask}

    def test_from_nifti_and_back(self, capsys, tmp_path, nifti):
        out = tmp_path / "case.medh5"
        code, printed = run(
            capsys,
            "convert",
            "from-nifti",
            str(out),
            "--image",
            f"CT={nifti['ct']}",
            "--mask",
            f"liver={nifti['mask']}",
            "--modality",
            "CT=CT",
        )
        assert code == EXIT_OK
        assert "DECISION" in printed.out
        assert out.exists()

        code, printed = run(
            capsys, "convert", "to-nifti", str(out), "CT", str(tmp_path / "b.nii.gz")
        )
        assert code == EXIT_OK
        assert (tmp_path / "b.nii.gz").exists()

    def test_the_report_is_written_and_machine_readable(self, capsys, tmp_path, nifti):
        report = tmp_path / "report.json"
        code, printed = run(
            capsys,
            "convert",
            "from-nifti",
            str(tmp_path / "c.medh5"),
            "--image",
            f"CT={nifti['ct']}",
            "--report",
            str(report),
            "--json",
        )
        assert code == EXIT_OK
        payload = json.loads(printed.out)
        assert payload["converter"] == "from-nifti"
        assert json.loads(report.read_text())["ok"] is True

    def test_a_malformed_pair_is_reported(self, capsys, tmp_path, nifti):
        code, printed = run(
            capsys, "convert", "from-nifti", str(tmp_path / "x.medh5"), "--image", "CT"
        )
        assert code == EXIT_ERROR
        assert "NAME=VALUE" in printed.err

    def test_convert_without_a_subcommand_explains_itself(self, capsys):
        code, printed = run(capsys, "convert")
        assert code == EXIT_ERROR
        assert "medh5 convert COMMAND" in printed.err

    def test_from_nnunet_and_to_nnunet(self, capsys, tmp_path):
        nib = pytest.importorskip("nibabel")
        root = tmp_path / "Dataset001_T"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (8, 6, 4)
        for channel in range(1):
            nib.save(
                nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
                str(root / "imagesTr" / f"C1_{channel:04d}.nii.gz"),
            )
        labels = np.zeros(shape, np.uint8)
        labels[1:4, 1:3, 1:2] = 1
        nib.save(
            nib.Nifti1Image(labels, np.eye(4)), str(root / "labelsTr" / "C1.nii.gz")
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "liver": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            )
        )
        code, printed = run(
            capsys, "convert", "from-nnunet", str(root), str(tmp_path / "out")
        )
        assert code == EXIT_OK
        assert (tmp_path / "out" / "C1.medh5").exists()

        code, printed = run(
            capsys,
            "convert",
            "to-nnunet",
            str(tmp_path / "back"),
            str(tmp_path / "out" / "C1.medh5"),
        )
        assert code == EXIT_OK
        assert (tmp_path / "back" / "Dataset001_medh5" / "dataset.json").exists()

    def test_migrate_writes_labels_then_migrates(self, capsys, tmp_path):
        shape = (6, 8, 10)
        mask = np.zeros(shape, bool)
        mask[1:4, 2:6, 3:8] = True
        old = tmp_path / "old.medh5"
        write_legacy_sample(
            old,
            images={"CT": np.zeros(shape, np.int16)},
            seg={"liver": mask},
            spacing=[1.0, 1.0, 1.0],
        )
        labels = tmp_path / "labels.json"
        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--write-labels",
            str(labels),
        )
        assert code == EXIT_OK
        assert "review before migrating" in printed.out
        assert json.loads(labels.read_text())["classes"]

        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--label-set",
            str(labels),
        )
        assert code == EXIT_OK
        written = sorted((tmp_path / "m").glob("*.medh5"))
        assert len(written) == 1
        with medh5.open(written[0]) as sample:
            assert sample.annotations["seg"].class_ids

    def test_migrate_reports_an_unreadable_file(self, capsys, tmp_path):
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        code, printed = run(capsys, "migrate", str(broken), "-o", str(tmp_path / "m"))
        assert code == EXIT_ERROR
        assert "WARNING" in printed.out
