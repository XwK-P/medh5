"""The command line."""

from __future__ import annotations

import errno
import io
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from medh5.labels import LabelClass, LabelSet
from tests.helpers import (
    block,
    encode_attr,
    write_legacy_sample,
    write_raters,
    write_sample,
    write_series,
)
from tests.kits import Framed, Hierarchy, Numbered, Organs


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


class TestEntryPoint:
    def test_no_command_prints_help(self, capsys):
        code, out = run(capsys)
        assert code == EXIT_USAGE
        assert "COMMAND" in out.out

    def test_version_and_help_return_their_code(self, capsys):
        assert main(["--version"]) == EXIT_OK
        assert "medh5" in capsys.readouterr().out
        assert main(["--help"]) == EXIT_OK
        assert "COMMAND" in capsys.readouterr().out
        assert main(["--no-such-flag"]) == EXIT_USAGE

    def test_an_argument_that_is_not_utf8_is_a_usage_error(self, capsys):
        """A Latin-1 file name a shell glob expanded reaches ``sys.argv`` as
        surrogate escapes.  It ended the console script in a traceback, and
        the standalone binary in a panic; it is refused as a usage error."""
        code, out = run(capsys, "info", "caf\udce9.medh5")
        assert code == EXIT_USAGE
        assert out.err == "medh5: argument 'caf�.medh5' is not valid UTF-8\n"

    def test_the_grammar_lists_each_option_once(self):
        """``command_tree()`` listed ``-h``, ``--help`` and ``--version``
        twice at every level: once as clap builds them, once by hand."""
        from medh5.cli import command_tree

        def check(node: dict[str, Any], where: str) -> None:
            options = node["options"]
            assert len(options) == len(set(options)), where
            assert {"-h", "--help"} <= set(options), where
            for name, sub in node["commands"].items():
                check(sub, f"{where} {name}")

        tree = command_tree()
        assert "--version" in tree["options"]
        check(tree, "medh5")

    @pytest.mark.skipif(sys.platform == "win32", reason="Ctrl-C is no signal there")
    def test_ctrl_c_ends_the_console_script_at_once(self, tmp_path):
        """The command runs in native code, where Python's handler only marks
        the interrupt: ``conformance build`` ran on to the end after Ctrl-C,
        then printed a ``KeyboardInterrupt`` traceback."""
        import signal
        import subprocess
        import time

        corpus = tmp_path / "corpus"
        child = subprocess.Popen(
            [sys.executable, "-m", "medh5.cli", "conformance", "build", str(corpus)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            # A case on disk: the command is under way, in native code.
            deadline = time.monotonic() + 120
            while not (corpus.is_dir() and any(corpus.iterdir())):
                assert child.poll() is None, child.stderr.read()
                assert time.monotonic() < deadline
                time.sleep(0.005)
            child.send_signal(signal.SIGINT)
            _, err = child.communicate(timeout=60)
        finally:
            child.kill()
            child.wait()
        assert child.returncode == -signal.SIGINT
        assert b"KeyboardInterrupt" not in err

    def test_only_the_console_script_takes_over_ctrl_c(self, monkeypatch):
        """A caller passing *argv* keeps its own handler; the console script
        puts back the one it found."""
        import signal

        from medh5 import cli

        during = []
        real = cli._core.cli_main

        def spy(args, host):
            during.append(signal.getsignal(signal.SIGINT))
            return real(args, host)

        monkeypatch.setattr(cli._core, "cli_main", spy)
        monkeypatch.setattr(sys, "argv", ["medh5", "--version"])
        before = signal.getsignal(signal.SIGINT)
        assert main() == EXIT_OK
        assert main(["--version"]) == EXIT_OK
        assert during == [signal.SIG_DFL, before]
        assert signal.getsignal(signal.SIGINT) is before

    def test_a_missing_name_is_a_handled_cli_error(self, tmp_path: Path, capsys):
        """`convert to-nifti FILE <unknown image>` ended in a KeyError traceback."""
        pytest.importorskip("nibabel")
        from medh5.cli import main

        path = tmp_path / "a.medh5"
        with Framed.writer(path):
            pass
        code = main(
            ["convert", "to-nifti", str(path), "PET", str(tmp_path / "x.nii.gz")]
        )
        assert code == 1
        assert capsys.readouterr().err.startswith("medh5: no image 'PET'")

    def test_a_bare_key_is_named(self):
        from medh5.cli import _what

        assert _what(KeyError("PET")) == "no such entry: 'PET'"
        assert _what(KeyError("no image 'PET'; available: []")).startswith("no image")

    def test_every_command_words_a_failed_lookup_alike(
        self, capsys, tmp_path, sample_path
    ):
        """``seg stats``, ``agree`` and the clinical commands printed the
        engine's sentence in quotes, and ``agree`` put a second "no such
        annotation" before it."""
        from medh5.collection import pack

        for argv in (
            ["seg", "stats", str(sample_path), "nope"],
            ["agree", str(sample_path), "nope", "organs_tp0"],
        ):
            code, out = run(capsys, *argv)
            assert code == EXIT_ERROR
            assert out.err.startswith("medh5: no annotation 'nope'; available: ["), argv
        shard = tmp_path / "s.medh5c"
        pack([sample_path], shard)
        code, expected = run(capsys, "info", str(shard), "--key", "nope")
        assert code == EXIT_ERROR
        assert expected.err.startswith("medh5: collection has no sample 'nope'")
        code, out = run(capsys, "clinical", "show", str(shard), "--key", "nope")
        assert (code, out.err) == (EXIT_ERROR, expected.err)


class TestOutput:
    """What the command line does when its stdout cannot take the output."""

    def test_output_lost_to_a_full_disk_fails_the_command(self, capsys, monkeypatch):
        class Full(io.StringIO):
            def write(self, text: str) -> int:
                raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(sys, "stdout", Full())
        assert main(["conformance", "list"]) == EXIT_ERROR
        assert "medh5: cannot write output: " in capsys.readouterr().err

    def test_a_reader_that_left_early_keeps_the_commands_own_code(
        self, monkeypatch, tmp_path
    ):
        """``medh5 validate ... | head`` exited 0 whatever the files held."""

        class Gone(io.StringIO):
            def write(self, text: str) -> int:
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")

        monkeypatch.setattr(sys, "stdout", Gone())
        assert main(["validate", str(tmp_path / "missing.medh5")]) == EXIT_ERROR
        assert main(["conformance", "list"]) == EXIT_OK

    def test_what_the_stdout_encoding_cannot_spell_is_escaped_not_lost(
        self, monkeypatch, longitudinal_path
    ):
        """Windows writes a redirected stdout in its ANSI code page, which has
        no box-drawing characters: ``tree`` dropped every line drawn with
        them, and exited 0."""
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)
        assert main(["tree", str(longitudinal_path)]) == EXIT_OK
        stream.flush()
        printed = buffer.getvalue().decode("cp1252")
        assert "\\u251c\\u2500\\u2500 meta" in printed
        assert "sample document (§2.4)" in printed


class TestUsageErrors:
    """A usage error exits 2, as documented, whichever rule it breaks; the
    files named are never opened."""

    @pytest.mark.parametrize("group", ["clinical", "task", "cache"])
    def test_a_command_of_commands_needs_one(self, capsys, group):
        code, out = run(capsys, group)
        assert code == EXIT_USAGE
        assert f"Usage: medh5 {group}" in out.err

    @pytest.mark.parametrize(
        "cutoffs", [[], ["--cutoff-us", "0", "--cutoff-hours", "1"]]
    )
    def test_a_selection_takes_its_cutoff_once(self, capsys, tmp_path, cutoffs):
        missing = str(tmp_path / "missing.medh5")
        assert run(capsys, "clinical", "select", missing, *cutoffs)[0] == EXIT_USAGE

    @pytest.mark.parametrize("hours", ["nan", "inf", "-inf", "1e300"])
    def test_a_cutoff_in_hours_is_a_time_on_the_subject_clock(
        self, capsys, tmp_path, hours
    ):
        """NaN selected at 0 us and infinity at the clock's ends, certified."""
        missing = str(tmp_path / "missing.medh5")
        code, out = run(
            capsys, "clinical", "select", missing, f"--cutoff-hours={hours}"
        )
        assert code == EXIT_USAGE
        assert "subject clock" in out.err

    @pytest.mark.parametrize(
        "argv",
        [
            ["index", "build", "--max-coords", "0"],
            ["index", "build", "--max-coords", "-7"],
            ["index", "build", "--occupancy", "-3"],
            ["dataset", "stats", "--stride", "0"],
            ["dataset", "stats", "--workers", "-1"],
            ["bench", "--patch", "0"],
            ["bench", "--repeats", "0"],
        ],
    )
    def test_a_count_below_its_least_value_is_refused(self, capsys, tmp_path, argv):
        """``--max-coords -7`` stored no coordinates, ``--occupancy -3`` built
        no map, and a stride of 0 or a negative worker count was taken as
        some other number, all without a word."""
        *command, option, value = argv
        missing = str(tmp_path / "missing")
        code, out = run(capsys, *command, missing, f"{option}={value}")
        assert code == EXIT_USAGE
        assert f"invalid value '{value}' for '{option}" in out.err

    def test_pseudonymising_ids_needs_apply_and_a_salt(self, capsys, tmp_path):
        missing = str(tmp_path / "missing.medh5")
        code, out = run(capsys, "scrub", missing, "--pseudonymise-ids")
        assert code == EXIT_USAGE and "--apply" in out.err
        code, out = run(capsys, "scrub", missing, "--pseudonymise-ids", "--apply")
        assert code == EXIT_USAGE and "needs --salt" in out.err


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

    @pytest.mark.parametrize(
        ("case", "member", "role"),
        [
            ("clinical-one-visit-history", "clinical", "clinical history (1.1 §3)"),
            ("collection-two-samples", "samples", "collection members (§2.2)"),
        ],
    )
    def test_tree_names_the_clinical_group_and_a_shards_members(
        self, capsys, tmp_path, case, member, role
    ):
        """Both were listed as extension objects (§16)."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[case])
        path = str(next(tmp_path.glob(f"{case}.*")))
        code, out = run(capsys, "tree", path)
        assert code == EXIT_OK
        assert f"├── {member}/" in out.out and f"# {role}" in out.out
        code, out = run(capsys, "tree", path, "--json")
        roles = {o["name"]: o["role"] for o in json.loads(out.out)["objects"]}
        assert roles[member] == role

    def test_the_inspection_commands_take_a_shard_with_a_key(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import main
        from medh5.collection import pack

        members = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.medh5"
            with Organs.writer(path) as w:
                w.add_segmentation("seg", grid="g", masks=Organs.blocks())
            members.append(path)
        shard = tmp_path / "s.medh5c"
        pack(members, shard)

        assert main(["info", str(shard)]) == 0
        assert "per-sample detail" in capsys.readouterr().out
        for command in ("info", "tree", "verify", "timeline", "track"):
            assert main([command, str(shard), "--key", "a"]) == 0, command
            capsys.readouterr()
        # A per-sample command with no key says which keys there are.
        assert main(["timeline", str(shard)]) == 1
        assert "--key" in capsys.readouterr().err

    def test_S2_2_a_later_minors_file_lists_its_projection(self, tmp_path, capsys):
        """One grid of an axis kind only a later minor defines failed `info`
        whole.  It lists what the projection reads, names what it leaves
        out, and `--json` says why; a file of 1.1 fails as it did."""
        path = tmp_path / "later.medh5"
        with Organs.writer(path) as w:
            w.add_grid(
                "h", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("MR", np.zeros(Organs.SHAPE, np.int16), grid="h", modality="MR")
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        with h5py.File(path, "r+") as handle:
            kinds = ["depth", "spatial", "spatial"]
            handle["grids/g"].attrs["axis_kinds"] = encode_attr(kinds)
            handle.attrs["medh5_version"] = encode_attr("1.2")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "MR" in out.out
        assert "left out     grids/g, images/CT, annotations/seg (W913" in out.out
        code, out = run(capsys, "info", str(path), "--json")
        payload = json.loads(out.out)
        assert [grid["id"] for grid in payload["grids"]] == ["h"]
        assert "axis kind 'depth'" in payload["left_out"]["grids/g"]
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("1.1")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_ERROR
        assert "E110" in out.err


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

    def test_S15_2_verbose_adds_what_each_code_means(self, capsys, longitudinal_path):
        """Every diagnostic is printed either way; ``-v``, which the help
        said printed every diagnostic rather than only the summary, adds
        under each the meaning of its code."""
        _, plain = run(capsys, "validate", str(longitudinal_path))
        _, verbose = run(capsys, "validate", str(longitudinal_path), "-v")
        lines = verbose.out.splitlines()
        meanings = [line for line in lines if line.lstrip().startswith("-> ")]
        assert meanings
        assert [
            line for line in lines if line not in meanings
        ] == plain.out.splitlines()
        assert main(["validate", "--help"]) == EXIT_OK
        assert "also print what each diagnostic's code means" in capsys.readouterr().out


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

    def test_verify_names_the_content_id_state(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        path = tmp_path / "v.medh5"
        with medh5.create(path, sample_id="v") as w:
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
        assert main(["verify", str(path)]) == 0
        assert "content_id ok" in capsys.readouterr().out
        assert main(["verify", str(path), "--partial", "images/CT"]) == 0
        assert "content_id not verified" in capsys.readouterr().out

    def test_S13_1_a_digest_that_cannot_be_checked_is_named(self, capsys, sample_path):
        """`verify` said FAILED and nothing else for a digest that is not
        ALGO:HEX, and for a `--partial` name with no digest (a group, a typo)."""
        code, out = run(
            capsys, "verify", str(sample_path), "--partial", "annotations/organs_tp0"
        )
        assert code == EXIT_ERROR
        assert "MALFORMED annotations/organs_tp0" in out.out
        with h5py.File(sample_path, "r+") as handle:
            handle["images/CT_tp0"].attrs["digest"] = encode_attr("sha256:nothex")
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_ERROR
        assert "MALFORMED images/CT_tp0" in out.out


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
        assert "no instance-carrying annotations in this sample" in out.out

    def test_track_says_why_the_instances_it_found_do_not_join(
        self, capsys, tmp_path, label_set
    ):
        """It said there were no instance-carrying annotations, then listed
        the ones it skipped."""
        from tests.helpers import SHAPE

        path = tmp_path / "kp.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_keypoints(
                "kp",
                np.zeros((1, 2, 3)),
                keypoint_classes=["lesion", "lesion"],
                class_ids=["lesion"],
                grid="g",
                instance_ids=[7],
            )
        code, out = run(capsys, "track", str(path))
        assert code == EXIT_OK
        assert "no instance-carrying annotation in this sample joins" in out.out
        assert "skipped 'kp'" in out.out

    def test_track_reports_persisted_resolved_and_new(
        self, capsys, tmp_path, label_set
    ):
        import medh5
        from tests.helpers import SHAPE

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
        from tests.helpers import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "labels", "show", str(path))
        assert "no label set" in out.out
        # `--json` is a document whatever there is to say: here, `null`.
        code, out = run(capsys, "labels", "show", str(path), "--json")
        assert code == EXIT_OK
        assert json.loads(out.out) is None

    def test_registry_list(self, capsys):
        code, out = run(capsys, "labels", "registry", "list")
        assert code == EXIT_OK
        assert "amos22-organs" in out.out

    def test_check_agrees_with_itself(self, capsys, sample_path):
        code, out = run(capsys, "labels", "check", str(sample_path), str(sample_path))
        assert code == EXIT_OK

    def test_check_detects_drift(self, capsys, tmp_path, label_set, masks):
        from tests.helpers import write_sample

        a = write_sample(tmp_path / "a.medh5", label_set=label_set, masks=masks)
        other = LabelSet("other-v1", classes=[LabelClass(1, "liver", "Liver")])
        b = write_sample(tmp_path / "b.medh5", label_set=other, masks={1: masks[1]})
        code, out = run(capsys, "labels", "check", str(a), str(b))
        assert code == EXIT_ERROR
        assert "distinct vocabularies" in out.out

    def test_usage_errors(self, capsys):
        for argv in (["labels"], ["labels", "registry"]):
            code, out = run(capsys, *argv)
            assert code == EXIT_USAGE, argv
            assert "Usage: medh5 " + " ".join(argv) in out.err


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

    @pytest.mark.parametrize(
        ("case", "annotation", "kept"),
        [("seg-instances", "lesions", "instances"), ("seg-probmap", "soft", "probmap")],
    )
    def test_S7_6_stats_recommends_only_what_keeps_the_annotation(
        self, capsys, tmp_path, case, annotation, kept
    ):
        """`seg stats` recommended a dense encoding for `instances` and called
        the re-encode lossless; the command it printed refuses (E404), as
        dropping every instance_id.  A probability map was recommended
        binary layers, which threshold it."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[case])
        path = str(tmp_path / f"{case}.medh5")
        code, out = run(capsys, "seg", "stats", path, annotation, "--json")
        assert code == EXIT_OK
        assert json.loads(out.out)["recommended"] == kept
        code, out = run(capsys, "seg", "stats", path, annotation)
        assert "seg convert" not in out.out and "losslessly" not in out.out

    def test_stats_names_a_lossless_re_encoding(self, capsys, tmp_path):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=["seg-bitmask"])
        path = str(tmp_path / "seg-bitmask.medh5")
        code, out = run(capsys, "seg", "stats", path, "organs")
        hint = f"`medh5 seg convert {path} organs --to layers` would re-encode it"
        assert hint in out.out
        assert run(capsys, "seg", "convert", path, "organs", "--to", "layers")[0] == 0

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

    def test_S7_6_a_probmap_is_converted_through_its_threshold(self, capsys, tmp_path):
        """§7.6 makes a probmap lossless only under its `threshold` (§7.5):
        another encoding keeps the voxels at or above it, and the help called
        every conversion lossless."""
        path = tmp_path / "soft.medh5"
        votes = np.zeros(Numbered.SHAPE, np.float32)
        votes[1], votes[2], votes[3] = 0.29, 0.3, 0.9
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "soft", grid="g", probabilities={1: votes}, threshold=0.3
            )
        argv = ["seg", "convert", str(path), "soft", "--to", "layers"]
        assert run(capsys, *argv)[0] == EXIT_OK
        with medh5.open(path) as sample:
            soft = sample.annotations["soft"]
            assert soft.kind == "layers"
            assert np.array_equal(soft.dense([1])[0], votes >= 0.3)
        code, out = run(capsys, "seg", "convert", "--help")
        assert code == EXIT_OK
        assert "a probmap is thresholded at its threshold (§7.5) and loses its " in (
            " ".join(out.out.split())
        )

    def test_index_build(self, capsys, sample_path):
        import medh5

        code, out = run(
            capsys, "index", "build", str(sample_path), "--max-coords", "32"
        )
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            assert "organs_tp0" in sample.index

    def test_S14_3_an_occupancy_block_of_0_builds_no_map(self, capsys, sample_path):
        code, out = run(capsys, "index", "build", str(sample_path), "--occupancy", "0")
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            assert not sample.index["organs_tp0"].has_occupancy

    def test_index_build_on_a_sample_without_annotations(self, capsys, tmp_path):
        import medh5
        from tests.helpers import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "index", "build", str(path))
        assert code == EXIT_ERROR

    def test_usage_errors(self, capsys):
        assert run(capsys, "seg")[0] == EXIT_USAGE
        assert run(capsys, "index")[0] == EXIT_USAGE


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
        assert out.out.startswith("wrote 1 cases and ")
        assert (tmp_path / "core-minimal.medh5").exists()
        assert (tmp_path / "expected.json").exists()
        code, out = run(
            capsys, "conformance", "run", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    @pytest.mark.parametrize("command", ["build", "run", "publish"])
    def test_a_case_name_the_corpus_lacks_is_refused(self, capsys, tmp_path, command):
        """A misspelt `--case` built, ran or published nothing and reported a
        whole corpus written, or "0/0 cases pass"."""
        code, out = run(
            capsys, "conformance", command, str(tmp_path / "out"), "--case", "nosuch"
        )
        assert code == EXIT_USAGE
        assert "no conformance case 'nosuch'" in out.err
        assert not (tmp_path / "out").exists()

    def test_publish_then_score(self, capsys, tmp_path):
        code, out = run(
            capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert ": 1 cases, see " in out.out
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "SHA256SUMS").exists()

        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    def test_score_reports_a_wrong_answer(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": ["E101"]}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR
        assert "unexpected" in out.out and "E101" in out.out

    def test_score_fails_when_the_files_have_drifted(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        case = tmp_path / "core-minimal.medh5"
        case.write_bytes(case.read_bytes() + b"\x00")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert "differ" in out.out and "core-minimal.medh5" in out.out
        # A score over files that are not the published ones is not a score.
        assert code == EXIT_ERROR
        code, out = run(
            capsys, "conformance", "score", str(tmp_path), str(results), "--json"
        )
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        assert payload["changed"] == ["core-minimal.medh5"]
        assert payload["ok"] is False and payload["passed"] == 1
        assert "differ" in out.err

    def test_score_without_a_suite_fails_cleanly(self, capsys, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("[]", encoding="utf-8")
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR

    def test_usage(self, capsys):
        assert run(capsys, "conformance")[0] == EXIT_USAGE


class TestFixAndScrub:
    def test_fix_reports_and_changes_nothing(self, capsys, sample_path):
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_OK
        assert "nothing to fix" in out.out

    def test_fix_refuses_to_restamp_without_a_reason(self, capsys, sample_path):
        before = sample_path.read_bytes()
        code, out = run(capsys, "fix", str(sample_path), "--rewrite-digests")
        assert code == EXIT_USAGE
        assert "--reason" in out.err
        code, out = run(
            capsys, "fix", str(sample_path), "--rewrite-digests", "--reason", ""
        )
        assert code == EXIT_ERROR
        assert "pass a reason" in out.err
        assert sample_path.read_bytes() == before

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

    def test_S13_2_fix_names_what_no_line_of_the_content_id_covers(
        self, capsys, sample_path
    ):
        """An undigested dataset in an attested object needs attention, and
        `fix` said only that, without naming it."""
        with h5py.File(sample_path, "r+") as handle:
            handle["annotations/organs_tp0"].create_dataset(
                "instance_ids", data=np.arange(3, dtype=np.int64)
            )
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_ERROR
        assert "unsigned: annotations/organs_tp0/instance_ids" in out.out

    def test_fix_rebuilds_a_stale_index(self, capsys, longitudinal_path):
        import h5py

        from tests.helpers import encode_attr

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


class TestErrorsNameTheFile:
    """An I/O or parse error names the file it is about, as the rest of the
    command line words one (`[Errno 2] No such file or directory: 'x'`);
    these said only "No such file or directory (os error 2)", or "EOF while
    parsing a value"."""

    def test_a_manifest_that_is_missing_or_not_json(self, capsys, tmp_path):
        missing = tmp_path / "missing.json"
        for command in ("split", "stats", "check"):
            code, out = run(capsys, "dataset", command, str(missing))
            assert code == EXIT_ERROR, command
            assert repr(str(missing)) in out.err, command
        truncated = tmp_path / "m.json"
        truncated.write_text("", encoding="utf-8")
        code, out = run(capsys, "dataset", "check", str(truncated))
        assert code == EXIT_ERROR
        assert f"{str(truncated)!r} is not JSON" in out.err

    def test_an_output_that_cannot_be_written(self, capsys, tmp_path, sample_path):
        manifest = tmp_path / "m.json"
        code, _ = run(
            capsys, "dataset", "index", str(sample_path.parent), "-o", str(manifest)
        )
        assert code == EXIT_OK
        target = tmp_path / "nodir" / "x.json"
        for command in ("split", "stats"):
            code, out = run(
                capsys, "dataset", command, str(manifest), "-o", str(target)
            )
            assert code == EXIT_ERROR, command
            assert repr(str(target)) in out.err, command

    def test_a_report_that_cannot_be_written(self, capsys, tmp_path):
        old = tmp_path / "old.medh5"
        write_legacy_sample(old, images={"CT": np.zeros((4, 4, 4), np.int16)})
        report = tmp_path / "nodir" / "r.json"
        code, out = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--report",
            str(report),
        )
        assert code == EXIT_ERROR
        assert repr(str(report)) in out.err

    def test_a_file_where_a_directory_is_wanted(self, capsys, sample_path):
        for argv in (
            ["unpack", str(sample_path), "-o", str(sample_path)],
            ["conformance", "build", str(sample_path), "--case", "core-minimal"],
        ):
            code, out = run(capsys, *argv)
            assert code == EXIT_ERROR, argv
            assert repr(str(sample_path)) in out.err, argv


def _strings(value):
    """Every string in a JSON document, keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


class TestBatches:
    """A command over several files reports on every one: a file it cannot
    process is an entry of its own, the files before it keep their reports,
    the files after it are still done, and the command fails at the end."""

    @pytest.fixture
    def batch(self, tmp_path, sample_path):
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        after = tmp_path / "after.medh5"
        after.write_bytes(sample_path.read_bytes())
        return [str(sample_path), str(broken), str(after)]

    def test_verify_goes_on_past_a_file_it_cannot_read(self, capsys, batch):
        code, out = run(capsys, "verify", *batch, "--json")
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        assert [entry["ok"] for entry in payload] == [True, False, True]
        assert payload[1]["path"] == batch[1]
        assert payload[1]["error"] in out.err

    @pytest.mark.parametrize(
        "command",
        [
            ["fix"],
            ["recompress", "--profile", "portable"],
            ["scrub"],
            ["index", "build"],
            ["labels", "check"],
        ],
    )
    def test_every_batch_command_goes_on(self, capsys, batch, command):
        code, out = run(capsys, *command, *batch, "--json")
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        # Looked for in the parsed document: JSON escapes the backslashes of
        # a Windows path, so the raw text never holds the path as given.
        named = set(_strings(payload))
        assert batch[1] in named and batch[2] in named, payload
        assert "medh5: " in out.err

    def test_a_refused_scrub_keeps_the_reports_of_files_already_changed(
        self, capsys, tmp_path, sample_path
    ):
        """A clinical sample is refused by design; the files scrubbed before
        it were changed, and their reports carry the only key to the
        pseudonyms written (`uid_map`)."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=["clinical-one-visit-history"])
        clinical = tmp_path / "clinical-one-visit-history.medh5"
        code, out = run(
            capsys,
            "scrub",
            str(sample_path),
            str(clinical),
            "--apply",
            "--salt",
            "s",
            "--json",
        )
        assert code == EXIT_ERROR
        first, refused = json.loads(out.out)
        assert first["path"] == str(sample_path) and first["applied"]
        assert "uid_map" in first
        assert refused["ok"] is False and "clinical profile" in refused["error"]


class TestDatasetCommands:
    """The cohort commands: index, split, stats, check."""

    @pytest.fixture
    def cohort(self, tmp_path, label_set):
        from tests.helpers import SHAPE

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
        assert json.loads(manifest.read_text(encoding="utf-8"))["samples"] == 6

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

    def test_index_of_a_root_that_is_not_there_says_so(self, capsys, tmp_path):
        """It wrote a manifest of nobody, and said nothing of the root."""
        missing = tmp_path / "no-such-cohort"
        manifest = tmp_path / "m.json"
        code, out = run(capsys, "dataset", "index", str(missing), "-o", str(manifest))
        assert code == EXIT_ERROR
        assert repr(str(missing)) in out.err
        assert not manifest.exists()

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
        payload = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
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
        payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
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

    @pytest.mark.parametrize(
        ("option", "what"), [("--image", "an image"), ("--annotation", "an annotation")]
    )
    def test_stats_refuses_an_id_no_sample_has(
        self, capsys, cohort, tmp_path, option, what
    ):
        """A misspelt id measured nothing and exited 0 with an empty table."""
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(capsys, "dataset", "stats", str(manifest), option, "CT")
        assert code == EXIT_ERROR
        assert out.err == f"medh5: no sample has {what} 'CT'\n"
        assert out.out == ""

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
        assert run(capsys, "dataset")[0] == EXIT_USAGE


class TestAnnotationKinds:
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


class TestTransforms:
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


class TestCuration:
    """Shards, provenance, agreement and the split audit (§2.2, §11, §12.3)."""

    def _members(self, tmp_path, label_set, masks, n=2):

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

    @pytest.mark.parametrize(
        "argv",
        [
            ["prov", "{}"],
            ["labels", "show", "{}"],
            ["labels", "check", "{}"],
            ["seg", "stats", "{}", "organs_tp0"],
            ["seg", "convert", "{}", "organs_tp0", "--to", "bitmask"],
            ["agree", "{}", "organs_tp0", "organs_tp0"],
            ["fix", "{}"],
            ["scrub", "{}"],
        ],
    )
    def test_S2_2_a_command_on_one_sample_names_the_way_to_a_member(
        self, capsys, tmp_path, label_set, masks, argv
    ):
        """A collection handed to a command that takes no ``--key`` was refused
        with the Python API's advice, "open it with open_collection()"."""
        shard = tmp_path / "s.medh5c"
        members = self._members(tmp_path, label_set, masks)
        assert run(capsys, "pack", *map(str, members), "-o", str(shard))[0] == EXIT_OK
        code, out = run(capsys, *(a.format(shard) for a in argv))
        assert code == EXIT_ERROR
        assert out.err.startswith(
            f"medh5: {shard} is a collection of 2 sample(s); this command works on one "
            f"sample file --- extract them with `medh5 unpack {shard} -o DIR`\n"
        )

    def test_S2_2_a_command_on_a_collection_names_a_sample_file(
        self, capsys, tmp_path, sample_path
    ):
        """``ls`` and ``unpack`` on a sample file answered with a wrong
        ``medh5_kind`` and "open it with `medh5.open`"."""
        refused = f"medh5: {sample_path} is a sample file, not a collection;"
        code, out = run(capsys, "ls", str(sample_path))
        assert code == EXIT_ERROR
        assert out.err == f"{refused} `medh5 info {sample_path}` describes it\n"
        code, out = run(capsys, "unpack", str(sample_path), "-o", str(tmp_path / "x"))
        assert code == EXIT_ERROR
        assert out.err == f"{refused} there is nothing to unpack\n"

    def test_pack_json_reports_sizes(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks, n=1)
        shard = tmp_path / "one.medh5c"
        code, out = run(capsys, "pack", str(members[0]), "-o", str(shard), "--json")
        payload = json.loads(out.out)
        assert payload["samples"] == 1
        assert payload["bytes"] > 0

    def test_S2_2_pack_never_writes_over_a_sample_it_packs(
        self, capsys, tmp_path, label_set, masks
    ):
        """`pack a.medh5 -o a.medh5` replaced the sample with a collection
        holding it, and exited 0."""
        first, second = self._members(tmp_path, label_set, masks)
        before = second.read_bytes()
        alias = tmp_path / "." / second.name
        code, out = run(capsys, "pack", str(first), str(second), "-o", str(alias))
        assert code == EXIT_ERROR
        assert "is one of the samples being packed" in out.err
        assert second.read_bytes() == before
        code, out = run(capsys, "pack", str(first), "-o", str(tmp_path))
        assert code == EXIT_ERROR
        assert repr(str(tmp_path)) in out.err

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
        from tests.helpers import SHAPE

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
        from tests.helpers import SHAPE

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

        path = write_raters(tmp_path / "raters.medh5", label_set)
        code, out = run(capsys, "agree", str(path), "r1", "r2")
        assert code == EXIT_OK
        assert "matched" in out.out and "instance_id" in out.out

    def test_agree_refuses_two_visits_grids(self, capsys, tmp_path, label_set):

        path = write_series(tmp_path / "series.medh5", label_set)
        code, out = run(capsys, "agree", str(path), "les_tp0", "les_tp1")
        assert code == EXIT_ERROR
        assert "different grids" in out.err

    def test_splits_audit_reports_leakage(self, capsys, tmp_path, label_set, masks):
        import medh5

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


class TestRecompressAndBench:
    """``recompress`` and ``bench`` (§14.2)."""

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
        assert code == EXIT_USAGE
        assert "single input" in out.err

    def test_recompress_rechunks_only_when_asked(self, capsys, tmp_path):
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

    def test_S14_2_recompress_names_what_it_copied_as_stored(self, capsys, tmp_path):
        """A dataset whose stored type a rebuild would lose --- an enumeration
        and its names --- is copied as it is, and the text output says so on a
        ``KEPT`` line, as ``--json`` lists it under ``kept``."""
        path = write_sample(tmp_path / "typed.medh5", codec="portable")
        colours = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="u1")
        with h5py.File(path, "r+") as f:
            f.create_dataset(
                "x_types/colours", data=np.zeros((48, 64, 64), "u1"), dtype=colours
            )
        code, out = run(capsys, "recompress", str(path), "--profile", "archive")
        assert code == EXIT_OK
        kept = "  KEPT      /x_types/colours (copied as stored: a rebuild would lose"
        assert f"{kept} its type)" in out.out.splitlines()
        code, out = run(
            capsys, "recompress", str(path), "--profile", "portable", "--json"
        )
        assert code == EXIT_OK
        assert json.loads(out.out)[0]["kept"] == ["/x_types/colours"]

    def test_S14_4_recompress_names_what_the_output_check_could_not_read(
        self, capsys, tmp_path
    ):
        """The output has replaced the input when it is checked, so what the
        check cannot read --- here a chunk whose deflate stream is damaged ---
        is reported, not raised.  The text output said ``FAILED (0)`` and
        nothing more; it names it on an ``UNCHECKED`` line, as ``--json`` lists
        it under ``unverified``."""

        def damaged(name: str) -> Path:
            path = write_sample(tmp_path / name, codec="portable")
            pair = np.dtype([("a", "<i4"), ("b", "<f4")])
            with h5py.File(path, "r+") as f:
                table = f.create_dataset(
                    "x_acme/table",
                    shape=(4,),
                    dtype=pair,
                    chunks=(4,),
                    compression="gzip",
                )
                table.id.write_direct_chunk((0,), b"\xa5" * 16)
                table.attrs["digest"] = "sha256:" + "0" * 64
            return path

        path = damaged("text.medh5")
        code, out = run(capsys, "recompress", str(path), "--profile", "archive")
        assert code == EXIT_ERROR
        assert "FAILED (1)" in out.out
        (unchecked,) = [ln for ln in out.out.splitlines() if "UNCHECKED" in ln]
        assert unchecked.startswith("  UNCHECKED /: ")
        assert unchecked.endswith("/x_acme/table")
        path = damaged("json.medh5")
        code, out = run(
            capsys, "recompress", str(path), "--profile", "archive", "--json"
        )
        assert code == EXIT_ERROR
        assert json.loads(out.out)[0]["unverified"] == [unchecked.split(maxsplit=1)[1]]

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
        rows = [line for line in out.out.splitlines() if "(target ≤" in line]
        assert len(rows) >= 4, "each targeted row prints its target"
        missed = [row.split()[1] for row in rows if row.startswith("!")]
        verdict = out.out.rstrip().splitlines()[-1]
        if missed:
            assert verdict.endswith(f"below target: {', '.join(missed)}")
            assert code == EXIT_ERROR
        else:
            assert verdict == "all targets met"
            assert code == EXIT_OK


class TestConvert:
    """``convert`` and ``migrate``."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["from-dicom-seg", "seg.dcm", "case.medh5"],
            ["to-dicom-seg", "case.medh5", "seg", "seg.dcm", "--source", "1.dcm"],
            ["from-rtstruct", "rt.dcm", "case.medh5"],
            ["to-rtstruct", "case.medh5", "contours", "rt.dcm", "--source", "1.dcm"],
        ],
    )
    def test_S3_4_every_dicom_frame_check_takes_the_scrub_salt(self, monkeypatch, argv):
        """``from-rtstruct`` and ``to-dicom-seg`` compare frames of reference
        as their converters do, but had no ``--frame-salt`` to compare one a
        scrub pseudonymised."""
        from medh5 import cli

        handed: dict[str, Any] = {}
        monkeypatch.setitem(
            cli._HANDLERS, argv[0], lambda a: handed.update(vars(a)) or EXIT_OK
        )
        assert main(["convert", *argv, "--frame-salt", "s3cret"]) == EXIT_OK
        assert handed["frame_salt"] == "s3cret"

    @pytest.mark.parametrize(
        "argv",
        [
            ["to-nifti", "{shard}", "CT_tp0", "{tmp}/x.nii.gz"],
            ["to-dicom-seg", "{shard}", "organs_tp0", "{tmp}/x.dcm", "--source", "1"],
            ["to-rtstruct", "{shard}", "organs_tp0", "{tmp}/x.dcm", "--source", "1"],
            ["from-dicom-seg", "{tmp}/seg.dcm", "{shard}"],
            ["from-rtstruct", "{tmp}/rt.dcm", "{shard}"],
            ["to-nnunet", "{tmp}/nn", "{tmp}/m0.medh5", "{shard}"],
        ],
    )
    def test_S2_2_a_converter_handed_a_collection_names_the_way_to_a_member(
        self, capsys, tmp_path, argv
    ):
        """The engine refuses a collection in the Python API's words, and a
        converter on one told the command line to "open it with
        open_collection()"; it names `medh5 unpack`, before reading anything."""
        member = write_sample(tmp_path / "m0.medh5")
        shard = tmp_path / "s.medh5c"
        assert run(capsys, "pack", str(member), "-o", str(shard))[0] == EXIT_OK
        argv = [a.format(shard=shard, tmp=tmp_path) for a in argv]
        code, out = run(capsys, "convert", *argv)
        assert code == EXIT_ERROR
        assert out.err == (
            f"medh5: {shard} is a collection of 1 sample(s); this command works "
            f"on sample files --- extract them with `medh5 unpack {shard} -o DIR`\n"
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["m0.medh5", "s.medh5c"]

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

    def test_S3_6_the_fourth_axis_can_be_settled_from_the_CLI(self, capsys, tmp_path):
        """`auto` guesses `time` for an unmarked series and says to override it.

        The override existed only in Python, so the one case the argument was
        added for could not be corrected by the interface people convert with.
        """
        nib = pytest.importorskip("nibabel")
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        source = tmp_path / "echo.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 4, 3), np.int16), affine), str(source))
        out = tmp_path / "echo.medh5"

        code, _ = run(
            capsys,
            "convert",
            "from-nifti",
            str(out),
            "--image",
            f"IM={source}",
            "--fourth-axis",
            "channel",
        )
        assert code == EXIT_OK
        with medh5.open(out) as sample:
            assert sample.grids["ref"].axis_kinds[0] == "channel"

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
        assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

    def test_a_malformed_pair_is_reported(self, capsys, tmp_path):
        code, printed = run(
            capsys, "convert", "from-nifti", str(tmp_path / "x.medh5"), "--image", "CT"
        )
        assert code == EXIT_ERROR
        assert "NAME=VALUE" in printed.err

    def test_convert_without_a_subcommand_explains_itself(self, capsys):
        code, printed = run(capsys, "convert")
        assert code == EXIT_USAGE
        assert "Usage: medh5 convert" in printed.err
        assert "from-nifti" in printed.err

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
            ),
            encoding="utf-8",
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
        assert json.loads(labels.read_text(encoding="utf-8"))["classes"]
        # With `--json` the note goes to stderr and stdout is the report alone.
        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--write-labels",
            str(labels),
            "--json",
        )
        assert code == EXIT_OK
        assert json.loads(printed.out)["converter"] == "migrate"
        assert "review before migrating" in printed.err

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
