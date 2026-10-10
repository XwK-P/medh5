"""``medh5 bench``: the measurements, and that each measures what it names."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

import medh5
from medh5.validate import validate_file
from tests.helpers import encode_attr


class TestBench:
    def test_the_metrics_run_and_are_reported(self, indexed_cohort):
        from medh5.bench import TARGETS, benchmark_file, report

        measurements = benchmark_file(indexed_cohort[0], patch=8, repeats=2)
        names = {m.name for m in measurements}
        # The many-class row is measured on the 63-class sample `medh5 bench`
        # writes for it, not on a file passed in.
        assert set(TARGETS) - {"foreground_sample_many_ms"} <= names
        assert all(m.value >= 0 for m in measurements)
        missed = [m.name for m in measurements if not m.ok]
        verdict = report(measurements).splitlines()[-1]
        assert verdict == (
            f"{len(missed)} metric(s) below target: {', '.join(missed)}"
            if missed
            else "all targets met"
        )

    def test_a_measurement_knows_whether_it_met_its_target(self):
        from medh5.bench import Measurement, report

        assert Measurement("x", 1.0, target=2.0).ok
        assert not Measurement("x", 3.0, target=2.0).ok
        assert Measurement("x", 3.0).ok
        assert "!" in str(Measurement("x", 3.0, target=2.0))
        assert Measurement("x", 1.0).to_json()["ok"] is True
        met = [Measurement("a", 1.0, target=2.0), Measurement("b", 9.0)]
        assert report(met).splitlines()[-1] == "all targets met"
        missed = [*met, Measurement("c", 3.0, target=2.0)]
        assert report(missed).splitlines()[-1] == "1 metric(s) below target: c"

    def test_S14_3_the_draw_row_says_what_the_sampler_used(self, indexed_cohort):
        """``used_index`` is the sampler's own answer.  The row reported an
        index entry's existence, so a stale entry, which the sampler scans
        past, was reported as used."""
        from medh5.bench import benchmark_file

        def used(path: Path) -> object:
            rows = {m.name: m for m in benchmark_file(path, patch=8, repeats=1)}
            detail = rows["foreground_sample_ms"].detail or {}
            return detail["used_index"]

        path = indexed_cohort[0]
        assert used(path) is True
        with h5py.File(path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        assert used(path) is False

    def test_timed_returns_a_median(self):
        from medh5.bench import timed

        assert timed(lambda: None, repeats=3, warmup=1) >= 0.0

    def test_the_throughput_run_measures_steady_state(self, indexed_cohort):
        pytest.importorskip("torch")  # the run goes through the real DataLoader
        from medh5.bench import throughput

        measured = throughput(
            indexed_cohort, patch=8, batches=2, batch_size=1, workers=0
        )
        assert measured.unit == "patches/s"
        assert measured.value > 0
        assert measured.detail["workers"] == 0

    def test_bench_measures_a_paired_centre(self, tmp_path: Path):
        from medh5.bench import benchmark_file, synthetic_pair

        pair = synthetic_pair(tmp_path, shape=(6, 8, 8))
        names = [m.name for m in benchmark_file(pair, patch=4, repeats=2)]
        assert "paired_center_ms" in names

    def test_T11_the_bench_has_a_many_class_row(self, tmp_path: Path):
        from medh5.bench import (
            MANY_CLASSES,
            TARGETS,
            many_class_measurement,
            synthetic_sample,
        )

        path = synthetic_sample(
            tmp_path, shape=(16, 32, 32), classes=MANY_CLASSES, name="many.medh5"
        )
        measured = many_class_measurement(path, patch=8, repeats=3)
        assert measured.name == "foreground_sample_many_ms"
        assert measured.target == TARGETS["foreground_sample_many_ms"][0]
        assert measured.detail == {"classes": MANY_CLASSES, "used_index": True}

    def test_Q19_bench_json_is_only_the_document(self, capsys):
        """Progress lines on stdout ahead of the JSON made `--json` unparsable.

        The engine writes the synthetic samples at their published size, so
        this run is the slow one."""
        from medh5.cli import main

        code = main(
            ["bench", "--no-throughput", "--json", "--repeats", "1", "--patch", "8"]
        )
        out, err = capsys.readouterr()
        payload = json.loads(out)
        names = {m["name"] for m in payload["measurements"]}
        assert "foreground_sample_many_ms" in names
        assert "writing a synthetic" in err
        assert code == (0 if payload["ok"] else 1)

    def test_bench_measures_throughput_through_the_dataloader(
        self, indexed_cohort, capsys
    ):
        """The command line hands the throughput run to the package it was
        started from, which runs the real DataLoader: the row is measured,
        not skipped."""
        pytest.importorskip("torch")
        from medh5.cli import main

        code = main(
            [
                "bench",
                str(indexed_cohort[0]),
                "--json",
                "--repeats",
                "1",
                "--patch",
                "8",
            ]
        )
        out, err = capsys.readouterr()
        assert "skipping throughput" not in err
        payload = json.loads(out)
        rows = {m["name"]: m for m in payload["measurements"]}
        assert rows["patch_throughput"]["unit"] == "patches/s"
        assert rows["patch_throughput"]["value"] > 0
        assert rows["patch_throughput"]["target"] is None
        assert code == (0 if payload["ok"] else 1)


class TestSyntheticBench:
    def test_the_synthetic_sample_is_valid_and_indexed(self, tmp_path):
        from medh5.bench import synthetic_sample

        path = synthetic_sample(
            tmp_path, shape=(8, 16, 16), classes=2, codec="portable"
        )
        assert not validate_file(path).errors
        with medh5.open(path) as sample:
            assert "training" in sample.profiles
            assert len(sample.annotations["organs"].class_ids) == 2
