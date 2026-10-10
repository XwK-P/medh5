"""``medh5 bench``: the measurements, and that each measures what it names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import medh5
from medh5.validate import validate_file


class TestBench:
    def test_the_metrics_run_and_are_reported(self, indexed_cohort):
        from medh5.bench import TARGETS, benchmark_file, report

        measurements = benchmark_file(indexed_cohort[0], patch=8, repeats=2)
        names = {m.name for m in measurements}
        # The many-class row is measured on the 63-class sample `medh5 bench`
        # writes for it, not on a file passed in (test_release_1_4_2, T-11).
        assert set(TARGETS) - {"foreground_sample_many_ms"} <= names
        assert all(m.value >= 0 for m in measurements)
        assert "target" in report(measurements) or "all targets met" in report(
            measurements
        )

    def test_a_measurement_knows_whether_it_met_its_target(self):
        from medh5.bench import Measurement

        assert Measurement("x", 1.0, target=2.0).ok
        assert not Measurement("x", 3.0, target=2.0).ok
        assert Measurement("x", 3.0).ok
        assert "!" in str(Measurement("x", 3.0, target=2.0))
        assert Measurement("x", 1.0).to_json()["ok"] is True

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

    def test_Q19_bench_json_is_only_the_document(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Progress lines on stdout ahead of the JSON made `--json` unparsable."""
        import medh5.bench as bench
        from medh5.cli import main

        small = bench.synthetic_sample
        pair = bench.synthetic_pair
        monkeypatch.setattr(
            bench,
            "synthetic_sample",
            lambda d, **kw: small(d, shape=(16, 32, 32), classes=3, **kw),
        )
        monkeypatch.setattr(
            bench, "synthetic_pair", lambda d, **kw: pair(d, shape=(8, 16, 16))
        )
        monkeypatch.setattr(
            bench,
            "synthetic_many_class_sample",
            lambda d: small(d, shape=(8, 16, 16), classes=5, name="many.medh5"),
        )
        code = main(
            ["bench", "--no-throughput", "--json", "--repeats", "1", "--patch", "8"]
        )
        out, err = capsys.readouterr()
        payload = json.loads(out)
        names = {m["name"] for m in payload["measurements"]}
        assert "foreground_sample_many_ms" in names
        assert "writing a synthetic" in err
        assert code in (0, 1)


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
