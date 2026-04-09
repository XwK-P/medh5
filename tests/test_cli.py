"""Tests for the CLI."""

import numpy as np
import pytest

from medh5 import MEDH5File
from medh5.cli import main


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "cli_test.medh5"
    images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
    MEDH5File.write(
        path,
        images=images,
        label=1,
        label_name="positive",
        spacing=[1.0, 1.0, 2.0],
        extra={"modality": "CT"},
        seg={"tumor": np.zeros((8, 8, 8), dtype=bool)},
    )
    return path


class TestCLI:
    def test_info(self, sample_file, capsys):
        ret = main(["info", str(sample_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "CT" in captured.out
        assert "float32" in captured.out
        assert "Spacing" in captured.out

    def test_validate_valid(self, sample_file, capsys):
        ret = main(["validate", str(sample_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "VALID" in captured.out

    def test_validate_invalid(self, tmp_path, capsys):
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        ret = main(["validate", str(bad)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "INVALID" in captured.err or len(captured.err) > 0

    def test_no_command(self, capsys):
        ret = main([])
        assert ret == 0

    def test_info_missing_file(self, tmp_path, capsys):
        ret = main(["info", str(tmp_path / "missing.medh5")])
        assert ret == 1
