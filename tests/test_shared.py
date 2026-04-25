"""Tests for medh5.open_shared ref-counted read handles."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5FileError, MEDH5ValidationError, open_shared
from medh5._shared import _registry


def _make_file(path):
    MEDH5File.write(
        path,
        images={"CT": np.zeros((2, 4, 4), dtype=np.float32)},
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    yield
    # Belt-and-braces: make sure one leaky test doesn't pollute the others.
    for entry in list(_registry.values()):
        entry.file.close()
    _registry.clear()


class TestOpenShared:
    def test_two_callers_share_one_handle(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        with open_shared(p) as f1, open_shared(p) as f2:
            assert f1 is f2
            assert f1.id.valid

    def test_closes_when_last_reference_released(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        with open_shared(p) as f:
            saved = f
            assert saved.id.valid
        assert not saved.id.valid
        assert p.resolve() not in _registry

    def test_inner_exit_keeps_outer_open(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        with open_shared(p) as outer:
            with open_shared(p) as inner:
                assert inner is outer
            # Outer still alive.
            assert outer.id.valid
            data = outer["images/CT"][0, 0, 0]
            assert float(data) == 0.0

    def test_exception_in_body_releases_ref(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        with pytest.raises(RuntimeError, match="boom"), open_shared(p):
            raise RuntimeError("boom")
        assert p.resolve() not in _registry

    def test_cross_thread_sharing(self, tmp_path):
        p = tmp_path / "s.medh5"
        _make_file(p)
        seen: list[int] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            with open_shared(p) as f:
                barrier.wait()
                seen.append(id(f))

        with open_shared(p) as main_f:
            t = threading.Thread(target=worker)
            t.start()
            barrier.wait()
            t.join()
            assert seen == [id(main_f)]

    def test_bad_suffix_rejected(self, tmp_path):
        bad = tmp_path / "nope.txt"
        bad.touch()
        with pytest.raises(MEDH5ValidationError, match="extension"), open_shared(bad):
            pass

    def test_missing_file_raises_medh5_error(self, tmp_path):
        p = tmp_path / "missing.medh5"
        with pytest.raises(MEDH5FileError, match="Failed to open"), open_shared(p):
            pass
