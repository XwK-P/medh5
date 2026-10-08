"""Fixtures for the whole suite; the builders behind them are in ``helpers``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medh5.labels import LabelClass, LabelSet
from tests.helpers import SEED, SHAPE, block, write_sample, write_series


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet(
        "test-v1",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver", category="organ"),
            LabelClass(2, "spleen", "Spleen", category="organ"),
            LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
            LabelClass(4, "vessel", "Vessel", category="vessel"),
        ],
    )


@pytest.fixture
def masks() -> dict[int, Any]:
    """Three classes where 1 and 3 overlap and 2 does not touch either."""
    return {
        1: block(SHAPE, (2, 2, 2), 8),
        2: block(SHAPE, (2, 14, 2), 6),
        3: block(SHAPE, (4, 4, 4), 3),
    }


@pytest.fixture
def ct() -> Any:
    rng = np.random.default_rng(SEED)
    return rng.integers(-1000, 1500, SHAPE).astype(np.int16)


@pytest.fixture
def sample_path(tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]) -> Path:
    return write_sample(tmp_path / "case.medh5", label_set=label_set, masks=masks)


@pytest.fixture
def longitudinal_path(
    tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]
) -> Path:
    return write_sample(
        tmp_path / "long.medh5",
        label_set=label_set,
        masks=masks,
        timepoints=("tp0", "tp1"),
        index=True,
    )


@pytest.fixture
def indexed_cohort(tmp_path: Path, label_set, masks) -> list[Path]:
    return [
        write_sample(
            tmp_path / f"case{i}.medh5",
            label_set=label_set,
            masks=masks,
            sample_id=f"case{i}",
            index=True,
        )
        for i in range(3)
    ]


@pytest.fixture
def series(tmp_path: Path, label_set: LabelSet) -> Path:
    """Two visits of one subject, lesions tracked across them."""
    return write_series(tmp_path / "series.medh5", label_set)
