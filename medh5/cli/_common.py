"""Shared helpers for the ``medh5`` CLI command modules."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

Handler = Callable[[argparse.Namespace], int]
ValidationPayload = dict[str, Any]


def iter_medh5(root: Path) -> Iterable[Path]:
    """Yield every ``*.medh5`` under *root* in sorted order."""
    yield from sorted(root.rglob("*.medh5"))
