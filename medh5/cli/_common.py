"""Shared CLI helpers: exit codes, JSON output, and error presentation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def emit(payload: Any, *, as_json: bool) -> None:
    """Print a JSON document, or nothing when the caller wants text output."""
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


def fail(message: str) -> int:
    print(f"medh5: {message}", file=sys.stderr)
    return EXIT_ERROR


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="machine-readable output on stdout"
    )


def add_paths(
    parser: argparse.ArgumentParser, help_text: str = "one or more files"
) -> None:
    parser.add_argument("paths", nargs="+", metavar="PATH", help=help_text)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GiB"  # pragma: no cover - unreachable


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    """A minimal fixed-width table --- no dependency, predictable in a pipe."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in cells))
        if cells
        else len(str(headers[i]))
        for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True))
    out = [line, "  ".join("-" * w for w in widths)]
    out.extend(
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells
    )
    return "\n".join(out)


__all__ = [
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_USAGE",
    "add_json_flag",
    "add_paths",
    "emit",
    "fail",
    "human_bytes",
    "table",
]
