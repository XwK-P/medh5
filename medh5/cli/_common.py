"""Shared CLI helpers: exit codes, JSON output, and the text the CLI prints.

The command line itself is native (``medh5-cli``); these serve the commands
the package runs on its behalf (the format converters) and keep the 1.x
helpers importable.  Sizes and tables are formatted by the native CLI's own
functions, so both halves print alike.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from medh5 import _core

EXIT_OK: int = _core.EXIT_OK
EXIT_ERROR: int = _core.EXIT_ERROR
EXIT_USAGE: int = _core.EXIT_USAGE


def emit(payload: Any, *, as_json: bool) -> None:
    """Print a JSON document, or nothing when the caller wants text output."""
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


def fail(message: str) -> int:
    """Report a handled error on stderr: ``medh5: <message>``, exit code 1."""
    print(f"medh5: {message}", file=sys.stderr)
    return EXIT_ERROR


def human_bytes(n: float) -> str:
    """``512 B``, ``2.0 KiB``, ... as the CLI prints sizes."""
    return str(_core.cli_human_bytes(float(n)))


def indent(text: str, prefix: str = "  ") -> str:
    """Indent every line of a block, for nesting a table under a heading."""
    return "\n".join(prefix + line for line in text.splitlines())


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    """A minimal fixed-width table --- no dependency, predictable in a pipe."""
    return str(_core.cli_table([list(row) for row in rows], [str(h) for h in headers]))


__all__ = [
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_USAGE",
    "emit",
    "fail",
    "human_bytes",
    "indent",
    "table",
]
