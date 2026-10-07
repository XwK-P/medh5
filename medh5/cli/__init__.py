"""The ``medh5`` command line.

One native application over the format engine (the ``medh5-cli`` crate): the
same grammar, output and exit codes --- 0 success, 1 a handled error, 2 a usage
error --- whether it runs as the standalone ``medh5`` binary or as this
package's console script.  The commands only Python can run are handed back to
the package: the format converters (NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net),
which wrap nibabel, pydicom and highdicom, and the PyTorch dataloader
benchmark.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from medh5 import _core
from medh5.cli._common import EXIT_ERROR, EXIT_OK, EXIT_USAGE


class _Host:
    """What only the Python package can run, on behalf of the native CLI."""

    def convert(self, argv: list[str], command: str, args: dict[str, Any]) -> int:
        from medh5.cli.convert import run

        return run(command, args)

    def throughput(
        self, path: str, patch: int, workers: int, annotation: str | None
    ) -> dict[str, Any]:
        from medh5.cli.perf import throughput

        return throughput(path, patch, workers, annotation)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``medh5`` on *argv* (``sys.argv[1:]`` by default).

    Returns the exit code.  ``--help``, ``--version`` and usage errors raise
    ``SystemExit`` instead, as an ``argparse`` command line does.
    """
    args = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    code, parser_exit = _core.cli_main(args, _Host())
    if parser_exit:
        raise SystemExit(code)
    return int(code)


def command_tree() -> dict[str, Any]:
    """The grammar as data: ``{"options", "positionals", "commands"}``,
    recursively --- what documentation is checked against."""
    found: dict[str, Any] = _core.cli_command_tree()
    return found


__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "command_tree", "main"]
