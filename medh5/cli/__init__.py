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
    """Run ``medh5`` on *argv* (``sys.argv[1:]`` by default); the exit code.

    ``--help`` and ``--version`` return 0 and a usage error 2, like every other
    outcome: nothing raises ``SystemExit`` but the console script itself.
    """
    args = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    return int(_core.cli_main(args, _Host()))


def _what(exc: LookupError) -> str:
    """The message a lookup failed with, or what was looked up.

    ``str(KeyError('x'))`` is ``"'x'"``: the key alone, in quotes.  Most of
    this package's lookups raise with a sentence that names what is available,
    which is printed as it is; a bare key is named as one.  The rule is the
    native CLI's, so a converter's error reads as an engine error does.
    """
    detail = exc.args[0] if len(exc.args) == 1 else exc
    return str(_core.cli_lookup_message(str(detail)))


def command_tree() -> dict[str, Any]:
    """The grammar as data: ``{"options", "positionals", "commands"}``,
    recursively --- what documentation is checked against."""
    found: dict[str, Any] = _core.cli_command_tree()
    return found


__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "command_tree", "main"]
