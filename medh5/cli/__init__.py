"""Command-line interface for medh5.

Subcommands are grouped across submodules:

- ``medh5.cli.inspect`` — ``info`` / ``validate`` / ``validate-all`` /
  ``audit`` / ``recompress``
- ``medh5.cli.dataset`` — ``index`` / ``split`` / ``stats``
- ``medh5.cli.convert`` — ``import`` and ``export`` subgroups
  (NIfTI, DICOM, nnU-Net v2)
- ``medh5.cli.review`` — ``review set`` / ``get`` / ``list`` / ``import-seg``

Each submodule exposes a ``register(sub)`` that adds its argparse parsers
and a ``dispatch(cmd, args)`` that returns an exit code when it owns the
command, or ``None`` otherwise.  The top-level :func:`main` composes them.
"""

from __future__ import annotations

import argparse
import sys

from medh5.cli import convert, dataset, inspect, review
from medh5.exceptions import MEDH5Error

__all__ = ["main"]

_MODULES = (inspect, dataset, convert, review)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medh5", description="medh5 file utility")
    sub = parser.add_subparsers(dest="command")
    for module in _MODULES:
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``medh5`` CLI.

    Exit codes follow the usual Unix convention so shell automation
    (``medh5 validate … || exit 1``) works:

    - ``0`` — command ran successfully.
    - ``1`` — a handler raised a known runtime error.
    - ``2`` — no command given, or an unknown subcommand.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        for module in _MODULES:
            rc = module.dispatch(cmd, args)
            if rc is not None:
                return rc
    except (ImportError, MEDH5Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2
