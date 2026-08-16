"""The ``medh5`` command line.

Each submodule exposes ``register(sub)`` and ``dispatch(command, args)``, and
this module composes them.  Exit codes are Unix-conventional: 0 success,
1 a handled error, 2 a usage error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from medh5.__about__ import __format_version__, __version__
from medh5.cli import conformance, curation, inspect, labels, perf, seg
from medh5.cli._common import EXIT_ERROR, EXIT_USAGE
from medh5.errors import MEDH5Error

MODULES = (inspect, seg, labels, curation, perf, conformance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medh5",
        description=(
            f"medh5 {__version__} --- tools for the MEDH5 {__format_version__} "
            "medical imaging container"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"medh5 {__version__} (format {__format_version__})",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for module in MODULES:
        module.register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    try:
        for module in MODULES:
            result: int | None = module.dispatch(args.command, args)
            if result is not None:
                return result
    except MEDH5Error as exc:
        print(f"medh5: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # pragma: no cover - `medh5 info | head`
        return 0
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
