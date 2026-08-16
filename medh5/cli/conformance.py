"""``conformance build`` and ``conformance run`` --- the corpus tools."""

from __future__ import annotations

import argparse

from medh5.cli._common import EXIT_ERROR, EXIT_OK, add_json_flag, emit, table
from medh5.conformance import CASES, build_corpus, run_corpus


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    conf = sub.add_parser("conformance", help="the conformance corpus (spec §15)")
    group = conf.add_subparsers(dest="conformance_command", metavar="COMMAND")

    listing = group.add_parser("list", help="list corpus cases")
    add_json_flag(listing)

    build = group.add_parser("build", help="write the corpus and its manifest")
    build.add_argument("outdir")
    build.add_argument("--case", action="append", dest="names")

    run = group.add_parser("run", help="build the corpus and check this validator")
    run.add_argument("outdir")
    run.add_argument("--case", action="append", dest="names")
    add_json_flag(run)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command != "conformance":
        return None
    sub = getattr(args, "conformance_command", None)
    if sub == "list":
        if args.json:
            emit([c.to_json() for c in CASES], as_json=True)
            return EXIT_OK
        print(
            table(
                [
                    [
                        c.name,
                        c.clause,
                        c.level,
                        "valid" if c.valid else "invalid",
                        ",".join(sorted((*c.errors, *c.warnings))) or "-",
                    ]
                    for c in CASES
                ],
                ["case", "clause", "level", "kind", "expected codes"],
            )
        )
        return EXIT_OK
    if sub == "build":
        manifest = build_corpus(args.outdir, names=args.names)
        print(f"wrote {len(CASES)} cases and {manifest}")
        return EXIT_OK
    if sub == "run":
        results = run_corpus(args.outdir, names=args.names)
        failures = [r for r in results if not r.ok]
        if args.json:
            emit([r.to_json() for r in results], as_json=True)
        else:
            for result in failures:
                print(f"FAIL {result.case.name}")
                if result.error:
                    print(f"     crash: {result.error}")
                if result.missing:
                    print(f"     not reported: {', '.join(result.missing)}")
                if result.unexpected:
                    print(f"     unexpected:   {', '.join(result.unexpected)}")
            print(f"{len(results) - len(failures)}/{len(results)} cases pass")
        return EXIT_OK if not failures else EXIT_ERROR
    print("usage: medh5 conformance {list|build|run}")
    return EXIT_ERROR


__all__ = ["dispatch", "register"]
