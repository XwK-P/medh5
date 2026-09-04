"""``conformance`` --- build the corpus, run it here, publish it, score others."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medh5.cli._common import EXIT_ERROR, EXIT_OK, add_json_flag, emit, fail, table
from medh5.conformance import (
    CASES,
    build_corpus,
    check_checksums,
    publish,
    run_corpus,
    score,
    summarize,
)
from medh5.conformance.corpus import CaseResult
from medh5.errors import MEDH5Error


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    conf = sub.add_parser("conformance", help="the conformance corpus (spec §15)")
    group = conf.add_subparsers(dest="conformance_command", metavar="COMMAND")

    listing = group.add_parser("list", help="list corpus cases")
    add_json_flag(listing)

    build = group.add_parser("build", help="write the corpus and its manifest")
    build.add_argument("outdir", help="directory to write the corpus into")
    build.add_argument(
        "--case",
        action="append",
        dest="names",
        help="only build these cases; repeatable",
    )

    run = group.add_parser("run", help="build the corpus and check this validator")
    run.add_argument("outdir", help="directory to build the corpus in")
    run.add_argument(
        "--case", action="append", dest="names", help="only run these cases; repeatable"
    )
    add_json_flag(run)

    release = group.add_parser(
        "publish", help="write the distributable suite: cases, codes, schema, checksums"
    )
    release.add_argument(
        "outdir", help="directory to write the distributable suite into"
    )
    release.add_argument(
        "--case",
        action="append",
        dest="names",
        help="only publish these cases; repeatable",
    )

    check = group.add_parser(
        "score", help="score any implementation's results against a published suite"
    )
    check.add_argument("suite", help="a directory written by `conformance publish`")
    check.add_argument("results", help="JSON: [{file, errors, warnings}, ...]")
    add_json_flag(check)


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
        return _report(run_corpus(args.outdir, names=args.names), args)
    if sub == "publish":
        root = publish(args.outdir, names=args.names)
        print(f"wrote the suite to {root}: {len(CASES)} cases, see {root}/README.md")
        return EXIT_OK
    if sub == "score":
        try:
            submitted = json.loads(Path(args.results).read_text(encoding="utf-8"))
            stale = check_checksums(args.suite)
            results = score(args.suite, submitted)
        except (MEDH5Error, OSError, json.JSONDecodeError) as exc:
            return fail(str(exc))
        if stale:
            # The scores mean nothing if the files scored are not the files
            # published, so say so before reporting any of them.
            print(f"WARNING: {len(stale)} published file(s) differ from {args.suite}")
            for name in stale:
                print(f"  changed: {name}")
        return _report(results, args)
    return fail("usage: medh5 conformance {list|build|run|publish|score}")


def _report(results: list[CaseResult], args: argparse.Namespace) -> int:
    failures = [r for r in results if not r.ok]
    if args.json:
        emit(summarize(results), as_json=True)
    else:
        for result in failures:
            print(f"FAIL {result.case.name}")
            if result.error:
                print(f"     {result.error}")
            if result.missing:
                print(f"     not reported: {', '.join(result.missing)}")
            if result.unexpected:
                print(f"     unexpected:   {', '.join(result.unexpected)}")
        print(f"{len(results) - len(failures)}/{len(results)} cases pass")
    return EXIT_OK if not failures else EXIT_ERROR


__all__ = ["dispatch", "register"]
