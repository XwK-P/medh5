"""``dataset`` --- the cohort commands: index, split, stats, check (plan §5).

Every one of these reads metadata and nothing else unless told otherwise, which
is what makes them usable on a cohort rather than on a demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medh5.cli._common import (
    EXIT_ERROR,
    EXIT_OK,
    add_json_flag,
    emit,
    fail,
    table,
)
from medh5.dataset.check import check
from medh5.dataset.manifest import GROUPABLE, Manifest, scan
from medh5.dataset.split import DEFAULT_RATIOS, Split, make_splits, write_claims
from medh5.dataset.stats import compute_stats
from medh5.errors import MEDH5Error


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("dataset", help="cohort manifests, splits and statistics")
    group = parser.add_subparsers(dest="dataset_command", metavar="COMMAND")

    index = group.add_parser("index", help="metadata-only scan of a directory tree")
    index.add_argument("root")
    index.add_argument("-o", "--out", required=True, help="manifest JSON to write")
    index.add_argument(
        "--strict", action="store_true", help="stop at the first unreadable file"
    )
    add_json_flag(index)

    split = group.add_parser("split", help="assign groups to partitions or folds")
    split.add_argument("manifest")
    split.add_argument("-o", "--out", help="split JSON to write")
    split.add_argument("--set-id", default="default")
    split.add_argument(
        "--group-by",
        default="group_id",
        help=f"never a file; one of {', '.join(GROUPABLE)}",
    )
    split.add_argument("--stratify-by")
    split.add_argument("--k-folds", type=int)
    split.add_argument(
        "--ratios",
        help=f"e.g. train=0.8,val=0.2 (default {_fmt_ratios(DEFAULT_RATIOS)})",
    )
    split.add_argument("--seed", type=int, default=0)
    split.add_argument(
        "--write-claims",
        action="store_true",
        help="stamp each sample with its partition and this manifest's digest",
    )
    split.add_argument(
        "--fold", type=int, help="with --k-folds --write-claims: the validation fold"
    )
    split.add_argument("--assigned-by")
    add_json_flag(split)

    stats = group.add_parser("stats", help="streaming intensity and class statistics")
    stats.add_argument("manifest")
    stats.add_argument("-o", "--out", help="statistics JSON to write")
    stats.add_argument("--image", action="append", dest="images")
    stats.add_argument("--annotation", action="append", dest="annotations")
    stats.add_argument("--workers", type=int, default=1)
    stats.add_argument(
        "--stride",
        type=int,
        default=1,
        help="read every Nth slab along the first axis (approximate, opt-in)",
    )
    stats.add_argument("--partition", help="restrict to one partition of --set-id")
    stats.add_argument("--set-id", default="default")
    add_json_flag(stats)

    checker = group.add_parser("check", help="cross-file consistency (C1xx codes)")
    checker.add_argument("manifest")
    checker.add_argument("--set-id")
    checker.add_argument(
        "--deep",
        action="store_true",
        help="re-read each content_id instead of trusting size and mtime",
    )
    add_json_flag(checker)


def _fmt_ratios(ratios: dict[str, float]) -> str:
    return ",".join(f"{k}={v}" for k, v in ratios.items())


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command != "dataset":
        return None
    handlers = {
        "index": _index,
        "split": _split,
        "stats": _stats,
        "check": _check,
    }
    handler = handlers.get(getattr(args, "dataset_command", None) or "")
    if handler is None:
        return fail("usage: medh5 dataset {index|split|stats|check} ... (see --help)")
    try:
        return handler(args)
    except (MEDH5Error, OSError, json.JSONDecodeError) as exc:
        return fail(str(exc))


def _index(args: argparse.Namespace) -> int:
    manifest, failures = scan(args.root, on_error="raise" if args.strict else "warn")
    target = manifest.save(args.out)
    payload = {
        "manifest": str(target),
        "samples": len(manifest),
        "subjects": len(manifest.subjects),
        "sha256": manifest.sha256(),
        "failed": list(failures),
    }
    if args.json:
        emit(payload, as_json=True)
    else:
        print(
            f"{target}: {len(manifest)} sample(s), "
            f"{len(manifest.subjects)} subject(s), sha256 {manifest.sha256()[:12]}"
        )
        for failure in failures:
            print(f"  unreadable: {failure}")
    return EXIT_OK if manifest.entries else EXIT_ERROR


def _split(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    ratios = _parse_ratios(args.ratios) if args.ratios else None
    split = make_splits(
        manifest,
        set_id=args.set_id,
        group_by=args.group_by,
        stratify_by=args.stratify_by,
        ratios=ratios,
        k_folds=args.k_folds,
        seed=args.seed,
    )
    written: tuple[str, ...] = ()
    if args.write_claims:
        written = write_claims(
            split, manifest, assigned_by=args.assigned_by, fold=args.fold
        )
    if args.out:
        Path(args.out).write_text(json.dumps(split.to_json(), indent=2) + "\n")
    if args.json:
        emit({**split.to_json(), "claims_written": list(written)}, as_json=True)
    else:
        rows = [
            [partition, str(n), _balance_cell(split, partition)]
            for partition, n in split.counts.items()
        ]
        print(table(rows, ["partition", "samples", args.stratify_by or "-"]))
        print(
            f"{len(split.assignments)} group(s) by {args.group_by}, "
            f"manifest {split.manifest_sha256[:12]}"
        )
        if written:
            print(f"wrote split claims into {len(written)} file(s)")
        if split.underfilled:
            print(
                f"WARNING: {', '.join(split.underfilled)} got no groups --- "
                f"{len(split.assignments)} indivisible group(s) cannot be split "
                f"in the ratios {_fmt_ratios(dict(split.ratios))}"
            )
        if split.leaks():
            print(f"LEAK: {', '.join(split.leaks())}")
    return EXIT_ERROR if split.leaks() else EXIT_OK


def _balance_cell(split: Split, partition: str) -> str:
    balance = split.balance().get(partition, {})
    return ", ".join(f"{k}:{v}" for k, v in balance.items() if k != "-") or "-"


def _parse_ratios(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        if "=" not in part:
            raise MEDH5Error(f"--ratios expects NAME=VALUE, got {part!r}")
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)
    return out


def _stats(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    entries = manifest.entries
    if args.partition:
        entries = [
            e
            for e in entries
            if any(
                c.get("set_id") == args.set_id and c.get("partition") == args.partition
                for c in e.splits
            )
        ]
        if not entries:
            return fail(
                f"no sample claims partition {args.partition!r} of set "
                f"{args.set_id!r} --- run `medh5 dataset split --write-claims` first"
            )
    result = compute_stats(
        [e.path for e in entries],
        images=args.images,
        annotations=args.annotations,
        workers=args.workers,
        sample_stride=args.stride,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_json(), indent=2) + "\n")
    if args.json:
        emit(result.to_json(), as_json=True)
    else:
        rows = [
            [
                key,
                f"{m.mean:.4g}",
                f"{m.std:.4g}",
                f"{m.minimum:.4g}",
                f"{m.maximum:.4g}",
            ]
            for key, m in sorted(result.images.items())
        ]
        print(table(rows, ["image", "mean", "std", "min", "max"]))
        if result.classes:
            rows = [
                [
                    str(s.class_id),
                    f"{s.voxels:,}",
                    f"{s.present_in}/{s.examined_in}",
                    f"{s.prevalence:.0%}",
                ]
                for s in sorted(result.classes.values(), key=lambda s: s.class_id)
            ]
            print(table(rows, ["class", "voxels", "present/examined", "prevalence"]))
        for failure in result.failures:
            print(f"  unreadable: {failure}")
    return EXIT_OK if result.samples else EXIT_ERROR


def _check(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    report = check(manifest, set_id=args.set_id, deep=args.deep)
    if args.json:
        emit(report.to_json(), as_json=True)
    else:
        print(report.format())
        if report.coverage:
            rows = [
                [
                    str(class_id),
                    str(v["examined_in"]),
                    str(v["present_in"]),
                    str(v["of"]),
                ]
                for class_id, v in sorted(report.coverage.items())
            ]
            print(table(rows, ["class", "examined in", "present in", "of"]))
    return EXIT_OK if report.ok else EXIT_ERROR


__all__ = ["dispatch", "register"]
