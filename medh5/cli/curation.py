"""``pack``, ``unpack``, ``ls``, ``prov``, ``agree``, ``splits`` and ``scrub``.

The curation half of the command line: shards (§2.2), the provenance graph and
quality records (§11), the cross-file split audit (§12.3) that no per-file
validator can perform, and the de-identification sweep (§11.4).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import medh5
from medh5.cli._common import (
    EXIT_ERROR,
    EXIT_OK,
    add_json_flag,
    add_paths,
    emit,
    fail,
    human_bytes,
    indent,
    table,
)
from medh5.collection import SUFFIX, open_collection, pack, unpack
from medh5.curation.agreement import compare_instances, compare_voxel
from medh5.curation.scrub import PROFILES as SCRUB_PROFILES
from medh5.curation.splits import audit_splits
from medh5.errors import MEDH5Error


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pack_cmd = sub.add_parser("pack", help=f"bundle sample files into one {SUFFIX}")
    add_paths(pack_cmd, "sample files to pack")
    pack_cmd.add_argument("-o", "--out", required=True, help=f"output {SUFFIX} file")
    pack_cmd.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="sample key for each source, in order; defaults to the file stem",
    )
    add_json_flag(pack_cmd)

    unpack_cmd = sub.add_parser("unpack", help="extract samples from a collection")
    unpack_cmd.add_argument("path", help="the .medh5c collection to extract from")
    unpack_cmd.add_argument("-o", "--out", required=True, help="output directory")
    unpack_cmd.add_argument(
        "--key", action="append", dest="keys", help="extract only these keys"
    )
    add_json_flag(unpack_cmd)

    ls_cmd = sub.add_parser("ls", help="list the samples in a collection")
    ls_cmd.add_argument("path", help="the .medh5c collection to list")
    add_json_flag(ls_cmd)

    prov = sub.add_parser("prov", help="who produced what, and how good it is (§11)")
    prov.add_argument("path", help="the sample whose provenance to print")
    add_json_flag(prov)

    agree = sub.add_parser("agree", help="measure agreement between two annotations")
    agree.add_argument("path", help="the sample holding both annotations")
    agree.add_argument("a", metavar="A", help="first annotation id")
    agree.add_argument("b", metavar="B", help="second annotation id")
    agree.add_argument(
        "--metric",
        choices=("dice", "iou"),
        default="dice",
        help="agreement metric: dice (default) or iou",
    )
    agree.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold when matching objects (default 0.5)",
    )
    agree.add_argument(
        "--record",
        action="store_true",
        help="print the `quality.agreement` record this measurement produces",
    )
    add_json_flag(agree)

    scrub = sub.add_parser(
        "scrub", help="find identifiers in the container and attest to it (§11.4)"
    )
    add_paths(scrub)
    scrub.add_argument(
        "--profile",
        choices=SCRUB_PROFILES,
        default="basic",
        help="how hard to look: basic (default) or strict",
    )
    scrub.add_argument(
        "--apply",
        action="store_true",
        dest="apply_changes",
        help="act on the actionable findings; without it, nothing is written",
    )
    scrub.add_argument(
        "--date-shift-days",
        type=int,
        help="shift dates by N days instead of dropping them, preserving intervals",
    )
    scrub.add_argument(
        "--salt",
        default="",
        help="salt the UID pseudonyms; keep it to reproduce the mapping",
    )
    scrub.add_argument(
        "--by",
        dest="performed_by",
        help="who performed the de-identification; recorded in the file",
    )
    add_json_flag(scrub)

    splits = sub.add_parser("splits", help="audit split claims across files (§12.3)")
    add_paths(splits, "sample or collection files to audit")
    add_json_flag(splits)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "pack":
        return _pack(args)
    if command == "unpack":
        return _unpack(args)
    if command == "ls":
        return _ls(args)
    if command == "prov":
        return _prov(args)
    if command == "agree":
        return _agree(args)
    if command == "splits":
        return _splits(args)
    if command == "scrub":
        return _scrub(args)
    return None


# -- collections -----------------------------------------------------------


def _pack(args: argparse.Namespace) -> int:
    try:
        out = pack(args.paths, args.out, keys=args.keys)
    except MEDH5Error as exc:
        return fail(str(exc))
    size = out.stat().st_size
    sources = sum(Path(p).stat().st_size for p in args.paths)
    payload = {
        "out": str(out),
        "samples": len(args.paths),
        "bytes": size,
        "source_bytes": sources,
    }
    if args.json:
        emit(payload, as_json=True)
        return EXIT_OK
    print(
        f"{out}: {len(args.paths)} samples, {human_bytes(size)} "
        f"(sources {human_bytes(sources)})"
    )
    return EXIT_OK


def _unpack(args: argparse.Namespace) -> int:
    try:
        written = unpack(args.path, args.out, keys=args.keys)
    except MEDH5Error as exc:
        return fail(str(exc))
    if args.json:
        emit([str(p) for p in written], as_json=True)
        return EXIT_OK
    for path in written:
        print(path)
    return EXIT_OK


def _ls(args: argparse.Namespace) -> int:
    try:
        with open_collection(args.path) as collection:
            summary = collection.summary()
            if args.json:
                emit(summary, as_json=True)
                return EXIT_OK
            print(f"{args.path}  ({len(collection)} samples, {summary['version']})")
            print(
                table(
                    [
                        [
                            entry["key"],
                            entry["subject_id"],
                            ",".join(entry["timepoints"]),
                            len(entry["images"]),
                            len(entry["annotations"]),
                            ",".join(entry["profiles"]),
                            (entry["content_id"] or "-")[:19],
                        ]
                        for entry in summary["samples"]
                    ],
                    ["key", "subject", "tp", "img", "ann", "profiles", "content_id"],
                )
            )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


# -- curation --------------------------------------------------------------


def _prov(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            document = sample.document
            graph = document.provenance
            payload: dict[str, Any] = {
                "provenance": graph.to_json(),
                "quality": {k: v.to_json() for k, v in document.quality.items()},
                "deidentification": (
                    None
                    if document.deidentification is None
                    else document.deidentification.to_json()
                ),
            }
            if args.json:
                emit(payload, as_json=True)
                return EXIT_OK
            if not graph:
                print("no provenance graph (§11.1)")
            else:
                print("agents")
                print(
                    indent(
                        table(
                            [
                                [a.id, a.type, a.name, a.version or "-", a.role or "-"]
                                for a in graph.agents
                            ],
                            ["id", "type", "name", "version", "role"],
                        )
                    )
                )
                print("\nactivities")
                print(
                    indent(
                        table(
                            [
                                [
                                    act.id,
                                    act.type,
                                    act.agent or "-",
                                    act.ended or act.started or "-",
                                    ",".join(act.outputs) or "-",
                                ]
                                for act in graph.activities
                            ],
                            ["id", "type", "agent", "when", "outputs"],
                        )
                    )
                )
            if document.quality:
                print("\nquality")
                print(
                    indent(
                        table(
                            [
                                [
                                    key,
                                    record.status,
                                    "-"
                                    if record.confidence is None
                                    else f"{record.confidence:.3g}",
                                    ",".join(record.reviewed_by) or "-",
                                    ";".join(
                                        f"{a.metric}={a.value:.3g}"
                                        for a in record.agreement
                                    )
                                    or "-",
                                    ";".join(i.code for i in record.issues) or "-",
                                ]
                                for key, record in sorted(document.quality.items())
                            ],
                            [
                                "key",
                                "status",
                                "conf",
                                "reviewers",
                                "agreement",
                                "issues",
                            ],
                        )
                    )
                )
            deid = document.deidentification
            print(
                f"\ndeidentification  {deid.method if deid else 'ABSENT (W903)'}"
                + (
                    f", dates shifted {deid.date_shift_days}d"
                    if deid and deid.date_shift_days is not None
                    else ""
                )
            )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


def _agree(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            first = sample.annotations[args.a]
            second = sample.annotations[args.b]
            if first.kind == "instances" or second.kind == "instances":
                result: Any = compare_instances(first, second, threshold=args.threshold)
            else:
                result = compare_voxel(first, second, metric=args.metric)
            payload = result.to_json()
            if args.record:
                payload = {
                    "quality_agreement": result.to_record().to_json(),
                    **payload,
                }
            if args.json:
                emit(payload, as_json=True)
                return EXIT_OK
            print(f"{args.a} vs {args.b}: {payload['metric']} = {result.value:.4f}")
            if hasattr(result, "per_class"):
                print(
                    indent(
                        table(
                            [
                                [k, f"{v:.4f}"]
                                for k, v in sorted(result.per_class.items())
                            ],
                            ["class", args.metric],
                        )
                    )
                )
                if result.skipped:
                    print("\nnot scored: " + ", ".join(result.skipped))
                    print(
                        "  a class one side never examined is not a disagreement "
                        "(§11.3)"
                    )
            else:
                print(
                    f"  matched {len(result.matched)} by {result.matched_by}, "
                    f"mean IoU {result.mean_iou:.4f}; "
                    f"{len(result.only_in_a)} only in {args.a}, "
                    f"{len(result.only_in_b)} only in {args.b}"
                )
                for instance_id, class_a, class_b in result.class_mismatches:
                    print(
                        f"  MISMATCH instance {instance_id}: "
                        f"class {class_a} vs {class_b}"
                    )
            return EXIT_OK
    except KeyError as exc:
        return fail(f"no such annotation: {exc}")
    except MEDH5Error as exc:
        return fail(str(exc))


def _splits(args: argparse.Namespace) -> int:
    audit = audit_splits(args.paths)
    if args.json:
        emit(audit.to_json(), as_json=True)
        return EXIT_OK if audit.ok else EXIT_ERROR
    if not audit.set_ids:
        print(f"no split claims in {len(args.paths)} file(s)")
    for set_id in audit.set_ids:
        counts = audit.counts()[set_id]
        total = sum(counts.values())
        print(
            f"{set_id}: {total} samples  "
            + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
    for conflict in audit.conflicts:
        print(f"W906  {conflict}")
        for manifest, paths in sorted(conflict.paths_by_manifest.items()):
            print(f"        {manifest[:16]}  {len(paths)} file(s), e.g. {paths[0]}")
    for leak in audit.leaks:
        print(f"LEAK  {leak}")
        for path in leak.paths:
            print(f"        {path}")
    if audit.unclaimed:
        print(f"\n{len(audit.unclaimed)} file(s) carry no split claim")
    for path, error in audit.unreadable:
        print(f"UNREADABLE  {path}: {error}")
    if audit.leaks:
        print(
            "\na grouping key in two partitions is train/test leakage (§12.2); "
            "re-split rather than re-stamp."
        )
    return EXIT_OK if audit.ok else EXIT_ERROR


__all__ = ["dispatch", "register"]


# -- de-identification ------------------------------------------------------


def _scrub(args: argparse.Namespace) -> int:
    from medh5.curation import scrub as scrubber

    reports = []
    for path in args.paths:
        try:
            if args.apply_changes:
                report = scrubber.apply(
                    path,
                    profile=args.profile,
                    salt=args.salt,
                    date_shift_days=args.date_shift_days,
                    performed_by=args.performed_by,
                )
            else:
                report = scrubber.scan(path, profile=args.profile)
        except MEDH5Error as exc:
            return fail(str(exc))
        reports.append(report.to_json())
        if not args.json:
            print(report.format())
            if not args.apply_changes and report.actionable:
                print(
                    f"  {len(report.actionable)} finding(s) can be acted on: "
                    "re-run with --apply"
                )
    emit(reports, as_json=args.json)
    # A scrub that only looked reports findings as a failure, so it is usable in
    # a pipeline gate.  A scrub that *acted* used to succeed merely for having
    # acted --- so a file whose patient name the tool had flagged and left in
    # place exited 0 carrying a fresh de-identification record.  Both forms now
    # report the state of the file: `ok` is "nothing found" for a scan and
    # "nothing actionable left, by re-scanning what was written" for an apply.
    return EXIT_OK if all(r["ok"] for r in reports) else EXIT_ERROR
