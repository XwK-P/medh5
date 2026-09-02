"""``labels show``, ``labels registry list``, ``labels check``."""

from __future__ import annotations

import argparse

import medh5
from medh5.cli._common import EXIT_ERROR, EXIT_OK, add_json_flag, emit, fail, table
from medh5.errors import MEDH5Error
from medh5.labels import registry


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    labels = sub.add_parser("labels", help="inspect label sets")
    group = labels.add_subparsers(dest="labels_command", metavar="COMMAND")

    show = group.add_parser("show", help="print a file's label set")
    show.add_argument("path", help="the sample whose label set to print")
    add_json_flag(show)

    check = group.add_parser("check", help="report vocabulary drift across files")
    check.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="the samples to compare label sets across",
    )
    add_json_flag(check)

    reg = group.add_parser("registry", help="bundled vocabularies")
    reg_sub = reg.add_subparsers(dest="registry_command", metavar="COMMAND")
    listing = reg_sub.add_parser("list", help="list bundled vocabularies")
    add_json_flag(listing)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command != "labels":
        return None
    sub = getattr(args, "labels_command", None)
    if sub == "show":
        return _show(args)
    if sub == "check":
        return _check(args)
    if sub == "registry":
        return _registry(args)
    return fail("usage: medh5 labels {show|check|registry}")


def _show(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            label_set = sample.label_set
    except MEDH5Error as exc:
        return fail(str(exc))
    if label_set is None:
        print(f"{args.path}: no label set")
        return EXIT_OK
    if args.json:
        emit(label_set.to_json(), as_json=True)
        return EXIT_OK
    print(
        f"{label_set.id} v{label_set.version}  form={label_set.form}  "
        f"sha256={label_set.digest()[:16]}...  ({len(label_set)} classes)"
    )
    print(
        table(
            [
                [
                    c.id,
                    c.key,
                    c.name,
                    c.category or "-",
                    ",".join(str(p) for p in c.parents) or "-",
                    ",".join(f"{code.system}:{code.code}" for code in c.codes) or "-",
                ]
                for c in label_set
            ],
            ["id", "key", "name", "category", "parents", "codes"],
        )
    )
    return EXIT_OK


def _check(args: argparse.Namespace) -> int:
    seen: dict[str, list[str]] = {}
    rows = []
    for path in args.paths:
        try:
            with medh5.open(path) as sample:
                label_set = sample.label_set
        except MEDH5Error as exc:
            return fail(str(exc))
        key = (
            f"{label_set.id}@{label_set.version}#{label_set.digest()[:16]}"
            if label_set
            else "<none>"
        )
        seen.setdefault(key, []).append(path)
        rows.append([path, key])
    if args.json:
        emit({"vocabularies": {k: v for k, v in seen.items()}}, as_json=True)
    else:
        print(table(rows, ["file", "vocabulary"]))
        if len(seen) > 1:
            print(
                f"\n{len(seen)} distinct vocabularies across {len(args.paths)} files; "
                "class ids are not comparable between them."
            )
    return EXIT_OK if len(seen) <= 1 else EXIT_ERROR


def _registry(args: argparse.Namespace) -> int:
    if getattr(args, "registry_command", None) != "list":
        return fail("usage: medh5 labels registry list")
    described = registry.describe()
    if args.json:
        emit(described, as_json=True)
        return EXIT_OK
    print(
        table(
            [
                [name, info["version"], info["classes"], info["sha256"][:16] + "..."]
                for name, info in described.items()
            ],
            ["name", "version", "classes", "sha256"],
        )
    )
    return EXIT_OK


__all__ = ["dispatch", "register"]
