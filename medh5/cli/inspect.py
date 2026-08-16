"""``info``, ``tree``, ``validate``, ``verify``, ``timeline``, ``track``."""

from __future__ import annotations

import argparse
from typing import Any

import h5py

import medh5
from medh5.cli._common import (
    EXIT_ERROR,
    EXIT_OK,
    add_json_flag,
    add_paths,
    emit,
    fail,
    human_bytes,
    table,
)
from medh5.errors import MEDH5Error
from medh5.storage.codecs import describe_filters
from medh5.validate import validate_paths
from medh5.validate.report import LEVELS


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    info = sub.add_parser("info", help="summarise a sample")
    info.add_argument("path")
    add_json_flag(info)

    tree = sub.add_parser("tree", help="annotated object listing with spec roles")
    tree.add_argument("path")

    validate = sub.add_parser("validate", help="check conformance (spec §15)")
    add_paths(validate)
    validate.add_argument("--level", choices=LEVELS, default="semantic")
    validate.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="override the declared profiles; repeatable",
    )
    validate.add_argument("-v", "--verbose", action="store_true")
    add_json_flag(validate)

    verify = sub.add_parser("verify", help="check digests and content_id (spec §13)")
    add_paths(verify)
    verify.add_argument(
        "--partial",
        action="append",
        dest="partial",
        metavar="OBJ",
        help="verify only these objects; repeatable",
    )
    add_json_flag(verify)

    timeline = sub.add_parser("timeline", help="timepoints and what belongs to each")
    timeline.add_argument("path")
    add_json_flag(timeline)

    track = sub.add_parser("track", help="join instance ids across timepoints")
    track.add_argument("path")
    track.add_argument("--class", dest="class_key", help="restrict to one class")
    add_json_flag(track)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "info":
        return _info(args)
    if command == "tree":
        return _tree(args)
    if command == "validate":
        return _validate(args)
    if command == "verify":
        return _verify(args)
    if command == "timeline":
        return _timeline(args)
    if command == "track":
        return _track(args)
    return None


def _info(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            summary = sample.summary()
            if args.json:
                emit(summary, as_json=True)
                return EXIT_OK
            print(f"{args.path}")
            print(f"  format      {summary['version']} ({summary['kind']})")
            print(f"  profiles    {', '.join(summary['profiles'])}")
            print(
                f"  sample      {summary['sample_id']}  subject {summary['subject_id']}"
            )
            print(f"  content_id  {summary['content_id'] or '-'}")
            print("\ntimepoints")
            print(
                _indent(
                    table(
                        [
                            [
                                t["id"],
                                t["index"],
                                t["label"] or "-",
                                t["days_from_baseline"]
                                if t["days_from_baseline"] is not None
                                else "-",
                            ]
                            for t in summary["timepoints"]
                        ],
                        ["id", "index", "label", "days"],
                    )
                )
            )
            print("\ngrids")
            print(
                _indent(
                    table(
                        [
                            [
                                g["id"],
                                "x".join(map(str, g["shape"])),
                                " ".join(f"{v:g}" for v in g["spacing"]),
                                g["coord_system"],
                                g["units"],
                                g["timepoint"] or "-",
                                g["frame_uid"] or "-",
                            ]
                            for g in summary["grids"]
                        ],
                        [
                            "id",
                            "shape",
                            "spacing",
                            "system",
                            "units",
                            "timepoint",
                            "frame",
                        ],
                    )
                )
            )
            print("\nimages")
            print(
                _indent(
                    table(
                        [
                            [
                                i["id"],
                                i["modality"],
                                "x".join(map(str, i["shape"])),
                                i["dtype"],
                                i["value_units"] or "-",
                                i["grid"],
                                describe_filters(sample.images[i["id"]].dataset),
                                human_bytes(i["nbytes"]),
                            ]
                            for i in summary["images"]
                        ],
                        [
                            "id",
                            "mod",
                            "shape",
                            "dtype",
                            "units",
                            "grid",
                            "codec",
                            "raw",
                        ],
                    )
                )
            )
            if summary["annotations"]:
                print("\nannotations")
                print(
                    _indent(
                        table(
                            [
                                [
                                    a["id"],
                                    a["kind"],
                                    a["task"],
                                    a["grid"] or "-",
                                    ",".join(a["timepoints"]) or "-",
                                    f"{a['annotated_classes']}/{a['classes']}",
                                    "yes" if a["fully_covered"] else "PARTIAL",
                                    a["quality"] or "-",
                                ]
                                for a in summary["annotations"]
                            ],
                            [
                                "id",
                                "kind",
                                "task",
                                "grid",
                                "tp",
                                "cover",
                                "full",
                                "quality",
                            ],
                        )
                    )
                )
            if summary["index"]:
                print(f"\nindex        {', '.join(summary['index'])}")
            if summary["label_set"]:
                label = summary["label_set"]
                print(
                    f"\nlabel set    {label['id']} v{label['version']} "
                    f"({label['classes']} classes, {label['form']})"
                )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


_ROLES = {
    "meta": "sample document (§2.4)",
    "grids": "geometry (§3.2)",
    "images": "image data (§4)",
    "annotations": "ground truth (§6-§9)",
    "transforms": "spatial mappings (§10)",
    "index": "derived sampling caches (§14.3)",
}


def _tree(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            print(args.path)
            root = sample.root
            for name in sorted(root, key=lambda n: (n != "meta", n)):
                role = _ROLES.get(name, "extension object (§16)")
                node = root[name]
                if isinstance(node, h5py.Dataset):
                    print(f"├── {name:22s} {_describe(node)}   # {role}")
                    continue
                print(f"├── {name}/{'':{max(0, 21 - len(name))}} # {role}")
                for child in sorted(node):
                    sub = node[child]
                    print(f"│   ├── {child:20s} {_describe(sub)}")
                    if isinstance(sub, h5py.Group):
                        for leaf in sorted(sub):
                            print(f"│   │   ├── {leaf:16s} {_describe(sub[leaf])}")
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


def _describe(node: Any) -> str:
    if isinstance(node, h5py.Dataset):
        shape = "scalar" if node.shape == () else "x".join(map(str, node.shape))
        return f"{shape} {node.dtype.str} {describe_filters(node)}"
    kind = node.attrs.get("kind")
    if kind is not None:
        from medh5._hdf5 import as_str

        return f"group kind={as_str(kind)}"
    return "group"


def _validate(args: argparse.Namespace) -> int:
    reports = validate_paths(args.paths, level=args.level, profiles=args.profiles)
    if args.json:
        emit([r.to_json() for r in reports], as_json=True)
    else:
        for report in reports:
            print(report.format(verbose=args.verbose))
    return EXIT_OK if all(r.ok for r in reports) else EXIT_ERROR


def _verify(args: argparse.Namespace) -> int:
    results: list[dict[str, Any]] = []
    ok = True
    for path in args.paths:
        try:
            with medh5.open(path) as sample:
                result = sample.verify(partial=args.partial)
        except MEDH5Error as exc:
            return fail(str(exc))
        summary = {"path": path, **result.summary()}
        results.append(summary)
        ok = ok and result.ok
        if not args.json:
            state = "OK" if result.ok else "FAILED"
            print(
                f"{path}: {state}  {len(result.checked)} objects, "
                f"content_id {'ok' if result.content_id_ok else result.content_id_ok}"
            )
            for name in result.mismatched:
                print(f"  MISMATCH  {name}")
            for name in result.stale_index:
                print(f"  STALE     index/{name} (rebuild with `medh5 index build`)")
    emit(results, as_json=args.json)
    return EXIT_OK if ok else EXIT_ERROR


def _timeline(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            rows = []
            payload = []
            for tp in sample.timepoints:
                view = sample.at(tp.id)
                rows.append(
                    [
                        tp.id,
                        tp.index,
                        tp.label or "-",
                        tp.days_from_baseline
                        if tp.days_from_baseline is not None
                        else "-",
                        ",".join(sorted(view.images)) or "-",
                        ",".join(sorted(view.annotations)) or "-",
                    ]
                )
                payload.append(
                    {
                        **tp.to_json(),
                        "images": sorted(view.images),
                        "annotations": sorted(view.annotations),
                        "grids": sorted(view.grids),
                    }
                )
            if args.json:
                emit(payload, as_json=True)
                return EXIT_OK
            print(
                table(rows, ["id", "index", "label", "days", "images", "annotations"])
            )
            spanning = sample.annotations.spanning()
            if spanning:
                print(
                    "\nspanning annotations: "
                    + ", ".join(
                        f"{a.ann_id} ({','.join(a.timepoints)})" for a in spanning
                    )
                )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


def _track(args: argparse.Namespace) -> int:
    try:
        with medh5.open(args.path) as sample:
            tracks = sample.track(args.class_key)
            declared = set(sample.timepoints.ids)
            payload = {}
            rows = []
            for instance_id, seen in sorted(tracks.items()):
                present = sorted(seen)
                state = (
                    "persisted"
                    if len(present) == len(declared)
                    else "new"
                    if present == [sample.timepoints[-1].id]
                    else "resolved"
                    if present == [sample.timepoints[0].id]
                    else "partial"
                )
                payload[str(instance_id)] = {
                    "timepoints": present,
                    "state": state,
                    "annotations": {tp: seen[tp][0] for tp in present},
                }
                rows.append(
                    [
                        instance_id,
                        ",".join(present),
                        state,
                        ",".join(seen[tp][0] for tp in present),
                    ]
                )
            if args.json:
                emit(payload, as_json=True)
                return EXIT_OK
            if not rows:
                print("no instance-carrying annotations in this sample")
                return EXIT_OK
            print(table(rows, ["instance", "timepoints", "state", "annotations"]))
            print(
                "\nabsence means `resolved` only where `annotated_class_ids` covers "
                "the class at that timepoint (spec §7.4)."
            )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


__all__ = ["dispatch", "register"]
