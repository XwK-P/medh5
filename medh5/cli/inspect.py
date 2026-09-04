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
    info.add_argument("path", help="the sample or collection to summarise")
    add_json_flag(info)

    tree = sub.add_parser("tree", help="annotated object listing with spec roles")
    tree.add_argument("path", help="the sample or collection to list")
    add_json_flag(tree)

    validate = sub.add_parser("validate", help="check conformance (spec §15)")
    add_paths(validate)
    validate.add_argument(
        "--level",
        choices=LEVELS,
        default="semantic",
        help="how much to check: structural, semantic (default), integrity or strict",
    )
    validate.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="override the declared profiles; repeatable",
    )
    validate.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print every diagnostic, not only the summary",
    )
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

    _register_fix(sub)

    timeline = sub.add_parser("timeline", help="timepoints and what belongs to each")
    timeline.add_argument("path", help="the sample whose visits to list")
    add_json_flag(timeline)

    track = sub.add_parser("track", help="join instance ids across timepoints")
    track.add_argument("path", help="the sample whose instances to join across visits")
    track.add_argument("--class", dest="class_key", help="restrict to one class")
    add_json_flag(track)


def _register_fix(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fixer = sub.add_parser("fix", help="rebuild derived data; restamp digests")
    add_paths(fixer)
    fixer.add_argument(
        "--rebuild-index",
        action="store_true",
        help="recompute stale sampling indices (§14.3)",
    )
    fixer.add_argument(
        "--rewrite-digests",
        action="store_true",
        help="restamp digests over the current bytes --- see --reason",
    )
    fixer.add_argument(
        "--reason",
        help="why the digests are being rewritten; recorded in the file",
    )
    fixer.add_argument(
        "--by",
        dest="performed_by",
        help="who is making the change; recorded in provenance",
    )
    add_json_flag(fixer)


def _fix(args: argparse.Namespace) -> int:
    from medh5.integrity.repair import fix

    results = []
    for path in args.paths:
        try:
            repair = fix(
                path,
                rebuild_index=args.rebuild_index,
                rewrite_digests=args.rewrite_digests,
                reason=args.reason,
                performed_by=args.performed_by,
            )
        except MEDH5Error as exc:
            return fail(str(exc))
        results.append(repair.to_json())
        if args.json:
            continue
        diagnosis = repair.diagnosis
        if repair.changed:
            done = []
            if repair.rebuilt_index:
                done.append(f"rebuilt index for {', '.join(repair.rebuilt_index)}")
            if repair.rewrote_digests:
                done.append("rewrote digests")
            print(f"{path}: {'; '.join(done)}")
            for note in repair.notes:
                print(f"  note: {note}")
        elif diagnosis.clean:
            print(f"{path}: nothing to fix")
        else:
            print(f"{path}: needs attention, nothing changed")
            if diagnosis.stale_index:
                print(
                    f"  stale index: {', '.join(diagnosis.stale_index)} "
                    "(--rebuild-index)"
                )
            if diagnosis.mismatched:
                print(
                    f"  digest mismatch: {', '.join(diagnosis.mismatched)} "
                    "(--rewrite-digests, and read what it means first)"
                )
            if diagnosis.content_id_ok is False:
                print("  content_id does not match the file's own contents")
    emit(results, as_json=args.json)
    # Non-zero when a file still needs attention: a fix run that found problems
    # and was not asked to act on them has not succeeded, it has reported.
    outstanding = [
        r
        for r in results
        if not r["changed"]
        and (r["diagnosis"]["needs_index"] or r["diagnosis"]["needs_digests"])
    ]
    return EXIT_OK if not outstanding else EXIT_ERROR


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "info":
        return _info(args)
    if command == "tree":
        return _tree(args)
    if command == "fix":
        return _fix(args)
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
                                    a.get("grid") or "-",
                                    ",".join(a.get("timepoints") or ()) or "-",
                                    f"{a.get('annotated_classes', 0)}"
                                    f"/{a.get('classes', 0)}",
                                    "yes" if a.get("fully_covered") else "PARTIAL",
                                    a.get("quality") or "-",
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
            if summary.get("transforms"):
                print("\ntransforms")
                print(
                    _indent(
                        table(
                            [
                                [
                                    t["id"],
                                    t["kind"],
                                    t["from_frame"],
                                    t["to_frame"],
                                    ",".join(t["timepoints"]) or "-",
                                    "yes" if t["invertible"] else "no",
                                    t["metrics"] or "-",
                                ]
                                for t in summary["transforms"]
                            ],
                            ["id", "kind", "from", "to", "tp", "inv", "metrics"],
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
            root = sample.root
            if args.json:
                emit(
                    {"path": args.path, "objects": _tree_json(root)},
                    as_json=True,
                )
                return EXIT_OK
            print(args.path)
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


def _tree_json(node: Any, prefix: str = "") -> list[dict[str, Any]]:
    """The same listing `tree` prints, as data.

    `tree` names each object and the spec clause that gives it its role, which
    is exactly what a tool auditing a cohort wants -- and it was the one
    inspection command with no machine-readable form.
    """
    out: list[dict[str, Any]] = []
    for name in sorted(node, key=lambda n: (n != "meta", n)):
        child = node[name]
        path = f"{prefix}{name}"
        entry: dict[str, Any] = {
            "path": path,
            "name": name,
            "kind": "dataset" if isinstance(child, h5py.Dataset) else "group",
            "describe": _describe(child),
        }
        if not prefix:
            entry["role"] = _ROLES.get(name, "extension object (§16)")
        if isinstance(child, h5py.Dataset):
            entry["shape"] = [int(v) for v in child.shape]
            entry["dtype"] = str(child.dtype)
        else:
            entry["children"] = _tree_json(child, f"{path}/")
        out.append(entry)
    return out


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
            # Three answers, not two: a partial pass does not recompute the
            # root, and a file may declare no `content_id` at all (§13.2).
            content = {True: "ok", False: "MISMATCH", None: "not verified"}[
                result.content_id_ok
            ]
            print(
                f"{path}: {state}  {len(result.checked)} objects, content_id {content}"
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
            tracking = sample.tracks(args.class_key)
            if args.json:
                emit(tracking.to_json(), as_json=True)
                return EXIT_OK
            if not len(tracking):
                print("no instance-carrying annotations in this sample")
                return EXIT_OK
            timepoints = tracking.timepoints or ("-",)
            rows = []
            for instance_id, track in sorted(tracking.items()):
                states = tracking.states(instance_id)
                rows.append(
                    [
                        instance_id,
                        track.class_key or track.class_id,
                        *(_cell(track, states, tp) for tp in timepoints),
                        _trend(tracking, instance_id, track, timepoints),
                    ]
                )
            print(
                table(
                    rows,
                    ["instance", "class", *timepoints, "trend"],
                )
            )
            conflicts = tracking.class_conflicts()
            for instance_id, class_ids in sorted(conflicts.items()):
                print(
                    f"\nW909  instance {instance_id} carries class ids "
                    f"{list(class_ids)}"
                )
            print(
                "\nvolume in the grid's units; `resolved` means the class was in "
                "`annotated_class_ids` at that timepoint and the object was not "
                "found, `unexamined` that nobody looked (spec §7.4, §11.3)."
            )
            return EXIT_OK
    except MEDH5Error as exc:
        return fail(str(exc))


def _cell(track: Any, states: dict[str, str], timepoint: str) -> str:
    state = states.get(timepoint, "unexamined")
    if state != "present":
        return state
    volume = track.volume(timepoint)
    return "present" if volume is None else f"{volume:.4g}"


def _trend(
    tracking: Any, instance_id: int, track: Any, timepoints: tuple[str, ...]
) -> str:
    """Relative volume change from first to last visit, where it is measurable."""
    if len(timepoints) < 2:  # noqa: PLR2004 - a trend needs two visits
        return "-"
    change = track.relative_change(timepoints[0], timepoints[-1])
    if change is not None:
        return f"{change:+.1%}"
    if tracking.is_new(instance_id):
        return "new"
    if tracking.is_resolved(instance_id):
        return "resolved"
    return "-"


__all__ = ["dispatch", "register"]
