"""``seg stats``, ``seg convert``, ``index build`` --- the encoding tools."""

from __future__ import annotations

import argparse

import medh5
from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.voxel.select import analyse, cost_model, select_encoding
from medh5.annotations.voxel.transcode import (
    TRANSCODABLE,
    annotation_to_masks,
)
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
from medh5.storage.index import DEFAULT_MAX_COORDS, DEFAULT_OCCUPANCY_FACTOR


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    seg = sub.add_parser("seg", help="voxel-annotation tools")
    group = seg.add_subparsers(dest="seg_command", metavar="COMMAND")

    stats = group.add_parser(
        "stats", help="per-class counts, overlap graph and encoding cost model"
    )
    stats.add_argument("path")
    stats.add_argument("annotation")
    add_json_flag(stats)

    convert = group.add_parser("convert", help="re-encode losslessly (spec §7.6)")
    convert.add_argument("path")
    convert.add_argument("annotation")
    convert.add_argument("--to", required=True, choices=TRANSCODABLE)
    convert.add_argument(
        "--dry-run", action="store_true", help="report the size delta, write nothing"
    )
    add_json_flag(convert)

    index = sub.add_parser("index", help="derived sampling caches (spec §14.3)")
    index_sub = index.add_subparsers(dest="index_command", metavar="COMMAND")
    build = index_sub.add_parser("build", help="build or refresh sampling indices")
    add_paths(build)
    build.add_argument("--max-coords", type=int, default=DEFAULT_MAX_COORDS)
    build.add_argument("--occupancy", type=int, default=DEFAULT_OCCUPANCY_FACTOR)
    build.add_argument("--seed", type=int, default=0)
    add_json_flag(build)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "seg":
        sub = getattr(args, "seg_command", None)
        if sub == "stats":
            return _stats(args)
        if sub == "convert":
            return _convert(args)
        return fail("usage: medh5 seg {stats|convert}")
    if command == "index":
        if getattr(args, "index_command", None) == "build":
            return _index_build(args)
        return fail("usage: medh5 index build PATH...")
    return None


def _open_voxel(path: str, ann_id: str) -> VoxelAnnotation:
    sample = medh5.open(path)
    annotation = sample.annotations[ann_id]
    if not isinstance(annotation, VoxelAnnotation):
        sample.close()
        raise MEDH5Error(
            f"annotation {ann_id!r} has kind {annotation.kind!r}, which is not a "
            "voxel encoding"
        )
    return annotation


def _stats(args: argparse.Namespace) -> int:
    try:
        annotation = _open_voxel(args.path, args.annotation)
    except (MEDH5Error, KeyError) as exc:
        return fail(str(exc))
    try:
        masks = annotation_to_masks(annotation)
        measured = analyse(masks, annotation.spatial_shape)
        costs = cost_model(measured)
        chosen, _ = select_encoding(stats=measured)
        payload = {
            "annotation": args.annotation,
            "kind": annotation.kind,
            "recommended": chosen,
            "stats": measured.summary(),
            "cost_bytes": {
                "labelmap": costs.labelmap,
                "layers": costs.layers,
                "bitmask": costs.bitmask,
                "instances": costs.instances,
                "probmap": costs.probmap,
            },
        }
        if args.json:
            emit(payload, as_json=True)
            return EXIT_OK
        print(f"{args.path} :: {args.annotation}  (stored as `{annotation.kind}`)")
        print(
            f"  classes {measured.n_classes}   voxels {measured.n_voxels}   "
            f"fill {measured.fill:.3g}   depth {measured.depth:.3g}"
        )
        print(
            f"  overlap graph: {len(measured.edges)} edges, mean degree "
            f"{measured.mean_degree:.2f} -> {measured.n_layers} layers, "
            f"{measured.n_planes} bitplanes"
        )
        print("\nper-class voxel counts")
        print(
            table(
                [
                    [cid, annotation.class_key(cid), measured.counts[cid]]
                    for cid in measured.class_ids
                ],
                ["id", "key", "voxels"],
            )
        )
        print("\nraw cost by encoding (pre-compression)")
        rows = [
            [
                name,
                "-" if value is None else human_bytes(value),
                "<- stored"
                if name == annotation.kind
                else ("<- recommended" if name == chosen else ""),
            ]
            for name, value in payload["cost_bytes"].items()
        ]
        print(table(rows, ["encoding", "bytes", ""]))
        if chosen != annotation.kind:
            print(
                f"\n`medh5 seg convert {args.path} {args.annotation} --to {chosen}` "
                "would re-encode it losslessly."
            )
        return EXIT_OK
    finally:
        annotation.group.file.close()


def _convert(args: argparse.Namespace) -> int:
    try:
        annotation = _open_voxel(args.path, args.annotation)
    except (MEDH5Error, KeyError) as exc:
        return fail(str(exc))
    try:
        before = sum(
            int(annotation.group[name].nbytes)
            for name in annotation.group
            if hasattr(annotation.group[name], "nbytes")
        )
        from medh5.annotations.voxel.transcode import transcode

        payload = transcode(annotation, args.to)
        after = payload.nbytes
        source_kind = annotation.kind
    except MEDH5Error as exc:
        return fail(str(exc))
    finally:
        annotation.group.file.close()

    result = {
        "path": args.path,
        "annotation": args.annotation,
        "from": source_kind,
        "to": args.to,
        "bytes_before": before,
        "bytes_after": after,
        "applied": not args.dry_run,
    }
    if not args.dry_run:
        try:
            with medh5.amend(args.path) as writer:
                writer.transcode_annotation(args.annotation, args.to)
        except MEDH5Error as exc:
            return fail(str(exc))
    if args.json:
        emit(result, as_json=True)
    else:
        verb = "would re-encode" if args.dry_run else "re-encoded"
        print(
            f"{verb} {args.annotation}: {source_kind} -> {args.to}  "
            f"{human_bytes(before)} -> {human_bytes(after)} raw"
        )
    return EXIT_OK


def _index_build(args: argparse.Namespace) -> int:
    built: dict[str, list[str]] = {}
    for path in args.paths:
        try:
            with medh5.amend(path) as writer:
                names = writer.build_index(
                    max_coords=args.max_coords,
                    occupancy=args.occupancy or None,
                    seed=args.seed,
                )
            built[path] = list(names)
        except MEDH5Error as exc:
            return fail(str(exc))
        if not args.json:
            print(
                f"{path}: built index for {', '.join(names) if names else '(nothing)'}"
            )
    emit(built, as_json=args.json)
    return EXIT_OK if any(built.values()) else EXIT_ERROR


__all__ = ["dispatch", "register"]
