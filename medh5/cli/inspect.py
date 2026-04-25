"""``info`` / ``validate`` / ``validate-all`` / ``audit`` / ``recompress``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py

from medh5.cli._common import Handler, ValidationPayload, iter_medh5
from medh5.core import MEDH5File
from medh5.exceptions import MEDH5Error
from medh5.integrity import VerifyResult


def _filter_summary(ds: object) -> list[str]:
    if not isinstance(ds, h5py.Dataset):
        return []
    plist = ds.id.get_create_plist()
    filters: list[str] = []
    for i in range(plist.get_nfilters()):
        filter_info = plist.get_filter(i)
        filter_name = filter_info[3]
        if isinstance(filter_name, bytes):
            filter_name = filter_name.decode("utf-8", errors="replace")
        filters.append(str(filter_name))
    return filters


def _build_info_payload(path: Path) -> dict[str, object]:
    with MEDH5File(path) as f:
        meta = f.meta
        img_grp = f.images
        first_key = next(iter(img_grp))
        first_ds = img_grp[first_key]
        review_status = "pending"
        if meta.extra and isinstance(meta.extra.get("review"), dict):
            raw = meta.extra["review"].get("status")
            if isinstance(raw, str) and raw:
                review_status = raw
        return {
            "file": str(path),
            "schema_version": meta.schema_version,
            "image_names": meta.image_names,
            "shape": list(first_ds.shape),
            "dtype": str(first_ds.dtype),
            "chunks": list(first_ds.chunks) if first_ds.chunks is not None else None,
            "filters": _filter_summary(first_ds),
            "label": meta.label,
            "label_name": meta.label_name,
            "seg_names": meta.seg_names,
            "has_bbox": meta.has_bbox,
            "spacing": meta.spatial.spacing,
            "origin": meta.spatial.origin,
            "coord_system": meta.spatial.coord_system,
            "patch_size": meta.patch_size,
            "review_status": review_status,
            "extra": meta.extra,
        }


def _cmd_info(args: argparse.Namespace) -> int:
    """Print metadata summary for a .medh5 file."""
    path = Path(args.file)
    try:
        payload = _build_info_payload(path)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0

        print(f"File:         {payload['file']}")
        print(f"Schema:       v{payload['schema_version']}")
        print(f"Images:       {payload['image_names']}")
        print(f"Shape:        {payload['shape']}")
        print(f"Dtype:        {payload['dtype']}")
        print(f"Chunks:       {payload['chunks']}")
        print(f"Filters:      {payload['filters']}")
        if payload["label"] is not None:
            label_str = repr(payload["label"])
            if payload["label_name"]:
                label_str += f" ({payload['label_name']})"
            print(f"Label:        {label_str}")
        if payload["seg_names"]:
            print(f"Seg masks:    {payload['seg_names']}")
        if payload["has_bbox"]:
            print("Bboxes:       yes")
        if payload["spacing"] is not None:
            print(f"Spacing:      {payload['spacing']}")
        if payload["origin"] is not None:
            print(f"Origin:       {payload['origin']}")
        if payload["coord_system"] is not None:
            print(f"Coord system: {payload['coord_system']}")
        if payload["patch_size"] is not None:
            print(f"Patch size:   {payload['patch_size']}")
        print(f"Review:       {payload['review_status']}")
        if payload["extra"] is not None:
            print(f"Extra:        {payload['extra']}")
    except MEDH5Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a .medh5 file structure and schema."""
    path = Path(args.file)
    report = MEDH5File.validate(path)
    ok = report.ok(strict=args.strict)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif ok:
        print(f"VALID: {path}")
    else:
        print(f"INVALID: {path}", file=sys.stderr)
        for issue in report.errors:
            print(f"  - error[{issue.code}]: {issue.message}", file=sys.stderr)
        for issue in report.warnings:
            prefix = sys.stderr if args.strict else sys.stdout
            print(f"  - warning[{issue.code}]: {issue.message}", file=prefix)
    return 0 if ok else 1


def _validate_one(path_str: str) -> tuple[str, ValidationPayload]:
    report = MEDH5File.validate(Path(path_str))
    return path_str, report.to_dict()


def _cmd_validate_all(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    paths = [str(p) for p in iter_medh5(root)]
    if not paths:
        print(f"No .medh5 files found under {root}", file=sys.stderr)
        return 1

    results: list[tuple[str, ValidationPayload]]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            result_iter = ex.map(_validate_one, paths)
            results = []
            for item in result_iter:
                results.append(item)
                if args.fail_fast and item[1]["errors"]:
                    break
    else:
        results = []
        for p in paths:
            item = _validate_one(p)
            results.append(item)
            if args.fail_fast and item[1]["errors"]:
                break

    invalid = [(p, payload) for p, payload in results if payload["errors"]]
    print(f"Checked: {len(results)}  Invalid: {len(invalid)}")
    for p, payload in invalid:
        print(f"INVALID: {p}", file=sys.stderr)
        for err in payload["errors"]:
            print(f"  - error[{err['code']}]: {err['message']}", file=sys.stderr)
    return 1 if invalid else 0


def _verify_one(path_str: str) -> tuple[str, bool, str | None]:
    try:
        result = MEDH5File.verify(Path(path_str))
        if result is VerifyResult.MISMATCH:
            return path_str, False, "checksum mismatch"
        return path_str, True, None
    except MEDH5Error as exc:
        return path_str, False, str(exc)


def _cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    paths = [str(p) for p in iter_medh5(root)]
    if not paths:
        print(f"No .medh5 files found under {root}", file=sys.stderr)
        return 1

    results: list[tuple[str, bool, str | None]]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_verify_one, paths))
    else:
        results = [_verify_one(p) for p in paths]

    bad = [r for r in results if not r[1]]
    print(f"Checked: {len(results)}  Failed: {len(bad)}")
    for p, _, err in bad:
        msg = err or "checksum mismatch"
        print(f"FAIL: {p}  ({msg})", file=sys.stderr)
    return 1 if bad else 0


def _cmd_recompress(args: argparse.Namespace) -> int:
    src_root = Path(args.dir_or_file)
    paths = [src_root] if src_root.is_file() else list(iter_medh5(src_root))
    if not paths:
        print(f"No .medh5 files found under {src_root}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for path in paths:
        try:
            sample = MEDH5File.read(path)
            if out_dir is not None:
                dest = out_dir / path.name
                tmp_path = dest
            else:
                fd, tmp_str = tempfile.mkstemp(
                    suffix=".medh5", prefix=".tmp_recompress_", dir=str(path.parent)
                )
                # Close the OS-level fd; h5py will reopen the path itself.
                os.close(fd)
                tmp_path = Path(tmp_str)
                dest = path

            MEDH5File.write(
                tmp_path,
                images=sample.images,
                seg=sample.seg,
                bboxes=sample.bboxes,
                bbox_scores=sample.bbox_scores,
                bbox_labels=sample.bbox_labels,
                label=sample.meta.label,
                label_name=sample.meta.label_name,
                spacing=sample.meta.spatial.spacing,
                origin=sample.meta.spatial.origin,
                direction=sample.meta.spatial.direction,
                axis_labels=sample.meta.spatial.axis_labels,
                coord_system=sample.meta.spatial.coord_system,
                patch_size=sample.meta.patch_size or 192,
                extra=sample.meta.extra,
                compression=args.compression,
                checksum=args.checksum,
            )
            if args.checksum and MEDH5File.verify(tmp_path) is not VerifyResult.OK:
                print(f"FAIL: post-write checksum mismatch for {path}", file=sys.stderr)
                if out_dir is None:
                    tmp_path.unlink(missing_ok=True)
                continue
            if out_dir is None:
                shutil.move(str(tmp_path), str(dest))
            n_ok += 1
            print(f"OK: {dest}")
        except MEDH5Error as exc:
            print(f"FAIL: {path}: {exc}", file=sys.stderr)

    print(f"Recompressed: {n_ok}/{len(paths)}")
    return 0 if n_ok == len(paths) else 1


HANDLERS: dict[str, Handler] = {
    "info": _cmd_info,
    "validate": _cmd_validate,
    "validate-all": _cmd_validate_all,
    "audit": _cmd_audit,
    "recompress": _cmd_recompress,
}


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("info", help="Print metadata summary")
    p.add_argument("file", help="Path to .medh5 file")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("validate", help="Validate file structure")
    p.add_argument("file", help="Path to .medh5 file")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")

    p = sub.add_parser("validate-all", help="Validate every .medh5 under a directory")
    p.add_argument("dir", help="Directory to scan recursively")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fail-fast", action="store_true")

    p = sub.add_parser("audit", help="Verify SHA-256 checksums under a directory")
    p.add_argument("dir", help="Directory to scan recursively")
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("recompress", help="Rewrite files with a new compression preset")
    p.add_argument("dir_or_file", help="Directory or single .medh5 file")
    p.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Write recompressed copies here (default: in place via tempfile)",
    )
    p.add_argument("--checksum", action="store_true")


def dispatch(cmd: str, args: argparse.Namespace) -> int | None:
    handler = HANDLERS.get(cmd)
    return handler(args) if handler else None
