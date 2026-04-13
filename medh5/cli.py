"""Command-line interface for medh5.

Subcommands:

- ``info``           — print metadata summary for one file
- ``validate``       — validate one file's structure
- ``validate-all``   — validate every ``.medh5`` under a directory
- ``audit``          — verify SHA-256 checksums under a directory
- ``recompress``     — rewrite files with a different compression preset
- ``index``          — scan a directory and write a manifest JSON
- ``split``          — split a manifest into train/val/test partitions
- ``stats``          — compute dataset-level statistics
- ``import nifti``   — convert NIfTI volumes → ``.medh5``
- ``import dicom``   — convert a DICOM series directory → ``.medh5``
- ``export nifti``   — export ``.medh5`` images/masks → NIfTI
- ``review set``     — record a review/QA status entry
- ``review get``     — print the current review status
- ``review list``    — list files in a directory matching a status filter
- ``review import-seg`` — import a (possibly edited) NIfTI mask back into a file
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

_Handler = Callable[[argparse.Namespace], int]

import h5py

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5Error

ValidationPayload = dict[str, Any]

# ---------------------------------------------------------------------------
# info / validate (single file)
# ---------------------------------------------------------------------------


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


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a .medh5 file structure and schema."""
    path = Path(args.file)
    report = MEDH5File.validate(path, strict=args.strict)
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


# ---------------------------------------------------------------------------
# Batch operations: validate-all / audit / recompress
# ---------------------------------------------------------------------------


def _iter_medh5(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.medh5"))


def _validate_one(path_str: str) -> tuple[str, ValidationPayload]:
    report = MEDH5File.validate(Path(path_str))
    return path_str, report.to_dict()


def _cmd_validate_all(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    paths = [str(p) for p in _iter_medh5(root)]
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
        ok = MEDH5File.verify(Path(path_str))
        return path_str, ok, None
    except MEDH5Error as exc:
        return path_str, False, str(exc)


def _cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    paths = [str(p) for p in _iter_medh5(root)]
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
    paths = [src_root] if src_root.is_file() else list(_iter_medh5(src_root))
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
                import os

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
            if args.checksum and not MEDH5File.verify(tmp_path):
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


# ---------------------------------------------------------------------------
# index / split / stats
# ---------------------------------------------------------------------------


def _cmd_index(args: argparse.Namespace) -> int:
    from medh5.dataset import Dataset

    ds = Dataset.from_directory(
        Path(args.dir), recursive=args.recursive, skip_invalid=args.skip_invalid
    )
    out = Path(args.output)
    ds.save(out)
    print(f"Indexed {len(ds)} files → {out}")
    return 0


def _parse_ratios(spec: str) -> dict[str, float]:
    parts = [float(p) for p in spec.split(",")]
    if len(parts) == 2:
        return {"train": parts[0], "val": parts[1]}
    if len(parts) == 3:
        return {"train": parts[0], "val": parts[1], "test": parts[2]}
    raise SystemExit(f"--ratios must be 2 or 3 comma-separated floats, got '{spec}'")


def _cmd_split(args: argparse.Namespace) -> int:
    from medh5.dataset import Dataset, make_splits

    ds = Dataset.load(Path(args.manifest))
    if args.k_folds:
        result = make_splits(
            ds,
            k_folds=args.k_folds,
            stratify_by=args.stratify,
            group_by=args.group,
            seed=args.seed,
        )
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        assert isinstance(result, list)
        for i, fm in enumerate(result):
            for name, partition in fm.items():
                partition.save(out_dir / f"fold{i}_{name}.json")
        print(f"Wrote {len(result)} folds → {out_dir}")
        return 0

    ratios = _parse_ratios(args.ratios)
    splits = make_splits(
        ds,
        ratios=ratios,
        stratify_by=args.stratify,
        group_by=args.group,
        seed=args.seed,
    )
    assert isinstance(splits, dict)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in splits.items():
        partition.save(out_dir / f"{name}.json")
        print(f"  {name}: {len(partition)} → {out_dir / f'{name}.json'}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from medh5.dataset import Dataset
    from medh5.stats import compute_stats

    src_path = Path(args.source)
    if src_path.is_dir():
        ds = Dataset.from_directory(src_path)
    elif src_path.suffix == ".json":
        ds = Dataset.load(src_path)
    else:
        ds = Dataset.from_paths([src_path])

    stats = compute_stats(
        ds,
        modalities=args.modality,
        foreground_mask=args.foreground,
        workers=args.workers,
    )
    payload = stats.to_dict()
    if args.output:
        stats.save(Path(args.output))
        print(f"Wrote stats → {args.output}")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------------
# import / export (NIfTI / DICOM)
# ---------------------------------------------------------------------------


def _cmd_import_nifti(args: argparse.Namespace) -> int:
    from medh5.io import from_nifti

    if not args.image:
        print("--image NAME PATH is required (one or more)", file=sys.stderr)
        return 2
    images: dict[str, str | Path] = {name: Path(path) for name, path in args.image}
    seg_d: dict[str, str | Path] = {name: Path(path) for name, path in (args.seg or [])}
    from_nifti(
        images=images,
        seg=seg_d or None,
        out_path=Path(args.output),
        label=args.label,
        label_name=args.label_name,
        compression=args.compression,
        checksum=args.checksum,
        resample_to=args.resample_to,
        interpolator=args.interpolator,
    )
    print(f"Wrote {args.output}")
    return 0


def _cmd_import_dicom(args: argparse.Namespace) -> int:
    from medh5.io import from_dicom

    from_dicom(
        Path(args.dicom_dir),
        Path(args.output),
        modality_name=args.modality,
        series_uid=args.series_uid,
        apply_modality_lut=not args.no_modality_lut,
        compression=args.compression,
        checksum=args.checksum,
    )
    print(f"Wrote {args.output}")
    return 0


def _cmd_export_nifti(args: argparse.Namespace) -> int:
    from medh5.io import to_nifti

    written = to_nifti(
        Path(args.file),
        Path(args.output),
        modalities=args.modalities,
        seg=args.seg,
    )
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


# ---------------------------------------------------------------------------
# review subcommands
# ---------------------------------------------------------------------------


def _cmd_review_set(args: argparse.Namespace) -> int:
    MEDH5File.set_review_status(
        Path(args.file),
        status=args.status,
        annotator=args.annotator,
        notes=args.notes,
    )
    print(f"OK: {args.file} → {args.status}")
    return 0


def _cmd_review_get(args: argparse.Namespace) -> int:
    st = MEDH5File.get_review_status(Path(args.file))
    if args.json:
        print(
            json.dumps(
                {
                    "status": st.status,
                    "annotator": st.annotator,
                    "timestamp": st.timestamp,
                    "notes": st.notes,
                    "history": st.history,
                },
                indent=2,
            )
        )
        return 0
    print(f"status:    {st.status}")
    print(f"annotator: {st.annotator}")
    print(f"timestamp: {st.timestamp}")
    print(f"notes:     {st.notes}")
    if st.history:
        n = len(st.history)
        word = "entry" if n == 1 else "entries"
        print(f"history:   {n} prior {word}")
    return 0


def _cmd_review_list(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    matches: list[Path] = []
    for path in _iter_medh5(root):
        try:
            st = MEDH5File.get_review_status(path)
        except MEDH5Error:
            continue
        if args.status is None or st.status == args.status:
            matches.append(path)
    for p in matches:
        print(p)
    return 0


def _cmd_review_import_seg(args: argparse.Namespace) -> int:
    from medh5.io import import_seg_nifti

    import_seg_nifti(
        Path(args.file),
        Path(args.from_nifti),
        name=args.name,
        resample=args.resample,
        replace=args.replace,
    )
    print(f"OK: added mask '{args.name}' to {args.file}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medh5", description="medh5 file utility")
    sub = parser.add_subparsers(dest="command")

    # info
    p = sub.add_parser("info", help="Print metadata summary")
    p.add_argument("file", help="Path to .medh5 file")
    p.add_argument("--json", action="store_true")

    # validate
    p = sub.add_parser("validate", help="Validate file structure")
    p.add_argument("file", help="Path to .medh5 file")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")

    # validate-all
    p = sub.add_parser("validate-all", help="Validate every .medh5 under a directory")
    p.add_argument("dir", help="Directory to scan recursively")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fail-fast", action="store_true")

    # audit
    p = sub.add_parser("audit", help="Verify SHA-256 checksums under a directory")
    p.add_argument("dir", help="Directory to scan recursively")
    p.add_argument("--workers", type=int, default=1)

    # recompress
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

    # index
    p = sub.add_parser("index", help="Scan a directory and write a manifest JSON")
    p.add_argument("dir", help="Directory to scan")
    p.add_argument("-o", "--output", required=True, help="Manifest output path")
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--skip-invalid", action="store_true")

    # split
    p = sub.add_parser("split", help="Split a manifest into partitions or k-fold")
    p.add_argument("manifest", help="Manifest JSON path")
    p.add_argument("--ratios", default="0.7,0.15,0.15")
    p.add_argument("--k-folds", type=int, default=None)
    p.add_argument("--stratify", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", required=True)

    # stats
    p = sub.add_parser("stats", help="Compute dataset statistics")
    p.add_argument("source", help="Directory, manifest JSON, or single file")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--modality", action="append", default=None)
    p.add_argument("--foreground", default=None)
    p.add_argument("--workers", type=int, default=1)

    # import (subgroup)
    p = sub.add_parser("import", help="Import external formats into .medh5")
    isub = p.add_subparsers(dest="import_command")

    pn = isub.add_parser("nifti", help="Import NIfTI volumes")
    pn.add_argument(
        "--image",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Modality name and NIfTI path (repeatable)",
    )
    pn.add_argument(
        "--seg",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Segmentation name and NIfTI path (repeatable)",
    )
    pn.add_argument("-o", "--output", required=True)
    pn.add_argument("--label", default=None)
    pn.add_argument("--label-name", default=None)
    pn.add_argument(
        "--resample-to",
        default=None,
        help="Reference modality name or NIfTI path for SimpleITK resampling",
    )
    pn.add_argument(
        "--interpolator",
        default="linear",
        choices=["linear", "nearest", "bspline"],
    )
    pn.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    pn.add_argument("--checksum", action="store_true")

    pd = isub.add_parser("dicom", help="Import a DICOM series directory")
    pd.add_argument("dicom_dir")
    pd.add_argument("-o", "--output", required=True)
    pd.add_argument("--modality", default="CT")
    pd.add_argument("--series-uid", default=None)
    pd.add_argument("--no-modality-lut", action="store_true")
    pd.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    pd.add_argument("--checksum", action="store_true")

    # export (subgroup)
    p = sub.add_parser("export", help="Export .medh5 to external formats")
    esub = p.add_subparsers(dest="export_command")
    pe = esub.add_parser("nifti", help="Export images and masks as NIfTI")
    pe.add_argument("file", help="Source .medh5 file")
    pe.add_argument("-o", "--output", required=True, help="Output directory")
    pe.add_argument("--modalities", nargs="*", default=None)
    pe.add_argument("--seg", nargs="*", default=None)

    # review (subgroup)
    p = sub.add_parser("review", help="Review/QA workflow helpers")
    rsub = p.add_subparsers(dest="review_command")

    rs = rsub.add_parser("set", help="Record a review status entry")
    rs.add_argument("file")
    rs.add_argument(
        "--status",
        required=True,
        choices=["pending", "reviewed", "flagged", "rejected"],
    )
    rs.add_argument("--annotator", default=None)
    rs.add_argument("--notes", default=None)

    rg = rsub.add_parser("get", help="Print the current review status")
    rg.add_argument("file")
    rg.add_argument("--json", action="store_true")

    rl = rsub.add_parser("list", help="List files matching a status filter")
    rl.add_argument("dir")
    rl.add_argument("--status", default=None)

    ri = rsub.add_parser("import-seg", help="Import a (NIfTI) mask back into a file")
    ri.add_argument("file")
    ri.add_argument("--name", required=True)
    ri.add_argument("--from", dest="from_nifti", required=True)
    ri.add_argument("--resample", action="store_true")
    ri.add_argument("--replace", action="store_true")

    return parser


_TOP_HANDLERS: dict[str, _Handler] = {
    "info": _cmd_info,
    "validate": _cmd_validate,
    "validate-all": _cmd_validate_all,
    "audit": _cmd_audit,
    "recompress": _cmd_recompress,
    "index": _cmd_index,
    "split": _cmd_split,
    "stats": _cmd_stats,
}

_IMPORT_HANDLERS: dict[str, _Handler] = {
    "nifti": _cmd_import_nifti,
    "dicom": _cmd_import_dicom,
}

_EXPORT_HANDLERS: dict[str, _Handler] = {
    "nifti": _cmd_export_nifti,
}

_REVIEW_HANDLERS: dict[str, _Handler] = {
    "set": _cmd_review_set,
    "get": _cmd_review_get,
    "list": _cmd_review_list,
    "import-seg": _cmd_review_import_seg,
}

_SUB_DISPATCH: dict[str, tuple[str, dict[str, _Handler]]] = {
    "import": ("import_command", _IMPORT_HANDLERS),
    "export": ("export_command", _EXPORT_HANDLERS),
    "review": ("review_command", _REVIEW_HANDLERS),
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``medh5`` CLI.

    Exit codes follow the usual Unix convention so shell automation
    (``medh5 validate … || exit 1``) works:

    - ``0`` — command ran successfully.
    - ``1`` — a handler raised a known runtime error.
    - ``2`` — no command given, or an unknown subcommand.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        if cmd in _TOP_HANDLERS:
            return _TOP_HANDLERS[cmd](args)

        if cmd in _SUB_DISPATCH:
            attr, table = _SUB_DISPATCH[cmd]
            sub = getattr(args, attr, None)
            if sub in table:
                return table[sub](args)
            print(
                f"medh5 {cmd}: missing subcommand "
                f"(choose from {', '.join(sorted(table))})",
                file=sys.stderr,
            )
            return 2
    except (ImportError, MEDH5Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
