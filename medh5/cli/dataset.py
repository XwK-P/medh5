"""Manifest operations: ``index`` / ``split`` / ``stats``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medh5.cli._common import Handler


def _parse_ratios(spec: str) -> dict[str, float]:
    parts = [float(p) for p in spec.split(",")]
    if len(parts) == 2:
        return {"train": parts[0], "val": parts[1]}
    if len(parts) == 3:
        return {"train": parts[0], "val": parts[1], "test": parts[2]}
    raise SystemExit(f"--ratios must be 2 or 3 comma-separated floats, got '{spec}'")


def _cmd_index(args: argparse.Namespace) -> int:
    from medh5.dataset import Dataset

    ds = Dataset.from_directory(
        Path(args.dir), recursive=args.recursive, skip_invalid=args.skip_invalid
    )
    out = Path(args.output)
    ds.save(out)
    print(f"Indexed {len(ds)} files → {out}")
    return 0


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


HANDLERS: dict[str, Handler] = {
    "index": _cmd_index,
    "split": _cmd_split,
    "stats": _cmd_stats,
}


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("index", help="Scan a directory and write a manifest JSON")
    p.add_argument("dir", help="Directory to scan")
    p.add_argument("-o", "--output", required=True, help="Manifest output path")
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--skip-invalid", action="store_true")

    p = sub.add_parser("split", help="Split a manifest into partitions or k-fold")
    p.add_argument("manifest", help="Manifest JSON path")
    p.add_argument("--ratios", default="0.7,0.15,0.15")
    p.add_argument("--k-folds", type=int, default=None)
    p.add_argument("--stratify", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", required=True)

    p = sub.add_parser("stats", help="Compute dataset statistics")
    p.add_argument("source", help="Directory, manifest JSON, or single file")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--modality", action="append", default=None)
    p.add_argument("--foreground", default=None)
    p.add_argument("--workers", type=int, default=1)


def dispatch(cmd: str, args: argparse.Namespace) -> int | None:
    handler = HANDLERS.get(cmd)
    return handler(args) if handler else None
