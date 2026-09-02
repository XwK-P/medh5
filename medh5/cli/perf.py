"""``recompress`` and ``bench`` --- the storage and performance commands (§14)."""

from __future__ import annotations

import argparse
import tempfile
from typing import Any

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
from medh5.storage.codecs import PROFILES


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    recompress = sub.add_parser(
        "recompress", help="re-encode bulk data under another codec profile (§14.2)"
    )
    add_paths(recompress)
    recompress.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILES),
        help="codec profile: training, balanced, archive or portable",
    )
    recompress.add_argument(
        "--rechunk",
        action="store_true",
        help="also re-derive chunk shapes; off by default, since a codec change "
        "is not an access-pattern change",
    )
    recompress.add_argument(
        "-o",
        "--out",
        help="write beside the source instead of replacing it (single input only)",
    )
    add_json_flag(recompress)

    bench = sub.add_parser(
        "bench", help="reproduce the performance targets (plan §4.3)"
    )
    bench.add_argument(
        "path",
        nargs="?",
        help="a sample to measure; omitted, a synthetic one is written and used",
    )
    bench.add_argument("--annotation", help="annotation to time label reads against")
    bench.add_argument(
        "--patch", type=int, default=64, help="patch side length to measure with"
    )
    bench.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="repetitions per measurement; the median is reported",
    )
    bench.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader workers for the throughput run",
    )
    bench.add_argument(
        "--no-throughput",
        action="store_true",
        help="skip the dataloader run (it needs PyTorch)",
    )
    add_json_flag(bench)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "recompress":
        return _recompress(args)
    if command == "bench":
        return _bench(args)
    return None


def _recompress(args: argparse.Namespace) -> int:
    from medh5.storage.recompress import recompress

    if args.out and len(args.paths) > 1:
        return fail("--out takes a single input file")
    results = []
    try:
        for path in args.paths:
            results.append(
                recompress(path, args.profile, out=args.out, rechunk=args.rechunk)
            )
    except MEDH5Error as exc:
        return fail(str(exc))
    if args.json:
        emit([r.to_json() for r in results], as_json=True)
        return EXIT_OK if all(r.content_id_preserved for r in results) else EXIT_ERROR
    print(
        table(
            [
                [
                    r.path,
                    r.profile,
                    r.datasets,
                    human_bytes(r.bytes_before),
                    human_bytes(r.bytes_after),
                    f"{r.ratio:.2f}x",
                    "yes" if r.content_id_preserved else "CHANGED",
                ]
                for r in results
            ],
            ["path", "profile", "sets", "before", "after", "ratio", "content_id"],
        )
    )
    print(
        "\ndigests cover decompressed content (§13.1), so `content_id` is "
        "unchanged by re-encoding; a cache keyed on it stays valid."
    )
    return EXIT_OK if all(r.content_id_preserved for r in results) else EXIT_ERROR


def _bench(args: argparse.Namespace) -> int:
    from medh5.bench import benchmark_file, synthetic_sample, throughput

    temporary: tempfile.TemporaryDirectory[str] | None = None
    path = args.path
    try:
        if path is None:
            temporary = tempfile.TemporaryDirectory(prefix="medh5-bench-")
            print("writing a synthetic 192x256x256 sample ...", flush=True)
            path = str(synthetic_sample(temporary.name))
        measurements = benchmark_file(
            path,
            annotation=args.annotation,
            patch=args.patch,
            repeats=args.repeats,
        )
        if not args.no_throughput:
            measured = _throughput(throughput, path, args)
            if measured is not None:
                measurements.append(measured)
        payload: dict[str, Any] = {
            "path": str(path),
            "measurements": [m.to_json() for m in measurements],
            "ok": all(m.ok for m in measurements),
        }
        if args.json:
            emit(payload, as_json=True)
        else:
            from medh5.bench import report

            print(f"\n{path}")
            print(report(measurements))
        return EXIT_OK if payload["ok"] else EXIT_ERROR
    except MEDH5Error as exc:
        return fail(str(exc))
    finally:
        if temporary is not None:
            temporary.cleanup()


def _throughput(fn: Any, path: str, args: argparse.Namespace) -> Any:
    """Run the dataloader benchmark, or say why it was skipped."""
    try:
        return fn(
            [path],
            patch=args.patch,
            workers=args.workers,
            annotation=args.annotation,
        )
    except ImportError as exc:
        print(f"skipping throughput: {exc}")
        return None


__all__ = ["dispatch", "register"]
