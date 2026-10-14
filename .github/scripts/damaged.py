"""Damage files and validate them with a `medh5` binary: every run must end in
a report, never in an abort.

    python .github/scripts/damaged.py MEDH5 CORPUS [--runs N] [--structural DIR]

HDF5 compiled without NDEBUG keeps its assertions, and a damaged file trips one
(`H5A__attr_release_table`, for one) and aborts the process where every other
build reports the damage.  The Windows build did, until MSVC builds were given
NDEBUG.  `medh5 validate --json` exits 0 or 1 with a JSON report whatever the
file holds; anything else --- an abort, a panic, a hang, no report --- fails
here.  The damage is seeded, so a failure names a file anyone can rebuild.

Random bytes rarely make the shapes 2.0.2's denial-of-service bugs needed: a
chain of groups thousands deep, a filter nobody registered, a chunk that
decodes to gigabytes.  `--structural DIR` reads those, as the binary writes
them (`medh5 conformance build --robustness DIR`), each validated at the
strictest level: each must be reported as an error (exit 1, a JSON report
holding one) within a time limit and, on POSIX, a bound on the memory the
binary reached.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import resource
except ImportError:  # Windows, which bounds no child's memory here
    resource = None  # type: ignore[assignment]

# What a structural file may take: well past what reporting it needs, and far
# short of what each bug took (a 64 KiB frame decoded to 2 GiB).
STRUCTURAL_SECONDS = 60
STRUCTURAL_RSS_BYTES = 1 << 30


def damage(data: bytes, rng: random.Random) -> bytes:
    """One to six bytes overwritten at random offsets."""
    damaged = bytearray(data)
    for _ in range(rng.randint(1, 6)):
        damaged[rng.randrange(len(damaged))] = rng.randrange(256)
    return bytes(damaged)


def outcome(medh5: str, path: Path) -> str | None:
    """None when the binary reported on `path`, else what went wrong."""
    try:
        run = subprocess.run(
            [medh5, "validate", str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "did not finish in 120 s"
    if run.returncode not in (0, 1):
        tail = run.stderr.strip().splitlines()[-3:]
        return f"exit {run.returncode}: {' | '.join(tail)}"
    try:
        json.loads(run.stdout)
    except json.JSONDecodeError:
        return f"exit {run.returncode} with no JSON report"
    return None


def peak_rss() -> int | None:
    """The most memory any child waited for reached, in bytes: kibibytes on
    Linux, bytes on macOS; None where nothing measures it."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def structural(medh5: str, path: Path) -> str | None:
    """None when the binary reported `path` as an error within the bounds,
    else what went wrong."""
    try:
        run = subprocess.run(
            [medh5, "validate", str(path), "--level", "strict", "--json"],
            capture_output=True,
            text=True,
            timeout=STRUCTURAL_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"did not finish in {STRUCTURAL_SECONDS} s"
    peak = peak_rss()
    if peak is not None and peak > STRUCTURAL_RSS_BYTES:
        return f"reached {peak >> 20} MiB, more than {STRUCTURAL_RSS_BYTES >> 20} MiB"
    if run.returncode != 1:
        tail = run.stderr.strip().splitlines()[-3:]
        return f"exit {run.returncode}, not 1: {' | '.join(tail)}"
    try:
        reports = json.loads(run.stdout)
    except json.JSONDecodeError:
        return "no JSON report"
    if not any(report.get("errors") for report in reports):
        return "reported no error"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("medh5", help="the binary")
    parser.add_argument(
        "corpus", type=Path, help="a directory of .medh5 files to damage"
    )
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--structural",
        type=Path,
        metavar="DIR",
        help="hostile files to report within bounds (conformance build --robustness)",
    )
    args = parser.parse_args(argv)

    # A path to the binary, made absolute so the child is found whatever the
    # platform resolves a relative one against; a bare name is looked up on PATH.
    medh5 = (
        str(Path(args.medh5).resolve()) if Path(args.medh5).is_file() else args.medh5
    )
    sources = sorted(args.corpus.rglob("*.medh5"))
    if not sources:
        parser.error(f"no .medh5 files under {args.corpus}")
    failures = []
    # First, while the children's peak is theirs alone.
    if args.structural is not None:
        hostile = sorted(args.structural.glob("*.medh5"))
        if not hostile:
            parser.error(f"no .medh5 files under {args.structural}")
        for path in hostile:
            problem = structural(medh5, path)
            if problem:
                failures.append(f"{path.name}: {problem}")
        print(f"{len(hostile)} structural files, {len(failures)} not reported")
    unreported = len(failures)
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as scratch:
        for run in range(args.runs):
            source = sources[run % len(sources)]
            victim = Path(scratch) / f"damaged-{run}.medh5"
            victim.write_bytes(damage(source.read_bytes(), rng))
            problem = outcome(medh5, victim)
            if problem:
                failures.append(
                    f"run {run} (seed {args.seed}, {source.name}): {problem}"
                )
            victim.unlink()
    print(f"{args.runs} damaged files, {len(failures) - unreported} not reported")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
