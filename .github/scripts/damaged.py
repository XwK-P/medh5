"""Damage files and validate them with a `medh5` binary: every run must end in
a report, never in an abort.

    python .github/scripts/damaged.py MEDH5 CORPUS [--runs N]

HDF5 compiled without NDEBUG keeps its assertions, and a damaged file trips one
(`H5A__attr_release_table`, for one) and aborts the process where every other
build reports the damage.  The Windows build did, until MSVC builds were given
NDEBUG.  `medh5 validate --json` exits 0 or 1 with a JSON report whatever the
file holds; anything else --- an abort, a panic, a hang, no report --- fails
here.  The damage is seeded, so a failure names a file anyone can rebuild.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("medh5", help="the binary")
    parser.add_argument(
        "corpus", type=Path, help="a directory of .medh5 files to damage"
    )
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    # A path to the binary, made absolute so the child is found whatever the
    # platform resolves a relative one against; a bare name is looked up on PATH.
    medh5 = (
        str(Path(args.medh5).resolve()) if Path(args.medh5).is_file() else args.medh5
    )
    sources = sorted(args.corpus.rglob("*.medh5"))
    if not sources:
        parser.error(f"no .medh5 files under {args.corpus}")
    rng = random.Random(args.seed)
    failures = []
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
    print(f"{args.runs} damaged files, {len(failures)} not reported")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
