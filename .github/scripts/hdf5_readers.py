"""Read what the engine writes with an older HDF5, and compare it with what
the newest one reads.

    python .github/scripts/hdf5_readers.py write SUITE OUT
    python .github/scripts/hdf5_readers.py read DIR --out READ.json
    python .github/scripts/hdf5_readers.py read DIR --hdf5 1.12 --compare READ.json

The specification promises files that HDF5 1.10 and later can read.  The engine
links HDF5 2.x, and writes with `libver_bounds(V110, latest)`, so only a reader
that old can show the promise is kept.  `write` (run where `medh5` is installed)
copies every valid case of a published conformance suite into OUT, and
re-encodes each valid, unmutated sample under the Blosc2 profiles, whose filter
the corpus does not use.  `read` (run under any h5py from 3.0 on, with
`hdf5plugin` for the filters) opens every file in DIR and reads every dataset
whole and every attribute, then records a digest of each value as read.  With
`--compare`, every value must read as it did for the run that wrote READ.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# The codec profiles of §14.2 that use the Blosc2 filter (32026).  The corpus
# is written `portable` (gzip), which needs no plugin.
PROFILES = ("training", "balanced", "archive")


def write(suite: Path, out: Path) -> int:
    manifest = json.loads((suite / "expected.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for case in manifest["cases"]:
        if not case["valid"]:
            continue
        source = suite / case["file"]
        shutil.copyfile(source, out / case["file"])
        count += 1
        # A mutated case no longer matches its own digests, so re-encoding it
        # fails verification; a later minor (W913) is never re-encoded (1.1 §2).
        if case["mutated"] or "W913" in case["expect_warnings"]:
            continue
        if case["file_suffix"] != ".medh5":
            continue
        for profile in PROFILES:
            target = out / f"{source.stem}.{profile}{source.suffix}"
            subprocess.run(
                ["medh5", "recompress", "--profile", profile, str(source)]
                + ["--out", str(target)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            count += 1
    print(f"{count} files in {out}")
    return 0


def digest(value: Any) -> str:
    """A value as read, independent of the NumPy version that holds it."""
    import h5py
    import numpy as np

    if isinstance(value, h5py.Empty):
        return f"empty {np.dtype(value.dtype).str}"
    array = np.asarray(value)
    kind = array.dtype.str if array.dtype.names is None else repr(array.dtype.descr)
    hashed = hashlib.sha256(f"{kind} {array.shape}".encode())
    if array.dtype.hasobject:
        # Variable-length strings: the text, not the objects that hold it.
        for item in array.ravel():
            data = item.encode() if isinstance(item, str) else bytes(item)
            hashed.update(len(data).to_bytes(8, "little") + data)
    else:
        hashed.update(np.ascontiguousarray(array).tobytes())
    return hashed.hexdigest()


def read_file(path: Path) -> dict[str, dict[str, Any]]:
    import h5py

    objects: dict[str, dict[str, Any]] = {}

    def visit(name: str, obj: Any) -> None:
        entry: dict[str, Any] = {
            "attrs": {key: digest(obj.attrs[key]) for key in obj.attrs}
        }
        if isinstance(obj, h5py.Dataset):
            entry["data"] = digest(obj[()])
        objects[name] = entry

    with h5py.File(path, "r") as file:
        visit("/", file)
        file.visititems(visit)
    return objects


def read(directory: Path, out: Path | None, compare: Path | None, hdf5: str) -> int:
    import h5py
    import hdf5plugin  # registers the Blosc2 and Zstandard filters
    import numpy as np

    running = h5py.version.hdf5_version
    print(
        f"h5py {h5py.__version__} on HDF5 {running}, hdf5plugin "
        f"{hdf5plugin.version}, NumPy {np.__version__}",
        flush=True,
    )
    # The version the job is about, so a wheel that bundles another one fails
    # here rather than passing as a test of what it is not.
    if hdf5 and not running.startswith(f"{hdf5}."):
        print(f"expected HDF5 {hdf5}.x, found {running}", file=sys.stderr)
        return 1
    files = sorted(p for p in directory.iterdir() if p.suffix in (".medh5", ".medh5c"))
    if not files:
        print(f"no .medh5 or .medh5c files in {directory}", file=sys.stderr)
        return 1
    found: dict[str, dict[str, dict[str, Any]]] = {}
    problems = []
    for path in files:
        try:
            found[path.name] = read_file(path)
        except Exception as error:  # every file is reported, not just the first
            problems.append(f"{path.name}: {type(error).__name__}: {error}")
    datasets = sum("data" in o for objs in found.values() for o in objs.values())
    attributes = sum(len(o["attrs"]) for objs in found.values() for o in objs.values())
    print(
        f"read {len(found)} of {len(files)} files: {datasets} datasets whole, "
        f"{attributes} attributes"
    )
    if out:
        out.write_text(json.dumps(found, indent=1, sort_keys=True), encoding="utf-8")
    if compare:
        expected = json.loads(compare.read_text(encoding="utf-8"))
        unread = {path.name for path in files} - set(found)
        for name in sorted(set(expected) | set(found)):
            if name in unread:
                continue  # a read failure, reported above
            if name not in found:
                problems.append(f"{name}: not in {directory}")
                continue
            if name not in expected:
                problems.append(f"{name}: not in {compare}")
                continue
            ours, theirs = found[name], expected[name]
            differ = sorted(
                path
                for path in set(ours) | set(theirs)
                if ours.get(path) != theirs.get(path)
            )
            if differ:
                problems.append(f"{name}: reads differently at {', '.join(differ)}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    writing = commands.add_parser("write", help="the files to read")
    writing.add_argument("suite", type=Path, help="a published conformance suite")
    writing.add_argument("out", type=Path, help="the directory to write")
    reading = commands.add_parser("read", help="read every file in a directory")
    reading.add_argument("directory", type=Path)
    reading.add_argument("--out", type=Path, help="write the digests read here")
    reading.add_argument("--compare", type=Path, help="digests every value must match")
    reading.add_argument(
        "--hdf5", default="", help="the HDF5 series h5py must run (e.g. 1.12)"
    )
    args = parser.parse_args(argv)
    if args.command == "write":
        return write(args.suite, args.out)
    return read(args.directory, args.out, args.compare, args.hdf5)


if __name__ == "__main__":
    sys.exit(main())
