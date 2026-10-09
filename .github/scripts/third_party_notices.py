"""Write THIRD_PARTY_NOTICES: the licences of what the engine compiles in.

    python .github/scripts/third_party_notices.py           # rewrite the file
    python .github/scripts/third_party_notices.py --check   # is it current?

The wheel and the `medh5` binary link everything statically: HDF5, C-Blosc2
and the codecs it calls, the HDF5-Blosc2 filter, and every Rust crate the two
depend on.  Most of those licences require their notice to travel with the
binary, so it does --- in the wheel's `.dist-info/licenses/`, beside the binary
in every release archive, and in the sdist.

The crates are Cargo.lock's, read through `cargo metadata` for the five targets
the project ships, so the file is regenerated rather than maintained: CI runs
`--check`, which fails when the checked-in file differs from what this writes.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "THIRD_PARTY_NOTICES"

# What ships: the wheel is the extension crate, the binary is the CLI.
SHIPPED = ("medh5-python", "medh5-cli")

# Where it runs: the targets CI builds and the release publishes.
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
)

# A -sys crate's own licence covers its Rust glue. The C library it compiles
# in carries its own, beside that library's sources.
NATIVE = {
    "hdf5-metno-src": ("HDF5", "ext/hdf5/LICENSE"),
    "libz-sys": ("zlib", "src/zlib/LICENSE"),
    "lz4-sys": ("LZ4", "liblz4/lib/LICENSE"),
    "zstd-sys": ("Zstandard", "zstd/LICENSE"),
}

# Compiled from this repository's tree, by crates/medh5-sys/build.rs.
VENDOR = "crates/medh5-sys/vendor"
VENDORED = (
    ("C-Blosc2", f"{VENDOR}/c-blosc2/LICENSE.txt"),
    ("BitShuffle, in C-Blosc2", f"{VENDOR}/c-blosc2/LICENSES/BITSHUFFLE.txt"),
    ("FastLZ, in C-Blosc2", f"{VENDOR}/c-blosc2/LICENSES/FASTLZ.txt"),
    ("LZ4, in C-Blosc2", f"{VENDOR}/c-blosc2/LICENSES/LZ4.txt"),
    ("zlib, in C-Blosc2", f"{VENDOR}/c-blosc2/LICENSES/ZLIB.txt"),
    ("Zstandard, in C-Blosc2", f"{VENDOR}/c-blosc2/LICENSES/ZSTD.txt"),
    ("HDF5-Blosc2 filter", f"{VENDOR}/hdf5-blosc2/LICENSE.txt"),
)
# A notice that lives only in a source file's header: C-Blosc2's pthreads
# emulation, compiled into Windows builds.
EXCERPTS = (
    (
        "pthreads for Windows, in C-Blosc2",
        f"{VENDOR}/c-blosc2/blosc/win32/threading.c",
        "Copyright (C) 2009 Andrzej K. Haczewski",
        "THE SOFTWARE.",
    ),
)

LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|notice|copyright|unlicense)", re.I)

# The permission notice of the MIT licence, for a crate whose manifest declares
# MIT (or offers it) and which ships no licence file of its own.
MIT = """\
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

HEADER = """\
THIRD-PARTY NOTICES
===================

The medh5 Python package and the `medh5` command line are built from this
repository, which is MIT-licensed (see LICENSE), and from the libraries below,
compiled in statically. Each distinct licence text is reproduced once, at the
end, and every library names the texts that apply to it.

This file is written by .github/scripts/third_party_notices.py from Cargo.lock
and the vendored sources; CI fails when it is out of date. Regenerate it rather
than editing it.
"""


def normalise(text: str) -> str:
    """One spelling per text: LF line ends, no trailing blanks or blank lines."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def read(path: Path) -> str:
    return normalise(path.read_bytes().decode("utf-8", errors="replace"))


def excerpt(path: Path, first: str, last: str) -> str:
    """The lines from the one holding `first` to the one holding `last`, with
    the comment's leading `*` removed."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, line in enumerate(lines) if first in line), None)
    if start is None:
        raise SystemExit(f"{path}: no line holds {first!r}")
    end = next((i for i in range(start, len(lines)) if last in lines[i]), None)
    if end is None:
        raise SystemExit(f"{path}: no line after {first!r} holds {last!r}")
    return normalise(
        "\n".join(re.sub(r"^\s*\*\s?", "", line) for line in lines[start : end + 1])
    )


def shipped(metadata: dict) -> list[dict]:
    """The packages the shipped crates depend on, as built: normal
    dependencies only, transitively, workspace members left out."""
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    members = set(metadata["workspace_members"])
    roots = [i for i in members if packages[i]["name"] in SHIPPED]
    if len(roots) != len(SHIPPED):
        raise SystemExit(f"the workspace does not have {', '.join(SHIPPED)}")
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for dep in nodes[node]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    return sorted(
        (packages[i] for i in seen - members), key=lambda p: (p["name"], p["version"])
    )


def offers_mit(expression: str | None) -> bool:
    """Whether an SPDX expression is MIT, or lets the recipient choose MIT
    alone.  A conjunction (`AND`, `WITH`) is never MIT alone."""
    if not expression or re.search(r"\b(AND|WITH)\b", expression):
        return False
    if expression.strip() in ("MIT", "MIT/Apache-2.0", "Apache-2.0/MIT"):
        return True
    alternatives = re.split(r"\s+OR\s+", expression.strip().strip("()"))
    return "MIT" in (a.strip().strip("()") for a in alternatives)


def crate_texts(package: dict) -> list[tuple[str, str]]:
    """(where, text) for every licence file the crate ships."""
    directory = Path(package["manifest_path"]).parent
    names = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and LICENCE_FILE.match(entry.name)
    )
    if package.get("license_file") and package["license_file"] not in names:
        names.append(package["license_file"])
    texts = [(name, read(directory / name)) for name in names]
    if texts:
        return texts
    if offers_mit(package.get("license")):
        return [("no licence file; the manifest declares MIT", MIT)]
    raise SystemExit(
        f"{package['name']} {package['version']} ships no licence file and its "
        f"licence ({package.get('license')}) is not MIT: add its text by hand"
    )


class Texts:
    """Distinct licence texts, numbered in order of first use."""

    def __init__(self) -> None:
        self.numbers: dict[str, int] = {}
        self.users: dict[int, list[str]] = {}

    def cite(self, text: str, user: str) -> int:
        number = self.numbers.setdefault(text, len(self.numbers) + 1)
        self.users.setdefault(number, []).append(user)
        return number

    def render(self) -> list[str]:
        out = []
        for text, number in self.numbers.items():
            users = textwrap.wrap(
                "; ".join(self.users[number]),
                width=79,
                initial_indent=f"[{number}] ",
                subsequent_indent="    ",
                break_on_hyphens=False,
            )
            out += ["-" * 79, *users, "-" * 79, "", text, ""]
        return out


def notices(metadata: dict, root: Path = ROOT) -> str:
    packages = shipped(metadata)
    texts = Texts()
    native: list[str] = []
    for name, (library, relative) in sorted(NATIVE.items()):
        found = [p for p in packages if p["name"] == name]
        if len(found) != 1:
            raise SystemExit(
                f"expected one {name} among the shipped crates, found {len(found)}"
            )
        package = found[0]
        path = Path(package["manifest_path"]).parent / relative
        if not path.is_file():
            raise SystemExit(f"{name} {package['version']} has no {relative}")
        crate = f"{name} {package['version']}"
        number = texts.cite(read(path), f"{library} ({crate}: {relative})")
        native.append(f"  {library:<36} {crate:<30} [{number}]")
    for library, relative in VENDORED:
        number = texts.cite(read(root / relative), f"{library} ({relative})")
        native.append(f"  {library:<36} {'vendored':<30} [{number}]")
    for library, relative, first, last in EXCERPTS:
        number = texts.cite(
            excerpt(root / relative, first, last), f"{library} ({relative})"
        )
        native.append(f"  {library:<36} {'vendored':<30} [{number}]")

    crates = []
    for package in packages:
        crate = f"{package['name']} {package['version']}"
        cited = [
            texts.cite(text, f"{crate}: {where}")
            for where, text in crate_texts(package)
        ]
        refs = ", ".join(str(n) for n in sorted(set(cited)))
        licence = package.get("license") or "see the text"
        crates.append(f"  {crate:<44} {licence:<32} [{refs}]")

    lines = [HEADER, "", "C libraries", "-----------", "", *native, "", ""]
    lines += [
        f"Rust crates ({len(packages)})",
        "-" * len(f"Rust crates ({len(packages)})"),
        "",
    ]
    lines += [*crates, "", "", "Licence texts", "-------------", "", *texts.render()]
    return "\n".join(lines).rstrip("\n") + "\n"


def metadata() -> dict:
    command = ["cargo", "metadata", "--format-version", "1", "--locked"]
    for target in TARGETS:
        command += ["--filter-platform", target]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the file is out of date"
    )
    args = parser.parse_args(argv)
    text = notices(metadata())
    if not args.check:
        OUT.write_text(text, encoding="utf-8", newline="\n")
        return 0
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if current == text:
        print(f"{OUT.name} is current")
        return 0
    diff = difflib.unified_diff(
        current.splitlines(),
        text.splitlines(),
        "checked in",
        "generated",
        lineterm="",
        n=1,
    )
    print("\n".join(list(diff)[:80]))
    script = ".github/scripts/third_party_notices.py"
    print(f"\n{OUT.name} is out of date: run `python {script}`", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
