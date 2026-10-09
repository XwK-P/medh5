"""The publishing steps of `release.yml`, each safe to run again.

    python .github/scripts/release.py crates
    python .github/scripts/release.py pypi DIST
    python .github/scripts/release.py github-release TAG DIST... [--prerelease]
    python .github/scripts/release.py verify TAG DIST CLI [--tap OWNER/REPO]

A release publishes to four places in turn --- PyPI, crates.io, the GitHub
Release, the Homebrew tap --- and none of them takes a version back.  When one
step fails after another succeeded, re-running the workflow must finish the
job rather than trip over what is already there, and must not paper over a
version that was published from something else.  So each step skips only what
is already published *and identical* to what this run built, publishes what is
missing, and fails on anything that differs; `verify` checks every place at the
end.  CONTRIBUTING.md ("If a release fails partway") is the runbook.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Dependency order: each needs the one before it on crates.io.  medh5-python
# is `publish = false`.
CRATES = ("medh5-sys", "medh5", "medh5-cli")

# What `cargo package` writes rather than copies.  Its spelling follows the
# cargo version (the normalised manifest, the lockfile), so a re-run with a
# newer toolchain would differ on these alone; everything else is the source.
GENERATED = frozenset({"Cargo.toml", "Cargo.lock"})

AGENT = "medh5-release (https://github.com/XwK-P/medh5)"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- what is compared --------------------------------------------------------


def crate_files(data: bytes) -> dict[str, str]:
    """A `.crate`'s files, by path inside the package, as SHA-256 digests;
    cargo's generated files left out."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if path in GENERATED:
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            out[path] = sha256(handle.read())
    return out


def differences(ours: dict[str, str], theirs: dict[str, str]) -> list[str]:
    """What differs between two listings of path -> digest."""
    problems = []
    for path in sorted(ours.keys() | theirs.keys()):
        if path not in theirs:
            problems.append(f"{path}: not in the published one")
        elif path not in ours:
            problems.append(f"{path}: only in the published one")
        elif ours[path] != theirs[path]:
            problems.append(f"{path}: differs")
    return problems


def dist_problems(local: dict[str, str], published: dict[str, str]) -> list[str]:
    """Every local file must be published with the same digest."""
    problems = []
    for name, digest in sorted(local.items()):
        if name not in published:
            problems.append(f"{name}: not published")
        elif published[name] != digest:
            problems.append(f"{name}: published with different content")
    return problems


def release_plan(
    local: dict[str, str], remote: dict[str, str | None]
) -> tuple[list[str], list[str]]:
    """(to upload, to compare by downloading) for a release that exists: a
    missing asset is uploaded; one whose digest GitHub does not report is
    downloaded and compared."""
    upload = sorted(name for name in local if name not in remote)
    unknown = sorted(name for name in local if name in remote and remote[name] is None)
    return upload, unknown


def formula_version(text: str) -> str | None:
    match = re.search(r'^\s*version\s+"([^"]+)"', text, re.M)
    return match.group(1) if match else None


def digests(directory: Path) -> dict[str, str]:
    return {
        p.name: sha256(p.read_bytes())
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def dist_version(names: list[str]) -> str:
    """The one version every distribution file carries (PEP 440 spelling)."""
    versions = {name.split("-")[1].removesuffix(".tar.gz") for name in names}
    if len(versions) != 1:
        raise SystemExit(
            f"the distributions carry {len(versions)} versions: {sorted(versions)}"
        )
    return versions.pop()


# --- the network --------------------------------------------------------------


def fetch(url: str, *, token: str | None = None, attempts: int = 4) -> bytes | None:
    """The body at `url`, or None for a 404.  Other failures retry, then raise."""
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def run(*command: str, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=capture
    )
    return result.stdout if capture else ""


def workspace_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = text.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    if not match:
        raise SystemExit("no [workspace.package] version")
    return match.group(1)


def on_crates_io(name: str, version: str) -> bool:
    body = fetch(f"https://crates.io/api/v1/crates/{name}/{version}")
    return body is not None and not json.loads(body)["version"]["yanked"]


def published_crate(name: str, version: str) -> bytes:
    body = fetch(f"https://static.crates.io/crates/{name}/{name}-{version}.crate")
    if body is None:
        raise SystemExit(
            f"crates.io lists {name} {version} but serves no .crate for it"
        )
    return body


def pypi_digests(version: str) -> dict[str, str]:
    body = fetch(f"https://pypi.org/pypi/medh5/{version}/json")
    if body is None:
        return {}
    return {
        entry["filename"]: entry["digests"]["sha256"]
        for entry in json.loads(body)["urls"]
    }


# --- the steps ------------------------------------------------------------------


def publish_crates() -> int:
    version = workspace_version()
    for name in CRATES:
        if not on_crates_io(name, version):
            # Packages, verifies, uploads, and waits until the index has it,
            # which the next crate's resolution needs.
            run("cargo", "publish", "-p", name, "--locked")
            continue
        run("cargo", "package", "-p", name, "--locked", "--no-verify")
        local = (ROOT / "target" / "package" / f"{name}-{version}.crate").read_bytes()
        problems = differences(
            crate_files(local), crate_files(published_crate(name, version))
        )
        if problems:
            raise SystemExit(
                f"{name} {version} is on crates.io with other content, and a version "
                "cannot be replaced; publish under a new version:\n  "
                + "\n  ".join(problems)
            )
        print(f"{name} {version} is already on crates.io, identical: skipped")
    return 0


def check_pypi(dist: Path, *, wait: int = 600) -> int:
    """Every file in `dist` is on PyPI with the same digest.  A fresh upload
    takes a moment to appear in the JSON API, so missing files are retried."""
    local = digests(dist)
    version = dist_version(list(local))
    deadline = time.monotonic() + wait
    while True:
        problems = dist_problems(local, pypi_digests(version))
        changed = [p for p in problems if "different content" in p]
        if not problems or changed or time.monotonic() > deadline:
            break
        time.sleep(30)
    if problems:
        raise SystemExit(
            f"PyPI does not have this run's medh5 {version}:\n  "
            + "\n  ".join(problems)
            + "\n(a file published with different content means an earlier run built "
            "this version: re-run only the failed jobs, which reuses that run's "
            "artifacts, or tag a new version)"
        )
    print(f"PyPI has all {len(local)} files of medh5 {version}, identical")
    return 0


def github_release(
    tag: str, files: list[Path], *, prerelease: bool, notes: Path
) -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    local = {path.name: sha256(path.read_bytes()) for path in files}
    view = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if view.returncode != 0:
        flags = ["--prerelease"] if prerelease else []
        run(
            "gh", "release", "create", tag, *map(str, files), "--repo", repository,
            "--title", f"medh5 {tag.removeprefix('v')}", "--notes-file", str(notes),
            "--verify-tag", *flags,
        )  # fmt: skip
        return 0
    assets = json.loads(view.stdout)["assets"]
    remote: dict[str, str | None] = {
        a["name"]: (a.get("digest") or "").removeprefix("sha256:") or None
        for a in assets
    }
    upload, unknown = release_plan(local, remote)
    with tempfile.TemporaryDirectory() as scratch:
        for name in unknown:
            run(
                "gh",
                "release",
                "download",
                tag,
                "--repo",
                repository,
                "-p",
                name,
                "-D",
                scratch,
            )
            remote[name] = sha256((Path(scratch) / name).read_bytes())
    changed = sorted(
        n for n in local if n in remote and remote[n] != local[n] and n not in upload
    )
    if changed:
        raise SystemExit(
            f"the {tag} release has other files under these names: {changed}"
        )
    by_name = {path.name: path for path in files}
    if upload:
        run(
            "gh",
            "release",
            "upload",
            tag,
            *(str(by_name[n]) for n in upload),
            "--repo",
            repository,
        )
    print(f"the {tag} release has all {len(local)} files ({len(upload)} uploaded now)")
    return 0


def verify(
    tag: str, dist: Path, cli: Path, *, tap: str | None, token: str | None
) -> int:
    """Every place has this version: PyPI every file of `dist`, crates.io every
    crate, the release page every file of `dist` and `cli`, and the tap (when
    named) a formula for it."""
    version = tag.removeprefix("v")
    problems = []
    local = digests(dist)
    problems += [
        f"PyPI: {p}"
        for p in dist_problems(local, pypi_digests(dist_version(list(local))))
    ]
    problems += [
        f"crates.io: {n} {version} is not published"
        for n in CRATES
        if not on_crates_io(n, version)
    ]
    repository = os.environ["GITHUB_REPOSITORY"]
    release = fetch(
        f"https://api.github.com/repos/{repository}/releases/tags/{tag}", token=token
    )
    if release is None:
        problems.append(f"GitHub: no release for {tag}")
    else:
        attached = {asset["name"] for asset in json.loads(release)["assets"]}
        expected = {*local, *digests(cli)}
        problems += [
            f"GitHub: the release lacks {name}" for name in sorted(expected - attached)
        ]
    if tap:
        formula = fetch(
            f"https://api.github.com/repos/{tap}/contents/Formula/medh5.rb",
            token=os.environ.get("TAP_TOKEN") or token,
        )
        text = ""
        if formula is not None:
            text = base64.b64decode(json.loads(formula)["content"]).decode("utf-8")
        if formula_version(text) != version:
            problems.append(
                f"Homebrew: {tap} has medh5 {formula_version(text)}, not {version}"
            )
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        f"medh5 {version} is on PyPI, crates.io, the release page"
        + (" and Homebrew" if tap else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("crates")
    pypi = sub.add_parser("pypi")
    pypi.add_argument("dist", type=Path)
    release = sub.add_parser("github-release")
    release.add_argument("tag")
    release.add_argument("files", type=Path, nargs="+")
    release.add_argument("--notes", type=Path, default=Path("release-notes.md"))
    release.add_argument("--prerelease", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("tag")
    check.add_argument("dist", type=Path)
    check.add_argument("cli", type=Path)
    check.add_argument("--tap")
    args = parser.parse_args(argv)
    if args.step == "crates":
        return publish_crates()
    if args.step == "pypi":
        return check_pypi(args.dist)
    if args.step == "github-release":
        return github_release(
            args.tag, args.files, prerelease=args.prerelease, notes=args.notes
        )
    return verify(
        args.tag, args.dist, args.cli, tap=args.tap, token=os.environ.get("GH_TOKEN")
    )


if __name__ == "__main__":
    sys.exit(main())
