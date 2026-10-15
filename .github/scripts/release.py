"""The publishing steps of `release.yml`, each safe to run again.

    python .github/scripts/release.py newest TAG
    python .github/scripts/release.py crates verify|publish
    python .github/scripts/release.py pypi DIST
    python .github/scripts/release.py github-release TAG DIST...
        [--prerelease | --not-latest]
    python .github/scripts/release.py verify TAG DIST CLI [--tap OWNER/REPO]
    python .github/scripts/release.py api-baseline [--registry crates|pypi]

A release publishes to four places in turn --- PyPI, crates.io, the GitHub
Release, the Homebrew tap --- and none of them takes a version back.  When one
step fails after another succeeded, re-running the workflow must finish the
job rather than trip over what is already there, and must not paper over a
version that was published from something else.  So each step skips only what
is already published *and identical* to what this run built, publishes what is
missing, and fails on anything that differs; `verify` checks every place at the
end.  CONTRIBUTING.md ("If a release fails partway") is the runbook.

`api-baseline` is CI's, not the release's: the published release the API gates
(`semver`, `python-api`) compare this tree with, and what kind of release it
would be.
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

# What PyPI serves: the wheels and the sdist.
DISTRIBUTIONS = (".whl", ".tar.gz")

# Where cargo finds a crates.io token.
TOKENS = frozenset({"CARGO_REGISTRY_TOKEN", "CARGO_REGISTRIES_CRATES_IO_TOKEN"})

AGENT = "medh5-release (https://github.com/medh5-project/medh5)"


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


# A final release's tag; a pre-release (`v2.1.0-rc.1`) is never the newest.
FINAL = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def newest(tag: str, tags: list[str]) -> bool:
    """Whether `tag` is a final release no older than any final release
    tagged: what GitHub's "Latest" and the Homebrew tap follow.  A backport
    --- a 2.0.x patch tagged after 2.1.0 --- is not, and must not move them
    back."""
    ours = FINAL.fullmatch(tag)
    if not ours:
        return False
    mine = tuple(map(int, ours.groups()))
    return all(
        mine >= tuple(map(int, other.groups()))
        for other in map(FINAL.fullmatch, tags)
        if other
    )


def digests(directory: Path) -> dict[str, str]:
    return {
        p.name: sha256(p.read_bytes())
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def dist_digests(directory: Path) -> dict[str, str]:
    """The wheels and the sdist in `directory`, as `digests` lists them.

    Nothing else there is a distribution: under Trusted Publishing the
    publishing action signs each one and writes the PEP 740 attestation beside
    it, as `<distribution>.publish.attestation`, in the directory it uploaded.
    """
    return {
        name: digest
        for name, digest in digests(directory).items()
        if name.endswith(DISTRIBUTIONS)
    }


def dist_version(names: list[str]) -> str:
    """The one version every distribution file carries (PEP 440 spelling)."""
    versions = {name.split("-")[1].removesuffix(".tar.gz") for name in names}
    if len(versions) != 1:
        raise SystemExit(
            f"the distributions carry {len(versions)} versions: {sorted(versions)}"
        )
    return versions.pop()


# The release kinds `cargo semver-checks --release-type` takes.
RELEASE_TYPES = ("major", "minor", "patch")

_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?")


def api_baseline(
    current: str, published: list[str], *, branch: str, override: str = ""
) -> tuple[str, str]:
    """(baseline, release type): the newest final release at or below
    `current`, and whether `current` would be a major, minor or patch release
    after it.  A pre-release `current` (2.1.0-rc.1) is below its final.  When
    the two are equal, the tree is the next release's work: a patch on a
    `release/**` line (`branch`, a name or a `refs/heads/` ref), a minor
    elsewhere.  `override` (the repository variable
    `API_RELEASE_TYPE`) names the type outright, for 3.0's work (D35)."""
    ours = _SEMVER.fullmatch(current)
    if ours is None:
        raise SystemExit(f"version {current!r} is not X.Y.Z")
    mine = tuple(map(int, ours.groups()[:3]))
    finals = []
    for version in published:
        found = _SEMVER.fullmatch(version)
        if found is None or found[4] is not None:
            continue  # a pre-release is never a baseline
        triple = tuple(map(int, found.groups()[:3]))
        if triple < mine or (triple == mine and ours[4] is None):
            finals.append(triple)
    if not finals:
        raise SystemExit(f"no final release at or below {current}")
    base = max(finals)
    if override:
        if override not in RELEASE_TYPES:
            raise SystemExit(
                f"API_RELEASE_TYPE {override!r} is not one of "
                f"{', '.join(RELEASE_TYPES)}"
            )
        kind = override
    elif base[0] != mine[0]:
        kind = "major"
    elif base[1] != mine[1]:
        kind = "minor"
    elif base[2] != mine[2]:
        kind = "patch"
    else:
        line = branch.removeprefix("refs/heads/")
        kind = "patch" if line.startswith("release/") else "minor"
    return ".".join(map(str, base)), kind


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


def run(*command: str, capture: bool = False, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=capture, env=env
    )
    return result.stdout if capture else ""


def workspace_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = text.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    if not match:
        raise SystemExit("no [workspace.package] version")
    return match.group(1)


def crates_io_status(name: str, version: str) -> str:
    """`name` at `version` on crates.io: "absent", "published" or "yanked".

    A yanked version is not absent: crates.io refuses to take it again.
    """
    body = fetch(f"https://crates.io/api/v1/crates/{name}/{version}")
    if body is None:
        return "absent"
    return "yanked" if json.loads(body)["version"]["yanked"] else "published"


def index_path(name: str) -> str:
    """Where crates.io's sparse index keeps a crate's versions (cargo's
    layout: by the length of the name, then its first letters)."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def in_index(name: str, version: str) -> bool:
    """Whether the index cargo resolves against lists `name` at `version`.
    The API has a version as soon as it is uploaded; the index, a moment
    later."""
    body = fetch(f"https://index.crates.io/{index_path(name)}")
    if body is None:
        return False
    return any(
        json.loads(line).get("vers") == version
        for line in body.decode("utf-8").splitlines()
        if line.strip()
    )


def wait_for_index(name: str, version: str, *, wait: int = 600) -> None:
    deadline = time.monotonic() + wait
    while not in_index(name, version):
        if time.monotonic() > deadline:
            raise SystemExit(
                f"{name} {version} is published but not yet in the crates.io index, "
                "which the next crate resolves against: re-run the job"
            )
        time.sleep(10)
    print(f"{name} {version} is in the crates.io index")


def published_crate(name: str, version: str) -> bytes:
    body = fetch(f"https://static.crates.io/crates/{name}/{name}-{version}.crate")
    if body is None:
        raise SystemExit(
            f"crates.io lists {name} {version} but serves no .crate for it"
        )
    return body


def published_versions(registry: str) -> list[str]:
    """Every version of medh5 the registry serves, yanked ones left out."""
    if registry == "crates":
        body = fetch("https://crates.io/api/v1/crates/medh5")
        if body is None:
            return []
        versions = json.loads(body)["versions"]
        return [v["num"] for v in versions if not v["yanked"]]
    body = fetch("https://pypi.org/pypi/medh5/json")
    if body is None:
        return []
    releases = json.loads(body)["releases"]
    return [
        version
        for version, files in releases.items()
        if files and not all(f.get("yanked") for f in files)
    ]


def origin_tags() -> list[str]:
    """Every `v*` tag the repository has (the checkout fetched only its own)."""
    listing = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", "origin", "v*"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return [line.split("refs/tags/", 1)[1] for line in listing.splitlines() if line]


def pypi_digests(version: str) -> dict[str, str]:
    body = fetch(f"https://pypi.org/pypi/medh5/{version}/json")
    if body is None:
        return {}
    return {
        entry["filename"]: entry["digests"]["sha256"]
        for entry in json.loads(body)["urls"]
    }


# --- the steps ------------------------------------------------------------------


def crate_statuses(version: str) -> dict[str, str]:
    """Each crate's status on crates.io at `version`; a yanked one stops the
    release, since crates.io never takes a version again."""
    statuses = {name: crates_io_status(name, version) for name in CRATES}
    for name, status in statuses.items():
        if status == "yanked":
            raise SystemExit(
                f"{name} {version} is on crates.io, yanked, and a version cannot be "
                "published twice: undo the yank (`cargo yank --undo`) if it holds "
                "this release, or publish under a new version"
            )
    return statuses


def verify_crates() -> int:
    """Stage one, without a token: build every crate from its package, as CI's
    `rust` job does, and hold any version crates.io already has to this
    commit's.  A failure here publishes nothing."""
    version = workspace_version()
    statuses = crate_statuses(version)
    untokened = {k: v for k, v in os.environ.items() if k not in TOKENS}
    packages = [arg for name in CRATES for arg in ("-p", name)]
    # Verifying builds each package, running the build script of every
    # dependency --- which need not see a token, and here none exists yet.
    run("cargo", "package", "--locked", *packages, env=untokened)
    for name, status in statuses.items():
        if status == "absent":
            print(f"{name} {version} is not on crates.io: to publish")
            continue
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


def publish_crates() -> int:
    """Stage three, with the token trusted publishing issued: each crate not
    on crates.io, in dependency order, each in the index before the next,
    which resolves it there.  Stage one verified them, so none is built
    again with the token in reach of a build script."""
    version = workspace_version()
    statuses = crate_statuses(version)
    for name in CRATES:
        if statuses[name] != "absent":
            continue
        run("cargo", "publish", "--locked", "--no-verify", "-p", name)
        # `cargo publish` waits for the index only so long, then warns and
        # succeeds; the next crate resolves this one through the index.
        wait_for_index(name, version)
    return 0


def check_pypi(dist: Path, *, wait: int = 600) -> int:
    """Every distribution in `dist` is on PyPI with the same digest.  A fresh
    upload takes a moment to appear in the JSON API, so missing files are
    retried."""
    local = dist_digests(dist)
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
    tag: str, files: list[Path], *, prerelease: bool, notes: Path, latest: bool = True
) -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    local = {path.name: sha256(path.read_bytes()) for path in files}
    view = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if view.returncode != 0:
        # GitHub makes a new final release "Latest" unless told otherwise,
        # backports included.
        flags = ["--prerelease"] if prerelease else []
        if not prerelease and not latest:
            flags.append("--latest=false")
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
    """Every place has this version: PyPI every distribution in `dist`,
    crates.io every crate, the release page those and every file of `cli`, and
    the tap (when named) a formula for it."""
    version = tag.removeprefix("v")
    problems = []
    local = dist_digests(dist)
    problems += [
        f"PyPI: {p}"
        for p in dist_problems(local, pypi_digests(dist_version(list(local))))
    ]
    statuses = {n: crates_io_status(n, version) for n in CRATES}
    problems += [
        f"crates.io: {n} {version} is {'yanked' if s == 'yanked' else 'not published'}"
        for n, s in statuses.items()
        if s != "published"
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
    # `latest=true|false`, for $GITHUB_OUTPUT.
    latest = sub.add_parser("newest")
    latest.add_argument("tag")
    crates = sub.add_parser("crates")
    crates.add_argument("stage", choices=("verify", "publish"))
    pypi = sub.add_parser("pypi")
    pypi.add_argument("dist", type=Path)
    release = sub.add_parser("github-release")
    release.add_argument("tag")
    release.add_argument("files", type=Path, nargs="+")
    release.add_argument("--notes", type=Path, default=Path("release-notes.md"))
    kind = release.add_mutually_exclusive_group()
    kind.add_argument("--prerelease", action="store_true")
    kind.add_argument("--not-latest", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("tag")
    check.add_argument("dist", type=Path)
    check.add_argument("cli", type=Path)
    check.add_argument("--tap")
    # `version=` and `release_type=`, for $GITHUB_OUTPUT.
    baseline = sub.add_parser("api-baseline")
    baseline.add_argument("--registry", choices=("crates", "pypi"), default="crates")
    args = parser.parse_args(argv)
    if args.step == "api-baseline":
        version, kind = api_baseline(
            workspace_version(),
            published_versions(args.registry),
            branch=os.environ.get("BASE_REF", ""),
            override=os.environ.get("API_RELEASE_TYPE", ""),
        )
        print(f"version={version}")
        print(f"release_type={kind}")
        return 0
    if args.step == "newest":
        print(f"latest={str(newest(args.tag, origin_tags())).lower()}")
        return 0
    if args.step == "crates":
        return verify_crates() if args.stage == "verify" else publish_crates()
    if args.step == "pypi":
        return check_pypi(args.dist)
    if args.step == "github-release":
        return github_release(
            args.tag,
            args.files,
            prerelease=args.prerelease,
            notes=args.notes,
            latest=not args.not_latest,
        )
    return verify(
        args.tag, args.dist, args.cli, tap=args.tap, token=os.environ.get("GH_TOKEN")
    )


if __name__ == "__main__":
    sys.exit(main())
