"""The package's own metadata.

`medh5.__version__` is not decoration: it is stamped into every file's
`generator` (§12) and into every dataset manifest.  A wheel whose declared
version disagrees with the string it writes into user data is a provenance
bug that no amount of format testing catches, because both halves are
internally consistent -- they just describe different releases.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import io
import json
import os
import random
import re
import shutil
import sysconfig
import tarfile
from importlib.metadata import distribution
from types import ModuleType

import pytest

import medh5
from tests.helpers import ROOT, write_sample

PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github/workflows/ci.yml"
RELEASE = ROOT / ".github/workflows/release.yml"


WORKSPACE = ROOT / "Cargo.toml"


def _declared() -> str:
    """The version the wheel is built with, read without `tomllib`.

    `tomllib` is 3.11+ and this package supports 3.10.  Guarding the import
    would make this skip on the oldest interpreter it claims to support --
    which is the failure mode the MONAI job exists to prevent -- so it reads
    the lines it needs instead.  Since 2.0 the one source is the Cargo
    workspace: `pyproject.toml` declares the version dynamic, maturin reads it
    from the extension crate, and every crate --- engine, CLI, extension ---
    inherits the workspace's.  What this checks is that the wiring points
    there and nowhere else.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert re.search(r'^dynamic\s*=\s*\["version"\]', text, re.M), (
        "pyproject.toml must declare version as dynamic"
    )
    assert not re.search(r"^version\s*=", text, re.M), (
        "a static [project] version would be a second source"
    )
    manifest = re.search(r'^manifest-path\s*=\s*"([^"]+)"', text, re.M)
    assert manifest, "maturin must build the extension crate"
    for crate in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        assert re.search(
            r"^version\.workspace\s*=\s*true", crate.read_text("utf-8"), re.M
        ), f"{crate.parent.name} must inherit the workspace version"
    assert (ROOT / manifest.group(1)).is_file()
    workspace = WORKSPACE.read_text(encoding="utf-8")
    section = workspace.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    assert match, "no [workspace.package] version"
    return match.group(1)


def _release_script(name: str) -> ModuleType:
    path = ROOT / ".github" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jobs(workflow: str) -> dict[str, str]:
    """A workflow's jobs by name, each with the text of its block."""
    body = workflow.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against the workspace version.

    The version is written once, in the Cargo workspace; the wheel, the CLI and
    the engine that stamps `generator` all take it from there, so a wheel that
    reports its version wrongly in the files it writes is no longer a thing a
    bump can produce.
    """
    assert medh5.__version__ == _declared()


def test_the_crates_pin_each_other_at_the_workspace_version():
    """The crates name each other at exactly the workspace version.

    crates.io publishes each crate on its own, so `medh5` depends on the
    `medh5-sys` of its own release through an `=X.Y.Z` pin in
    `[workspace.dependencies]`, and `medh5-cli` on the `medh5` of its own.
    The runbook said the version lived in one place; a bump that set it alone
    would publish an engine depending on the previous release's HDF5 build.
    """
    workspace = WORKSPACE.read_text(encoding="utf-8")
    section = workspace.split("[workspace.dependencies]", 1)[1].split("\n[", 1)[0]
    pins = dict(
        re.findall(
            r'^(medh5[\w-]*)\s*=\s*\{[^}]*version\s*=\s*"=([^"]+)"', section, re.M
        )
    )
    assert set(pins) == {"medh5", "medh5-sys", "medh5-cli"}, pins
    assert set(pins.values()) == {_declared()}, pins


def test_the_format_version_is_not_the_package_version():
    """§1: the *format* is 1.1 (1.0 plus the optional clinical profile). The
    package ships releases against it.

    Tying them together would force a format-version bump for every package
    release, and the format version is what tells a reader whether it can open
    the file at all.

    The assertion used to be ``_declared().startswith("1.0")``, which tied the
    two together in exactly the way the paragraph above forbids -- it only
    looked correct while the package happened to sit on 1.0.x, and the first
    package minor bump against an unchanged format failed it.  1.x then held
    the package MAJOR equal to the format's; 2.0 is the architectural reset
    that broke that on purpose (a Rust engine, one format), so what has to
    hold is that the newest format version is 1.1, the package version is a
    well-formed release of its own, and the package MAJOR never trails the
    format MAJOR it writes.  A file is still written at the lowest version its
    content needs, so an imaging-only sample stays 1.0.
    """
    assert medh5.__format_version__ == "1.1"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?[0-9A-Za-z.]+)?", _declared())
    package_major = int(_declared().split(".")[0])
    assert package_major >= int(medh5.__format_version__.split(".")[0])


def test_release_the_homebrew_formula_names_the_binaries_ci_builds() -> None:
    """The release writes the formula from the checksums of the binaries CI
    built.  A target the formula expects and CI never builds --- or an archive
    named one way by the build and another by the formula --- would publish a
    formula whose download 404s."""
    brew = _release_script("homebrew_formula.py")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
    built = set(re.findall(r"- target: (\S+)", ci))
    assert set(brew.TARGETS.values()) <= built
    assert all(target in release for target in built)
    assert 'name="medh5-$version-$target"' in build
    assert brew.archive("2.0.0", "x") == "medh5-2.0.0-x.tar.gz"

    sums = {brew.archive("2.0.0", t): f"{i:064x}" for i, t in enumerate(sorted(built))}
    text = brew.formula("2.0.0", sums, "XwK-P/medh5")
    base = "https://github.com/XwK-P/medh5/releases/download/v2.0.0"
    for target in brew.TARGETS.values():
        name = brew.archive("2.0.0", target)
        assert f'url "{base}/{name}"' in text
        assert f'sha256 "{sums[name]}"' in text
    assert 'bin.install "medh5"' in text

    del sums[brew.archive("2.0.0", "x86_64-apple-darwin")]
    with pytest.raises(SystemExit, match="x86_64-apple-darwin"):
        brew.formula("2.0.0", sums, "XwK-P/medh5")
    listing = f"{'a' * 64}  medh5-2.0.0-x.tar.gz\n\n{'b' * 64} *y.zip\n"
    assert brew.read_sums(listing) == {
        "medh5-2.0.0-x.tar.gz": "a" * 64,
        "y.zip": "b" * 64,
    }


def test_R03_the_2_0_0_notes_describe_the_format_it_writes() -> None:
    """The release publishes its CHANGELOG section, and only that section.

    2.0.0's said "The file format is unchanged: 1.0" while the engine writes
    1.1 for a clinical sample, and the clinical work sat under [Unreleased],
    which no release extracts.  The release now refuses a CHANGELOG with
    entries still under [Unreleased].
    """
    notes = _release_script("release_notes.py")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    body = notes.notes(text, "2.0.0", "XwK-P/medh5", "v2.0.0")
    assert "**format 1.1**" in body and "**The `clinical` profile**" in body
    assert "format is unchanged" not in body
    pinned = "https://github.com/XwK-P/medh5/blob/v2.0.0/docs/spec/medh5-1.1.md"
    assert f"]({pinned})" in body
    with pytest.raises(SystemExit, match="no section for 0.0.0"):
        notes.notes(text, "0.0.0", "o/r", "v0.0.0")
    notes.check_unreleased("## [Unreleased]\n\n## [9.9.9]\n\n- shipped\n")
    with pytest.raises(SystemExit, match="Unreleased"):
        notes.check_unreleased("## [Unreleased]\n\n- not yet\n\n## [9.9.9]\n\n- x\n")


class TestPublicNames:
    """Where the public names live."""

    def test_the_writer_lives_beside_the_reader(self):
        import medh5.sample as sample_module

        assert medh5.SampleWriter is sample_module.SampleWriter
        assert medh5.create is sample_module.create
        assert medh5.amend is sample_module.amend
        assert {"SampleWriter", "create", "amend"} <= set(sample_module.__all__)

    def test_optional_dependencies_name_their_extra(self):
        from medh5._optional import require

        with pytest.raises(ImportError, match=r"medh5\[dicomseg\]"):
            require("no_such_module_medh5", extra="dicomseg", purpose="testing")
        assert require("json", extra="x", purpose="y").dumps({}) == "{}"

    def test_Q11_the_dead_names_are_gone(self):
        import medh5.geometry as grid
        import medh5.transforms as apply

        assert not hasattr(grid, "iter_spatial_slices")
        assert not hasattr(grid, "KNOWN_COORD_SYSTEMS")
        assert not hasattr(apply, "_refuse_outside")


class TestRepositoryGates:
    """The gates the repository holds itself to."""

    def test_the_lint_gate_refuses_suppressions_that_suppress_nothing(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert '"RUF100"' in text

    def test_K03_actions_are_pinned_by_commit(self):
        uses = []
        for name in ("ci.yml", "release.yml"):
            text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            uses += re.findall(r"uses:\s*(\S+)(.*)", text)
        external = [(u, rest) for u, rest in uses if not u.startswith("./")]
        assert external
        for action, rest in external:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
            assert re.search(r"#\s*v\d", rest), f"{action} carries no version comment"
        dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        assert "package-ecosystem: github-actions" in dependabot

    def test_the_minimum_rust_is_a_full_version(self):
        """`rust-version` is published with every crate and cannot be
        corrected there.  It said `1.85` --- 1.85.0 to cargo --- while the HDF5
        crates need 1.85.1, and the job that keeps the promise installed the
        newest 1.85.x for `1.85` and passed."""
        workspace = WORKSPACE.read_text(encoding="utf-8")
        section = workspace.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
        assert re.search(r'^rust-version = "\d+\.\d+\.\d+"$', section, re.M)
        msrv = _jobs(CI.read_text(encoding="utf-8"))["msrv"]
        assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in msrv

    def test_K03_the_release_runs_ci_on_the_tagged_commit(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "workflow_call:" in ci and "concurrency:" in ci
        assert "uses: ./.github/workflows/ci.yml" in release
        build = release[release.index("  build:") :]
        assert "needs: ci" in build.split("\n  publish:")[0]
        # The two builds upload in one run when CI is called; names must differ.
        assert "name: ci-dist" in ci and "name: dist" in release


class TestDistribution:
    """What the release ships, and how it ships it (R01--R08, W02 of the 2.0
    audit)."""

    def test_R01_the_required_check_needs_every_job(self):
        """The ruleset required `build` and `test-macos (3.12)`, names no job
        reports --- a matrix reports `build (<target>)` --- so no pull request
        could merge, and a job added later would not have been required.  One
        aggregate job is required instead; it must need every other one, and
        run when one failed (a skipped required check counts as passing)."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        gate = jobs.pop("ci-ok")
        needs = re.findall(r"^      - ([\w-]+)$", gate.split("steps:")[0], re.M)
        assert sorted(needs) == sorted(jobs)
        assert "if: always()" in gate and 'result != "success"' in gate

    def test_R02_the_notices_name_what_is_compiled_in(self):
        """Every crate the notices list is Cargo.lock's, at its version, and
        the C libraries are there; CI regenerates the file and fails on a
        difference (`--check`), which needs the crate sources this cannot."""
        notices = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        locked = set(re.findall(r'^name = "([^"]+)"\nversion = "([^"]+)"', lock, re.M))
        table = notices.split("\nRust crates (", 1)[1].split("\nLicence texts\n")[0]
        listed = set(re.findall(r"^  (\S+) (\S+) ", table, re.M))
        assert len(listed) > 100 and listed <= locked, sorted(listed - locked)
        for library in ("HDF5", "C-Blosc2", "HDF5-Blosc2 filter", "zlib", "LZ4"):
            assert re.search(rf"^  {re.escape(library)} ", notices, re.M), library
        assert "--check" in _jobs(CI.read_text(encoding="utf-8"))["rust"]

    def test_R02_every_artifact_carries_the_licences(self):
        """The wheel (as `license-files`), the sdist (with them), each binary
        archive, and each crate --- whose MIT licence must travel with it."""
        assert (
            'license-files = ["LICENSE", "THIRD_PARTY_NOTICES"]'
            in PYPROJECT.read_text(encoding="utf-8")
        )
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        assert 'cp "$exe" LICENSE THIRD_PARTY_NOTICES' in build
        release = RELEASE.read_text(encoding="utf-8")
        assert ".dist-info/licenses/{notice}" in release and 'f"/{notice}"' in release
        licence = (ROOT / "LICENSE").read_bytes()
        for crate in ("medh5", "medh5-cli", "medh5-sys"):
            assert (ROOT / "crates" / crate / "LICENSE").read_bytes() == licence, crate
        sys_manifest = (ROOT / "crates/medh5-sys/Cargo.toml").read_text(
            encoding="utf-8"
        )
        assert '"LICENSE",' in sys_manifest.split("include = [", 1)[1].split("]")[0]

    def test_R02_the_installed_package_carries_the_notices(self):
        names = {str(f) for f in distribution("medh5").files or []}
        for notice in ("LICENSE", "THIRD_PARTY_NOTICES"):
            assert any(n.endswith(f".dist-info/licenses/{notice}") for n in names), (
                notice
            )

    def test_R02_the_generator_covers_every_crate_or_refuses(self, tmp_path):
        notices = _release_script("third_party_notices.py")

        def package(name: str, licence: str, files: dict[str, str]) -> dict:
            directory = tmp_path / name
            directory.mkdir()
            for file, text in {"Cargo.toml": "", **files}.items():
                (directory / file).write_bytes(text.encode())
            return {
                "id": name,
                "name": name,
                "version": "1.0.0",
                "license": licence,
                "manifest_path": str(directory / "Cargo.toml"),
            }

        def dep(name: str, kind: str | None = None) -> dict:
            return {"pkg": name, "dep_kinds": [{"kind": kind, "target": None}]}

        apache = "Apache License\r\nVersion 2.0   \r\n\r\n"
        packages = [
            package("medh5-python", "MIT", {}),
            package("medh5-cli", "MIT", {}),
            package("a", "MIT OR Apache-2.0", {"LICENSE-APACHE": apache}),
            package("b", "Apache-2.0", {"LICENSE": "Apache License\nVersion 2.0\n"}),
            package("c", "MIT", {}),
            package("build-only", "GPL-3.0-only", {}),
        ]
        nodes = [
            {
                "id": "medh5-python",
                "deps": [dep("a"), dep("c"), dep("build-only", "build")],
            },
            {"id": "medh5-cli", "deps": [dep("b")]},
            *({"id": n, "deps": []} for n in ("a", "b", "c", "build-only")),
        ]
        metadata = {
            "packages": packages,
            "workspace_members": ["medh5-python", "medh5-cli"],
            "resolve": {"nodes": nodes},
        }
        shipped = notices.shipped(metadata)
        assert [p["name"] for p in shipped] == ["a", "b", "c"]
        texts = notices.Texts()
        for p in shipped:
            for where, text in notices.crate_texts(p):
                texts.cite(text, f"{p['name']}: {where}")
        # One Apache text however its lines end, and the MIT notice for the
        # crate that ships no file.
        assert len(texts.numbers) == 2
        assert notices.crate_texts(shipped[2])[0][1] == notices.MIT
        with pytest.raises(SystemExit, match="is not MIT"):
            notices.crate_texts(package("d", "BSD-3-Clause", {}))
        assert notices.offers_mit("Apache-2.0 OR MIT")
        assert not notices.offers_mit("(MIT OR Apache-2.0) AND Unicode-3.0")

    def test_R04_the_maturin_floor_writes_pep_639_metadata(self):
        """1.9.0 wrote the legacy `License` field and 1.8 could not read the
        table; the floor is 1.9.3, and CI builds the sdist with exactly it."""
        text = PYPROJECT.read_text(encoding="utf-8")
        assert re.findall(r'"maturin>=([\d.]+),<2"', text) == ["1.9.3", "1.9.3"]
        sdist = _jobs(CI.read_text(encoding="utf-8"))["sdist"]
        assert '["build-system"]["requires"]' in sdist
        assert "--no-build-isolation" in sdist
        assert 'f"License-Expression: {expression}"' in sdist
        assert "['License-Expression'] ==" in sdist and "['license']" in sdist

    def test_the_declared_licence_covers_what_is_compiled_in(self):
        """The wheels declared `License-Expression: MIT` while compiling in
        HDF5, C-Blosc2, Zstandard and `subtle` (BSD-3-Clause), LZ4 and
        rust-numpy (BSD-2-Clause), zlib and `foldhash` (Zlib),
        `unicode-general-category` (Apache-2.0) and `borrow-or-share` (MIT-0)
        --- and PyPI cannot correct a release's metadata.  The binary links the
        same libraries, so the Homebrew formula states the same set."""
        (expression,) = re.findall(
            r'^license = "([^"]+)"$', PYPROJECT.read_text(encoding="utf-8"), re.M
        )
        declared = expression.split(" AND ")
        assert declared[0] == "MIT", "the package's own licence leads"
        assert set(declared) == {
            "MIT", "BSD-3-Clause", "BSD-2-Clause", "Zlib", "Apache-2.0", "MIT-0"
        }  # fmt: skip
        notices = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
        for licence in ("BSD-3-Clause", "BSD-2-Clause", "Zlib", "MIT-0"):
            assert re.search(rf"  {re.escape(licence)} +\[", notices), licence
        brew = _release_script("homebrew_formula.py")
        assert list(brew.LICENSES) == declared
        sums = {brew.archive("2.0.0", t): "0" * 64 for t in brew.TARGETS.values()}
        formula = brew.formula("2.0.0", sums, "XwK-P/medh5")
        assert f"  license all_of: {json.dumps(declared)}\n" in formula

    def test_R05_a_published_crate_is_skipped_only_when_identical(self):
        """A re-run after a partial release skips a crate already on crates.io
        only if its contents are the tagged commit's; cargo's generated files,
        whose spelling follows the cargo version, are not compared."""
        release = _release_script("release.py")

        def crate(files: dict[str, bytes]) -> bytes:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for name, data in files.items():
                    info = tarfile.TarInfo(f"medh5-2.0.0/{name}")
                    info.size, info.mtime = len(data), random.randrange(2**31)
                    archive.addfile(info, io.BytesIO(data))
            return buffer.getvalue()

        source = {"src/lib.rs": b"fn f() {}", "Cargo.toml.orig": b"[package]"}
        ours = release.crate_files(crate({**source, "Cargo.toml": b"cargo 1.97"}))
        theirs = release.crate_files(crate({**source, "Cargo.toml": b"cargo 1.95"}))
        assert release.differences(ours, theirs) == []
        edited = release.crate_files(crate({**source, "src/lib.rs": b"fn g() {}"}))
        assert release.differences(ours, edited) == ["src/lib.rs: differs"]
        extra = release.crate_files(crate({**source, "build.rs": b""}))
        assert release.differences(ours, extra) == [
            "build.rs: only in the published one"
        ]

    def test_R05_dists_and_assets_are_compared_by_digest(self):
        release = _release_script("release.py")
        local = {"medh5-2.0.0.tar.gz": "a", "medh5-2.0.0-cp310-abi3-x.whl": "b"}
        assert release.dist_version(list(local)) == "2.0.0"
        assert release.dist_problems(local, dict(local)) == []
        assert release.dist_problems(local, {"medh5-2.0.0.tar.gz": "z"}) == [
            "medh5-2.0.0-cp310-abi3-x.whl: not published",
            "medh5-2.0.0.tar.gz: published with different content",
        ]
        upload, download = release.release_plan(
            {"x": "1", "y": "2", "z": "3"}, {"x": "1", "y": None}
        )
        assert (upload, download) == (["z"], ["y"])
        assert release.formula_version(
            'class Medh5 < Formula\n  version "2.0.0"\n'
        ) == ("2.0.0")
        with pytest.raises(SystemExit, match="2 versions"):
            release.dist_version(["medh5-2.0.0.tar.gz", "medh5-2.0.1.tar.gz"])

    def test_only_the_distributions_are_compared_with_pypi(self, tmp_path, monkeypatch):
        """Under Trusted Publishing the upload signs every distribution and
        writes its attestation beside it, as `<file>.publish.attestation`.
        The check after the upload read those as distributions of a second
        version and failed with PyPI already holding the release, so neither
        the release page nor the formula followed --- and a re-run signed, and
        failed, again."""
        release = _release_script("release.py")
        dist, cli = tmp_path / "dist", tmp_path / "cli"
        dist.mkdir()
        cli.mkdir()
        names = ["medh5-2.0.0-cp310-abi3-win_amd64.whl", "medh5-2.0.0.tar.gz"]
        for name in names:
            (dist / name).write_bytes(name.encode())
            (dist / f"{name}.publish.attestation").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="2 versions"):
            release.dist_version(list(release.digests(dist)))
        local = release.dist_digests(dist)
        assert sorted(local) == sorted(names)
        assert release.dist_version(list(local)) == "2.0.0"
        published = {name: release.sha256(name.encode()) for name in names}
        monkeypatch.setattr(release, "pypi_digests", lambda version: published)
        assert release.check_pypi(dist, wait=0) == 0
        # And `verify`, which reads the same directory.
        (cli / "medh5-2.0.0-x.tar.gz").write_bytes(b"x")
        assets = {"assets": [{"name": n} for n in [*names, "medh5-2.0.0-x.tar.gz"]]}
        monkeypatch.setattr(release, "on_crates_io", lambda name, version: True)
        monkeypatch.setattr(
            release, "fetch", lambda url, **_: json.dumps(assets).encode()
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "XwK-P/medh5")
        assert release.verify("v2.0.0", dist, cli, tap=None, token=None) == 0

    def test_a_crate_is_verified_without_the_token_and_indexed_before_the_next(
        self, monkeypatch
    ):
        """`cargo publish` verified each crate with the token in the
        environment of every dependency's build script, and waited for the
        index only so long before warning and succeeding --- after which the
        next crate, which resolves the one before through the index, failed."""
        release = _release_script("release.py")
        steps: list[tuple[str, ...]] = []

        def run(*command: str, capture: bool = False, env=None) -> str:
            token = "CARGO_REGISTRY_TOKEN" in (os.environ if env is None else env)
            steps.append((*command, "token" if token else "no token"))
            return ""

        monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "secret")
        monkeypatch.setattr(release, "run", run)
        monkeypatch.setattr(release, "on_crates_io", lambda name, version: False)
        monkeypatch.setattr(
            release,
            "wait_for_index",
            lambda name, version: steps.append(("index", name)),
        )
        assert release.publish_crates() == 0
        assert steps == [
            step
            for name in ("medh5-sys", "medh5", "medh5-cli")
            for step in (
                ("cargo", "package", "-p", name, "--locked", "no token"),
                ("cargo", "publish", "-p", name, "--locked", "--no-verify", "token"),
                ("index", name),
            )
        ]
        # Where cargo looks a crate up, and what it finds there.
        names = ("medh5-sys", "Serde", "abc", "ab", "a")
        assert [release.index_path(n) for n in names] == [
            "me/dh/medh5-sys", "se/rd/serde", "3/a/abc", "2/ab", "1/a"
        ]  # fmt: skip
        listing = b"".join(
            b'{"name":"medh5-sys","vers":"%s"}\n' % v for v in (b"1.9.0", b"2.0.0")
        )
        monkeypatch.setattr(
            release,
            "fetch",
            lambda url, **_: listing if url.endswith("/me/dh/medh5-sys") else None,
        )
        assert release.in_index("medh5-sys", "2.0.0")
        assert not release.in_index("medh5-sys", "2.0.1")
        assert not release.in_index("medh5", "2.0.0")

    def test_R05_R06_every_publishing_step_can_run_again(self):
        """`cargo publish --workspace` refused a re-run once one crate was up,
        `gh release create` once the release existed, and an unchanged formula
        failed `git commit`; a missing token in this repository only warned."""
        release = RELEASE.read_text(encoding="utf-8")
        jobs = _jobs(release)
        assert "skip-existing: true" in jobs["publish"]
        assert "release.py pypi dist/" in jobs["publish"]
        assert "release.py crates" in jobs["crates"]
        assert "cargo publish --workspace" not in release
        assert "release.py github-release" in jobs["github-release"]
        assert "gh release create" not in release
        assert "git diff --cached --quiet" in jobs["homebrew"]
        assert (
            "release.py verify" in jobs["verify"] and "!cancelled()" in jobs["verify"]
        )
        for job in ("crates", "homebrew"):
            assert "CANONICAL: ${{ github.repository == 'XwK-P/medh5' }}" in jobs[job]
            fails = (
                r'"\$CANONICAL" == true \]\]; then\n\s+echo "::error::[^\n]*\n\s+exit 1'
            )
            assert re.search(fails, jobs[job]), job
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "### If a release fails partway" in contributing

    def test_R07_the_sdist_carries_what_its_project_tests_read(self):
        """Four of the shipped project tests read files the sdist left out, so
        they failed from the archive; CI now runs them from it."""
        include = re.findall(
            r'\{ path = "([^"]+)", format = "sdist" \}',
            PYPROJECT.read_text(encoding="utf-8"),
        )
        needed = [
            ".github/dependabot.yml",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            *(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / ".github/scripts").iterdir()
                if p.suffix in (".py", ".sh")
            ),
            "mkdocs.yml",
            "hooks/mkdocs_hooks.py",
            "docs/requirements.txt",
            "CHANGELOG.md",
        ]
        for path in needed:
            assert (ROOT / path).is_file(), path
            assert any(fnmatch.fnmatch(path, pattern) for pattern in include), path
        sdist = _jobs(CI.read_text(encoding="utf-8"))["sdist"]
        assert "rm -rf medh5" in sdist and "pytest tests/project" in sdist

    def test_W02_a_damaged_file_is_reported_by_every_binary(self, tmp_path):
        """An HDF5 compiled without NDEBUG aborts on a damaged file; CI damages
        files for every platform's binary, and for a build from a copy of the
        workspace without `.cargo/config.toml` --- a crates.io build --- which
        medh5-sys must stop until HDF5 is rebuilt with the flag."""
        damaged = _release_script("damaged.py")
        data = bytes(range(256)) * 4
        once = damaged.damage(data, random.Random(5))
        assert once != data and len(once) == len(data)
        assert sum(a != b for a, b in zip(data, once, strict=True)) <= 6
        assert damaged.damage(data, random.Random(5)) == once
        # The console script beside this interpreter, whether or not its
        # directory is on PATH.
        binary = shutil.which("medh5", path=sysconfig.get_path("scripts"))
        assert binary, "the package's `medh5` command is not installed"
        (tmp_path / "corpus").mkdir()
        write_sample(tmp_path / "corpus" / "case.medh5")
        assert damaged.main([binary, str(tmp_path / "corpus"), "--runs", "6"]) == 0
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        assert "damaged.py" in jobs["build"]
        msvc = jobs["msvc-from-crates-io"]
        assert "rm -r ws/.cargo" in msvc and "damaged.py" in msvc
        assert "cargo clean --release -p hdf5-metno-src" in msvc
        # The job's bash rewrote `/DNDEBUG` into `C:/Program Files/Git/DNDEBUG`.
        assert "CFLAGS_x86_64_pc_windows_msvc: -DNDEBUG" in msvc
