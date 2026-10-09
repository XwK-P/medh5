"""The package's own metadata.

`medh5.__version__` is not decoration: it is stamped into every file's
`generator` (§12) and into every dataset manifest.  A wheel whose declared
version disagrees with the string it writes into user data is a provenance
bug that no amount of format testing catches, because both halves are
internally consistent -- they just describe different releases.
"""

from __future__ import annotations

import importlib.util
import re
from types import ModuleType

import pytest

import medh5
from tests.helpers import ROOT

PYPROJECT = ROOT / "pyproject.toml"


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


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against the workspace version.

    The version is written once, in the Cargo workspace; the wheel, the CLI and
    the engine that stamps `generator` all take it from there, so a wheel that
    reports its version wrongly in the files it writes is no longer a thing a
    bump can produce.
    """
    assert medh5.__version__ == _declared()


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

    def test_K03_the_release_runs_ci_on_the_tagged_commit(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "workflow_call:" in ci and "concurrency:" in ci
        assert "uses: ./.github/workflows/ci.yml" in release
        build = release[release.index("  build:") :]
        assert "needs: ci" in build.split("\n  publish:")[0]
        # The two builds upload in one run when CI is called; names must differ.
        assert "name: ci-dist" in ci and "name: dist" in release
