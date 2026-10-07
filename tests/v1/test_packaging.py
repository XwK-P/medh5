"""The package's own metadata.

`medh5.__version__` is not decoration: it is stamped into every file's
`generator` (§12) and into every dataset manifest.  A wheel whose declared
version disagrees with the string it writes into user data is a provenance
bug that no amount of format testing catches, because both halves are
internally consistent -- they just describe different releases.
"""

from __future__ import annotations

import re
from pathlib import Path

import medh5

ROOT = Path(__file__).resolve().parents[2]
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


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against the workspace version.

    The version is written once, in the Cargo workspace; the wheel, the CLI and
    the engine that stamps `generator` all take it from there, so a wheel that
    reports its version wrongly in the files it writes is no longer a thing a
    bump can produce.
    """
    assert medh5.__version__ == _declared()


def test_the_format_version_is_not_the_package_version():
    """§1: the *format* is 1.0. The package ships releases against it.

    Tying them together would force a format-version bump for every package
    release, and the format version is what tells a reader whether it can open
    the file at all.

    The assertion used to be ``_declared().startswith("1.0")``, which tied the
    two together in exactly the way the paragraph above forbids -- it only
    looked correct while the package happened to sit on 1.0.x, and the first
    package minor bump against an unchanged format failed it.  1.x then held
    the package MAJOR equal to the format's; 2.0 is the architectural reset
    that broke that on purpose (a Rust engine, one format), so what has to
    hold is that the format version is 1.0, the package version is a
    well-formed release of its own, and the package MAJOR never trails the
    format MAJOR it writes.
    """
    assert medh5.__format_version__ == "1.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?[0-9A-Za-z.]+)?", _declared())
    package_major = int(_declared().split(".")[0])
    assert package_major >= int(medh5.__format_version__.split(".")[0])
