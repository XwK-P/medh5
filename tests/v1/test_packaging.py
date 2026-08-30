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

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared() -> str:
    """The version in `pyproject.toml`, read without `tomllib`.

    `tomllib` is 3.11+ and this package supports 3.10.  Guarding the import
    would make this skip on the oldest interpreter it claims to support --
    which is the failure mode the MONAI job exists to prevent -- so it reads
    the one line it needs instead.
    """
    in_project = False
    for line in PYPROJECT.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.fullmatch(r'version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    raise AssertionError("no [project] version found in pyproject.toml")


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against `pyproject.toml` only.

    Bumping one of the two and not the other publishes a wheel that reports
    its own version wrongly in every file it writes.
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
    package minor bump against an unchanged format failed it. What actually
    has to hold is that the format version is 1.0 and the package version is a
    well-formed release of its own.
    """
    assert medh5.__format_version__ == "1.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?[0-9A-Za-z.]+)?", _declared())
    # The package's MAJOR tracks the format's MAJOR: a 1.x package writes and
    # reads 1.x files (§16 -- readers reject an unknown MAJOR).
    assert _declared().split(".")[0] == medh5.__format_version__.split(".")[0]
