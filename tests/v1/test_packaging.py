"""The package's own metadata.

`medh5.__version__` is not decoration: it is stamped into every file's
`generator` (§12) and into every dataset manifest.  A wheel whose declared
version disagrees with the string it writes into user data is a provenance
bug that no amount of format testing catches, because both halves are
internally consistent -- they just describe different releases.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

import medh5

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared() -> str:
    with PYPROJECT.open("rb") as handle:
        version: str = tomllib.load(handle)["project"]["version"]
    return version


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against `pyproject.toml` only.

    Bumping one of the two and not the other publishes a wheel that reports
    its own version wrongly in every file it writes.
    """
    assert medh5.__version__ == _declared()


def test_the_format_version_is_not_the_package_version():
    """§1: the *format* is 1.0. The package ships fixes against it.

    Tying them together would force a format-version bump for every patch
    release, which is what tells a reader whether it can open the file at all.
    """
    assert medh5.__format_version__ == "1.0"
    assert _declared().startswith("1.0")
