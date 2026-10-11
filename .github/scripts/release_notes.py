"""Write a release's notes from its CHANGELOG section:

    python .github/scripts/release_notes.py      (in the release workflow)

reads the tag from `GITHUB_REF_NAME` and writes `release-notes.md`.  It runs
before anything is published, so a tag whose version has no section fails
there --- and so does a CHANGELOG with entries still under [Unreleased]: those
are changes the release ships and its notes would not describe --- and notes
longer than a GitHub release takes.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

# GitHub refuses a release body longer than this, and the release is created
# after the packages are published: the notes are measured here, before.
GITHUB_LIMIT = 125_000


def section(text: str, heading: str) -> str:
    """The body of the `## [heading]` section, stripped ('' when absent)."""
    found = re.search(
        rf"^## \[{re.escape(heading)}\][^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    return found.group(1).strip() if found else ""


def check_unreleased(text: str) -> None:
    """Refuse a CHANGELOG whose [Unreleased] section has entries."""
    if section(text, "Unreleased"):
        sys.exit(
            "CHANGELOG.md has entries under [Unreleased]: move them into the "
            "section of the version being released"
        )


def notes(text: str, version: str, repository: str, tag: str) -> str:
    """`version`'s section, with relative links pinned to the tagged commit.

    A relative link resolves in GitHub's file view and on the docs site, but
    not on a release page.
    """
    body = section(text, version)
    if not body:
        sys.exit(f"CHANGELOG.md has no section for {version}")
    blob = f"https://github.com/{repository}/blob/{tag}/"
    body = re.sub(r"\]\((?!https?://|#|mailto:)", "](" + blob, body)
    if len(body) > GITHUB_LIMIT:
        sys.exit(
            f"the release notes for {version} are {len(body)} characters; "
            f"GitHub takes at most {GITHUB_LIMIT}"
        )
    return body


def main() -> None:
    tag = os.environ["GITHUB_REF_NAME"]
    version = tag.removeprefix("v")
    text = pathlib.Path("CHANGELOG.md").read_text(encoding="utf-8")
    check_unreleased(text)
    body = notes(text, version, os.environ["GITHUB_REPOSITORY"], tag)
    pathlib.Path("release-notes.md").write_text(body + "\n", encoding="utf-8")
    print(f"release notes: {len(body)} characters from CHANGELOG.md [{version}]")


if __name__ == "__main__":
    main()
