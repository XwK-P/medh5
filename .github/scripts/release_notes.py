"""Write a release's notes from its CHANGELOG section:

    python .github/scripts/release_notes.py          (in the release workflow)
    python .github/scripts/release_notes.py vX.Y.Z   (for a published release)

In the workflow it reads the tag from `GITHUB_REF_NAME` and writes
`release-notes.md`.  It runs before anything is published, so a tag whose
version has no section fails there --- and so does a CHANGELOG with entries
still under [Unreleased]: those are changes the release ships and its notes
would not describe --- and notes longer than a GitHub release takes.

Given a tag, it writes the notes of a release already published, for `gh
release edit vX.Y.Z --notes-file release-notes.md` after its section changed.
The [Unreleased] check belongs to the release being made, and is skipped.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

from markdown_it import MarkdownIt

# GitHub refuses a release body longer than this, and the release is created
# after the packages are published: the notes are measured here, before.
GITHUB_LIMIT = 125_000

# The repository the notes link into, outside the workflow.
REPOSITORY = "XwK-P/medh5"

# GitHub's Markdown is CommonMark with tables (and inline extensions).
MARKDOWN = MarkdownIt("commonmark").enable("table")

# How a line that could open a block begins, in some context and to some
# parser: a heading, a quote, HTML, a table row, a code fence, a footnote.
_OPENS = re.compile(r"[ \t]*(?:#{1,6}(?:[ \t]|$)|[>|<]|```|~~~|\[\^)")
# A line that opens or closes a code fence.
_FENCE = re.compile(r"[ \t]*(?:```|~~~)")
# A list item's marker, and a numbered one's number.
_MARKER = re.compile(r"[ \t]*(?:[-+*]|(\d{1,9})[.)])(?:[ \t]|$)")
# A line of nothing else: a thematic break, a setext underline, a table's
# delimiter row --- each of which changes what the line above it is.
_RULE = re.compile(r"[ \t]*(?:[-*_=:|][ \t]*)+")
# What comes before a line's text: its indentation, and any list markers.
_LEAD = re.compile(r"[ \t]*(?:(?:[-+*]|\d{1,9}[.)])[ \t]+)*")


def _joins(line: str, column: int, *, item: bool) -> bool:
    """Whether `line`, continuing a paragraph whose text starts at `column`
    (past the marker, if the paragraph opens a list `item`), joins the line
    above it the same way to any reader.

    Not if it could open a block, in some context and to some parser ---
    unless it is a numbered item other than 1, in a list item's paragraph and
    indented to its text: that cannot interrupt a paragraph (CommonMark 5.2),
    so "a target of" wrapped before "400." is one sentence.  Nor if it is
    indented short of the text: GitHub keeps that indentation inside a code
    span.
    """
    indent = len(line) - len(line.lstrip(" \t"))
    if 0 < indent < column or _OPENS.match(line) or _RULE.fullmatch(line):
        return False
    marker = _MARKER.match(line)
    if marker is None:
        return True
    number = marker.group(1)
    return item and number is not None and int(number) != 1 and indent >= column


def shown(source: str) -> str:
    """`source` as HTML, with each run of whitespace outside a code block one
    space: what a reader sees, to whom a line break and a space are alike."""
    pieces = re.split(r"(<pre>.*?</pre>)", MARKDOWN.render(source), flags=re.S)
    return "".join(
        piece if piece.startswith("<pre>") else re.sub(r"[ \t\r\n]+", " ", piece)
        for piece in pieces
    )


def unwrap(body: str) -> str:
    """`body` with each paragraph on one line.

    The changelog wraps its prose at 80 columns, which Markdown reads as one
    paragraph, and a release page --- like an issue or a comment --- breaks at
    every newline.  The lines of each paragraph, as the parser finds them, are
    joined up to a line that could open a block, since GitHub may read what
    follows as that block's.  A line after a hard break, or above a rule
    (which can make it a table's header), keeps its own line.  Code, tables
    and quotes are left as written, and so is everything after HTML (which
    runs past a blank line) or after a fence the parser does not see.  The
    parser is not GitHub's, so the notes are rendered both ways, and kept as
    written should the two differ.
    """
    lines = body.split("\n")
    lines_after = [*lines[1:], ""]
    tokens = MARKDOWN.parse(body)
    code = {
        n
        for token in tokens
        if token.type in ("fence", "code_block") and token.map
        for n in range(*token.map)
    }
    barrier = min(
        [t.map[0] for t in tokens if t.type == "html_block" and t.map]
        + [n for n, line in enumerate(lines) if _FENCE.match(line) and n not in code],
        default=len(lines),
    )
    continues: set[int] = set()  # lines that join the line before
    quotes = 0
    for token in tokens:
        if token.type == "blockquote_open":
            quotes += 1
        elif token.type == "blockquote_close":
            quotes -= 1
        elif token.type == "paragraph_open" and token.map and not quotes:
            start, end = token.map
            if end > barrier:
                break
            lead = _LEAD.match(lines[start])
            column = lead.end() if lead else 0
            item = _MARKER.match(lines[start]) is not None
            for n in range(start + 1, end):
                if not _joins(lines[n], column, item=item):
                    break
                if not (
                    lines[n - 1].endswith(("  ", "\\"))
                    or _RULE.fullmatch(lines_after[n])
                ):
                    continues.add(n)
    out: list[str] = []
    for n, line in enumerate(lines):
        if n in continues:
            out[-1] = out[-1].rstrip(" \t") + " " + line.lstrip(" \t")
        else:
            out.append(line)
    flowed = "\n".join(out)
    if shown(flowed) != shown(body):
        print(
            "release notes: kept as written; unwrapped, they read differently",
            file=sys.stderr,
        )
        return body
    return flowed


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
    """`version`'s section, unwrapped, with relative links pinned to the tag.

    A relative link resolves in GitHub's file view and on the docs site, but
    not on a release page.
    """
    body = section(text, version)
    if not body:
        sys.exit(f"CHANGELOG.md has no section for {version}")
    blob = f"https://github.com/{repository}/blob/{tag}/"
    body = unwrap(re.sub(r"\]\((?!https?://|#|mailto:)", "](" + blob, body))
    if len(body) > GITHUB_LIMIT:
        sys.exit(
            f"the release notes for {version} are {len(body)} characters; "
            f"GitHub takes at most {GITHUB_LIMIT}"
        )
    return body


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        sys.exit("usage: release_notes.py [vX.Y.Z]")
    tag = argv[0] if argv else os.environ["GITHUB_REF_NAME"]
    version = tag.removeprefix("v")
    text = pathlib.Path("CHANGELOG.md").read_text(encoding="utf-8")
    if not argv:
        check_unreleased(text)
    repository = os.environ.get("GITHUB_REPOSITORY", REPOSITORY)
    body = notes(text, version, repository, tag)
    pathlib.Path("release-notes.md").write_text(body + "\n", encoding="utf-8")
    print(f"release notes: {len(body)} characters from CHANGELOG.md [{version}]")


if __name__ == "__main__":
    main(sys.argv[1:])
