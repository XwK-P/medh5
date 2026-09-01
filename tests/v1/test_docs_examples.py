"""The documentation is checked against the code, not proofread.

Every defect this file guards against was found in review after being written
confidently and shipped: a `--source ct/*.dcm` that argparse rejects before doing
anything, a `--out cold/` that dies with `IsADirectoryError`, a response label
naming one visit when it describes an interval.  Each was a claim about the code
that the code disagreed with, and each was reachable by asking the code.

So the documentation gets the same treatment as anything else here.  A prose page
is not a comment; it is an assertion about behaviour, and an assertion nobody
executes decays.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from medh5.cli import build_parser

DOCS = Path(__file__).resolve().parents[2] / "docs"
README = Path(__file__).resolve().parents[2] / "README.md"

# The normative specification is excluded: its examples are illustrative of the
# format rather than of this package's API, and it is versioned separately.
PAGES = sorted(p for p in DOCS.rglob("*.md") if "spec/medh5-1.0.md" not in str(p))


def _shell_lines() -> list[tuple[Path, str]]:
    """Every `medh5 ...` invocation in a fenced block, line continuations joined."""
    out: list[tuple[Path, str]] = []
    for path in [*PAGES, README]:
        for block in re.findall(
            r"```(?:bash|sh|console)?\n(.*?)```", path.read_text(), re.S
        ):
            for line in re.sub(r"\\\n\s*", " ", block).splitlines():
                line = line.strip().removeprefix("$ ").strip()
                if not line.startswith("medh5 "):
                    continue
                # Placeholder forms (`PATH...`, `medh5 COMMAND [args]`) and
                # pipelines are prose, not invocations.
                if line.endswith("...") or any(c in line for c in "|<>&"):
                    continue
                out.append((path, line))
    return out


def _parser_tree() -> dict[tuple[str, ...], object]:
    tree: dict[tuple[str, ...], object] = {}

    def walk(parser: object, prefix: tuple[str, ...]) -> None:
        tree[prefix] = parser
        for action in parser._actions:  # type: ignore[attr-defined]
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for name, sub in choices.items():
                    walk(sub, (*prefix, name))

    walk(build_parser(), ())
    return tree


@pytest.mark.parametrize(("path", "line"), _shell_lines(), ids=lambda v: str(v)[:60])
def test_documented_cli_flags_exist(path: Path, line: str) -> None:
    """A flag in the documentation is a flag the CLI defines.

    `--source ct/*.dcm` looked right for years and never worked: the option takes
    one value per occurrence, so the glob expanded and argparse exited with
    `unrecognized arguments` before the export began.
    """
    tree = _parser_tree()
    tokens = shlex.split(line)[1:]
    command: tuple[str, ...] = ()
    for token in tokens:
        if token.startswith("-") or (*command, token) not in tree:
            break
        command = (*command, token)
    parser = tree.get(command)
    if parser is None:  # a placeholder command name; nothing to check
        return
    defined = {
        option
        for action in parser._actions  # type: ignore[attr-defined]
        for option in action.option_strings
    }
    used = [
        token
        for token in tokens[len(command) :]
        # A negative number is a value (`--date-shift-days -117`), not a flag.
        if token.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", token)
    ]
    unknown = [f for f in used if f.split("=")[0] not in defined]
    assert not unknown, f"{path.name}: `{line}` uses undefined flag(s) {unknown}"


# Claims corrected during review, each with the shape that was wrong.  A pattern
# here is cheaper than the round trip that found it the first time.
STALE_CLAIMS: tuple[tuple[str, str, str], ...] = (
    (
        "ignore region read as a class",
        r"dense\(\[?65535",
        "the ignore id is not a class; `layers` gives an all-zero plane. "
        "Use ignore_mask().",
    ),
    (
        "worker_init_fn called mandatory",
        r"worker_init_fn[^.]{0,80}?(required|mandatory|not optional|must not omit)",
        "the handle cache is PID-keyed, so the callback is an eager reset, "
        "not a requirement.",
    ),
    (
        "validation described as metadata-only",
        r"touches no voxel|never touches a voxel|costs nothing to run",
        "`structural` and `semantic` do bounded payload scans; "
        "measured 62 ms on 12.6 Mvox.",
    ),
    (
        "glob passed to --source",
        r"--source [^\s]*\*\.dcm",
        "`--source` takes one value per occurrence; a glob is rejected "
        "as extra arguments.",
    ),
    (
        "recompress --out given a directory",
        r"--out [^\s]*/(\s|$)",
        "`--out` is a single destination filename, not a directory.",
    ),
    (
        "PairReport credited with resolving transforms",
        r"aligned by a transform",
        "`PairReport` holds files, pairs and cross-sectional skips; "
        "it resolves nothing.",
    ),
    (
        "change label naming one endpoint",
        r'"response"[^)]*timepoints=\["tp[01]"\]',
        "a label describing an interval names both visits, in acquisition order.",
    ),
    (
        "entry-level coverage used as a loss mask",
        r"e\.annotated_class_ids for e in manifest",
        "`Entry.annotated_class_ids` unions every annotation; a loss mask needs "
        "the one being trained on.",
    ),
    (
        "registration preflight resolved per timepoint",
        r"transform_between\(pair\.first, pair\.second\)",
        "a visit may hold several grids on different frames; resolve between the "
        "grids the images are on.",
    ),
    (
        "inverse_id offered for graph resolution",
        r"store it under `inverse_id`",
        "`inverse_id` adds a second one-hop path and makes the reverse "
        "direction raise E501.",
    ),
)


@pytest.mark.parametrize(
    ("label", "pattern", "why"), STALE_CLAIMS, ids=lambda v: str(v)[:40]
)
def test_corrected_claims_stay_corrected(label: str, pattern: str, why: str) -> None:
    """Each of these was true of the documentation once, and is not true of the code.

    They recur because a correction lands on the page that was reported and not
    on its three siblings.  Instances kept coming back until the class was
    checked, which is what this does.
    """
    # Scoped to the paragraph, not the line: a correction names the wrong form
    # in order to warn against it, and the warning is usually a sentence away
    # from the mention.  Line-scoped matching flags the fix as the defect.
    disclaimed = re.compile(
        r"\bnot\b|instead|rather than|does not|never|finds nothing|all-zero|"
        r"expands to|is rejected|no longer|used to",
        re.I,
    )
    hits: list[str] = []
    for path in [*PAGES, README]:
        text = path.read_text()
        offset = 0
        for para in text.split("\n\n"):
            if re.search(pattern, para, re.I) and not disclaimed.search(para):
                line_no = text.count("\n", 0, offset) + 1
                first = next(
                    (ln for ln in para.splitlines() if re.search(pattern, ln, re.I)),
                    para,
                )
                hits.append(f"{path.name}:{line_no}: {first.strip()[:80]}")
            offset += len(para) + 2
    assert not hits, f"{label}: {why}\n  " + "\n  ".join(hits)
