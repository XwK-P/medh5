"""Build-time hooks for the documentation site.

Two jobs, both in service of one rule: **a fact belongs in exactly one place.**

`CHANGELOG.md` lives at the repository root, because that is where packaging,
GitHub and every convention for finding one expect it.  It is also worth having
on the documentation site, and copying it into ``docs/`` would leave two files
to drift apart --- exactly the failure the rest of this project is built to
avoid.  So it is added to the build here instead: one source, rendered twice.

Relocating a file relocates its links.  At the repository root a link into the
documentation reads ``docs/spec/medh5-1.0.md``; on the site that same page is
``spec/medh5-1.0.md``, because the documentation *is* the site root.  The prefix
is rewritten on the way in, so `CHANGELOG.md` stays correct on GitHub and the
copy served here resolves too.  `mkdocs build --strict` fails if it does not.

The second job is the tables.  The diagnostic codes, the cohort check codes and
the sample-document schema are all *already* defined precisely somewhere in the
repository, and were all *also* being retyped by hand into prose pages --- in
two different wordings, in the case of the C1xx codes.  `medh5/errors.py` says
in its own docstring that "the validator, the conformance corpus and the
documentation all read it rather than repeating literals"; that was true of the
first two and false of the third.  It is true of all three now: pages carry a
marker comment, and the table is rendered from the source at build time.

Reading those sources costs the build nothing, which matters because
`docs/requirements.txt` deliberately does not install `medh5` --- the site is
hand-written Markdown and a build should not need h5py, torch or pydicom.  So
`medh5/errors.py` is loaded as a standalone module (it imports only the standard
library) and `medh5/dataset/check.py`, which does pull h5py transitively, is
*parsed* rather than imported.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File, Files

# (path relative to the repository root, path it is served at on the site)
INCLUDED: tuple[tuple[str, str], ...] = (("CHANGELOG.md", "changelog.md"),)

# The sample-document schema, published verbatim so that it has a stable URL to
# cite and to download, next to the page that explains it.
SCHEMA_SOURCE = "schemas/medh5-sample-1.0.schema.json"
SCHEMA_TARGET = "reference/medh5-sample-1.0.schema.json"

# A Markdown link target beginning `docs/`, which is repository-root-relative.
_DOCS_LINK = re.compile(r"(?<=]\()docs/")

# A Markdown link target into `design/`, which is in the repository but *not* on
# the site: those documents are historical records of the 1.0 design and are
# deliberately not published.  On the site the link has to leave for GitHub.
_OFF_SITE_LINK = re.compile(r"(?<=]\()(design/[^)]+)")


def _relocated(markdown: str, repo_url: str) -> str:
    """Rewrite repository-root-relative doc links for a page served at the site root."""
    markdown = _DOCS_LINK.sub("", markdown)
    return _OFF_SITE_LINK.sub(rf"{repo_url.rstrip('/')}/blob/main/\1", markdown)


# --------------------------------------------------------------------------
# Reading the sources of truth without installing the package
# --------------------------------------------------------------------------


def _load_errors(root: Path) -> Any:
    """Import ``medh5/errors.py`` as a standalone module.

    It imports only ``dataclasses``, ``typing`` and ``__future__``, so it loads
    with no h5py, no numpy and no installed package.  Two details are not
    optional:

    * the module name is **not** ``medh5.errors``.  That name would make Python
      import the parent package first, which does pull h5py in, defeating the
      whole point.
    * it is registered in ``sys.modules`` *before* execution.  ``Code`` is a
      ``@dataclass(slots=True)``, and building a slots dataclass resolves its
      annotations through ``sys.modules[cls.__module__]``; with the module
      absent that lookup returns ``None`` and the import dies in `dataclasses`.
    """
    path = root / "medh5" / "errors.py"
    if not path.is_file():
        raise FileNotFoundError(f"the diagnostic code table is not at {path}")
    spec = importlib.util.spec_from_file_location("_medh5_errors_for_docs", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cohort_codes(root: Path) -> dict[str, str]:
    """``CHECK_CODES`` from ``medh5/dataset/check.py``, read as a literal.

    That module imports `medh5.dataset.manifest`, which needs h5py, so it is
    parsed rather than imported --- the table itself is a plain dict of string
    literals and `ast` can read it without running anything.
    """
    path = root / "medh5" / "dataset" / "check.py"
    if not path.is_file():
        raise FileNotFoundError(f"the cohort check table is not at {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CHECK_CODES" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):  # pragma: no cover - defensive
                raise TypeError(f"CHECK_CODES is a {type(value).__name__}, not a dict")
            return value
    raise ValueError(
        f"CHECK_CODES is no longer a module-level literal in {path}; the cohort "
        "check reference page cannot be generated from it"
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

# `|` would end a table cell.  `{` is worse and less obvious: `attr_list` is
# enabled, and it treats a trailing `{...}` as an attribute block --- which is
# exactly the shape of E003's summary, `identifier does not match
# [A-Za-z0-9_.-]{1,128}`.  Unescaped, that code's meaning silently disappears
# from the rendered table and becomes an HTML attribute.
_CELL_ESCAPES = str.maketrans({"|": r"\|", "{": r"\{", "}": r"\}"})


def _cell(text: str) -> str:
    """Escape one table cell's worth of prose."""
    return text.translate(_CELL_ESCAPES)


_DOMAIN_TITLES: tuple[tuple[str, str], ...] = (
    ("container", "Container"),
    ("geometry", "Geometry"),
    ("images", "Images"),
    ("labels", "Label set"),
    ("annotations", "Annotations"),
    ("transforms", "Transforms"),
    ("curation", "Curation"),
    ("integrity", "Integrity"),
)


def _codes_markdown(root: Path) -> str:
    """The §15.2 diagnostic code table, grouped by domain."""
    errors = _load_errors(root)
    codes = errors.CODES
    warnings = sum(c.severity == "warning" for c in codes.values())
    out: list[str] = [
        f"{len(codes)} codes: **{len(codes) - warnings} errors** and "
        f"**{warnings} warnings**. Every one has a conformance case.",
        "",
    ]
    seen = 0
    for domain, title in _DOMAIN_TITLES:
        rows = errors.codes_for(domain)
        if not rows:  # pragma: no cover - every domain is populated today
            continue
        seen += len(rows)
        out += [
            f"### {title}",
            "",
            "| Code | Severity | Meaning |",
            "|---|---|---|",
        ]
        out += [
            f"| `{c.code}`{{ #{c.code.lower()} }} | {c.severity} | {_cell(c.summary)} |"
            for c in rows
        ]
        out.append("")
    if seen != len(codes):
        raise ValueError(
            f"{len(codes) - seen} diagnostic code(s) are in a domain this page does "
            f"not render; add it to _DOMAIN_TITLES in {__file__}"
        )
    return "\n".join(out)


def _cohort_codes_markdown(root: Path) -> str:
    """The C1xx--C5xx cohort check table."""
    codes = _cohort_codes(root)
    out = ["| Code | Meaning |", "|---|---|"]
    out += [
        f"| `{code}`{{ #{code.lower()} }} | {_cell(summary)} |"
        for code, summary in sorted(codes.items())
    ]
    return "\n".join(out)


def _type_of(schema: dict[str, Any]) -> str:
    """A one-cell description of what a property holds."""
    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        return f"[`{name}`](#{name.lower()})"
    kind = schema.get("type", "")
    if kind == "array":
        item = schema.get("items", {})
        return f"array of {_type_of(item)}" if item else "array"
    return f"`{kind}`" if kind else ""


_CONSTRAINTS = (
    "const",
    "enum",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "minLength",
    "minItems",
    "maxItems",
)


def _constraints_of(schema: dict[str, Any]) -> str:
    """Every constraint keyword the schema actually uses, as one cell."""
    parts: list[str] = []
    for key in _CONSTRAINTS:
        if key not in schema:
            continue
        value = schema[key]
        if key == "enum":
            parts.append(" \\| ".join(f"`{v}`" for v in value))
        elif key == "const":
            parts.append(f"`{value}`")
        else:
            parts.append(f"{key} `{value}`")
    return _cell("; ".join(parts)) if parts else ""


def _property_table(schema: dict[str, Any]) -> list[str]:
    """A `name | type | required | constraints | description` table."""
    required = set(schema.get("required", ()))
    rows = [
        "| Property | Type | Required | Constraints | Description |",
        "|---|---|---|---|---|",
    ]
    for name, prop in schema.get("properties", {}).items():
        rows.append(
            f"| `{name}` | {_type_of(prop)} "
            f"| {'yes' if name in required else ''} "
            f"| {_constraints_of(prop)} "
            f"| {_cell(prop.get('description', ''))} |"
        )
    return rows


def _schema_markdown(root: Path) -> str:
    """The sample document schema, as prose tables plus the document itself."""
    path = root / SCHEMA_SOURCE
    if not path.is_file():
        raise FileNotFoundError(f"the sample-document schema is not at {path}")
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    defs = doc.get("$defs", {})

    out: list[str] = ["## The document", ""]
    out += _property_table(doc)
    out += [
        "",
        "`additionalProperties` is `false` at every level: an unrecognised key "
        "is a schema failure, not a silently ignored extension. Use `extra` for "
        "anything the schema does not define.",
        "",
        "## Definitions",
        "",
        f"{len(defs)} shared definitions, referenced by `$ref` above and by each "
        "other.",
        "",
    ]
    for name, sub in defs.items():
        out += [f"### {name}", ""]
        if description := sub.get("description"):
            out += [description, ""]
        if "properties" in sub:
            out += _property_table(sub)
        else:
            out += [
                "| Type | Constraints |",
                "|---|---|",
                f"| {_type_of(sub)} | {_constraints_of(sub)} |",
            ]
        out.append("")

    out += [
        "## The schema itself",
        "",
        f"Published verbatim at [`{SCHEMA_TARGET.rsplit('/', 1)[-1]}`]"
        f"({SCHEMA_TARGET.rsplit('/', 1)[-1]}), and shipped inside the package.",
        "",
        "```json",
        raw.rstrip(),
        "```",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Substitution, with a loud failure if a marker goes missing
# --------------------------------------------------------------------------

_MARKERS: dict[str, Any] = {
    "<!--@diagnostic-codes-->": _codes_markdown,
    "<!--@cohort-codes-->": _cohort_codes_markdown,
    "<!--@schema-->": _schema_markdown,
}

_SEEN: dict[str, int] = dict.fromkeys(_MARKERS, 0)


def _root(config: Any) -> Path:
    return Path(config["config_file_path"]).parent


def on_files(files: Files, config: Any) -> Files:
    """Add repository-root files to the build without copying them into `docs/`."""
    root = _root(config)
    repo_url = config.get("repo_url") or ""
    for source, destination in INCLUDED:
        origin = root / source
        if not origin.is_file():
            # Loud, not silent: a missing source would otherwise produce a site
            # with a nav entry leading nowhere, and `strict` only checks links
            # between pages it already knows about.
            raise FileNotFoundError(
                f"{source} is listed in hooks/mkdocs_hooks.py:INCLUDED but is not in "
                f"the repository at {origin}"
            )
        content = _relocated(origin.read_text(encoding="utf-8"), repo_url)
        files.append(
            File.generated(config, destination, content=content.encode("utf-8"))
        )

    schema = root / SCHEMA_SOURCE
    if not schema.is_file():
        raise FileNotFoundError(f"the sample-document schema is not at {schema}")
    files.append(File.generated(config, SCHEMA_TARGET, content=schema.read_bytes()))
    return files


def on_page_markdown(markdown: str, page: Any, config: Any, files: Files) -> str:
    """Replace each table marker with the table rendered from its source."""
    root = _root(config)
    for marker, render in _MARKERS.items():
        if marker in markdown:
            _SEEN[marker] += 1
            markdown = markdown.replace(marker, render(root))
    return markdown


def on_post_build(config: Any) -> None:
    """Fail if a generated table stopped being generated.

    `strict` catches a link to a page that vanished.  It cannot catch a *table*
    that vanished: a page whose marker was deleted or misspelled still builds,
    still renders, and is simply missing the content it exists to carry.  So the
    count is checked here instead.
    """
    missing = sorted(marker for marker, count in _SEEN.items() if count != 1)
    if missing:
        raise RuntimeError(
            f"generated-table marker(s) {missing} did not appear exactly once "
            f"across the site (counts: { {m: _SEEN[m] for m in missing} }). A page "
            "was renamed, deleted, or had its marker edited, and its table "
            "silently vanished."
        )
    for marker in _SEEN:
        _SEEN[marker] = 0
