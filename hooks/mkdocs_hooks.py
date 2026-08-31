"""Build-time hooks for the documentation site.

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
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File, Files

# (path relative to the repository root, path it is served at on the site)
INCLUDED: tuple[tuple[str, str], ...] = (("CHANGELOG.md", "changelog.md"),)

# A Markdown link target beginning `docs/`, which is repository-root-relative.
_DOCS_LINK = re.compile(r"(?<=]\()docs/")


def _relocated(markdown: str) -> str:
    """Rewrite repository-root-relative doc links for a page served at the site root."""
    return _DOCS_LINK.sub("", markdown)


def on_files(files: Files, config: Any) -> Files:
    """Add repository-root files to the build without copying them into `docs/`."""
    root = Path(config["config_file_path"]).parent
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
        content = _relocated(origin.read_text(encoding="utf-8"))
        files.append(
            File.generated(config, destination, content=content.encode("utf-8"))
        )
    return files
