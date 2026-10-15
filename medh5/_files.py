"""A file written whole: to a temporary beside it, renamed over it.

`write_text` is the engine's replace (`_core.write_atomic`).
"""

from __future__ import annotations

from pathlib import Path


def write_text(target: Path, text: str) -> None:
    """*text* as *target*'s content, as UTF-8 with its ``\n`` kept on every
    platform, through the engine's replace (§14.4): a link at *target* is
    kept and the file it names replaced, with that file's mode."""
    from medh5 import _core

    _core.write_atomic(target, text)
