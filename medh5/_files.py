"""A file written whole: to a temporary beside it, renamed over it.

`write_text` is the engine's replace (`_core.write_atomic`).
`temporary_beside` is the Python copy the engine's replaced; it stays while
`TaskManifest.save` and `write_nifti` still use it, and goes with the last
of them.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

_EXCLUSIVE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


def temporary_beside(target: Path) -> tuple[Path, int]:
    """A new, empty file beside *target*, and a descriptor that writes it.

    Created exclusively, under a name with a random part: a link or a file
    already at the name fails the creation rather than being written
    through, as one named for the process id alone let anyone who could
    write the directory arrange.  The name ends in *target*'s, so a writer
    that picks a format by suffix (nibabel) picks *target*'s.
    """
    while True:
        name = f".{os.getpid()}-{secrets.token_hex(4)}-{target.name}"
        temporary = target.with_name(name)
        try:
            return temporary, os.open(temporary, _EXCLUSIVE, 0o666)
        except FileExistsError:
            continue


def write_text(target: Path, text: str) -> None:
    """*text* as *target*'s content, as UTF-8 with its ``\n`` kept on every
    platform, through the engine's replace (§14.4): a link at *target* is
    kept and the file it names replaced, with that file's mode."""
    from medh5 import _core

    _core.write_atomic(target, text)
