"""A file written whole: to a temporary beside it, renamed over it."""

from __future__ import annotations

import contextlib
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
    """*text* as *target*'s content, through a temporary renamed over it: a
    link at *target* is replaced, not written through."""
    temporary, fd = temporary_beside(target)
    try:
        with open(fd, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
