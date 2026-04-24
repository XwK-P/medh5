"""Ref-counted read-only handle sharing for ``.medh5`` files.

HDF5 refuses to reopen a file that is already open elsewhere in the same
process. Lazy-read consumers (napari plugins, dashboards, viewers) that
want to hand out independent "handles" while keeping the underlying file
single-open end up writing their own reference-counting registries.

:func:`open_shared` ships that registry in the library. The first caller
opens the file; subsequent callers in the same process receive the same
:class:`h5py.File` object. The file is closed only after the last caller
releases its reference.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py

from medh5.exceptions import MEDH5FileError
from medh5.meta import _validate_suffix


@dataclass
class _Entry:
    file: h5py.File
    refcount: int


_lock = threading.RLock()
_registry: dict[Path, _Entry] = {}


def _acquire(path: Path) -> h5py.File:
    key = path.resolve()
    with _lock:
        entry = _registry.get(key)
        if entry is None:
            try:
                handle = h5py.File(str(path), "r")
            except OSError as exc:
                raise MEDH5FileError(f"Failed to open '{path}': {exc}") from exc
            _registry[key] = _Entry(file=handle, refcount=1)
            return handle
        entry.refcount += 1
        return entry.file


def _release(path: Path) -> None:
    key = path.resolve()
    with _lock:
        entry = _registry.get(key)
        if entry is None:
            return
        entry.refcount -= 1
        if entry.refcount <= 0:
            del _registry[key]
            entry.file.close()


@contextmanager
def open_shared(path: str | Path) -> Iterator[h5py.File]:
    """Return a process-shared read-only :class:`h5py.File` handle.

    Multiple callers in the same process (and across threads) share one
    underlying file. The handle stays open until the last caller exits
    its ``with`` block, at which point the file is closed.

    Example::

        with open_shared("sample.medh5") as f:
            patch = f["images/CT"][10:42, 20:84, 20:84]

    The registry is keyed by :func:`Path.resolve`, so symlinks that point
    at the same inode share a handle. Callers must treat the returned
    object as read-only and must not close it themselves.

    Raises
    ------
    MEDH5ValidationError
        If the path has the wrong extension.
    MEDH5FileError
        If the file cannot be opened.
    """
    path = Path(path)
    _validate_suffix(path)
    handle = _acquire(path)
    try:
        yield handle
    finally:
        _release(path)
