"""Per-worker file handles (spec §14.4).

An open ``h5py.File`` **must not** cross a ``fork``.  The HDF5 library keeps
state behind the file descriptor, and a child that inherits and then uses a
parent's handle produces corrupt reads that look like data errors, not
concurrency errors --- which is why they are usually diagnosed as a broken
dataset months later.  The cache is therefore keyed by PID and abandoned
(never closed --- the descriptors belong to the parent) as soon as it notices
it is running somewhere else.

Reopening per ``__getitem__`` would be correct but costs an open per patch;
with a 1000-file dataset and 8 workers that is the dominant cost in the
dataloader.  An LRU of open samples per worker removes it while keeping the
number of descriptors bounded.
"""

from __future__ import annotations

import atexit
import contextlib
import os
from collections import OrderedDict
from pathlib import Path

from medh5.sample import Sample, open_sample

DEFAULT_MAXSIZE = 32


class HandleCache:
    """PID-scoped LRU cache of open samples."""

    __slots__ = ("_items", "_owner_pid", "maxsize", "opens")

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self.maxsize = int(maxsize)
        self._items: OrderedDict[str, Sample] = OrderedDict()
        self._owner_pid = os.getpid()
        self.opens = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    def _ensure_owner(self) -> None:
        pid = os.getpid()
        if pid != self._owner_pid:
            # Abandon, never close: the descriptors are the parent's.
            self._items = OrderedDict()
            self._owner_pid = pid

    def get(self, path: str | os.PathLike[str]) -> Sample:
        self._ensure_owner()
        key = str(Path(path))
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached
        sample = open_sample(key)
        self.opens += 1
        self._items[key] = sample
        while len(self._items) > self.maxsize:
            _, evicted = self._items.popitem(last=False)
            with contextlib.suppress(Exception):  # pragma: no cover - best effort
                evicted.close()
        return sample

    def clear(self) -> None:
        """Drop every handle without closing it --- the post-fork reset."""
        self._items = OrderedDict()
        self._owner_pid = os.getpid()

    def close_all(self) -> None:
        while self._items:
            _, sample = self._items.popitem(last=True)
            with contextlib.suppress(Exception):  # pragma: no cover - best effort
                sample.close()


CACHE = HandleCache()


@atexit.register
def _close_cache() -> None:  # pragma: no cover - process exit hook
    CACHE.close_all()


def open_cached(path: str | os.PathLike[str]) -> Sample:
    """Open a sample through the per-worker cache."""
    return CACHE.get(path)


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """``DataLoader(worker_init_fn=...)``: drop handles inherited across ``fork``."""
    CACHE.clear()


def set_cache_size(maxsize: int) -> None:
    """Resize the cache; useful when file descriptors are scarce."""
    CACHE.maxsize = int(maxsize)


__all__ = [
    "CACHE",
    "DEFAULT_MAXSIZE",
    "HandleCache",
    "open_cached",
    "set_cache_size",
    "worker_init_fn",
]
