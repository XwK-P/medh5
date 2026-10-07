"""Per-worker file handles (spec §14.4).

An open HDF5 file **must not** cross a ``fork``.  The HDF5 library keeps
state behind the file descriptor, and a child that inherits and then uses a
parent's handle produces corrupt reads that look like data errors, not
concurrency errors --- which is why they are usually diagnosed as a broken
dataset months later.  The cache is therefore keyed by PID and abandoned
(never closed --- the descriptors belong to the parent) as soon as it notices
it is running somewhere else.  Abandoning is explicit: each inherited sample
pins its engine handle so that nothing in this process closes the file, and is
kept alive for the life of the process, so no destructor of what it holds runs
either.

Reopening per ``__getitem__`` would be correct but costs an open per patch;
with a 1000-file dataset and 8 workers that is the dominant cost in the
dataloader.  An LRU of open samples per worker removes it while keeping the
number of descriptors bounded.

**Threads share the cache, so it is locked, and a handle in use is never
closed.**  A thread-based loader calls ``__getitem__`` from several threads of
one process; without a lock one thread's eviction could close the ``Sample``
another was reading --- and §14.4 already requires external locking to share a
handle across threads.  :meth:`HandleCache.lease` holds a handle against
eviction for the length of an item, and the datasets read through it; a
cache full of leased handles grows past ``maxsize`` rather than closing one,
and shrinks back as they are released.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

from medh5.sample import Sample, open_sample

DEFAULT_MAXSIZE = 32

_ABANDONED: list[Sample] = []
"""Samples inherited across a fork: pinned, never closed, never collected."""


def _abandon(samples: Iterator[Sample] | list[Sample]) -> None:
    for sample in samples:
        with contextlib.suppress(Exception):  # pragma: no cover - best effort
            sample._handle.abandon()
        _ABANDONED.append(sample)


class HandleCache:
    """PID-scoped, thread-safe LRU cache of open samples."""

    __slots__ = ("_items", "_lock", "_owner_pid", "_pins", "maxsize", "opens")

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self.maxsize = int(maxsize)
        self._items: OrderedDict[str, Sample] = OrderedDict()
        self._pins: dict[str, int] = {}
        self._lock = threading.Lock()
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
            # Abandon, never close: the descriptors are the parent's.  The lock
            # is replaced too --- one held by another of the parent's threads
            # at the fork is held forever in the child.
            _abandon(list(self._items.values()))
            self._lock = threading.Lock()
            self._items = OrderedDict()
            self._pins = {}
            self._owner_pid = pid

    def _acquire(self, key: str, *, pin: bool) -> Sample:
        """The handle for *key*, opened if need be; the caller holds the lock."""
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
        else:
            cached = open_sample(key)
            self.opens += 1
            self._items[key] = cached
        if pin:
            self._pins[key] = self._pins.get(key, 0) + 1
        return cached

    def _overflow(self, keep: str | None = None) -> list[Sample]:
        """Evict least-recently-used idle handles past ``maxsize``; under the lock.

        *keep* is the handle being handed out: evicting it would return a
        closed file to the caller that asked for it.
        """
        evicted: list[Sample] = []
        for key in list(self._items):
            if len(self._items) <= self.maxsize:
                break
            if key == keep or self._pins.get(key):
                continue
            evicted.append(self._items.pop(key))
        return evicted

    @staticmethod
    def _close(samples: list[Sample]) -> None:
        for sample in samples:
            with contextlib.suppress(Exception):  # pragma: no cover - best effort
                sample.close()

    def get(self, path: str | os.PathLike[str]) -> Sample:
        """An open handle, not held against eviction; see :meth:`lease`."""
        self._ensure_owner()
        key = str(Path(path))
        with self._lock:
            sample = self._acquire(key, pin=False)
            evicted = self._overflow(keep=key)
        self._close(evicted)
        return sample

    @contextlib.contextmanager
    def lease(self, path: str | os.PathLike[str]) -> Iterator[Sample]:
        """An open handle that no eviction closes until the block ends."""
        self._ensure_owner()
        key = str(Path(path))
        with self._lock:
            sample = self._acquire(key, pin=True)
            evicted = self._overflow(keep=key)
        self._close(evicted)
        try:
            yield sample
        finally:
            with self._lock:
                remaining = self._pins.get(key, 0) - 1
                if remaining > 0:
                    self._pins[key] = remaining
                else:
                    self._pins.pop(key, None)
                evicted = self._overflow()
            self._close(evicted)

    def resize(self, maxsize: int) -> None:
        """Set ``maxsize``, closing idle handles past it now rather than later."""
        self._ensure_owner()
        with self._lock:
            self.maxsize = int(maxsize)
            evicted = self._overflow()
        self._close(evicted)

    def clear(self) -> None:
        """Drop every handle without closing it --- the post-fork reset."""
        _abandon(list(self._items.values()))
        self._lock = threading.Lock()
        self._items = OrderedDict()
        self._pins = {}
        self._owner_pid = os.getpid()

    def close_all(self) -> None:
        """Close what *this* process opened, and abandon anything inherited.

        The PID check is not optional here.  A forked child that exits through
        normal interpreter shutdown runs the ``atexit`` hook below, and without
        this it would call into HDF5 to close descriptors belonging to its
        parent --- the one thing the module docstring says must never happen.
        ``worker_init_fn`` prevents it only for callers who remember to pass it,
        and a plain ``os.fork()`` never had the chance.
        """
        self._ensure_owner()
        with self._lock:
            samples = list(reversed(self._items.values()))
            self._items = OrderedDict()
            self._pins = {}
        self._close(samples)


CACHE = HandleCache()


@atexit.register
def _close_cache() -> None:  # pragma: no cover - process exit hook
    CACHE.close_all()


def open_cached(path: str | os.PathLike[str]) -> Sample:
    """Open a sample through the per-worker cache."""
    return CACHE.get(path)


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``: drop handles inherited across ``fork``."""
    CACHE.clear()


def set_cache_size(maxsize: int) -> None:
    """Resize this process's cache: how many files each worker keeps open.

    Lower it when file descriptors are scarce; raise it toward the number of
    files a worker cycles through, so a shuffled epoch does not re-open them
    (see :class:`~medh5.torch.FileGroupedSampler` for the other way round).
    Call it in ``worker_init_fn`` to size each worker's cache.
    """
    if int(maxsize) < 1:
        raise ValueError("the handle cache needs room for at least one file")
    CACHE.resize(int(maxsize))


__all__ = [
    "CACHE",
    "DEFAULT_MAXSIZE",
    "HandleCache",
    "open_cached",
    "set_cache_size",
    "worker_init_fn",
]
