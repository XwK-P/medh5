"""Cache-aware chunk sizing (spec §14.1).

HDF5 decompresses a whole chunk to serve any element of it, so the chunk is the
real unit of I/O.  Sizing it to the L3 cache keeps a patch read inside cache
after decompression; sizing it near the training patch keeps read amplification
low.  Those two pull in opposite directions, and this module resolves them the
way the 0.x optimizer did --- start at the patch, grow toward the cache budget,
stop before the chunk is much larger than the patch.

The decomposition differs from 0.x: :func:`spatial_chunk_for` answers only the
spatial question, and :func:`optimize_chunks` composes that answer with the
non-spatial axes.  That is what makes the spec's ``(1, *spatial_chunk)``
requirement for ``layers``/``bitmask``/``probmap``/``displacement`` a one-liner
rather than a special case inside the optimizer.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from medh5.errors import MEDH5ValidationError

DEFAULT_L3_BYTES = 1_441_792
"""Fallback L3 slice, ~1.375 MiB (Intel Xeon Silver 4110)."""

MIN_CHUNK_BYTES = 512 * 1024
MAX_CHUNK_BYTES = 4 * 1024 * 1024
CACHE_SAFETY = 0.8
OVERSHOOT_LIMIT = 1.5
"""Stop growing once the mean chunk/patch ratio exceeds this."""

DEFAULT_PATCH = 64


@lru_cache(maxsize=1)
def detect_l3_bytes() -> int:
    """Best-effort L3-per-core detection, falling back to :data:`DEFAULT_L3_BYTES`.

    sysfs is read first because it answers on Linux without leaving the process.
    ``sysctl`` is asked only on Darwin, where sysfs does not exist, and through
    ``subprocess.run`` rather than ``os.popen``: the latter forks ``/bin/sh`` on
    every platform before the read that would have answered, and signals failure
    by returning empty output rather than by raising the ``OSError`` the handler
    was catching.  Forking is worth avoiding on principle in a package whose
    handle cache exists because HDF5 state must not cross one.
    """
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index3/size") as fh:
            raw = fh.read().strip()
        if raw.endswith("K"):
            return int(raw[:-1]) * 1024
        if raw.endswith("M"):
            return int(raw[:-1]) * 1024 * 1024
        if raw.isdigit():
            return int(raw)
    except (OSError, ValueError):  # pragma: no cover - platform dependent
        pass
    if sys.platform == "darwin":  # pragma: no cover - platform dependent
        try:
            probe = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.l3cachesize"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            raw = probe.stdout.strip()
            if raw.isdigit() and int(raw) > 0:
                return int(raw)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return DEFAULT_L3_BYTES


def _budget(l3_bytes: int | None) -> float:
    target = (l3_bytes if l3_bytes is not None else detect_l3_bytes()) * CACHE_SAFETY
    return float(min(max(target, MIN_CHUNK_BYTES), MAX_CHUNK_BYTES))


def spatial_chunk_for(
    spatial_shape: Sequence[int],
    patch: Sequence[int] | int | None = None,
    *,
    itemsize: int = 4,
    l3_bytes: int | None = None,
) -> tuple[int, ...]:
    """Chunk extents for the spatial axes alone.

    Starts at ``2**ceil(log2(patch))`` per axis, then grows the axis with the
    smallest chunk/patch ratio until the cache budget is reached, the array
    extent is hit, or the mean ratio would exceed :data:`OVERSHOOT_LIMIT`.
    """
    shape = tuple(int(s) for s in spatial_shape)
    if not shape or any(s <= 0 for s in shape):
        raise MEDH5ValidationError(f"spatial shape must be positive, got {shape!r}")

    if patch is None:
        patch_t = tuple(min(DEFAULT_PATCH, s) for s in shape)
    elif isinstance(patch, int):
        patch_t = tuple(min(patch, s) for s in shape)
    else:
        patch_t = tuple(int(p) for p in patch)
        if len(patch_t) != len(shape):
            raise MEDH5ValidationError(
                f"patch {patch_t!r} does not match spatial shape {shape!r}"
            )
    patch_t = tuple(max(1, min(p, s)) for p, s in zip(patch_t, shape, strict=True))

    chunk = [
        min(2 ** max(0, math.ceil(math.log2(p))), s)
        for p, s in zip(patch_t, shape, strict=True)
    ]
    budget = _budget(l3_bytes)

    def nbytes(c: Sequence[int]) -> float:
        return float(np.prod(c, dtype=np.float64)) * itemsize

    while nbytes(chunk) < budget:
        growable = [
            i for i, (c, s) in enumerate(zip(chunk, shape, strict=True)) if c < s
        ]
        if not growable:
            break
        axis = min(growable, key=lambda i: chunk[i] / patch_t[i])
        step = 2 ** max(0, math.ceil(math.log2(patch_t[axis])))
        grown = list(chunk)
        grown[axis] = min(grown[axis] + step, shape[axis])
        ratios = [c / p for c, p in zip(grown, patch_t, strict=True)]
        if float(np.mean(ratios)) > OVERSHOOT_LIMIT:
            break
        chunk = grown

    return tuple(int(min(c, s)) for c, s in zip(chunk, shape, strict=True))


def optimize_chunks(
    shape: Sequence[int],
    axis_kinds: Sequence[str],
    patch: Sequence[int] | int | None = None,
    *,
    itemsize: int = 4,
    l3_bytes: int | None = None,
    leading: int = 0,
) -> tuple[int, ...]:
    """Full chunk shape for an array whose axes are described by *axis_kinds*.

    Non-spatial axes (``channel``, ``time``, ``other``) get extent 1: they are
    addressed independently far more often than they are read together, and a
    chunk that spans them forces a reader wanting one channel to decompress all
    of them.  *leading* prepends that many extra size-1 axes, which is how the
    ``(1, *spatial_chunk)`` requirement for stacked encodings is expressed.
    """
    shape_t = tuple(int(s) for s in shape)
    kinds = tuple(str(k) for k in axis_kinds)
    if len(kinds) + leading != len(shape_t):
        raise MEDH5ValidationError(
            f"axis_kinds {kinds!r} (+{leading} leading) does not describe shape "
            f"{shape_t!r}"
        )
    prefix = shape_t[:leading]
    body = shape_t[leading:]
    spatial_idx = [i for i, k in enumerate(kinds) if k == "spatial"]
    spatial_shape = tuple(body[i] for i in spatial_idx)
    spatial_chunk = spatial_chunk_for(
        spatial_shape, patch, itemsize=itemsize, l3_bytes=l3_bytes
    )
    out = [1] * len(body)
    for slot, i in enumerate(spatial_idx):
        out[i] = spatial_chunk[slot]
    return (1,) * len(prefix) + tuple(out)


def chunk_report(
    shape: Sequence[int], chunks: Sequence[int], itemsize: int
) -> dict[str, Any]:
    """Diagnostics for ``medh5 info``: chunk bytes, count, and per-axis fit."""
    chunk_bytes = int(np.prod(chunks, dtype=np.int64)) * itemsize
    n_chunks = int(
        np.prod(
            [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)],
            dtype=np.int64,
        )
    )
    return {
        "chunks": tuple(int(c) for c in chunks),
        "chunk_bytes": chunk_bytes,
        "chunk_mib": round(chunk_bytes / 1024 / 1024, 3),
        "n_chunks": n_chunks,
    }


__all__ = [
    "CACHE_SAFETY",
    "DEFAULT_L3_BYTES",
    "MAX_CHUNK_BYTES",
    "MIN_CHUNK_BYTES",
    "OVERSHOOT_LIMIT",
    "chunk_report",
    "detect_l3_bytes",
    "optimize_chunks",
    "spatial_chunk_for",
]
