"""Cache-aware chunk sizing (spec §14.1).

HDF5 decompresses a whole chunk to serve any element of it, so the chunk is
the real unit of I/O.  Sizing it to the L3 cache keeps a patch read inside
cache after decompression; sizing it near the training patch keeps read
amplification low.  The engine resolves the two by starting at the patch,
growing toward the cache budget, and stopping before the chunk is much larger
than the patch.
"""

from __future__ import annotations

from medh5 import _core

DEFAULT_L3_BYTES: int = _core.DEFAULT_L3_BYTES
MIN_CHUNK_BYTES: float = _core.MIN_CHUNK_BYTES
MAX_CHUNK_BYTES: float = _core.MAX_CHUNK_BYTES
CACHE_SAFETY: float = _core.CACHE_SAFETY
OVERSHOOT_LIMIT: float = _core.OVERSHOOT_LIMIT
DEFAULT_PATCH: int = _core.DEFAULT_PATCH

detect_l3_bytes = _core.detect_l3_bytes
spatial_chunk_for = _core.spatial_chunk_for
optimize_chunks = _core.optimize_chunks
grid_chunks = _core.grid_chunks
fit_chunks = _core.fit_chunks
field_chunks = _core.field_chunks
chunk_report = _core.chunk_report

__all__ = [
    "CACHE_SAFETY",
    "DEFAULT_L3_BYTES",
    "DEFAULT_PATCH",
    "MAX_CHUNK_BYTES",
    "MIN_CHUNK_BYTES",
    "OVERSHOOT_LIMIT",
    "chunk_report",
    "detect_l3_bytes",
    "field_chunks",
    "fit_chunks",
    "grid_chunks",
    "optimize_chunks",
    "spatial_chunk_for",
]
