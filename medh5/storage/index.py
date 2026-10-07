"""Sampling indices: derived, rebuildable acceleration data (spec §14.3).

An index entry holds per-class voxel counts, boxes, a uniform subsample of
foreground coordinates and an occupancy map, so a patch sampler draws a
foreground-centred patch without scanning the annotation.  Each entry records
the ``source_digest`` of the annotation it was built from; a stale entry is
ignored, never trusted.
"""

from __future__ import annotations

from medh5 import _core

DEFAULT_MAX_COORDS: int = _core.DEFAULT_MAX_COORDS
DEFAULT_OCCUPANCY_FACTOR: int = _core.DEFAULT_OCCUPANCY_FACTOR

SamplingIndex = _core.SamplingIndex
"""One stored index entry: ``voxel_counts()``, ``bbox()``, ``coords()``,
``sample_foreground(class_id, n, rng)``, ``class_weights(mode)``."""

IndexPayload = _core.IndexPayload
build_index = _core.build_index
occupancy = _core.occupancy

__all__ = [
    "DEFAULT_MAX_COORDS",
    "DEFAULT_OCCUPANCY_FACTOR",
    "IndexPayload",
    "SamplingIndex",
    "build_index",
    "occupancy",
]
