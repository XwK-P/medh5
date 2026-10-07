"""Grids: the named sampling lattices a sample's arrays live on (spec §3.1-§3.2).

A grid is an **empty HDF5 group carrying only attributes**.  That makes grids
essentially free, which is what lets the format say "two acquisitions with the
same lattice in different timepoints are two grids, not one shared grid" ---
keeping ``timepoint`` and ``frame_uid`` single-valued per grid and removing any
need for per-image geometry overrides.

:class:`Grid` is the format engine's grid: its checks (orthonormal direction,
positive spacing, axis kinds) and its index/world mappings are the engine's.
Grids are written with ``SampleWriter.add_grid`` and read from
``Sample.grids``.
"""

from __future__ import annotations

from medh5._core import AXIS_KINDS, KNOWN_UNITS, SPEC_GRID_ATTRS, TIME_UNITS, Grid

MIN_SPATIAL, MAX_SPATIAL = 2, 3

__all__ = [
    "AXIS_KINDS",
    "KNOWN_UNITS",
    "MAX_SPATIAL",
    "MIN_SPATIAL",
    "SPEC_GRID_ATTRS",
    "TIME_UNITS",
    "Grid",
]
