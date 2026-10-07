"""``SampleWriter``, ``create`` and ``amend`` --- writing a sample (spec §14.4).

The write model: ``create`` writes to a sibling temporary file and atomically
replaces the target on ``commit``, so a reader never sees a half-written
sample and a crash leaves the previous file intact.  ``amend`` is
copy-on-write: it builds a new file from the old one and replaces it, because
HDF5 does not reclaim space on delete --- and anything holding the old file
open keeps reading the old inode.

Every ``add_*`` validates as it goes, and ``commit`` refuses to write a file
the validator would reject.  The builder is the format engine's
``SampleWriter``; this module is its Python face::

    with medh5.create("case.medh5", sample_id="case-001") as w:
        w.add_grid("ct", shape=(64, 64, 32), spacing=(0.8, 0.8, 2.0))
        w.add_image("CT", volume, grid="ct", modality="CT")
"""

from __future__ import annotations

from medh5 import _core

SampleWriter = _core.SampleWriter
create = _core.create
amend = _core.amend

MANAGED_ROOT_ATTRS: tuple[str, ...] = _core.MANAGED_ROOT_ATTRS
"""Root attributes ``commit`` writes itself; an amend does not copy them."""

STANDARD_GROUPS: tuple[str, ...] = _core.STANDARD_GROUPS

__all__ = ["MANAGED_ROOT_ATTRS", "STANDARD_GROUPS", "SampleWriter", "amend", "create"]
