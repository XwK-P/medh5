"""0.x → 1.0 migration (spec Appendix B).

The mapping is mostly mechanical; four steps are not, and each is reported per
file so a curator can audit a cohort rather than trust it:

1. **Voxel encoding.**  0.x stored one boolean volume per mask name.  1.0
   measures the overlap graph and picks an encoding (§7.6), which changes the
   size and nothing else.
2. **Box corners.**  0.x boxes were slice-like integers ``[min, max)``; 1.0
   boxes sit at voxel edges, so ``lo = min − 0.5`` and ``hi = max − 0.5``.  That
   is a real half-voxel shift in the numbers, and it is reported as one.
3. **Label set.**  0.x had names, not classes.  Mask names and ``bbox_labels``
   become keys with minted ids, written to a sidecar so a curator can review,
   edit and reapply them cohort-wide before converting the rest.
4. **Grouping.**  A 0.x file is study-scoped and carries no subject key, so the
   default is one sample per file with a single declared ``tp0``.  Nothing about
   time is invented.  ``--group-by subject`` merges files that share a key the
   curator names, and orders them by date when there is one, by mtime otherwise
   — and says which.

Instance correspondence is **never** inferred across merged files: each file's
objects keep independent ids, because asserting that lesion 2 at baseline is
lesion 2 at follow-up would fabricate the tracking ground truth §7.4 exists to
record.

The migration is one-way, and that is deliberate.  A 0.x reader opening a 1.0
file raises on the missing ``schema_version``, which is the correct loud failure.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from medh5 import _core
from medh5.io._legacy_reader import (
    LegacyMeta,
    LegacySample,
    is_legacy,
)
from medh5.io._legacy_reader import read_meta as _read_meta
from medh5.io._legacy_reader import read_sample as _read_sample
from medh5.io.report import ConversionReport

BOX_SHIFT: float = _core.LEGACY_BOX_SHIFT
"""0.x ``[min, max)`` integer boxes sit at voxel edges once shifted (§8.1)."""


def read_legacy(path: str | os.PathLike[str]) -> LegacySample:
    """Read a whole 0.x file (:mod:`medh5.io._legacy_reader`)."""
    return _read_sample(path)


def legacy_meta(path: str | os.PathLike[str]) -> LegacyMeta:
    """Read a 0.x file's metadata without its arrays."""
    return _read_meta(path)


def build_label_set(
    paths: Sequence[str | os.PathLike[str]],
    *,
    report: ConversionReport | None = None,
) -> Any:
    """Mint one label set covering a whole cohort's mask names and box labels.

    Cohort-wide rather than per file: ids minted independently per file would
    make ``liver`` id 1 in one sample and id 2 in the next, which is exactly the
    inconsistency a label set exists to prevent.  Ids an
    ``extra.nnunetv2.labels`` mapping already fixed are reused.
    """
    label_set, notes = _core.legacy_build_label_set([os.fspath(p) for p in paths])
    if report is not None:
        report._extend(notes)
    return label_set


def migrate(
    path: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    label_set: Any = None,
    codec: str = "balanced",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Migrate one 0.x file into one 1.0 sample (Appendix B).

    *report*, when given, is the one written to and returned.
    """
    log = report if report is not None else ConversionReport(converter="migrate")
    fields = _core.legacy_migrate(
        os.fspath(path),
        os.fspath(out),
        label_set=label_set,
        codec=codec,
        report=log,
    )
    return log._update(fields)


def migrate_paths(
    paths: Sequence[str | os.PathLike[str]],
    outdir: str | os.PathLike[str],
    *,
    group_by: str = "study",
    subject_key: str | None = None,
    label_set: Any = None,
    codec: str = "balanced",
) -> ConversionReport:
    """Migrate a cohort, minting one label set for all of it.

    A file that is not 0.x, or is broken, is reported (``unreadable``) and the
    rest of the cohort is migrated.
    """
    return ConversionReport._from_fields(
        _core.legacy_migrate_paths(
            [os.fspath(p) for p in paths],
            os.fspath(outdir),
            group_by=group_by,
            subject_key=subject_key,
            label_set=label_set,
            codec=codec,
        )
    )


def write_sidecar(label_set: Any, path: str | os.PathLike[str]) -> Path:
    """Write the minted label set for review before a cohort-wide migration."""
    return Path(_core.legacy_write_sidecar(label_set, os.fspath(path)))


def load_sidecar(path: str | os.PathLike[str]) -> Any:
    """Read a reviewed label-set sidecar back."""
    return _core.legacy_load_sidecar(os.fspath(path))


__all__ = [
    "BOX_SHIFT",
    "is_legacy",
    "build_label_set",
    "legacy_meta",
    "load_sidecar",
    "migrate",
    "migrate_paths",
    "read_legacy",
    "write_sidecar",
]
