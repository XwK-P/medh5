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

1.0 ships no 0.x implementation.  It ships a *reader*, because ``medh5
migrate`` has to open files somebody wrote last year, and because a curator
migrating a cohort should not be able to keep writing the format they are
migrating away from --- which is exactly what shipping the old package inside
the new one would allow.  The 0.x layout is small enough to state in full,
and stating it here is the point: this module is the format's obituary,
written down, rather than a dependency on code that still runs.

    /images/<name>          one dataset per modality, all the same shape
    /seg/<name>             one boolean dataset per mask name
    /bboxes                 (n, ndim, 2) integers, slice-like [min, max)
    /bbox_scores            (n,) floats            (optional)
    /bbox_labels            (n,) strings           (optional)

    root attrs              schema_version, image_names, label, label_name,
                            has_seg, seg_names, has_bbox, extra (JSON)
    /images attrs           shape, spacing, origin, direction (flattened
                            row-major), axis_labels, coord_system, patch_size

Only reading is implemented, and the denormalised flags (``has_seg``,
``seg_names``, ``image_names``) are *ignored* in favour of what the file
actually contains --- 0.x could and did drift between the two, and a migration
that trusts the flag silently drops the data.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.io.report import ConversionReport

SUFFIX = ".medh5"
SCHEMA_VERSION: str = _core.LEGACY_SCHEMA_VERSION
"""The only 0.x schema version that ever shipped."""

BOX_SHIFT: float = _core.LEGACY_BOX_SHIFT
"""0.x ``[min, max)`` integer boxes sit at voxel edges once shifted (§8.1)."""


# -- reading the 0.x layout ------------------------------------------------------


@dataclass
class LegacySpatial:
    """0.x geometry: one grid, shared by every image in the file."""

    spacing: list[float] | None = None
    origin: list[float] | None = None
    direction: list[list[float]] | None = None
    axis_labels: list[str] | None = None
    coord_system: str | None = None


@dataclass
class LegacyMeta:
    """0.x metadata, as far as a migration needs it."""

    spatial: LegacySpatial = field(default_factory=LegacySpatial)
    shape: list[int] | None = None
    image_names: list[str] = field(default_factory=list)
    seg_names: list[str] = field(default_factory=list)
    label: int | str | None = None
    label_name: str | None = None
    patch_size: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def _from_fields(cls, fields: Mapping[str, Any]) -> LegacyMeta:
        values = dict(fields)
        values["spatial"] = LegacySpatial(**values["spatial"])
        return cls(**values)


@dataclass
class LegacySample:
    """A whole 0.x file in memory."""

    images: dict[str, npt.NDArray[Any]]
    seg: dict[str, npt.NDArray[np.bool_]]
    bboxes: npt.NDArray[Any] | None
    bbox_scores: npt.NDArray[Any] | None
    bbox_labels: list[str] | None
    meta: LegacyMeta


def is_legacy(path: str | os.PathLike[str]) -> bool:
    """True when *path* is a readable 0.x file."""
    return bool(_core.legacy_is(os.fspath(path)))


def legacy_meta(path: str | os.PathLike[str]) -> LegacyMeta:
    """A 0.x file's metadata, without its arrays.

    A 1.0 file, a file without 0.x ``/images``, an unknown schema version, a
    malformed ``direction`` or ``extra`` --- each is refused by name rather
    than read as nonsense.
    """
    return LegacyMeta._from_fields(_core.legacy_read_meta(os.fspath(path)))


def read_legacy(path: str | os.PathLike[str]) -> LegacySample:
    """A whole 0.x file.  The file beats its own denormalised flags: masks
    are what ``/seg`` holds, whatever ``has_seg`` and ``seg_names`` say."""
    fields = dict(_core.legacy_read_sample(os.fspath(path)))
    fields["meta"] = LegacyMeta._from_fields(fields["meta"])
    return LegacySample(**fields)


# -- migrating -------------------------------------------------------------------------


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
    "SCHEMA_VERSION",
    "SUFFIX",
    "LegacyMeta",
    "LegacySample",
    "LegacySpatial",
    "build_label_set",
    "is_legacy",
    "legacy_meta",
    "load_sidecar",
    "migrate",
    "migrate_paths",
    "read_legacy",
    "write_sidecar",
]
