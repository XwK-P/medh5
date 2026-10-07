"""A self-contained reader for the 0.x on-disk layout.

1.0 ships no 0.x implementation.  It ships a *reader*, because ``medh5
migrate`` has to open files somebody wrote last year, and because a curator
migrating a cohort should not be able to keep writing the format they are
migrating away from --- which is exactly what shipping the old package inside
the new one would allow.

The 0.x layout is small enough to state in full, and stating it here is the
point: this module is the format's obituary, written down, rather than a
dependency on code that still runs.

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core

SUFFIX = ".medh5"
SCHEMA_VERSION: str = _core.LEGACY_SCHEMA_VERSION
"""The only 0.x schema version that ever shipped."""


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


def read_meta(path: str | os.PathLike[str]) -> LegacyMeta:
    """A 0.x file's metadata, without its arrays.

    A 1.0 file, a file without 0.x ``/images``, an unknown schema version, a
    malformed ``direction`` or ``extra`` --- each is refused by name rather
    than read as nonsense.
    """
    return LegacyMeta._from_fields(_core.legacy_read_meta(os.fspath(path)))


def read_sample(path: str | os.PathLike[str]) -> LegacySample:
    """A whole 0.x file.  The file beats its own denormalised flags: masks
    are what ``/seg`` holds, whatever ``has_seg`` and ``seg_names`` say."""
    fields = dict(_core.legacy_read_sample(os.fspath(path)))
    fields["meta"] = LegacyMeta._from_fields(fields["meta"])
    return LegacySample(**fields)


__all__ = [
    "SCHEMA_VERSION",
    "SUFFIX",
    "LegacyMeta",
    "LegacySample",
    "LegacySpatial",
    "is_legacy",
    "read_meta",
    "read_sample",
]
