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

import json
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5FileError, MEDH5SchemaError

SUFFIX = ".medh5"
SCHEMA_VERSION = "1"
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


@dataclass
class LegacySample:
    """A whole 0.x file in memory."""

    images: dict[str, npt.NDArray[Any]]
    seg: dict[str, npt.NDArray[np.bool_]]
    bboxes: npt.NDArray[Any] | None
    bbox_scores: npt.NDArray[Any] | None
    bbox_labels: list[str] | None
    meta: LegacyMeta


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _json(value: Any, what: str) -> Any:
    try:
        return json.loads(_text(value))
    except json.JSONDecodeError as exc:
        raise MEDH5SchemaError(f"0.x attribute {what!r} is not JSON: {exc}") from exc


def _open(path: str | os.PathLike[str]) -> h5py.File:
    """Open a 0.x file, refusing anything that is not one.

    A 1.0 file carries ``/meta`` and no ``schema_version``; a 0.x file carries
    ``schema_version`` and an ``images`` group.  Reading either as the other
    produces nonsense, so the discriminator is checked before anything else.
    """
    text = os.fspath(path)
    try:
        handle = h5py.File(text, "r")
    except OSError as exc:
        raise MEDH5FileError(f"cannot open {text!r} as a 0.x file: {exc}") from exc
    try:
        if "meta" in handle:
            raise MEDH5SchemaError(
                f"{text!r} is a 1.0 file (it has `/meta`), not a 0.x file"
            )
        if "images" not in handle or not isinstance(handle["images"], h5py.Group):
            raise MEDH5SchemaError(f"{text!r} has no 0.x `/images` group")
        version = _text(handle.attrs.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise MEDH5SchemaError(
                f"{text!r} declares 0.x schema version {version!r}; this reader "
                f"understands {SCHEMA_VERSION!r}"
            )
    except BaseException:
        handle.close()
        raise
    return handle


def is_legacy(path: str | os.PathLike[str]) -> bool:
    """True when *path* is a readable 0.x file."""
    try:
        _open(path).close()
    except (MEDH5FileError, MEDH5SchemaError):
        return False
    return True


def read_meta(path: str | os.PathLike[str]) -> LegacyMeta:
    """Read 0.x metadata without touching the arrays."""
    with _open(path) as handle:
        return _meta(handle)


def read_sample(path: str | os.PathLike[str]) -> LegacySample:
    """Read a whole 0.x file: images, masks, boxes and metadata."""
    with _open(path) as handle:
        images = {name: handle["images"][name][...] for name in handle["images"]}
        seg: dict[str, npt.NDArray[np.bool_]] = {}
        group = handle.get("seg")
        if isinstance(group, h5py.Group):
            seg = {name: np.asarray(group[name][...], dtype=bool) for name in group}
        boxes = handle["bboxes"][...] if "bboxes" in handle else None
        scores = handle["bbox_scores"][...] if "bbox_scores" in handle else None
        labels = (
            [_text(v) for v in handle["bbox_labels"][...]]
            if "bbox_labels" in handle
            else None
        )
        return LegacySample(
            images=images,
            seg=seg,
            bboxes=boxes,
            bbox_scores=scores,
            bbox_labels=labels,
            meta=_meta(handle),
        )


def _meta(handle: h5py.File) -> LegacyMeta:
    spatial = LegacySpatial()
    meta = LegacyMeta(spatial=spatial)

    group = handle["images"]
    meta.image_names = sorted(group)
    attrs = group.attrs
    if "spacing" in attrs:
        spatial.spacing = [float(v) for v in np.asarray(attrs["spacing"]).ravel()]
    if "origin" in attrs:
        spatial.origin = [float(v) for v in np.asarray(attrs["origin"]).ravel()]
    if "axis_labels" in attrs:
        spatial.axis_labels = [_text(v) for v in attrs["axis_labels"]]
    if "coord_system" in attrs:
        spatial.coord_system = _text(attrs["coord_system"])
    if "shape" in attrs:
        meta.shape = [int(v) for v in np.asarray(attrs["shape"]).ravel()]
    if "patch_size" in attrs:
        meta.patch_size = [int(v) for v in np.asarray(attrs["patch_size"]).ravel()]
    if "direction" in attrs and meta.image_names:
        ndim = len(group[meta.image_names[0]].shape)
        raw = np.asarray(attrs["direction"], dtype=np.float64).ravel()
        if raw.size != ndim * ndim:
            raise MEDH5SchemaError(
                f"0.x `direction` has {raw.size} element(s), not {ndim * ndim} "
                f"for a {ndim}-D volume"
            )
        spatial.direction = raw.reshape(ndim, ndim).tolist()

    root = handle.attrs
    if "label" in root:
        value = root["label"]
        meta.label = value.item() if isinstance(value, np.generic) else _label(value)
    if "label_name" in root:
        meta.label_name = _text(root["label_name"])
    if "extra" in root:
        extra = _json(root["extra"], "extra")
        meta.extra = dict(extra) if isinstance(extra, dict) else {}
    meta.schema_version = _text(root.get("schema_version", SCHEMA_VERSION))

    seg = handle.get("seg")
    meta.seg_names = sorted(seg) if isinstance(seg, h5py.Group) else []
    return meta


def _label(value: Any) -> int | str:
    text = _text(value)
    try:
        return int(text)
    except ValueError:
        return text


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
