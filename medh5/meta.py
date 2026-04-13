"""Metadata helpers – plain dataclasses + HDF5 attribute read/write.

All metadata is stored as native HDF5 attributes (scalars, arrays, or
JSON strings) so it is inspectable with ``h5dump`` / ``h5ls`` / HDFView
without any custom library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from medh5.exceptions import MEDH5SchemaError

SCHEMA_VERSION = "1"

_ROOT_META_ATTRS = (
    "schema_version",
    "image_names",
    "label",
    "label_name",
    "has_seg",
    "seg_names",
    "has_bbox",
    "extra",
)

_IMAGE_META_ATTRS = (
    "shape",
    "spacing",
    "origin",
    "direction",
    "axis_labels",
    "coord_system",
    "patch_size",
)


@dataclass
class SpatialMeta:
    """Geometry metadata shared by all images in a sample."""

    spacing: list[float] | None = None
    origin: list[float] | None = None
    direction: list[list[float]] | None = None
    axis_labels: list[str] | None = None
    coord_system: str | None = None


@dataclass
class SampleMeta:
    """Top-level metadata for a single ``.medh5`` sample."""

    spatial: SpatialMeta = field(default_factory=SpatialMeta)
    image_names: list[str] | None = None
    label: int | str | None = None
    label_name: str | None = None
    shape: list[int] | None = None
    has_seg: bool = False
    seg_names: list[str] | None = None
    has_bbox: bool = False
    patch_size: list[int] | None = None
    extra: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __repr__(self) -> str:
        parts = [f"schema_version={self.schema_version!r}"]
        if self.image_names is not None:
            parts.append(f"image_names={self.image_names}")
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        if self.label is not None:
            parts.append(f"label={self.label!r}")
        if self.has_seg and self.seg_names is not None:
            parts.append(f"seg_names={self.seg_names}")
        if self.has_bbox:
            parts.append("has_bbox=True")
        if self.spatial.spacing is not None:
            parts.append(f"spacing={self.spatial.spacing}")
        if self.extra is not None:
            parts.append(f"extra={self.extra!r}")
        return f"SampleMeta({', '.join(parts)})"

    def validate(self, ndim: int | None = None) -> None:
        """Raise on obviously invalid metadata."""
        s = self.spatial
        if s.spacing is not None:
            if not all(isinstance(v, (int, float)) for v in s.spacing):
                raise TypeError("spacing must contain only numbers")
            if ndim is not None and len(s.spacing) != ndim:
                raise ValueError(f"spacing length ({len(s.spacing)}) != ndim ({ndim})")
        if s.origin is not None:
            if not all(isinstance(v, (int, float)) for v in s.origin):
                raise TypeError("origin must contain only numbers")
            if ndim is not None and len(s.origin) != ndim:
                raise ValueError(f"origin length ({len(s.origin)}) != ndim ({ndim})")
        if s.direction is not None:
            for row in s.direction:
                if not all(isinstance(v, (int, float)) for v in row):
                    raise TypeError("direction must contain only numbers")
            if ndim is not None and (
                len(s.direction) != ndim or any(len(row) != ndim for row in s.direction)
            ):
                bad_cols = len(s.direction[0]) if s.direction else 0
                raise ValueError(
                    f"direction must be a {ndim}x{ndim} matrix, "
                    f"got {len(s.direction)}x{bad_cols}"
                )
        if s.axis_labels is not None:
            if not all(isinstance(v, str) for v in s.axis_labels):
                raise TypeError("axis_labels must be strings")
            if ndim is not None and len(s.axis_labels) != ndim:
                raise ValueError(
                    f"axis_labels length ({len(s.axis_labels)}) != ndim ({ndim})"
                )
        if self.patch_size is not None:
            if not all(isinstance(v, int) for v in self.patch_size):
                raise TypeError("patch_size must contain only integers")
            if ndim is not None and len(self.patch_size) != ndim:
                raise ValueError(
                    f"patch_size length ({len(self.patch_size)}) != ndim ({ndim})"
                )
        if self.extra is not None:
            json.dumps(self.extra)  # raises on non-serializable values


# ---------------------------------------------------------------------------
# HDF5 attribute I/O
# ---------------------------------------------------------------------------


def write_meta(f: h5py.File, meta: SampleMeta) -> None:
    """Persist *meta* as HDF5 attributes on *f*.

    Root attributes hold scalar/label/flag metadata.  Shape and spatial
    metadata are stored on the ``images`` group so they apply to every
    modality equally.
    """
    for key in _ROOT_META_ATTRS:
        f.attrs.pop(key, None)
    f.attrs["schema_version"] = meta.schema_version
    if meta.image_names is not None:
        f.attrs["image_names"] = json.dumps(meta.image_names)
    if meta.label is not None:
        f.attrs["label"] = meta.label
    if meta.label_name is not None:
        f.attrs["label_name"] = meta.label_name
    f.attrs["has_seg"] = meta.has_seg
    if meta.seg_names is not None:
        f.attrs["seg_names"] = json.dumps(meta.seg_names)
    f.attrs["has_bbox"] = meta.has_bbox
    if meta.extra is not None:
        f.attrs["extra"] = json.dumps(meta.extra)

    if "images" not in f:
        return
    grp = f["images"]
    for key in _IMAGE_META_ATTRS:
        grp.attrs.pop(key, None)
    if meta.shape is not None:
        grp.attrs["shape"] = np.asarray(meta.shape, dtype=np.int64)
    s = meta.spatial
    if s.spacing is not None:
        grp.attrs["spacing"] = np.asarray(s.spacing, dtype=np.float64)
    if s.origin is not None:
        grp.attrs["origin"] = np.asarray(s.origin, dtype=np.float64)
    if s.direction is not None:
        grp.attrs["direction"] = np.asarray(s.direction, dtype=np.float64).ravel()
    if s.axis_labels is not None:
        grp.attrs["axis_labels"] = s.axis_labels
    if s.coord_system is not None:
        grp.attrs["coord_system"] = s.coord_system
    if meta.patch_size is not None:
        grp.attrs["patch_size"] = np.asarray(meta.patch_size, dtype=np.int64)


def read_meta(f: h5py.File) -> SampleMeta:
    """Reconstruct a :class:`SampleMeta` from HDF5 attributes."""

    spatial = SpatialMeta()
    shape: list[int] | None = None
    patch_size = None

    if "images" in f:
        a = f["images"].attrs
        if "spacing" in a:
            spatial.spacing = a["spacing"].tolist()
        if "origin" in a:
            spatial.origin = a["origin"].tolist()
        if "direction" in a:
            raw = np.asarray(a["direction"])
            first_key = next(iter(f["images"]))
            ndim = len(f["images"][first_key].shape)
            if raw.size != ndim * ndim:
                raise MEDH5SchemaError(
                    f"Malformed 'direction' attribute: expected "
                    f"{ndim * ndim} elements for a {ndim}D volume but "
                    f"got {raw.size}."
                )
            spatial.direction = raw.reshape(ndim, ndim).tolist()
        if "axis_labels" in a:
            spatial.axis_labels = list(a["axis_labels"])
        if "coord_system" in a:
            spatial.coord_system = str(a["coord_system"])
        if "shape" in a:
            shape = a["shape"].tolist()
        if "patch_size" in a:
            patch_size = a["patch_size"].tolist()

    ra = f.attrs

    image_names: list[str] | None = None
    if "image_names" in ra:
        raw_image_names = ra["image_names"]
        if isinstance(raw_image_names, bytes):
            raw_image_names = raw_image_names.decode()
        image_names = json.loads(raw_image_names)

    label = ra.get("label")
    if isinstance(label, bytes):
        label = label.decode()
    elif isinstance(label, np.generic):
        label = label.item()

    label_name = ra.get("label_name")
    if isinstance(label_name, bytes):
        label_name = label_name.decode()

    has_seg = bool(ra.get("has_seg", False))

    seg_names: list[str] | None = None
    if "seg_names" in ra:
        raw_seg_names = ra["seg_names"]
        if isinstance(raw_seg_names, bytes):
            raw_seg_names = raw_seg_names.decode()
        seg_names = json.loads(raw_seg_names)

    has_bbox = bool(ra.get("has_bbox", False))

    extra = None
    if "extra" in ra:
        raw_extra = ra["extra"]
        if isinstance(raw_extra, bytes):
            raw_extra = raw_extra.decode()
        extra = json.loads(raw_extra)

    raw_schema_version = ra.get("schema_version", SCHEMA_VERSION)
    if isinstance(raw_schema_version, bytes):
        raw_schema_version = raw_schema_version.decode()
    schema_version = str(raw_schema_version)

    try:
        schema_version_num = int(schema_version)
        current_schema_num = int(SCHEMA_VERSION)
    except ValueError as exc:
        raise MEDH5SchemaError(
            f"Invalid schema version '{schema_version}'. Expected an integer string."
        ) from exc

    if schema_version_num > current_schema_num:
        raise MEDH5SchemaError(
            f"File has schema version '{schema_version}' but this library "
            f"only supports up to '{SCHEMA_VERSION}'. Upgrade medh5."
        )

    return SampleMeta(
        spatial=spatial,
        image_names=image_names,
        shape=shape,
        label=label,
        label_name=label_name,
        has_seg=has_seg,
        seg_names=seg_names,
        has_bbox=has_bbox,
        patch_size=patch_size,
        extra=extra,
        schema_version=schema_version,
    )
