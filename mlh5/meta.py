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

SCHEMA_VERSION = "1"


@dataclass
class SpatialMeta:
    """Geometry metadata attached to the *image* dataset."""

    spacing: list[float] | None = None
    origin: list[float] | None = None
    direction: list[list[float]] | None = None
    axis_labels: list[str] | None = None
    coord_system: str | None = None


@dataclass
class SampleMeta:
    """Top-level metadata for a single ``.mlh5`` sample."""

    spatial: SpatialMeta = field(default_factory=SpatialMeta)
    label: int | str | None = None
    label_name: str | None = None
    has_seg: bool = False
    has_bbox: bool = False
    patch_size: list[int] | None = None
    extra: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

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
        if s.axis_labels is not None:
            if not all(isinstance(v, str) for v in s.axis_labels):
                raise TypeError("axis_labels must be strings")
        if self.extra is not None:
            json.dumps(self.extra)  # raises on non-serializable values


# ---------------------------------------------------------------------------
# HDF5 attribute I/O
# ---------------------------------------------------------------------------

def write_meta(f: h5py.File, meta: SampleMeta) -> None:
    """Persist *meta* as HDF5 attributes on *f* (root + ``image`` dataset)."""

    f.attrs["schema_version"] = meta.schema_version
    if meta.label is not None:
        f.attrs["label"] = meta.label
    if meta.label_name is not None:
        f.attrs["label_name"] = meta.label_name
    f.attrs["has_seg"] = meta.has_seg
    f.attrs["has_bbox"] = meta.has_bbox
    if meta.extra is not None:
        f.attrs["extra"] = json.dumps(meta.extra)

    if "image" not in f:
        return
    ds = f["image"]
    s = meta.spatial
    if s.spacing is not None:
        ds.attrs["spacing"] = np.asarray(s.spacing, dtype=np.float64)
    if s.origin is not None:
        ds.attrs["origin"] = np.asarray(s.origin, dtype=np.float64)
    if s.direction is not None:
        ds.attrs["direction"] = np.asarray(s.direction, dtype=np.float64).ravel()
    if s.axis_labels is not None:
        ds.attrs["axis_labels"] = s.axis_labels
    if s.coord_system is not None:
        ds.attrs["coord_system"] = s.coord_system
    if meta.patch_size is not None:
        ds.attrs["patch_size"] = np.asarray(meta.patch_size, dtype=np.int64)


def read_meta(f: h5py.File) -> SampleMeta:
    """Reconstruct a :class:`SampleMeta` from HDF5 attributes."""

    spatial = SpatialMeta()
    patch_size = None

    if "image" in f:
        a = f["image"].attrs
        if "spacing" in a:
            spatial.spacing = a["spacing"].tolist()
        if "origin" in a:
            spatial.origin = a["origin"].tolist()
        if "direction" in a:
            raw = np.asarray(a["direction"])
            ndim = len(f["image"].shape)
            spatial.direction = raw.reshape(ndim, ndim).tolist()
        if "axis_labels" in a:
            spatial.axis_labels = list(a["axis_labels"])
        if "coord_system" in a:
            spatial.coord_system = str(a["coord_system"])
        if "patch_size" in a:
            patch_size = a["patch_size"].tolist()

    ra = f.attrs
    label = ra.get("label")
    if isinstance(label, bytes):
        label = label.decode()
    elif isinstance(label, np.generic):
        label = label.item()

    label_name = ra.get("label_name")
    if isinstance(label_name, bytes):
        label_name = label_name.decode()

    has_seg = bool(ra.get("has_seg", False))
    has_bbox = bool(ra.get("has_bbox", False))

    extra = None
    if "extra" in ra:
        raw_extra = ra["extra"]
        if isinstance(raw_extra, bytes):
            raw_extra = raw_extra.decode()
        extra = json.loads(raw_extra)

    schema_version = str(ra.get("schema_version", SCHEMA_VERSION))
    if isinstance(schema_version, bytes):
        schema_version = schema_version.decode()

    return SampleMeta(
        spatial=spatial,
        label=label,
        label_name=label_name,
        has_seg=has_seg,
        has_bbox=has_bbox,
        patch_size=patch_size,
        extra=extra,
        schema_version=schema_version,
    )
