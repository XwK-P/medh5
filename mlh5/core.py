"""Core read/write API for ``.mlh5`` files.

A single ``.mlh5`` file stores image + segmentation + bounding boxes +
image-level label using HDF5 datasets and attributes with Blosc2
compression provided by *hdf5plugin*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 – registers the Blosc2 filter
import numpy as np

from mlh5.chunks import optimize_chunks
from mlh5.meta import SampleMeta, SpatialMeta, read_meta, write_meta

_SUFFIX = ".mlh5"


@dataclass
class MLH5Sample:
    """In-memory representation of everything in a ``.mlh5`` file."""

    image: np.ndarray
    seg: np.ndarray | None
    bboxes: np.ndarray | None
    bbox_scores: np.ndarray | None
    bbox_labels: list[str] | None
    meta: SampleMeta


class MLH5File:
    """Static helpers for reading / writing ``.mlh5`` files.

    Usage::

        MLH5File.write("sample.mlh5", image=arr, seg=seg, label=1)
        sample = MLH5File.read("sample.mlh5")
        f = MLH5File.open("sample.mlh5")  # lazy h5py.File
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def write(
        path: str | Path,
        image: np.ndarray,
        *,
        seg: np.ndarray | None = None,
        bboxes: np.ndarray | None = None,
        bbox_scores: np.ndarray | None = None,
        bbox_labels: list[str] | None = None,
        label: int | str | None = None,
        label_name: str | None = None,
        spacing: list[float] | None = None,
        origin: list[float] | None = None,
        direction: list[list[float]] | None = None,
        axis_labels: list[str] | None = None,
        coord_system: str | None = None,
        patch_size: int | list[int] | tuple[int, ...] = 192,
        spatial_axis_mask: list[bool] | None = None,
        extra: dict[str, Any] | None = None,
        cname: str = "lz4hc",
        clevel: int = 8,
    ) -> None:
        """Write a complete sample to a ``.mlh5`` file.

        Parameters
        ----------
        path : str or Path
            Destination file path (must end with ``.mlh5``).
        image : np.ndarray
            Image array (required).
        seg : np.ndarray, optional
            Segmentation mask with the same spatial shape as *image*.
        bboxes : np.ndarray, optional
            Bounding boxes shaped ``(n, ndims, 2)``  (min/max per axis).
        bbox_scores : np.ndarray, optional
            Confidence scores aligned with *bboxes*.
        bbox_labels : list[str], optional
            String labels aligned with *bboxes*.
        label : int or str, optional
            Image-level classification label.
        label_name : str, optional
            Human-readable name for *label*.
        spacing, origin, direction, axis_labels, coord_system :
            Spatial metadata written as HDF5 attributes on the image
            dataset.
        patch_size : int or sequence of ints
            Spatial patch size for chunk optimization (default 192).
        spatial_axis_mask : list[bool], optional
            Per-axis boolean mask (True = spatial).  Defaults to all-True.
        extra : dict, optional
            Arbitrary JSON-serializable metadata.
        cname : str
            Blosc2 compressor name (``'lz4hc'``, ``'zstd'``, …).
        clevel : int
            Compression level 0-9.
        """
        path = Path(path)
        if path.suffix != _SUFFIX:
            raise ValueError(f"File must have '{_SUFFIX}' extension, got '{path.suffix}'")

        image = np.ascontiguousarray(image)

        if isinstance(patch_size, int):
            spatial_ndim = (
                sum(spatial_axis_mask) if spatial_axis_mask else image.ndim
            )
            ps_tuple = (patch_size,) * spatial_ndim
        else:
            ps_tuple = tuple(patch_size)

        chunks = optimize_chunks(
            image.shape,
            ps_tuple,
            bytes_per_element=image.dtype.itemsize,
            spatial_axis_mask=spatial_axis_mask,
        )

        blosc2_opts: dict[str, Any] = {
            **hdf5plugin.Blosc2(cname=cname, clevel=clevel),
        }

        with h5py.File(str(path), "w") as f:
            f.create_dataset(
                "image",
                data=image,
                chunks=chunks,
                **blosc2_opts,
            )

            if seg is not None:
                seg = np.ascontiguousarray(seg)
                seg_chunks = tuple(
                    min(c, s) for c, s in zip(chunks, seg.shape)
                )
                f.create_dataset(
                    "seg",
                    data=seg,
                    chunks=seg_chunks,
                    **blosc2_opts,
                )

            if bboxes is not None:
                f.create_dataset("bboxes", data=np.asarray(bboxes))
            if bbox_scores is not None:
                f.create_dataset("bbox_scores", data=np.asarray(bbox_scores))
            if bbox_labels is not None:
                dt = h5py.string_dtype()
                f.create_dataset(
                    "bbox_labels",
                    data=np.array(bbox_labels, dtype=object),
                    dtype=dt,
                )

            meta = SampleMeta(
                spatial=SpatialMeta(
                    spacing=list(spacing) if spacing is not None else None,
                    origin=list(origin) if origin is not None else None,
                    direction=(
                        [list(row) for row in direction]
                        if direction is not None
                        else None
                    ),
                    axis_labels=(
                        list(axis_labels) if axis_labels is not None else None
                    ),
                    coord_system=coord_system,
                ),
                label=label,
                label_name=label_name,
                has_seg=seg is not None,
                has_bbox=bboxes is not None,
                patch_size=list(ps_tuple),
                extra=extra,
            )
            meta.validate(ndim=image.ndim)
            write_meta(f, meta)

    # ------------------------------------------------------------------
    # Read (eager)
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> MLH5Sample:
        """Read all contents of a ``.mlh5`` file into memory.

        Returns an :class:`MLH5Sample` with numpy arrays and metadata.
        """
        path = Path(path)
        if path.suffix != _SUFFIX:
            raise ValueError(f"File must have '{_SUFFIX}' extension")

        with h5py.File(str(path), "r") as f:
            image = f["image"][...]

            seg = None
            if "seg" in f:
                seg = f["seg"][...]

            bboxes = None
            if "bboxes" in f:
                bboxes = f["bboxes"][...]

            bbox_scores = None
            if "bbox_scores" in f:
                bbox_scores = f["bbox_scores"][...]

            bbox_labels = None
            if "bbox_labels" in f:
                raw = f["bbox_labels"][...]
                bbox_labels = [
                    v.decode() if isinstance(v, bytes) else str(v) for v in raw
                ]

            meta = read_meta(f)

        return MLH5Sample(
            image=image,
            seg=seg,
            bboxes=bboxes,
            bbox_scores=bbox_scores,
            bbox_labels=bbox_labels,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Open (lazy)
    # ------------------------------------------------------------------

    @staticmethod
    def open(path: str | Path, mode: str = "r") -> h5py.File:
        """Open a ``.mlh5`` file for lazy / partial access.

        Returns a raw :class:`h5py.File`.  The caller is responsible for
        closing it (or using it as a context manager).

        Example::

            with MLH5File.open("sample.mlh5") as f:
                patch = f["image"][10:20, 50:60, 50:60]
        """
        path = Path(path)
        if path.suffix != _SUFFIX:
            raise ValueError(f"File must have '{_SUFFIX}' extension")
        return h5py.File(str(path), mode)

    # ------------------------------------------------------------------
    # Metadata-only read
    # ------------------------------------------------------------------

    @staticmethod
    def read_meta(path: str | Path) -> SampleMeta:
        """Read only metadata without loading array data."""
        path = Path(path)
        if path.suffix != _SUFFIX:
            raise ValueError(f"File must have '{_SUFFIX}' extension")

        with h5py.File(str(path), "r") as f:
            return read_meta(f)
