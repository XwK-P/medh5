"""Core read/write API for ``.medh5`` files.

A single ``.medh5`` file stores one or more co-registered images (e.g.
different modalities), optional segmentation masks, bounding boxes, and
an image-level label using HDF5 datasets and attributes with Blosc2
compression provided by *hdf5plugin*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 – registers the Blosc2 filter
import numpy as np

from medh5.chunks import optimize_chunks
from medh5.exceptions import MEDH5FileError, MEDH5ValidationError
from medh5.integrity import verify_checksum, write_checksum
from medh5.meta import SampleMeta, SpatialMeta, read_meta, write_meta

_SUFFIX = ".medh5"


class _UnsetType:
    """Sentinel type used so ``None`` can be distinguished from 'not provided'."""

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _UnsetType()

_COMPRESSION_PRESETS: dict[str, tuple[str, int]] = {
    "fast": ("lz4", 3),
    "balanced": ("lz4hc", 8),
    "max": ("zstd", 9),
}


def _validate_suffix(path: Path) -> None:
    if path.suffix != _SUFFIX:
        raise MEDH5ValidationError(
            f"File must have '{_SUFFIX}' extension, got '{path.suffix}'"
        )


def _validate_write_inputs(
    images: dict[str, np.ndarray],
    seg: dict[str, np.ndarray] | None,
    bboxes: np.ndarray | None,
    bbox_scores: np.ndarray | None,
    bbox_labels: list[str] | None,
    clevel: int,
) -> tuple[int, ...]:
    """Validate all write inputs and return the common image shape."""
    if not images:
        raise MEDH5ValidationError("images dict must contain at least one entry")

    image_names = sorted(images.keys())
    ref_shape: tuple[int, ...] | None = None
    for name in image_names:
        arr = images[name]
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise MEDH5ValidationError(
                f"All images must share the same shape. "
                f"'{image_names[0]}' has shape {ref_shape} but "
                f"'{name}' has shape {arr.shape}"
            )
    assert ref_shape is not None

    if not (0 <= clevel <= 9):
        raise MEDH5ValidationError(f"clevel must be 0-9, got {clevel}")

    ndim = len(ref_shape)

    if seg is not None:
        for seg_name, mask in seg.items():
            if mask.shape != ref_shape:
                raise MEDH5ValidationError(
                    f"Segmentation mask '{seg_name}' has shape {mask.shape} "
                    f"but images have shape {ref_shape}"
                )

    if bboxes is not None:
        bboxes_arr = np.asarray(bboxes)
        if bboxes_arr.ndim != 3 or bboxes_arr.shape[1:] != (ndim, 2):
            raise MEDH5ValidationError(
                f"bboxes must have shape (n, {ndim}, 2), got {bboxes_arr.shape}"
            )
        n_boxes = bboxes_arr.shape[0]

        if bbox_scores is not None and len(bbox_scores) != n_boxes:
            raise MEDH5ValidationError(
                f"bbox_scores length ({len(bbox_scores)}) must match "
                f"bboxes count ({n_boxes})"
            )
        if bbox_labels is not None and len(bbox_labels) != n_boxes:
            raise MEDH5ValidationError(
                f"bbox_labels length ({len(bbox_labels)}) must match "
                f"bboxes count ({n_boxes})"
            )

    return ref_shape


@dataclass
class MEDH5Sample:
    """In-memory representation of everything in a ``.medh5`` file."""

    images: dict[str, np.ndarray]
    seg: dict[str, np.ndarray] | None
    bboxes: np.ndarray | None
    bbox_scores: np.ndarray | None
    bbox_labels: list[str] | None
    meta: SampleMeta

    def __repr__(self) -> str:
        names = sorted(self.images)
        first = self.images[names[0]]
        parts = [
            f"modalities={names}",
            f"shape={first.shape}",
            f"dtype={first.dtype}",
        ]
        if self.seg is not None:
            parts.append(f"seg={sorted(self.seg)}")
        if self.meta.label is not None:
            parts.append(f"label={self.meta.label!r}")
        return f"MEDH5Sample({', '.join(parts)})"


class MEDH5File:
    """Read / write ``.medh5`` files.

    Can be used as static helpers or as a context manager for lazy access::

        # Static helpers
        MEDH5File.write("sample.medh5", images={"CT": ct, "PET": pet}, label=1)
        sample = MEDH5File.read("sample.medh5")

        # Context-manager for lazy / partial reads
        with MEDH5File("sample.medh5") as f:
            meta = f.meta
            patch = f.images["CT"][10:20, ...]
            seg_patch = f.seg["tumor"][10:20, ...]
    """

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        path = Path(path)
        _validate_suffix(path)
        try:
            self._h5: h5py.File = h5py.File(str(path), mode)
        except OSError as exc:
            raise MEDH5FileError(f"Failed to open '{path}': {exc}") from exc
        self._path = path

    def __enter__(self) -> MEDH5File:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._h5.close()

    @property
    def meta(self) -> SampleMeta:
        """Read metadata from the open file."""
        return read_meta(self._h5)

    @property
    def images(self) -> h5py.Group:
        """The ``images`` HDF5 group (dict-like, supports slicing)."""
        return self._h5["images"]

    @property
    def seg(self) -> h5py.Group | None:
        """The ``seg`` HDF5 group, or *None* if absent."""
        return self._h5.get("seg")

    @property
    def h5(self) -> h5py.File:
        """The underlying raw :class:`h5py.File`."""
        return self._h5

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def write(
        path: str | Path,
        images: dict[str, np.ndarray],
        *,
        seg: dict[str, np.ndarray] | None = None,
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
        compression: str | None = None,
        cname: str = "lz4hc",
        clevel: int = 8,
        checksum: bool = False,
    ) -> None:
        """Write a complete sample to a ``.medh5`` file.

        Parameters
        ----------
        path : str or Path
            Destination file path (must end with ``.medh5``).
        images : dict[str, np.ndarray]
            Named image arrays (at least one required).  All arrays must
            share the same shape.  Keys are modality names (e.g.
            ``"CT"``, ``"MRI_T1"``, ``"PET"``).
        seg : dict[str, np.ndarray], optional
            Named binary segmentation masks.  Keys are mask names, values
            are bool arrays with the same spatial shape as the images.
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
            Spatial metadata written as HDF5 attributes on the ``images``
            group (shared by all modalities).
        patch_size : int or sequence of ints
            Spatial patch size for chunk optimization (default 192).
        spatial_axis_mask : list[bool], optional
            Per-axis boolean mask (True = spatial).  Defaults to all-True.
        extra : dict, optional
            Arbitrary JSON-serializable metadata.
        compression : str, optional
            Named compression preset: ``"fast"`` (lz4, level 3),
            ``"balanced"`` (lz4hc, level 8), or ``"max"`` (zstd, level 9).
            Overrides *cname* and *clevel* when given.
        cname : str
            Blosc2 compressor name (``'lz4hc'``, ``'zstd'``, …).
        clevel : int
            Compression level 0-9.
        checksum : bool
            If *True*, compute and store a SHA-256 checksum of image
            data for later verification with :meth:`verify`.

        Raises
        ------
        MEDH5ValidationError
            If any inputs fail validation (shape mismatch, bad extension,
            empty images dict, inconsistent bbox/score/label counts, or
            clevel out of range).
        """
        if compression is not None:
            if compression not in _COMPRESSION_PRESETS:
                raise MEDH5ValidationError(
                    f"Unknown compression preset '{compression}'. "
                    f"Choose from: {sorted(_COMPRESSION_PRESETS)}"
                )
            cname, clevel = _COMPRESSION_PRESETS[compression]

        path = Path(path)
        _validate_suffix(path)

        ref_shape = _validate_write_inputs(
            images,
            seg,
            bboxes,
            bbox_scores,
            bbox_labels,
            clevel,
        )

        image_names = sorted(images.keys())
        prepared: dict[str, np.ndarray] = {
            name: np.ascontiguousarray(images[name]) for name in image_names
        }
        ref_arr = prepared[image_names[0]]

        if isinstance(patch_size, int):
            spatial_ndim = sum(spatial_axis_mask) if spatial_axis_mask else ref_arr.ndim
            ps_tuple = (patch_size,) * spatial_ndim
        else:
            ps_tuple = tuple(patch_size)

        chunks = optimize_chunks(
            ref_shape,
            ps_tuple,
            bytes_per_element=ref_arr.dtype.itemsize,
            spatial_axis_mask=spatial_axis_mask,
        )

        blosc2_opts: dict[str, Any] = {
            **hdf5plugin.Blosc2(cname=cname, clevel=clevel),
        }

        try:
            with h5py.File(str(path), "w") as f:
                img_grp = f.create_group("images")
                for name in image_names:
                    img_grp.create_dataset(
                        name,
                        data=prepared[name],
                        chunks=chunks,
                        **blosc2_opts,
                    )

                seg_names: list[str] | None = None
                if seg is not None:
                    seg_grp = f.create_group("seg")
                    seg_names = sorted(seg.keys())
                    for name in seg_names:
                        mask = np.ascontiguousarray(seg[name], dtype=bool)
                        mask_chunks = tuple(
                            min(c, s) for c, s in zip(chunks, mask.shape, strict=True)
                        )
                        seg_grp.create_dataset(
                            name,
                            data=mask,
                            chunks=mask_chunks,
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
                    image_names=image_names,
                    label=label,
                    label_name=label_name,
                    has_seg=seg is not None,
                    seg_names=seg_names,
                    has_bbox=bboxes is not None,
                    patch_size=list(ps_tuple),
                    extra=extra,
                )
                meta.validate(ndim=ref_arr.ndim)
                write_meta(f, meta)

                if checksum:
                    write_checksum(f)
        except MEDH5ValidationError:
            raise
        except OSError as exc:
            raise MEDH5FileError(f"Failed to write '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Read (eager)
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> MEDH5Sample:
        """Read all contents of a ``.medh5`` file into memory.

        Returns an :class:`MEDH5Sample` with numpy arrays and metadata.

        Raises
        ------
        MEDH5ValidationError
            If the file extension is wrong.
        MEDH5FileError
            If the file cannot be opened or is corrupt.
        MEDH5SchemaError
            If the file's schema version is unsupported.
        """
        path = Path(path)
        _validate_suffix(path)

        try:
            with h5py.File(str(path), "r") as f:
                img_grp = f["images"]
                images = {name: img_grp[name][...] for name in img_grp}

                seg = None
                if "seg" in f:
                    seg_grp = f["seg"]
                    seg = {name: seg_grp[name][...] for name in seg_grp}

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
        except OSError as exc:
            raise MEDH5FileError(f"Failed to read '{path}': {exc}") from exc

        return MEDH5Sample(
            images=images,
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
        """Open a ``.medh5`` file for lazy / partial access.

        Returns a raw :class:`h5py.File`.  The caller is responsible for
        closing it (or using it as a context manager).

        Example::

            with MEDH5File.open("sample.medh5") as f:
                patch = f["images/CT"][10:20, 50:60, 50:60]

        Raises
        ------
        MEDH5ValidationError
            If the file extension is wrong.
        MEDH5FileError
            If the file cannot be opened.
        """
        path = Path(path)
        _validate_suffix(path)
        try:
            return h5py.File(str(path), mode)
        except OSError as exc:
            raise MEDH5FileError(f"Failed to open '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Metadata-only read
    # ------------------------------------------------------------------

    @staticmethod
    def read_meta(path: str | Path) -> SampleMeta:
        """Read only metadata without loading array data.

        Raises
        ------
        MEDH5ValidationError
            If the file extension is wrong.
        MEDH5FileError
            If the file cannot be opened.
        MEDH5SchemaError
            If the file's schema version is unsupported.
        """
        path = Path(path)
        _validate_suffix(path)

        try:
            with h5py.File(str(path), "r") as f:
                return read_meta(f)
        except OSError as exc:
            raise MEDH5FileError(f"Failed to read '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    @staticmethod
    def verify(path: str | Path) -> bool:
        """Verify the checksum of a ``.medh5`` file.

        Returns *True* if the stored checksum matches the data, or if
        no checksum was stored (opt-in).  Returns *False* if the data
        has been corrupted.
        """
        path = Path(path)
        _validate_suffix(path)

        try:
            with h5py.File(str(path), "r") as f:
                return verify_checksum(f)
        except OSError as exc:
            raise MEDH5FileError(f"Failed to verify '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # In-place modifications
    # ------------------------------------------------------------------

    @staticmethod
    def update_meta(
        path: str | Path,
        *,
        label: int | str | None | _UnsetType = _UNSET,
        label_name: str | None | _UnsetType = _UNSET,
        extra: dict[str, Any] | None | _UnsetType = _UNSET,
    ) -> None:
        """Update metadata attributes on an existing file.

        Only the provided keyword arguments are updated; omitted fields
        are left unchanged.  Uses an ``_UNSET`` sentinel so that ``None``
        can be written explicitly.

        Raises
        ------
        MEDH5ValidationError
            If the file extension is wrong.
        MEDH5FileError
            If the file cannot be opened.
        """
        import json

        path = Path(path)
        _validate_suffix(path)

        try:
            with h5py.File(str(path), "a") as f:
                if not isinstance(label, _UnsetType):
                    if label is None:
                        f.attrs.pop("label", None)
                    else:
                        f.attrs["label"] = label
                if not isinstance(label_name, _UnsetType):
                    if label_name is None:
                        f.attrs.pop("label_name", None)
                    else:
                        f.attrs["label_name"] = label_name
                if not isinstance(extra, _UnsetType):
                    if extra is None:
                        f.attrs.pop("extra", None)
                    else:
                        f.attrs["extra"] = json.dumps(extra)
        except OSError as exc:
            raise MEDH5FileError(f"Failed to update '{path}': {exc}") from exc

    @staticmethod
    def add_seg(
        path: str | Path,
        name: str,
        mask: np.ndarray,
        *,
        cname: str = "lz4hc",
        clevel: int = 8,
    ) -> None:
        """Add a segmentation mask to an existing ``.medh5`` file.

        Parameters
        ----------
        path : str or Path
            Path to the existing file.
        name : str
            Name for the new segmentation mask.
        mask : np.ndarray
            Boolean mask array (must match image shape).
        cname, clevel :
            Blosc2 compression parameters.

        Raises
        ------
        MEDH5ValidationError
            If the mask shape doesn't match the image shape or if
            a mask with the same name already exists.
        MEDH5FileError
            If the file cannot be opened.
        """
        import json

        path = Path(path)
        _validate_suffix(path)

        mask = np.ascontiguousarray(mask, dtype=bool)

        try:
            with h5py.File(str(path), "a") as f:
                img_grp = f["images"]
                first_key = next(iter(img_grp))
                ref_shape = img_grp[first_key].shape

                if mask.shape != ref_shape:
                    raise MEDH5ValidationError(
                        f"Mask shape {mask.shape} does not match "
                        f"image shape {ref_shape}"
                    )

                if "seg" not in f:
                    seg_grp = f.create_group("seg")
                    f.attrs["has_seg"] = True
                else:
                    seg_grp = f["seg"]

                if name in seg_grp:
                    raise MEDH5ValidationError(
                        f"Segmentation mask '{name}' already exists"
                    )

                blosc2_opts: dict[str, Any] = {
                    **hdf5plugin.Blosc2(cname=cname, clevel=clevel),
                }

                chunks = img_grp[first_key].chunks
                if chunks is not None:
                    mask_chunks = tuple(
                        min(c, s) for c, s in zip(chunks, mask.shape, strict=True)
                    )
                else:
                    mask_chunks = None

                seg_grp.create_dataset(
                    name,
                    data=mask,
                    chunks=mask_chunks,
                    **blosc2_opts,
                )

                existing_names: list[str] = []
                if "seg_names" in f.attrs:
                    raw = f.attrs["seg_names"]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    existing_names = json.loads(raw)
                existing_names.append(name)
                existing_names.sort()
                f.attrs["seg_names"] = json.dumps(existing_names)
        except MEDH5ValidationError:
            raise
        except OSError as exc:
            raise MEDH5FileError(f"Failed to update '{path}': {exc}") from exc
