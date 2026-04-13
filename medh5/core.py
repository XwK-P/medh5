"""Core read/write API for ``.medh5`` files.

A single ``.medh5`` file stores one or more co-registered images (e.g.
different modalities), optional segmentation masks, bounding boxes, and
an image-level label using HDF5 datasets and attributes with Blosc2
compression provided by *hdf5plugin*.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 – registers the Blosc2 filter
import numpy as np

from medh5.chunks import optimize_chunks
from medh5.exceptions import MEDH5FileError, MEDH5ValidationError
from medh5.integrity import _CHECKSUM_ATTR, verify_checksum, write_checksum
from medh5.meta import SampleMeta, SpatialMeta, read_meta, write_meta
from medh5.review import get_review_status, set_review_status

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


@dataclass
class ValidationIssue:
    """A single validation finding."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass
class ValidationReport:
    """Structured validation result for one ``.medh5`` file."""

    path: str
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def ok(self, *, strict: bool = False) -> bool:
        return self.is_valid and (not strict or not self.warnings)

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(ValidationIssue(code=code, message=message))

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(ValidationIssue(code=code, message=message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "is_valid": self.is_valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _coerce_seg_write_kwargs(
    image_ds: h5py.Dataset,
    *,
    cname: str,
    clevel: int,
) -> dict[str, Any]:
    chunks = image_ds.chunks
    blosc2_opts: dict[str, Any] = {**hdf5plugin.Blosc2(cname=cname, clevel=clevel)}
    return {"chunks": chunks, **blosc2_opts}


def _normalize_patch_size(
    patch_size: int | list[int] | tuple[int, ...],
    ref_arr: np.ndarray,
    spatial_axis_mask: list[bool] | None,
) -> tuple[int, ...]:
    if isinstance(patch_size, int):
        spatial_ndim = sum(spatial_axis_mask) if spatial_axis_mask else ref_arr.ndim
        return (patch_size,) * spatial_ndim
    return tuple(patch_size)


def _sync_meta_from_file(f: h5py.File, meta: SampleMeta) -> None:
    img_grp = f["images"]
    image_names = sorted(img_grp.keys())
    first_key = image_names[0]
    ref_shape = list(img_grp[first_key].shape)
    meta.image_names = image_names
    meta.shape = ref_shape

    seg_grp = f.get("seg")
    if isinstance(seg_grp, h5py.Group) and len(seg_grp) > 0:
        meta.has_seg = True
        meta.seg_names = sorted(seg_grp.keys())
    else:
        meta.has_seg = False
        meta.seg_names = None

    meta.has_bbox = "bboxes" in f


def _set_bbox_datasets(
    f: h5py.File,
    ref_shape: tuple[int, ...],
    *,
    bboxes: np.ndarray | None,
    bbox_scores: np.ndarray | None,
    bbox_labels: list[str] | None,
) -> None:
    ndim = len(ref_shape)
    if bboxes is None:
        if bbox_scores is not None or bbox_labels is not None:
            raise MEDH5ValidationError("bbox_scores/bbox_labels require bboxes")
        for key in ("bboxes", "bbox_scores", "bbox_labels"):
            if key in f:
                del f[key]
        return

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

    if "bboxes" in f:
        del f["bboxes"]
    f.create_dataset("bboxes", data=bboxes_arr)

    if "bbox_scores" in f:
        del f["bbox_scores"]
    if bbox_scores is not None:
        f.create_dataset("bbox_scores", data=np.asarray(bbox_scores))

    if "bbox_labels" in f:
        del f["bbox_labels"]
    if bbox_labels is not None:
        dt = h5py.string_dtype()
        f.create_dataset(
            "bbox_labels",
            data=np.array(bbox_labels, dtype=object),
            dtype=dt,
        )


def _validate_open_file(f: h5py.File, path: str | Path) -> ValidationReport:
    report = ValidationReport(path=str(path))
    if "images" not in f:
        report.add_error("missing_images_group", "Missing required 'images' group")
        return report

    img_grp = f["images"]
    if len(img_grp) == 0:
        report.add_error("empty_images_group", "'images' group is empty")
        return report

    image_names = sorted(img_grp.keys())
    first_key = image_names[0]
    ref_shape = img_grp[first_key].shape
    ndim = len(ref_shape)
    for name in image_names[1:]:
        if img_grp[name].shape != ref_shape:
            report.add_error(
                "image_shape_mismatch",
                f"Image '{name}' has shape {img_grp[name].shape}; expected {ref_shape}",
            )

    if "schema_version" not in f.attrs:
        report.add_error("missing_schema_version", "Missing 'schema_version' attribute")

    try:
        meta = read_meta(f)
    except Exception as exc:
        report.add_error("metadata_read_failed", f"Failed to read metadata: {exc}")
        return report

    try:
        meta.validate(ndim=ndim)
    except Exception as exc:
        report.add_error("metadata_invalid", f"Invalid metadata: {exc}")

    if meta.image_names is None:
        report.add_warning("missing_image_names", "Missing 'image_names' metadata")
    elif sorted(meta.image_names) != image_names:
        report.add_error(
            "image_names_mismatch",
            f"Metadata image_names {meta.image_names} != datasets {image_names}",
        )

    if meta.shape is not None and tuple(meta.shape) != ref_shape:
        report.add_error(
            "shape_mismatch",
            f"Metadata shape {meta.shape} != image shape {list(ref_shape)}",
        )

    seg_grp = f.get("seg")
    actual_seg_names = []
    if isinstance(seg_grp, h5py.Group) and len(seg_grp) > 0:
        actual_seg_names = sorted(seg_grp.keys())
    if meta.has_seg != bool(actual_seg_names):
        report.add_error(
            "seg_presence_mismatch",
            "Metadata has_seg="
            f"{meta.has_seg} != actual presence={bool(actual_seg_names)}",
        )
    if meta.seg_names is not None and sorted(meta.seg_names) != actual_seg_names:
        report.add_error(
            "seg_names_mismatch",
            f"Metadata seg_names {meta.seg_names} != datasets {actual_seg_names}",
        )
    if isinstance(seg_grp, h5py.Group):
        for name in actual_seg_names:
            if seg_grp[name].shape != ref_shape:
                report.add_error(
                    "seg_shape_mismatch",
                    "Segmentation "
                    f"'{name}' has shape {seg_grp[name].shape}; expected {ref_shape}",
                )

    has_bbox = "bboxes" in f
    if meta.has_bbox != has_bbox:
        report.add_error(
            "bbox_presence_mismatch",
            f"Metadata has_bbox={meta.has_bbox} != actual presence={has_bbox}",
        )
    if has_bbox:
        bbox_ds = f["bboxes"]
        bbox_shape = bbox_ds.shape
        if len(bbox_shape) != 3 or bbox_shape[1:] != (ndim, 2):
            report.add_error(
                "bbox_shape_invalid",
                f"'bboxes' has shape {bbox_shape}; expected (n, {ndim}, 2)",
            )
        if "bbox_scores" in f and len(f["bbox_scores"]) != bbox_shape[0]:
            report.add_error(
                "bbox_scores_mismatch",
                "'bbox_scores' length must match 'bboxes' count",
            )
        if "bbox_labels" in f and len(f["bbox_labels"]) != bbox_shape[0]:
            report.add_error(
                "bbox_labels_mismatch",
                "'bbox_labels' length must match 'bboxes' count",
            )
    elif "bbox_scores" in f or "bbox_labels" in f:
        report.add_error(
            "bbox_dependents_without_bboxes",
            "'bbox_scores'/'bbox_labels' present without 'bboxes'",
        )

    if _CHECKSUM_ATTR not in f.attrs:
        report.add_warning("missing_checksum", "No checksum stored")
    elif not verify_checksum(f):
        report.add_error(
            "checksum_mismatch", "Stored checksum does not match file contents"
        )

    return report


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

        ps_tuple = _normalize_patch_size(patch_size, ref_arr, spatial_axis_mask)

        chunks = optimize_chunks(
            ref_shape,
            ps_tuple,
            bytes_per_element=ref_arr.dtype.itemsize,
            spatial_axis_mask=spatial_axis_mask,
        )

        blosc2_opts: dict[str, Any] = {
            **hdf5plugin.Blosc2(cname=cname, clevel=clevel),
        }

        # Atomic write: stage to a sibling temp file, fsync, then os.replace
        # into the target. An interrupted write (Ctrl-C, OOM, crash) leaves
        # the destination path untouched; the temp file is unlinked below.
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp_str = str(tmp_path)
        done = False
        try:
            with h5py.File(tmp_str, "w") as f:
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
                    bboxes_arr = np.asarray(bboxes)
                    # Compression only pays off for large bbox arrays;
                    # tiny ones (the common case) pay Blosc2 chunk
                    # overhead for no win, so store them raw.
                    if bboxes_arr.shape[0] > 64:
                        f.create_dataset(
                            "bboxes",
                            data=bboxes_arr,
                            chunks=True,
                            **blosc2_opts,
                        )
                    else:
                        f.create_dataset("bboxes", data=bboxes_arr)
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
                    shape=list(ref_shape),
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

                f.flush()

            # Fsync the staged file before the rename so a crash
            # between os.replace and fs flush leaves either the old
            # file or the fully-written new one — never a torso.
            fd = os.open(tmp_str, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

            os.replace(tmp_str, str(path))
            done = True
        except MEDH5ValidationError:
            raise
        except OSError as exc:
            raise MEDH5FileError(f"Failed to write '{path}': {exc}") from exc
        finally:
            if not done:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp_str)

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
                    if len(seg_grp) > 0:
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
        meta_updates: dict[str, Any] = {}
        if not isinstance(label, _UnsetType):
            meta_updates["label"] = label
        if not isinstance(label_name, _UnsetType):
            meta_updates["label_name"] = label_name
        if not isinstance(extra, _UnsetType):
            meta_updates["extra"] = extra
        MEDH5File.update(path, meta=meta_updates)

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
        MEDH5File.update(
            path,
            seg_ops={"add": {name: mask}, "cname": cname, "clevel": clevel},
        )

    @staticmethod
    def update(
        path: str | Path,
        *,
        meta: dict[str, Any] | None = None,
        seg_ops: dict[str, Any] | None = None,
        bbox_ops: dict[str, Any] | None = None,
        recompute_checksum: bool = True,
        force: bool = False,
    ) -> None:
        """Apply in-place metadata / seg / bbox updates to an existing file.

        If the file has a stored checksum, it is verified **before** any
        mutation.  A mismatch raises :class:`MEDH5FileError` — re-hashing
        over pre-corrupted data would otherwise silently bake bad data
        into the stored digest.  Pass ``force=True`` to bypass the
        pre-verify (e.g. when intentionally repairing a file).
        """
        path = Path(path)
        _validate_suffix(path)
        allowed_meta = {
            "label",
            "label_name",
            "extra",
            "spacing",
            "origin",
            "direction",
            "axis_labels",
            "coord_system",
            "patch_size",
        }
        meta_updates = dict(meta or {})
        unknown_meta = sorted(set(meta_updates) - allowed_meta)
        if unknown_meta:
            raise MEDH5ValidationError(f"Unknown meta update fields: {unknown_meta}")

        seg_updates = dict(seg_ops or {})
        bbox_updates = dict(bbox_ops or {})

        try:
            with h5py.File(str(path), "a") as f:
                if not force and _CHECKSUM_ATTR in f.attrs and not verify_checksum(f):
                    raise MEDH5FileError(
                        f"Refusing to update '{path}': stored checksum does "
                        "not match current contents. Pass force=True to "
                        "override (e.g. when repairing)."
                    )
                img_grp = f["images"]
                first_key = next(iter(img_grp))
                ref_ds = img_grp[first_key]
                ref_shape = ref_ds.shape
                current_meta = read_meta(f)

                for key, value in meta_updates.items():
                    if key in {"label", "label_name", "extra"}:
                        setattr(current_meta, key, value)
                    elif key == "patch_size":
                        if value is None:
                            current_meta.patch_size = None
                        elif isinstance(value, int):
                            current_meta.patch_size = [value] * len(ref_shape)
                        else:
                            current_meta.patch_size = [int(v) for v in value]
                    else:
                        setattr(current_meta.spatial, key, value)

                if seg_updates:
                    cname = str(seg_updates.get("cname", "lz4hc"))
                    clevel = int(seg_updates.get("clevel", 8))
                    write_kwargs = _coerce_seg_write_kwargs(
                        ref_ds, cname=cname, clevel=clevel
                    )
                    seg_grp = f.get("seg")
                    existing_names = (
                        set(seg_grp.keys())
                        if isinstance(seg_grp, h5py.Group)
                        else set()
                    )
                    future_names = set(existing_names)
                    remove_names = list(seg_updates.get("remove", []))
                    prepared_masks: dict[str, tuple[bool, np.ndarray]] = {}

                    for name in remove_names:
                        if name not in future_names:
                            raise MEDH5ValidationError(
                                f"Segmentation mask '{name}' does not exist"
                            )
                        future_names.remove(name)

                    for action in ("add", "replace"):
                        replace = action == "replace"
                        for name, mask in dict(seg_updates.get(action, {})).items():
                            mask_arr = np.ascontiguousarray(mask, dtype=bool)
                            if mask_arr.shape != ref_shape:
                                raise MEDH5ValidationError(
                                    "Mask shape "
                                    f"{mask_arr.shape} does not match image shape "
                                    f"{ref_shape}"
                                )
                            if name in future_names:
                                if not replace:
                                    raise MEDH5ValidationError(
                                        f"Segmentation mask '{name}' already exists"
                                    )
                            elif replace:
                                raise MEDH5ValidationError(
                                    f"Segmentation mask '{name}' does not exist"
                                )
                            else:
                                future_names.add(name)
                            prepared_masks[name] = (replace, mask_arr)

                    if remove_names or prepared_masks:
                        if seg_grp is None and future_names:
                            seg_grp = f.create_group("seg")

                        if isinstance(seg_grp, h5py.Group):
                            for name in remove_names:
                                del seg_grp[name]

                            for name, (replace, mask_arr) in prepared_masks.items():
                                if replace:
                                    del seg_grp[name]
                                mask_chunks = write_kwargs["chunks"]
                                if mask_chunks is not None:
                                    mask_chunks = tuple(
                                        min(c, s)
                                        for c, s in zip(
                                            mask_chunks, mask_arr.shape, strict=True
                                        )
                                    )
                                seg_grp.create_dataset(
                                    name,
                                    data=mask_arr,
                                    chunks=mask_chunks,
                                    **{
                                        k: v
                                        for k, v in write_kwargs.items()
                                        if k != "chunks"
                                    },
                                )

                            if len(seg_grp) == 0:
                                del f["seg"]

                if bbox_updates:
                    clear = bool(bbox_updates.get("clear", False))
                    if clear:
                        _set_bbox_datasets(
                            f,
                            ref_shape,
                            bboxes=None,
                            bbox_scores=None,
                            bbox_labels=None,
                        )
                    else:
                        bboxes = (
                            bbox_updates["bboxes"]
                            if "bboxes" in bbox_updates
                            else (f["bboxes"][...] if "bboxes" in f else None)
                        )
                        bbox_scores = (
                            bbox_updates["bbox_scores"]
                            if "bbox_scores" in bbox_updates
                            else (f["bbox_scores"][...] if "bbox_scores" in f else None)
                        )
                        bbox_labels = (
                            list(f["bbox_labels"].asstr()[...])
                            if "bbox_labels" in f and "bbox_labels" not in bbox_updates
                            else bbox_updates.get("bbox_labels")
                        )
                        if (
                            "bboxes" in bbox_updates
                            or "bbox_scores" in bbox_updates
                            or "bbox_labels" in bbox_updates
                        ):
                            _set_bbox_datasets(
                                f,
                                ref_shape,
                                bboxes=bboxes,
                                bbox_scores=bbox_scores,
                                bbox_labels=bbox_labels,
                            )

                _sync_meta_from_file(f, current_meta)
                current_meta.validate(ndim=len(ref_shape))
                write_meta(f, current_meta)

                if recompute_checksum and (
                    _CHECKSUM_ATTR in f.attrs
                    or meta_updates
                    or seg_updates
                    or bbox_updates
                ):
                    write_checksum(f)
        except MEDH5ValidationError:
            raise
        except MEDH5FileError:
            raise
        except OSError as exc:
            raise MEDH5FileError(f"Failed to update '{path}': {exc}") from exc

    @staticmethod
    def validate(path: str | Path, *, strict: bool = False) -> ValidationReport:
        """Validate a ``.medh5`` file and return a structured report."""
        path = Path(path)
        _validate_suffix(path)
        try:
            with h5py.File(str(path), "r") as f:
                return _validate_open_file(f, path)
        except OSError as exc:
            report = ValidationReport(path=str(path))
            report.add_error("file_open_failed", f"Failed to open '{path}': {exc}")
        return report

    @staticmethod
    def is_valid(path: str | Path) -> bool:
        """Return ``True`` if *path* passes :meth:`validate` with no errors.

        Thin convenience wrapper for the common "is this file OK?"
        check — saves callers from constructing a
        :class:`ValidationReport` just to read its ``is_valid`` flag.
        Returns ``False`` (not raises) when the path is missing, has
        the wrong extension, or cannot be opened, so it can be used
        freely in ``filter(MEDH5File.is_valid, paths)`` style pipelines.
        """
        try:
            return MEDH5File.validate(path).is_valid
        except MEDH5ValidationError:
            return False

    # ------------------------------------------------------------------
    # Review / curation helpers (delegated to medh5.review)
    # ------------------------------------------------------------------

    set_review_status = staticmethod(set_review_status)
    get_review_status = staticmethod(get_review_status)
