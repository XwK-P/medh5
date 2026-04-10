"""DICOM series → medh5 converter.

Requires ``pydicom`` (install with ``pip install medh5[dicom]``).

The converter reads a directory of single-frame DICOM files belonging to
one series, sorts them along the through-plane axis using
``ImagePositionPatient`` projected onto the slice normal, and stacks them
into a single 3D volume. Spatial metadata is reconstructed from
``ImageOrientationPatient``, ``ImagePositionPatient``, and ``PixelSpacing``.

Selected DICOM tags can be preserved under ``extra["dicom"]`` for
downstream curation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5ValidationError

try:
    import pydicom

    _PYDICOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDICOM_AVAILABLE = False


_DEFAULT_TAGS: tuple[str, ...] = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "StudyDate",
    "Manufacturer",
)


def _require_pydicom() -> None:
    if not _PYDICOM_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "pydicom is required for DICOM ingestion. "
            "Install it with: pip install medh5[dicom]"
        )


def _read_series(
    dicom_dir: Path,
) -> tuple[np.ndarray, list[float], list[float], list[list[float]], Any]:
    """Load every DICOM file under *dicom_dir*, sort, and stack.

    Returns ``(volume, spacing, origin, direction, first_dataset)``.
    """
    paths = sorted(p for p in dicom_dir.rglob("*") if p.is_file())
    if not paths:
        raise MEDH5ValidationError(f"No DICOM files found under '{dicom_dir}'")

    datasets: list[Any] = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False)
        except Exception:  # pragma: no cover - pydicom raises many things
            continue
        if not hasattr(ds, "PixelData"):
            continue
        datasets.append(ds)

    if not datasets:
        raise MEDH5ValidationError(
            f"No DICOM files with PixelData found under '{dicom_dir}'"
        )

    first = datasets[0]
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        # Fall back to identity orientation
        row_cosine = np.array([1.0, 0.0, 0.0])
        col_cosine = np.array([0.0, 1.0, 0.0])
    else:
        row_cosine = np.asarray(iop[:3], dtype=np.float64)
        col_cosine = np.asarray(iop[3:], dtype=np.float64)
    slice_normal = np.cross(row_cosine, col_cosine)

    def _slice_position(ds: Any) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None or len(ipp) != 3:
            return 0.0
        return float(np.dot(np.asarray(ipp, dtype=np.float64), slice_normal))

    datasets.sort(key=_slice_position)
    sorted_first = datasets[0]

    arrays = [np.asarray(ds.pixel_array) for ds in datasets]
    ref_shape = arrays[0].shape
    for i, arr in enumerate(arrays):
        if arr.shape != ref_shape:
            raise MEDH5ValidationError(
                f"In-plane shape mismatch in DICOM series at slice {i}: "
                f"{arr.shape} vs reference {ref_shape}"
            )
    volume = np.stack(arrays, axis=0)

    pixel_spacing = getattr(sorted_first, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) != 2:
        row_spacing = 1.0
        col_spacing = 1.0
    else:
        row_spacing = float(pixel_spacing[0])
        col_spacing = float(pixel_spacing[1])

    if len(datasets) >= 2:
        p0 = _slice_position(datasets[0])
        p1 = _slice_position(datasets[1])
        slice_spacing = abs(p1 - p0) or float(
            getattr(sorted_first, "SliceThickness", 1.0) or 1.0
        )
    else:
        slice_spacing = float(getattr(sorted_first, "SliceThickness", 1.0) or 1.0)

    spacing = [slice_spacing, row_spacing, col_spacing]

    ipp = getattr(sorted_first, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        origin = [float(v) for v in ipp]
    else:
        origin = [0.0, 0.0, 0.0]

    direction = [
        slice_normal.tolist(),
        row_cosine.tolist(),
        col_cosine.tolist(),
    ]

    return volume, spacing, origin, direction, sorted_first


def from_dicom(
    dicom_dir: str | Path,
    out_path: str | Path,
    *,
    modality_name: str = "CT",
    extra_tags: list[str] | None = None,
    label: int | str | None = None,
    label_name: str | None = None,
    extra: dict[str, Any] | None = None,
    compression: str | None = "balanced",
    checksum: bool = False,
) -> None:
    """Convert a single DICOM series directory into a ``.medh5`` file.

    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing single-frame DICOM files for one series.
        Files are auto-sorted along the slice normal.
    out_path : str or Path
        Destination ``.medh5`` file.
    modality_name : str
        Modality key under ``images/`` (default ``"CT"``).
    extra_tags : list[str], optional
        DICOM tag names to copy into ``extra["dicom"]``. Defaults to a
        small set of identifying tags (PatientID, study/series UIDs,
        Modality, StudyDate, Manufacturer).
    label, label_name :
        Image-level label/name forwarded to :meth:`MEDH5File.write`.
    extra : dict, optional
        User-supplied extra metadata. Merged with the DICOM tag dump under
        the top-level key (DICOM tags live at ``extra["dicom"]``).
    compression : str
        Compression preset (default ``"balanced"``).
    checksum : bool
        If True, compute and store a SHA-256 checksum.

    Raises
    ------
    MEDH5ValidationError
        If no DICOM files are found, or in-plane shapes disagree.
    """
    _require_pydicom()

    src = Path(dicom_dir)
    if not src.is_dir():
        raise MEDH5ValidationError(f"DICOM directory not found: '{src}'")

    volume, spacing, origin, direction, first_ds = _read_series(src)

    tag_names = list(extra_tags) if extra_tags is not None else list(_DEFAULT_TAGS)
    dicom_tags: dict[str, Any] = {}
    for tag in tag_names:
        value = getattr(first_ds, tag, None)
        if value is None:
            continue
        try:
            # JSON-friendly conversion (UIDs, MultiValue, ints, floats, strings)
            dicom_tags[tag] = (
                str(value) if not isinstance(value, (int, float, str, bool)) else value
            )
        except Exception:  # pragma: no cover
            dicom_tags[tag] = repr(value)

    merged_extra: dict[str, Any] = dict(extra) if extra else {}
    if dicom_tags:
        merged_extra["dicom"] = dicom_tags

    MEDH5File.write(
        out_path,
        images={modality_name: volume},
        label=label,
        label_name=label_name,
        spacing=spacing,
        origin=origin,
        direction=direction,
        coord_system="LPS",  # DICOM patient coordinates
        extra=merged_extra or None,
        compression=compression,
        checksum=checksum,
    )
