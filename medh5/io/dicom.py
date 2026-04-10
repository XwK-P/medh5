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

from collections.abc import Sequence
from dataclasses import dataclass
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
    "SeriesDescription",
    "Modality",
    "StudyDate",
    "Manufacturer",
    "SOPClassUID",
    "RescaleSlope",
    "RescaleIntercept",
)
_GEOMETRY_TOL = 1e-5


@dataclass
class _DicomInstance:
    path: Path
    dataset: Any


def _require_pydicom() -> None:
    if not _PYDICOM_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "pydicom is required for DICOM ingestion. "
            "Install it with: pip install medh5[dicom]"
        )


def _json_friendly(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_friendly(v) for v in value]
    return str(value)


def _read_pixel_array(ds: Any, *, apply_modality_lut: bool) -> np.ndarray:
    arr = np.asarray(ds.pixel_array)
    if apply_modality_lut:
        arr = np.asarray(pydicom.pixels.apply_modality_lut(arr, ds))
    return arr


def _read_candidates(dicom_dir: Path) -> list[_DicomInstance]:
    paths = sorted(p for p in dicom_dir.rglob("*") if p.is_file())
    if not paths:
        raise MEDH5ValidationError(f"No DICOM files found under '{dicom_dir}'")

    instances: list[_DicomInstance] = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False)
        except Exception:  # pragma: no cover - pydicom raises many things
            continue
        if not hasattr(ds, "PixelData"):
            continue
        instances.append(_DicomInstance(path=p, dataset=ds))

    if not instances:
        raise MEDH5ValidationError(
            f"No DICOM files with PixelData found under '{dicom_dir}'"
        )
    return instances


def _group_series(
    instances: list[_DicomInstance],
) -> dict[str, list[_DicomInstance]]:
    groups: dict[str, list[_DicomInstance]] = {}
    for inst in instances:
        uid = str(getattr(inst.dataset, "SeriesInstanceUID", "<missing-series-uid>"))
        groups.setdefault(uid, []).append(inst)
    return groups


def _select_series(
    groups: dict[str, list[_DicomInstance]],
    *,
    series_uid: str | None,
) -> tuple[str, list[_DicomInstance], list[str]]:
    available = sorted(groups)
    if series_uid is not None:
        if series_uid not in groups:
            raise MEDH5ValidationError(
                f"SeriesInstanceUID '{series_uid}' not found. Available: {available}"
            )
        return series_uid, groups[series_uid], available
    selected_uid = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[0][
        0
    ]
    return selected_uid, groups[selected_uid], available


def _orientation(ds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = getattr(ds, "ImageOrientationPatient", None)
    if raw is None or len(raw) != 6:
        raise MEDH5ValidationError(
            "DICOM series is missing ImageOrientationPatient; cannot infer geometry"
        )
    row = np.asarray(raw[:3], dtype=np.float64)
    col = np.asarray(raw[3:], dtype=np.float64)
    row_norm = np.linalg.norm(row)
    col_norm = np.linalg.norm(col)
    if row_norm <= 0 or col_norm <= 0:
        raise MEDH5ValidationError("Invalid ImageOrientationPatient vectors")
    row = row / row_norm
    col = col / col_norm
    normal = np.cross(row, col)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 0:
        raise MEDH5ValidationError("Degenerate DICOM orientation vectors")
    return row, col, normal / normal_norm


def _slice_position(ds: Any, slice_normal: np.ndarray) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) != 3:
        raise MEDH5ValidationError(
            "DICOM series is missing ImagePositionPatient; cannot infer slice order"
        )
    return float(np.dot(np.asarray(ipp, dtype=np.float64), slice_normal))


def _read_series(
    dicom_dir: Path,
    *,
    series_uid: str | None,
    apply_modality_lut: bool,
) -> tuple[
    np.ndarray,
    list[float],
    list[float],
    list[list[float]],
    Any,
    dict[str, Any],
]:
    """Load one validated DICOM series and return volume, geometry, and provenance."""
    instances = _read_candidates(dicom_dir)
    groups = _group_series(instances)
    selected_uid, selected, available = _select_series(groups, series_uid=series_uid)

    first = selected[0].dataset
    row_cosine, col_cosine, slice_normal = _orientation(first)

    pixel_spacing = getattr(first, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) != 2:
        raise MEDH5ValidationError(
            "DICOM series is missing PixelSpacing; cannot infer in-plane spacing"
        )
    ref_pixel_spacing = np.asarray(pixel_spacing, dtype=np.float64)
    if np.any(ref_pixel_spacing <= 0):
        raise MEDH5ValidationError(f"Invalid PixelSpacing values: {pixel_spacing}")

    ordered: list[tuple[float, np.ndarray, _DicomInstance]] = []
    ref_shape: tuple[int, ...] | None = None
    for inst in selected:
        ds = inst.dataset
        if int(getattr(ds, "NumberOfFrames", 1) or 1) != 1:
            raise MEDH5ValidationError(
                "Multi-frame DICOM is not supported; provide a single-frame series"
            )
        if int(getattr(ds, "SamplesPerPixel", 1) or 1) != 1:
            raise MEDH5ValidationError(
                "Only grayscale single-sample DICOM images are supported"
            )
        interpretation = str(getattr(ds, "PhotometricInterpretation", ""))
        if interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
            raise MEDH5ValidationError(
                f"Unsupported PhotometricInterpretation '{interpretation}'"
            )

        row_i, col_i, _ = _orientation(ds)
        if not np.allclose(row_i, row_cosine, atol=_GEOMETRY_TOL) or not np.allclose(
            col_i, col_cosine, atol=_GEOMETRY_TOL
        ):
            raise MEDH5ValidationError(
                "Inconsistent ImageOrientationPatient across the selected series"
            )

        spacing_i = getattr(ds, "PixelSpacing", None)
        if spacing_i is None or len(spacing_i) != 2:
            raise MEDH5ValidationError("Missing PixelSpacing on one or more slices")
        if not np.allclose(
            np.asarray(spacing_i, dtype=np.float64),
            ref_pixel_spacing,
            atol=_GEOMETRY_TOL,
        ):
            raise MEDH5ValidationError(
                "Inconsistent PixelSpacing across the selected series"
            )

        arr = _read_pixel_array(ds, apply_modality_lut=apply_modality_lut)
        if arr.ndim != 2:
            raise MEDH5ValidationError(
                f"Only single-frame 2D slices are supported, got shape {arr.shape}"
            )
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise MEDH5ValidationError(
                "In-plane shape mismatch in DICOM series: "
                f"{arr.shape} vs reference {ref_shape}"
            )
        ordered.append((_slice_position(ds, slice_normal), arr, inst))

    ordered.sort(key=lambda item: item[0])
    positions = np.asarray([item[0] for item in ordered], dtype=np.float64)
    if positions.size >= 2:
        diffs = np.diff(positions)
        if np.any(np.abs(diffs) <= _GEOMETRY_TOL):
            raise MEDH5ValidationError(
                "Duplicate or overlapping slice positions detected in the series"
            )
        spacing_abs = np.abs(diffs)
        if not np.allclose(spacing_abs, spacing_abs[0], atol=1e-4, rtol=1e-4):
            raise MEDH5ValidationError(
                "Inconsistent slice spacing detected in the selected series"
            )
        slice_spacing = float(spacing_abs[0])
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)
        if slice_spacing <= 0:
            raise MEDH5ValidationError(
                "SliceThickness must be positive for single-slice import"
            )

    volume = np.stack([arr for _, arr, _ in ordered], axis=0)

    first_ds = ordered[0][2].dataset
    origin_raw = getattr(first_ds, "ImagePositionPatient", None)
    if origin_raw is None or len(origin_raw) != 3:
        raise MEDH5ValidationError("Missing ImagePositionPatient on the first slice")
    spacing = [slice_spacing, float(ref_pixel_spacing[0]), float(ref_pixel_spacing[1])]
    origin = [float(v) for v in origin_raw]
    direction = [slice_normal.tolist(), row_cosine.tolist(), col_cosine.tolist()]
    provenance = {
        "selected_series_uid": selected_uid,
        "available_series_uids": available,
        "n_instances": len(ordered),
        "applied_modality_lut": apply_modality_lut,
    }
    return volume, spacing, origin, direction, first_ds, provenance


def from_dicom(
    dicom_dir: str | Path,
    out_path: str | Path,
    *,
    modality_name: str = "CT",
    series_uid: str | None = None,
    apply_modality_lut: bool = True,
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
    series_uid : str, optional
        Explicit ``SeriesInstanceUID`` to import when multiple series exist.
    apply_modality_lut : bool
        Apply DICOM modality LUT / rescale slope-intercept before writing.
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

    volume, spacing, origin, direction, first_ds, provenance = _read_series(
        src, series_uid=series_uid, apply_modality_lut=apply_modality_lut
    )

    tag_names = list(extra_tags) if extra_tags is not None else list(_DEFAULT_TAGS)
    dicom_tags: dict[str, Any] = {}
    for tag in tag_names:
        value = getattr(first_ds, tag, None)
        if value is None:
            continue
        try:
            dicom_tags[tag] = _json_friendly(value)
        except Exception:  # pragma: no cover
            dicom_tags[tag] = repr(value)

    merged_extra: dict[str, Any] = dict(extra) if extra else {}
    merged_extra["dicom"] = {**dicom_tags, **provenance}

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
