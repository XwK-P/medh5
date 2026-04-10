"""NIfTI ⇄ medh5 converters.

Requires ``nibabel`` (install with ``pip install medh5[nifti]``).

The converters preserve the spatial geometry recorded in the NIfTI affine:
``spacing`` is taken from the column norms, ``origin`` from the affine
translation, and ``direction`` from the unit-length column vectors. The
NIfTI canonical coordinate system (RAS+) is recorded as ``coord_system``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5ValidationError

try:
    import nibabel as nib

    _NIBABEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NIBABEL_AVAILABLE = False


def _require_nibabel() -> None:
    if not _NIBABEL_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "nibabel is required for NIfTI I/O. "
            "Install it with: pip install medh5[nifti]"
        )


def _decompose_affine(
    affine: np.ndarray,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Split a 4x4 NIfTI affine into spacing / origin / direction."""
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise MEDH5ValidationError(
            f"Expected a 4x4 NIfTI affine, got shape {affine.shape}"
        )
    rotation = affine[:3, :3]
    spacing = np.linalg.norm(rotation, axis=0)
    # Avoid division by zero on degenerate axes; treat them as unit-length.
    safe_spacing = np.where(spacing > 0, spacing, 1.0)
    direction = rotation / safe_spacing
    origin = affine[:3, 3]
    return (
        spacing.astype(float).tolist(),
        origin.astype(float).tolist(),
        direction.astype(float).tolist(),
    )


def _compose_affine(
    spacing: list[float] | None,
    origin: list[float] | None,
    direction: list[list[float]] | None,
    ndim: int,
) -> np.ndarray:
    """Build a 4x4 NIfTI affine from spacing / origin / direction.

    Falls back to an identity affine when components are missing. Only
    3D volumes get a meaningful affine; lower-dimensional inputs are
    embedded into the upper-left block.
    """
    affine = np.eye(4, dtype=np.float64)
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        if d.shape == (ndim, ndim):
            affine[:ndim, :ndim] = d
    if spacing is not None:
        s = np.asarray(spacing, dtype=np.float64)
        if s.shape == (ndim,):
            affine[:ndim, :ndim] = affine[:ndim, :ndim] * s
    if origin is not None:
        o = np.asarray(origin, dtype=np.float64)
        if o.shape == (ndim,):
            affine[:ndim, 3] = o
    return affine


def _load_nifti(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return ``(data, affine)``."""
    img = nib.load(str(path))
    data: np.ndarray = np.asarray(img.dataobj)
    affine: np.ndarray = np.asarray(img.affine, dtype=np.float64)
    return data, affine


def from_nifti(
    images: dict[str, str | Path],
    out_path: str | Path,
    *,
    seg: dict[str, str | Path] | None = None,
    label: int | str | None = None,
    label_name: str | None = None,
    extra: dict[str, Any] | None = None,
    compression: str | None = "balanced",
    checksum: bool = False,
    require_same_grid: bool = True,
) -> None:
    """Convert one or more NIfTI volumes into a single ``.medh5`` file.

    Parameters
    ----------
    images : dict[str, path]
        Modality name → NIfTI file path. All volumes must share the same
        spatial grid (shape + affine) when ``require_same_grid=True``.
    out_path : str or Path
        Destination ``.medh5`` file.
    seg : dict[str, path], optional
        Mask name → NIfTI file path. Each mask is converted to ``bool``;
        any non-zero voxel counts as foreground.
    label, label_name, extra :
        Forwarded to :meth:`MEDH5File.write`.
    compression : str
        Compression preset (default ``"balanced"``).
    checksum : bool
        If True, write a SHA-256 checksum.
    require_same_grid : bool
        If True (default), raise :class:`MEDH5ValidationError` when image
        shapes or affines disagree across modalities/masks.

    Raises
    ------
    MEDH5ValidationError
        On grid mismatch (when ``require_same_grid=True``) or empty input.
    """
    _require_nibabel()

    if not images:
        raise MEDH5ValidationError("images dict must contain at least one entry")

    image_arrays: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}
    ref_shape: tuple[int, ...] | None = None
    ref_affine: np.ndarray | None = None
    ref_name: str | None = None

    for name in sorted(images.keys()):
        data, aff = _load_nifti(images[name])
        if ref_shape is None:
            ref_shape = data.shape
            ref_affine = aff
            ref_name = name
        elif require_same_grid:
            if data.shape != ref_shape:
                raise MEDH5ValidationError(
                    f"NIfTI shape mismatch: '{ref_name}' has shape {ref_shape} "
                    f"but '{name}' has shape {data.shape}"
                )
            if ref_affine is not None and not np.allclose(aff, ref_affine, atol=1e-5):
                raise MEDH5ValidationError(
                    f"NIfTI affine mismatch between '{ref_name}' and '{name}'. "
                    "Pass require_same_grid=False to skip this check."
                )
        image_arrays[name] = data
        affines[name] = aff

    assert ref_affine is not None
    assert ref_shape is not None

    seg_arrays: dict[str, np.ndarray] | None = None
    if seg:
        seg_arrays = {}
        for name in sorted(seg.keys()):
            data, aff = _load_nifti(seg[name])
            if require_same_grid:
                if data.shape != ref_shape:
                    raise MEDH5ValidationError(
                        f"Segmentation '{name}' shape {data.shape} does not "
                        f"match image shape {ref_shape}"
                    )
                if not np.allclose(aff, ref_affine, atol=1e-5):
                    raise MEDH5ValidationError(
                        f"Segmentation '{name}' affine does not match images. "
                        "Pass require_same_grid=False to skip this check."
                    )
            seg_arrays[name] = data.astype(bool)

    spacing, origin, direction = _decompose_affine(ref_affine)
    ndim = len(ref_shape)
    # Trim spatial metadata if the volume is < 3D (NIfTI affines are always 4x4).
    spacing = spacing[:ndim]
    origin = origin[:ndim]
    direction = [row[:ndim] for row in direction[:ndim]]

    MEDH5File.write(
        out_path,
        images=image_arrays,
        seg=seg_arrays,
        label=label,
        label_name=label_name,
        spacing=spacing,
        origin=origin,
        direction=direction,
        coord_system="RAS",
        extra=extra,
        compression=compression,
        checksum=checksum,
    )


def to_nifti(
    path: str | Path,
    out_dir: str | Path,
    *,
    modalities: list[str] | None = None,
    seg: list[str] | None = None,
    suffix: str = ".nii.gz",
) -> dict[str, Path]:
    """Export images and masks from a ``.medh5`` file as NIfTI files.

    Useful for round-tripping into 3D Slicer or ITK-SNAP for human edits;
    edited masks can be re-imported with :func:`medh5.MEDH5File.add_seg`
    or via the ``medh5 review import-seg`` CLI.

    Parameters
    ----------
    path : str or Path
        Source ``.medh5`` file.
    out_dir : str or Path
        Destination directory (created if missing).
    modalities : list[str], optional
        Subset of modality names to export. Defaults to all.
    seg : list[str], optional
        Subset of mask names to export. Defaults to all.
    suffix : str
        File suffix for NIfTI outputs (``.nii`` or ``.nii.gz``).

    Returns
    -------
    dict[str, Path]
        Map from output key (``"image:CT"``, ``"seg:tumor"``) to written
        file path.
    """
    _require_nibabel()

    src = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample = MEDH5File.read(src)
    s = sample.meta.spatial
    ndim = next(iter(sample.images.values())).ndim
    affine = _compose_affine(s.spacing, s.origin, s.direction, ndim)

    written: dict[str, Path] = {}

    image_keys = modalities if modalities is not None else sorted(sample.images.keys())
    for name in image_keys:
        if name not in sample.images:
            raise MEDH5ValidationError(f"Modality '{name}' not found in {src}")
        out_path = out / f"image_{name}{suffix}"
        nib.save(nib.Nifti1Image(np.asarray(sample.images[name]), affine), out_path)
        written[f"image:{name}"] = out_path

    if sample.seg is not None:
        seg_keys = seg if seg is not None else sorted(sample.seg.keys())
        for name in seg_keys:
            if name not in sample.seg:
                raise MEDH5ValidationError(f"Segmentation '{name}' not found in {src}")
            mask_u8 = np.asarray(sample.seg[name], dtype=np.uint8)
            out_path = out / f"seg_{name}{suffix}"
            nib.save(nib.Nifti1Image(mask_u8, affine), out_path)
            written[f"seg:{name}"] = out_path

    return written
