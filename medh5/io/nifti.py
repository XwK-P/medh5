"""NIfTI import and export (spec §3, §4, §7).

The one thing this converter must not get wrong is the **handedness**.  NIfTI's
affine is RAS+; MEDH5 grids declare their own ``coord_system`` and default to
LPS, the DICOM convention.  Converting between them is a sign flip on the first
two world axes --- applied to the affine, never to the voxels.

So the choice is made explicitly and recorded: ``coord_system="LPS"`` (the
default) converts and says so in the report; ``coord_system="RAS"`` keeps
NIfTI's own frame and stores that on the grid.  Either is correct; silently
picking one and not writing it down is how a segmentation ends up mirrored
against the image it was drawn on, which no unit test catches because both
volumes flip together.

Axis order is likewise stated rather than assumed.  NIfTI arrays are
``(x, y, z)`` fastest-first; MEDH5 spatial axes are trailing and, for a 3-D
volume written by a medical tool, conventionally ``(z, y, x)``.  ``transpose``
controls it, defaults to reordering, and the resulting axis names are written
onto the grid so a reader never has to infer them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import build_affine, decompose_affine
from medh5.io.report import ConversionReport

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""World-axis sign flip. ``LPS = RAS_TO_LPS @ RAS``, and it is its own inverse."""

COORD_SYSTEMS = ("LPS", "RAS")


def require_nibabel() -> Any:
    try:
        import nibabel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "nibabel is required for NIfTI conversion. Install it with: "
            "pip install 'medh5[nifti]'"
        ) from exc
    return nibabel


def convert_world(
    affine: npt.ArrayLike, *, source: str, target: str
) -> npt.NDArray[np.float64]:
    """Re-express a 3-D index→world affine between RAS and LPS."""
    for name in (source, target):
        if name not in COORD_SYSTEMS:
            raise MEDH5ValidationError(
                f"unknown coordinate system {name!r}; expected one of "
                f"{list(COORD_SYSTEMS)}"
            )
    matrix = np.asarray(affine, dtype=np.float64)
    if source == target:
        return matrix
    if matrix.shape != (4, 4):
        raise MEDH5ValidationError(
            f"RAS↔LPS conversion is defined for 3-D affines; got {matrix.shape}"
        )
    return np.asarray(RAS_TO_LPS @ matrix, dtype=np.float64)


def read_nifti(
    path: str | os.PathLike[str],
    *,
    coord_system: str = "LPS",
    transpose: bool = True,
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One NIfTI file as ``(array, geometry)`` in MEDH5 conventions."""
    nib = require_nibabel()
    image = nib.load(os.fspath(path))
    data = np.asanyarray(image.dataobj)
    affine = convert_world(image.affine, source="RAS", target=coord_system)
    spacing, origin, direction = decompose_affine(affine)
    order = tuple(range(data.ndim))
    if transpose and data.ndim >= 3:  # noqa: PLR2004 - 3-D and up reorder
        # NIfTI puts i, j, k *first* and time (dim[4]) after them, so the
        # spatial block is the leading three axes, not the trailing three.
        # Reversing the trailing three would move t into a spatial slot and
        # hand the grid a spacing that belongs to a different axis --- silently,
        # for every cine, DCE and 4-D CT series.
        spatial = (2, 1, 0)
        order = tuple(range(3, data.ndim)) + spatial
        data = np.transpose(data, order)
        spacing = spacing[list(spatial)]
        direction = direction[:, list(spatial)]
    return np.ascontiguousarray(data), {
        "spacing": [float(v) for v in spacing],
        "origin": [float(v) for v in origin],
        "direction": [[float(v) for v in row] for row in direction],
        "coord_system": coord_system,
        "shape": tuple(int(v) for v in data.shape),
        "axis_order": order,
        "units": _units(image),
        "dtype": str(data.dtype),
        "header": {
            "descrip": _text(image.header.get("descrip")),
            "scl_slope": _number(image.header.get("scl_slope")),
            "scl_inter": _number(image.header.get("scl_inter")),
        },
    }


def grid_axes(
    shape: Sequence[int], *, transposed: bool = True
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(axis_names, axis_kinds)`` for a grid built from a converted NIfTI array.

    A grid of more than three axes has no unambiguous default (§3.1), so the
    converter has to name the axes rather than let ``add_grid`` guess --- and
    what it should name them depends on whether ``read_nifti`` reordered the
    array, which is why *transposed* is not optional information.

    Reordered, the spatial block trails and any NIfTI axis beyond the third
    leads, so a 4-D series is ``(t, z, y, x)``.  Left alone, the array is still
    in NIfTI order and the axes are ``(x, y, z)`` --- declaring the reordered
    names over it labelled the x axis ``time`` and gave the grid a spacing
    belonging to a different axis on every one.

    Anything past a single time axis is refused rather than guessed at:
    NIfTI's dim[5] carries vector or tensor components whose MEDH5 kind depends
    on what the producer meant by them.
    """
    extra = len(shape) - 3
    if not transposed:
        if extra > 0:
            raise MEDH5ValidationError(
                f"a {len(shape)}-D NIfTI keeps its time axis last, and §3.1 "
                "requires the spatial axes to be trailing and contiguous, so "
                "this array cannot be declared as a grid without reordering it "
                "--- drop `transpose=False` for 4-D input"
            )
        return ("x", "y", "z")[: len(shape)], ("spatial",) * len(shape)
    if extra <= 0:
        return ("z", "y", "x")[-len(shape) :], ("spatial",) * len(shape)
    if extra == 1:
        return ("t", "z", "y", "x"), ("time", "spatial", "spatial", "spatial")
    raise MEDH5ValidationError(
        f"a {len(shape)}-D NIfTI has {extra} axes beyond (x, y, z); only a "
        "single trailing time axis converts without a decision about what the "
        "others mean --- convert the volumes separately, or declare the axes"
    )


def _units(image: Any) -> str:
    """The NIfTI spatial unit, mapped onto the §3.5 vocabulary."""
    try:
        spatial, _ = image.header.get_xyzt_units()
    except Exception:  # noqa: BLE001 - a malformed header is not fatal
        return "mm"
    return {"meter": "m", "mm": "mm", "micron": "um", "unknown": "mm"}.get(
        str(spatial), "mm"
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    raw = (
        bytes(value) if isinstance(value, (bytes, np.ndarray)) else str(value).encode()
    )
    return raw.split(b"\x00")[0].decode("utf-8", "replace")


def _number(value: Any) -> float | None:
    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except Exception:  # noqa: BLE001 - absent or malformed
        return None
    return None if not np.isfinite(out) else out


def from_nifti(
    images: Mapping[str, str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    masks: Mapping[str, str | os.PathLike[str]] | None = None,
    sample_id: str | None = None,
    subject_id: str | None = None,
    modalities: Mapping[str, str] | None = None,
    coord_system: str = "LPS",
    transpose: bool = True,
    value_units: Mapping[str, str] | None = None,
    codec: str = "balanced",
    annotated_classes: Sequence[str] | str = "all_given",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Write one sample from a set of co-registered NIfTI volumes.

    Every volume must share one grid.  A disagreement is refused rather than
    resampled: resampling changes the data, and a converter that does it
    silently produces a file whose labels no longer sit on the voxels they were
    drawn on.  Resample first, deliberately, with a tool that records it.
    """
    import medh5

    log = report or ConversionReport(converter="from-nifti")
    log.source = ", ".join(str(p) for p in images.values())
    if not images:
        raise MEDH5ValidationError("from_nifti needs at least one image")

    arrays: dict[str, npt.NDArray[Any]] = {}
    geometry: dict[str, Any] | None = None
    for name, path in images.items():
        data, geo = read_nifti(path, coord_system=coord_system, transpose=transpose)
        geometry = _same_grid(geometry, geo, name, log) if geometry else geo
        arrays[name] = data
    assert geometry is not None

    mask_arrays: dict[str, npt.NDArray[np.bool_]] = {}
    for name, path in (masks or {}).items():
        data, geo = read_nifti(path, coord_system=coord_system, transpose=transpose)
        _same_grid(geometry, geo, name, log)
        mask_arrays[name] = np.asarray(data) != 0

    if coord_system != "RAS":
        log.decision(
            "coord_system",
            f"NIfTI is RAS+; the grid was written in {coord_system} "
            "(sign flip on the first two world axes, applied to the affine only)",
            {"source": "RAS", "target": coord_system},
        )
    if transpose and len(geometry["shape"]) >= 3:  # noqa: PLR2004
        log.decision(
            "axis_order",
            "NIfTI (x, y, z) was reordered to (z, y, x), any trailing time "
            "axis moved in front of them, and the spacing, direction and axis "
            "names follow",
            {"order": list(geometry["axis_order"])},
        )

    target = Path(os.fspath(out))
    stem = sample_id or target.stem
    label_set = None
    if mask_arrays:
        label_set = _mint_label_set(sorted(mask_arrays), log)

    with medh5.create(
        target, sample_id=stem, subject_id=subject_id or stem, codec=codec
    ) as writer:
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import",
            agent=tool,
            tool="medh5 convert from-nifti",
            inputs=[str(p) for p in images.values()],
            params={"coord_system": coord_system, "transpose": bool(transpose)},
        )
        if label_set is not None:
            writer.label_set(label_set)
        # Whether the axes actually moved, read off what `read_nifti` recorded
        # rather than re-derived from `transpose`: the two differ for arrays
        # below 3-D, which are never reordered whatever the caller asked for.
        order = tuple(geometry["axis_order"])
        axis_names, axis_kinds = grid_axes(
            geometry["shape"], transposed=order != tuple(range(len(order)))
        )
        writer.add_grid(
            "ref",
            shape=geometry["shape"],
            spacing=geometry["spacing"],
            origin=geometry["origin"],
            direction=geometry["direction"],
            axis_names=axis_names,
            axis_kinds=axis_kinds,
            coord_system=geometry["coord_system"],
            units=geometry["units"],
        )
        for name, array in arrays.items():
            writer.add_image(
                name,
                array,
                grid="ref",
                modality=(modalities or {}).get(name, "OT"),
                value_units=(value_units or {}).get(name),
                value_type="quantitative"
                if (value_units or {}).get(name)
                else "intensity",
                prov=activity,
            )
        if mask_arrays and label_set is not None:
            resolved = {label_set[key].id: mask for key, mask in mask_arrays.items()}
            kind, stats = writer.add_segmentation(
                "seg",
                grid="ref",
                masks=resolved,
                annotated_classes=annotated_classes,
                prov=activity,
            )
            log.decision(
                "encoding",
                f"{len(resolved)} mask(s) were measured and stored as {kind!r}",
                {
                    "kind": kind,
                    "overlapping": None if stats is None else len(stats.edges),
                },
            )
            log.guess(
                "coverage",
                "annotated_class_ids was set to the masks supplied; NIfTI cannot "
                "say which classes were searched for and not found",
                {"classes": sorted(mask_arrays)},
            )
    log.outputs.append(str(target))
    return log


def _same_grid(
    first: dict[str, Any], other: dict[str, Any], name: str, log: ConversionReport
) -> dict[str, Any]:
    """Refuse volumes that do not share a grid (spec §3.2)."""
    if tuple(first["shape"]) != tuple(other["shape"]):
        raise MEDH5ValidationError(
            f"{name!r} has shape {other['shape']}, but the first volume has "
            f"{first['shape']}; resample before converting",
            code="E202",
        )
    for key in ("spacing", "origin", "direction"):
        if not np.allclose(
            np.asarray(first[key], dtype=np.float64),
            np.asarray(other[key], dtype=np.float64),
            atol=1e-4,
        ):
            raise MEDH5ValidationError(
                f"{name!r} disagrees with the first volume on {key}; resample "
                "before converting rather than letting a converter do it silently",
                code="E101",
            )
    return first


def _mint_label_set(keys: Sequence[str], log: ConversionReport) -> Any:
    """Mask filenames become class keys with minted ids (Appendix B)."""
    from medh5.labels.labelset import LabelClass, LabelSet

    classes = [
        LabelClass(i + 1, key, key.replace("_", " ").title())
        for i, key in enumerate(keys)
    ]
    log.decision(
        "label_set",
        "class ids were minted from the mask names in sorted order; review the "
        "generated label set before applying it cohort-wide",
        {"ids": {c.key: c.id for c in classes}},
    )
    return LabelSet("converted", version="1.0.0", classes=classes)


def to_nifti(
    path: str | os.PathLike[str],
    image_id: str,
    out: str | os.PathLike[str],
    *,
    physical: bool = True,
    annotation: str | None = None,
    class_key: int | str | None = None,
) -> Path:
    """Export one image, or one class of one annotation, as a NIfTI file.

    The affine is converted back to RAS+ and the axes back to ``(x, y, z)``, so
    the file opens in FSL, ITK-SNAP or 3D Slicer at the location it came from.
    """
    import medh5

    nib = require_nibabel()
    target = Path(os.fspath(out))
    with medh5.open(path) as sample:
        image = sample.images[image_id]
        grid = image.grid
        if annotation is not None:
            ann = sample.annotations[annotation]
            data = (
                np.asarray(ann.labelmap(), dtype=np.uint16)
                if class_key is None
                else np.asarray(ann.dense([class_key])[0], dtype=np.uint8)
            )
        else:
            data = np.asarray(image.read(physical=physical))
        affine = convert_world(grid.affine, source=grid.coord_system, target="RAS")
        spacing, origin, direction = decompose_affine(affine)
        if data.ndim >= 3:  # noqa: PLR2004 - undo the (z, y, x) reordering
            # MEDH5 leads with time and trails with (z, y, x); NIfTI is the
            # other way round.  Reversing only the trailing three sent time to
            # a *spatial* slot and left the affine describing (x, y, z), so a
            # 4-D export carried geometry that was wrong on every axis.
            spatial = (data.ndim - 1, data.ndim - 2, data.ndim - 3)
            data = np.transpose(data, spatial + tuple(range(data.ndim - 3)))
            index = [2, 1, 0]
            spacing = spacing[index]
            direction = direction[:, index]
        out_affine = build_affine(spacing, origin, direction)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), out_affine), str(target))
    return target


def import_seg_nifti(
    path: str | os.PathLike[str],
    masks: Mapping[str, str | os.PathLike[str]],
    *,
    ann_id: str = "seg",
    grid: str | None = None,
    coord_system: str | None = None,
    transpose: bool = True,
    annotated_classes: Sequence[str] | str = "all_given",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Add NIfTI masks to an existing sample, checking they sit on its grid."""
    import medh5

    log = report or ConversionReport(converter="import-seg-nifti")
    log.source = ", ".join(str(p) for p in masks.values())
    with medh5.open(path) as sample:
        grid_id = grid or sample.reference_grid.grid_id
        target_grid = sample.grids[grid_id]
        system = coord_system or target_grid.coord_system
        existing = sample.label_set

    arrays: dict[str, npt.NDArray[np.bool_]] = {}
    for name, mask_path in masks.items():
        data, geo = read_nifti(mask_path, coord_system=system, transpose=transpose)
        if tuple(geo["shape"]) != tuple(target_grid.spatial_shape):
            raise MEDH5ValidationError(
                f"mask {name!r} has shape {geo['shape']}, but grid {grid_id!r} is "
                f"{target_grid.spatial_shape}",
                code="E405",
            )
        if not np.allclose(
            np.asarray(geo["spacing"]), np.asarray(target_grid.spacing), atol=1e-4
        ):
            log.warn(
                "geometry",
                f"mask {name!r} declares spacing {geo['spacing']}, the grid says "
                f"{list(target_grid.spacing)}; the voxels were taken as aligned",
                {"mask": name},
            )
        arrays[name] = np.asarray(data) != 0

    label_set = existing
    if label_set is None:
        label_set = _mint_label_set(sorted(arrays), log)
    missing = [k for k in arrays if label_set.get(k) is None]
    if missing:
        raise MEDH5ValidationError(
            f"the sample's label set has no class(es) {missing}; add them or "
            "convert without an existing label set",
            code="E402",
        )
    with medh5.amend(path) as writer:
        if existing is None:
            writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import", agent=tool, tool="medh5 convert import-seg-nifti"
        )
        kind, _ = writer.add_segmentation(
            ann_id,
            grid=grid_id,
            masks={label_set[k].id: v for k, v in arrays.items()},
            annotated_classes=annotated_classes,
            prov=activity,
        )
        log.decision("encoding", f"masks were stored as {kind!r}", {"kind": kind})
    log.outputs.append(str(path))
    return log


__all__ = [
    "COORD_SYSTEMS",
    "RAS_TO_LPS",
    "convert_world",
    "from_nifti",
    "import_seg_nifti",
    "read_nifti",
    "require_nibabel",
    "to_nifti",
]
