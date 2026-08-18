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
from medh5.geometry.affine import ORTHONORMAL_TOL, build_affine, decompose_affine
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
    fourth_axis: str = "auto",
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One NIfTI file as ``(array, geometry)`` in MEDH5 conventions.

    *fourth_axis* decides what a 4-D series' extra axis is --- ``"time"`` for
    cine, DCE and 4-D CT, ``"channel"`` for multi-b-value DWI and multi-echo
    (§3.6).  ``"auto"`` reads the answer out of the file where it can and
    reports a guess where it cannot.
    """
    nib = require_nibabel()
    image = nib.load(os.fspath(path))
    data = np.asanyarray(image.dataobj)
    # A trailing axis of extent 1 beyond the spatial block carries nothing:
    # writers emit `dim[4] = 1` routinely, and keeping it turns a plain volume
    # into a grid with a degenerate one-frame time axis --- and a 5-D
    # `(x, y, z, 1, n)` multi-echo into something no grid can describe.
    squeezed = [
        axis
        for axis in range(data.ndim - 1, 2, -1)  # noqa: PLR2004 - past x, y, z
        if data.shape[axis] == 1
    ]
    for axis in squeezed:
        data = np.squeeze(data, axis=axis)
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
    if data.ndim == 2:  # noqa: PLR2004 - a radiograph (§3.6)
        spacing, origin, direction = _reduce_plane(spacing, origin, direction)
    times: tuple[float, ...] | None = None
    time_units: str | None = None
    measured = False
    kind = "spatial"
    b_values: tuple[float, ...] | None = None
    if data.ndim == 4:  # noqa: PLR2004 - three spatial axes and one more
        kind, stated = _fourth_axis(image, path, fourth_axis)
        if kind == "time":
            times, time_units, measured = _time_axis(image, int(data.shape[0]))
        else:
            sidecar = _bval_sidecar(path)
            b_values = None if sidecar is None else _read_bvals(sidecar)
        measured = measured if kind == "time" else stated
    return np.ascontiguousarray(data), {
        "spacing": [float(v) for v in spacing],
        "origin": [float(v) for v in origin],
        "direction": [[float(v) for v in row] for row in direction],
        "coord_system": coord_system,
        "shape": tuple(int(v) for v in data.shape),
        "axis_order": order,
        "time_values": None if times is None else list(times),
        "time_units": time_units,
        "time_measured": measured,
        "leading_kind": kind,
        "leading_stated": measured,
        "b_values": None if b_values is None else list(b_values),
        "squeezed": squeezed,
        "units": _units(image),
        "dtype": str(data.dtype),
        "header": {
            "descrip": _text(image.header.get("descrip")),
            "scl_slope": _number(image.header.get("scl_slope")),
            "scl_inter": _number(image.header.get("scl_inter")),
        },
    }


# NIfTI intent codes whose extra dimension holds *components* rather than
# frames --- §3.6's "channel" kind rather than its "time" kind.  The RGB and
# RGBA vectors (2003, 2004) belong here as much as the numeric ones: they are
# the "RGB / multi-echo" row, and without them that row fell through to the
# time guess despite the header stating the answer.
CHANNEL_INTENTS = frozenset({1001, 1004, 1005, 1006, 1007, 2003, 2004})
TIME_SERIES_INTENT = 2001

FOURTH_AXES = ("auto", "time", "channel")


def _bval_sidecar(path: str | os.PathLike[str]) -> Path | None:
    """The ``.bval`` file beside a NIfTI, which is what makes a series DWI.

    NIfTI-1 says ``dim[4]`` is time, but every diffusion pipeline in practice
    puts the gradient index there and writes the b-values alongside.  The
    sidecar is the only reliable signal, because such files carry no intent
    code and often a meaningless ``pixdim[4]``.
    """
    name = Path(os.fspath(path))
    stem = name.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            candidate = name.with_name(stem[: -len(suffix)] + ".bval")
            return candidate if candidate.exists() else None
    return None


def _read_bvals(path: Path) -> tuple[float, ...]:
    text = path.read_text().replace("\n", " ")
    return tuple(float(v) for v in text.split())


def _fourth_axis(
    image: Any, path: str | os.PathLike[str], requested: str
) -> tuple[str, bool]:
    """Whether a 4-D NIfTI's leading axis is ``time`` or ``channel`` (§3.6).

    §3.6 gives both a row --- cine/DCE/4-D CT under ``time``, multi-b-value DWI
    and multi-echo under ``channel`` --- and NIfTI does not distinguish them in
    ``dim[4]``.  Reading every 4-D series as time labelled every DWI gradient
    axis a time axis and handed it invented per-frame timings.

    Returns the kind and whether the source stated it.  A guess is reported as
    one, and ``fourth_axis=`` overrides it outright.
    """
    if requested not in FOURTH_AXES:
        raise MEDH5ValidationError(
            f"unknown fourth_axis {requested!r}; expected one of {list(FOURTH_AXES)}"
        )
    if requested != "auto":
        return requested, True
    if _bval_sidecar(path) is not None:
        return "channel", True
    intent = int(image.header["intent_code"])
    if intent in CHANNEL_INTENTS:
        return "channel", True
    if intent == TIME_SERIES_INTENT:
        return "time", True
    if _temporal(image)[3]:
        return "time", True
    return "time", False


def _reduce_plane(
    spacing: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A 2-D radiograph's geometry reduced to two dimensions, or a refusal.

    §3.6 gives a 2-D grid ``S = 2``: two spacings, a 2-D origin and a 2x2
    ``direction``.  A NIfTI plane carries a 3-D affine regardless, so the
    converter has to reduce it --- and can only do so where the two in-plane
    axes have no component along the third world axis.  A plane tilted in 3-D
    has no 2-D grid that describes it, and flattening it anyway would move
    every pixel to somewhere it is not.
    """
    if float(np.max(np.abs(direction[2, :2]))) > ORTHONORMAL_TOL:
        raise MEDH5ValidationError(
            "this 2-D NIfTI is a plane tilted in 3-D, and §3.6 gives a 2-D grid "
            "a 2x2 direction --- there is no 2-D grid that describes it; import "
            "it as a single-slice 3-D volume instead",
            code="E102",
        )
    return spacing[:2], origin[:2], direction[:2, :2]


def grid_axes(
    shape: Sequence[int], *, transposed: bool = True, leading_kind: str = "time"
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
        name = "t" if leading_kind == "time" else "c"
        return (name, "z", "y", "x"), (leading_kind, "spatial", "spatial", "spatial")
    raise MEDH5ValidationError(
        f"a {len(shape)}-D NIfTI has {extra} axes beyond (x, y, z); only a "
        "single trailing time axis converts without a decision about what the "
        "others mean --- convert the volumes separately, or declare the axes"
    )


def _temporal(image: Any) -> tuple[float, float, str, bool]:
    """The temporal zoom, offset and unit, and whether the file states them.

    ``pixdim[4]`` is 1.0 in a freshly built header, so a positive zoom on its
    own is not a statement about timing --- it is the default.  The unit has to
    be set as well before the number is the scanner's and not nibabel's, which
    is what separates a frame time this converter read from one it assumed.
    """
    zooms = image.header.get_zooms()
    step = float(zooms[3]) if len(zooms) > 3 else 0.0  # noqa: PLR2004 - dim[4]
    try:
        _, temporal = image.header.get_xyzt_units()
    except Exception:  # noqa: BLE001 - a malformed header is not fatal
        temporal = "unknown"
    known = {"sec": ("s", 1.0), "msec": ("ms", 1.0), "usec": ("ms", 1e-3)}
    if step <= 0 or str(temporal) not in known:
        return 0.0, 0.0, "s", False
    units, scale = known[str(temporal)]
    try:
        offset = float(image.header["toffset"])
    except (KeyError, ValueError, TypeError):  # pragma: no cover - header variants
        offset = 0.0
    # Both scaled here, by the same factor.  `toffset` is in the header's own
    # temporal unit, so converting the zoom and not the offset put a microsecond
    # series a thousand frames from where it starts.
    return step * scale, offset * scale, units, True


def _time_axis(image: Any, frames: int) -> tuple[tuple[float, ...], str, bool]:
    """Per-frame acquisition times, and whether they were read or assumed.

    A grid carrying a time axis **MUST** carry ``time_values`` (§3.2), and §3.6
    puts them "per frame" for exactly the cine, DCE and 4-D CT series this
    path converts, so leaving a 4-D import with no frame timing at all was not
    an option --- the source states a temporal zoom and the converter was
    throwing it away.
    """
    step, offset, units, stated = _temporal(image)
    if not stated:
        # Nothing to read.  Frame indices are all that is left, and they are an
        # assumption about timing rather than a measurement of it.
        return tuple(float(k) for k in range(frames)), "s", False
    return tuple(offset + k * step for k in range(frames)), units, True


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


def _same_axis_kind(per_image: Mapping[str, Mapping[str, Any]]) -> None:
    """Every volume on one grid must agree what its non-spatial axis is.

    They share a single grid, and the grid states ``axis_kinds`` once.  A DWI
    and a cine series of the same shape would otherwise both be written under
    whichever kind the first file happened to have, so one of them ends up
    declared as something it is not.
    """
    kinds = {
        name: geo.get("leading_kind", "spatial") for name, geo in per_image.items()
    }
    distinct = sorted(set(kinds.values()))
    if len(distinct) > 1:
        listed = ", ".join(f"{n} is {k}" for n, k in sorted(kinds.items()))
        raise MEDH5ValidationError(
            f"these volumes share a grid but disagree about their non-spatial "
            f"axis ({listed}); one grid states one set of `axis_kinds`, so "
            "convert them separately or pass `fourth_axis=` to settle it",
            code="E110",
        )


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
    fourth_axis: str = "auto",
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
    # Kept per image.  `_same_grid` compares the *grid*, which the volumes share
    # by definition here; b-values and the leading axis kind belong to the file
    # they came from, and reading them off the first geometry handed every DWI
    # in the set the first one's gradients without noticing.
    per_image: dict[str, dict[str, Any]] = {}
    geometry: dict[str, Any] | None = None
    for name, path in images.items():
        data, geo = read_nifti(
            path,
            coord_system=coord_system,
            transpose=transpose,
            fourth_axis=fourth_axis,
        )
        geometry = _same_grid(geometry, geo, name, log) if geometry else geo
        per_image[name] = geo
        arrays[name] = data
    assert geometry is not None
    _same_axis_kind(per_image)

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
    if geometry.get("squeezed"):
        log.decision(
            "axis_order",
            "trailing axes of extent 1 beyond (x, y, z) carry nothing and were "
            "dropped, so the grid describes the data rather than the file",
            {"axes": list(geometry["squeezed"])},
        )
    if geometry.get("leading_kind") == "channel":
        detail = {
            "channels": geometry["shape"][0],
            "b_values": geometry.get("b_values"),
        }
        log.decision(
            "axis_kinds",
            "the fourth axis holds components rather than frames, so it is a "
            "`channel` axis (§3.6) --- b-values or an intent code said so",
            detail,
        )
    elif geometry.get("leading_kind") == "time" and not geometry.get("leading_stated"):
        log.guess(
            "axis_kinds",
            "the source does not say whether the fourth axis is time or channel; "
            "it was read as time (§3.6) --- pass `fourth_axis=` to say otherwise",
            {"frames": geometry["shape"][0]},
        )
    if geometry.get("time_values") is not None:
        detail = {
            "time_units": geometry["time_units"],
            "frames": len(geometry["time_values"]),
        }
        if geometry.get("time_measured"):
            log.decision(
                "time_values",
                "frame times were read from the NIfTI temporal zoom and toffset",
                detail,
            )
        else:
            log.guess(
                "time_values",
                "the NIfTI declares no temporal zoom, so frame indices were used "
                "as times; §3.2 requires `time_values` where there is a time axis",
                detail,
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
            geometry["shape"],
            transposed=order != tuple(range(len(order))),
            leading_kind=geometry.get("leading_kind", "time"),
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
            time_values=geometry.get("time_values"),
            time_units=geometry.get("time_units"),
        )
        for name, array in arrays.items():
            b_values = per_image[name].get("b_values")
            channel_names = (
                tuple(f"b={v:g}" for v in b_values)
                if b_values and len(b_values) == geometry["shape"][0]
                else None
            )
            if b_values:
                # §3.6 puts the b-values in `acquisition` (§4.5); they are what
                # the channel axis *means*, and dropping them leaves a stack of
                # unlabelled volumes.
                writer.acquisition(name, b_values=list(b_values))
            writer.add_image(
                name,
                array,
                grid="ref",
                modality=(modalities or {}).get(name, "OT"),
                channel_names=channel_names,
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
