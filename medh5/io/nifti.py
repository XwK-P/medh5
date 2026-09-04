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

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._optional import require
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import ORTHONORMAL_TOL, build_affine, decompose_affine
from medh5.io.report import ConversionReport

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""World-axis sign flip. ``LPS = RAS_TO_LPS @ RAS``, and it is its own inverse."""

RAS_TO_LPS_2D = np.diag([-1.0, -1.0, 1.0])
"""The same flip for a 2-D grid (§3.6).

A 2-D grid's two world axes are the first two of the frame it was reduced
from --- :func:`_reduce_plane` keeps a plane only when it has no component
along the third --- and those are exactly the two axes RAS↔LPS flips.
"""

COORD_SYSTEMS = ("LPS", "RAS")


def require_nibabel() -> Any:
    return require("nibabel", extra="nifti", purpose="NIfTI conversion")


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
    if matrix.shape == (3, 3):
        # A 2-D grid, which §3.6 gives an S = 2 affine.  Both importers accept
        # 2-D input and both exporters called this, so refusing here meant no
        # 2-D sample could be written back out at all --- a radiograph could go
        # in and never come out.
        return np.asarray(RAS_TO_LPS_2D @ matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise MEDH5ValidationError(
            f"RAS↔LPS conversion is defined for 2-D and 3-D affines; got {matrix.shape}"
        )
    return np.asarray(RAS_TO_LPS @ matrix, dtype=np.float64)


def embed_plane(affine: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """A 2-D grid's 3x3 affine as the 4x4 a NIfTI file carries.

    The exact inverse of :func:`_reduce_plane`: the third index axis is unit
    and the third world coordinate zero, which is the only embedding that
    round-trips, because a plane keeps no information about the axis it was
    reduced along.  A 2-D array is written as ``(x, y)`` with ``dim[3] = 1``,
    which ``read_nifti`` squeezes back off.
    """
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 3):
        raise MEDH5ValidationError(
            f"embed_plane takes a 2-D grid's 3x3 affine; got {matrix.shape}"
        )
    out = np.eye(4, dtype=np.float64)
    out[:2, :2] = matrix[:2, :2]
    out[:2, 3] = matrix[:2, 2]
    return out


def write_nifti(
    array: npt.NDArray[Any],
    affine: npt.ArrayLike,
    path: str | os.PathLike[str],
    *,
    rescale: tuple[float, float] | None = None,
) -> Path:
    """Write one NIfTI file, recording *rescale* in the header (§4.2).

    Every export goes through here, because the alternative is what shipped:
    ``image.read()`` returns **stored** values and the exporters wrote them
    with no ``scl_slope``/``scl_inter``, so a CT imported from DICOM with
    intercept −1024 left every voxel 1024 HU too high in the exported file and
    nothing in it said so.  NIfTI has the two fields for exactly this, and
    every reader --- nibabel, SimpleITK, and therefore nnU-Net --- applies them
    on load, so writing them makes the stored volume mean what it meant here.
    """
    nib = require_nibabel()
    target = Path(os.fspath(path))
    data = np.ascontiguousarray(array)
    image = nib.Nifti1Image(data, np.asarray(affine, dtype=np.float64))
    image.header.set_data_dtype(data.dtype)
    if rescale is not None:
        image.header.set_slope_inter(float(rescale[0]), float(rescale[1]))
    nib.save(image, str(target))
    return target


def _geometry_notes(
    image: Any, path: str | os.PathLike[str], *, assume_geometry: bool
) -> list[tuple[str, str, dict[str, Any]]]:
    """What the header left ambiguous, as ``(kind, message, detail)`` triples.

    Two cases, both of which used to pass silently with the report saying "0
    guesses":

    ``sform_code == qform_code == 0`` is NIfTI stating that the file carries no
    spatial mapping at all --- the data has voxel indices and nothing else.
    nibabel still hands back an affine, built from ``pixdim``, and importing
    that mints a world grid nobody measured.  Refused unless the caller says
    otherwise, because "geometry is never invented" is the rule this converter
    exists to keep.

    An sform and a qform that *disagree* is the classic signature of a file
    edited by a tool that updated one and not the other.  Preferring the sform
    is conventional and is what nibabel does; it is still a decision made from
    contradictory data, and the file that records it should say so, because a
    downstream tool preferring the qform will place the volume somewhere else.
    """
    header = image.header
    sform_code = int(header["sform_code"])
    qform_code = int(header["qform_code"])
    notes: list[tuple[str, str, dict[str, Any]]] = []
    if sform_code == 0 and qform_code == 0:
        if not assume_geometry:
            raise MEDH5ValidationError(
                f"{os.fspath(path)!r} declares no spatial mapping "
                "(sform_code = qform_code = 0), so it has no world geometry to "
                "import; nibabel's fallback affine is built from pixdim and is "
                "not a measurement. Pass assume_geometry=True (CLI: "
                "--assume-geometry) to accept that fallback deliberately, and "
                "it will be recorded as a guess."
            )
        notes.append(
            (
                "geometry",
                "the file declares no spatial mapping (sform_code = qform_code "
                "= 0); the grid was taken from pixdim and is assumed, not measured",
                {"sform_code": 0, "qform_code": 0},
            )
        )
        return notes
    if sform_code and qform_code:
        sform = np.asarray(header.get_sform(), dtype=np.float64)
        qform = np.asarray(header.get_qform(), dtype=np.float64)
        if not np.allclose(sform, qform, atol=1e-4, rtol=0):
            notes.append(
                (
                    "geometry",
                    "sform and qform describe different geometry; the sform was "
                    "used. A reader preferring the qform will place this volume "
                    "elsewhere.",
                    {
                        "sform_code": sform_code,
                        "qform_code": qform_code,
                        "max_abs_difference": float(np.abs(sform - qform).max()),
                    },
                )
            )
    return notes


def _replay_notes(log: ConversionReport, name: str, geo: Mapping[str, Any]) -> None:
    """Put `read_nifti`'s geometry notes into the conversion report."""
    for kind, message, detail in geo.get("notes", ()):
        log.guess(kind, f"{name}: {message}", detail)


def read_nifti(
    path: str | os.PathLike[str],
    *,
    coord_system: str = "LPS",
    transpose: bool = True,
    fourth_axis: str = "auto",
    assume_geometry: bool = False,
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One NIfTI file as ``(array, geometry)`` in MEDH5 conventions.

    *fourth_axis* decides what a 4-D series' extra axis is --- ``"time"`` for
    cine, DCE and 4-D CT, ``"channel"`` for multi-b-value DWI and multi-echo
    (§3.6).  ``"auto"`` reads the answer out of the file where it can and
    reports a guess where it cannot.

    *assume_geometry* allows a file that declares **no** spatial mapping
    (``sform_code == qform_code == 0``) to be imported anyway, taking nibabel's
    fallback from ``pixdim``.  It is off by default because that fallback is an
    invented grid, and a grid this library invented is indistinguishable
    downstream from one a scanner measured (§3.3).

    The returned geometry carries ``notes``: what the reader had to decide from
    ambiguous headers, for the caller to put in its conversion report.
    """
    nib = require_nibabel()
    image = nib.load(os.fspath(path))
    notes = _geometry_notes(image, path, assume_geometry=assume_geometry)
    rescale = _rescale(image)
    data = _stored_data(image, rescale)
    # A trailing axis of extent 1 beyond the spatial block carries nothing:
    # writers emit `dim[4] = 1` routinely, and keeping it turns a plain volume
    # into a grid with a degenerate one-frame time axis --- and a 5-D
    # `(x, y, z, 1, n)` multi-echo into something no grid can describe.
    squeezed = [axis for axis in range(data.ndim - 1, 2, -1) if data.shape[axis] == 1]
    for axis in squeezed:
        data = np.squeeze(data, axis=axis)
    affine = convert_world(image.affine, source="RAS", target=coord_system)
    spacing, origin, direction = decompose_affine(affine)
    order = tuple(range(data.ndim))
    if transpose and data.ndim >= 3:
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
    if data.ndim == 2:
        spacing, origin, direction = _reduce_plane(spacing, origin, direction)
    times: tuple[float, ...] | None = None
    time_units: str | None = None
    measured = False
    kind = "spatial"
    stated_by: str | None = None
    b_values: tuple[float, ...] | None = None
    channel_field: str | None = None
    channel_values: tuple[float, ...] | None = None
    if data.ndim == 4:
        frames = int(data.shape[0])
        bids = _bids_axis(path, frames, fourth_axis)
        kind, stated = _fourth_axis(image, path, fourth_axis, bids)
        stated_by = bids[1] if bids is not None and bids[0] == kind else None
        if kind == "time":
            stamps = None if bids is None or bids[0] != "time" else bids[2]
            times, time_units, measured = _time_axis(image, frames, stamps)
        else:
            sidecar = _bval_sidecar(path)
            b_values = None if sidecar is None else _read_bvals(sidecar)
            if bids is not None and bids[0] == "channel":
                channel_field, channel_values = bids[1], bids[2]
        measured = measured if kind == "time" else stated
    return np.ascontiguousarray(data), {
        "notes": notes,
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
        "channel_field": channel_field,
        "channel_values": None if channel_values is None else list(channel_values),
        "stated_by": stated_by,
        "squeezed": squeezed,
        "units": _units(image),
        "dtype": str(data.dtype),
        "rescale": rescale,
        "header": {
            "descrip": _text(image.header.get("descrip")),
            "scl_slope": _number(image.header.get("scl_slope")),
            "scl_inter": _number(image.header.get("scl_inter")),
        },
    }


def _rescale(image: Any) -> tuple[float, float] | None:
    """``scl_slope``/``scl_inter`` as a §4.2 rescale, or ``None`` when nothing scales.

    Read from the array proxy, not the header.  nibabel *consumes* the header
    fields on load --- it moves them onto ``dataobj.slope``/``dataobj.inter``
    and resets ``scl_slope`` to NaN --- so the header of a loaded image says
    "unscaled" whatever the file said.  NIfTI-1's "no scaling" (``scl_slope``
    of 0 or NaN) arrives here as slope 1, intercept 0.
    """
    proxy = image.dataobj
    slope = _number(getattr(proxy, "slope", None))
    inter = _number(getattr(proxy, "inter", None))
    if slope is None or slope == 0.0:
        return None
    inter = 0.0 if inter is None else inter
    if slope == 1.0 and inter == 0.0:
        return None
    return float(slope), float(inter)


def _stored_data(image: Any, rescale: tuple[float, float] | None) -> npt.NDArray[Any]:
    """The voxels as the file stores them; the scaling goes into ``rescale``.

    ``np.asanyarray(image.dataobj)`` applies ``scl_slope``/``scl_inter`` and
    hands back ``float64``, so an ``int16`` CT scaled by its header used to be
    imported as floats with no ``rescale`` attribute --- three times the bytes
    for the same numbers, W907 on the converter's own output, and the two
    header fields read into the report and then ignored.  §4.2 keeps the stored
    dtype and records the scale, exactly as the DICOM importer does with the
    modality LUT.
    """
    proxy = image.dataobj
    if rescale is not None and hasattr(proxy, "get_unscaled"):
        return np.asanyarray(proxy.get_unscaled())
    return np.asanyarray(proxy)


def _as_mask(
    data: npt.NDArray[Any], geometry: Mapping[str, Any]
) -> npt.NDArray[np.bool_]:
    """Non-zero **physical** voxels.

    A mask file may carry a rescale of its own, and the stored values are only
    the mask once it is applied: stored ``1`` under intercept ``-1`` is ``0``.
    """
    values = np.asarray(data)
    rescale = geometry.get("rescale")
    if rescale is not None:
        values = values.astype(np.float64) * rescale[0] + rescale[1]
    out: npt.NDArray[np.bool_] = np.asarray(values != 0)
    return out


# NIfTI intent codes whose extra dimension holds *components* rather than
# frames --- §3.6's "channel" kind rather than its "time" kind.  The RGB and
# RGBA vectors (2003, 2004) belong here as much as the numeric ones: they are
# the "RGB / multi-echo" row, and without them that row fell through to the
# time guess despite the header stating the answer.
CHANNEL_INTENTS = frozenset({1001, 1004, 1005, 1006, 1007, 2003, 2004})
TIME_SERIES_INTENT = 2001

FOURTH_AXES = ("auto", "time", "channel")


def _sidecar(path: str | os.PathLike[str], suffix: str) -> Path | None:
    """The file beside a NIfTI sharing its stem and ending in *suffix*."""
    name = Path(os.fspath(path))
    stem = name.name
    for extension in (".nii.gz", ".nii"):
        if stem.endswith(extension):
            candidate = name.with_name(stem[: -len(extension)] + suffix)
            return candidate if candidate.exists() else None
    return None


def _bval_sidecar(path: str | os.PathLike[str]) -> Path | None:
    """The ``.bval`` file beside a NIfTI, which is what makes a series DWI.

    NIfTI-1 says ``dim[4]`` is time, but every diffusion pipeline in practice
    puts the gradient index there and writes the b-values alongside.  The
    sidecar is the only reliable signal, because such files carry no intent
    code and often a meaningless ``pixdim[4]``.
    """
    return _sidecar(path, ".bval")


def _read_bvals(path: Path) -> tuple[float, ...]:
    text = path.read_text(encoding="utf-8").replace("\n", " ")
    return tuple(float(v) for v in text.split())


# BIDS sidecar fields carrying **one entry per volume**, mapped onto the DICOM
# keyword `acquisition` wants (§4.5) and a short channel label.
#
# Per-volume is the whole test.  A scalar ``EchoTime`` sits in the sidecar of
# every MRI ever converted and says nothing about the fourth axis; a list of
# four of them beside a four-frame file says what those four frames *are*.
# Matching on length is what separates the two, and it is why this reads the
# sidecar's shape rather than merely noting that a sidecar exists.
PER_VOLUME_CHANNEL = {
    "EchoTime": ("EchoTime", "TE"),
    "EchoTimes": ("EchoTime", "TE"),
    "EchoNumber": ("EchoNumbers", "echo"),
    "InversionTime": ("InversionTime", "TI"),
    "FlipAngle": ("FlipAngle", "FA"),
}
PER_VOLUME_TIME = ("VolumeTiming",)


def _read_json(path: Path) -> dict[str, Any]:
    """A BIDS sidecar as a mapping; malformed or unreadable reads as absent."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # a broken sidecar is not a broken NIfTI
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _per_volume(
    sidecar: Mapping[str, Any], fields: Iterable[str], frames: int
) -> tuple[str, tuple[float, ...]] | None:
    """The first *field* holding exactly *frames* numbers, and its values."""
    for field_name in fields:
        value = sidecar.get(field_name)
        if not isinstance(value, (list, tuple)) or len(value) != frames:
            continue
        try:
            return field_name, tuple(float(v) for v in value)
        except (TypeError, ValueError):  # a list of something else
            continue
    return None


def _bids_axis(
    path: str | os.PathLike[str], frames: int, requested: str
) -> tuple[str, str, tuple[float, ...]] | None:
    """What a BIDS JSON sidecar states the fourth axis is, if anything.

    Returns ``(kind, field, values)`` --- the §3.6 kind, the sidecar field that
    settled it, and its per-volume numbers, which are the frame times for a
    time axis and what the channels *mean* for a channel one.

    This is the multi-echo counterpart of the ``.bval`` rule above.  Both rows
    of §3.6's fourth-axis entry are written by the same converters, and the
    JSON sidecar is the convention those converters already emit, so reading it
    needs no new argument and no new guess.
    """
    beside = _sidecar(path, ".json")
    if beside is None:
        return None
    fields = _read_json(beside)
    channel = _per_volume(fields, PER_VOLUME_CHANNEL, frames)
    timing = _per_volume(fields, PER_VOLUME_TIME, frames)
    if channel is not None and timing is not None:
        if requested == "auto":
            raise MEDH5ValidationError(
                f"the sidecar beside {Path(os.fspath(path)).name} states both "
                f"{channel[0]} and {timing[0]} per volume, so it says the "
                "fourth axis is a channel axis and a time axis at once (§3.6) "
                "--- pass `fourth_axis=` to settle it"
            )
        # The caller has settled it, so the refusal above would contradict the
        # advice it gives.  Keep only the evidence that agrees with them.
        channel = channel if requested == "channel" else None
        timing = timing if requested == "time" else None
    if channel is not None:
        return "channel", channel[0], channel[1]
    if timing is not None:
        return "time", timing[0], timing[1]
    return None


def _fourth_axis(
    image: Any,
    path: str | os.PathLike[str],
    requested: str,
    bids: tuple[str, str, tuple[float, ...]] | None,
) -> tuple[str, bool]:
    """Whether a 4-D NIfTI's leading axis is ``time`` or ``channel`` (§3.6).

    §3.6 gives both a row --- cine/DCE/4-D CT under ``time``, multi-b-value DWI
    and multi-echo under ``channel`` --- and NIfTI does not distinguish them in
    ``dim[4]``.  Reading every 4-D series as time labelled every DWI gradient
    axis a time axis and handed it invented per-frame timings.

    Returns the kind and whether the source stated it.  A guess is reported as
    one, and ``fourth_axis=`` overrides it outright.

    Sidecars rank above the header intent because they are what the converters
    that write these files actually populate: a DWI or multi-echo series
    normally carries ``intent_code = 0``, and the sidecar is derived from the
    DICOM the series came from.
    """
    if requested not in FOURTH_AXES:
        raise MEDH5ValidationError(
            f"unknown fourth_axis {requested!r}; expected one of {list(FOURTH_AXES)}"
        )
    if requested != "auto":
        return requested, True
    if _bval_sidecar(path) is not None:
        return "channel", True
    if bids is not None:
        return bids[0], True
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
            "it as a single-slice 3-D volume instead"
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
    step = float(zooms[3]) if len(zooms) > 3 else 0.0
    try:
        _, temporal = image.header.get_xyzt_units()
    except Exception:
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


def _time_axis(
    image: Any, frames: int, stamps: tuple[float, ...] | None = None
) -> tuple[tuple[float, ...], str, bool]:
    """Per-frame acquisition times, and whether they were read or assumed.

    A grid carrying a time axis **MUST** carry ``time_values`` (§3.2), and §3.6
    puts them "per frame" for exactly the cine, DCE and 4-D CT series this
    path converts, so leaving a 4-D import with no frame timing at all was not
    an option --- the source states a temporal zoom and the converter was
    throwing it away.
    """
    if stamps is not None:
        # BIDS `VolumeTiming` is the acquisition time of each volume in
        # seconds.  It is a measurement of exactly what §3.2 asks for, so it
        # beats a ramp rebuilt from `pixdim[4]`, which assumes frames are
        # evenly spaced --- the assumption sparse-sampled fMRI breaks.
        return stamps, "s", True
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
    except Exception:
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
    except Exception:
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
            "convert them separately or pass `fourth_axis=` to settle it"
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
    assume_geometry: bool = False,
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
            assume_geometry=assume_geometry,
        )
        _replay_notes(log, name, geo)
        geometry = _same_grid(geometry, geo, name, log) if geometry else geo
        per_image[name] = geo
        arrays[name] = data
    assert geometry is not None
    _same_axis_kind(per_image)

    mask_arrays: dict[str, npt.NDArray[np.bool_]] = {}
    for name, path in (masks or {}).items():
        data, geo = read_nifti(
            path,
            coord_system=coord_system,
            transpose=transpose,
            assume_geometry=assume_geometry,
        )
        _replay_notes(log, name, geo)
        _same_grid(geometry, geo, name, log)
        mask_arrays[name] = _as_mask(data, geo)

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
            "stated_by": geometry.get("stated_by"),
        }
        log.decision(
            "axis_kinds",
            "the fourth axis holds components rather than frames, so it is a "
            "`channel` axis (§3.6) --- b-values, a sidecar field or an intent "
            "code said so",
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
    if transpose and len(geometry["shape"]) >= 3:
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
            timepoint="tp0",
            time_values=geometry.get("time_values"),
            time_units=geometry.get("time_units"),
        )
        for name, array in arrays.items():
            b_values = per_image[name].get("b_values")
            field = per_image[name].get("channel_field")
            values = per_image[name].get("channel_values")
            frames = geometry["shape"][0]
            channel_names = None
            if b_values and len(b_values) == frames:
                channel_names = tuple(f"b={v:g}" for v in b_values)
            elif field and values and len(values) == frames:
                channel_names = tuple(
                    f"{PER_VOLUME_CHANNEL[field][1]}={v:g}" for v in values
                )
            if b_values:
                # §3.6 puts the b-values in `acquisition` (§4.5); they are what
                # the channel axis *means*, and dropping them leaves a stack of
                # unlabelled volumes.
                writer.acquisition(name, b_values=list(b_values))
            if field and values:
                # Same reasoning, same place, under the DICOM keyword §4.5 asks
                # for: echo times are to a multi-echo stack what b-values are
                # to a DWI one.
                writer.acquisition(name, **{PER_VOLUME_CHANNEL[field][0]: list(values)})
            rescale = per_image[name].get("rescale")
            if rescale is not None:
                log.decision(
                    "value_scale",
                    f"{name}: scl_slope/scl_inter were stored as the image's rescale, "
                    "not applied; read(physical=True) applies them and read() "
                    "returns what the file stores (§4.2)",
                    {"image": name, "slope": rescale[0], "intercept": rescale[1]},
                )
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
                rescale_slope=None if rescale is None else rescale[0],
                rescale_intercept=None if rescale is None else rescale[1],
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
    """Refuse volumes that do not share a grid (spec §3.2).

    Uncoded, deliberately.  §15.2's table describes conditions found *in a MEDH5
    file*, and these are two NIfTI volumes that are not one yet: the shape
    mismatch is not `E202` (an image disagreeing with its grid --- neither of
    these is a grid), and the geometry mismatch is not `E101` (a reference to a
    grid that does not exist --- nothing here is referenced). Reporting them
    under those codes told anything branching on the code an untrue story about
    what had gone wrong.
    """
    if tuple(first["shape"]) != tuple(other["shape"]):
        raise MEDH5ValidationError(
            f"{name!r} has shape {other['shape']}, but the first volume has "
            f"{first['shape']}; resample before converting"
        )
    for key in ("spacing", "origin", "direction"):
        if not np.allclose(
            np.asarray(first[key], dtype=np.float64),
            np.asarray(other[key], dtype=np.float64),
            atol=1e-4,
        ):
            raise MEDH5ValidationError(
                f"{name!r} disagrees with the first volume on {key}; resample "
                "before converting rather than letting a converter do it silently"
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

    ``physical=False`` writes the **stored** values, and then the image's
    rescale is written into the header, so the numbers a conforming reader
    gets are the physical ones either way.  A label volume has no rescale and
    is written as it is.
    """
    import medh5

    target = Path(os.fspath(out))
    with medh5.open(path) as sample:
        image = sample.images[image_id]
        grid = image.grid
        rescale: tuple[float, float] | None = None
        if annotation is not None:
            ann = sample.annotations[annotation]
            data = (
                np.asarray(ann.labelmap(), dtype=np.uint16)
                if class_key is None
                else np.asarray(ann.dense([class_key])[0], dtype=np.uint8)
            )
        else:
            data = np.asarray(image.read(physical=physical))
            if not physical:
                rescale = image.rescale
        data, out_affine = for_export(grid, data)
    return write_nifti(data, out_affine, target, rescale=rescale)


def for_export(
    grid: Any, data: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """An array and affine in MEDH5 conventions, as NIfTI wants them.

    MEDH5 leads with time and trails with ``(z, y, x)``; NIfTI is the other way
    round, so the spatial axes are reversed and the affine's columns with them.
    A 2-D plane is *not* reversed --- ``read_nifti`` does not transpose one
    either, so reversing here would mirror every radiograph --- and its 3x3
    affine is embedded in the 4x4 a NIfTI file carries (§3.6).

    Shared by both exporters, because both had their own copy of the reversal
    and only one of them was ever fixed.
    """
    affine = convert_world(grid.affine, source=grid.coord_system, target="RAS")
    spacing, origin, direction = decompose_affine(affine)
    if data.ndim >= 3:
        spatial = (data.ndim - 1, data.ndim - 2, data.ndim - 3)
        data = np.transpose(data, spatial + tuple(range(data.ndim - 3)))
        index = [2, 1, 0]
        spacing = spacing[index]
        direction = direction[:, index]
    return data, embed_plane(build_affine(spacing, origin, direction))


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
        arrays[name] = _as_mask(data, geo)

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
