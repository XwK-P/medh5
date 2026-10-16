"""RT Structure Set import and export (spec §8.6).

An RTSTRUCT is a set of **planar polygons in patient coordinates**, one per ROI
per slice.  It is not a mask, and the difference matters: a contour is exact at
sub-voxel resolution and a mask is not, so converting one to the other loses
information in one direction and invents it in the other.

This converter therefore stores contours as contours (§8.6, ``space="world"``),
in the world coordinates of the grid they belong to: ``ContourData`` is DICOM
patient coordinates --- millimetres, LPS --- so an import scales them to the
grid's units and, for an RAS grid, flips x and y, and an export does the
inverse.  Rasterising into a voxel annotation is available and **opt-in**,
because the rasterisation rule — winding, holes, whether a boundary voxel
counts — is a decision that belongs in the provenance record rather than in a
library's defaults.  When it is used, the rule is written down and the contours
stay in the file beside the mask.

Holes are read the way planners write them: a contour on a slice that lies
inside an odd number of others of its class is stored with ``role="hole"``,
and the rasteriser fills each slice by the even-odd rule over those nesting
depths, so an island inside a hole is a region again (§8.6).  Treating every
contour as an outer boundary is how a segmented vessel lumen ends up filled
in.  ROIs of one name are one class, and are combined only where that makes
no contour of one a hole in another.

Areas, points and, on request, curves are imported.  A medh5 contour is a
closed polygon in one of its grid's slices (§8.6), so an import takes the ROIs
drawn as ``CLOSED_PLANAR`` or ``CLOSEDPLANAR_XOR`` contours, each in a slice of
the grid, as contours, and the ROIs drawn as ``POINT`` contours --- markers ---
as a ``points`` annotation (§8.5).  An ROI of open curves --- a brachytherapy
applicator, a vessel centreline --- is a ``polylines`` annotation with
``open_curves="polylines"`` (format 1.2 §3), which raises the sample's version
to 1.2; by default it is left out with a warning, rather than fill a line into
an area, or mark a class examined with nothing in it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.errors import MEDH5ValidationError
from medh5.geometry import MM_PER_UNIT
from medh5.io._common import examined_classes, sanitize_key, save_dicom
from medh5.io._patient import (
    _one_frame,
    _require_same_frame,
    _salt_hint,
    frames_agree,
    patient_mapping,
)
from medh5.io.report import ConversionReport

MIN_POLYGON_VERTICES = 3

# The `ContourGeometricType`s that enclose an area (PS3.3 C.8.8.6), which is
# what a medh5 contour is; POINT, OPEN_PLANAR and OPEN_NONPLANAR are not.
_CLOSED = "CLOSED_PLANAR"
_XOR = "CLOSEDPLANAR_XOR"
_POINT = "POINT"
_OPEN_PLANAR = "OPEN_PLANAR"
_OPEN_NONPLANAR = "OPEN_NONPLANAR"
_OPEN = (_OPEN_PLANAR, _OPEN_NONPLANAR)

# What `from_rtstruct(open_curves=...)` takes: leave an ROI of open curves out,
# or import it as `polylines`, the kind format 1.2 adds for them.
_OPEN_CURVES = ("omit", "polylines")

# A polyline joins at least two vertices (format 1.2 §3, E404).
_MIN_POLYLINE_VERTICES = 2

# A DICOM Short String (SH) holds at most 16 characters: `StructureSetLabel`.
_SH_LENGTH = 16


def read_rtstruct(
    path: str | os.PathLike[str],
) -> tuple[dict[int, list[npt.NDArray[np.float64]]], dict[str, Any]]:
    """An RTSTRUCT as ``{roi number: [contour, ...]}`` in patient coordinates.

    The contours are those of three points or more, and ``meta`` says what
    each one is and what was left out:

    - ``types``: per ROI, each contour's ``ContourGeometricType``, in the
      order of its contours --- ``None`` where one states none;
    - ``dropped``: per ROI, the type of each contour of fewer than three
      points;
    - ``points``: per ROI, the points of each ``POINT`` contour, as a
      ``(k, 3)`` array whatever *k* is (one, in a well-formed file);
    - ``curves``: per ROI, each ``OPEN_PLANAR`` or ``OPEN_NONPLANAR``
      contour as ``(type, points)``, the points a ``(k, 3)`` array whatever
      *k* is, in the order of its contours;
    - ``roi_frames``: per ROI, the frame of reference it is in --- its own
      ``ReferencedFrameOfReferenceUID``, else the one frame the file lists,
      else ``None``;
    - ``frames``: the frames the file lists
      (``ReferencedFrameOfReferenceSequence``);
    - ``frame_uid``: the frame every ROI with contours is in, or ``None``
      where none states one or they are in several (``roi_frames`` says
      which ROI is in which).

    ``frame_uid`` was the first frame the file listed, whichever frame each
    ROI stated, and the contours' types were not read.  An open curve of two
    points was in ``dropped`` alone, and could not be imported.
    """
    from medh5.io.dicom import _dcmread

    dataset = _dcmread(path)
    if getattr(dataset, "Modality", None) != "RTSTRUCT":
        raise MEDH5ValidationError(
            f"{path} declares Modality="
            f"{getattr(dataset, 'Modality', None)!r}, not 'RTSTRUCT'"
        )
    listed = list(
        dict.fromkeys(
            str(item.FrameOfReferenceUID)
            for item in getattr(dataset, "ReferencedFrameOfReferenceSequence", [])
            if getattr(item, "FrameOfReferenceUID", None)
        )
    )
    names: dict[int, str] = {}
    stated: dict[int, str] = {}
    for roi in getattr(dataset, "StructureSetROISequence", []):
        number = int(roi.ROINumber)
        names[number] = str(getattr(roi, "ROIName", f"roi_{number}"))
        if getattr(roi, "ReferencedFrameOfReferenceUID", None):
            stated[number] = str(roi.ReferencedFrameOfReferenceUID)
    colors: dict[int, tuple[int, int, int, int]] = {}
    contours: dict[int, list[npt.NDArray[np.float64]]] = {}
    types: dict[int, list[str | None]] = {}
    dropped: dict[int, list[str | None]] = {}
    marked: dict[int, list[npt.NDArray[np.float64]]] = {}
    curves: dict[int, list[tuple[str, npt.NDArray[np.float64]]]] = {}
    for item in getattr(dataset, "ROIContourSequence", []):
        number = int(item.ReferencedROINumber)
        rgb = getattr(item, "ROIDisplayColor", None)
        if rgb is not None and len(rgb) >= 3:
            colors[number] = (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
        for contour in getattr(item, "ContourSequence", []):
            data = np.asarray([float(v) for v in contour.ContourData], dtype=np.float64)
            if data.size % 3:  # pragma: no cover - malformed source
                raise MEDH5ValidationError(
                    f"ROI {number} has a contour whose ContourData is not a "
                    "multiple of 3"
                )
            points = data.reshape(-1, 3)
            kind = str(getattr(contour, "ContourGeometricType", "") or "") or None
            if kind == _POINT:
                marked.setdefault(number, []).append(points)
            elif kind in _OPEN:
                curves.setdefault(number, []).append((kind, points))
            if points.shape[0] >= MIN_POLYGON_VERTICES:
                contours.setdefault(number, []).append(points)
                types.setdefault(number, []).append(kind)
            else:
                dropped.setdefault(number, []).append(kind)
    # An ROI stating no frame is in the file's, where the file lists one.
    only = listed[0] if len(listed) == 1 else None
    roi_frames = {
        number: stated.get(number, only)
        for number in sorted({*names, *contours, *dropped})
    }
    used = {roi_frames[number] for number in contours} - {None}
    return contours, {
        "names": names,
        "colors": colors,
        "types": types,
        "dropped": dropped,
        "points": marked,
        "curves": curves,
        "roi_frames": roi_frames,
        "frames": listed,
        "frame_uid": used.pop() if len(used) == 1 else None,
        "label": str(getattr(dataset, "StructureSetLabel", "")),
    }


def from_rtstruct(
    path: str | os.PathLike[str],
    sample: str | os.PathLike[str],
    *,
    ann_id: str = "contours",
    grid: str | None = None,
    rasterize: bool = False,
    mask_id: str | None = None,
    annotated_classes: Sequence[str] | str = "all_given",
    frame_salt: str | None = None,
    report: ConversionReport | None = None,
    roi_classes: Mapping[int, str] | None = None,
    points_id: str | None = None,
    open_curves: str = "omit",
    curves_id: str | None = None,
) -> ConversionReport:
    """Import an RTSTRUCT's ROIs as contours, and optionally rasterise them.

    The ROIs drawn as areas are imported: ``CLOSED_PLANAR`` contours, and
    ``CLOSEDPLANAR_XOR`` ones that are nested or apart on every slice, where
    their exclusive-or is the holes this importer marks.  An ROI of
    ``OPEN_PLANAR`` or ``OPEN_NONPLANAR`` contours is left out with a
    warning by default (*open_curves* ``"omit"``): a medh5 contour is a
    closed polygon (§8.6), and a filled line, or a class examined with
    nothing in it, would be invented.  An ROI mixing areas with anything else, or
    exclusive-or with another type (PS3.3 C.8.8.6.1), is refused, and so are
    exclusive-or contours that cross; a contour stating no type is read as
    ``CLOSED_PLANAR``, on a recorded guess.  A contour of fewer than three
    points encloses nothing, and its drop is a warning.

    An ROI of ``POINT`` contours --- markers --- is imported as rows of a
    ``points`` annotation (§8.5), *points_id* (``f"{ann_id}_points"`` by
    default): one row per contour, in world coordinates of the contours'
    frame, of the ROI's class, named for the ROI, with ``task="other"``
    (a marker is no detection finding).  A POINT contour of more than one
    point is refused.  A structure set of points alone writes no contours
    annotation.  An explicit *annotated_classes* list is split: a class
    drawn as points is examined in the points annotation, every other class
    it names in the contours one; ``"all"`` applies to the contours only.
    A class the list names that no ROI supplies was examined and found
    absent (§11.3): minted after the ROIs' classes when the sample has no
    label set, else one its label set must hold (E402).

    With *open_curves* ``"polylines"``, an ROI of open curves is imported as
    rows of a ``polylines`` annotation (format 1.2 §3), *curves_id*
    (``f"{ann_id}_curves"`` by default): one polyline per contour, in world
    coordinates of the contours' frame, of the ROI's class, with
    ``task="other"``.  An ``OPEN_PLANAR`` curve records the plane of the grid
    it lies in (``polyline_plane``), found as a contour's slice is, along any
    of the grid's axes; one that lies in none, and every ``OPEN_NONPLANAR``
    curve, has ``axis = -1``.  A curve of fewer than two points joins
    nothing, and its drop is a warning.  *annotated_classes* is split as for
    points: a class drawn as curves is examined in the curves annotation.
    The option is opt-in because a ``polylines`` annotation raises the
    sample's version to 1.2, which a 1.1 reader reads as its projection,
    leaving the curves out (``W913``); the report records the raise.

    Every contour must lie in a slice of the grid, along its first axis: one
    between two slices, across several (drawn on another orientation, or not
    planar), or outside the grid is refused rather than moved onto a slice.

    The contours go onto the grid that shares the ROIs' frame of reference,
    or the sample's only grid; a grid --- matched or named with *grid* ---
    whose frame is known and is another is refused (E414) before the sample
    is touched: equal numbers in two frames are no correspondence (§3.4).
    Each ROI is in the frame it states, else the one the file lists, and ROIs
    in several frames are refused (E414) before the sample is opened.  Where
    the RTSTRUCT or the grid states no frame, the placement rests on the
    coordinates alone and the report records a guess.  *frame_salt* is the
    salt a ``medh5 scrub`` pseudonymised the sample's frames with
    (:func:`~medh5.io.dicom_seg.frames_agree`).

    ``ContourData`` is patient coordinates (LPS, millimetres), and the
    contours are stored in the grid's world coordinates: scaled to its units
    and, for an RAS grid, flipped in x and y --- the inverse of
    :func:`to_rtstruct`.  A grid in pixels or another convention has no
    patient coordinates, and is refused (E414).  Stored as read, a contour on
    an RAS grid was mirrored through the origin in x and y, and one on a
    metre grid lay a thousand times as far out.

    Each ROI is a class: the one whose key its ``ROIName`` makes, or, in a
    sample that has a label set, the class of that key, name or display
    name.  ROIs that come to one class are combined into it, with a
    ``merged_rois`` decision naming them, and their contours' roles come
    from nesting over the whole class, plane by plane (§8.6).  Outlines of
    two such ROIs that cross are their union; a contour of one inside a
    contour of another on a plane would become a hole in what both drew as
    an area, so that import is refused, naming the ROIs and the plane.
    *roi_classes* (``{ROINumber: key}``; ``--roi NUMBER=KEY`` on the command
    line) puts an ROI in the class of that key instead --- minted, or one
    the sample's label set must hold (E402) --- which imports same-named
    ROIs apart.  Two ROIs of one name, one inside the other, were imported
    as a hole, and minting a class for each failed (E302).
    """
    import medh5
    from medh5.labels import LabelClass, LabelSet

    if open_curves not in _OPEN_CURVES:
        raise MEDH5ValidationError(
            f"open_curves={open_curves!r}: it is 'omit', which leaves an ROI of "
            "open curves out with a warning, or 'polylines', which imports it "
            "as a polylines annotation (format 1.2)"
        )
    log = report or ConversionReport(converter="from-rtstruct")
    log.source = os.fspath(path)
    contours, meta = read_rtstruct(path)
    contours, xor, marked, curved = _areas(contours, meta, log, path, open_curves)
    frame = _contour_frame({**contours, **marked, **curved}, meta, log, path)

    with medh5.open(sample) as opened:
        grid_id = grid or _match_grid(opened, frame, log, frame_salt)
        target = opened.grids[grid_id]
        existing = opened.label_set
        version = opened.version
    _require_same_frame(frame, target, frame_salt, source="the RTSTRUCT")
    signs, scale = patient_mapping(target, "RTSTRUCT contours")
    if frames_agree(frame, target.frame_uid, frame_salt) is None:
        log.guess(
            "frame",
            f"{'the RTSTRUCT' if not frame else f'grid {grid_id!r}'} "
            "states no frame of reference, so the frames could not be compared; "
            "the contours were placed by their coordinates alone",
            {"grid": grid_id, "rtstruct_frame": frame},
        )
    # Patient coordinates into the grid's world: `patient = signs * scale *
    # world`, and the signs are their own inverse.
    contours = {
        number: [signs * points / scale for points in planar]
        for number, planar in contours.items()
    }
    marked = {number: signs * points / scale for number, points in marked.items()}
    curved = {
        number: [(kind, signs * points / scale) for kind, points in drawn]
        for number, drawn in curved.items()
    }
    rois = sorted({*contours, *marked, *curved})

    names = {number: meta["names"].get(number, f"roi_{number}") for number in rois}
    keys = _roi_keys(rois, names, roi_classes, meta, path)
    label_set = existing
    ids: dict[int, int] = {}
    if label_set is None:
        # ROIs whose names make one key are one class: a key is unique in a
        # label set (E302), and minting a class for each failed the write.
        firsts: dict[str, int] = {}
        for number in rois:
            firsts.setdefault(keys[number], number)
        colours = {
            key: next(
                (
                    meta["colors"][n]
                    for n in rois
                    if keys[n] == key and n in meta["colors"]
                ),
                None,
            )
            for key in firsts
        }
        classes = [
            LabelClass(i + 1, key, names[first], color=colours[key])
            for i, (key, first) in enumerate(firsts.items())
        ]
        label_set = LabelSet("rtstruct", version="1.0.0", classes=classes)
        minted = {key: i + 1 for i, key in enumerate(firsts)}
        ids = {number: minted[keys[number]] for number in rois}
        log.decision(
            "label_set",
            f"{len(classes)} class(es) were minted from ROIName, keeping each "
            "ROI's display colour",
            {"names": {str(k): meta["names"].get(k) for k in rois}},
        )
    else:
        missing = []
        for number in rois:
            name = names[number]
            if roi_classes is not None and number in roi_classes:
                found = label_set.get(keys[number])
                if found is None:
                    raise MEDH5ValidationError(
                        f"roi_classes maps ROI {number} ({name!r}) to class "
                        f"{keys[number]!r}, which the sample's label set does "
                        "not hold",
                        code="E402",
                    )
            else:
                found = (
                    label_set.get(_key(name))
                    or label_set.get(name)
                    or _named(label_set, name)
                )
            if found is None:
                missing.append(name)
            else:
                ids[number] = found.id
        if missing:
            raise MEDH5ValidationError(
                f"the sample's label set has no class for ROI(s) {missing}",
                code="E402",
            )

    polygons = _combine(contours, ids, names, label_set, target, log, xor)
    points_id = points_id or f"{ann_id}_points"
    curves_id = curves_id or f"{ann_id}_curves"
    polylines = _polylines(curved, ids, names, target, log)
    label_set, examined = examined_classes(
        annotated_classes,
        label_set,
        mint=existing is None,
        log=log,
        supplied="the ROIs",
    )
    on_contours, on_points, on_curves = _split_examined(
        examined,
        [ids[n] for n in sorted(contours)],
        [ids[n] for n in sorted(marked)],
        label_set,
        log,
        on_curves=[ids[n] for n in sorted(curved)],
    )
    if xor:
        rois = [meta["names"].get(number, f"roi_{number}") for number in sorted(xor)]
        log.decision(
            "contour_type",
            f"ROI(s) {rois} combine their contours by exclusive-or "
            "(CLOSEDPLANAR_XOR); on every slice they are nested or apart, so "
            "their exclusive-or is the nesting the holes record",
            {"rois": rois},
        )
    if contours:
        log.decision(
            "contours",
            f"{len(polygons)} contour(s) were stored in grid {grid_id!r}'s world "
            f"coordinates ({target.coord_system}, {target.units}), converted from "
            "patient coordinates (LPS, mm) but otherwise exactly as the planner "
            "drew them; a contour is sub-voxel and a mask is not",
            {
                "polygons": len(polygons),
                "rois": len(contours),
                "coord_system": target.coord_system,
                "units": target.units,
            },
        )
    if marked:
        log.decision(
            "points",
            f"{sum(len(p) for p in marked.values())} POINT contour(s) of "
            f"{len(marked)} ROI(s) were stored as annotation {points_id!r}, one "
            f"row each, in grid {grid_id!r}'s world coordinates, each row named "
            "for its ROI and of its class, with task 'other': a marker is a "
            "landmark (§8.5), not a detection finding",
            {
                "annotation": points_id,
                "rois": {str(n): names[n] for n in sorted(marked)},
                "task": "other",
            },
        )
    if curved:
        planar = sum(1 for line in polylines if line.plane[0] >= 0)
        log.decision(
            "curves",
            f"{len(polylines)} open curve(s) of {len(curved)} ROI(s) were stored "
            f"as polylines annotation {curves_id!r} in grid {grid_id!r}'s world "
            f"coordinates, {planar} of them with the grid plane they lie in, "
            "each of its ROI's class, with task 'other'",
            {
                "annotation": curves_id,
                "rois": {str(n): names[n] for n in sorted(curved)},
                "planar": planar,
                "task": "other",
            },
        )
        if tuple(int(part) for part in version.split(".")[:2]) < (1, 2):
            log.decision(
                "version",
                f"a polylines annotation is format 1.2 (§3), so the sample, "
                f"written at {version}, is now written at 1.2: a 1.1 reader "
                "reads it as its projection, leaving the curves out (W913)",
                {"from": version, "to": "1.2"},
            )

    masks: dict[int, npt.NDArray[np.bool_]] | None = None
    if rasterize and not contours:
        log.warn(
            "rasterization",
            "the structure set holds no areas, only points or curves, so there "
            "was nothing to rasterise",
            {"rois": len(marked) + len(curved)},
        )
    elif rasterize:
        masks = _rasterize(polygons, target, log)
        empty = [label_set[c].key for c, mask in masks.items() if not mask.any()]
        if empty:
            log.warn(
                "rasterization",
                f"the contours of class(es) {empty} enclose no voxel centre of "
                f"grid {grid_id!r}, so their rasterised masks are empty --- read "
                "as examined and absent, where the contours outline something "
                "smaller than a voxel",
                {"classes": empty},
            )

    with medh5.amend(sample) as writer:
        if existing is None:
            writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import", agent=tool, tool="medh5 convert from-rtstruct"
        )
        # The grid's frame, which the coordinates are now in: a frame
        # compared through the scrub's salt is the grid's pseudonym, and the
        # RTSTRUCT's own UID is what the scrub removed.
        frame_uid = target.frame_uid or frame
        if contours:
            writer.add_contours(
                ann_id,
                polygons,
                grid=grid_id,
                space="world",
                frame_uid=frame_uid,
                annotated_classes=on_contours,
                prov=activity,
            )
        if marked:
            numbers = sorted(marked)
            writer.add_points(
                points_id,
                np.concatenate([marked[n] for n in numbers]).astype(np.float32),
                grid=grid_id,
                space="world",
                frame_uid=frame_uid,
                class_ids=[ids[n] for n in numbers for _ in marked[n]],
                names=[names[n] for n in numbers for _ in marked[n]],
                annotated_classes=on_points,
                task="other",
                prov=activity,
            )
        if curved:
            writer.add_polylines(
                curves_id,
                polylines,
                grid=grid_id,
                space="world",
                frame_uid=frame_uid,
                annotated_classes=on_curves,
                prov=activity,
            )
        if masks:
            raster = writer.activity(
                "derive",
                agent=tool,
                tool="medh5 convert from-rtstruct --rasterize",
                inputs=[f"annotations/{ann_id}"],
                params={
                    "rule": "even-odd fill at voxel centres, per plane",
                    "sampling": "voxel centre",
                },
            )
            kind, _ = writer.add_segmentation(
                mask_id or f"{ann_id}_mask",
                grid=grid_id,
                masks=masks,
                annotated_classes=on_contours,
                prov=raster,
                derived_from=[ann_id],
            )
            log.decision(
                "rasterization",
                "contours were rasterised by even-odd fill at voxel centres, per "
                f"plane, stored as {kind!r}; the contours were kept, so the lossy "
                "step is reversible by re-deriving it",
                {"kind": kind, "rule": "even-odd at voxel centres"},
            )
    log.outputs.append(os.fspath(sample))
    return log


def _split_examined(
    annotated_classes: Sequence[int | str] | str,
    on_contours: Sequence[int],
    on_points: Sequence[int],
    label_set: Any,
    log: ConversionReport,
    *,
    on_curves: Sequence[int] = (),
) -> tuple[
    Sequence[int | str] | str, Sequence[int | str] | str, Sequence[int | str] | str
]:
    """*annotated_classes* for the contours annotation, the points one and
    the curves one.

    A list is split: a class drawn as points is examined in the points
    annotation, one drawn as curves in the curves annotation, every other
    class it names in the contours annotation (and a class drawn both as
    areas and otherwise in both).  ``"all"`` is the contours annotation's:
    a structure set's markers and curves are not a search of every class.
    A structure set with no areas writes no contours annotation, so there
    every class a list names that no ROI draws goes to the points
    annotation, else the curves one: it was examined and found absent, and
    would otherwise be lost.
    """
    others = [
        name for name, drawn in (("points", on_points), ("curves", on_curves)) if drawn
    ]
    if isinstance(annotated_classes, str):
        if annotated_classes == "all" and others:
            which = " and ".join(others)
            log.decision(
                "annotated_classes",
                "annotated_classes='all' applies to the contours annotation; "
                + (
                    f"the {which} annotation examined the classes it holds"
                    if len(others) == 1
                    else f"the {which} annotations examined the classes they hold"
                ),
                {name: "all_given" for name in others},
            )
        return annotated_classes, "all_given", "all_given"
    points, curves, contours = set(on_points), set(on_curves), set(on_contours)

    def class_id(entry: Any) -> int | None:
        found = label_set.get(entry)
        return None if found is None else int(found.id)

    def shown(entries: Sequence[int | str]) -> list[str]:
        return [label_set[c].key if c in label_set else str(c) for c in entries]

    found = {entry: class_id(entry) for entry in annotated_classes}
    nowhere = [c for c in annotated_classes if found[c] not in points | curves]
    to_points = [c for c in annotated_classes if found[c] in points]
    to_curves = [c for c in annotated_classes if found[c] in curves]
    to_contours = [c for c in annotated_classes if c in nowhere or found[c] in contours]
    if not on_contours:
        if on_points:
            to_points = [
                c for c in annotated_classes if found[c] in points or c in nowhere
            ]
        elif on_curves:
            to_curves = [
                c for c in annotated_classes if found[c] in curves or c in nowhere
            ]
        if len(others) < 2:
            return [], to_points, to_curves
        to_contours = []
    parts = []
    if to_points and (on_contours or on_curves):
        parts.append(
            f"class(es) {shown(to_points)} are drawn as points, so they are "
            "examined in the points annotation"
        )
    if to_curves and (on_contours or on_points):
        parts.append(
            f"class(es) {shown(to_curves)} are drawn as open curves, so they are "
            "examined in the curves annotation"
        )
    if parts:
        rest = "; the others named, in the contours annotation" if on_contours else ""
        detail = {"points": shown(to_points), "contours": shown(to_contours)}
        if on_curves:
            detail["curves"] = shown(to_curves)
        log.decision("annotated_classes", "; ".join(parts) + rest, detail)
    return to_contours, to_points, to_curves


def _areas(
    contours: Mapping[int, list[npt.NDArray[np.float64]]],
    meta: Mapping[str, Any],
    log: ConversionReport,
    path: str | os.PathLike[str],
    open_curves: str = "omit",
) -> tuple[
    dict[int, list[npt.NDArray[np.float64]]],
    set[int],
    dict[int, npt.NDArray[Any]],
    dict[int, list[tuple[str, npt.NDArray[np.float64]]]],
]:
    """The ROIs drawn as areas, which a medh5 contour can hold, which of them
    combine their contours by exclusive-or, the ROIs drawn as points, each as
    its ``(n, 3)`` points (§8.5), and, with *open_curves* ``"polylines"``, the
    ROIs drawn as open curves, each as its ``(type, points)`` curves of two
    points or more (format 1.2 §3).

    ``ContourGeometricType`` was never read: an open curve was filled as an
    area, an applicator's path flattened onto one slice --- its class then
    marked examined with nothing in it --- and a POINT ROI vanished, as every
    contour of fewer than three points did, without a note.  Until 2.1 a
    POINT ROI was left out with a warning, and until 2.2 every open curve.
    """
    names = meta["names"]
    kept: dict[int, list[npt.NDArray[np.float64]]] = {}
    xor: set[int] = set()
    marked: dict[int, npt.NDArray[Any]] = {}
    curved: dict[int, list[tuple[str, npt.NDArray[np.float64]]]] = {}
    untyped: dict[str, int] = {}
    skipped: list[str] = []
    types: Mapping[int, list[str | None]] = meta["types"]
    dropped: Mapping[int, list[str | None]] = meta["dropped"]
    for number in sorted({*types, *dropped}):
        name = names.get(number, f"roi_{number}")
        stated = [*types.get(number, []), *dropped.get(number, [])]
        if None in stated:
            untyped[name] = stated.count(None)
        # The type is Type 1: one stating none is read as the type nearly
        # every structure set is drawn in, on the guess recorded below.
        found = sorted({kind or _CLOSED for kind in stated})
        areas = [kind for kind in found if kind in (_CLOSED, _XOR)]
        others = [kind for kind in found if kind not in (_CLOSED, _XOR)]
        if areas and others:
            raise MEDH5ValidationError(
                f"ROI {name!r} in {path} mixes {' and '.join(areas)} contours "
                f"with {' and '.join(others)} ones: an ROI is an area or a set "
                "of points or curves, and which one this is cannot be told"
            )
        if _POINT in others and len(others) > 1:
            raise MEDH5ValidationError(
                f"ROI {name!r} in {path} mixes {' and '.join(others)} contours: "
                "an ROI is a set of points or a curve, and which one this is "
                "cannot be told"
            )
        if others == [_POINT]:
            drawn = meta["points"].get(number, [])
            for points in drawn:
                if len(points) != 1:
                    raise MEDH5ValidationError(
                        f"ROI {name!r} has a POINT contour of {len(points)} "
                        f"points in {path}: a POINT contour is one point "
                        "(PS3.3 C.8.8.6), and which one it marks cannot be told"
                    )
            marked[number] = np.concatenate(drawn)
            continue
        if others and open_curves == "polylines" and set(others) <= set(_OPEN):
            drawn = meta["curves"].get(number, [])
            lines = [(k, c) for k, c in drawn if len(c) >= _MIN_POLYLINE_VERTICES]
            if len(lines) < len(drawn):
                short = len(drawn) - len(lines)
                log.warn(
                    "dropped_contours",
                    f"{short} open curve(s) of ROI {name!r} have fewer than "
                    f"{_MIN_POLYLINE_VERTICES} points and join nothing, so they "
                    "were not imported" + ("" if lines else ", and nor was the ROI"),
                    {"roi": name, "contours": short},
                )
            if lines:
                curved[number] = lines
            else:
                skipped.append(f"{name!r} has no curve of two points")
            continue
        if others:
            for kind in others:
                count = stated.count(kind)
                hint = (
                    "open_curves='omit'; open_curves='polylines' imports it as "
                    "a polylines annotation (format 1.2)"
                    if kind in _OPEN
                    else f"open_curves={open_curves!r}"
                )
                log.warn(
                    "contour_type",
                    f"ROI {name!r} is drawn as {count} {kind} contour(s), and "
                    "a medh5 contour is a closed polygon (§8.6), so the ROI was "
                    f"not imported ({hint}), and no class is marked examined "
                    "for it",
                    {"roi": name, "type": kind, "contours": count},
                )
            skipped.append(f"{name!r} is {' and '.join(others)}")
            continue
        if len(areas) > 1:
            raise MEDH5ValidationError(
                f"ROI {name!r} in {path} mixes CLOSEDPLANAR_XOR contours with "
                "CLOSED_PLANAR ones, which PS3.3 C.8.8.6.1 forbids --- an ROI "
                "with one exclusive-or contour is all exclusive-or --- so how "
                "they combine is not stated"
            )
        short = len(dropped.get(number, []))
        if short:
            log.warn(
                "dropped_contours",
                f"{short} contour(s) of ROI {name!r} have fewer than "
                f"{MIN_POLYGON_VERTICES} points and enclose no area, so they "
                "were not imported"
                + ("" if number in contours else ", and nor was the ROI"),
                {"roi": name, "contours": short},
            )
        if number in contours:
            kept[number] = list(contours[number])
            if areas == [_XOR]:
                xor.add(number)
    if untyped:
        log.guess(
            "contour_type",
            f"{sum(untyped.values())} contour(s) of ROI(s) {sorted(untyped)} "
            "state no ContourGeometricType and were read as CLOSED_PLANAR",
            {"rois": untyped},
        )
    if not kept and not marked and not curved:
        why = f": {'; '.join(skipped)}" if skipped else ""
        if open_curves == "omit" and any(
            kind in _OPEN
            for kinds in (*types.values(), *dropped.values())
            for kind in kinds
        ):
            why += " (open_curves='polylines' imports open curves)"
        raise MEDH5ValidationError(f"{path} contains no usable contours or points{why}")
    return kept, xor, marked, curved


def _polylines(
    curved: Mapping[int, list[tuple[str, npt.NDArray[np.float64]]]],
    ids: Mapping[int, int],
    names: Mapping[int, str],
    grid: Any,
    log: ConversionReport,
) -> list[Any]:
    """Every ROI's open curves as polylines of its class, in the grid's world
    coordinates (format 1.2 §3), each ``OPEN_PLANAR`` curve with the plane of
    *grid* it lies in."""
    from medh5.annotations.geometric import Polyline

    out: list[Any] = []
    loose: dict[str, int] = {}
    for number in sorted(curved):
        for kind, points in curved[number]:
            plane = (-1, 0)
            if kind == _OPEN_PLANAR:
                plane = _curve_plane(grid.world_to_index(points), grid)
                if plane[0] < 0:
                    loose[names[number]] = loose.get(names[number], 0) + 1
            out.append(
                Polyline(
                    vertices=points.astype(np.float32),
                    class_id=ids[number],
                    plane=plane,
                )
            )
    if loose:
        log.decision(
            "curve_plane",
            f"{sum(loose.values())} OPEN_PLANAR curve(s) of ROI(s) {sorted(loose)} "
            f"lie in no plane of grid {grid.grid_id!r}, so they were stored "
            "with axis -1, as drawn, rather than moved onto one",
            {"rois": loose, "grid": grid.grid_id},
        )
    return out


def _curve_plane(index: npt.NDArray[np.float64], grid: Any) -> tuple[int, int]:
    """The ``(axis, index)`` of the plane of *grid* that a curve lies in, given
    its vertices in the grid's index space, the first axis first; ``(-1, 0)``
    where it lies in none.

    Within a tenth of a slice, and at least 0.01 mm, as :func:`_slice` holds
    a contour to its slice.
    """
    from medh5.io.dicom import _position_tolerance

    shape = grid.spatial_shape
    for axis in range(len(shape)):
        step = float(grid.spacing[grid.spatial_axes[axis]]) * MM_PER_UNIT.get(
            grid.units, 1.0
        )
        along = index[:, axis]
        at = int(round(float(np.median(along))))
        off = float(np.max(np.abs(along - at))) * step
        if 0 <= at < shape[axis] and off <= _position_tolerance(step):
            return axis, at
    return -1, 0


def _contour_frame(
    contours: Mapping[int, Any],
    meta: Mapping[str, Any],
    log: ConversionReport,
    path: str | os.PathLike[str],
) -> str | None:
    """The one frame of reference the imported ROIs are in (§3.4).

    Each ROI is in the frame it states; one stating none is in the frame the
    file lists, where it lists one.  The first frame the file listed stood
    for every ROI, so a structure set in two frames was accepted or refused
    whole, by the order of its list.
    """
    names = meta["names"]
    listed: list[str] = meta["frames"]
    frames = {number: meta["roi_frames"].get(number) for number in sorted(contours)}
    unknown = [names.get(n, f"roi_{n}") for n, f in frames.items() if f is None]
    if unknown and len(listed) > 1:
        raise MEDH5ValidationError(
            f"ROI(s) {unknown} in {path} state no frame of reference, and the "
            f"file lists {len(listed)} ({', '.join(listed)}), so which one their "
            "contours are in is not stated",
            code="E414",
        )
    used = sorted({f for f in frames.values() if f})
    if len(used) > 1:
        rois = "; ".join(
            f"{uid}: "
            + ", ".join(
                repr(names.get(n, f"roi_{n}")) for n, f in frames.items() if f == uid
            )
            for uid in used
        )
        raise MEDH5ValidationError(
            f"the ROIs of {path} are in {len(used)} frames of reference "
            f"({rois}); one contours annotation is in one frame, and relating "
            "two is a registration, which this importer does not apply",
            code="E414",
        )
    if unknown and used:
        log.guess(
            "frame",
            f"ROI(s) {unknown} state no frame of reference, and {path} lists "
            f"none; they were taken to be in {used[0]}, the other ROIs' frame",
            {"rois": unknown, "frame": used[0]},
        )
    unlisted = [
        names.get(n, f"roi_{n}") for n, f in frames.items() if f and f not in listed
    ]
    if unlisted:
        log.warn(
            "frame",
            f"ROI(s) {unlisted} are in frame of reference {used[0]}, which "
            f"{path} does not list in its ReferencedFrameOfReferenceSequence, "
            "as every frame its ROIs are in must be (PS3.3 C.8.8.5); the "
            "ROIs' own frame was used",
            {"rois": unlisted, "frame": used[0], "listed": listed},
        )
    return used[0] if used else None


def _named(label_set: Any, name: str) -> Any:
    """The one class whose display name is *name*, which :func:`to_rtstruct`
    writes as the ROIName; ``None`` for none, or several."""
    found = [c for c in label_set if c.name == name]
    return found[0] if len(found) == 1 else None


def _roi_keys(
    numbers: Sequence[int],
    names: Mapping[int, str],
    roi_classes: Mapping[int, str] | None,
    meta: Mapping[str, Any],
    path: str | os.PathLike[str],
) -> dict[int, str]:
    """Each ROI's class key: the one *roi_classes* gives it, else the one its
    name makes.  An ROI number the file does not hold is refused, so a typo
    does not import the ROI it meant into the class of its name."""
    given = dict(roi_classes or {})
    held = {*meta["names"], *meta["types"], *meta["dropped"]}
    unknown = sorted(n for n in given if n not in held)
    if unknown:
        raise MEDH5ValidationError(
            f"roi_classes names ROI number(s) {unknown}, which {path} does not "
            f"hold (it holds {sorted(held)})"
        )
    blank = sorted(n for n, k in given.items() if not isinstance(k, str) or not k)
    if blank:
        raise MEDH5ValidationError(
            f"roi_classes gives ROI number(s) {blank} no class key"
        )
    return {n: given[n] if n in given else _key(names[n]) for n in numbers}


def _combine(
    contours: Mapping[int, list[npt.NDArray[np.float64]]],
    ids: Mapping[int, int],
    names: Mapping[int, str],
    label_set: Any,
    grid: Any,
    log: ConversionReport,
    xor: set[int],
) -> list[Any]:
    """Every ROI's polygons, their roles from nesting over each class, plane
    by plane (§8.6).

    Roles came from nesting within each ROI while the rasteriser nested per
    class, so an ROI drawn inside another of its name became a hole: an 11×11
    "Bladder" holding a 4×4 one was 105 voxels where 121 were drawn.  ROIs of
    one class are combined only where no contour's depth in the class differs
    from its depth within its own ROI; anywhere else the import is refused.
    """
    from medh5.annotations.geometric import Polygon

    placed = {
        number: _place(contours[number], grid, names[number], xor=number in xor)
        for number in sorted(contours)
    }
    by_class: dict[int, list[int]] = {}
    for number in sorted(contours):
        by_class.setdefault(ids[number], []).append(number)
    depth: dict[tuple[int, int], int] = {}
    for class_id, numbers in by_class.items():
        planes: dict[int, list[tuple[int, int]]] = {}
        for number in numbers:
            for position, (_, _, plane) in enumerate(placed[number]):
                planes.setdefault(plane, []).append((number, position))
        for plane, members in planes.items():
            merged = _core.polygon_depths([placed[n][p][1][:, 1:] for n, p in members])
            own: dict[tuple[int, int], int] = {}
            for number in dict.fromkeys(n for n, _ in members):
                mine = [(n, p) for n, p in members if n == number]
                found = _core.polygon_depths([placed[n][p][1][:, 1:] for n, p in mine])
                own.update(zip(mine, found, strict=True))
            if any(d != own[member] for member, d in zip(members, merged, strict=True)):
                rois = sorted({n for n, _ in members})
                shown = ", ".join(f"{n} ({names[n]!r})" for n in rois)
                raise MEDH5ValidationError(
                    f"ROIs {shown} are one class "
                    f"{label_set[class_id].key!r}, and on slice {plane} of "
                    f"grid {grid.grid_id!r} a contour of one lies inside a "
                    "contour of another: combined, it would be a hole in what "
                    "each ROI drew as an area (§8.6). Import them apart with "
                    "roi_classes= (--roi NUMBER=KEY), a class key for each"
                )
            depth.update(zip(members, merged, strict=True))
        if len(numbers) > 1:
            key = label_set[class_id].key
            log.decision(
                "merged_rois",
                f"ROIs {numbers} "
                f"({', '.join(repr(names[n]) for n in numbers)}) are one class "
                f"{key!r}, so their contours were combined into it, plane by "
                "plane; outlines of two of them that cross are their union",
                {"class": key, "rois": numbers, "names": [names[n] for n in numbers]},
            )
    polygons: list[Polygon] = []
    for number in sorted(contours):
        holes = 0
        for position, (points, _, plane) in enumerate(placed[number]):
            hole = depth[(number, position)] % 2 == 1
            holes += hole
            polygons.append(
                Polygon(
                    vertices=points.astype(np.float32),
                    class_id=ids[number],
                    # `(0, k)`: the polygon lies in slice k along the grid's
                    # first spatial axis, in that grid's index space (§8.6).
                    # It was computed to find holes and then thrown away, so
                    # `by_plane()` on an imported RTSTRUCT answered nothing.
                    plane=(0, plane),
                    role="hole" if hole else "outer",
                )
            )
        _note_holes(log, holes, names[number])
    return polygons


def _assign_roles(
    contours: Sequence[npt.NDArray[np.float64]],
    grid: Any,
    log: ConversionReport,
    name: str | None,
    *,
    xor: bool = False,
) -> list[tuple[npt.NDArray[np.float64], str, int]]:
    """Mark contours inside an odd number of others on the same slice as holes.

    Nesting depth, not mere enclosure: an island drawn inside a hole is inside
    two contours and is a region again.  Marking every enclosed contour a hole
    erased it.

    Returns ``(vertices, role, plane)`` with the plane index that grouping
    found, so the writer can record it rather than lose it.
    """
    placed = _place(contours, grid, name, xor=xor)
    planes: dict[int, list[int]] = {}
    for position, (_, _, plane) in enumerate(placed):
        planes.setdefault(plane, []).append(position)
    depth: dict[int, int] = {}
    for positions in planes.values():
        found = _core.polygon_depths([placed[p][1][:, 1:] for p in positions])
        depth.update(zip(positions, found, strict=True))
    out = [
        (points, "hole" if depth[p] % 2 else "outer", plane)
        for p, (points, _, plane) in enumerate(placed)
    ]
    _note_holes(log, sum(role == "hole" for _, role, _ in out), name)
    return out


def _note_holes(log: ConversionReport, holes: int, name: str | None) -> None:
    """The decision that *holes* contours of ROI *name* are holes."""
    if holes:
        log.decision(
            "holes",
            f"{holes} contour(s) of ROI {name!r} lie inside an odd number of others "
            "on the same slice and were marked as holes; one inside a hole is a "
            "region again",
            {"holes": holes, "roi": name},
        )


def _place(
    contours: Sequence[npt.NDArray[np.float64]],
    grid: Any,
    name: str | None,
    *,
    xor: bool = False,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]]:
    """One ROI's contours as ``(vertices, index, plane)``, grouped by the
    slice each lies in, the slices in the order they first appear.

    The grouping and the enclosure test are both done in the grid's **index**
    space, where a planar contour has one constant axis.  Doing it on world
    coordinates would only work for axial acquisitions: under the orientation
    a DICOM series actually declares, the world *z* of a contour varies within
    its own slice, and grouping on it puts every vertex in its own plane.

    For an *xor* ROI (``CLOSEDPLANAR_XOR``) nesting parity is the
    exclusive-or only while no two contours on a slice cross, so two that do
    are refused, naming the slice: read as nested, they were their union.
    """
    index = [grid.world_to_index(points) for points in contours]
    by_plane: dict[int, list[int]] = {}
    for position, points in enumerate(index):
        by_plane.setdefault(_slice(points, grid, name), []).append(position)
    out: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]] = []
    for plane, positions in by_plane.items():
        if xor:
            for first, second in combinations(positions, 2):
                if _crosses(index[first][:, 1:], index[second][:, 1:]):
                    raise MEDH5ValidationError(
                        f"ROI {name!r} has CLOSEDPLANAR_XOR contours that "
                        f"cross on slice {plane} of grid {grid.grid_id!r}: "
                        "their exclusive-or is not a set of nested outlines, "
                        "which is all a medh5 contour can hold (§8.6), and "
                        "read as nested they would be their union"
                    )
        out.extend((contours[p], index[p], plane) for p in positions)
    return out


def _slice(index: npt.NDArray[np.float64], grid: Any, name: str | None) -> int:
    """The slice of *grid*, along its first axis, that a contour lies in.

    Within a tenth of a slice, and at least 0.01 mm, as the slices of a DICOM
    series are held to their stack (:mod:`medh5.io.dicom`).  The plane was
    the median vertex's, rounded: a contour between two slices went onto one
    of them, one across several (drawn on another orientation, or not
    planar) onto the middle one, and one past the grid's last slice was
    stored there and rasterised to nothing.
    """
    from medh5.io.dicom import _position_tolerance

    shape = grid.spatial_shape
    step = float(grid.spacing[grid.spatial_axes[0]]) * MM_PER_UNIT.get(grid.units, 1.0)
    tolerance = _position_tolerance(step)
    along = index[:, 0]
    plane = int(round(float(np.median(along))))
    where = f"ROI {name!r} has a contour"
    if float(np.ptp(along)) * step > tolerance:
        raise MEDH5ValidationError(
            f"{where} that runs from slice {float(along.min()):.4g} to "
            f"{float(along.max()):.4g} of grid {grid.grid_id!r}: it does not lie "
            "in one of the grid's slices, as a contour on it must (§8.6) --- it "
            "was drawn on another orientation, or is not planar --- and it is "
            "not projected onto one"
        )
    off = float(np.max(np.abs(along - plane))) * step
    if off > tolerance:
        raise MEDH5ValidationError(
            f"{where} {off:.4g} mm from slice {plane} of grid {grid.grid_id!r}, "
            f"the nearest, beyond the {tolerance:.4g} mm tolerance (a tenth of a "
            "slice, and at least 0.01 mm): it was drawn between the grid's "
            "slices, and is not moved onto one"
        )
    rows, columns = index[:, 1], index[:, 2]
    outside = (
        not 0 <= plane < shape[0]
        or rows.max() < -0.5
        or rows.min() > shape[1] - 0.5
        or columns.max() < -0.5
        or columns.min() > shape[2] - 0.5
    )
    if outside:
        raise MEDH5ValidationError(
            f"{where} on slice {plane} that lies outside grid {grid.grid_id!r} "
            f"({shape[0]} slices of {shape[1]}×{shape[2]}): it outlines nothing "
            "of the grid's image"
        )
    return plane


def _crosses(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> bool:
    """Whether the outlines of two closed polygons cross or touch, in-plane.

    An edge-against-edge test: a vertex-in-polygon one misses two bars
    crossing as a plus, which have no vertex inside each other.  Touching
    counts, since a vertex on the other's outline is neither inside nor out.
    """
    if np.any(
        np.maximum(first.min(0), second.min(0))
        > np.minimum(first.max(0), second.max(0))
    ):
        return False
    extent = float(np.ptp(np.concatenate([first, second]), axis=0).max()) + 1.0
    tolerance, slack = 1e-9 * extent * extent, 1e-9 * extent
    q0, q1 = second, np.roll(second, -1, axis=0)
    ends = np.roll(first, -1, axis=0)
    for start in range(0, len(first), 256):
        p0 = first[start : start + 256, None, :]
        p1 = ends[start : start + 256, None, :]
        d1 = _side(q0[None], q1[None], p0, tolerance)
        d2 = _side(q0[None], q1[None], p1, tolerance)
        d3 = _side(p0, p1, q0[None], tolerance)
        d4 = _side(p0, p1, q1[None], tolerance)
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            return True
        touching = (
            ((d1 == 0) & _between(q0[None], q1[None], p0, slack))
            | ((d2 == 0) & _between(q0[None], q1[None], p1, slack))
            | ((d3 == 0) & _between(p0, p1, q0[None], slack))
            | ((d4 == 0) & _between(p0, p1, q1[None], slack))
        )
        if np.any(touching):
            return True
    return False


def _side(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    point: npt.NDArray[np.float64],
    tolerance: float,
) -> npt.NDArray[np.int8]:
    """Which side of the line *a*→*b* each *point* is on: ±1, or 0 on it."""
    cross = (b[..., 0] - a[..., 0]) * (point[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (point[..., 0] - a[..., 0])
    side: npt.NDArray[np.int8] = np.where(
        np.abs(cross) <= tolerance, 0, np.sign(cross)
    ).astype(np.int8)
    return side


def _between(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    point: npt.NDArray[np.float64],
    slack: float,
) -> npt.NDArray[np.bool_]:
    """Whether each *point*, on the line *a*→*b*, lies within the segment."""
    lo, hi = np.minimum(a, b) - slack, np.maximum(a, b) + slack
    within: npt.NDArray[np.bool_] = np.all((point >= lo) & (point <= hi), axis=-1)
    return within


def _inside(
    polygon: npt.NDArray[np.float64], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Even-odd point-in-polygon test, vectorised over *points*."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        straddles = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, np.nan, y1 - y0)
        hit = straddles & (x < crossing)
        inside ^= np.nan_to_num(hit, nan=0.0).astype(bool)
    return inside


def _rasterize(
    polygons: Sequence[Any], grid: Any, log: ConversionReport
) -> dict[int, npt.NDArray[np.bool_]]:
    """Fill contours at voxel centres by the even-odd rule, per plane (§8.6).

    A voxel is inside when an odd number of nesting levels contain it: a hole
    subtracts from the region around it, and an island inside the hole adds
    back.  Contours at one depth are unioned first, so two outlines that
    overlap without nesting are one region rather than cancelling where they
    overlap.  Unioning every outer and subtracting every hole erased an island
    inside a hole.

    Voxel centres, not areas: a partial-coverage rule would need a threshold,
    and any threshold chosen here would be an unrecorded decision applied to
    every dataset the tool ever converts.
    """
    shape = grid.spatial_shape
    rows, columns = np.meshgrid(
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    centres = np.stack([rows.ravel(), columns.ravel()], axis=1)

    # Contours are collected per (class, plane) and combined only once that
    # plane is complete.  DICOM does not require an outer contour to precede
    # the hole it encloses, and combining in file order lost the cavity.
    seen: set[int] = set()
    planes: dict[tuple[int, int], list[npt.NDArray[np.float64]]] = {}
    for polygon in polygons:
        index = grid.world_to_index(np.asarray(polygon.vertices, dtype=np.float64))
        plane = int(round(float(np.median(index[:, 0]))))
        if not 0 <= plane < shape[0]:
            continue
        class_id = int(polygon.class_id)
        seen.add(class_id)
        planes.setdefault((class_id, plane), []).append(index)

    # A class contributing only holes still gets its (empty) mask: "examined and
    # absent" and "never looked at" are different facts (§11.3).
    masks: dict[int, npt.NDArray[np.bool_]] = {
        class_id: np.zeros(shape, dtype=bool) for class_id in sorted(seen)
    }
    for (class_id, plane), contours in planes.items():
        levels: dict[int, npt.NDArray[np.bool_]] = {}
        depths = _core.polygon_depths([c[:, 1:] for c in contours])
        for depth, index in zip(depths, contours, strict=True):
            filled = _inside(index[:, 1:], centres).reshape(shape[1], shape[2])
            levels[depth] = filled if depth not in levels else levels[depth] | filled
        region = np.zeros((shape[1], shape[2]), dtype=bool)
        for filled in levels.values():
            region ^= filled
        masks[class_id][plane] |= region
    log.guess(
        "rasterization",
        "a rasterised mask is an approximation of the contours it came from; the "
        "contours remain in the file as the authoritative geometry",
        {"classes": sorted(masks)},
    )
    return masks


def _key(name: str) -> str:
    return sanitize_key(name, fallback="roi")


def _match_grid(
    sample: Any,
    frame: str | None,
    log: ConversionReport,
    salt: str | None = None,
) -> str:
    """The grid sharing the ROIs' frame, else the sample's only grid.

    The only grid is taken whatever its frame; one in another known frame is
    refused afterwards, as a grid named explicitly is.  Used with a recorded
    guess, a grid in another frame received every contour.
    """

    matches = (
        [g for g in sample.grids.values() if frames_agree(frame, g.frame_uid, salt)]
        if frame
        else []
    )
    if len(matches) == 1:
        return str(matches[0].grid_id)
    names: list[str] = sorted(sample.grids)
    if len(names) == 1:
        if frame and not sample.grids[names[0]].frame_uid:
            log.guess(
                "grid",
                f"the RTSTRUCT names frame {frame!r}, which no grid declares; the "
                f"sample's only grid {names[0]!r} was used",
                {"grid": names[0]},
            )
        return names[0]
    raise MEDH5ValidationError(
        f"cannot tell which grid the RTSTRUCT belongs to: frame {frame!r} matches "
        f"{len(matches)} of the sample's {len(names)} grids. Pass grid= explicitly.",
        code="E101",
    )


def to_rtstruct(
    sample: str | os.PathLike[str],
    ann_id: str,
    source_images: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    label: str = "medh5 export",
    frame_salt: str | None = None,
    report: ConversionReport | None = None,
    points: str | None = None,
    curves: str | None = None,
) -> Path:
    """Export a ``contours`` annotation as an RT Structure Set.

    Only ``contours`` is exportable: an RTSTRUCT is polygons, and turning a mask
    into polygons requires a marching-squares rule that would be invented here
    rather than recorded.  Rasterise deliberately in the other direction if you
    need both.

    ``ContourData`` is DICOM patient coordinates --- millimetres, LPS --- so
    the grid's world coordinates are scaled from its units and, from RAS,
    flipped in x and y; a grid in pixels or another convention is refused, and
    so are world-space contours with no grid to say their units.  The source
    images must share one frame of reference, which the contours' --- the
    annotation's ``frame_uid`` in world space, the grid's in index space ---
    must be where one is stated (*frame_salt* compares a frame ``medh5 scrub``
    pseudonymised, as :func:`~medh5.io.dicom_seg.frames_agree` does).
    The contours were written as stored --- a metre grid's 0.003 for 3 mm,
    RAS signs unflipped --- and filed under the first image's frame, whatever
    the grid's.

    Each ROI is named for its class's display name, where every class has its
    own one that an ``ROIName`` can hold, else for its key, and takes the
    label set's colour; text beyond ASCII is written as UTF-8 under
    ``SpecificCharacterSet`` ``ISO_IR 192``.  The key, red, and text in no
    stated character set were written.  Every contour is ``CLOSED_PLANAR``,
    a hole a contour nested inside its outer, which readers that combine an
    ROI's contours by exclusive-or or even-odd fill read as medh5 does; two
    contours of one class that cross on a plane are a warning, because
    medh5 means their union and those readers cancel their overlap.

    *points* names a ``points`` annotation (§8.5) to write beside the
    contours: one ROI per class, of ``POINT`` contours of one point each,
    with ``RTROIInterpretedType`` ``"MARKER"`` (a contours ROI is
    ``"ORGAN"``).  Its rows must each have a class, and a class drawn in both
    annotations is refused: one ROI is areas or points, not both.

    *curves* names a ``polylines`` annotation (format 1.2 §3) to write beside
    them: one ROI per class, a contour per polyline, ``OPEN_PLANAR`` for one
    that records the grid plane it lies in and ``OPEN_NONPLANAR`` for one
    that does not, with an empty ``RTROIInterpretedType`` (a curve's role,
    an applicator's channel or a vessel, is not recorded).  A class it shares
    with the contours or the points is refused, as for points.

    *label* is the ``StructureSetLabel``, a DICOM SH: one longer than 16
    characters is refused before anything is read, where it was written and
    the file was not valid DICOM.
    """
    import medh5
    from medh5.io.dicom import _dcmread, require_pydicom

    pydicom = require_pydicom()
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    if len(label) > _SH_LENGTH:
        raise MEDH5ValidationError(
            f"label={label!r} (--label) is {len(label)} characters, and the "
            f"StructureSetLabel it is written as is a DICOM SH of at most "
            f"{_SH_LENGTH}: shorten it"
        )
    log = report or ConversionReport(converter="to-rtstruct")
    log.source = os.fspath(sample)
    datasets = [_dcmread(p) for p in source_images]
    if not datasets:
        raise MEDH5ValidationError("to_rtstruct needs the source DICOM images")
    reference = datasets[0]
    frame_uid = _one_frame(datasets)

    with medh5.open(sample) as opened:
        annotation, grid, to_patient = _exported(
            opened, ann_id, "contours", frame_uid, frame_salt, log
        )
        polygons = list(annotation.polygons())
        class_ids = sorted({int(p.class_id) for p in polygons})
        marked: list[tuple[int, npt.NDArray[np.float64]]] = []
        marker_ids: list[int] = []
        if points is not None:
            pointed, _, place = _exported(
                opened, points, "points", frame_uid, frame_salt, log
            )
            rows = np.asarray(pointed.points, dtype=np.float64)
            owners = [int(c) for c in pointed.object_class_ids]
            if 0 in owners:
                raise MEDH5ValidationError(
                    f"annotation {points!r} has {owners.count(0)} point(s) of no "
                    "class, and an RTSTRUCT ROI is a class: give its rows "
                    "class_ids to export it",
                    code="E402",
                )
            marker_ids = sorted(set(owners))
            both = sorted(set(marker_ids) & set(class_ids))
            if both:
                raise MEDH5ValidationError(
                    f"class(es) {[annotation.class_key(c) for c in both]} are in "
                    f"both {ann_id!r} and {points!r}: an RTSTRUCT ROI is drawn "
                    "as areas or as points, not both, so one class cannot be "
                    "written as one ROI"
                )
            marked = list(zip(owners, place(rows), strict=True))
        lined: list[tuple[int, str, npt.NDArray[np.float64]]] = []
        curve_ids: list[int] = []
        if curves is not None:
            curved, _, place = _exported(
                opened, curves, "polylines", frame_uid, frame_salt, log
            )
            lines = list(curved.polylines())
            owners = [int(line.class_id) for line in lines]
            if 0 in owners:
                raise MEDH5ValidationError(
                    f"annotation {curves!r} has {owners.count(0)} polyline(s) of "
                    "no class, and an RTSTRUCT ROI is a class: give them "
                    "class ids to export them",
                    code="E402",
                )
            curve_ids = sorted(set(owners))
            both = sorted(set(curve_ids) & {*class_ids, *marker_ids})
            if both:
                raise MEDH5ValidationError(
                    f"class(es) {[annotation.class_key(c) for c in both]} are in "
                    f"{curves!r} and in {ann_id!r} or {points!r}: an RTSTRUCT "
                    "ROI is drawn as areas, points or curves, one of them, so "
                    "one class cannot be written as one ROI"
                )
            lined = [
                (
                    int(line.class_id),
                    _OPEN_PLANAR if int(line.plane[0]) >= 0 else _OPEN_NONPLANAR,
                    place(np.asarray(line.vertices, dtype=np.float64)),
                )
                for line in lines
            ]
        exported = [*class_ids, *marker_ids, *curve_ids]
        names = _roi_names(annotation, exported, log)
        found = annotation.label_set
        colors = {
            c: found[c].color if found is not None and c in found else None
            for c in exported
        }
        for class_id, planes in _crossings(polygons, grid, annotation.space).items():
            key = annotation.class_key(class_id)
            shown = ", ".join(map(str, planes[:8])) + (", ..." if planes[8:] else "")
            log.warn(
                "crossing_contours",
                f"contours of class {key!r} cross on {len(planes)} plane(s) of "
                f"grid {grid.grid_id!r}, as (axis, index): {shown}; medh5 means "
                "their union, and a reader that combines an ROI's contours by "
                "exclusive-or or even-odd fill cancels their overlap --- merge "
                "them into one outline where such a reader matters",
                {"class": key, "planes": [list(plane) for plane in planes]},
            )
        world = [
            (int(p.class_id), to_patient(np.asarray(p.vertices, dtype=np.float64)))
            for p in polygons
        ]

    structure = Dataset()
    structure.file_meta = FileMetaDataset()
    structure.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    structure.SOPClassUID = pydicom.uid.RTStructureSetStorage
    structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
    structure.Modality = "RTSTRUCT"
    structure.StructureSetLabel = label
    structure.StructureSetDate = getattr(reference, "StudyDate", "")
    structure.StructureSetTime = "000000"
    for tag in (
        "PatientID",
        "PatientName",
        "PatientBirthDate",
        "PatientSex",
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "StudyID",
        "AccessionNumber",
    ):
        setattr(structure, tag, getattr(reference, tag, ""))
    structure.SeriesInstanceUID = generate_uid()
    structure.SeriesNumber = 1
    structure.Manufacturer = "medh5"

    frame = Dataset()
    frame.FrameOfReferenceUID = frame_uid
    structure.ReferencedFrameOfReferenceSequence = [frame]

    structure.StructureSetROISequence = []
    structure.ROIContourSequence = []
    structure.RTROIObservationsSequence = []
    drawn: list[tuple[int, str, list[tuple[str, npt.NDArray[np.float64]]]]] = (
        [
            (c, "ORGAN", [(_CLOSED, p) for owner, p in world if owner == c])
            for c in class_ids
        ]
        + [
            (c, "MARKER", [(_POINT, p) for owner, p in marked if owner == c])
            for c in marker_ids
        ]
        + [
            (c, "", [(kind, p) for owner, kind, p in lined if owner == c])
            for c in curve_ids
        ]
    )
    for number, (class_id, interpreted, shapes) in enumerate(drawn, start=1):
        roi = Dataset()
        roi.ROINumber = number
        roi.ReferencedFrameOfReferenceUID = frame_uid
        roi.ROIName = names[class_id]
        roi.ROIGenerationAlgorithm = "MANUAL"
        structure.StructureSetROISequence.append(roi)

        contour_item = Dataset()
        contour_item.ReferencedROINumber = number
        color = colors[class_id]
        contour_item.ROIDisplayColor = list(color[:3]) if color else [255, 0, 0]
        contour_item.ContourSequence = []
        for kind, shape in shapes:
            points_of = np.asarray(shape, dtype=np.float64).reshape(-1, 3)
            entry = Dataset()
            entry.ContourGeometricType = kind
            entry.NumberOfContourPoints = int(points_of.shape[0])
            entry.ContourData = [float(v) for v in points_of.reshape(-1)]
            contour_item.ContourSequence.append(entry)
        structure.ROIContourSequence.append(contour_item)

        observation = Dataset()
        observation.ObservationNumber = number
        observation.ReferencedROINumber = number
        observation.RTROIInterpretedType = interpreted
        observation.ROIInterpreter = ""
        structure.RTROIObservationsSequence.append(observation)

    if not all(
        str(element.value).isascii()
        for element in structure.iterall()
        if element.VR in _TEXT
    ):
        # Without it, pydicom writes Latin-1 under the default repertoire,
        # which is ASCII: a name with an umlaut read back as other letters.
        structure.SpecificCharacterSet = "ISO_IR 192"
    target = Path(os.fspath(out))
    save_dicom(structure, target, enforce_file_format=True)
    log.decision(
        "contours",
        f"{len(world)} contour(s) across {len(class_ids)} ROI(s) were written in "
        "patient coordinates",
        {"rois": [names[c] for c in class_ids]},
    )
    if marker_ids:
        log.decision(
            "points",
            f"{len(marked)} point(s) of {points!r} were written as POINT "
            f"contours of {len(marker_ids)} ROI(s), interpreted as MARKER",
            {"rois": [names[c] for c in marker_ids]},
        )
    if curve_ids:
        planar = sum(1 for _, kind, _ in lined if kind == _OPEN_PLANAR)
        log.decision(
            "curves",
            f"{len(lined)} polyline(s) of {curves!r} were written as open "
            f"contours of {len(curve_ids)} ROI(s), {planar} OPEN_PLANAR (with a "
            "plane) and the rest OPEN_NONPLANAR, with no RTROIInterpretedType",
            {
                "rois": [names[c] for c in curve_ids],
                "planar": planar,
                "nonplanar": len(lined) - planar,
            },
        )
    log.outputs.append(str(target))
    return target


def _exported(
    opened: Any,
    ann_id: str,
    kind: str,
    frame_uid: str,
    frame_salt: str | None,
    log: ConversionReport,
) -> tuple[Any, Any, Any]:
    """Annotation *ann_id* of *kind*, its grid, and the map of its stored
    coordinates to patient ones (mm, LPS), once its grid and frame are
    found to allow it."""
    annotation = opened.annotations[ann_id]
    if annotation.kind != kind:
        what = {
            "contours": "ROIs are polygons, so export a `contours` annotation "
            "(or derive one)",
            "points": "POINT ROIs are markers, so pass a `points` annotation "
            "as points=",
            "polylines": "open ROIs are curves, so pass a `polylines` "
            "annotation as curves=",
        }[kind]
        raise MEDH5ValidationError(
            f"annotation {ann_id!r} is {annotation.kind!r}; an RTSTRUCT's {what}",
            code="E401",
        )
    if annotation.grid_id is None:
        raise MEDH5ValidationError(
            f"annotation {ann_id!r} names no grid, so its coordinates have no "
            "units or convention to map to patient coordinates (mm, LPS)",
            code="E414",
        )
    grid = opened.grids[annotation.grid_id]
    if len(grid.spatial_shape) != 3:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} has {len(grid.spatial_shape)} spatial axes; "
            "RTSTRUCT contours are points of the 3-D patient space",
            code="E414",
        )
    signs, scale = patient_mapping(grid, "RTSTRUCT contours")
    # World coordinates live in the annotation's own frame, index
    # coordinates in the grid's (§6.2, §8).
    stated = (
        annotation.header.frame_uid if annotation.space == "world" else grid.frame_uid
    )
    agree = frames_agree(frame_uid, stated, frame_salt)
    if agree is False:
        hint = _salt_hint(stated, frame_salt, "The annotation's", "the images' frame")
        raise MEDH5ValidationError(
            f"annotation {ann_id!r} is in frame of reference {stated!r} and "
            f"the source images in {frame_uid!r}: contours drawn in one frame "
            "are not filed under another without a registration, which this "
            f"exporter does not apply.{hint}",
            code="E414",
        )
    if agree is None:
        log.guess(
            "frame_of_reference",
            f"annotation {ann_id!r} and grid {grid.grid_id!r} state no frame "
            "of reference for the contours, so they were assumed to be in the "
            f"source images' frame {frame_uid!r}",
            {"grid": grid.grid_id, "frame": frame_uid},
        )
    world = annotation.space == "world"

    def to_patient(coordinates: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        found = coordinates if world else grid.index_to_world(coordinates)
        mapped: npt.NDArray[np.float64] = signs * scale * found
        return mapped

    return annotation, grid, to_patient


# The value representations that hold text in a character set (PS3.5 6.1.2).
_TEXT = frozenset({"SH", "LO", "ST", "LT", "UC", "UT", "PN"})


def _roi_names(
    annotation: Any, class_ids: Sequence[int], log: ConversionReport
) -> dict[int, str]:
    """Each class's ``ROIName``: its display name, where every class exported
    has its own one that an ``ROIName`` holds (``LO``: up to 64 characters, no
    backslash), else its key --- every class's, so that no ROI is named for
    another class's key."""
    keys = {c: annotation.class_key(c) for c in class_ids}
    found = annotation.label_set
    shown = {
        c: found[c].name if found is not None and c in found else "" for c in class_ids
    }
    fits = all(
        0 < len(name) <= 64 and "\\" not in name and name.isprintable()
        for name in shown.values()
    )
    if fits and len(set(shown.values())) == len(shown):
        return shown
    log.decision(
        "roi_names",
        "the classes' display names are "
        + ("not distinct" if fits else "not all names an ROIName holds")
        + ", so each ROI is named for its class key",
        {"names": {keys[c]: shown[c] for c in class_ids}},
    )
    return keys


def _crossings(
    polygons: Sequence[Any], grid: Any, space: str
) -> dict[int, list[tuple[int, int]]]:
    """Per class, the ``(axis, index)`` planes on which two of its contours
    cross or touch, compared in the grid's index space."""
    outlines: dict[tuple[int, tuple[int, int]], list[npt.NDArray[np.float64]]] = {}
    for polygon in polygons:
        vertices = np.asarray(polygon.vertices, dtype=np.float64)
        index = vertices if space == "index" else grid.world_to_index(vertices)
        axis, at = (int(v) for v in polygon.plane)
        if axis < 0:
            # No plane recorded: the axis it is constant along, if any.
            spread = np.ptp(index, axis=0)
            axis = int(np.argmin(spread))
            if spread[axis] > 1e-3:
                continue
            at = int(round(float(np.median(index[:, axis]))))
        key = (int(polygon.class_id), (axis, at))
        outlines.setdefault(key, []).append(np.delete(index, axis, axis=1))
    found: dict[int, list[tuple[int, int]]] = {}
    for (class_id, plane), drawn in sorted(outlines.items()):
        if any(_crosses(a, b) for a, b in combinations(drawn, 2)):
            found.setdefault(class_id, []).append(plane)
    return found


__all__ = [
    "MIN_POLYGON_VERTICES",
    "from_rtstruct",
    "read_rtstruct",
    "to_rtstruct",
]
