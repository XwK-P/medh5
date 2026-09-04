"""RT Structure Set import and export (spec §8.6).

An RTSTRUCT is a set of **planar polygons in patient coordinates**, one per ROI
per slice.  It is not a mask, and the difference matters: a contour is exact at
sub-voxel resolution and a mask is not, so converting one to the other loses
information in one direction and invents it in the other.

This converter therefore stores contours as contours (§8.6, ``space="world"``),
which round-trips exactly.  Rasterising into a voxel annotation is available and
**opt-in**, because the rasterisation rule — winding, holes, whether a boundary
voxel counts — is a decision that belongs in the provenance record rather than
in a library's defaults.  When it is used, the rule is written down and the
contours stay in the file beside the mask.

Holes are read the way planners write them: a second contour on the same slice
that lies inside another is stored with ``role="hole"``, and the rasteriser
excludes it.  Treating every contour as an outer boundary is how a segmented
vessel lumen ends up filled in.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.io._common import sanitize_key
from medh5.io.report import ConversionReport

MIN_POLYGON_VERTICES = 3


def read_rtstruct(
    path: str | os.PathLike[str],
) -> tuple[dict[int, list[npt.NDArray[np.float64]]], dict[str, Any]]:
    """An RTSTRUCT as ``{roi number: [contour, ...]}`` in patient coordinates."""
    from medh5.io.dicom import require_pydicom

    pydicom = require_pydicom()
    dataset = pydicom.dcmread(os.fspath(path))
    if getattr(dataset, "Modality", None) != "RTSTRUCT":
        raise MEDH5ValidationError(
            f"{path} declares Modality="
            f"{getattr(dataset, 'Modality', None)!r}, not 'RTSTRUCT'"
        )
    names = {
        int(roi.ROINumber): str(getattr(roi, "ROIName", f"roi_{roi.ROINumber}"))
        for roi in getattr(dataset, "StructureSetROISequence", [])
    }
    colors: dict[int, tuple[int, int, int, int]] = {}
    contours: dict[int, list[npt.NDArray[np.float64]]] = {}
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
            if points.shape[0] >= MIN_POLYGON_VERTICES:
                contours.setdefault(number, []).append(points)
    return contours, {
        "names": names,
        "colors": colors,
        "frame_uid": _referenced_frame(dataset),
        "label": str(getattr(dataset, "StructureSetLabel", "")),
    }


def _referenced_frame(dataset: Any) -> str | None:
    for item in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        uid = getattr(item, "FrameOfReferenceUID", None)
        if uid:
            return str(uid)
    return None


def from_rtstruct(
    path: str | os.PathLike[str],
    sample: str | os.PathLike[str],
    *,
    ann_id: str = "contours",
    grid: str | None = None,
    rasterize: bool = False,
    mask_id: str | None = None,
    annotated_classes: Sequence[str] | str = "all_given",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Import an RTSTRUCT's ROIs as contours, and optionally rasterise them."""
    import medh5
    from medh5.annotations.geometric import Polygon
    from medh5.labels.labelset import LabelClass, LabelSet

    log = report or ConversionReport(converter="from-rtstruct")
    log.source = os.fspath(path)
    contours, meta = read_rtstruct(path)
    if not contours:
        raise MEDH5ValidationError(f"{path} contains no usable contours")

    with medh5.open(sample) as opened:
        grid_id = grid or _match_grid(opened, meta, log)
        target = opened.grids[grid_id]
        existing = opened.label_set

    label_set = existing
    ids: dict[int, int] = {}
    if label_set is None:
        classes = [
            LabelClass(
                i + 1,
                _key(meta["names"].get(number, f"roi_{number}")),
                meta["names"].get(number, f"roi_{number}"),
                color=meta["colors"].get(number),
            )
            for i, number in enumerate(sorted(contours))
        ]
        label_set = LabelSet("rtstruct", version="1.0.0", classes=classes)
        ids = {number: i + 1 for i, number in enumerate(sorted(contours))}
        log.decision(
            "label_set",
            f"{len(classes)} class(es) were minted from ROIName, keeping each "
            "ROI's display colour",
            {"names": {str(k): meta["names"].get(k) for k in sorted(contours)}},
        )
    else:
        missing = []
        for number in sorted(contours):
            name = meta["names"].get(number, f"roi_{number}")
            found = label_set.get(_key(name)) or label_set.get(name)
            if found is None:
                missing.append(name)
            else:
                ids[number] = found.id
        if missing:
            raise MEDH5ValidationError(
                f"the sample's label set has no class for ROI(s) {missing}",
                code="E402",
            )

    polygons: list[Polygon] = []
    for number in sorted(contours):
        planar = _assign_roles(contours[number], target, log, meta["names"].get(number))
        for points, role, plane in planar:
            polygons.append(
                Polygon(
                    vertices=points.astype(np.float32),
                    class_id=ids[number],
                    # `(0, k)`: the polygon lies in slice k along the grid's
                    # first spatial axis, in that grid's index space (§8.6).
                    # It was computed to find holes and then thrown away, so
                    # `by_plane()` on an imported RTSTRUCT answered nothing.
                    plane=(0, plane),
                    role=role,
                )
            )
    log.decision(
        "contours",
        f"{len(polygons)} contour(s) were stored in world coordinates, exactly "
        "as the planner drew them; a contour is sub-voxel and a mask is not",
        {"polygons": len(polygons), "rois": len(contours)},
    )

    masks: dict[int, npt.NDArray[np.bool_]] | None = None
    if rasterize:
        masks = _rasterize(polygons, target, log)

    with medh5.amend(sample) as writer:
        if existing is None:
            writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import", agent=tool, tool="medh5 convert from-rtstruct"
        )
        writer.add_contours(
            ann_id,
            polygons,
            grid=grid_id,
            space="world",
            frame_uid=meta["frame_uid"],
            annotated_classes=annotated_classes,
            prov=activity,
        )
        if masks:
            raster = writer.activity(
                "derive",
                agent=tool,
                tool="medh5 convert from-rtstruct --rasterize",
                inputs=[f"annotations/{ann_id}"],
                params={
                    "rule": "even-odd fill at voxel centres, holes excluded",
                    "sampling": "voxel centre",
                },
            )
            kind, _ = writer.add_segmentation(
                mask_id or f"{ann_id}_mask",
                grid=grid_id,
                masks=masks,
                annotated_classes=annotated_classes,
                prov=raster,
                derived_from=[ann_id],
            )
            log.decision(
                "rasterization",
                "contours were rasterised by even-odd fill at voxel centres with "
                f"holes excluded, stored as {kind!r}; the contours were kept, so "
                "the lossy step is reversible by re-deriving it",
                {"kind": kind, "rule": "even-odd at voxel centres"},
            )
    log.outputs.append(os.fspath(sample))
    return log


def _assign_roles(
    contours: Sequence[npt.NDArray[np.float64]],
    grid: Any,
    log: ConversionReport,
    name: str | None,
) -> list[tuple[npt.NDArray[np.float64], str, int]]:
    """Mark contours enclosed by another on the same slice as holes.

    The grouping and the enclosure test are both done in the grid's **index**
    space, where a planar contour has one constant axis.  Doing it on world
    coordinates would only work for axial acquisitions: under the orientation
    a DICOM series actually declares, the world *z* of a contour varies within
    its own slice, and grouping on it puts every vertex in its own plane.

    Returns ``(vertices, role, plane)`` with the plane index that grouping
    found, so the writer can record it rather than lose it.
    """
    out: list[tuple[npt.NDArray[np.float64], str, int]] = []
    index = [grid.world_to_index(points) for points in contours]
    by_plane: dict[int, list[int]] = {}
    for position, points in enumerate(index):
        by_plane.setdefault(int(round(float(np.median(points[:, 0])))), []).append(
            position
        )
    holes = 0
    for plane, positions in by_plane.items():
        for position in positions:
            role = "outer"
            for other in positions:
                if other != position and _encloses(index[other], index[position]):
                    role = "hole"
                    holes += 1
                    break
            out.append((contours[position], role, plane))
    if holes:
        log.decision(
            "holes",
            f"{holes} contour(s) of ROI {name!r} lie inside another on the same "
            "slice and were marked as holes rather than as separate regions",
            {"holes": holes, "roi": name},
        )
    return out


def _encloses(outer: npt.NDArray[np.float64], inner: npt.NDArray[np.float64]) -> bool:
    """Whether every vertex of *inner* lies inside *outer*, in-plane."""
    return bool(np.all(_inside(outer[:, 1:], inner[:, 1:])))


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
    """Fill contours at voxel centres, subtracting holes (§8.6).

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

    # Outers and holes are collected per (class, plane) and combined only once
    # that plane is complete.  DICOM does not require an outer contour to
    # precede the hole it encloses, and subtracting a hole from a mask whose
    # outer has not been drawn yet loses the cavity: the outer then fills it
    # back in and the conversion silently turns holes into foreground.
    seen: set[int] = set()
    outers: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
    holes: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
    for polygon in polygons:
        index = grid.world_to_index(np.asarray(polygon.vertices, dtype=np.float64))
        plane = int(round(float(np.median(index[:, 0]))))
        if not 0 <= plane < shape[0]:
            continue
        class_id = int(polygon.class_id)
        seen.add(class_id)
        filled = _inside(index[:, 1:], centres).reshape(shape[1], shape[2])
        into = holes if polygon.role == "hole" else outers
        key = (class_id, plane)
        into[key] = filled if key not in into else (into[key] | filled)

    # A class contributing only holes still gets its (empty) mask: "examined and
    # absent" and "never looked at" are different facts (§6.4).
    masks: dict[int, npt.NDArray[np.bool_]] = {
        class_id: np.zeros(shape, dtype=bool) for class_id in sorted(seen)
    }
    for (class_id, plane), filled in outers.items():
        hole = holes.get((class_id, plane))
        masks[class_id][plane] |= filled if hole is None else filled & ~hole
    log.guess(
        "rasterization",
        "a rasterised mask is an approximation of the contours it came from; the "
        "contours remain in the file as the authoritative geometry",
        {"classes": sorted(masks)},
    )
    return masks


def _key(name: str) -> str:
    return sanitize_key(name, fallback="roi")


def _match_grid(sample: Any, meta: Mapping[str, Any], log: ConversionReport) -> str:
    frame = meta.get("frame_uid")
    if frame:
        matches = [g for g in sample.grids.values() if g.frame_uid == frame]
        if len(matches) == 1:
            return str(matches[0].grid_id)
    names: list[str] = sorted(sample.grids)
    if len(names) == 1:
        if frame:
            log.guess(
                "grid",
                f"the RTSTRUCT names frame {frame!r}, which no grid declares; the "
                f"sample's only grid {names[0]!r} was used",
                {"grid": names[0]},
            )
        return names[0]
    raise MEDH5ValidationError(
        f"cannot tell which grid the RTSTRUCT belongs to: frame {frame!r} matches "
        f"no grid and the sample has {len(names)} grids. Pass grid= explicitly.",
        code="E101",
    )


def to_rtstruct(
    sample: str | os.PathLike[str],
    ann_id: str,
    source_images: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    label: str = "medh5 export",
    report: ConversionReport | None = None,
) -> Path:
    """Export a ``contours`` annotation as an RT Structure Set.

    Only ``contours`` is exportable: an RTSTRUCT is polygons, and turning a mask
    into polygons requires a marching-squares rule that would be invented here
    rather than recorded.  Rasterise deliberately in the other direction if you
    need both.
    """
    import medh5
    from medh5.io.dicom import require_pydicom

    pydicom = require_pydicom()
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    log = report or ConversionReport(converter="to-rtstruct")
    log.source = os.fspath(sample)
    datasets = [pydicom.dcmread(os.fspath(p)) for p in source_images]
    if not datasets:
        raise MEDH5ValidationError("to_rtstruct needs the source DICOM images")
    reference = datasets[0]

    with medh5.open(sample) as opened:
        annotation = opened.annotations[ann_id]
        if annotation.kind != "contours":
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} is {annotation.kind!r}; an RTSTRUCT is "
                "polygons, so export a `contours` annotation (or derive one)",
                code="E401",
            )
        if annotation.space != "world" and annotation.grid_id is None:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} stores index coordinates but names no "
                "grid, so they cannot be mapped to patient coordinates",
                code="E414",
            )
        grid = opened.grids[annotation.grid_id] if annotation.grid_id else None
        polygons = list(annotation.polygons())
        class_ids = sorted({int(p.class_id) for p in polygons})
        names = {c: annotation.class_key(c) for c in class_ids}
        world = [
            (
                int(p.class_id),
                p.vertices
                if annotation.space == "world" or grid is None
                else grid.index_to_world(np.asarray(p.vertices, dtype=np.float64)),
            )
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

    frame_uid = str(getattr(reference, "FrameOfReferenceUID", generate_uid()))
    frame = Dataset()
    frame.FrameOfReferenceUID = frame_uid
    structure.ReferencedFrameOfReferenceSequence = [frame]

    structure.StructureSetROISequence = []
    structure.ROIContourSequence = []
    structure.RTROIObservationsSequence = []
    for number, class_id in enumerate(class_ids, start=1):
        roi = Dataset()
        roi.ROINumber = number
        roi.ReferencedFrameOfReferenceUID = frame_uid
        roi.ROIName = names[class_id]
        roi.ROIGenerationAlgorithm = "MANUAL"
        structure.StructureSetROISequence.append(roi)

        contour_item = Dataset()
        contour_item.ReferencedROINumber = number
        contour_item.ROIDisplayColor = [255, 0, 0]
        contour_item.ContourSequence = []
        for owner, points in world:
            if owner != class_id:
                continue
            entry = Dataset()
            entry.ContourGeometricType = "CLOSED_PLANAR"
            entry.NumberOfContourPoints = int(np.asarray(points).shape[0])
            entry.ContourData = [
                float(v) for v in np.asarray(points, dtype=np.float64).reshape(-1)
            ]
            contour_item.ContourSequence.append(entry)
        structure.ROIContourSequence.append(contour_item)

        observation = Dataset()
        observation.ObservationNumber = number
        observation.ReferencedROINumber = number
        observation.RTROIInterpretedType = "ORGAN"
        observation.ROIInterpreter = ""
        structure.RTROIObservationsSequence.append(observation)

    target = Path(os.fspath(out))
    structure.save_as(str(target), enforce_file_format=True)
    log.decision(
        "contours",
        f"{len(world)} contour(s) across {len(class_ids)} ROI(s) were written in "
        "patient coordinates",
        {"rois": [names[c] for c in class_ids]},
    )
    log.outputs.append(str(target))
    return target


__all__ = [
    "MIN_POLYGON_VERTICES",
    "from_rtstruct",
    "read_rtstruct",
    "to_rtstruct",
]
