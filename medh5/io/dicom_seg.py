"""DICOM Segmentation import and export (spec §7, §5).

A DICOM SEG stores one binary frame per (segment, source slice), each carrying
its own position, and the frames need not be complete or ordered.  Reading one
therefore means *placing* frames into a volume by their geometry rather than
reshaping them by their order — the same rule as §3.3 everywhere else.

Two things are preserved that a naive conversion loses:

* **Segment identity.**  ``SegmentLabel`` becomes the label-set ``key``, and the
  coded concept (``SegmentedPropertyTypeCodeSequence``) becomes an ontology
  binding when it is present.  Nothing is invented: a SEG with no coded concept
  yields a class with no binding, and W912 says so, which is better than a wrong
  SNOMED code that no validator can detect.
* **Overlap.**  ``SegmentationType`` ``FRACTIONAL`` and overlapping segments are
  both representable — the first as ``probmap``, the second by whichever §7
  encoding the overlap graph calls for. A converter that flattened them into one
  labelmap would silently drop the tumour inside the organ.

Writing needs `highdicom`, which is the reference implementation of the
Segmentation IOD; hand-rolling the per-frame functional groups is how invalid
SEGs get published.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._optional import require
from medh5.errors import MEDH5ValidationError
from medh5.io._common import sanitize_key
from medh5.io.report import ConversionReport

BINARY = "BINARY"
FRACTIONAL = "FRACTIONAL"


def require_highdicom() -> Any:
    return require("highdicom", extra="dicomseg", purpose="writing DICOM SEG")


SLICE_TOLERANCE = 0.25
"""How far off a slice centre a frame may sit and still be that slice.

A quarter of a slice: comfortably inside the rounding and floating-point error
of an ``ImagePositionPatient``, and far enough from half a slice that a frame
lying *between* two slices is refused rather than assigned to one.
"""


def read_dicom_seg_frames(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every frame of a SEG, each with the segment and the position it carries.

    The lower-level half of :func:`read_dicom_seg`, and what an import wants.
    A SEG stores one frame per (segment, source slice) and **need not store
    the empty ones**: ``omit_empty_frames=True`` is highdicom's default and
    the common form in the wild, so a file segmenting two slices of an
    eight-slice series has two frames --- and a volume assembled from the
    frames present is a two-slice volume whose through-plane spacing is the
    distance between those two slices.  That is not a smaller reconstruction
    of the study, it is the study with its gaps closed up, and comparing its
    shape to the grid refused an ordinary file for a reason that was not true.
    Frames carry their own ``ImagePositionPatient``, so :func:`place_frames`
    puts each one where it belongs instead.
    """
    from medh5.io.dicom import require_pydicom

    pydicom = require_pydicom()
    dataset = pydicom.dcmread(os.fspath(path))
    if getattr(dataset, "Modality", None) != "SEG":
        raise MEDH5ValidationError(
            f"{path} declares Modality={getattr(dataset, 'Modality', None)!r}, "
            "not 'SEG'"
        )
    segments = _segments(dataset)
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[None]
    rows, columns = int(dataset.Rows), int(dataset.Columns)
    placement = _frame_placement(dataset, pixels.shape[0])
    fractional = getattr(dataset, "SegmentationType", BINARY) == FRACTIONAL
    scale = float(getattr(dataset, "MaximumFractionalValue", 255) or 255)

    frames = [
        {
            "index": index,
            "segment": placement["segments"][index],
            "position": placement["positions"][index],
            "plane": placement["indices"][index],
            "data": (
                pixels[index].astype(np.float32) / scale
                if fractional
                else pixels[index].astype(bool)
            ),
        }
        for index in range(pixels.shape[0])
    ]
    geometry: dict[str, Any] = {
        "shape": (len(placement["planes"]), rows, columns),
        "rows": rows,
        "columns": columns,
        "spacing": placement["spacing"],
        "origin": placement["origin"],
        "direction": placement["direction"],
        "coord_system": "LPS",
        "units": "mm",
        "frame_uid": _first(dataset, "FrameOfReferenceUID"),
        "segments": segments,
        "fractional": fractional,
        "source_series": placement["source_series"],
        "planes": placement["planes"],
        "frames": len(frames),
        "scale": scale,
    }
    return frames, geometry


def place_frames(
    frames: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    grid: Any,
) -> dict[int, npt.NDArray[Any]]:
    """Put each frame on the slice of *grid* that its position names (§3.3).

    Shape agreement becomes a question about rows and columns, which a SEG and
    its source series must share, and never about depth, which a SEG with
    omitted frames does not have.  A frame whose position is not a slice of
    this grid is refused **by name** rather than force-fitted: it is either a
    different reconstruction or a different series, and both are answers the
    caller has to act on.
    """
    shape = tuple(grid.spatial_shape)
    if len(shape) != 3:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} has {len(shape)} spatial axes; a DICOM SEG "
            "is placed into a 3-D grid",
            code="E405",
        )
    if shape[1:] != (int(geometry["rows"]), int(geometry["columns"])):
        raise MEDH5ValidationError(
            f"the SEG's frames are {geometry['rows']}x{geometry['columns']} but "
            f"grid {grid.grid_id!r} is {shape[1]}x{shape[2]} in plane; it was "
            "drawn on a different reconstruction",
            code="E405",
        )
    dtype = np.float32 if geometry["fractional"] else bool
    volumes: dict[int, npt.NDArray[Any]] = {
        number: np.zeros(shape, dtype=dtype) for number in geometry["segments"]
    }
    for frame in frames:
        segment = int(frame["segment"])
        if segment not in volumes:
            continue
        index = np.asarray(grid.world_to_index(frame["position"]), dtype=np.float64)
        position = float(index.reshape(-1)[0])
        nearest = int(np.round(position))
        if abs(position - nearest) > SLICE_TOLERANCE or not 0 <= nearest < shape[0]:
            raise MEDH5ValidationError(
                f"SEG frame {frame['index']} (segment {segment}) sits at index "
                f"{position:.3f} along grid {grid.grid_id!r}'s first axis, which "
                "is not one of its slices; the segmentation was drawn on a "
                "different reconstruction",
                code="E405",
            )
        volumes[segment][nearest] = frame["data"]
    return volumes


def read_dicom_seg(
    path: str | os.PathLike[str],
    *,
    report: ConversionReport | None = None,
) -> tuple[dict[int, npt.NDArray[Any]], dict[str, Any]]:
    """A SEG file as ``{segment number: volume}`` plus its geometry and segments.

    The volume spans the planes the file carries, which is what a reader with
    no other information can honestly build.  An import into an existing
    sample goes through :func:`read_dicom_seg_frames` and :func:`place_frames`
    instead, because the grid it is going onto is what knows how many slices
    the study has.
    """
    frames, geometry = read_dicom_seg_frames(path)
    segments = geometry["segments"]
    depth, rows, columns = geometry["shape"]
    volumes: dict[int, npt.NDArray[Any]] = {
        number: np.zeros(
            (depth, rows, columns),
            dtype=np.float32 if geometry["fractional"] else bool,
        )
        for number in segments
    }
    for frame in frames:
        if frame["segment"] in volumes:
            volumes[frame["segment"]][frame["plane"]] = frame["data"]
    fractional = geometry["fractional"]
    scale = float(geometry.get("scale", 255))
    if report is not None:
        report.decision(
            "segments",
            f"{len(segments)} segment(s) were read by frame geometry, not by "
            "frame order; a SEG may store its frames in any order",
            {"segments": {k: v["label"] for k, v in segments.items()}},
        )
        if fractional:
            report.decision(
                "fractional",
                "SegmentationType is FRACTIONAL, so the segments became a "
                "`probmap` rather than being thresholded",
                {"scale": scale},
            )
    return volumes, geometry


def _segments(dataset: Any) -> dict[int, dict[str, Any]]:
    """Segment number → label, algorithm and coded concept, where given."""
    out: dict[int, dict[str, Any]] = {}
    for item in getattr(dataset, "SegmentSequence", []):
        number = int(item.SegmentNumber)
        code = None
        sequence = getattr(item, "SegmentedPropertyTypeCodeSequence", None)
        if sequence:
            entry = sequence[0]
            code = {
                "scheme": str(getattr(entry, "CodingSchemeDesignator", "")),
                "code": str(getattr(entry, "CodeValue", "")),
                "meaning": str(getattr(entry, "CodeMeaning", "")),
            }
        out[number] = {
            "label": str(getattr(item, "SegmentLabel", f"segment_{number}")),
            "algorithm": str(getattr(item, "SegmentAlgorithmType", "")),
            "code": code,
        }
    return out


def _frame_placement(dataset: Any, n_frames: int) -> dict[str, Any]:
    """Where each frame belongs, from the per-frame functional groups."""
    shared = _first_item(dataset, "SharedFunctionalGroupsSequence")
    per_frame = list(getattr(dataset, "PerFrameFunctionalGroupsSequence", []))
    orientation = _orientation(shared, per_frame)
    normal = np.cross(orientation[:3], orientation[3:])
    positions: list[npt.NDArray[np.float64]] = []
    segments: list[int] = []
    for index in range(n_frames):
        group = per_frame[index] if index < len(per_frame) else None
        positions.append(_position(group, shared))
        segments.append(_segment_number(group))
    projected = [float(np.dot(p, normal)) for p in positions]
    planes = sorted(set(np.round(projected, 4)))
    indices = [planes.index(round(v, 4)) for v in projected]
    spacing = _spacing(shared, planes)
    origin_index = indices.index(0) if 0 in indices else 0
    return {
        "planes": planes,
        "indices": indices,
        "segments": segments,
        # Each frame's own `ImagePositionPatient`, kept rather than collapsed
        # into a plane index: placing a frame on the grid it is being imported
        # onto needs the position, and the index is only meaningful among the
        # planes this file happens to carry.
        "positions": positions,
        "spacing": spacing,
        "origin": [float(v) for v in positions[origin_index]],
        "direction": [
            [float(v) for v in row]
            for row in np.stack([normal, orientation[3:], orientation[:3]], axis=1)
        ],
        "source_series": _source_series(dataset),
    }


def _orientation(shared: Any, per_frame: Sequence[Any]) -> npt.NDArray[np.float64]:
    for holder in (shared, *per_frame):
        group = _first_item(holder, "PlaneOrientationSequence")
        if group is not None and hasattr(group, "ImageOrientationPatient"):
            return np.asarray(
                [float(v) for v in group.ImageOrientationPatient], dtype=np.float64
            )
    raise MEDH5ValidationError(
        "the SEG carries no ImageOrientationPatient, so its frames cannot be placed"
    )


def _position(group: Any, shared: Any) -> npt.NDArray[np.float64]:
    for holder in (group, shared):
        plane = _first_item(holder, "PlanePositionSequence")
        if plane is not None and hasattr(plane, "ImagePositionPatient"):
            return np.asarray(
                [float(v) for v in plane.ImagePositionPatient], dtype=np.float64
            )
    raise MEDH5ValidationError("a SEG frame carries no ImagePositionPatient")


def _segment_number(group: Any) -> int:
    identification = _first_item(group, "SegmentIdentificationSequence")
    if identification is None:
        return 1
    return int(getattr(identification, "ReferencedSegmentNumber", 1))


def _spacing(shared: Any, planes: Sequence[float]) -> list[float]:
    measures = _first_item(shared, "PixelMeasuresSequence")
    in_plane = [1.0, 1.0]
    if measures is not None and hasattr(measures, "PixelSpacing"):
        in_plane = [float(v) for v in measures.PixelSpacing]
    if len(planes) > 1:
        gaps = np.diff(np.asarray(planes, dtype=np.float64))
        through = float(np.median(np.abs(gaps)))
    elif measures is not None and getattr(measures, "SpacingBetweenSlices", None):
        through = float(measures.SpacingBetweenSlices)
    else:
        through = 1.0
    return [through, in_plane[0], in_plane[1]]


def _first_item(holder: Any, name: str) -> Any:
    sequence = getattr(holder, name, None) if holder is not None else None
    return sequence[0] if sequence else None


def _first(dataset: Any, name: str) -> str | None:
    value = getattr(dataset, name, None)
    return None if value is None else str(value)


def _source_series(dataset: Any) -> str | None:
    for item in getattr(dataset, "ReferencedSeriesSequence", []):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid:
            return str(uid)
    return None


def from_dicom_seg(
    path: str | os.PathLike[str],
    sample: str | os.PathLike[str],
    *,
    ann_id: str = "seg",
    grid: str | None = None,
    annotated_classes: Sequence[str] | str = "all_given",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Add a DICOM SEG's segments to an existing sample (§7).

    The SEG is matched to a grid by frame of reference where it declares one,
    and its shape is checked against that grid: a SEG drawn on a different
    reconstruction is refused rather than force-fitted.
    """
    import medh5
    from medh5.labels.labelset import LabelClass, LabelSet, OntologyCode

    log = report or ConversionReport(converter="from-dicom-seg")
    log.source = os.fspath(path)
    frames, geometry = read_dicom_seg_frames(path)

    with medh5.open(sample) as opened:
        grid_id = grid or _match_grid(opened, geometry, log)
        target = opened.grids[grid_id]
        existing = opened.label_set
    # Placed by position, not reshaped by count: a SEG that omits its empty
    # frames covers only the slices it labels, and the grid is what says how
    # many slices the study has (§3.3).
    volumes = place_frames(frames, geometry, target)
    log.decision(
        "frames",
        f"{len(frames)} frame(s) were placed on grid {grid_id!r} by "
        "ImagePositionPatient; a SEG may omit its empty frames and store the "
        "rest in any order",
        {"frames": len(frames), "slices": int(target.spatial_shape[0])},
    )
    if geometry["fractional"]:
        log.decision(
            "fractional",
            "SegmentationType is FRACTIONAL, so the segments became a "
            "`probmap` rather than being thresholded",
            {"scale": geometry["scale"]},
        )

    segments = geometry["segments"]
    log.decision(
        "segments",
        f"{len(segments)} segment(s) were read by frame geometry, not by "
        "frame order; a SEG may store its frames in any order",
        {"segments": {k: v["label"] for k, v in segments.items()}},
    )
    label_set = existing
    if label_set is not None:
        # DICOM segment numbers are positional --- a SEG numbers its segments
        # 1..N whatever the classes mean --- so an existing label set is matched
        # by SegmentLabel, never by number.  Matching on the number would import
        # "lesion" as whatever class happens to hold id 2.
        mapping = _match_segments(segments, label_set)
        volumes = {mapping[number]: volume for number, volume in volumes.items()}
        log.decision(
            "segment_mapping",
            "segments were matched to the sample's label set by SegmentLabel; "
            "DICOM segment numbers are positional and carry no identity",
            {"mapping": {str(k): v for k, v in mapping.items()}},
        )
    if label_set is None:
        classes = []
        for number, info in sorted(segments.items()):
            code = info["code"]
            classes.append(
                LabelClass(
                    number,
                    _key(info["label"]),
                    info["label"],
                    codes=(
                        (
                            OntologyCode(
                                system=code["scheme"],
                                code=code["code"],
                                name=code["meaning"] or None,
                            ),
                        )
                        if code and code["scheme"] and code["code"]
                        else ()
                    ),
                )
            )
        label_set = LabelSet("dicom-seg", version="1.0.0", classes=classes)
        bound = sum(1 for c in classes if c.codes)
        log.decision(
            "label_set",
            f"{len(classes)} class(es) were minted from SegmentLabel; "
            f"{bound} carried a coded concept and got an ontology binding",
            {"bound": bound, "total": len(classes)},
        )
        if bound < len(classes):
            log.guess(
                "ontology",
                "segments without a coded concept were left unbound rather than "
                "given a guessed code: a wrong binding is undetectable (W912)",
                {"unbound": [c.key for c in classes if not c.codes]},
            )

    with medh5.amend(sample) as writer:
        if existing is None:
            writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import",
            agent=tool,
            tool="medh5 convert from-dicom-seg",
            inputs=[f"dicom:{_first_or(geometry, 'source_series')}"],
        )
        payload = {number: volumes[number] for number in sorted(volumes)}
        if geometry["fractional"]:
            kind, _ = writer.add_segmentation(
                ann_id,
                grid=grid_id,
                probabilities=payload,
                annotated_classes=annotated_classes,
                prov=activity,
            )
        else:
            kind, stats = writer.add_segmentation(
                ann_id,
                grid=grid_id,
                masks=payload,
                annotated_classes=annotated_classes,
                prov=activity,
            )
            if stats is not None and stats.edges:
                log.decision(
                    "overlap",
                    f"{len(stats.edges)} segment pair(s) overlap; the encoding "
                    f"chosen ({kind!r}) represents them without flattening",
                    {"overlapping_pairs": len(stats.edges)},
                )
        log.decision("encoding", f"segments were stored as {kind!r}", {"kind": kind})
    log.outputs.append(os.fspath(sample))
    return log


def _match_segments(
    segments: Mapping[int, dict[str, Any]], label_set: Any
) -> dict[int, int]:
    """Segment number → class id, resolved through the label set by name."""
    mapping: dict[int, int] = {}
    unmatched: list[str] = []
    for number, info in sorted(segments.items()):
        label = info["label"]
        found = label_set.get(_key(label)) or label_set.get(label)
        if found is None:
            unmatched.append(label)
            continue
        mapping[number] = found.id
    if unmatched:
        raise MEDH5ValidationError(
            f"the sample's label set has no class for segment(s) {unmatched}; add "
            "them, or import into a sample with no label set to mint them",
            code="E402",
        )
    return mapping


def _first_or(geometry: Mapping[str, Any], key: str) -> str:
    return str(geometry.get(key) or "unknown")


def _key(label: str) -> str:
    """A SegmentLabel as a label-set key (§5.2)."""
    return sanitize_key(label, fallback="segment")


def _in_plane_match(grid: Any, geometry: Mapping[str, Any]) -> bool:
    """Whether a grid's rows and columns are the SEG's.

    Depth is deliberately not compared: a SEG that omits its empty frames has
    fewer planes than the series it was drawn on, and matching on the full
    shape rejected exactly the grid it belonged to.
    """
    shape = tuple(grid.spatial_shape)
    return len(shape) == 3 and shape[1:] == (
        int(geometry["rows"]),
        int(geometry["columns"]),
    )


def _match_grid(sample: Any, geometry: Mapping[str, Any], log: ConversionReport) -> str:
    """Pick the grid the SEG was drawn on, by frame of reference then by shape."""
    frame = geometry.get("frame_uid")
    if frame:
        matches = [g for g in sample.grids.values() if g.frame_uid == frame]
        if len(matches) == 1:
            return str(matches[0].grid_id)
        if len(matches) > 1:
            same = [g for g in matches if _in_plane_match(g, geometry)]
            if len(same) == 1:
                return str(same[0].grid_id)
    candidates = [g for g in sample.grids.values() if _in_plane_match(g, geometry)]
    if len(candidates) == 1:
        log.guess(
            "grid",
            f"the SEG names frame {frame!r}, which no grid declares; it was "
            f"matched to grid {candidates[0].grid_id!r} on shape alone",
            {"grid": candidates[0].grid_id},
        )
        return str(candidates[0].grid_id)
    raise MEDH5ValidationError(
        f"cannot tell which grid the SEG belongs to: frame {frame!r} matches no "
        f"grid and {len(candidates)} grid(s) share its rows and columns. Pass "
        "grid= explicitly rather than letting the converter guess.",
        code="E101",
    )


def to_dicom_seg(
    sample: str | os.PathLike[str],
    ann_id: str,
    source_images: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    classes: Sequence[int | str] | None = None,
    series_description: str = "medh5 segmentation",
    report: ConversionReport | None = None,
) -> Path:
    """Export one voxel annotation as a DICOM SEG against its source slices.

    *source_images* are the DICOM files the segmentation was drawn on: a SEG is
    defined by reference to them, and one written without them is orphaned in
    every PACS that receives it.
    """
    import medh5

    hd = require_highdicom()
    from medh5.io.dicom import require_pydicom

    pydicom = require_pydicom()
    log = report or ConversionReport(converter="to-dicom-seg")
    log.source = os.fspath(sample)

    datasets = [pydicom.dcmread(os.fspath(p)) for p in source_images]
    if not datasets:
        raise MEDH5ValidationError("to_dicom_seg needs the source DICOM images")
    normal = _source_normal(datasets[0])
    datasets.sort(key=lambda d: float(np.dot(_ipp(d), normal)))

    with medh5.open(sample) as opened:
        annotation = opened.annotations[ann_id]
        ids = annotation.resolve_classes(classes)
        planes = np.asarray(annotation.dense(list(ids)), dtype=bool)
        descriptions = [
            hd.seg.SegmentDescription(
                segment_number=i + 1,
                segment_label=annotation.class_key(class_id),
                segmented_property_category=hd.sr.CodedConcept(
                    "91723000", "SCT", "Anatomical Structure"
                ),
                segmented_property_type=hd.sr.CodedConcept(
                    "91723000", "SCT", "Anatomical Structure"
                ),
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
            )
            for i, class_id in enumerate(ids)
        ]
        # highdicom wants (frames, rows, columns, segments); `dense` returns
        # (segments, z, y, x).
        pixels = np.transpose(planes, (1, 2, 3, 0))
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=pixels,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=descriptions,
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="medh5",
            manufacturer_model_name="medh5",
            software_versions=medh5.__version__,
            device_serial_number="0",
            series_description=series_description,
            omit_empty_frames=False,
        )
    target = Path(os.fspath(out))
    segmentation.save_as(str(target))
    log.decision(
        "segments",
        f"{len(ids)} class(es) were written as separate segments, so overlap "
        "survives the export",
        {"classes": [str(i) for i in ids]},
    )
    log.guess(
        "coded_concepts",
        "segments were given a generic 'Anatomical Structure' concept; MEDH5 "
        "ontology bindings are not automatically translated into DICOM codes, "
        "and inventing one would be worse than leaving it generic",
        {},
    )
    log.outputs.append(str(target))
    return target


def _source_normal(dataset: Any) -> npt.NDArray[np.float64]:
    orientation = np.asarray(
        [float(v) for v in dataset.ImageOrientationPatient], dtype=np.float64
    )
    return np.cross(orientation[:3], orientation[3:])


def _ipp(dataset: Any) -> npt.NDArray[np.float64]:
    return np.asarray(
        [float(v) for v in dataset.ImagePositionPatient], dtype=np.float64
    )


__all__ = [
    "BINARY",
    "FRACTIONAL",
    "SLICE_TOLERANCE",
    "from_dicom_seg",
    "place_frames",
    "read_dicom_seg",
    "read_dicom_seg_frames",
    "require_highdicom",
    "to_dicom_seg",
]
