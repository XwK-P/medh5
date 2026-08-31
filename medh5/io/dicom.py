"""DICOM import (spec §3, §4, §12).

A DICOM series is a pile of slices that *claims* to be a volume.  Turning it
into a grid means answering three questions the files do not answer directly,
and this converter answers each one by measurement rather than by assumption:

**Slice order.**  ``InstanceNumber`` is unreliable --- it is a display hint and
routinely disagrees with geometry.  Slices are ordered by their projection onto
the normal of ``ImageOrientationPatient``, which is the only ordering the
geometry itself defines.

**Spacing along the normal.**  ``SliceThickness`` is the thickness of the *slab*,
not the distance between slice origins; a series with 5 mm slices at 2.5 mm
increments is common and using the thickness would halve the volume's extent.
The z spacing is measured from consecutive positions, and a series whose gaps
vary by more than a tolerance is refused rather than averaged --- an irregular
stack is not a grid, and pretending otherwise puts every label at the wrong
depth.

**Values.**  ``RescaleSlope``/``RescaleIntercept`` are stored, never applied
(§4.2), so ``read()`` returns what the file stored and ``read(physical=True)``
returns HU.  A converter that bakes the rescale in makes the two indistinguishable
afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.io.grouping import (
    Occasion,
    SubjectGroup,
    group_by_subject,
    note_instance_ids,
    output_name,
)
from medh5.io.report import ConversionReport

SPACING_TOLERANCE = 0.01
"""Relative tolerance on gaps between consecutive slices before a stack is refused."""

STORED_TAGS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "XRayTubeCurrent",
    "ExposureTime",
    "ConvolutionKernel",
    "SliceThickness",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "MagneticFieldStrength",
    "SequenceName",
    "ContrastBolusAgent",
    "PatientPosition",
    "BodyPartExamined",
)
"""Acquisition parameters copied to ``/meta → acquisition`` (§4.5).

An explicit list, because §11.4 forbids copying DICOM tags wholesale: a bulk
copy carries names, dates and identifiers into a file that claims to be
de-identified.
"""


def require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pydicom is required for DICOM conversion. Install it with: "
            "pip install 'medh5[dicom]'"
        ) from exc
    return pydicom


@dataclass(slots=True)
class Series:
    """One DICOM series, resolved into a volume and its geometry."""

    series_uid: str
    study_uid: str
    modality: str
    patient_id: str | None
    study_date: str | None
    paths: list[str] = field(default_factory=list)
    description: str = ""
    frame_uid: str | None = None

    @property
    def n_slices(self) -> int:
        return len(self.paths)

    def to_json(self) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "modality": self.modality,
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "slices": self.n_slices,
            "description": self.description,
            "frame_uid": self.frame_uid,
        }

    def __repr__(self) -> str:
        return (
            f"Series({self.modality}, {self.n_slices} slices, "
            f"{self.description or self.series_uid[:16]})"
        )


def scan_dicom(root: str | os.PathLike[str]) -> list[Series]:
    """Walk a directory tree and collect its image series."""
    pydicom = require_pydicom()
    found: dict[str, Series] = {}
    for path in sorted(Path(os.fspath(root)).rglob("*")):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:  # noqa: BLE001 - a tree holds more than DICOM
            continue
        uid = getattr(header, "SeriesInstanceUID", None)
        if uid is None or not hasattr(header, "ImagePositionPatient"):
            continue
        series = found.get(uid)
        if series is None:
            series = Series(
                series_uid=str(uid),
                study_uid=str(getattr(header, "StudyInstanceUID", "")),
                modality=str(getattr(header, "Modality", "OT")),
                patient_id=_text(getattr(header, "PatientID", None)),
                study_date=_text(getattr(header, "StudyDate", None)),
                description=_text(getattr(header, "SeriesDescription", "")) or "",
                frame_uid=_text(getattr(header, "FrameOfReferenceUID", None)),
            )
            found[uid] = series
        series.paths.append(str(path))
    return sorted(found.values(), key=lambda s: (s.study_uid, s.series_uid))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_series(
    series: Series, *, report: ConversionReport | None = None
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One series as ``(volume, geometry)``, ordered and measured (§3.3)."""
    pydicom = require_pydicom()
    if not series.paths:
        raise MEDH5ValidationError(f"series {series.series_uid} has no files")
    slices = [pydicom.dcmread(p) for p in series.paths]
    orientation = np.asarray(
        _agreed(
            slices,
            lambda s: [float(v) for v in s.ImageOrientationPatient],
            "ImageOrientationPatient",
            series,
            code="E102",
        ),
        dtype=np.float64,
    )
    row, column = orientation[:3], orientation[3:]
    normal = np.cross(row, column)
    positions = np.asarray(
        [[float(v) for v in s.ImagePositionPatient] for s in slices], dtype=np.float64
    )
    # Geometry, not InstanceNumber: the projection onto the slice normal is the
    # only ordering the files themselves define.
    order = np.argsort(positions @ normal)
    slices = [slices[i] for i in order]
    positions = positions[order]

    z_spacing, gaps = _slice_spacing(
        positions,
        normal,
        series,
        report,
        _scalar(getattr(slices[0], "SliceThickness", None)),
    )
    pixel_spacing = _agreed(
        slices,
        lambda s: [float(v) for v in getattr(s, "PixelSpacing", (1.0, 1.0))],
        "PixelSpacing",
        series,
        code="E104",
    )
    volume = np.stack([np.asarray(s.pixel_array) for s in slices])
    direction = np.stack([normal, column, row], axis=1)
    geometry = {
        "shape": tuple(int(v) for v in volume.shape),
        "spacing": [z_spacing, pixel_spacing[0], pixel_spacing[1]],
        "origin": [float(v) for v in positions[0]],
        "direction": [[float(v) for v in r] for r in direction],
        "coord_system": "LPS",
        "units": "mm",
        "frame_uid": series.frame_uid,
        "rescale": _agreed(slices, _rescale, "RescaleSlope/Intercept", series),
        "acquisition": {
            tag: _scalar(getattr(slices[0], tag, None))
            for tag in STORED_TAGS
            if getattr(slices[0], tag, None) is not None
        },
        "gaps": gaps,
    }
    return volume, geometry


def _slice_spacing(
    positions: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    series: Series,
    report: ConversionReport | None,
    thickness: Any = None,
) -> tuple[float, list[float]]:
    """Spacing measured between slice origins, refusing an irregular stack."""
    if positions.shape[0] == 1:
        # No neighbour to measure an increment against.  SliceThickness is the
        # slab rather than the increment, which is why it is not trusted for a
        # stack --- but for a lone slice it is the only physical extent the
        # source states, and inventing 1 mm over a declared 5 mm gives the grid
        # a wrong extent that later resampling and export propagate.
        declared = _positive(thickness)
        if declared is None:
            if report is not None:
                report.guess(
                    "slice_spacing",
                    "a single-slice series carries no increment to measure and "
                    "declared no usable SliceThickness; 1 mm was assumed",
                    {"spacing": 1.0, "thickness": thickness, "slices": 1},
                )
            return 1.0, []
        if report is not None:
            report.decision(
                "slice_spacing",
                "a single-slice series has no neighbouring position to measure "
                f"against, so its declared SliceThickness ({declared:.4g} mm) "
                "was used as the through-plane spacing",
                {"spacing": declared, "thickness": declared, "slices": 1},
            )
        return declared, []
    projected = positions @ normal
    gaps = np.diff(projected)
    spacing = float(np.median(np.abs(gaps)))
    if spacing <= 0:
        raise MEDH5ValidationError(
            f"series {series.series_uid} has coincident slice positions",
            code="E104",
        )
    spread = float(np.max(np.abs(np.abs(gaps) - spacing)))
    if spread > SPACING_TOLERANCE * spacing:
        raise MEDH5ValidationError(
            f"series {series.series_uid} has irregular slice gaps "
            f"(median {spacing:.4g} mm, worst deviation {spread:.4g} mm); an "
            "irregular stack is not a grid --- resample it deliberately, or "
            "convert the slices as separate images",
            code="E104",
        )
    if report is not None:
        report.decision(
            "slice_spacing",
            f"z spacing {spacing:.4g} mm was measured between slice origins, not "
            "taken from SliceThickness (which is the slab, not the increment)",
            {"spacing": spacing, "thickness": thickness, "slices": len(positions)},
        )
    return spacing, [float(g) for g in gaps]


def _agreed(
    slices: Sequence[Any],
    extract: Callable[[Any], Any],
    what: str,
    series: Series,
    *,
    code: str = "E204",
) -> Any:
    """The value every slice agrees on, or a refusal.

    A series is one volume with one geometry and one modality LUT, so each of
    these is read once --- but reading it from ``slices[0]`` *without* checking
    is what makes that assumption dangerous.  A stack whose slices disagree then
    silently takes the first slice's answer for all of them: a PET series with a
    per-slice rescale (which is ordinary) reports the wrong activity everywhere
    slice 0's LUT does not apply, and a stack with one rotated slice places that
    slice's voxels somewhere the scanner never put them.  Neither is visible in
    the output, so both are refused here rather than averaged or assumed.
    """
    values = [extract(s) for s in slices]
    first = values[0]
    for slice_, value in zip(slices[1:], values[1:], strict=True):
        if _differs(first, value):
            # Named by SOPInstanceUID, not by position: these checks run either
            # side of the geometric sort, so an index would mean two things.
            uid = getattr(slice_, "SOPInstanceUID", "?")
            raise MEDH5ValidationError(
                f"series {series.series_uid}: instance {uid} declares "
                f"{what} {value!r}, but the first slice declares {first!r}. "
                f"A series is "
                f"one volume with one geometry and one modality LUT; storing the "
                f"first slice's value for all of them would misplace or misscale "
                f"the rest without saying so. Split the series, or resample it "
                f"deliberately with a tool that records what it did.",
                code=code,
            )
    return first


def _differs(first: Any, other: Any) -> bool:
    if (first is None) != (other is None):
        return True
    if first is None:
        return False
    return not np.allclose(
        np.asarray(first, dtype=np.float64),
        np.asarray(other, dtype=np.float64),
        atol=1e-6,
    )


def _rescale(dataset: Any) -> tuple[float, float] | None:
    slope = _scalar(getattr(dataset, "RescaleSlope", None))
    intercept = _scalar(getattr(dataset, "RescaleIntercept", None))
    if slope is None and intercept is None:
        return None
    return float(slope if slope is not None else 1.0), float(
        intercept if intercept is not None else 0.0
    )


def _positive(value: Any) -> float | None:
    """*value* as a usable physical length, or ``None`` if it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def from_dicom(
    root: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    group_by: str = "subject",
    modalities: Sequence[str] | None = None,
    series_uids: Sequence[str] | None = None,
    codec: str = "balanced",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Convert a DICOM tree into one sample per subject (or per study).

    *out* is a directory when more than one sample results, and a file path when
    exactly one does.
    """
    log = report or ConversionReport(converter="from-dicom")
    log.source = os.fspath(root)
    series = scan_dicom(root)
    if series_uids is not None:
        wanted = set(series_uids)
        series = [s for s in series if s.series_uid in wanted]
    if modalities is not None:
        allowed = {m.upper() for m in modalities}
        series = [s for s in series if s.modality.upper() in allowed]
    if not series:
        raise MEDH5ValidationError(f"no DICOM image series found under {root}")

    studies: dict[str, list[Series]] = {}
    for entry in series:
        studies.setdefault(entry.study_uid, []).append(entry)
    occasions = [
        Occasion(
            key=study_uid,
            subject_id=_consistent_patient(entries, log),
            date=next((e.study_date for e in entries if e.study_date), None),
            payload=entries,
        )
        for study_uid, entries in sorted(studies.items())
    ]
    groups = group_by_subject(occasions, mode=group_by, report=log)
    target = Path(os.fspath(out))
    multiple = len(groups) > 1
    if multiple:
        target.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for group in groups:
        path = (
            target / f"{output_name(group, used, safe=_safe)}.medh5"
            if multiple
            else target
        )
        _write_subject(group, path, codec=codec, log=log)
    return log


def _consistent_patient(entries: Sequence[Series], log: ConversionReport) -> str | None:
    """One PatientID per study, or none --- never a majority vote."""
    ids = {e.patient_id for e in entries if e.patient_id}
    if len(ids) == 1:
        return next(iter(ids))
    if len(ids) > 1:
        log.warn(
            "identity",
            f"study {entries[0].study_uid} carries {len(ids)} different "
            "PatientIDs; it was left ungrouped rather than merged on a guess",
            {"patient_ids": sorted(ids)},
        )
    return None


def _safe(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)[:200]


def _write_subject(
    group: SubjectGroup,
    path: Path,
    *,
    codec: str,
    log: ConversionReport,
) -> None:
    import medh5

    note_instance_ids(group, log)
    days = group.days_from_baseline()
    with medh5.create(
        path,
        sample_id=_safe(group.subject_id),
        subject_id=group.subject_id,
        codec=codec,
    ) as writer:
        tool = writer.software("medh5", medh5.__version__)
        for index, occasion in enumerate(group.occasions):
            tp = f"tp{index}"
            entries: list[Series] = occasion.payload
            writer.add_timepoint(
                tp,
                index=index,
                date=_iso(occasion.date),
                days_from_baseline=days[index],
                study_uid=occasion.key,
                series_uids={e.modality: e.series_uid for e in entries},
            )
        for index, occasion in enumerate(group.occasions):
            tp = f"tp{index}"
            entries = occasion.payload
            for entry in entries:
                volume, geometry = read_series(entry, report=log)
                grid_id = f"{_safe(entry.modality).lower()}_{tp}"
                image_id = f"{_safe(entry.modality)}_{tp}"
                activity = writer.activity(
                    "import",
                    agent=tool,
                    tool="medh5 convert from-dicom",
                    inputs=[f"dicom:{entry.series_uid}"],
                    params={"slices": entry.n_slices, "modality": entry.modality},
                )
                writer.add_grid(
                    grid_id,
                    shape=geometry["shape"],
                    spacing=geometry["spacing"],
                    origin=geometry["origin"],
                    direction=geometry["direction"],
                    coord_system=geometry["coord_system"],
                    units=geometry["units"],
                    timepoint=tp,
                    frame_uid=geometry["frame_uid"],
                )
                rescale = geometry["rescale"]
                writer.add_image(
                    image_id,
                    volume,
                    grid=grid_id,
                    modality=entry.modality,
                    value_type="quantitative"
                    if entry.modality == "CT"
                    else "intensity",
                    value_units="HU" if entry.modality == "CT" else None,
                    rescale_slope=None if rescale is None else rescale[0],
                    rescale_intercept=None if rescale is None else rescale[1],
                    prov=activity,
                )
                if geometry["acquisition"]:
                    writer.acquisition(image_id, **geometry["acquisition"])
        log.decision(
            "value_scale",
            "RescaleSlope/Intercept were stored, not applied; read(physical=True) "
            "returns HU and read() returns stored values (§4.2)",
            {"subject": group.subject_id},
        )
        log.warn(
            "deidentification",
            "no de-identification record was written: this converter does not "
            "de-identify, and a file without the record must be treated as "
            "potentially identifying (§11.4, W903)",
            {"subject": group.subject_id},
        )
    log.outputs.append(str(path))


def _iso(value: str | None) -> str | None:
    """A DICOM ``YYYYMMDD`` date as ISO ``YYYY-MM-DD``."""
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():  # noqa: PLR2004 - DICOM DA length
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def select_series(
    series: Iterable[Series], *, thinnest: bool = True
) -> dict[str, Series]:
    """One series per modality --- the thinnest, or the first seen.

    "Thinnest" means most slices for the same anatomy, which is the usual proxy
    for the reconstruction a reader wants when a study holds several.
    """
    best: dict[str, Series] = {}
    for entry in series:
        current = best.get(entry.modality)
        if current is None or (thinnest and entry.n_slices > current.n_slices):
            best[entry.modality] = entry
    return best


__all__ = [
    "SPACING_TOLERANCE",
    "STORED_TAGS",
    "Series",
    "from_dicom",
    "read_series",
    "require_pydicom",
    "scan_dicom",
    "select_series",
]
