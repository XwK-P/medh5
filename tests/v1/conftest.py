"""Fixtures for the 1.0 test suite.

Samples are built by the public writer, so every test that reads one is also a
test that the writer produces something readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.labels.labelset import LabelClass, LabelSet

SEED = 20260815
SHAPE = (16, 24, 24)


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet(
        "test-v1",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver", category="organ"),
            LabelClass(2, "spleen", "Spleen", category="organ"),
            LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
            LabelClass(4, "vessel", "Vessel", category="vessel"),
        ],
    )


def block(shape: tuple[int, ...], origin: tuple[int, ...], size: int = 6) -> Any:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(o, o + size) for o in origin)] = True
    return mask


@pytest.fixture
def masks() -> dict[int, Any]:
    """Three classes where 1 and 3 overlap and 2 does not touch either."""
    return {
        1: block(SHAPE, (2, 2, 2), 8),
        2: block(SHAPE, (2, 14, 2), 6),
        3: block(SHAPE, (4, 4, 4), 3),
    }


@pytest.fixture
def ct() -> Any:
    rng = np.random.default_rng(SEED)
    return rng.integers(-1000, 1500, SHAPE).astype(np.int16)


def write_sample(
    path: Path,
    *,
    label_set: LabelSet | None = None,
    masks: dict[int, Any] | None = None,
    ct: Any = None,
    timepoints: tuple[str, ...] = ("tp0",),
    index: bool = False,
    encoding: str = "auto",
    annotated: Any = "all_given",
    codec: str = "portable",
    sample_id: str | None = None,
) -> Path:
    """A complete, valid sample --- the base every reader test starts from."""
    rng = np.random.default_rng(SEED)
    image = ct if ct is not None else rng.integers(-1000, 1500, SHAPE).astype(np.int16)
    with medh5.create(
        path, sample_id=sample_id or path.stem, subject_id="subj-A", codec=codec
    ) as w:
        w.identity(sex="F", bodypart="abdomen")
        w.cohort(dataset_id="test", site_id="site-A")
        for i, tp in enumerate(timepoints):
            w.add_timepoint(
                tp, label="baseline" if i == 0 else f"fu{i}", days_from_baseline=90 * i
            )
        if label_set is not None:
            w.label_set(label_set)
        tool = w.software("medh5", medh5.__version__)
        act = w.activity("import", agent=tool, tool="test suite")
        for tp in timepoints:
            w.add_grid(
                f"ct_{tp}",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                origin=(-12.0, -9.6, -9.6),
                timepoint=tp,
                frame_uid=f"pseudo:frame-{tp}",
                patch_hint=(8, 8, 8),
            )
            w.add_image(
                f"CT_{tp}",
                image,
                grid=f"ct_{tp}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                prov=act,
            )
            if masks is not None:
                w.add_segmentation(
                    f"organs_{tp}",
                    grid=f"ct_{tp}",
                    masks=masks,
                    encoding=encoding,
                    annotated_classes=annotated,
                    prov=act,
                    quality={"status": "approved"},
                )
        if index and masks is not None:
            w.build_index(max_coords=64)
        w.deidentification(method="dicom-psi-profile", date_shift_days=-117)
    return path


@pytest.fixture
def sample_path(tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]) -> Path:
    return write_sample(tmp_path / "case.medh5", label_set=label_set, masks=masks)


@pytest.fixture
def longitudinal_path(
    tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]
) -> Path:
    return write_sample(
        tmp_path / "long.medh5",
        label_set=label_set,
        masks=masks,
        timepoints=("tp0", "tp1"),
        index=True,
    )


def write_dicom_series(
    directory: Path,
    *,
    patient_id: str,
    study_uid: str,
    study_date: str,
    modality: str = "CT",
    shape: tuple[int, int, int] = (6, 16, 20),
    spacing: tuple[float, float, float] = (2.5, 0.8, 0.9),
    origin: tuple[float, float, float] = (-10.0, -20.0, 30.0),
    frame_uid: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """A minimal but valid CT/PT series.

    Two details are deliberate, because the converter's job is to survive them:
    ``SliceThickness`` is twice the slice increment (it is the slab, not the
    step), and ``InstanceNumber`` counts *down*, so a converter that trusts it
    rather than geometry produces a flipped volume.
    """
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    series_uid = generate_uid()
    frame = frame_uid or generate_uid()
    volume = rng.integers(0, 2000, shape).astype(np.uint16)
    for k in range(shape[0]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient_id
        ds.PatientName = "ANON^ANON"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.StudyDate = study_date
        ds.StudyTime = "120000"
        ds.ContentDate = study_date
        ds.ContentTime = "120000"
        ds.SeriesNumber = 1
        ds.Modality = modality
        ds.SeriesDescription = f"{modality} axial"
        ds.Manufacturer = "SYNTH"
        ds.ConvolutionKernel = "B30f"
        ds.SliceThickness = spacing[0] * 2
        ds.PixelSpacing = [spacing[1], spacing[2]]
        ds.ImageOrientationPatient = [0, 0, 1, 0, 1, 0]
        ds.ImagePositionPatient = [origin[0] - k * spacing[0], origin[1], origin[2]]
        ds.InstanceNumber = shape[0] - k
        ds.Rows, ds.Columns = shape[1], shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.PixelData = volume[k].tobytes()
        ds.save_as(str(directory / f"{modality}_{k:03d}.dcm"), enforce_file_format=True)
    assert pydicom is not None
    return {
        "series_uid": series_uid,
        "frame_uid": frame,
        "volume": volume,
        "paths": sorted(str(p) for p in directory.glob("*.dcm")),
    }
