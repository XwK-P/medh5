"""RTSTRUCT import and export (``medh5.io.rtstruct``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.io.report import ConversionReport
from tests.helpers import write_dicom_series
from tests.kits import Flat

pydicom = pytest.importorskip("pydicom")


class TestRtstruct:
    @pytest.fixture
    def prepared(self, tmp_path: Path) -> dict[str, Any]:
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root,
            patient_id="PSEUDO-003",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        path = tmp_path / "case.medh5"
        from_dicom(root, path, group_by="study")
        return {"path": path, "series": series, "root": root}

    def _rtstruct(
        self, prepared, tmp_path, *, hole: bool = True, hole_first: bool = False
    ) -> Path:
        """A square ROI on two slices, optionally with a square hole.

        ``hole_first`` lists the inner contour before its outer, which DICOM
        permits and no ordering rule forbids.
        """
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        reference = pydicom.dcmread(prepared["series"]["paths"][0])
        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[sorted(sample.grids)[0]]

        def square(centre_y: int, centre_x: int, radius: int, plane: int):
            corners = [
                (centre_y - radius, centre_x - radius),
                (centre_y - radius, centre_x + radius),
                (centre_y + radius, centre_x + radius),
                (centre_y + radius, centre_x - radius),
            ]
            return grid.index_to_world(
                np.array([[plane, y, x] for y, x in corners], dtype=float)
            )

        structure = Dataset()
        structure.file_meta = FileMetaDataset()
        structure.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        structure.SOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
        structure.Modality = "RTSTRUCT"
        structure.StructureSetLabel = "test"
        structure.PatientID = reference.PatientID
        structure.PatientName = reference.PatientName
        structure.StudyInstanceUID = reference.StudyInstanceUID
        structure.SeriesInstanceUID = generate_uid()
        frame = Dataset()
        frame.FrameOfReferenceUID = reference.FrameOfReferenceUID
        structure.ReferencedFrameOfReferenceSequence = [frame]
        roi = Dataset()
        roi.ROINumber = 1
        roi.ROIName = "liver"
        roi.ReferencedFrameOfReferenceUID = frame.FrameOfReferenceUID
        structure.StructureSetROISequence = [roi]
        item = Dataset()
        item.ReferencedROINumber = 1
        item.ROIDisplayColor = [200, 90, 70]
        item.ContourSequence = []
        for plane in (2, 3):
            shapes = [square(8, 10, 5, plane)]
            if hole:
                inner = square(8, 10, 2, plane)
                shapes = [inner, *shapes] if hole_first else [*shapes, inner]
            for points in shapes:
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = len(points)
                contour.ContourData = [float(v) for v in points.reshape(-1)]
                item.ContourSequence.append(contour)
        structure.ROIContourSequence = [item]
        out = tmp_path / "rt.dcm"
        structure.save_as(str(out), enforce_file_format=True)
        return out

    def test_S8_6_contours_are_stored_as_contours(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois")
        with medh5.open(prepared["path"]) as sample:
            annotation = sample.annotations["rois"]
            assert annotation.kind == "contours"
            assert annotation.space == "world"
            assert len(list(annotation.polygons())) == 4
            assert {c.key for c in sample.label_set} == {"liver"}
        assert report.of_kind("contours")

    def test_S8_6_a_contour_inside_another_is_a_hole(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=True)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            roles = [p.role for p in sample.annotations["rois"].polygons()]
            assert roles.count("hole") == 2
            mask = sample.annotations["rois_mask"].dense(["liver"])[0]
            assert mask[2, 5, 6], "the ring is filled"
            assert not mask[2, 8, 10], "the hole is not"
        assert report.of_kind("holes")

    def test_S8_6_a_hole_listed_before_its_outer_is_still_a_hole(
        self, prepared, tmp_path
    ):
        """DICOM does not order outer contours before the holes they enclose.

        Subtracting a hole from a mask whose outer has not been drawn yet does
        nothing, and the outer then fills the cavity back in --- a conversion
        that turns holes into foreground without saying so.
        """
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=True, hole_first=True)
        from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            roles = [p.role for p in sample.annotations["rois"].polygons()]
            assert roles.count("hole") == 2, "containment, not order, makes a hole"
            mask = sample.annotations["rois_mask"].dense(["liver"])[0]
            assert mask[2, 5, 6], "the ring is filled"
            assert not mask[2, 8, 10], "and the hole was not filled back in"

    def test_rasterization_is_opt_in_and_recorded(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=False)
        report = from_rtstruct(rt, prepared["path"], ann_id="rois")
        with medh5.open(prepared["path"]) as sample:
            assert "rois_mask" not in sample.annotations
        assert not report.of_kind("rasterization")

        report = from_rtstruct(rt, prepared["path"], ann_id="raster", rasterize=True)
        with medh5.open(prepared["path"]) as sample:
            derived = sample.annotations["raster_mask"]
            # §6.2: `derived_from` holds annotation ids, not paths.
            assert derived.header.derived_from == ("raster",)
            activity = sample.document.provenance.activity(derived.prov)
            assert "even-odd" in activity.params["rule"]
        assert report.guesses

    def test_the_round_trip_preserves_the_contours(self, prepared, tmp_path):
        from medh5.io.rtstruct import from_rtstruct, read_rtstruct, to_rtstruct

        rt = self._rtstruct(prepared, tmp_path, hole=False)
        original, _ = read_rtstruct(rt)
        from_rtstruct(rt, prepared["path"], ann_id="rois")
        back = to_rtstruct(
            prepared["path"],
            "rois",
            prepared["series"]["paths"],
            tmp_path / "back.dcm",
        )
        recovered, meta = read_rtstruct(back)
        assert meta["names"] == {1: "liver"}
        assert len(recovered[1]) == len(original[1])
        for first, second in zip(
            sorted(original[1], key=lambda a: a[0, 0].item()),
            sorted(recovered[1], key=lambda a: a[0, 0].item()),
            strict=True,
        ):
            assert np.allclose(first, second, atol=1e-3)

    def test_exporting_a_mask_is_refused(self, prepared, tmp_path):
        from medh5.io.rtstruct import to_rtstruct

        rt = self._rtstruct(prepared, tmp_path)
        from medh5.io.rtstruct import from_rtstruct

        from_rtstruct(rt, prepared["path"], ann_id="rois", rasterize=True)
        with pytest.raises(MEDH5ValidationError, match="polygons"):
            to_rtstruct(
                prepared["path"],
                "rois_mask",
                prepared["series"]["paths"],
                tmp_path / "x.dcm",
            )

    def test_a_non_rtstruct_is_refused(self, prepared):
        from medh5.io.rtstruct import read_rtstruct

        with pytest.raises(MEDH5ValidationError, match="not 'RTSTRUCT'"):
            read_rtstruct(prepared["series"]["paths"][0])

    def test_S8_6_imported_rtstruct_polygons_record_their_plane(self, tmp_path: Path):
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        from medh5.io.rtstruct import from_rtstruct

        path = tmp_path / "ct.medh5"
        # No label set: the importer mints one from ROIName, which is where
        # the key sanitiser runs.
        with medh5.create(path, sample_id="ct") as w:
            w.add_grid(
                "g",
                shape=Flat.SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="f0",
            )
            w.add_image("CT", Flat.image(), grid="g", modality="CT")
        structure = Dataset()
        structure.file_meta = FileMetaDataset()
        structure.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        structure.SOPClassUID = pydicom.uid.RTStructureSetStorage
        structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
        structure.Modality = "RTSTRUCT"
        structure.StructureSetLabel = "test"
        frame = Dataset()
        frame.FrameOfReferenceUID = "f0"
        structure.ReferencedFrameOfReferenceSequence = [frame]
        roi = Dataset()
        roi.ROINumber = 1
        roi.ROIName = "GTV-1"
        roi.ReferencedFrameOfReferenceUID = "f0"
        structure.StructureSetROISequence = [roi]
        item = Dataset()
        item.ReferencedROINumber = 1
        contours = []
        for z in (2.0, 5.0):
            c = Dataset()
            c.ContourGeometricType = "CLOSED_PLANAR"
            # Grid spacing is 1 mm from origin 0, so world z is the slice index.
            square = [[z, 2.0, 2.0], [z, 2.0, 6.0], [z, 6.0, 6.0], [z, 6.0, 2.0]]
            c.NumberOfContourPoints = 4
            c.ContourData = [v for p in square for v in p]
            contours.append(c)
        item.ContourSequence = contours
        structure.ROIContourSequence = [item]
        rt = tmp_path / "rt.dcm"
        structure.save_as(str(rt), enforce_file_format=True)
        from_rtstruct(rt, path, ann_id="rt")
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["rt"]
            assert sorted(ann.by_plane()) == [(0, 2), (0, 5)]
            # §5.2: the minted key is schema-valid, so the write did not fail E005.
            assert sample.label_set is not None
            assert sample.label_set[1].key == "gtv_1"

    def test_Q18_S8_6_an_island_inside_a_hole_survives_rasterisation(
        self, tmp_path: Path
    ):
        """Unioning outers and subtracting every hole erased the island."""
        from medh5.io.rtstruct import _assign_roles, _rasterize

        path = tmp_path / "plane.medh5"
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.add_grid("g", shape=(1, 20, 20), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((1, 20, 20), np.int16), grid="g", modality="CT")
        with medh5.open(path) as s:
            grid = s.grids["g"]

        def square(lo: float, hi: float) -> np.ndarray:
            return np.array(
                [[0.0, lo, lo], [0.0, lo, hi], [0.0, hi, hi], [0.0, hi, lo]]
            )

        contours = [square(2, 17), square(5, 14), square(8, 11)]
        log = ConversionReport()
        roles = [role for _, role, _ in _assign_roles(contours, grid, log, "organ")]
        assert roles == ["outer", "hole", "outer"]
        polygons = [
            SimpleNamespace(vertices=c, class_id=1, role=r)
            for c, r in zip(contours, roles, strict=True)
        ]
        mask = _rasterize(polygons, grid, log)[1][0]
        assert mask[3, 3]  # inside the outer only
        assert not mask[6, 6]  # inside the hole
        assert mask[9, 9]  # inside the island inside the hole

    def test_Q18_two_overlapping_outlines_do_not_cancel(self, tmp_path: Path):
        from medh5.io.rtstruct import _rasterize

        path = tmp_path / "plane.medh5"
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.add_grid("g", shape=(1, 20, 20), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((1, 20, 20), np.int16), grid="g", modality="CT")
        with medh5.open(path) as s:
            grid = s.grids["g"]
        left = np.array([[0.0, 2, 2], [0.0, 2, 12], [0.0, 12, 12], [0.0, 12, 2]])
        right = left + np.array([0.0, 0.0, 6.0])
        polygons = [
            SimpleNamespace(vertices=v, class_id=1, role="outer") for v in (left, right)
        ]
        mask = _rasterize(polygons, grid, ConversionReport())[1][0]
        assert mask[5, 10]  # in both: one region, not a hole
