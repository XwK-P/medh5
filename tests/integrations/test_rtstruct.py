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
from medh5.validate import validate_file
from tests.helpers import write_dicom_series
from tests.kits import Flat, NestedSameNameRoiKit

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

    def test_an_rtstruct_comes_in_through_the_command_line(
        self, prepared, tmp_path, capsys
    ):
        """`medh5 convert from-rtstruct`; a file that is not DICOM is refused by
        name, where pydicom's error was a traceback."""
        from medh5.cli import EXIT_ERROR, EXIT_OK, main

        rt = self._rtstruct(prepared, tmp_path, hole=False)
        path = str(prepared["path"])
        argv = ["convert", "from-rtstruct", str(rt), path, "--id", "rois"]
        assert main(argv) == EXIT_OK
        with medh5.open(prepared["path"]) as sample:
            assert sample.annotations["rois"].kind == "contours"
        text = tmp_path / "notes.dcm"
        text.write_text("not DICOM", encoding="utf-8")
        capsys.readouterr()
        assert main(["convert", "from-rtstruct", str(text), path]) == EXIT_ERROR
        assert capsys.readouterr().err.startswith(f"medh5: {text} is not a DICOM file")

    def test_a_non_rtstruct_is_refused(self, prepared):
        from medh5.io.rtstruct import read_rtstruct

        with pytest.raises(MEDH5ValidationError, match="not 'RTSTRUCT'"):
            read_rtstruct(prepared["series"]["paths"][0])

    def test_S8_6_imported_rtstruct_polygons_record_their_plane(self, tmp_path: Path):
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

    def test_S8_6_an_island_inside_a_hole_survives_rasterisation(self, tmp_path: Path):
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

    def test_S8_6_two_overlapping_outlines_do_not_cancel(self, tmp_path: Path):
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


FRAME_A = "1.2.826.0.1.3680043.8.498.111"
FRAME_B = "1.2.826.0.1.3680043.8.498.222"
CLOSED = "CLOSED_PLANAR"
XOR = "CLOSEDPLANAR_XOR"


def _dicom_case(tmp_path: Path, frame: str = FRAME_A) -> SimpleNamespace:
    """A 6×16×16 CT of 1 mm voxels in *frame*, imported by ``from_dicom``."""
    from pydicom.uid import generate_uid

    from medh5.io.dicom import from_dicom

    series = write_dicom_series(
        tmp_path / "dcm",
        patient_id="PSEUDO-9",
        study_uid=generate_uid(),
        study_date="20260101",
        shape=(6, 16, 16),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        frame_uid=frame,
    )
    path = tmp_path / "case.medh5"
    from_dicom(tmp_path / "dcm", path, group_by="study")
    with medh5.open(path) as sample:
        grid = sample.grids[sorted(sample.grids)[0]]
    return SimpleNamespace(path=path, grid=grid, series=series, tmp=tmp_path)


def _square(
    grid: Any, plane: float, rows: tuple[int, int], columns: tuple[int, int]
) -> np.ndarray:
    """A square on *plane* whose edges sit on voxel edges, around the voxels
    of *rows* and *columns* (inclusive), in patient coordinates."""
    (r0, r1), (c0, c1) = rows, columns
    corners = [
        (r0 - 0.5, c0 - 0.5),
        (r0 - 0.5, c1 + 0.5),
        (r1 + 0.5, c1 + 0.5),
        (r1 + 0.5, c0 - 0.5),
    ]
    return grid.index_to_world(np.array([[plane, r, c] for r, c in corners], float))


def _structure_set(
    out: Path,
    rois: dict[int, tuple[str, list[tuple[str | None, np.ndarray]]]],
    frames: list[str],
    *,
    roi_frames: dict[int, str | None] | None = None,
) -> Path:
    """An RTSTRUCT of *rois* --- ``{number: (name, [(type, points), ...])}``,
    a type of ``None`` stating none --- listing *frames*.  Each ROI states the
    frame *roi_frames* gives it (``None``: none), else the first listed."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    structure = Dataset()
    structure.file_meta = FileMetaDataset()
    structure.file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    structure.SOPClassUID = pydicom.uid.RTStructureSetStorage
    structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
    structure.Modality = "RTSTRUCT"
    structure.StructureSetLabel = out.stem
    structure.ReferencedFrameOfReferenceSequence = []
    for uid in frames:
        item = Dataset()
        item.FrameOfReferenceUID = uid
        structure.ReferencedFrameOfReferenceSequence.append(item)
    structure.StructureSetROISequence = []
    structure.ROIContourSequence = []
    for number, (name, contours) in rois.items():
        roi = Dataset()
        roi.ROINumber = number
        roi.ROIName = name
        frame = (roi_frames or {}).get(number, frames[0] if frames else None)
        if frame:
            roi.ReferencedFrameOfReferenceUID = frame
        structure.StructureSetROISequence.append(roi)
        item = Dataset()
        item.ReferencedROINumber = number
        item.ContourSequence = []
        for kind, points in contours:
            contour = Dataset()
            if kind is not None:
                contour.ContourGeometricType = kind
            contour.NumberOfContourPoints = len(points)
            contour.ContourData = [float(v) for v in np.asarray(points).reshape(-1)]
            item.ContourSequence.append(contour)
        structure.ROIContourSequence.append(item)
    structure.save_as(str(out), enforce_file_format=True)
    return out


def _voxels(path: Path, ann_id: str = "rois") -> dict[str, int]:
    """Each class the rasterised mask examined, and how many voxels it has."""
    with medh5.open(path) as sample:
        mask = sample.annotations[f"{ann_id}_mask"]
        return {
            c.key: int(mask.dense([c.key])[0].sum()) for c in mask.annotated_classes
        }


class TestContourTypes:
    """A medh5 contour is a closed polygon in a slice of its grid (§8.6), so an
    import takes what an RTSTRUCT draws as areas and says what it leaves out.

    ``ContourGeometricType`` (PS3.3 C.8.8.6) was never read: crossing
    exclusive-or contours were their union, an open curve was filled, an
    applicator's path was flattened onto one slice and its class marked
    examined with nothing in it, and a POINT ROI vanished without a note.
    """

    @pytest.fixture
    def case(self, tmp_path: Path) -> SimpleNamespace:
        return _dicom_case(tmp_path)

    def _import(self, case: SimpleNamespace, rois: dict, **kw: Any) -> ConversionReport:
        from medh5.io.rtstruct import from_rtstruct

        rt = _structure_set(case.tmp / "rt.dcm", rois, [FRAME_A])
        return from_rtstruct(rt, case.path, ann_id="rois", rasterize=True, **kw)

    def _refused(self, case: SimpleNamespace, rois: dict, match: str) -> None:
        before = case.path.read_bytes()
        with pytest.raises(MEDH5ValidationError, match=match):
            self._import(case, rois)
        assert case.path.read_bytes() == before, "the sample is untouched"

    @pytest.mark.parametrize("shape", ["overlapping", "plus"])
    def test_S8_6_crossing_xor_contours_are_refused(self, case, shape):
        """Two exclusive-or squares overlapping by 3×3 voxels are 54 voxels,
        and were read as their union, 63.  A plus has no vertex inside the
        other bar, so only an edge test sees it cross."""
        grid = case.grid
        first, second = (
            {
                "overlapping": ((2, 7), (2, 7)),
                "plus": ((4, 5), (1, 10)),
            }[shape],
            {
                "overlapping": ((5, 10), (5, 10)),
                "plus": ((1, 10), (4, 5)),
            }[shape],
        )
        a, b = _square(grid, 2, *first), _square(grid, 2, *second)
        self._refused(
            case,
            {1: ("gtv", [(XOR, a), (XOR, b)])},
            r"ROI 'gtv' has CLOSEDPLANAR_XOR contours that cross on slice 2 ",
        )
        # The same outlines, CLOSED_PLANAR, are medh5's union, as they were.
        self._import(case, {1: ("gtv", [(CLOSED, a), (CLOSED, b)])})
        assert _voxels(case.path) == {"gtv": 63 if shape == "overlapping" else 36}

    def test_S8_6_nested_and_apart_xor_contours_import_as_their_exclusive_or(
        self, case
    ):
        grid = case.grid
        outer = _square(grid, 3, (2, 11), (2, 11))  # 100 voxels
        inner = _square(grid, 3, (5, 8), (5, 8))  # 16
        apart = _square(grid, 3, (13, 14), (13, 14))  # 4
        report = self._import(
            case, {1: ("gtv", [(XOR, inner), (XOR, outer), (XOR, apart)])}
        )
        assert _voxels(case.path) == {"gtv": 100 - 16 + 4}
        with medh5.open(case.path) as sample:
            roles = [p.role for p in sample.annotations["rois"].polygons()]
        assert roles == ["hole", "outer", "outer"]
        [note] = report.of_kind("contour_type")
        assert note.severity == "decision" and note.detail == {"rois": ["gtv"]}
        assert report.ok

    def test_S8_6_xor_mixed_with_another_type_is_refused(self, case):
        """PS3.3 C.8.8.6.1: an ROI with one exclusive-or contour is all
        exclusive-or."""
        outer = _square(case.grid, 3, (2, 11), (2, 11))
        inner = _square(case.grid, 3, (5, 8), (5, 8))
        self._refused(
            case,
            {1: ("gtv", [(XOR, outer), (CLOSED, inner)])},
            "ROI 'gtv' in .* mixes CLOSEDPLANAR_XOR contours with CLOSED_PLANAR",
        )

    def test_S8_6_open_rois_are_left_out_with_a_warning_naming_the_option(self, case):
        """Open curves are left out (``open_curves="omit"``); points are not
        (§8.5): a POINT ROI vanished, with a warning, until 2.1."""
        grid = case.grid
        organ = _square(grid, 2, (2, 7), (2, 7))
        line = grid.index_to_world(
            np.array([[2, 2.0, 2.0], [2, 2.0, 8.0], [2, 8.0, 8.0]])
        )
        path = grid.index_to_world(np.array([[k, 4.0 + k, 6.0] for k in range(6)]))
        point = grid.index_to_world(np.array([[2, 4.0, 4.0]]))
        report = self._import(
            case,
            {
                1: ("organ", [(CLOSED, organ)]),
                2: ("line", [("OPEN_PLANAR", line)]),
                3: ("applicator", [("OPEN_NONPLANAR", path)]),
                4: ("marker", [("POINT", point), ("POINT", point + 1.0)]),
            },
        )
        with medh5.open(case.path) as sample:
            assert [c.key for c in sample.label_set] == ["organ", "marker"]
            assert len(list(sample.annotations["rois"].polygons())) == 1
            assert len(sample.annotations["rois_points"]) == 2
        assert _voxels(case.path) == {"organ": 36}, "no class examined and empty"
        notes = report.of_kind("contour_type")
        assert all(n.severity == "warning" for n in notes)
        assert sorted(
            (n.detail["roi"], n.detail["type"], n.detail["contours"]) for n in notes
        ) == [
            ("applicator", "OPEN_NONPLANAR", 1),
            ("line", "OPEN_PLANAR", 1),
        ]
        assert all("open_curves='omit'" in n.message for n in notes)

    def test_S8_6_an_rtstruct_of_open_curves_alone_is_refused_by_name(self, case):
        line = case.grid.index_to_world(
            np.array([[2, 2.0, 2.0], [2, 2.0, 8.0], [2, 8.0, 8.0]])
        )
        self._refused(
            case,
            {2: ("line", [("OPEN_PLANAR", line)])},
            "contains no usable contours or points: 'line' is OPEN_PLANAR",
        )

    def test_S8_6_an_roi_of_areas_and_curves_is_refused(self, case):
        grid = case.grid
        organ = _square(grid, 2, (2, 7), (2, 7))
        line = grid.index_to_world(
            np.array([[3, 2.0, 2.0], [3, 2.0, 8.0], [3, 8.0, 8.0]])
        )
        self._refused(
            case,
            {1: ("organ", [(CLOSED, organ), ("OPEN_PLANAR", line)])},
            "ROI 'organ' in .* mixes CLOSED_PLANAR contours with OPEN_PLANAR ones",
        )

    def test_S8_6_a_contour_stating_no_type_is_closed_on_a_recorded_guess(self, case):
        organ = _square(case.grid, 2, (2, 7), (2, 7))
        report = self._import(case, {1: ("organ", [(None, organ)])})
        assert _voxels(case.path) == {"organ": 36}
        [note] = report.of_kind("contour_type")
        assert note.severity == "guess" and note.detail == {"rois": {"organ": 1}}

    def test_S8_6_a_contour_of_fewer_than_three_points_is_reported(self, case):
        grid = case.grid
        organ = _square(grid, 2, (2, 7), (2, 7))
        two = grid.index_to_world(np.array([[3, 2.0, 2.0], [3, 2.0, 8.0]]))
        report = self._import(case, {1: ("organ", [(CLOSED, organ), (CLOSED, two)])})
        assert _voxels(case.path) == {"organ": 36}
        [note] = report.of_kind("dropped_contours")
        assert note.severity == "warning"
        assert note.detail == {"roi": "organ", "contours": 1}

    @pytest.mark.parametrize(
        ("where", "match"),
        [
            ("between", r"0\.5 mm from slice 2 of grid .*beyond the 0\.1 mm tolerance"),
            ("across", r"runs from slice 0 to 5 of grid"),
            ("past", r"on slice 9 that lies outside grid .* \(6 slices of 16×16\)"),
            ("beside", r"on slice 2 that lies outside grid"),
        ],
    )
    def test_S8_6_a_contour_off_the_grids_slices_is_refused(self, case, where, match):
        """It was stored on the median vertex's slice, rounded: between two
        slices onto one of them, across six onto the middle one --- then
        rasterised to nothing --- and past the last slice as is."""
        index = {
            "between": [[2.5, 3, 3], [2.5, 3, 10], [2.5, 10, 10], [2.5, 10, 3]],
            "across": [[0, 3, 5], [5, 3, 5], [5, 12, 5], [0, 12, 5]],
            "past": [[9, 3, 3], [9, 3, 10], [9, 10, 10], [9, 10, 3]],
            "beside": [[2, 20, 20], [2, 20, 25], [2, 25, 25], [2, 25, 20]],
        }[where]
        organ = case.grid.index_to_world(np.array(index, float))
        self._refused(
            case,
            {1: ("organ", [(CLOSED, organ)])},
            f"ROI 'organ' has a contour .*{match}",
        )

    def test_S8_6_crossing_contours_of_one_class_are_a_warning_on_export(self, case):
        """medh5 means the union of two crossing outlines; a reader combining
        an ROI's contours by exclusive-or or even-odd fill cancels the
        overlap, and the export said nothing."""
        from medh5.io.rtstruct import to_rtstruct

        grid = case.grid
        self._import(
            case,
            {
                1: (
                    "gtv",
                    [
                        (CLOSED, _square(grid, 2, (2, 7), (2, 7))),
                        (CLOSED, _square(grid, 2, (5, 10), (5, 10))),
                        (CLOSED, _square(grid, 3, (2, 11), (2, 11))),
                        (CLOSED, _square(grid, 3, (5, 8), (5, 8))),
                    ],
                )
            },
        )
        report = ConversionReport()
        out = case.tmp / "out.dcm"
        to_rtstruct(case.path, "rois", case.series["paths"], out, report=report)
        [note] = report.of_kind("crossing_contours")
        assert note.severity == "warning"
        assert note.detail == {"class": "gtv", "planes": [[0, 2]]}, (
            "nested is no crossing"
        )
        assert out.exists(), "a warning, not a refusal"

    def test_S11_3_a_class_rasterised_to_nothing_is_a_warning(self, case):
        """Examined and absent is what an empty mask says; the contours say
        there is something there, smaller than a voxel."""
        grid = case.grid
        organ = _square(grid, 2, (2, 7), (2, 7))
        seed = grid.index_to_world(
            np.array([[2, 9.1, 9.1], [2, 9.1, 9.4], [2, 9.4, 9.4], [2, 9.4, 9.1]])
        )
        report = self._import(
            case, {1: ("organ", [(CLOSED, organ)]), 2: ("seed", [(CLOSED, seed)])}
        )
        assert _voxels(case.path) == {"organ": 36, "seed": 0}
        [note] = [n for n in report.of_kind("rasterization") if n.severity == "warning"]
        assert note.detail == {"classes": ["seed"]}


class TestRoiFrames:
    """Each ROI is in the frame of reference it states (PS3.3 C.8.8.5), and
    one stating none is in the one its file lists.  The first frame the file
    listed stood for every ROI, so the order of that list decided whether a
    structure set in two frames was refused, or imported whole onto a grid in
    one of them."""

    def _rois(self, grid: Any) -> dict:
        return {
            1: ("in_a", [(CLOSED, _square(grid, 2, (2, 7), (2, 7)))]),
            2: ("in_b", [(CLOSED, _square(grid, 3, (8, 13), (8, 13)))]),
        }

    def _import(self, case: SimpleNamespace, rt: Path) -> ConversionReport:
        from medh5.io.rtstruct import from_rtstruct

        return from_rtstruct(rt, case.path, ann_id="rois", rasterize=True)

    @pytest.mark.parametrize("listed", [[FRAME_A, FRAME_B], [FRAME_B, FRAME_A]])
    def test_S3_4_rois_in_two_frames_are_refused(self, tmp_path, listed):
        case = _dicom_case(tmp_path)
        rt = _structure_set(
            tmp_path / "rt.dcm",
            self._rois(case.grid),
            listed,
            roi_frames={1: FRAME_A, 2: FRAME_B},
        )
        before = case.path.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="2 frames of reference") as exc:
            self._import(case, rt)
        assert exc.value.code == "E414"
        assert f"{FRAME_A}: 'in_a'" in str(exc.value)
        assert f"{FRAME_B}: 'in_b'" in str(exc.value)
        assert case.path.read_bytes() == before

    def test_S3_4_rois_of_one_frame_import_from_a_file_listing_two(self, tmp_path):
        case = _dicom_case(tmp_path)
        rt = _structure_set(
            tmp_path / "rt.dcm",
            self._rois(case.grid),
            [FRAME_B, FRAME_A],
            roi_frames={1: FRAME_A, 2: FRAME_A},
        )
        report = self._import(case, rt)
        assert _voxels(case.path) == {"in_a": 36, "in_b": 36}
        with medh5.open(case.path) as sample:
            assert sample.annotations["rois"].header.frame_uid == FRAME_A
        assert report.ok and not report.of_kind("frame"), "the frames were compared"

    def test_S3_4_an_roi_naming_another_frame_than_its_file_is_refused(self, tmp_path):
        """The file lists frame A and its only ROI says B: the ROI's frame is
        compared with the grid's, where the file's was."""
        case = _dicom_case(tmp_path)
        rois = {2: self._rois(case.grid)[2]}
        rt = _structure_set(
            tmp_path / "rt.dcm", rois, [FRAME_A], roi_frames={2: FRAME_B}
        )
        before = case.path.read_bytes()
        with pytest.raises(
            MEDH5ValidationError, match=f"drawn in frame of reference '{FRAME_B}'"
        ) as exc:
            self._import(case, rt)
        assert exc.value.code == "E414"
        assert case.path.read_bytes() == before

    def test_S3_4_the_roi_frame_is_used_when_the_file_lists_none(self, tmp_path):
        in_b = _dicom_case(tmp_path / "b", FRAME_B)
        in_a = _dicom_case(tmp_path / "a", FRAME_A)
        rois = {2: self._rois(in_b.grid)[2]}
        rt = _structure_set(tmp_path / "rt.dcm", rois, [], roi_frames={2: FRAME_B})
        report = self._import(in_b, rt)
        assert _voxels(in_b.path) == {"in_b": 36}
        [note] = report.of_kind("frame")
        assert note.severity == "warning" and note.detail["rois"] == ["in_b"]
        with pytest.raises(MEDH5ValidationError, match="registration") as exc:
            self._import(in_a, rt)
        assert exc.value.code == "E414"

    def test_S3_4_an_roi_stating_no_frame_in_a_file_listing_two_is_refused(
        self, tmp_path
    ):
        case = _dicom_case(tmp_path)
        rois = {1: self._rois(case.grid)[1]}
        rt = _structure_set(
            tmp_path / "rt.dcm", rois, [FRAME_A, FRAME_B], roi_frames={1: None}
        )
        with pytest.raises(
            MEDH5ValidationError, match="state no frame of reference"
        ) as exc:
            self._import(case, rt)
        assert exc.value.code == "E414"


class TestPatientCoordinates:
    """``ContourData`` is DICOM patient coordinates: millimetres, LPS, and the
    one frame of reference the source images share.

    The exporter wrote each contour as stored --- a metre grid's 0.003 for
    3 mm, an RAS grid's x and y unflipped --- and filed it under the first
    source image's frame whatever the grid's.
    """

    # A square on slice 2, in the grid's index space.
    SQUARE = np.array(
        [[2.0, 4.0, 4.0], [2.0, 4.0, 10.0], [2.0, 10.0, 10.0], [2.0, 10.0, 4.0]]
    )

    @pytest.fixture
    def series(self, tmp_path: Path) -> dict[str, Any]:
        from pydicom.uid import generate_uid

        return write_dicom_series(
            tmp_path / "dcm",
            patient_id="PSEUDO-009",
            study_uid=generate_uid(),
            study_date="20260101",
        )

    def _sample(
        self,
        tmp_path: Path,
        *,
        frame_uid: str | None,
        units: str = "mm",
        coord_system: str = "LPS",
        space: str = "index",
        grid: bool = True,
    ) -> Path:
        from medh5.annotations.geometric import Polygon
        from medh5.labels import LabelClass, LabelSet

        spacing = (2.5, 0.8, 0.9) if units == "mm" else (0.0025, 0.0008, 0.0009)
        origin = (-10.0, -20.0, 30.0) if units == "mm" else (-0.01, -0.02, 0.03)
        path = tmp_path / "contours.medh5"
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.label_set(LabelSet("rt", [LabelClass(1, "liver", "Liver")]))
            declared = w.add_grid(
                "g",
                shape=(6, 16, 20),
                spacing=spacing,
                origin=origin,
                coord_system=coord_system,
                units=units,
                frame_uid=frame_uid,
            )
            w.add_image("CT", np.zeros((6, 16, 20), np.int16), grid="g", modality="CT")
            vertices = (
                self.SQUARE
                if space == "index"
                else declared.index_to_world(self.SQUARE)
            )
            w.add_contours(
                "rois",
                [Polygon(vertices.astype(np.float32), 1, plane=(0, 2))],
                grid="g" if grid else None,
                space=space,
                frame_uid=None if grid else frame_uid,
            )
        return path

    def _exported(self, path: Path, sources: list[str], out: Path, **kw: Any):
        from medh5.io.rtstruct import read_rtstruct, to_rtstruct

        report = ConversionReport(converter="to-rtstruct")
        to_rtstruct(path, "rois", sources, out, report=report, **kw)
        contours, meta = read_rtstruct(out)
        return contours[1][0], meta, report

    def _expected_mm(self, path: Path) -> np.ndarray:
        """The square in the grid's world coordinates, as stored."""
        with medh5.open(path) as sample:
            stored = sample.annotations["rois"]
            grid = sample.grids["g"]
            vertices = np.asarray(stored.polygon(0), dtype=np.float64)
            return (
                vertices if stored.space == "world" else grid.index_to_world(vertices)
            )

    def test_S8_6_an_lps_millimetre_grid_is_written_as_stored(self, tmp_path, series):
        path = self._sample(tmp_path, frame_uid=series["frame_uid"])
        points, meta, report = self._exported(
            path, series["paths"], tmp_path / "rt.dcm"
        )
        assert np.allclose(points, self._expected_mm(path), atol=1e-4)
        assert meta["frame_uid"] == series["frame_uid"]
        assert not report.of_kind("frame_of_reference"), "the frames were compared"

    @pytest.mark.parametrize("space", ["index", "world"])
    def test_S8_6_a_metre_grid_is_written_in_millimetres(self, tmp_path, series, space):
        path = self._sample(
            tmp_path, frame_uid=series["frame_uid"], units="m", space=space
        )
        points, _, _ = self._exported(path, series["paths"], tmp_path / "rt.dcm")
        stored = self._expected_mm(path)
        assert np.allclose(points, 1000.0 * stored, atol=1e-3)
        # Slice 2 at 2.5 mm spacing from -10 mm: the slice axis is world x.
        assert np.allclose(points[:, 0], -5.0, atol=1e-3)

    @pytest.mark.parametrize("space", ["index", "world"])
    def test_S8_6_an_ras_grid_is_flipped_into_lps(self, tmp_path, series, space):
        path = self._sample(
            tmp_path, frame_uid=series["frame_uid"], coord_system="RAS", space=space
        )
        points, _, _ = self._exported(path, series["paths"], tmp_path / "rt.dcm")
        stored = self._expected_mm(path)
        assert np.allclose(points, stored * np.array([-1.0, -1.0, 1.0]), atol=1e-4)

    def test_S3_4_contours_in_another_frame_are_refused(self, tmp_path, series):
        path = self._sample(tmp_path, frame_uid="1.2.826.0.1.3680043.2.1125.9")
        out = tmp_path / "rt.dcm"
        with pytest.raises(MEDH5ValidationError, match="registration") as exc:
            self._exported(path, series["paths"], out)
        assert exc.value.code == "E414"
        assert "frame_salt" not in str(exc.value), "no scrub wrote this frame"
        assert not out.exists(), "nothing is filed under the wrong frame"

    def test_S3_4_world_contours_are_judged_by_their_own_frame(self, tmp_path, series):
        """World coordinates live in the annotation's frame: a gridless
        annotation in another frame is refused like a grid's would be."""
        path = self._sample(
            tmp_path, frame_uid="1.2.826.0.1.3680043.2.1125.9", space="world"
        )
        with medh5.open(path) as sample:
            assert sample.annotations["rois"].header.frame_uid.endswith(".9")
        with pytest.raises(MEDH5ValidationError, match="registration") as exc:
            self._exported(path, series["paths"], tmp_path / "rt.dcm")
        assert exc.value.code == "E414"

    def test_S3_4_a_pseudonymised_frame_is_compared_through_its_salt(
        self, tmp_path, series
    ):
        """Refused without its salt, the export said nothing of one, where the
        SEG paths name it."""
        from medh5.curation.scrub import pseudonymise

        hidden = pseudonymise(series["frame_uid"], "s3cret")
        path = self._sample(tmp_path, frame_uid=hidden)
        with pytest.raises(MEDH5ValidationError) as exc:
            self._exported(path, series["paths"], tmp_path / "a.dcm")
        assert exc.value.code == "E414"
        assert "frame_salt= (`--frame-salt` on the command line)" in str(exc.value)
        with pytest.raises(MEDH5ValidationError) as exc:
            self._exported(path, series["paths"], tmp_path / "a.dcm", frame_salt="no")
        assert "frame_salt=" not in str(exc.value), "a salt was given: the wrong one"
        points, meta, _ = self._exported(
            path, series["paths"], tmp_path / "b.dcm", frame_salt="s3cret"
        )
        assert meta["frame_uid"] == series["frame_uid"], "filed under the real frame"
        assert np.allclose(points, self._expected_mm(path), atol=1e-4)

    def test_S3_4_a_frameless_grid_is_a_recorded_guess(self, tmp_path, series):
        path = self._sample(tmp_path, frame_uid=None)
        _, meta, report = self._exported(path, series["paths"], tmp_path / "rt.dcm")
        assert meta["frame_uid"] == series["frame_uid"]
        assert report.of_kind("frame_of_reference")

    def test_S3_4_sources_in_two_frames_are_refused(self, tmp_path, series):
        from pydicom.uid import generate_uid

        other = write_dicom_series(
            tmp_path / "other",
            patient_id="PSEUDO-009",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        path = self._sample(tmp_path, frame_uid=series["frame_uid"])
        sources = [*series["paths"], other["paths"][0]]
        with pytest.raises(MEDH5ValidationError, match="frames of reference") as exc:
            self._exported(path, sources, tmp_path / "rt.dcm")
        assert exc.value.code == "E414"

    def test_S3_4_a_source_without_a_frame_is_refused(self, tmp_path, series):
        import pydicom

        bare = tmp_path / "bare.dcm"
        image = pydicom.dcmread(series["paths"][0])
        del image.FrameOfReferenceUID
        image.save_as(str(bare), enforce_file_format=True)
        path = self._sample(tmp_path, frame_uid=series["frame_uid"])
        with pytest.raises(MEDH5ValidationError, match="none") as exc:
            self._exported(path, [*series["paths"], str(bare)], tmp_path / "rt.dcm")
        assert exc.value.code == "E414"

    @pytest.mark.parametrize(
        ("units", "coord_system"), [("px", "LPS"), ("mm", "custom")]
    )
    def test_S8_6_a_grid_without_patient_coordinates_is_refused(
        self, tmp_path, series, units, coord_system
    ):
        path = self._sample(
            tmp_path,
            frame_uid=series["frame_uid"],
            units=units,
            coord_system=coord_system,
        )
        with pytest.raises(MEDH5ValidationError, match="grid 'g' is in") as exc:
            self._exported(path, series["paths"], tmp_path / "rt.dcm")
        assert exc.value.code == "E414"

    def test_S8_6_gridless_contours_are_refused(self, tmp_path, series):
        """World coordinates are in their grid's units (§8); with no grid there
        is nothing to scale them by."""
        path = self._sample(
            tmp_path, frame_uid=series["frame_uid"], space="world", grid=False
        )
        with pytest.raises(MEDH5ValidationError, match="names no grid") as exc:
            self._exported(path, series["paths"], tmp_path / "rt.dcm")
        assert exc.value.code == "E414"

    def _imported(self, path: Path, rt: Path, **kw: Any) -> tuple[Any, Any]:
        """Import *rt* into *path* as ``back``: its world vertices, its frame
        and the report."""
        from medh5.io.rtstruct import from_rtstruct

        report = from_rtstruct(rt, path, ann_id="back", **kw)
        with medh5.open(path) as sample:
            back = sample.annotations["back"]
            vertices = np.asarray(back.polygon(0), dtype=np.float64)
            return (vertices, back.header.frame_uid), report

    @pytest.mark.parametrize(
        ("units", "coord_system"), [("mm", "LPS"), ("mm", "RAS"), ("m", "LPS")]
    )
    def test_S8_6_imported_contours_are_in_their_grids_world_coordinates(
        self, tmp_path, series, units, coord_system
    ):
        """`ContourData` is LPS millimetres: an import scales it to the grid's
        units and flips it for an RAS grid, so a round trip lands where it
        started.  Stored as read, the RAS square came back mirrored off the
        grid, and the metre one read its millimetres as metres."""
        path = self._sample(
            tmp_path,
            frame_uid=series["frame_uid"],
            units=units,
            coord_system=coord_system,
        )
        rt = tmp_path / "rt.dcm"
        self._exported(path, series["paths"], rt)
        (vertices, frame), report = self._imported(path, rt)
        assert np.allclose(vertices, self._expected_mm(path), rtol=1e-5, atol=1e-6)
        with medh5.open(path) as sample:
            index = sample.grids["g"].world_to_index(vertices)
        assert np.allclose(index, self.SQUARE, atol=1e-3)
        assert frame == series["frame_uid"]
        assert not report.guesses, "the frames were compared"

    @pytest.mark.parametrize(
        ("units", "coord_system"), [("px", "LPS"), ("mm", "custom")]
    )
    def test_S8_6_an_import_onto_a_grid_without_patient_coordinates_is_refused(
        self, tmp_path, series, units, coord_system
    ):
        source = self._sample(tmp_path, frame_uid=series["frame_uid"])
        rt = tmp_path / "rt.dcm"
        self._exported(source, series["paths"], rt)
        (tmp_path / "other").mkdir()
        path = self._sample(
            tmp_path / "other",
            frame_uid=series["frame_uid"],
            units=units,
            coord_system=coord_system,
        )
        with pytest.raises(MEDH5ValidationError, match="grid 'g' is in") as exc:
            self._imported(path, rt)
        assert exc.value.code == "E414"
        with medh5.open(path) as sample:
            assert "back" not in sample.annotations

    @pytest.mark.parametrize("named", [False, True])
    def test_S3_4_an_rtstruct_in_another_frame_is_refused(
        self, tmp_path, series, named
    ):
        """The sample's only grid, or one named with `grid=`, in another known
        frame took the contours with a guess at most."""
        source = self._sample(tmp_path, frame_uid=series["frame_uid"])
        rt = tmp_path / "rt.dcm"
        self._exported(source, series["paths"], rt)
        (tmp_path / "other").mkdir()
        path = self._sample(
            tmp_path / "other", frame_uid="1.2.826.0.1.3680043.2.1125.9"
        )
        with pytest.raises(MEDH5ValidationError, match="registration") as exc:
            self._imported(path, rt, **({"grid": "g"} if named else {}))
        assert exc.value.code == "E414"
        with medh5.open(path) as sample:
            assert "back" not in sample.annotations

    def test_S3_4_an_import_compares_a_pseudonymised_frame_through_its_salt(
        self, tmp_path, series
    ):
        """The contours take the grid's pseudonym as their frame: the
        RTSTRUCT's own UID is what the scrub removed."""
        from medh5.curation.scrub import pseudonymise

        hidden = pseudonymise(series["frame_uid"], "s3cret")
        path = self._sample(tmp_path, frame_uid=hidden)
        rt = tmp_path / "rt.dcm"
        self._exported(path, series["paths"], rt, frame_salt="s3cret")
        with pytest.raises(MEDH5ValidationError, match="frame_salt") as exc:
            self._imported(path, rt)
        assert exc.value.code == "E414"
        assert "--frame-salt" in str(exc.value), "the command line's name too"
        (vertices, frame), report = self._imported(path, rt, frame_salt="s3cret")
        assert frame == hidden
        assert np.allclose(vertices, self._expected_mm(path), atol=1e-4)
        assert not report.guesses

    def test_S3_4_a_frameless_grid_takes_the_contours_on_a_recorded_guess(
        self, tmp_path, series
    ):
        path = self._sample(tmp_path, frame_uid=None)
        rt = tmp_path / "rt.dcm"
        self._exported(path, series["paths"], rt)
        (vertices, frame), report = self._imported(path, rt)
        assert np.allclose(vertices, self._expected_mm(path), atol=1e-4)
        assert frame == series["frame_uid"], "the frame the coordinates are in"
        assert report.of_kind("grid") and report.of_kind("frame")

    def test_S3_4_the_command_line_names_the_frame_salt_and_takes_it(
        self, tmp_path, series, capsys
    ):
        from medh5.cli import EXIT_ERROR, EXIT_OK, main
        from medh5.curation.scrub import pseudonymise

        path = self._sample(
            tmp_path, frame_uid=pseudonymise(series["frame_uid"], "s3cret")
        )
        argv = ["convert", "to-rtstruct", str(path), "rois"]
        sources = [arg for p in series["paths"] for arg in ("--source", p)]
        assert main([*argv, str(tmp_path / "a.dcm"), *sources]) == EXIT_ERROR
        err = capsys.readouterr().err
        assert err.startswith("medh5: [E414]") and "--frame-salt" in err, err
        out = tmp_path / "b.dcm"
        salted = [*argv, str(out), *sources, "--frame-salt", "s3cret"]
        assert main(salted) == EXIT_OK
        assert out.exists()


class TestRoiNamesAndText:
    """An export names each ROI for its class's display name, colours it as
    the label set does, and states the character set of text beyond ASCII.
    The key and red were written, and a name with an umlaut went out in
    Latin-1 under the default repertoire, which is ASCII."""

    @pytest.fixture
    def series(self, tmp_path: Path) -> dict[str, Any]:
        from pydicom.uid import generate_uid

        return write_dicom_series(
            tmp_path / "dcm",
            patient_id="PSEUDO-010",
            study_uid=generate_uid(),
            study_date="20260101",
        )

    def _sample(self, path: Path, series: dict[str, Any], classes: list) -> Path:
        from medh5.annotations.geometric import Polygon
        from medh5.labels import LabelSet

        square = np.array(
            [[2.0, 4.0, 4.0], [2.0, 4.0, 10.0], [2.0, 10.0, 10.0], [2.0, 10.0, 4.0]]
        )
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.label_set(LabelSet("rt", classes))
            w.add_grid(
                "g",
                shape=(6, 16, 20),
                spacing=(2.5, 0.8, 0.9),
                origin=(-10.0, -20.0, 30.0),
                frame_uid=series["frame_uid"],
            )
            w.add_image("CT", np.zeros((6, 16, 20), np.int16), grid="g", modality="CT")
            w.add_contours(
                "rois",
                [
                    Polygon(
                        (square + [k, 0.0, 0.0]).astype(np.float32), c.id, (0, 2 + k)
                    )
                    for k, c in enumerate(classes)
                ],
                grid="g",
                space="index",
            )
        return path

    def test_S8_6_rois_take_the_label_sets_names_and_colours(self, tmp_path, series):
        """And come back into that label set by display name, which no key
        matched: the import refused the export (E402)."""
        from medh5.io.rtstruct import from_rtstruct, to_rtstruct
        from medh5.labels import LabelClass

        path = self._sample(
            tmp_path / "s.medh5",
            series,
            [
                LabelClass(1, "kidney_left", "Left kidney", color=(10, 200, 30, 255)),
                LabelClass(2, "liver", "Liver"),
            ],
        )
        out = to_rtstruct(path, "rois", series["paths"], tmp_path / "rt.dcm")
        written = pydicom.dcmread(out)
        rois = [r.ROIName for r in written.StructureSetROISequence]
        assert rois == ["Left kidney", "Liver"]
        colours = [list(c.ROIDisplayColor) for c in written.ROIContourSequence]
        assert colours == [[10, 200, 30], [255, 0, 0]], "red where none is given"
        assert "SpecificCharacterSet" not in written, "ASCII states none, as before"

        from_rtstruct(out, path, ann_id="back")
        with medh5.open(path) as sample:
            back = sample.annotations["back"]
            keys = sorted(back.class_key(p.class_id) for p in back.polygons())
        assert keys == ["kidney_left", "liver"]

    def test_S8_6_rois_are_named_for_keys_where_display_names_repeat(
        self, tmp_path, series
    ):
        from medh5.io.rtstruct import to_rtstruct
        from medh5.labels import LabelClass

        path = self._sample(
            tmp_path / "s.medh5",
            series,
            [LabelClass(1, "lesion_1", "Lesion"), LabelClass(2, "lesion_2", "Lesion")],
        )
        report = ConversionReport()
        out = to_rtstruct(
            path, "rois", series["paths"], tmp_path / "rt.dcm", report=report
        )
        rois = [r.ROIName for r in pydicom.dcmread(out).StructureSetROISequence]
        assert rois == ["lesion_1", "lesion_2"]
        [note] = report.of_kind("roi_names")
        assert note.severity == "decision" and "not distinct" in note.message

    def test_S8_6_text_beyond_ascii_is_written_as_utf8(self, tmp_path, series):
        from medh5.io.rtstruct import to_rtstruct
        from medh5.labels import LabelClass

        sources = []
        for k, source in enumerate(series["paths"]):
            image = pydicom.dcmread(source)
            image.SpecificCharacterSet = "ISO_IR 100"
            image.PatientName = "Müller^Jürgen"
            sources.append(str(tmp_path / f"latin1_{k}.dcm"))
            image.save_as(sources[-1], enforce_file_format=True)
        path = self._sample(
            tmp_path / "s.medh5", series, [LabelClass(1, "spinal_cord", "Rückenmark")]
        )
        out = to_rtstruct(path, "rois", sources, tmp_path / "rt.dcm")
        raw = out.read_bytes()
        assert "Müller".encode() in raw and "Rückenmark".encode() in raw
        assert "Müller".encode("latin-1") not in raw
        written = pydicom.dcmread(out)
        assert written.SpecificCharacterSet == "ISO_IR 192"
        assert str(written.PatientName) == "Müller^Jürgen"
        assert written.StructureSetROISequence[0].ROIName == "Rückenmark"


def _bladder_label_set(*keys: str) -> Any:
    from medh5.labels import LabelClass, LabelSet

    return LabelSet(
        "bladder-1",
        version="1.0.0",
        classes=[LabelClass(i, key, key.title()) for i, key in enumerate(keys, 1)],
    )


class TestSameNamedRois:
    """ROIs whose names make one key are one class (§8.6).

    Minting, two ROIs named "GTV" failed "[E302] duplicate class keys"; with
    a label set they merged without a note, their roles nested per ROI while
    the rasteriser nested per class --- so a "Bladder" drawn inside another
    became a hole in it.
    """

    @pytest.fixture(params=["minted", "label_set"])
    def given(self, request) -> Any:
        """No label set (the importer mints one), or one the sample holds."""
        if request.param == "minted":
            return None
        return _bladder_label_set("gtv", "bladder", "bladder_inner")

    def test_S8_6_same_named_rois_are_one_class(self, tmp_path, given):
        from medh5.io.rtstruct import from_rtstruct

        case = _dicom_case(tmp_path)
        if given is not None:
            with medh5.amend(case.path) as w:
                w.label_set(given)
        grid = case.grid
        rt = _structure_set(
            case.tmp / "rt.dcm",
            {
                1: ("GTV", [(CLOSED, _square(grid, 2, (2, 5), (2, 5)))]),
                3: ("gtv", [(CLOSED, _square(grid, 2, (8, 10), (8, 10)))]),
                4: ("GTV", [(CLOSED, _square(grid, 3, (2, 7), (2, 7)))]),
            },
            [FRAME_A],
        )
        report = from_rtstruct(rt, case.path, ann_id="rois", rasterize=True)
        with medh5.open(case.path) as sample:
            polygons = list(sample.annotations["rois"].polygons())
            gtv = sample.label_set["gtv"]
            assert {p.class_id for p in polygons} == {gtv.id}
            assert [p.role for p in polygons] == ["outer"] * 3
            if given is None:
                assert [c.key for c in sample.label_set] == ["gtv"]
                assert gtv.name == "GTV"
        assert _voxels(case.path) == {"gtv": 16 + 9 + 36}
        [note] = report.of_kind("merged_rois")
        assert note.severity == "decision"
        assert note.detail == {
            "class": "gtv",
            "rois": [1, 3, 4],
            "names": ["GTV", "gtv", "GTV"],
        }

    def test_S8_6_nested_same_named_rois_are_refused(self, tmp_path, given):
        """The Bladder kit: drawn, 121 voxels; 2.0.2 stored 105."""
        from medh5.io.rtstruct import from_rtstruct

        kit = NestedSameNameRoiKit
        sample = kit.sample(tmp_path / "ct.medh5", label_set=given)
        rt = kit.rtstruct(tmp_path / "rt.dcm", sample)
        before = sample.read_bytes()
        with pytest.raises(
            MEDH5ValidationError,
            match=r"ROIs 1 \('Bladder'\), 2 \('Bladder'\) are one class 'bladder', "
            r"and on slice 0 of grid 'ct' .*\(--roi NUMBER=KEY\)",
        ):
            from_rtstruct(rt, sample, rasterize=True)
        assert sample.read_bytes() == before, "the sample is untouched"

    def test_S8_6_roi_classes_import_rois_apart(self, tmp_path, given):
        from medh5.io.rtstruct import from_rtstruct

        kit = NestedSameNameRoiKit
        sample = kit.sample(tmp_path / "ct.medh5", label_set=given)
        rt = kit.rtstruct(tmp_path / "rt.dcm", sample)
        report = from_rtstruct(
            rt, sample, rasterize=True, roi_classes={2: "bladder_inner"}
        )
        assert _voxels(sample, "contours") == {
            "bladder": kit.DRAWN,
            "bladder_inner": 16,
        }
        assert not report.of_kind("merged_rois")
        with medh5.open(sample) as opened:
            assert opened.annotations["contours"].roles == ("outer", "outer")
            if given is None:
                assert [(c.key, c.name) for c in opened.label_set] == [
                    ("bladder", "Bladder"),
                    ("bladder_inner", "Bladder"),
                ]

    def test_S8_6_roi_classes_name_a_class_of_the_label_set(self, tmp_path):
        """With a label set, the key must be one of its classes (E402); an ROI
        number the file does not hold is refused, not ignored."""
        from medh5.io.rtstruct import from_rtstruct

        kit = NestedSameNameRoiKit
        sample = kit.sample(
            tmp_path / "ct.medh5", label_set=_bladder_label_set("bladder")
        )
        rt = kit.rtstruct(tmp_path / "rt.dcm", sample)
        with pytest.raises(MEDH5ValidationError, match="'bladder_inner'") as exc:
            from_rtstruct(rt, sample, roi_classes={2: "bladder_inner"})
        assert exc.value.code == "E402"
        with pytest.raises(MEDH5ValidationError, match=r"ROI number\(s\) \[7\]"):
            from_rtstruct(rt, sample, roi_classes={7: "bladder"})

    def test_roi_classes_come_in_through_the_command_line(self, tmp_path, capsys):
        from medh5.cli import main

        kit = NestedSameNameRoiKit
        sample = kit.sample(tmp_path / "ct.medh5")
        rt = kit.rtstruct(tmp_path / "rt.dcm", sample)
        command = ["convert", "from-rtstruct", str(rt), str(sample), "--rasterize"]
        assert main(command) == 1, "refused without --roi"
        assert "--roi NUMBER=KEY" in capsys.readouterr().err
        assert main([*command, "--roi", "two=x"]) == 1
        assert "--roi" in capsys.readouterr().err
        assert main([*command, "--roi", "2=bladder_inner"]) == 0
        assert _voxels(sample, "contours") == {"bladder": 121, "bladder_inner": 16}


class TestPointRois:
    """An RTSTRUCT's POINT ROIs are a `points` annotation (§8.5): landmarks,
    in world coordinates of the contours' frame.  They were left out with a
    warning."""

    @pytest.fixture
    def case(self, tmp_path: Path) -> SimpleNamespace:
        return _dicom_case(tmp_path)

    @staticmethod
    def _rois(grid: Any) -> dict:
        organ = _square(grid, 2, (2, 7), (2, 7))
        first = grid.index_to_world(np.array([[2, 4.0, 4.0]]))
        second = grid.index_to_world(np.array([[3, 9.25, 11.5]]))
        return {
            1: ("organ", [(CLOSED, organ)]),
            4: ("marker", [("POINT", first), ("POINT", second)]),
        }

    def _import(self, case: SimpleNamespace, rois: dict, **kw: Any):
        from medh5.io.rtstruct import from_rtstruct

        rt = _structure_set(case.tmp / "rt.dcm", rois, [FRAME_A])
        return from_rtstruct(rt, case.path, ann_id="rois", **kw)

    def test_S8_5_rtstruct_point_rois_import_as_points(self, case):
        rois = self._rois(case.grid)
        report = self._import(case, rois)
        expected = np.concatenate([points for _, points in rois[4][1]])
        with medh5.open(case.path) as sample:
            points = sample.annotations["rois_points"]
            marker = sample.label_set["marker"]
            assert points.kind == "points" and points.space == "world"
            assert points.header.frame_uid == sample.grids[points.grid_id].frame_uid
            assert points.grid_id == sample.annotations["rois"].grid_id
            assert points.task == "other", "a marker is no detection finding"
            assert points.object_class_ids.tolist() == [marker.id, marker.id]
            assert points.names == ("marker", "marker")
            assert np.allclose(points.points, expected, atol=1e-4)
            assert [c.key for c in points.annotated_classes] == ["marker"]
            contours = sample.annotations["rois"]
            assert [c.key for c in contours.annotated_classes] == ["organ"]
        assert not report.of_kind("contour_type"), "no warning"
        [note] = report.of_kind("points")
        assert note.severity == "decision"
        assert note.detail["rois"] == {"4": "marker"}
        assert validate_file(case.path).ok

    def test_S8_5_point_only_structure_set_writes_no_contours(self, case):
        rois = {4: self._rois(case.grid)[4]}
        report = self._import(case, rois, points_id="landmarks", rasterize=True)
        with medh5.open(case.path) as sample:
            assert sorted(sample.annotations) == ["landmarks"]
            assert len(sample.annotations["landmarks"]) == 2
        [note] = report.of_kind("rasterization")
        assert note.severity == "warning"

    def test_S8_5_a_point_contour_of_several_points_is_refused(self, case):
        grid = case.grid
        two = grid.index_to_world(np.array([[2, 4.0, 4.0], [2, 5.0, 5.0]]))
        before = case.path.read_bytes()
        with pytest.raises(
            MEDH5ValidationError,
            match=r"ROI 'marker' has a POINT contour of 2 points",
        ):
            self._import(case, {4: ("marker", [("POINT", two)])})
        assert case.path.read_bytes() == before

    def test_S8_5_open_curves_take_only_omit(self, case):
        with pytest.raises(MEDH5ValidationError, match="2.2.0"):
            self._import(case, self._rois(case.grid), open_curves="polylines")

    def test_S8_5_annotated_classes_split_across_points_and_contours(self, case):
        rois = self._rois(case.grid)
        rois[5] = ("seed", [("POINT", rois[4][1][0][1] + 2.0)])
        self._import(case, rois, annotated_classes=["organ", "marker"])
        with medh5.open(case.path) as sample:
            seen = {
                ann: [c.key for c in sample.annotations[ann].annotated_classes]
                for ann in ("rois", "rois_points")
            }
        assert seen == {"rois": ["organ"], "rois_points": ["marker"]}

    def test_S8_5_all_applies_to_the_contours_only(self, case):
        self._import(case, self._rois(case.grid), annotated_classes="all")
        with medh5.open(case.path) as sample:
            contours = sample.annotations["rois"].annotated_classes
            points = sample.annotations["rois_points"].annotated_classes
            assert [c.key for c in contours] == ["organ", "marker"]
            assert [c.key for c in points] == ["marker"]

    def test_S8_5_points_round_trip_through_to_rtstruct(self, case):
        from medh5.io.rtstruct import read_rtstruct, to_rtstruct

        rois = self._rois(case.grid)
        self._import(case, rois)
        out = case.tmp / "out.dcm"
        report = ConversionReport()
        to_rtstruct(
            case.path,
            "rois",
            case.series["paths"],
            out,
            points="rois_points",
            report=report,
        )
        written = pydicom.dcmread(out)
        names = {r.ROINumber: r.ROIName for r in written.StructureSetROISequence}
        kinds = {
            o.ReferencedROINumber: o.RTROIInterpretedType
            for o in written.RTROIObservationsSequence
        }
        marker = next(n for n, name in names.items() if name == "marker")
        organ = next(n for n, name in names.items() if name == "organ")
        assert kinds == {organ: "ORGAN", marker: "MARKER"}
        [item] = [
            i for i in written.ROIContourSequence if i.ReferencedROINumber == marker
        ]
        assert [c.ContourGeometricType for c in item.ContourSequence] == ["POINT"] * 2
        assert [int(c.NumberOfContourPoints) for c in item.ContourSequence] == [1, 1]
        found = np.array(
            [list(map(float, c.ContourData)) for c in item.ContourSequence]
        )
        expected = np.concatenate([points for _, points in rois[4][1]])
        assert np.allclose(found, expected.astype(np.float32), rtol=0, atol=1e-4)
        _, meta = read_rtstruct(out)
        assert [len(p) for p in meta["points"][marker]] == [1, 1]

    def test_S8_5_a_class_in_both_annotations_is_refused_on_export(self, case):
        from medh5.io.rtstruct import to_rtstruct

        grid = case.grid
        rois = self._rois(grid)
        rois[6] = ("marker", [(CLOSED, _square(grid, 4, (2, 3), (2, 3)))])
        self._import(case, rois)
        with pytest.raises(MEDH5ValidationError, match="'marker'"):
            to_rtstruct(
                case.path,
                "rois",
                case.series["paths"],
                case.tmp / "out.dcm",
                points="rois_points",
            )

    def test_points_come_in_and_go_out_through_the_command_line(self, case):
        from medh5.cli import main

        rt = _structure_set(case.tmp / "rt.dcm", self._rois(case.grid), [FRAME_A])
        assert (
            main(
                [
                    "convert",
                    "from-rtstruct",
                    str(rt),
                    str(case.path),
                    "--points-id",
                    "landmarks",
                    "--open-curves",
                    "omit",
                ]
            )
            == 0
        )
        out = case.tmp / "out.dcm"
        sources = [arg for p in case.series["paths"] for arg in ("--source", str(p))]
        command = ["convert", "to-rtstruct", str(case.path), "contours", str(out)]
        assert main([*command, *sources, "--points", "landmarks"]) == 0
        written = pydicom.dcmread(out)
        assert {o.RTROIInterpretedType for o in written.RTROIObservationsSequence} == {
            "ORGAN",
            "MARKER",
        }
