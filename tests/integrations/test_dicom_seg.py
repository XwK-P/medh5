"""DICOM SEG import and export (``medh5.io.dicom_seg``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from tests.helpers import write_dicom_series
from tests.kits import Organs

pydicom = pytest.importorskip("pydicom")


@pytest.fixture
def prepared(tmp_path: Path) -> dict[str, Any]:
    """A one-series sample plus two overlapping masks on its grid."""
    from pydicom.uid import generate_uid

    from medh5.io.dicom import from_dicom
    from tests.helpers import write_dicom_series

    root = tmp_path / "dcm"
    series = write_dicom_series(
        root,
        patient_id="PSEUDO-002",
        study_uid=generate_uid(),
        study_date="20260101",
    )
    path = tmp_path / "case.medh5"
    from_dicom(root, path, group_by="study")
    with medh5.open(path) as sample:
        grid_id = sorted(sample.grids)[0]
        shape = sample.grids[grid_id].spatial_shape
    liver = np.zeros(shape, bool)
    liver[1:5, 2:12, 3:16] = True
    lesion = np.zeros(shape, bool)
    lesion[2:4, 5:8, 6:10] = True  # entirely inside the liver
    with medh5.amend(path) as writer:
        writer.label_set(
            LabelSet(
                "d",
                version="1.0.0",
                classes=[
                    LabelClass(1, "liver", "Liver"),
                    LabelClass(3, "lesion", "Lesion"),
                ],
            )
        )
        writer.add_segmentation("organs", grid=grid_id, masks={1: liver, 3: lesion})
    return {
        "path": path,
        "series": series,
        "liver": liver,
        "lesion": lesion,
        "grid": grid_id,
    }


class TestDicomSeg:
    def test_S7_overlapping_segments_survive_the_round_trip(self, prepared, tmp_path):
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        report = from_dicom_seg(out, prepared["path"], ann_id="imported")
        with medh5.open(prepared["path"]) as sample:
            imported = sample.annotations["imported"]
            assert np.array_equal(imported.dense(["liver"])[0], prepared["liver"])
            assert np.array_equal(imported.dense(["lesion"])[0], prepared["lesion"])
            overlap = imported.dense(["liver"])[0] & imported.dense(["lesion"])[0]
            assert overlap.any(), "the lesion is inside the liver and must stay there"
        assert report.of_kind("overlap")

    def test_S5_segments_are_matched_by_label_not_by_number(self, prepared, tmp_path):
        """DICOM numbers segments 1..N; the sample's ids are 1 and 3."""
        from medh5.io.dicom_seg import from_dicom_seg, read_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        _, geometry = read_dicom_seg(out)
        assert sorted(geometry["segments"]) == [1, 2]
        report = from_dicom_seg(out, prepared["path"], ann_id="matched")
        with medh5.open(prepared["path"]) as sample:
            assert sample.annotations["matched"].class_ids == (1, 3)
        assert report.of_kind("segment_mapping")

    def test_frames_are_placed_by_geometry(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        volumes, geometry = read_dicom_seg(out)
        assert geometry["shape"] == prepared["liver"].shape
        assert np.isclose(geometry["spacing"][0], 2.5)
        assert np.array_equal(volumes[1], prepared["liver"])

    def test_a_segment_the_label_set_lacks_is_refused(self, prepared, tmp_path):
        import pydicom

        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        dataset = pydicom.dcmread(str(out))
        dataset.SegmentSequence[0].SegmentLabel = "pancreas"
        dataset.save_as(str(out), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="pancreas"):
            from_dicom_seg(out, prepared["path"], ann_id="x")

    def test_a_non_seg_file_is_refused(self, prepared):
        from medh5.io.dicom_seg import read_dicom_seg

        with pytest.raises(MEDH5ValidationError, match="not 'SEG'"):
            read_dicom_seg(prepared["series"]["paths"][0])

    def test_a_seg_from_another_reconstruction_is_refused(self, prepared, tmp_path):
        """A SEG only means anything against the grid it was drawn on."""
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        other = tmp_path / "other.medh5"
        with medh5.create(other, sample_id="other", codec="portable") as writer:
            writer.add_grid("g", shape=(4, 8, 10), spacing=(1.0, 1.0, 1.0))
            writer.add_image(
                "CT", np.zeros((4, 8, 10), dtype=np.int16), grid="g", modality="CT"
            )
        with pytest.raises(MEDH5ValidationError, match="different reconstruction"):
            from_dicom_seg(out, other, ann_id="x", grid="g")

    def test_F12_S3_3_a_seg_with_omitted_frames_imports(self, tmp_path: Path):
        """highdicom's default omits empty frames, and the import refused it.

        The reader assembled a volume from the planes present --- two slices,
        spaced by the distance between them --- and `from_dicom_seg` compared
        that shape with the grid's and reported E405 "drawn on a different
        reconstruction".  Frames carry their own position; they are placed.
        """
        hd = pytest.importorskip("highdicom")
        pydicom = pytest.importorskip("pydicom")
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root,
            patient_id="p",
            study_uid=generate_uid(),
            study_date="20260101",
            shape=(8, 16, 20),
        )
        path = tmp_path / "case.medh5"
        from_dicom(root, path, group_by="study")
        with medh5.open(path) as sample:
            shape = sample.grids[sorted(sample.grids)[0]].spatial_shape

        datasets = [pydicom.dcmread(p) for p in series["paths"]]
        normal = np.cross([0, 0, 1], [0, 1, 0])
        datasets.sort(
            key=lambda d: float(
                np.dot([float(v) for v in d.ImagePositionPatient], normal)
            )
        )
        mask = np.zeros((*shape, 1), np.uint8)
        mask[3, 4:8, 5:9, 0] = 1
        mask[4, 4:8, 5:9, 0] = 1  # two non-adjacent-from-the-edges slices
        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label="liver",
            segmented_property_category=hd.sr.CodedConcept(
                "91723000", "SCT", "Anatomical Structure"
            ),
            segmented_property_type=hd.sr.CodedConcept("10200004", "SCT", "Liver"),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=mask,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[description],
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="t",
            manufacturer_model_name="t",
            software_versions="1",
            device_serial_number="0",
            omit_empty_frames=True,
        )
        out = tmp_path / "omit.dcm"
        segmentation.save_as(str(out))

        report = from_dicom_seg(out, path, ann_id="imported")
        assert report.of_kind("frames")
        with medh5.open(path) as sample:
            dense = sample.annotations["imported"].dense(["liver"])[0]
            assert dense.shape == shape
            counts = [int(dense[k].sum()) for k in range(shape[0])]
            assert counts == [int(mask[k, ..., 0].sum()) for k in range(shape[0])]

    def test_F12_a_frame_that_is_not_a_slice_is_refused_by_name(self, tmp_path: Path):
        """The refusal stays, and says which frame and why."""
        from medh5.io.dicom_seg import place_frames

        path = tmp_path / "grid.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            geometry = {
                "rows": Organs.SHAPE[1],
                "columns": Organs.SHAPE[2],
                "segments": {1: {"label": "liver"}},
                "fractional": False,
                "direction": np.asarray(grid.direction).tolist(),
                "spacing": list(grid.spacing),
            }
            good = [
                {
                    "index": 0,
                    "segment": 1,
                    "position": grid.index_to_world([2, 0, 0]),
                    "data": np.ones(Organs.SHAPE[1:], bool),
                }
            ]
            placed = place_frames(good, geometry, grid)
            assert placed[1][2].all() and not placed[1][3].any()

            off = [
                {
                    "index": 7,
                    "segment": 1,
                    "position": grid.index_to_world([2.5, 0, 0]),
                    "data": np.ones(Organs.SHAPE[1:], bool),
                }
            ]
            with pytest.raises(MEDH5ValidationError) as exc:
                place_frames(off, geometry, grid)
            assert exc.value.code == "E405"
            assert "frame 7" in str(exc.value)


class TestDicomSegExtras:
    def test_a_seg_into_a_sample_without_a_label_set_mints_one(self, tmp_path):
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg
        from tests.helpers import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root, patient_id="p", study_uid=generate_uid(), study_date="20260101"
        )
        source = tmp_path / "src.medh5"
        from_dicom(root, source, group_by="study")
        with medh5.open(source) as sample:
            grid_id = sorted(sample.grids)[0]
            shape = sample.grids[grid_id].spatial_shape
        mask = np.zeros(shape, bool)
        mask[1:4, 2:8, 3:9] = True
        with medh5.amend(source) as writer:
            writer.label_set(
                LabelSet(
                    "d", version="1.0.0", classes=[LabelClass(7, "liver", "Liver")]
                )
            )
            writer.add_segmentation("organs", grid=grid_id, masks={7: mask})
        seg = to_dicom_seg(source, "organs", series["paths"], tmp_path / "s.dcm")

        blank = tmp_path / "blank.medh5"
        from_dicom(root, blank, group_by="study")
        report = from_dicom_seg(seg, blank, ann_id="imported")
        with medh5.open(blank) as sample:
            minted = sample.label_set
            assert [c.key for c in minted] == ["liver"]
            assert np.array_equal(
                sample.annotations["imported"].dense(["liver"])[0], mask
            )
        note = report.of_kind("label_set")[0]
        assert note.detail["bound"] == 1, "the SEG's coded concept came through"

    def test_a_fractional_seg_becomes_a_probmap(self, tmp_path):
        pytest.importorskip("highdicom")
        import highdicom as hd
        import pydicom
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg, read_dicom_seg

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root, patient_id="p", study_uid=generate_uid(), study_date="20260101"
        )
        sample_path = tmp_path / "case.medh5"
        from_dicom(root, sample_path, group_by="study")
        with medh5.open(sample_path) as sample:
            shape = sample.grids[sorted(sample.grids)[0]].spatial_shape

        datasets = [pydicom.dcmread(p) for p in series["paths"]]
        normal = np.cross([0, 0, 1], [0, 1, 0])
        datasets.sort(
            key=lambda d: float(
                np.dot([float(v) for v in d.ImagePositionPatient], normal)
            )
        )
        probabilities = np.zeros((*shape, 1), dtype=np.float32)
        probabilities[1:4, 2:8, 3:9, 0] = 0.6
        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label="liver",
            segmented_property_category=hd.sr.CodedConcept(
                "91723000", "SCT", "Anatomical Structure"
            ),
            segmented_property_type=hd.sr.CodedConcept("10200004", "SCT", "Liver"),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="test",
                version="1",
                family=hd.sr.CodedConcept("123037004", "SCT", "Body Structure"),
            ),
        )
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=probabilities,
            segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
            segment_descriptions=[description],
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="test",
            manufacturer_model_name="test",
            software_versions="1",
            device_serial_number="0",
            omit_empty_frames=False,
        )
        out = tmp_path / "frac.dcm"
        segmentation.save_as(str(out))

        volumes, geometry = read_dicom_seg(out)
        assert geometry["fractional"]
        assert 0.0 < float(volumes[1].max()) <= 1.0
        report = from_dicom_seg(out, sample_path, ann_id="prob")
        with medh5.open(sample_path) as sample:
            annotation = sample.annotations["prob"]
            assert annotation.kind == "probmap"
            assert [c.key for c in sample.label_set] == ["liver"]
            assert sample.label_set["liver"].codes[0].code == "10200004"
        assert report.of_kind("fractional")


class TestSegFramesSitOnTheGrid:
    """A frame's position, its rows and columns, and its pixel spacing are the
    grid's (L02 of the 2.0 audit).  Only the first index of a frame's position
    was read, and rows and columns compared by count, so a SEG shifted in
    plane, turned, or at another pixel spacing was laid on the grid as it was."""

    @staticmethod
    def _grid(tmp_path: Path, **options: Any) -> Path:
        path = tmp_path / "grid.medh5"
        spacing = options.pop("spacing", (2.0, 0.8, 0.8))
        with medh5.create(path, sample_id="g") as w:
            w.add_grid(
                "g", shape=Organs.SHAPE, spacing=spacing, timepoint="tp0", **options
            )
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
        return path

    @staticmethod
    def _geometry(grid: Any, *, direction: Any = None, spacing: Any = None) -> dict:
        return {
            "rows": Organs.SHAPE[1],
            "columns": Organs.SHAPE[2],
            "segments": {1: {"label": "liver"}},
            "fractional": False,
            "direction": np.asarray(
                grid.direction if direction is None else direction
            ).tolist(),
            "spacing": list(grid.spacing if spacing is None else spacing),
        }

    @staticmethod
    def _frame(position: Any, index: int = 0) -> dict[str, Any]:
        return {
            "index": index,
            "segment": 1,
            "position": position,
            "data": np.ones(Organs.SHAPE[1:], bool),
        }

    def test_L02_a_frame_shifted_in_plane_is_refused(self, tmp_path):
        from medh5.io.dicom_seg import place_frames

        with medh5.open(self._grid(tmp_path)) as sample:
            grid = sample.grids["g"]
            shifted = [self._frame(grid.index_to_world([2, 3, 0]), index=4)]
            with pytest.raises(
                MEDH5ValidationError, match=r"\(3\.000, 0\.000\)"
            ) as exc:
                place_frames(shifted, self._geometry(grid), grid)
            assert exc.value.code == "E405" and "frame 4" in str(exc.value)

    def test_L02_another_orientation_or_pixel_spacing_is_refused(self, tmp_path):
        from medh5.io.dicom_seg import place_frames

        with medh5.open(self._grid(tmp_path)) as sample:
            grid = sample.grids["g"]
            frames = [self._frame(grid.index_to_world([2, 0, 0]))]
            swapped = np.asarray(grid.direction)[:, [0, 2, 1]]
            with pytest.raises(MEDH5ValidationError, match="rows and columns run"):
                place_frames(frames, self._geometry(grid, direction=swapped), grid)
            with pytest.raises(MEDH5ValidationError, match="pixel spacing"):
                place_frames(
                    frames, self._geometry(grid, spacing=(2.0, 0.7, 0.7)), grid
                )

    def test_L02_patient_positions_are_LPS_millimetres_whatever_the_grid(
        self, tmp_path
    ):
        """An RAS grid in metres takes the same SEG, converted rather than refused."""
        from medh5.io.dicom_seg import place_frames

        path = self._grid(
            tmp_path, coord_system="RAS", units="m", spacing=(0.002, 0.0008, 0.0008)
        )
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            lps = np.array([-1.0, -1.0, 1.0])
            position = lps * np.asarray(grid.index_to_world([2, 0, 0])) * 1000.0
            direction = lps[:, None] * np.asarray(grid.direction)
            spacing = [v * 1000.0 for v in grid.spacing]
            geometry = self._geometry(grid, direction=direction, spacing=spacing)
            placed = place_frames([self._frame(position)], geometry, grid)
            assert placed[1][2].all() and not placed[1][3].any()


def _rewrite_frames(seg: Path, out: Path, change: Any = None) -> Path:
    """The SEG with its plane orientation and pixel measures moved out of the
    shared functional groups into each frame's own, as PS3.3 C.7.6.16 permits;
    *change*, given each frame's item and index, may then alter them."""
    import copy

    dataset = pydicom.dcmread(str(seg))
    shared = dataset.SharedFunctionalGroupsSequence[0]
    orientation = shared.PlaneOrientationSequence
    measures = shared.PixelMeasuresSequence
    del shared.PlaneOrientationSequence
    del shared.PixelMeasuresSequence
    items = dataset.PerFrameFunctionalGroupsSequence
    for index, item in enumerate(items):
        item.PlaneOrientationSequence = copy.deepcopy(orientation)
        item.PixelMeasuresSequence = copy.deepcopy(measures)
        if change is not None:
            change(item, index, len(items))
    dataset.save_as(str(out), enforce_file_format=True)
    return out


class TestSegGeometryPerFrame:
    """A functional group may sit in the shared item or in each frame's (L02 of
    the 2.0 re-audit): the import took the first orientation it found and the
    shared pixel spacing for every frame, so a frame turned in plane was
    placed as the first was, and spacing kept per frame read as 1 mm."""

    @staticmethod
    def _exported(prepared: dict[str, Any], tmp_path: Path) -> Path:
        from medh5.io.dicom_seg import to_dicom_seg

        return to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )

    def test_L02_geometry_kept_in_each_frame_imports_as_shared(
        self, prepared, tmp_path
    ):
        """The series is 0.8 x 0.9 mm in plane, which the shared default of
        1 mm refused as another reconstruction."""
        from medh5.io.dicom_seg import from_dicom_seg

        out = _rewrite_frames(self._exported(prepared, tmp_path), tmp_path / "f.dcm")
        from_dicom_seg(out, prepared["path"], ann_id="per_frame")
        with medh5.open(prepared["path"]) as sample:
            imported = sample.annotations["per_frame"]
            assert np.array_equal(imported.dense(["liver"])[0], prepared["liver"])
            assert np.array_equal(imported.dense(["lesion"])[0], prepared["lesion"])

    @pytest.mark.parametrize("what", ["orientation", "spacing"])
    def test_L02_a_later_frame_unlike_the_first_is_refused(
        self, prepared, tmp_path, what
    ):
        from medh5.io.dicom_seg import from_dicom_seg

        def turn(item: Any, index: int, count: int) -> None:
            if index != count - 1:
                return
            if what == "orientation":  # 180 degrees in plane
                plane = item.PlaneOrientationSequence[0]
                plane.ImageOrientationPatient = [
                    -float(v) for v in plane.ImageOrientationPatient
                ]
            else:
                item.PixelMeasuresSequence[0].PixelSpacing = [0.7, 0.7]

        out = _rewrite_frames(
            self._exported(prepared, tmp_path), tmp_path / "f.dcm", turn
        )
        with pytest.raises(MEDH5ValidationError, match="not one volume") as caught:
            from_dicom_seg(out, prepared["path"], ann_id="turned")
        assert caught.value.code == "E405"
        with medh5.open(prepared["path"]) as sample:
            assert "turned" not in sample.annotations

    def test_L02_a_frame_without_pixel_spacing_is_refused(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg_frames

        def drop(item: Any, index: int, count: int) -> None:
            del item.PixelMeasuresSequence

        out = _rewrite_frames(
            self._exported(prepared, tmp_path), tmp_path / "f.dcm", drop
        )
        with pytest.raises(MEDH5ValidationError, match="no PixelSpacing"):
            read_dicom_seg_frames(out)


class TestSegExportGeometry:
    """A SEG is written against its source images, frame k with image k's
    geometry, and the export compared the annotation's grid with neither:
    an annotation on a grid turned in plane, or shifted 10 mm in the same
    frame, was written onto the source geometry (N01 of the 2.0 re-audit)."""

    @staticmethod
    def _variant(prepared: dict[str, Any], name: str, how: str) -> None:
        """An annotation of the liver on a grid of the sample's frame: turned
        in plane over the same lattice, shifted 10 mm, or resampled."""
        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[prepared["grid"]]
            shape, spacing = grid.shape, np.asarray(grid.spacing, np.float64)
            direction = np.array(grid.direction, np.float64)
            origin = np.asarray(grid.origin, np.float64)
            corner = np.asarray(
                grid.index_to_world([0, shape[1] - 1, shape[2] - 1]), np.float64
            ).reshape(-1)
            frame, timepoint = grid.frame_uid, grid.timepoint
        liver = prepared["liver"]
        if how == "turned":
            direction[:, 1:] *= -1
            origin, liver = corner, liver[:, ::-1, ::-1]
        elif how == "shifted":
            origin = origin + 10.0 * direction[:, 2]
        else:
            spacing = spacing * np.array([1.0, 1.1, 1.1])
        with medh5.amend(prepared["path"]) as w:
            w.add_grid(
                name,
                shape=shape,
                spacing=tuple(spacing),
                origin=tuple(origin),
                direction=direction,
                frame_uid=frame,
                timepoint=timepoint,
            )
            w.add_segmentation(name, grid=name, masks={1: np.ascontiguousarray(liver)})

    @pytest.mark.parametrize("how", ["turned", "shifted", "resampled"])
    def test_N01_an_annotation_on_another_grid_is_refused(
        self, prepared, tmp_path, how
    ):
        from medh5.io.dicom_seg import to_dicom_seg

        self._variant(prepared, how, how)
        out = tmp_path / "s.dcm"
        with pytest.raises(MEDH5ValidationError) as caught:
            to_dicom_seg(prepared["path"], how, prepared["series"]["paths"], out)
        assert caught.value.code == "E405" and not out.exists()

    def test_N01_exported_frames_sit_where_the_annotation_does(
        self, prepared, tmp_path
    ):
        """Every labelled SEG pixel, placed by its frame's own position,
        orientation and spacing, is a labelled voxel of the annotation."""
        from medh5.io.dicom_seg import to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        seg = pydicom.dcmread(str(out))
        shared = seg.SharedFunctionalGroupsSequence[0]
        cosines = np.asarray(
            shared.PlaneOrientationSequence[0].ImageOrientationPatient, np.float64
        )
        rows, columns = (float(v) for v in shared.PixelMeasuresSequence[0].PixelSpacing)
        pixels = np.asarray(seg.pixel_array).reshape(-1, seg.Rows, seg.Columns)
        exported = set()
        for item, frame in zip(
            seg.PerFrameFunctionalGroupsSequence, pixels, strict=True
        ):
            segment = item.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            if segment != 1:
                continue
            position = np.asarray(
                item.PlanePositionSequence[0].ImagePositionPatient, np.float64
            )
            for r, c in zip(*np.nonzero(frame), strict=True):
                world = position + r * rows * cosines[3:] + c * columns * cosines[:3]
                exported.add(tuple(np.round(world, 3)))
        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[prepared["grid"]]
            assert grid.coord_system == "LPS"
            voxels = np.argwhere(prepared["liver"])
            drawn = {
                tuple(np.round(np.asarray(grid.index_to_world(v)).reshape(-1), 3))
                for v in voxels
            }
        assert exported == drawn
