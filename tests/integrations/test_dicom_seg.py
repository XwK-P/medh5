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
def highdicom() -> Any:
    """highdicom, which writing a SEG takes: a test that writes one skips
    without it, as on a Python it publishes no wheel for yet, and the tests
    that only read run all the same."""
    return pytest.importorskip("highdicom")


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
    @pytest.mark.usefixtures("highdicom")
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

    @pytest.mark.usefixtures("highdicom")
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

    @pytest.mark.usefixtures("highdicom")
    def test_frames_are_placed_by_geometry(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg, to_dicom_seg

        out = to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )
        volumes, geometry = read_dicom_seg(out)
        assert geometry["shape"] == prepared["liver"].shape
        assert np.isclose(geometry["spacing"][0], 2.5)
        assert np.array_equal(volumes[1], prepared["liver"])

    @pytest.mark.usefixtures("highdicom")
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

    @pytest.mark.usefixtures("highdicom")
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

    @pytest.mark.usefixtures("highdicom")
    @pytest.mark.parametrize("encoding", ["labelmap", "bitmask"])
    def test_S7_7_an_annotation_with_an_ignore_region_is_refused(
        self, prepared, tmp_path, encoding
    ):
        """A binary segment's 0 says examined and absent: written from
        `dense()`, every voxel the annotation ignores went out as that 0 ---
        in band (`labelmap`) or in its sibling mask (`bitmask`) alike."""
        from medh5.io.dicom_seg import to_dicom_seg

        ignore = np.zeros_like(prepared["liver"])
        ignore[0] = True  # a slice nobody read, clear of the liver
        with medh5.amend(prepared["path"]) as writer:
            writer.add_segmentation(
                "partial",
                grid=prepared["grid"],
                masks={1: prepared["liver"]},
                ignore=ignore,
                encoding=encoding,
            )
        out = tmp_path / "s.dcm"
        sources = prepared["series"]["paths"]
        with pytest.raises(MEDH5ValidationError, match="ignores") as caught:
            to_dicom_seg(prepared["path"], "partial", sources, out)
        assert f"ignores {int(ignore.sum())} voxel(s)" in str(caught.value)
        assert not out.exists()

    @staticmethod
    def _without_segments(prepared: dict[str, Any]) -> None:
        """Boxes, which have no voxels, and a `mask`, which has no classes."""
        with medh5.amend(prepared["path"]) as writer:
            writer.add_boxes(
                "det",
                grid=prepared["grid"],
                boxes=[[[1.0, 3.0], [2.0, 6.0], [3.0, 8.0]]],
                class_ids=[1],
                task="detection",
            )
            writer.add_mask("fov", prepared["liver"], grid=prepared["grid"])

    @pytest.mark.usefixtures("highdicom")
    @pytest.mark.parametrize(
        ("ann_id", "why"),
        [
            ("det", "is 'boxes', not a voxel annotation"),
            ("fov", "is a 'mask', which has no classes"),
        ],
    )
    def test_S6_3_a_kind_a_seg_cannot_hold_is_refused_before_anything_is_read(
        self, prepared, tmp_path, ann_id, why
    ):
        """A SEG holds one segment per class. Boxes reached `dense()` and
        failed with an `AttributeError`; a mask wrote no segment, and
        highdicom failed with an `IndexError`."""
        from medh5.io.dicom_seg import to_dicom_seg

        self._without_segments(prepared)
        out = tmp_path / "s.dcm"
        # Read, a source that is not there would be refused on its own.
        with pytest.raises(MEDH5ValidationError) as caught:
            to_dicom_seg(prepared["path"], ann_id, [tmp_path / "absent.dcm"], out)
        assert caught.value.code == "E401"
        message = str(caught.value)
        assert why in message
        assert "'labelmap', 'layers', 'bitmask', 'instances' or 'probmap'" in message
        assert not out.exists()

    @pytest.mark.usefixtures("highdicom")
    def test_S6_3_the_command_line_refuses_it_by_name(self, prepared, tmp_path, capsys):
        from medh5.cli import EXIT_ERROR, main

        self._without_segments(prepared)
        sources = [a for p in prepared["series"]["paths"] for a in ("--source", p)]
        for ann_id in ("det", "fov"):
            out = tmp_path / f"{ann_id}.dcm"
            path = str(prepared["path"])
            argv = ["convert", "to-dicom-seg", path, ann_id, str(out), *sources]
            assert main(argv) == EXIT_ERROR
            err = capsys.readouterr().err
            assert err.startswith(f"medh5: [E401] annotation {ann_id!r} is"), err
            assert "Traceback" not in err
            assert not out.exists()

    def test_S3_3_a_seg_with_omitted_frames_imports(self, tmp_path: Path):
        """highdicom's default omits empty frames, and the import refused it.

        The reader assembled a volume from the planes present --- two slices,
        spaced by the distance between them --- and `from_dicom_seg` compared
        that shape with the grid's and reported E405 "drawn on a different
        reconstruction".  Frames carry their own position; they are placed.
        """
        hd = pytest.importorskip("highdicom")
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

    def test_S3_3_a_frame_that_is_not_a_slice_is_refused_by_name(self, tmp_path: Path):
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
    @pytest.mark.usefixtures("highdicom")
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
    grid's.  Only the first index of a frame's position
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

    def test_S3_2_a_frame_shifted_in_plane_is_refused(self, tmp_path):
        from medh5.io.dicom_seg import place_frames

        with medh5.open(self._grid(tmp_path)) as sample:
            grid = sample.grids["g"]
            shifted = [self._frame(grid.index_to_world([2, 3, 0]), index=4)]
            with pytest.raises(
                MEDH5ValidationError, match=r"\(3\.000, 0\.000\)"
            ) as exc:
                place_frames(shifted, self._geometry(grid), grid)
            assert exc.value.code == "E405" and "frame 4" in str(exc.value)

    def test_S3_2_another_orientation_or_pixel_spacing_is_refused(self, tmp_path):
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

    def test_S3_5_patient_positions_are_LPS_millimetres_whatever_the_grid(
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

    def test_S3_5_a_grid_without_patient_coordinates_is_E414(self, tmp_path):
        """A grid in pixels has no place for a patient position: its space is
        what is wrong, as the RTSTRUCT importer says, not a shape (E405)."""
        from medh5.io.dicom_seg import place_frames

        path = self._grid(tmp_path, units="px", spacing=(1.0, 1.0, 1.0))
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            frames = [self._frame(grid.index_to_world([2, 0, 0]))]
            with pytest.raises(MEDH5ValidationError, match="uncalibrated") as caught:
                place_frames(frames, self._geometry(grid), grid)
            assert caught.value.code == "E414"


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


@pytest.mark.usefixtures("highdicom")
class TestSegGeometryPerFrame:
    """A functional group may sit in the shared item or in each frame's: the
    import took the first orientation it found and the
    shared pixel spacing for every frame, so a frame turned in plane was
    placed as the first was, and spacing kept per frame read as 1 mm."""

    @staticmethod
    def _exported(prepared: dict[str, Any], tmp_path: Path) -> Path:
        from medh5.io.dicom_seg import to_dicom_seg

        return to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )

    def test_S3_2_geometry_kept_in_each_frame_imports_as_shared(
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
    def test_S3_2_a_later_frame_unlike_the_first_is_refused(
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
        # About the SEG alone, so no §15.2 code (there is no sample yet).
        assert caught.value.code is None
        with medh5.open(prepared["path"]) as sample:
            assert "turned" not in sample.annotations

    def test_S3_2_a_frame_without_pixel_spacing_is_refused(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg_frames

        def drop(item: Any, index: int, count: int) -> None:
            del item.PixelMeasuresSequence

        out = _rewrite_frames(
            self._exported(prepared, tmp_path), tmp_path / "f.dcm", drop
        )
        with pytest.raises(MEDH5ValidationError, match="no PixelSpacing"):
            read_dicom_seg_frames(out)


@pytest.mark.usefixtures("highdicom")
class TestSegExportGeometry:
    """A SEG is written against its source images, frame k with image k's
    geometry, and the export compared the annotation's grid with neither:
    an annotation on a grid turned in plane, or shifted 10 mm in the same
    frame, was written onto the source geometry."""

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

    def test_S3_4_source_images_in_another_frame_are_E414(self, prepared, tmp_path):
        """Another frame of reference is E414, as on import; a frame `medh5
        scrub` pseudonymised is compared through the scrub's salt."""
        from medh5.curation.scrub import pseudonymise
        from medh5.io.dicom_seg import to_dicom_seg

        frame = prepared["series"]["frame_uid"]
        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[prepared["grid"]]
            lattice = {
                "shape": grid.shape,
                "spacing": grid.spacing,
                "origin": grid.origin,
                "direction": np.asarray(grid.direction),
                "timepoint": grid.timepoint,
            }
        with medh5.amend(prepared["path"]) as w:
            for name, uid in (
                ("other", "9.9.9"),
                ("scrubbed", pseudonymise(frame, "s")),
            ):
                w.add_grid(name, frame_uid=uid, **lattice)
                w.add_segmentation(name, grid=name, masks={1: prepared["liver"]})
        sources = prepared["series"]["paths"]
        with pytest.raises(MEDH5ValidationError, match="across frames") as caught:
            to_dicom_seg(prepared["path"], "other", sources, tmp_path / "a.dcm")
        assert caught.value.code == "E414"
        with pytest.raises(MEDH5ValidationError, match="frame_salt=") as caught:
            to_dicom_seg(prepared["path"], "scrubbed", sources, tmp_path / "b.dcm")
        assert caught.value.code == "E414"
        assert "--frame-salt" in str(caught.value), "the command line's name too"
        out = to_dicom_seg(
            prepared["path"], "scrubbed", sources, tmp_path / "c.dcm", frame_salt="s"
        )
        assert pydicom.dcmread(str(out)).FrameOfReferenceUID == frame

    @pytest.mark.parametrize("how", ["turned", "shifted", "resampled"])
    def test_S3_2_an_annotation_on_another_grid_is_refused(
        self, prepared, tmp_path, how
    ):
        from medh5.io.dicom_seg import to_dicom_seg

        self._variant(prepared, how, how)
        out = tmp_path / "s.dcm"
        with pytest.raises(MEDH5ValidationError) as caught:
            to_dicom_seg(prepared["path"], how, prepared["series"]["paths"], out)
        assert caught.value.code == "E405" and not out.exists()

    def test_S3_3_exported_frames_sit_where_the_annotation_does(
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


def _highdicom_seg(paths: list[Any], mask: Any, out: Path) -> Path:
    """A BINARY SEG of *mask* (slices in the series' geometric order, one
    segment) over the DICOM files *paths*, empty frames omitted."""
    hd = pytest.importorskip("highdicom")
    datasets = [pydicom.dcmread(p) for p in paths]
    normal = np.cross([0, 0, 1], [0, 1, 0])
    datasets.sort(
        key=lambda d: float(np.dot([float(v) for v in d.ImagePositionPatient], normal))
    )
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
        pixel_array=np.asarray(mask, np.uint8)[..., None],
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
    segmentation.save_as(str(out))
    return out


def _labelled_seg(
    paths: list[Any],
    segments: list[tuple[str, Any]],
    out: Path,
    *,
    fractional: bool = False,
    codes: list[tuple[str, str]] | None = None,
) -> Path:
    """A SEG of *segments* --- ``(SegmentLabel, volume)`` in the series'
    geometric order --- over the DICOM files *paths*, numbered 1..N.  *codes*
    are each segment's SNOMED ``(code, meaning)``; Liver by default."""
    hd = pytest.importorskip("highdicom")
    datasets = [pydicom.dcmread(p) for p in paths]
    normal = np.cross([0, 0, 1], [0, 1, 0])
    datasets.sort(
        key=lambda d: float(np.dot([float(v) for v in d.ImagePositionPatient], normal))
    )
    codes = codes or [("10200004", "Liver")] * len(segments)
    descriptions = [
        hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=label,
            segmented_property_category=hd.sr.CodedConcept(
                "91723000", "SCT", "Anatomical Structure"
            ),
            segmented_property_type=hd.sr.CodedConcept(code, "SCT", meaning),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        for number, ((label, _), (code, meaning)) in enumerate(
            zip(segments, codes, strict=True), start=1
        )
    ]
    pixels = np.stack([np.asarray(volume) for _, volume in segments], axis=-1)
    segmentation = hd.seg.Segmentation(
        source_images=datasets,
        pixel_array=pixels.astype(np.float32 if fractional else np.uint8),
        segmentation_type=(
            hd.seg.SegmentationTypeValues.FRACTIONAL
            if fractional
            else hd.seg.SegmentationTypeValues.BINARY
        ),
        segment_descriptions=descriptions,
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
    segmentation.save_as(str(out))
    return out


class TestSegmentsOfOneClass:
    """Segments that make one class are combined into it.  One "Lesion"
    segment per lesion is common, and keyed by class each segment replaced the
    one before: two lesions imported as the second alone, with no warning."""

    @staticmethod
    def _lesions(prepared: dict[str, Any]) -> tuple[Any, Any]:
        first = np.zeros_like(prepared["lesion"])
        first[1, 2:4, 3:6] = True
        second = np.zeros_like(prepared["lesion"])
        second[4, 9:12, 12:15] = True  # another slice, another place
        return first, second

    def test_S5_two_segments_of_one_label_are_both_imported(self, prepared, tmp_path):
        from medh5.io.dicom_seg import from_dicom_seg

        first, second = self._lesions(prepared)
        seg = _labelled_seg(
            prepared["series"]["paths"],
            [("Lesion", first), ("Lesion", second)],
            tmp_path / "lesions.dcm",
        )
        report = from_dicom_seg(seg, prepared["path"], ann_id="lesions")
        with medh5.open(prepared["path"]) as sample:
            imported = sample.annotations["lesions"]
            assert imported.class_ids == (3,)
            lesion = imported.dense(["lesion"])[0]
        assert np.array_equal(lesion, first | second), "both lesions, every voxel"
        (note,) = report.of_kind("segment_union")
        assert note.severity == "decision"
        assert [m["segment"] for m in note.detail["classes"]["lesion"]] == [1, 2]
        assert report.ok

    def test_S5_minted_classes_combine_one_label_too(self, tmp_path):
        """With no label set to match, two segments keyed `lesion` minted two
        classes of one key, refused at commit (E302)."""
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg

        series = write_dicom_series(
            tmp_path / "dcm", patient_id="p", study_uid=generate_uid(),
            study_date="20260101",
        )  # fmt: skip
        blank = tmp_path / "blank.medh5"
        from_dicom(tmp_path / "dcm", blank, group_by="study")
        first = np.zeros((6, 16, 20), bool)
        first[1, 2:4, 3:6] = True
        second = np.zeros((6, 16, 20), bool)
        second[4, 9:12, 12:15] = True
        seg = _labelled_seg(
            series["paths"],
            [("Lesion", first), ("lesion", second)],
            tmp_path / "lesions.dcm",
        )
        report = from_dicom_seg(seg, blank, ann_id="lesions")
        with medh5.open(blank) as sample:
            assert [(c.id, c.key, c.name) for c in sample.label_set] == [
                (1, "lesion", "Lesion")
            ]
            dense = sample.annotations["lesions"].dense(["lesion"])[0]
        assert np.array_equal(dense, first | second)
        assert report.of_kind("segment_union")

    def test_S5_one_key_coding_two_concepts_is_refused(self, tmp_path):
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg

        series = write_dicom_series(
            tmp_path / "dcm", patient_id="p", study_uid=generate_uid(),
            study_date="20260101",
        )  # fmt: skip
        blank = tmp_path / "blank.medh5"
        from_dicom(tmp_path / "dcm", blank, group_by="study")
        mask = np.zeros((6, 16, 20), bool)
        mask[2, 4:8, 5:9] = True
        seg = _labelled_seg(
            series["paths"],
            [("Lesion", mask), ("Lesion", mask)],
            tmp_path / "two.dcm",
            codes=[("108369006", "Neoplasm"), ("441457006", "Cyst")],
        )
        before = blank.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="different concepts") as exc:
            from_dicom_seg(seg, blank)
        assert "SCT 108369006" in str(exc.value) and "SCT 441457006" in str(exc.value)
        assert blank.read_bytes() == before

    def test_S7_5_fractional_segments_of_one_class_take_their_maximum(
        self, prepared, tmp_path
    ):
        from medh5.io.dicom_seg import from_dicom_seg

        first = np.zeros(prepared["lesion"].shape, np.float32)
        first[2, 4:8, 5:9] = 0.6
        second = np.zeros_like(first)
        second[2, 6:10, 5:9] = 0.8  # overlaps the first in two rows
        seg = _labelled_seg(
            prepared["series"]["paths"],
            [("Lesion", first), ("Lesion", second)],
            tmp_path / "frac.dcm",
            fractional=True,
        )
        report = from_dicom_seg(seg, prepared["path"], ann_id="prob")
        with medh5.open(prepared["path"]) as sample:
            probabilities = sample.annotations["prob"].probabilities(["lesion"])[0]
        expected = np.maximum(first, second)
        assert np.allclose(probabilities, expected, atol=1 / 255)
        assert "maximum" in report.of_kind("segment_union")[0].message


@pytest.mark.usefixtures("highdicom")
class TestSegExportCoverage:
    """A segment's 0 says "examined, absent".  Every class went out, and a
    class nobody examined became an empty segment: a verified negative to
    whoever read the SEG back."""

    @staticmethod
    def _partial(prepared: dict[str, Any]) -> None:
        """Liver examined; lesion a class of the annotation nobody looked for."""
        with medh5.amend(prepared["path"]) as writer:
            writer.add_segmentation(
                "partial",
                grid=prepared["grid"],
                masks={1: prepared["liver"], 3: np.zeros_like(prepared["liver"])},
                annotated_classes=["liver"],
            )

    def test_S11_3_only_examined_classes_are_exported(self, prepared, tmp_path):
        from medh5.io.dicom_seg import from_dicom_seg, read_dicom_seg, to_dicom_seg
        from medh5.io.report import ConversionReport

        self._partial(prepared)
        with medh5.open(prepared["path"]) as sample:
            partial = sample.annotations["partial"]
            assert partial.class_ids == (1, 3) and partial.annotated_class_ids == (1,)
        report = ConversionReport()
        out = to_dicom_seg(
            prepared["path"],
            "partial",
            prepared["series"]["paths"],
            tmp_path / "s.dcm",
            report=report,
        )
        _, geometry = read_dicom_seg(out)
        assert [s["label"] for s in geometry["segments"].values()] == ["liver"]
        (note,) = report.of_kind("unexamined")
        assert note.detail["left_out"] == ["lesion"]
        # Read back, the lesion is still a class nobody examined.
        from_dicom_seg(out, prepared["path"], ann_id="back")
        with medh5.open(prepared["path"]) as sample:
            assert sample.annotations["back"].annotated_class_ids == (1,)
            assert not sample.annotations["back"].is_annotated("lesion")

    def test_S11_3_naming_an_unexamined_class_is_refused(self, prepared, tmp_path):
        from medh5.io.dicom_seg import to_dicom_seg

        self._partial(prepared)
        out = tmp_path / "s.dcm"
        sources = prepared["series"]["paths"]
        with pytest.raises(MEDH5ValidationError, match="never examined") as exc:
            to_dicom_seg(prepared["path"], "partial", sources, out, classes=["lesion"])
        assert "['lesion']" in str(exc.value)
        assert not out.exists()
        to_dicom_seg(prepared["path"], "partial", sources, out, classes=["liver"])
        assert out.exists()


class TestSegFrameIdentity:
    """A SEG lands only in its own frame of reference: a grid in another
    known frame, with the same numbers, received
    every voxel --- from an explicit ``grid=`` without a word, from automatic
    matching on a recorded guess."""

    def test_S3_4_placement_compares_frames_first(self, tmp_path: Path):
        from medh5.curation.scrub import pseudonymise
        from medh5.io.dicom_seg import place_frames

        def grid(name: str, frame: str | None) -> Any:
            folder = tmp_path / name
            folder.mkdir()
            options = {} if frame is None else {"frame_uid": frame}
            path = TestSegFramesSitOnTheGrid._grid(folder, **options)
            with medh5.open(path) as sample:
                return sample.grids["g"]

        own, other = grid("own", "1.2.3"), grid("other", "9.9.9")
        frameless = grid("none", None)
        scrubbed = grid("scrubbed", pseudonymise("1.2.3", "salt"))
        geometry = {**TestSegFramesSitOnTheGrid._geometry(own), "frame_uid": "1.2.3"}
        frames = [TestSegFramesSitOnTheGrid._frame(own.index_to_world([2, 0, 0]))]
        for target in (own, frameless):
            assert place_frames(frames, geometry, target)[1][2].all()
        with pytest.raises(MEDH5ValidationError, match="no correspondence") as caught:
            place_frames(frames, geometry, other)
        assert caught.value.code == "E414"
        with pytest.raises(MEDH5ValidationError, match="frame_salt="):
            place_frames(frames, geometry, scrubbed)
        with pytest.raises(MEDH5ValidationError, match="no correspondence"):
            place_frames(frames, geometry, scrubbed, frame_salt="another")
        placed = place_frames(frames, geometry, scrubbed, frame_salt="salt")
        assert placed[1][2].all()

    def test_S3_4_an_import_into_another_frame_leaves_the_sample(self, tmp_path: Path):
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg

        series = write_dicom_series(
            tmp_path / "dcm", patient_id="p", study_uid=generate_uid(),
            study_date="20260101", shape=(6, 16, 20),
        )  # fmt: skip
        source = tmp_path / "source.medh5"
        from_dicom(tmp_path / "dcm", source, group_by="study")
        with medh5.open(source) as sample:
            g = sample.grids[sorted(sample.grids)[0]]
            lattice = {
                "shape": g.spatial_shape,
                "spacing": g.spacing,
                "origin": g.origin,
                "direction": np.array(g.direction),
                "coord_system": g.coord_system,
            }
        mask = np.zeros(lattice["shape"], bool)
        mask[2, 4:8, 5:9] = True
        seg = _highdicom_seg(series["paths"], mask, tmp_path / "seg.dcm")

        def target(name: str, frame: str | None) -> Path:
            path = tmp_path / f"{name}.medh5"
            with medh5.create(path, sample_id=name) as w:
                options = {} if frame is None else {"frame_uid": frame}
                w.add_grid("g", timepoint="tp0", **lattice, **options)
                w.add_image(
                    "CT", np.zeros(lattice["shape"], np.int16), grid="g", modality="CT"
                )
            return path

        other = target("other", "9.9.9")
        before = other.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="no correspondence") as caught:
            from_dicom_seg(seg, other, grid="g")
        assert caught.value.code == "E414"
        with pytest.raises(MEDH5ValidationError, match="another frame") as caught:
            from_dicom_seg(seg, other)
        assert caught.value.code == "E101"
        assert other.read_bytes() == before
        report = from_dicom_seg(seg, target("frameless", None))
        assert report.of_kind("frame"), "a placement on geometry alone is a guess"
        report = from_dicom_seg(seg, source)
        assert not report.of_kind("frame")
        with medh5.open(source) as sample:
            assert np.array_equal(sample.annotations["seg"].dense(["liver"])[0], mask)


@pytest.mark.usefixtures("highdicom")
class TestSegKinds:
    """A SEG is read only as what its frames hold: a LABELMAP frame holds
    every segment by pixel value and names none, and was read as a BINARY
    frame of segment 1 --- as was any frame that named no segment."""

    @staticmethod
    def _exported(prepared: dict[str, Any], tmp_path: Path) -> Path:
        from medh5.io.dicom_seg import to_dicom_seg

        return to_dicom_seg(
            prepared["path"], "organs", prepared["series"]["paths"], tmp_path / "s.dcm"
        )

    def test_S7_a_labelmap_seg_is_refused_by_name(self, prepared, tmp_path):
        from medh5.io.dicom_seg import from_dicom_seg

        out = self._exported(prepared, tmp_path)
        dataset = pydicom.dcmread(str(out))
        dataset.SegmentationType = "LABELMAP"
        dataset.save_as(str(out), enforce_file_format=True)
        before = prepared["path"].read_bytes()
        with pytest.raises(MEDH5ValidationError, match="LABELMAP") as caught:
            from_dicom_seg(out, prepared["path"], ann_id="labelmap")
        assert caught.value.code is None
        assert prepared["path"].read_bytes() == before

    def test_S7_a_frame_naming_no_segment_is_refused(self, prepared, tmp_path):
        from medh5.io.dicom_seg import read_dicom_seg_frames

        out = self._exported(prepared, tmp_path)
        dataset = pydicom.dcmread(str(out))
        del dataset.PerFrameFunctionalGroupsSequence[-1].SegmentIdentificationSequence
        dataset.save_as(str(out), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="names no segment"):
            read_dicom_seg_frames(out)

    def test_S7_a_segment_named_in_the_shared_groups_is_read(self, prepared, tmp_path):
        """A functional group may sit in the shared item (PS3.3 C.7.6.16), and
        highdicom names a single-segment SEG's one segment there."""
        from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

        with medh5.amend(prepared["path"]) as w:
            w.add_segmentation(
                "liver", grid=prepared["grid"], masks={1: prepared["liver"]}
            )
        out = to_dicom_seg(
            prepared["path"], "liver", prepared["series"]["paths"], tmp_path / "l.dcm"
        )
        dataset = pydicom.dcmread(str(out))
        shared = dataset.SharedFunctionalGroupsSequence[0]
        if "SegmentIdentificationSequence" not in shared:
            frames = dataset.PerFrameFunctionalGroupsSequence
            identification = frames[0].SegmentIdentificationSequence
            shared.SegmentIdentificationSequence = identification
            for item in frames:
                del item.SegmentIdentificationSequence
            dataset.save_as(str(out), enforce_file_format=True)
        from_dicom_seg(out, prepared["path"], ann_id="back")
        with medh5.open(prepared["path"]) as sample:
            back = sample.annotations["back"].dense(["liver"])[0]
            assert np.array_equal(back, prepared["liver"])


@pytest.mark.usefixtures("highdicom")
class TestSegCommandLine:
    """`medh5 convert to-dicom-seg` and `from-dicom-seg`."""

    def test_a_seg_goes_out_and_back_through_the_command_line(self, prepared, tmp_path):
        from medh5.cli import EXIT_OK, main

        out = tmp_path / "s.dcm"
        sources = [a for p in prepared["series"]["paths"] for a in ("--source", p)]
        path = str(prepared["path"])
        assert (
            main(["convert", "to-dicom-seg", path, "organs", str(out), *sources]) == 0
        )
        argv = ["convert", "from-dicom-seg", str(out), path, "--id", "back"]
        assert main(argv) == EXIT_OK
        with medh5.open(prepared["path"]) as sample:
            back = sample.annotations["back"]
            assert np.array_equal(back.dense(["liver"])[0], prepared["liver"])
            assert np.array_equal(back.dense(["lesion"])[0], prepared["lesion"])

    def test_a_file_that_is_not_dicom_is_refused_by_name(
        self, prepared, tmp_path, capsys
    ):
        """pydicom's error named no file and was no `MEDH5Error`: the command
        line printed a traceback."""
        from medh5.cli import EXIT_ERROR, main

        text = tmp_path / "notes.medh5"
        text.write_text("not DICOM", encoding="utf-8")
        path = str(prepared["path"])
        for argv in (
            ["from-dicom-seg", str(text), path],
            [
                "to-dicom-seg",
                path,
                "organs",
                str(tmp_path / "x.dcm"),
                "--source",
                str(text),
            ],
        ):
            assert main(["convert", *argv]) == EXIT_ERROR
            err = capsys.readouterr().err
            assert err.startswith(f"medh5: {text} is not a DICOM file"), err
        assert not (tmp_path / "x.dcm").exists()

    def test_S3_4_a_scrubbed_frame_names_the_flag_that_compares_it(
        self, prepared, tmp_path, capsys
    ):
        """Refused for a frame `medh5 scrub` pseudonymised, the command line
        was told to pass `frame_salt=`, a keyword it has no way to take."""
        from medh5.cli import EXIT_ERROR, EXIT_OK, main
        from medh5.curation.scrub import pseudonymise

        with medh5.open(prepared["path"]) as sample:
            grid = sample.grids[prepared["grid"]]
            lattice = {
                "shape": grid.shape,
                "spacing": grid.spacing,
                "origin": grid.origin,
                "direction": np.asarray(grid.direction),
                "timepoint": grid.timepoint,
            }
        frame = pseudonymise(prepared["series"]["frame_uid"], "s")
        with medh5.amend(prepared["path"]) as w:
            w.add_grid("scrubbed", frame_uid=frame, **lattice)
            w.add_segmentation(
                "scrubbed", grid="scrubbed", masks={1: prepared["liver"]}
            )
        path = str(prepared["path"])
        sources = [a for p in prepared["series"]["paths"] for a in ("--source", p)]
        seg = str(tmp_path / "s.dcm")
        export = ["convert", "to-dicom-seg", path, "scrubbed", seg, *sources]
        assert main(export) == EXIT_ERROR
        assert "--frame-salt" in capsys.readouterr().err
        assert main([*export, "--frame-salt", "s"]) == EXIT_OK
        back = ["convert", "from-dicom-seg", seg, path, "--grid", "scrubbed"]
        assert main(back) == EXIT_ERROR
        assert "--frame-salt" in capsys.readouterr().err
        assert main([*back, "--frame-salt", "s"]) == EXIT_OK


class TestSparseSegReader:
    """A SEG that omits its empty frames reads back on the regular stack its
    planes sit on: the median gap stood for the
    step, so planes at 10, 12.5 and 17.5 mm read as a stack 3.75 mm apart."""

    def test_lattice(self):
        from medh5.io.dicom_seg import _lattice

        assert _lattice([10.0, 12.5, 17.5], None) == (2.5, [0, 1, 3])
        assert _lattice([0.0, 5.0], 2.5) == (2.5, [0, 2])
        assert _lattice([7.0], 3.0) == (3.0, [0])
        assert _lattice([10.0, 12.5, 16.25], None) is None
        assert _lattice([0.0, 2.5], 2.0) is None

    def test_S3_3_planes_are_held_to_the_bound_a_series_is(self):
        """A tenth of a step, and at least 0.01 mm, per plane --- as a DICOM
        series is.  Each gap was held to 10⁻³ mm, which refused the planes of
        a series whose positions were written to two decimals."""
        from medh5.io.dicom_seg import _lattice

        rounded = list(np.round(30.0 + 0.625 * np.arange(120), 2))
        found = _lattice(rounded, None)
        assert found is not None, "rounding to 0.01 mm accumulates nothing"
        from medh5.io.dicom import _position_tolerance

        step, indices = found
        assert step == pytest.approx(0.625, abs=1e-4) and indices == list(range(120))
        gapped = _lattice(np.delete(rounded, [3, 4]).tolist(), None)
        assert gapped is not None, "two omitted planes are not irregularity"
        assert gapped[0] == pytest.approx(step, abs=1e-4)
        assert gapped[1] == [k for k in range(120) if k not in (3, 4)]
        # A plane farther off than the bound is refused, not snapped.
        off = 0.625 + 1.2 * _position_tolerance(0.625)
        assert _lattice([0.0, off, 1.25, 1.875], 0.625) is None
        assert _lattice([0.0, 0.625 + 0.9 * _position_tolerance(0.625), 1.25], 0.625)

    def test_S3_3_a_one_plane_seg_takes_the_step_it_states_or_is_refused(
        self, tmp_path
    ):
        """One plane has no step to measure, and 1 mm was assumed without a
        note: `SpacingBetweenSlices`, else the slab `SliceThickness` states,
        as for a lone DICOM slice, else a refusal."""
        from pydicom.uid import generate_uid

        from medh5.io.dicom_seg import read_dicom_seg
        from medh5.io.report import ConversionReport

        series = write_dicom_series(
            tmp_path / "dcm", patient_id="p", study_uid=generate_uid(),
            study_date="20260101", shape=(6, 16, 20),
        )  # fmt: skip
        mask = np.zeros((6, 16, 20), bool)
        mask[2, 4:8, 5:9] = True
        seg = _highdicom_seg(series["paths"], mask, tmp_path / "one.dcm")
        dataset = pydicom.dcmread(str(seg))
        measures = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        # highdicom 0.23 states no step on one plane, and later releases the
        # series': stated here, so the test holds the reader, not the writer.
        measures.SpacingBetweenSlices = 2.5
        dataset.save_as(str(seg), enforce_file_format=True)
        _, stated = read_dicom_seg(seg)
        assert stated["spacing"][0] == 2.5

        del measures.SpacingBetweenSlices
        dataset.save_as(str(seg), enforce_file_format=True)
        report = ConversionReport()
        volumes, geometry = read_dicom_seg(seg, report=report)
        assert geometry["spacing"][0] == 5.0, "the slab, as for a lone slice"
        assert report.of_kind("slice_spacing")[0].severity == "decision"
        assert volumes[1].shape == (1, 16, 20)
        assert stated["spacing_source"] == "SpacingBetweenSlices"
        assert geometry["spacing_source"] == "SliceThickness"

        del measures.SliceThickness
        dataset.save_as(str(seg), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="none is invented") as exc:
            read_dicom_seg(seg)
        assert exc.value.code is None

    def test_S3_3_omitted_planes_come_back_empty_where_they_were(self, tmp_path):
        from pydicom.uid import generate_uid

        from medh5.io.dicom_seg import read_dicom_seg, read_dicom_seg_frames

        series = write_dicom_series(
            tmp_path / "dcm", patient_id="p", study_uid=generate_uid(),
            study_date="20260101", shape=(8, 16, 20), spacing=(2.5, 0.8, 0.9),
        )  # fmt: skip
        mask = np.zeros((8, 16, 20), bool)
        mask[[0, 1, 3], 4:8, 5:9] = True
        seg = _highdicom_seg(series["paths"], mask, tmp_path / "sparse.dcm")
        volumes, geometry = read_dicom_seg(seg)
        assert geometry["shape"][0] == 4 and geometry["spacing"][0] == pytest.approx(
            2.5
        )
        counts = [int(volumes[1][k].sum()) for k in range(4)]
        assert sorted(counts) == [0, 16, 16, 16] and counts[0] and counts[3]
        # Every plane is where the returned geometry says it is.
        origin = np.asarray(geometry["origin"])
        step = np.asarray(geometry["direction"])[:, 0] * geometry["spacing"][0]
        frames, _ = read_dicom_seg_frames(seg)
        for frame in frames:
            assert np.allclose(frame["position"], origin + frame["plane"] * step)

        # Planes no regular stack holds are refused, not spaced by a median.
        dataset = pydicom.dcmread(str(seg))
        last = max(
            range(len(dataset.PerFrameFunctionalGroupsSequence)),
            key=lambda i: frames[i]["plane"],
        )
        group = dataset.PerFrameFunctionalGroupsSequence[last]
        position = [
            float(v) for v in group.PlanePositionSequence[0].ImagePositionPatient
        ]
        group.PlanePositionSequence[0].ImagePositionPatient = list(
            np.asarray(position) + 1.0 * step / np.linalg.norm(step)
        )
        dataset.save_as(str(seg), enforce_file_format=True)
        with pytest.raises(MEDH5ValidationError, match="no regular stack") as caught:
            read_dicom_seg(seg)
        # About the SEG alone, so no §15.2 code (there is no sample yet).
        assert caught.value.code is None
