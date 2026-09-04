"""What 1.3.0 changed, held to by the workstream that changed it.

The corpus carries the validator side (twelve new cases).  This module holds
the writer side of every new refusal, the API loose ends, the performance
rewrites --- each of which must return exactly what the slow path returned ---
and the structure moves, which must leave every public name where it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5._hdf5 import _temporary_name, open_h5
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError, MEDH5VersionError
from medh5.labels.labelset import LabelClass, LabelSet

SHAPE = (8, 10, 10)


def _label_set() -> LabelSet:
    return LabelSet(
        "rel-1.3.0",
        version="1.0.0",
        classes=[LabelClass(1, "liver", "Liver"), LabelClass(3, "lesion", "Lesion")],
    )


def _image() -> Any:
    return np.random.default_rng(3).integers(-1000, 1500, SHAPE).astype(np.int16)


def _mask(origin: tuple[int, int, int] = (2, 2, 2), size: int = 3) -> Any:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[tuple(slice(o, o + size) for o in origin)] = True
    return mask


def _open_writer(path: Path, *, frames: bool = False) -> Any:
    w = medh5.create(path, sample_id=path.stem)
    w.label_set(_label_set())
    w.add_grid(
        "g",
        shape=SHAPE,
        spacing=(1.0, 1.0, 1.0),
        timepoint="tp0",
        frame_uid="f0" if frames else None,
    )
    w.add_image("CT", _image(), grid="g", modality="CT")
    return w


class TestW1SpecificationCorrections:
    def test_S2_1_an_unknown_digest_algo_is_E703_everywhere(self, tmp_path: Path):
        path = tmp_path / "algo.medh5"
        with _open_writer(path):
            pass
        with open_h5(path, "r+") as handle:
            handle.attrs["digest_algo"] = "blake3"
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.compute_content_id()
            assert exc.value.code == "E703"
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.verify()
            assert exc.value.code == "E703"
        from medh5.validate import validate_file

        report = validate_file(path, level="integrity")
        assert "E703" in report.codes
        # The pass still reports, rather than dying inside the recompute.
        assert "E001" not in report.codes

    def test_S7_5_the_threshold_is_written_read_and_addressed(self, tmp_path: Path):
        path = tmp_path / "soft.medh5"
        soft = np.zeros(SHAPE, dtype=np.float32)
        soft[_mask()] = 0.4
        with _open_writer(path) as w:
            w.add_segmentation("soft", grid="g", probabilities={3: soft}, threshold=0.3)
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["soft"]
            assert ann.threshold == pytest.approx(0.3)
            assert float(ann.group.attrs["threshold"]) == pytest.approx(0.3)
            # 0.4 is above 0.3 and below the 0.5 default: the declaration decides.
            assert ann.contains(3, (3, 3, 3))
            assert ann.dense([3])[0].sum() == int(_mask().sum())
            with_threshold = sample.content_id
        plain = tmp_path / "plain.medh5"
        with _open_writer(plain) as w:
            w.add_segmentation("soft", grid="g", probabilities={3: soft})
        with medh5.open(plain) as sample:
            assert not sample.annotations["soft"].contains(3, (3, 3, 3))
            assert "threshold" not in sample.annotations["soft"].group.attrs
            # It is a spec attribute, so it is part of the address (§13.2).
            assert sample.content_id != with_threshold

    def test_S7_5_threshold_applies_to_probabilities_only(self, tmp_path: Path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _open_writer(tmp_path / "x.medh5") as w,
        ):
            w.add_segmentation("s", grid="g", masks={3: _mask()}, threshold=0.3)
        assert exc.value.code == "E404"
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _open_writer(tmp_path / "y.medh5") as w,
        ):
            w.add_segmentation(
                "s", grid="g", probabilities={3: np.zeros(SHAPE)}, threshold=1.5
            )
        assert exc.value.code == "E404"

    def test_S15_2_E603_names_agents_too(self):
        assert medh5.CODES["E603"].summary == "unknown agent or activity type"


class TestW2WriterSideRefusals:
    """Every new validator rule is refused by the writer at commit as well."""

    def _refused(self, tmp_path: Path, code: str, build: Any) -> None:
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _open_writer(tmp_path / f"{code}.medh5") as w,
        ):
            build(w)
        assert exc.value.code == code, exc.value

    def test_S7_7_ignore_mask_must_exist(self, tmp_path: Path):
        self._refused(
            tmp_path,
            "E413",
            lambda w: w.add_segmentation(
                "s", grid="g", masks={3: _mask()}, ignore_mask="nope"
            ),
        )

    def test_S7_7_ignore_mask_must_be_a_mask_on_the_same_grid(self, tmp_path: Path):
        def not_a_mask(w: Any) -> None:
            w.add_segmentation("other", grid="g", masks={1: _mask()})
            w.add_segmentation("s", grid="g", masks={3: _mask()}, ignore_mask="other")

        self._refused(tmp_path, "E413", not_a_mask)

        def other_grid(w: Any) -> None:
            w.add_grid("h", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_mask("fov", np.ones(SHAPE, dtype=bool), grid="h")
            w.add_segmentation("s", grid="g", masks={3: _mask()}, ignore_mask="fov")

        self._refused(tmp_path, "E413", other_grid)

    def test_S7_7_a_correct_ignore_mask_is_accepted(self, tmp_path: Path):
        path = tmp_path / "ok.medh5"
        with _open_writer(path) as w:
            w.add_mask("uncertain", _mask((5, 5, 5), 2), grid="g")
            w.add_segmentation(
                "s", grid="g", masks={3: _mask()}, ignore_mask="uncertain"
            )
        from medh5.validate import validate_file

        assert "E413" not in validate_file(path, level="strict").codes

    def test_S6_2_derived_from_must_exist(self, tmp_path: Path):
        self._refused(
            tmp_path,
            "E413",
            lambda w: w.add_segmentation(
                "s", grid="g", masks={3: _mask()}, derived_from=["ghost"]
            ),
        )

    def test_S6_2_derived_from_accepts_the_path_spelling(self, tmp_path: Path):
        path = tmp_path / "paths.medh5"
        with _open_writer(path) as w:
            w.add_segmentation("a", grid="g", masks={3: _mask()})
            w.add_segmentation(
                "b", grid="g", masks={3: _mask()}, derived_from=["annotations/a"]
            )
        from medh5.validate import validate_file

        assert "E413" not in validate_file(path).codes

    def test_S4_4_valid_mask_must_be_a_mask(self, tmp_path: Path):
        path = tmp_path / "vm.medh5"
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(path, sample_id="vm") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT", valid_mask="nope")
        assert exc.value.code == "E413"

    def test_S10_1_transform_links_are_checked(self, tmp_path: Path):
        def dangling_prov(w: Any) -> None:
            w.add_transform(
                "t",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=np.eye(4),
                prov="act_nope",
            )

        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _open_writer(tmp_path / "p.medh5", frames=True) as w,
        ):
            dangling_prov(w)
        assert exc.value.code == "E601"

        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _open_writer(tmp_path / "m.medh5", frames=True) as w,
        ):
            w.add_transform(
                "t",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=np.eye(4),
                metrics="nope",
            )
        assert exc.value.code == "E602"

    def test_S12_3_and_S11_4_timestamps_are_RFC_3339(self):
        from medh5.curation.identity import Deidentification, SplitClaim

        with pytest.raises(MEDH5ValidationError) as exc:
            SplitClaim(set_id="cv", partition="train", assigned_at="yesterday")
        assert exc.value.code == "E604"
        with pytest.raises(MEDH5ValidationError) as exc:
            Deidentification(method="m", date="yesterday")
        assert exc.value.code == "E604"
        assert SplitClaim(
            set_id="cv", partition="train", assigned_at="2026-09-04T10:00:00Z"
        )

    def test_S3_2_a_time_axis_carries_its_timings(self, tmp_path: Path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "t.medh5", sample_id="t") as w,
        ):
            w.add_grid(
                "dce",
                shape=(3, *SHAPE),
                spacing=(1.0, 1.0, 1.0),
                axis_kinds=("time", "spatial", "spatial", "spatial"),
                axis_names=("t", "z", "y", "x"),
                timepoint="tp0",
            )
        assert exc.value.code == "E109"


class TestW3LooseEnds:
    def test_open_is_read_only(self, tmp_path: Path):
        path = tmp_path / "ro.medh5"
        with _open_writer(path):
            pass
        with pytest.raises(MEDH5ValidationError, match="read-only"):
            medh5.open(path, "r+")
        with medh5.open(path, "r") as sample:
            assert sample.identity.sample_id == "ro"

    def test_S9_an_empty_contours_annotation_is_a_verified_negative(
        self, tmp_path: Path
    ):
        path = tmp_path / "empty.medh5"
        with _open_writer(path) as w:
            w.add_contours("outlines", [], grid="g", annotated_classes=["lesion"])
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["outlines"]
            assert len(ann) == 0
            assert list(ann.polygons()) == []
            assert ann.annotated_class_ids == (3,)
            assert ann.is_annotated("lesion")
            assert ann.vertices.shape == (0, 3)
        from medh5.annotations.geometric import encode_contours

        with pytest.raises(MEDH5ValidationError) as exc:
            encode_contours([])  # no grid to shape it from
        assert exc.value.code == "E410"
        assert encode_contours([], ndim=2).datasets["vertices"].shape == (0, 2)

    def test_S2_1_a_collection_from_another_major_is_refused(self, tmp_path: Path):
        from medh5.collection import open_any, open_collection, pack

        member = tmp_path / "m.medh5"
        with _open_writer(member):
            pass
        shard = tmp_path / "s.medh5c"
        pack([member], shard)
        with open_h5(shard, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            open_collection(shard)
        with pytest.raises(MEDH5VersionError):
            open_any(shard)
        with open_h5(member, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            pack([member], tmp_path / "t.medh5c")

    def test_S14_4_temporary_names_are_unique_within_a_process(self):
        names = {_temporary_name("x.medh5") for _ in range(50)}
        assert len(names) == 50
        assert all(n.startswith(".x.medh5.tmp-") for n in names)

    def test_manifest_fields_are_fields(self, tmp_path: Path):
        from medh5.dataset.manifest import scan

        with _open_writer(tmp_path / "a.medh5"):
            pass
        manifest, _ = scan(tmp_path)
        entry = manifest.entries[0]
        assert entry.field("cohort.site_id") is None
        assert entry.field("subject_id") == "a"
        with pytest.raises(MEDH5ValidationError, match="not a manifest field"):
            entry.field("to_json")

    def test_S10_frame_graph_draws_what_the_resolver_walks(self, tmp_path: Path):
        from medh5.transforms.base import frame_graph

        path = tmp_path / "graph.medh5"
        with _open_writer(path, frames=True) as w:
            w.add_grid(
                "h",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="f1",
            )
            # Declared invertible, no stored inverse: not traversable backwards.
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="f0",
                to_frame="f1",
                field=np.zeros((3, *SHAPE), dtype=np.float32),
                field_grid="g",
                invertible=True,
            )
        with medh5.open(path) as sample:
            graph = frame_graph(dict(sample.transforms))
            assert graph["f0"] == ["f1"]
            assert graph["f1"] == []
            assert sample.resolve_frames("f1", "f0") is None

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
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="tp0",
                frame_uid="f0",
            )
            w.add_image("CT", _image(), grid="g", modality="CT")
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

    def test_dicom_enhanced_objects_and_monochrome1_are_reported(self, tmp_path: Path):
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        from medh5.io.dicom import from_dicom, read_series, scan_dicom
        from medh5.io.report import ConversionReport
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dicom"
        series = write_dicom_series(
            root / "ct", patient_id="P", study_uid=generate_uid(), study_date="20260101"
        )
        for file in series["paths"]:
            ds = pydicom.dcmread(file)
            ds.PhotometricInterpretation = "MONOCHROME1"
            ds.save_as(file, enforce_file_format=True)
        enhanced = Dataset()
        enhanced.file_meta = FileMetaDataset()
        enhanced.file_meta.MediaStorageSOPClassUID = pydicom.uid.EnhancedMRImageStorage
        enhanced.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        enhanced.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        enhanced.SOPClassUID = pydicom.uid.EnhancedMRImageStorage
        enhanced.SOPInstanceUID = enhanced.file_meta.MediaStorageSOPInstanceUID
        enhanced.Modality = "MR"
        enhanced.SeriesInstanceUID = generate_uid()
        enhanced.NumberOfFrames = 4
        (root / "mr").mkdir()
        enhanced.save_as(str(root / "mr" / "enh.dcm"), enforce_file_format=True)

        report = ConversionReport(converter="test")
        found = scan_dicom(root, report=report)
        assert [s.modality for s in found] == ["CT"]
        unsupported = report.of_kind("unsupported")
        assert unsupported and unsupported[0].severity == "warning"
        assert "MR" not in [s.modality for s in found]

        report = ConversionReport(converter="test")
        read_series(found[0], report=report)
        assert report.of_kind("photometric")

        report = from_dicom(root, tmp_path / "out.medh5")
        assert not report.ok  # both are warnings, and warnings fail the command
        with medh5.open(tmp_path / "out.medh5") as sample:
            acquisition = sample.document.acquisition["CT_tp0"]
            assert acquisition["PhotometricInterpretation"] == "MONOCHROME1"

    def test_a_mask_is_never_the_foreground_annotation(self, tmp_path: Path):
        from medh5.sampling import PatchSampler

        path = tmp_path / "fov.medh5"
        with _open_writer(path) as w:
            w.add_mask("fov", np.ones(SHAPE, dtype=bool), grid="g")
        with medh5.open(path) as sample:
            sampler = PatchSampler(4, strategy="foreground")
            assert sampler._annotation(sample, None) is None
            patch = sampler.draw(sample, None, np.random.default_rng(0))
            assert patch.strategy == "uniform"
            assert patch.used_index is None


class TestW4PerformanceRewritesAreExact:
    @pytest.mark.parametrize("extrapolation", ["zero", "nearest", "error"])
    def test_S10_4_windowed_sampling_equals_the_whole_field(
        self, tmp_path: Path, extrapolation: str
    ):
        from medh5.transforms.apply import sample_field

        rng = np.random.default_rng(7)
        field = rng.normal(0.0, 2.0, (3, 12, 14, 16)).astype(np.float32)
        path = tmp_path / "field.medh5"
        with medh5.create(path, sample_id="f") as w:
            w.add_grid("g", shape=(12, 14, 16), spacing=(1.0, 1.0, 1.0), frame_uid="f0")
            w.add_image("CT", np.zeros((12, 14, 16), np.int16), grid="g", modality="CT")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="f0",
                to_frame="f1",
                field=field,
                field_grid="g",
                extrapolation=extrapolation,
            )
        inside = rng.uniform([0, 0, 0], [11, 13, 15], size=(40, 3))
        edges = np.array(
            [
                [-0.5, 0.0, 0.0],
                [11.5, 13.5, 15.5],
                [0.0, -0.49, 15.4],
                [11.0, 13.0, 0.0],
            ]
        )
        outside = np.array([[-3.0, 2.0, 2.0], [14.0, 2.0, 2.0], [2.0, 2.0, 40.0]])
        with medh5.open(path) as sample:
            transform: Any = sample.transforms["warp"]
            for points in (inside, edges, np.vstack([inside, edges])):
                expected = sample_field(field, points, extrapolation=extrapolation)
                assert np.array_equal(transform._sample(points), expected)
            if extrapolation == "error":
                with pytest.raises(MEDH5ValidationError, match="outside"):
                    transform._sample(outside)
            else:
                expected = sample_field(field, outside, extrapolation=extrapolation)
                assert np.array_equal(transform._sample(outside), expected)
                mixed = np.vstack([inside[:3], outside])
                assert np.array_equal(
                    transform._sample(mixed),
                    sample_field(field, mixed, extrapolation=extrapolation),
                )
            # And a single far-away point along one axis only.
            lone = np.array([[2.0, 30.0, 2.0]])
            if extrapolation != "error":
                assert np.array_equal(
                    transform._sample(lone),
                    sample_field(field, lone, extrapolation=extrapolation),
                )

    def test_S7_4_instance_columns_are_read_once(self, tmp_path: Path):
        path = tmp_path / "inst.medh5"
        with _open_writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(class_id=3, instance_id=7, mask=_mask(), score=0.5),
                    InstanceInput(class_id=3, instance_id=8, mask=_mask((5, 5, 5), 2)),
                ],
            )
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["les"]
            first = ann.boxes
            assert ann._table() is ann._table()
            assert ann.boxes is first
            assert [o.instance_id for o in ann.instances()] == [7, 8]
            assert ann.dense([3])[0].sum() == int(_mask().sum() + 8)
            assert ann.scores is not None and np.isnan(ann.scores[1])
            crop = ann.crop(1)
            assert crop is not None and crop.shape == (2, 2, 2) and crop.all()

    def test_S7_7_labelmap_and_mask_scan_in_slabs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("medh5.annotations.voxel.payload.SLAB_BYTES", 1)
        ignore = np.zeros(SHAPE, dtype=bool)
        ignore[-1] = True
        path = tmp_path / "slabs.medh5"
        with _open_writer(path) as w:
            w.add_segmentation(
                "lm", grid="g", masks={3: _mask()}, encoding="labelmap", ignore=ignore
            )
            w.add_mask("fov", _mask((0, 0, 0), 4), grid="g")
        with medh5.open(path) as sample:
            assert sample.annotations["lm"].has_ignore_region
            assert sample.annotations["fov"].summary()["true_voxels"] == 64

    def test_S10_resolve_frames_is_memoised_per_handle(self, tmp_path: Path):
        path = tmp_path / "memo.medh5"
        with _open_writer(path, frames=True) as w:
            w.add_transform(
                "t", kind="affine", from_frame="f0", to_frame="f1", matrix=np.eye(4)
            )
        with medh5.open(path) as sample:
            first = sample.resolve_frames("f0", "f1")
            assert first is sample.resolve_frames("f0", "f1")
            assert first is not None and first.transform_id == "t"
            assert sample.resolve_frames("f0", "f0") is None
            # The reverse is an analytic inverse, resolved and memoised too.
            back = sample.resolve_frames("f1", "f0")
            assert back is sample.resolve_frames("f1", "f0")

    def test_bench_measures_a_paired_centre(self, tmp_path: Path):
        from medh5.bench import benchmark_file, synthetic_pair

        pair = synthetic_pair(tmp_path, shape=(6, 8, 8))
        names = [m.name for m in benchmark_file(pair, patch=4, repeats=2)]
        assert "paired_center_ms" in names


class TestW5Structure:
    def test_the_writer_moved_and_every_name_stayed(self):
        import medh5.sample as sample_module
        import medh5.writer as writer_module

        assert sample_module.SampleWriter is writer_module.SampleWriter
        assert medh5.SampleWriter is writer_module.SampleWriter
        assert medh5.create is writer_module.create
        assert medh5.amend is writer_module.amend
        assert "SampleWriter" in sample_module.__all__

    def test_S5_2_keys_are_schema_valid_wherever_they_are_minted(self):
        from medh5.io._common import sanitize_key, sanitize_stem

        assert sanitize_key("GTV-1") == "gtv_1"
        assert sanitize_key("Tumour Core") == "tumour_core"
        assert sanitize_key("Liver.L") == "liver_l"
        assert sanitize_key("_x") == "x"
        assert sanitize_key("", fallback="roi") == "roi"
        assert sanitize_key("é") == "class"
        assert sanitize_stem("a b/c", limit=4) == "a_b_"

    def test_optional_dependencies_name_their_extra(self):
        from medh5._optional import require

        with pytest.raises(ImportError, match=r"medh5\[dicomseg\]"):
            require("no_such_module_medh5", extra="dicomseg", purpose="testing")
        assert require("json", extra="x", purpose="y").dumps({}) == "{}"

    def test_one_extrapolations_vocabulary(self):
        from medh5.transforms import apply, base

        assert base.EXTRAPOLATIONS is apply.EXTRAPOLATIONS


class TestW6Tooling:
    def test_fsync_leaves_a_trailing_control_z_alone(self, tmp_path: Path):
        """`_fsync_path` opens a writable descriptor on Windows, and the C runtime's
        text mode treats a trailing 0x1A as an end-of-file mark it strips on open.
        The Windows job found four samples in eleven hundred one byte short."""
        from medh5._hdf5 import _fsync_path

        path = tmp_path / "ctrlz.bin"
        payload = bytes(range(256)) + b"\x1a"
        path.write_bytes(payload)
        _fsync_path(path)
        assert path.read_bytes() == payload

    def test_the_lint_gate_refuses_suppressions_that_suppress_nothing(self):
        text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        assert '"RUF100"' in text
