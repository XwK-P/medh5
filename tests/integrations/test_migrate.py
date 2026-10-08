"""0.x files: the reader and the migration (spec Appendix B)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5FileError, MEDH5SchemaError
from medh5.io.report import ConversionReport
from medh5.validate import validate_file
from tests.helpers import write_legacy_sample


class TestMigrate:
    @pytest.fixture
    def legacy(self, tmp_path: Path) -> list[Path]:
        shape = (8, 12, 16)
        rng = np.random.default_rng(3)
        liver = np.zeros(shape, bool)
        liver[1:6, 2:9, 3:12] = True
        lesion = np.zeros(shape, bool)
        lesion[2:4, 4:6, 5:8] = True
        paths = []
        for index, patient in enumerate(("PAT-A", "PAT-A", "PAT-B")):
            path = tmp_path / f"old_{index}.medh5"
            write_legacy_sample(
                path,
                images={"CT": rng.integers(-1000, 1500, shape).astype(np.int16)},
                seg={"liver": liver, "lesion": lesion},
                bboxes=np.array([[[2, 6], [4, 9], [5, 11]]], dtype=np.int32),
                bbox_labels=["lesion"],
                bbox_scores=np.array([0.9], np.float32),
                spacing=[2.0, 0.8, 0.9],
                origin=[1.0, 2.0, 3.0],
                coord_system="LPS",
                label=1,
                label_name="lesion",
                extra={
                    "patient_id": patient,
                    "study_date": f"2026-0{index + 1}-15",
                    "review": {
                        "reviewer": "RAD-07",
                        "status": "approved",
                        "date": "2026-03-01",
                    },
                },
            )
            paths.append(path)
        return paths

    def test_appendix_B_one_file_becomes_one_sample(self, legacy, tmp_path):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.timepoints.ids == ("tp0",)
            assert sample.grids["ref"].timepoint == "tp0"
            assert {"core", "seg", "det", "cls"} <= sample.profiles
            assert sample.annotations["seg"].kind in ("layers", "labelmap", "bitmask")
        assert report.of_kind("encoding")
        assert not validate_file(tmp_path / "new.medh5").errors

    def test_S8_1_boxes_shift_by_half_a_voxel_and_round_trip_to_slices(
        self, legacy, tmp_path
    ):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            boxes = sample.annotations["boxes"]
            assert np.allclose(boxes.boxes[0], [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]])
            assert boxes.as_slices()[0] == (
                slice(2, 6),
                slice(4, 9),
                slice(5, 11),
            )
        note = report.of_kind("box_convention")[0]
        assert note.detail["shift"] == -0.5

    def test_masks_survive_and_coverage_is_flagged(self, legacy, tmp_path):
        from medh5.io.legacy import migrate, read_legacy

        migrate(legacy[0], tmp_path / "new.medh5")
        original = read_legacy(legacy[0])
        with medh5.open(tmp_path / "new.medh5") as sample:
            seg = sample.annotations["seg"]
            for name, mask in original.seg.items():
                assert np.array_equal(seg.dense([name])[0], mask), name

    def test_review_becomes_provenance_and_quality(self, legacy, tmp_path):
        from medh5.io.legacy import migrate

        report = migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.document.quality["seg"].status == "approved"
            types = [a.type for a in sample.document.provenance.activities]
            assert "review" in types
            assert sample.document.extra["legacy"]["patient_id"] == "PAT-A"
        assert report.of_kind("review")

    def test_S3_7_subject_grouping_merges_a_patients_files(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        report = migrate_paths(
            legacy,
            tmp_path / "out",
            group_by="subject",
            subject_key="extra.patient_id",
        )
        written = sorted((tmp_path / "out").glob("*.medh5"))
        assert [p.stem for p in written] == ["pat-a", "pat-b"]
        with medh5.open(tmp_path / "out" / "pat-a.medh5") as sample:
            assert sample.timepoints.ids == ("tp0", "tp1")
            assert sample.timepoints["tp1"].days_from_baseline == 31
            assert sorted(sample.images) == ["CT_tp0", "CT_tp1"]
        assert report.of_kind("instance_ids"), "merging must say it did not join ids"

    def test_the_default_is_one_sample_per_file(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        migrate_paths(legacy, tmp_path / "out")
        assert len(list((tmp_path / "out").glob("*.medh5"))) == 3

    def test_subject_grouping_without_a_key_warns(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        report = migrate_paths(legacy, tmp_path / "out", group_by="subject")
        assert report.warnings
        assert any("subject-key" in w.message for w in report.warnings)

    def test_the_label_set_is_minted_once_for_the_cohort(self, legacy, tmp_path):
        from medh5.io.legacy import build_label_set, load_sidecar, write_sidecar

        report = ConversionReport()
        label_set = build_label_set(legacy, report=report)
        assert {c.key for c in label_set} == {"liver", "lesion"}
        sidecar = write_sidecar(label_set, tmp_path / "labels.json")
        assert {c.key for c in load_sidecar(sidecar)} == {"liver", "lesion"}
        assert report.of_kind("label_set")

    def test_an_unreadable_file_is_reported_not_fatal(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        report = migrate_paths([*legacy, broken], tmp_path / "out")
        assert report.warnings
        assert len(list((tmp_path / "out").glob("*.medh5"))) == 3


class TestLegacyReader:
    """1.0 ships a reader for the 0.x layout, not an implementation of it."""

    def test_the_0x_package_is_gone(self):
        with pytest.raises(ImportError):
            import medh5.legacy  # noqa: F401

    def test_a_1_0_file_is_refused_as_0_x(self, sample_path):
        from medh5.io.legacy import is_legacy, read_legacy

        with pytest.raises(MEDH5SchemaError, match="1.0 file"):
            read_legacy(sample_path)
        assert not is_legacy(sample_path)

    def test_a_non_hdf5_file_is_refused(self, tmp_path):
        from medh5.io.legacy import is_legacy, read_legacy

        path = tmp_path / "junk.medh5"
        path.write_bytes(b"not hdf5")
        with pytest.raises(MEDH5FileError):
            read_legacy(path)
        assert not is_legacy(path)

    def test_a_future_0_x_schema_version_is_refused(self, tmp_path):
        from medh5.io.legacy import legacy_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["schema_version"] = "2"
        with pytest.raises(MEDH5SchemaError, match="schema version"):
            legacy_meta(path)

    def test_the_file_beats_its_own_flags(self, tmp_path):
        """0.x denormalised `has_seg`/`seg_names`, and they could drift."""
        from medh5.io.legacy import read_legacy

        mask = np.zeros((2, 3, 4), bool)
        mask[0, 1, 2] = True
        path = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((2, 3, 4), np.int16)},
            seg={"liver": mask},
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["has_seg"] = False
            del handle.attrs["seg_names"]
        sample = read_legacy(path)
        assert list(sample.seg) == ["liver"]
        assert sample.meta.seg_names == ["liver"]

    def test_geometry_and_extra_round_trip(self, tmp_path):
        from medh5.io.legacy import read_legacy

        direction = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        path = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((2, 3, 4), np.int16)},
            spacing=[2.0, 0.8, 0.9],
            origin=[1.0, 2.0, 3.0],
            direction=direction,
            coord_system="LPS",
            patch_size=[2, 2, 2],
            label=1,
            label_name="lesion",
            extra={"patient_id": "PAT-A"},
        )
        sample = read_legacy(path)
        assert sample.meta.spatial.direction == direction
        assert sample.meta.spatial.spacing == [2.0, 0.8, 0.9]
        assert sample.meta.patch_size == [2, 2, 2]
        assert sample.meta.label == 1
        assert sample.meta.extra["patient_id"] == "PAT-A"

    def test_a_malformed_direction_is_named(self, tmp_path):
        from medh5.io.legacy import legacy_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle["images"].attrs["direction"] = np.zeros(4, np.float64)
        with pytest.raises(MEDH5SchemaError, match="4 element"):
            legacy_meta(path)

    def test_malformed_extra_is_named(self, tmp_path):
        from medh5.io.legacy import legacy_meta

        path = write_legacy_sample(
            tmp_path / "old.medh5", images={"CT": np.zeros((2, 3, 4), np.int16)}
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["extra"] = "{not json"
        with pytest.raises(MEDH5SchemaError, match="not JSON"):
            legacy_meta(path)
