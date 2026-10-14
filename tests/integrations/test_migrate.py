"""0.x files: the reader and the migration (spec Appendix B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5FileError, MEDH5SchemaError, MEDH5ValidationError
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
            assert sample.document.extra["legacy"]["tp0"]["patient_id"] == "PAT-A"
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

    def test_a_sidecar_is_replaced_whole(self, legacy, tmp_path):
        """A sidecar was written in place, so a write cut short left the
        label set under review truncated."""
        from medh5.io.legacy import build_label_set, load_sidecar, write_sidecar

        label_set = build_label_set(legacy)
        sidecar = write_sidecar(label_set, tmp_path / "labels.json")
        before = sidecar.stat().st_ino
        assert write_sidecar(label_set, sidecar) == sidecar
        assert sidecar.stat().st_ino != before, "a new file, not the old rewritten"
        assert {c.key for c in load_sidecar(sidecar)} == {"liver", "lesion"}

    def test_an_unreadable_file_is_reported_not_fatal(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        report = migrate_paths([*legacy, broken], tmp_path / "out")
        assert report.warnings
        assert len(list((tmp_path / "out").glob("*.medh5"))) == 3

    def test_a_label_set_names_the_files_it_could_not_read(self, legacy, tmp_path):
        """``migrate --write-labels`` passed over a file it could not read
        and wrote a label set without that file's classes, with exit status
        0.  Each run that mints a label set reports such a file once."""
        from medh5.cli import EXIT_ERROR, main
        from medh5.io.legacy import build_label_set, migrate_paths

        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        report = ConversionReport()
        label_set = build_label_set([*legacy, broken], report=report)
        assert {c.key for c in label_set} == {"liver", "lesion"}
        (note,) = report.of_kind("unreadable")
        assert note.severity == "warning" and note.detail == {"path": str(broken)}
        report = migrate_paths([*legacy, broken], tmp_path / "out")
        assert [n.detail["path"] for n in report.of_kind("unreadable")] == [str(broken)]
        labels = tmp_path / "labels.json"
        argv = ["migrate", *map(str, [*legacy, broken]), "-o", str(tmp_path / "o")]
        assert main([*argv, "--write-labels", str(labels)]) == EXIT_ERROR
        assert labels.exists(), "the label set of the files it could read"

    def test_a_file_that_cannot_be_migrated_is_reported_and_the_rest_are(
        self, legacy, tmp_path
    ):
        """The first file whose content the writer refused ended the cohort,
        with every file after it unwritten, where the docstring promised a
        report and the rest."""
        from medh5.io.legacy import migrate_paths

        with h5py.File(legacy[1], "r+") as handle:
            del handle["seg/liver"]
            handle["seg"].create_dataset("liver", data=np.zeros((2, 2, 2), bool))
            # A guess made before the failure, about a file never written.
            del handle["images"].attrs["coord_system"]
        report = migrate_paths(legacy, tmp_path / "out")
        written = sorted(p.stem for p in (tmp_path / "out").glob("*.medh5"))
        assert written == ["old_0", "old_2"]
        failed = report.of_kind("failed")
        assert len(failed) == 1 and failed[0].severity == "warning"
        assert failed[0].detail["inputs"] == [str(legacy[1])]
        assert "E405" in failed[0].message
        # Nothing was written for it, so nothing else describes it.
        about = [n.kind for n in report.notes if str(legacy[1]) in n.message]
        assert about == ["failed"]

    def test_review_is_the_quality_of_each_migrated_annotation(self, legacy, tmp_path):
        """The review became a quality record keyed ``seg`` that no annotation
        named, and an activity whose output was ``seg`` whatever the file
        held.  Each annotation migrated from the file carries it now, and the
        activity names them; the dict stays verbatim in ``extra.legacy``."""
        from medh5.io.legacy import migrate

        migrate(legacy[0], tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            for name in ("seg", "boxes", "label"):
                annotation = sample.annotations[name]
                assert annotation.quality_key == name
                assert sample.document.quality[name].status == "approved"
            (review,) = sample.document.provenance.activities_by_type("review")
            assert sorted(review.outputs) == [
                "annotations/boxes",
                "annotations/label",
                "annotations/seg",
            ]
            legacy_review = sample.document.extra["legacy"]["tp0"]["review"]
            assert legacy_review == {
                "reviewer": "RAD-07",
                "status": "approved",
                "date": "2026-03-01",
            }

    def test_a_review_of_boxes_alone_names_no_segmentation(self, tmp_path):
        from medh5.io.legacy import migrate

        old = write_legacy_sample(
            tmp_path / "boxes.medh5",
            images={"CT": np.zeros((4, 6, 6), np.int16)},
            bboxes=np.array([[[1, 3], [1, 4], [2, 5]]], dtype=np.int32),
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
            extra={"review": {"reviewer": "RAD-07", "status": "approved"}},
        )
        migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sorted(sample.document.quality) == ["boxes"]
            (review,) = sample.document.provenance.activities_by_type("review")
            assert review.outputs == ("annotations/boxes",)
        assert not validate_file(tmp_path / "new.medh5").errors

    @pytest.mark.parametrize(
        ("status", "quality", "reviewed_by", "guessed"),
        [
            ("pending", "submitted", False, True),
            ("flagged", "submitted", True, True),
            ("reviewed", "reviewed", True, False),
            ("rejected", "rejected", True, False),
        ],
    )
    def test_appendix_B_a_review_the_0x_api_wrote_migrates(
        self, tmp_path, status, quality, reviewed_by, guessed
    ):
        """0.x's ``set_review_status`` wrote ``annotator`` and ``timestamp``,
        which were never read: the activity had no agent and every such file
        failed (E005), and ``pending`` or ``flagged`` became ``reviewed``."""
        from medh5.io.legacy import migrate

        record = {
            "status": status,
            "annotator": "RAD-07",
            "timestamp": "2026-01-01T10:00:00.123456+00:00",
            "notes": "margin unclear",
            "history": [
                {
                    "status": "pending",
                    "annotator": None,
                    "timestamp": "2025-12-01T09:00:00+00:00",
                    "notes": None,
                }
            ],
        }
        old = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((4, 5, 6), np.int16)},
            seg={"liver": np.ones((4, 5, 6), bool)},
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
            extra={"review": record},
        )
        report = migrate(old, tmp_path / "new.medh5")
        assert not validate_file(tmp_path / "new.medh5").errors
        with medh5.open(tmp_path / "new.medh5") as sample:
            document = sample.document
            seg = document.quality["seg"]
            assert seg.status == quality
            assert [i.code for i in seg.issues] == (
                ["flagged"] if status == "flagged" else []
            )
            (review,) = document.provenance.activities_by_type("review")
            assert document.provenance.agent(review.agent).name == "RAD-07"
            assert bool(seg.reviewed_by) is reviewed_by
            assert review.ended == record["timestamp"]
            assert review.params == {"verdict": status}
            assert document.extra["legacy"]["tp0"]["review"] == record
        guesses = [n for n in report.guesses if n.kind == "review"]
        assert bool(guesses) is guessed

    def test_a_review_naming_no_one_is_the_import_tools(self, tmp_path):
        """With no reviewer, the activity had no agent, which the schema
        refuses: the import tool that recorded it is its agent, as a guess."""
        from medh5.io.legacy import migrate

        old = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((4, 5, 6), np.int16)},
            seg={"liver": np.ones((4, 5, 6), bool)},
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
            extra={"review": {"status": "reviewed"}},
        )
        report = migrate(old, tmp_path / "new.medh5")
        assert not validate_file(tmp_path / "new.medh5").errors
        with medh5.open(tmp_path / "new.medh5") as sample:
            provenance = sample.document.provenance
            (review,) = provenance.activities_by_type("review")
            assert provenance.agent(review.agent).type == "software"
            assert sample.document.quality["seg"].reviewed_by == ()
        assert any("import tool" in n.message for n in report.guesses)

    def test_S12_1_each_study_of_a_subject_has_its_own_sample_id(
        self, legacy, tmp_path
    ):
        """In study mode with a subject key, every study of a subject was
        given the subject's sample id; the id is now the output's name."""
        from medh5.io.legacy import migrate_paths

        migrate_paths(legacy, tmp_path / "out", subject_key="extra.patient_id")
        identities = []
        for path in sorted((tmp_path / "out").glob("*.medh5")):
            with medh5.open(path) as sample:
                identity = sample.document.identity
                assert identity.sample_id == path.stem
                identities.append((identity.sample_id, identity.subject_id))
        assert len({sample_id for sample_id, _ in identities}) == 3
        assert sorted(s for _, s in identities) == ["PAT-A", "PAT-A", "PAT-B"]

    def test_W1_no_local_path_reaches_the_sample(self, legacy, tmp_path):
        """A study with no subject was ``study:<its absolute path>``, and its
        sample id the path made an identifier; `migrate` used the bare stem
        as the subject.  Both now name it for its stem, the same way."""
        from medh5.io.legacy import migrate, migrate_paths

        where = tmp_path / "MRN-0042 Doe"
        where.mkdir()
        old = where / "Case 7.medh5"
        old.write_bytes(legacy[0].read_bytes())
        migrate_paths([old], tmp_path / "out")
        migrate(old, tmp_path / "single.medh5")
        for path in (tmp_path / "out" / "case_7.medh5", tmp_path / "single.medh5"):
            with medh5.open(path) as sample:
                identity = sample.document.identity
                assert (identity.sample_id, identity.subject_id) == (
                    "case_7",
                    "study:case_7",
                )
                assert "MRN-0042" not in json.dumps(sample.document.to_json())

    def test_an_mtime_order_is_reported_once_per_subject(self, legacy, tmp_path):
        from medh5.io.legacy import migrate_paths

        for path in legacy:
            with h5py.File(path, "r+") as handle:
                extra = json.loads(handle.attrs["extra"])
                del extra["study_date"]
                handle.attrs["extra"] = json.dumps(extra)
        report = migrate_paths(
            legacy, tmp_path / "out", group_by="subject", subject_key="extra.patient_id"
        )
        orders = report.of_kind("timepoint_order")
        assert [n.detail["subject"] for n in orders] == ["PAT-A"]


class TestUncalibrated:
    """A 0.x file need not state its geometry (Appendix B)."""

    def test_S3_5_a_file_without_spacing_becomes_a_px_grid(self, tmp_path: Path):
        """It became a millimetre grid of unit voxels, a size nobody measured,
        and the report said nothing of it."""
        from medh5.io.legacy import migrate

        old = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"XR": np.zeros((6, 8), np.uint16)},
            bboxes=np.array([[[1, 3], [2, 5]]], dtype=np.int32),
        )
        report = migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            grid = sample.grids["ref"]
            assert grid.units == "px"
            assert grid.spacing == (1.0, 1.0)
            assert grid.coord_system == "LPS"
        notes = {n.severity: n for n in report.of_kind("geometry")}
        assert notes["decision"].detail["units"] == "px"
        assert notes["guess"].detail["coord_system"] == "LPS"
        assert not validate_file(tmp_path / "new.medh5").errors

    def test_a_stated_geometry_is_kept_without_a_note(self, tmp_path: Path):
        from medh5.io.legacy import migrate

        old = write_legacy_sample(
            tmp_path / "old.medh5",
            images={"CT": np.zeros((2, 3, 4), np.int16)},
            spacing=[2.0, 0.8, 0.9],
            coord_system="RAS",
        )
        report = migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.grids["ref"].units == "mm"
            assert sample.grids["ref"].coord_system == "RAS"
        assert not report.of_kind("geometry")


class TestLabelKeys:
    """A class is its key: names that sanitize alike are one class."""

    @staticmethod
    def _old(path: Path, masks: dict[str, Any], **extra: Any) -> Path:
        return write_legacy_sample(
            path,
            images={"CT": np.zeros((4, 6, 6), np.int16)},
            seg=masks,
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
            extra=extra or None,
        )

    @staticmethod
    def _mask(*index: slice) -> Any:
        mask = np.zeros((4, 6, 6), bool)
        mask[index] = True
        return mask

    def test_appendix_B_names_of_one_key_are_one_class(self, tmp_path: Path):
        """The guide's own example: ``liver`` and ``Liver`` both keyed
        ``liver``, the label set held the key twice, and step 1 failed (E302)
        without writing the file it tells the curator to review."""
        from medh5.io.legacy import build_label_set, load_sidecar, write_sidecar

        paths = [
            self._old(tmp_path / f"old_{i}.medh5", {name: self._mask(slice(0, 2))})
            for i, name in enumerate(("liver", "Liver", "liver_1"))
        ]
        report = ConversionReport()
        label_set = build_label_set(paths, report=report)
        assert {c.key for c in label_set} == {"liver", "liver_1"}
        assert label_set["liver"].name == "liver"
        merged = [n for n in report.guesses if n.kind == "label_set"]
        assert [n.detail["names"] for n in merged] == [["liver", "Liver"]]
        sidecar = write_sidecar(label_set, tmp_path / "labels.json")
        assert {c.key for c in load_sidecar(sidecar)} == {"liver", "liver_1"}

    def test_two_masks_of_one_class_are_stored_as_their_union(self, tmp_path: Path):
        """One file's ``liver`` and ``Liver``: the second replaced the first."""
        from medh5.io.legacy import migrate

        upper, lower = self._mask(slice(0, 2)), self._mask(slice(2, 4))
        old = self._old(tmp_path / "old.medh5", {"liver": upper, "Liver": lower})
        report = migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            seg = sample.annotations["seg"]
            assert len(seg.class_ids) == 1
            assert np.array_equal(seg.dense(["liver"])[0], upper | lower)
        assert any(n.detail.get("classes") == 1 for n in report.of_kind("label_set"))

    def test_conflicting_nnunet_ids_keep_the_first_and_warn(self, tmp_path: Path):
        """``liver`` was id 1 in one file's dataset.json and 2 in another's:
        the last silently won.  And two keys asking for one id made a label
        set with a duplicate id."""
        from medh5.io.legacy import build_label_set

        mask = self._mask(slice(0, 2))
        paths = [
            self._old(
                tmp_path / "a.medh5",
                {"liver": mask},
                nnunetv2={"labels": {"background": 0, "liver": 1}},
            ),
            self._old(
                tmp_path / "b.medh5",
                {"liver": mask, "spleen": mask},
                nnunetv2={"labels": {"background": 0, "liver": 2, "spleen": 1}},
            ),
        ]
        report = ConversionReport()
        label_set = build_label_set(paths, report=report)
        assert label_set["liver"].id == 1
        assert label_set["spleen"].id not in (0, 1)
        warnings = [w for w in report.warnings if w.kind == "label_set"]
        assert len(warnings) == 2
        assert warnings[0].detail["kept"] == 1 and warnings[0].detail["ignored"] == 2
        assert warnings[1].detail["kept_by"] == "liver"


class TestOlderLayouts:
    @staticmethod
    def _old(path: Path) -> Path:
        return write_legacy_sample(
            path,
            images={"CT": np.zeros((4, 5, 6), np.int16)},
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
        )

    def test_appendix_B_0x_up_to_0_5_kept_shape_on_the_root(self, tmp_path: Path):
        from medh5.io.legacy import legacy_meta

        old = self._old(tmp_path / "old.medh5")
        with h5py.File(old, "r+") as handle:
            del handle["images"].attrs["shape"]
            handle.attrs["shape"] = np.asarray([4, 5, 6], np.int64)
        assert legacy_meta(old).shape == [4, 5, 6]

    def test_appendix_B_an_extra_that_is_not_an_object_is_kept(self, tmp_path: Path):
        from medh5.io.legacy import migrate

        old = self._old(tmp_path / "old.medh5")
        with h5py.File(old, "r+") as handle:
            handle.attrs["extra"] = json.dumps(["kept", 7])
        migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.document.extra["legacy"]["tp0"] == {"extra": ["kept", 7]}


class TestNoOutputAliasesAnInput:
    """F11: ``migrate -o .`` replaced the 0.x originals, and a ``--report`` or
    ``--write-labels`` naming an input made it JSON; each exited 0."""

    @pytest.fixture
    def old(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        monkeypatch.chdir(tmp_path)
        paths = [
            TestOlderLayouts._old(tmp_path / f"old_{index}.medh5") for index in (0, 1)
        ]
        (tmp_path / "link.json").symlink_to(paths[0])
        return paths

    @pytest.mark.parametrize(
        ("args", "converts"),
        [
            (["-o", "."], True),
            (["-o", "out", "--report", "old_0.medh5"], False),
            (["-o", "out", "--report", "../{here}/old_1.medh5"], False),
            (["-o", "out", "--report", "link.json"], False),
            (["-o", "out", "--write-labels", "old_1.medh5"], False),
            (["-o", "out", "--report", "."], False),
            (["-o", "out", "--report", "nodir/r.json"], False),
            (["-o", "out", "--report", "out/old_0.medh5"], True),
        ],
    )
    def test_F11_no_output_path_aliases_an_input(self, old, tmp_path, args, converts):
        from medh5.cli import EXIT_ERROR, main

        before = [p.read_bytes() for p in old]
        argv = [a.replace("{here}", tmp_path.name) for a in args]
        assert main(["migrate", "old_0.medh5", "old_1.medh5", *argv]) == EXIT_ERROR
        assert [p.read_bytes() for p in old] == before
        # A refusal comes before anything is written.
        assert (tmp_path / "out").exists() is (converts and "out" in argv)
        for written in (tmp_path / "out").glob("*.medh5"):
            with medh5.open(written) as sample:
                assert sample.document.identity.sample_id == written.stem

    def test_a_report_may_go_in_the_output_directory(self, old, tmp_path):
        from medh5.cli import EXIT_OK, main

        argv = ["migrate", "old_0.medh5", "-o", "out", "--report", "./out/r.json"]
        assert main(argv) == EXIT_OK
        assert json.loads((tmp_path / "out" / "r.json").read_text())["outputs"]

    def test_F11_migrate_refuses_to_write_over_its_source(self, old):
        from medh5.io.legacy import migrate

        before = old[0].read_bytes()
        with pytest.raises(MEDH5ValidationError, match="being migrated"):
            migrate(old[0], Path(".") / old[0].name)
        assert old[0].read_bytes() == before


class TestSampleLabel:
    """Appendix B maps ``label`` and ``label_name`` to a sample-scope
    classification.  The label set minted only mask and box names, so the
    label was dropped unless its name happened to be one of them."""

    @staticmethod
    def _old(path: Path, **label: Any) -> Path:
        shape = (4, 5, 6)
        tumor = np.zeros(shape, bool)
        tumor[1:3, 1:3, 1:3] = True
        return write_legacy_sample(
            path,
            images={"CT": np.zeros(shape, np.int16)},
            seg={"tumor": tumor, "liver": np.ones(shape, bool)},
            spacing=[1.0, 1.0, 1.0],
            coord_system="LPS",
            **label,
        )

    def test_appendix_B_the_0x_readme_example_keeps_its_label(self, tmp_path: Path):
        from medh5.io.legacy import migrate

        old = self._old(tmp_path / "old.medh5", label=1, label_name="malignant")
        report = migrate(old, tmp_path / "new.medh5")
        assert report.ok, [n.message for n in report.warnings]
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.annotations["label"].positives == ("malignant",)
            keys = sorted(c.key for c in sample.label_set)
            assert keys == ["liver", "malignant", "tumor"]
        (note,) = report.of_kind("label")
        assert note.severity == "decision"
        assert note.detail["label"] == 1
        assert not validate_file(tmp_path / "new.medh5").errors

    def test_appendix_B_a_classification_cohort_keeps_every_label(self, tmp_path: Path):
        from medh5.io.legacy import migrate_paths

        paths = [
            self._old(tmp_path / f"case_{value}.medh5", label=value) for value in (0, 1)
        ]
        report = migrate_paths(paths, tmp_path / "out")
        assert report.ok, [n.message for n in report.warnings]
        for value in (0, 1):
            with medh5.open(tmp_path / "out" / f"case_{value}.medh5") as sample:
                assert sample.annotations["label"].positives == (str(value),)
        # A bare value is not a class name anyone chose.
        assert len([n for n in report.guesses if n.kind == "label"]) == 2

    def test_a_label_name_without_a_label_is_kept_as_a_guess(self, tmp_path: Path):
        from medh5.io.legacy import migrate

        old = self._old(tmp_path / "old.medh5", label_name="benign")
        report = migrate(old, tmp_path / "new.medh5")
        with medh5.open(tmp_path / "new.medh5") as sample:
            assert sample.annotations["label"].positives == ("benign",)
        (note,) = report.of_kind("label")
        assert note.severity == "guess"


class TestLegacyExtra:
    """Appendix B puts a 0.x ``extra.nnunetv2`` at ``/meta → extra.nnunetv2``,
    where the nnU-Net exporter reads a stashed ``dataset.json``.  The whole
    ``extra`` went to ``extra.legacy`` instead, and a grouped migration kept
    only the last file's."""

    STASH = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "lesion": 2},
        "file_ending": ".nii.gz",
        "description": "a 0.x dataset",
    }

    @staticmethod
    def _old(path: Path, **extra: Any) -> Path:
        shape = (6, 8, 10)
        liver = np.zeros(shape, bool)
        liver[1:5, 2:6, 2:8] = True
        lesion = np.zeros(shape, bool)
        lesion[2:3, 6:8, 8:10] = True  # beside the liver: nnU-Net holds one class
        return write_legacy_sample(
            path,
            images={"CT": np.zeros(shape, np.int16)},
            seg={"liver": liver, "lesion": lesion},
            spacing=[2.0, 1.0, 1.0],
            coord_system="LPS",
            extra=extra,
        )

    def _migrated(self, tmp_path: Path) -> Path:
        from medh5.io.legacy import migrate

        old = self._old(tmp_path / "old.medh5", patient_id="PAT-A", nnunetv2=self.STASH)
        migrate(old, tmp_path / "case_a.medh5")
        return tmp_path / "case_a.medh5"

    def test_appendix_B_the_stash_is_at_meta_extra(self, tmp_path: Path):
        with medh5.open(self._migrated(tmp_path)) as sample:
            extra = sample.document.extra
            assert extra["nnunetv2"] == self.STASH
            assert extra["legacy"] == {"tp0": {"patient_id": "PAT-A"}}

    def test_appendix_B_the_stash_is_where_the_exporter_reads_it(self, tmp_path: Path):
        pytest.importorskip("nibabel")
        import json

        from medh5.io.nnunetv2 import to_nnunetv2

        report = to_nnunetv2([self._migrated(tmp_path)], tmp_path / "nnunet")
        written = json.loads(
            (tmp_path / "nnunet" / "Dataset001_medh5" / "dataset.json").read_text()
        )
        assert written["description"] == "a 0.x dataset"
        assert written["labels"] == self.STASH["labels"]
        assert report.of_kind("dataset_json")[0].detail["reused"]

    @pytest.mark.parametrize("same", [True, False])
    def test_appendix_B_grouped_files_keep_every_files_extra(
        self, tmp_path: Path, same: bool
    ):
        from medh5.io.legacy import migrate_paths

        other = {**self.STASH, "description": "re-exported"}
        paths = [
            self._old(
                tmp_path / f"old_{index}.medh5",
                patient_id="PAT-A",
                study_date=f"2026-0{index + 1}-15",
                site=site,
                nnunetv2=self.STASH if same or index == 0 else other,
            )
            for index, site in enumerate(("north", "south"))
        ]
        report = migrate_paths(
            paths, tmp_path / "out", group_by="subject", subject_key="extra.patient_id"
        )
        with medh5.open(tmp_path / "out" / "pat-a.medh5") as sample:
            extra = sample.document.extra
            assert extra["nnunetv2"] == self.STASH, "the first file's"
            legacy = extra["legacy"]
            assert sorted(legacy) == ["tp0", "tp1"]
            assert legacy["tp0"]["site"] == "north"
            assert legacy["tp1"]["site"] == "south", "the second did not replace it"
            assert "nnunetv2" not in legacy["tp0"]
            if same:
                assert "nnunetv2" not in legacy["tp1"]
            else:
                assert legacy["tp1"]["nnunetv2"] == other, "kept, not dropped"
        conflicts = [w for w in report.warnings if w.kind == "nnunetv2"]
        assert len(conflicts) == (0 if same else 1)


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
