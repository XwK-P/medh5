"""Longitudinal joins, agreement and split audits (spec §7.4, §11.2, §12.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.curation.agreement import (
    box_iou,
    compare,
    compare_instances,
    compare_voxel,
    dice,
    iou,
)
from medh5.curation.splits import audit_splits
from medh5.curation.tracking import PRESENT, RESOLVED, UNEXAMINED, carries_instance_ids
from medh5.errors import MEDH5ValidationError
from medh5.validate import validate_file
from tests.v1.conftest import SHAPE, block, write_sample


def lesion(z: int, y: int, x: int, r: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[z - r : z + r, y - r : y + r, x - r : x + r] = True
    return mask


def write_series(
    path: Path,
    label_set,
    *,
    follow_up: list[InstanceInput] | None = None,
    annotated_tp1: list[int] | str = "all_given",
) -> Path:
    """Two visits: lesion 7 grows, lesion 8 vanishes, lesion 9 appears."""
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0", days_from_baseline=0)
        w.add_timepoint("tp1", days_from_baseline=90)
        w.label_set(label_set)
        for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
            w.add_grid(
                f"g_{tp}",
                shape=SHAPE,
                spacing=(2.0, 1.0, 1.0),
                timepoint=tp,
                frame_uid=f"pseudo:{frame}",
            )
            w.add_image(
                f"CT_{tp}",
                np.zeros(SHAPE, dtype=np.int16),
                grid=f"g_{tp}",
                modality="CT",
            )
        w.add_segmentation(
            "les_tp0",
            grid="g_tp0",
            instances=[
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 2)),
                InstanceInput(class_id=3, instance_id=8, mask=lesion(8, 16, 16, 1)),
            ],
            annotated_classes=[3],
        )
        w.add_segmentation(
            "les_tp1",
            grid="g_tp1",
            instances=follow_up
            or [
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 3)),
                InstanceInput(class_id=3, instance_id=9, mask=lesion(9, 4, 18, 1)),
            ],
            annotated_classes=annotated_tp1,
        )
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            matrix=np.eye(4),
            from_frame="pseudo:f0",
            to_frame="pseudo:f1",
        )
    return path


@pytest.fixture
def series(tmp_path: Path, label_set) -> Path:
    return write_series(tmp_path / "series.medh5", label_set)


class TestTrackingJoin:
    def test_S7_4_the_join_recovers_each_object(self, series):
        with medh5.open(series) as sample:
            tracking = sample.tracks()
            assert sorted(tracking) == [7, 8, 9]
            assert tracking[7].timepoints == ("tp0", "tp1")
            assert tracking[7].class_key == "lesion"
            assert len(tracking[7]) == 2
            assert [o.timepoint for o in tracking[7]] == ["tp0", "tp1"]
            assert "seen at" in repr(tracking[7])
            assert "2 timepoints" in repr(tracking)

    def test_S7_4_volumes_use_the_grid_spacing(self, series):
        with medh5.open(series) as sample:
            track = sample.tracks()[7]
            # 4x4x4 voxels of 2.0 x 1.0 x 1.0 at baseline, 6x6x6 at follow-up.
            assert track.volume("tp0") == pytest.approx(4 * 4 * 4 * 2.0)
            assert track.volume("tp1") == pytest.approx(6 * 6 * 6 * 2.0)
            assert track.observations[0].units == "mm"
            assert track.observations[0].voxel_count == 64

    def test_relative_change_is_the_growth_a_reader_wants(self, series):
        with medh5.open(series) as sample:
            change = sample.tracks()[7].relative_change("tp0", "tp1")
            assert change == pytest.approx((216 - 64) / 64)

    def test_relative_change_is_none_where_a_visit_is_missing(self, series):
        with medh5.open(series) as sample:
            assert sample.tracks()[8].relative_change("tp0", "tp1") is None

    def test_class_filter_selects_one_class(self, series, tmp_path, label_set):
        with medh5.open(series) as sample:
            assert sorted(sample.tracks("lesion")) == [7, 8, 9]
            assert sorted(sample.tracks("liver")) == []

    def test_measurement_can_be_skipped(self, series):
        with medh5.open(series) as sample:
            tracking = sample.tracks(measure=False)
            assert tracking[7].volume("tp0") is None
            assert tracking[7].at("tp0") is not None

    def test_centroid_and_extent_come_from_the_box(self, series):
        with medh5.open(series) as sample:
            obs = sample.tracks()[7].at("tp0")
            assert obs.centroid.shape == (3,)
            assert np.allclose(obs.extent, [4.0, 4.0, 4.0])

    def test_to_json_is_serializable(self, series):
        import json

        with medh5.open(series) as sample:
            json.dumps(sample.tracks().to_json())


class TestThreeStates:
    """§7.4 with §11.3: absence measures something only where someone looked."""

    def test_present_resolved_and_new(self, series):
        with medh5.open(series) as sample:
            tracking = sample.tracks()
            assert tracking.state_at(7, "tp1") == PRESENT
            assert tracking.state_at(8, "tp1") == RESOLVED
            assert tracking.state_at(9, "tp0") == RESOLVED
            assert tracking.is_persistent(7)
            assert tracking.is_resolved(8)
            assert tracking.is_new(9)
            assert tracking.summary()["resolved"] == [8]

    def test_withdrawn_coverage_makes_absence_unexamined(self, tmp_path, label_set):
        path = write_series(tmp_path / "partial.medh5", label_set, annotated_tp1=[1])
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking.state_at(8, "tp1") == UNEXAMINED
            assert not tracking.is_resolved(8)
            assert tracking.unexamined()["tp1"] == (8,)

    def test_a_single_visit_is_never_resolved(self, sample_path):
        with medh5.open(sample_path) as sample:
            tracking = sample.tracks()
            assert len(tracking) == 0
            assert tracking.summary()["tracks"] == 0

    def test_coverage_travels_with_the_join(self, series):
        with medh5.open(series) as sample:
            coverage = sample.tracks().coverage
            assert coverage["tp0"] == frozenset({3})
            assert coverage["tp1"] == frozenset({3})


class TestClassConflicts:
    def test_S7_4_reclassification_across_visits_is_reported(self, tmp_path, label_set):
        path = write_series(
            tmp_path / "conflict.medh5",
            label_set,
            follow_up=[
                InstanceInput(class_id=1, instance_id=7, mask=lesion(8, 8, 8, 3))
            ],
            annotated_tp1=[1, 3],
        )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking[7].has_class_conflict
            assert tracking.class_conflicts() == {7: (1, 3)}
            assert tracking[7].class_id == 1
        assert "W909" in validate_file(path).codes

    def test_W909_names_both_annotations(self, tmp_path, label_set):
        path = write_series(
            tmp_path / "conflict.medh5",
            label_set,
            follow_up=[
                InstanceInput(class_id=1, instance_id=7, mask=lesion(8, 8, 8, 3))
            ],
            annotated_tp1=[1, 3],
        )
        report = validate_file(path)
        message = next(d.message for d in report.warnings if d.code == "W909")
        assert "les_tp0" in message and "les_tp1" in message

    def test_a_clean_series_reports_nothing(self, series):
        assert "W909" not in validate_file(series).codes


class TestInstanceIdentity:
    def test_row_indices_are_not_treated_as_identity(self, tmp_path, label_set):
        """A `boxes` annotation without ids must not fabricate correspondences."""
        path = tmp_path / "boxes.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1")
            w.label_set(label_set)
            for tp in ("tp0", "tp1"):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(SHAPE, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
                w.add_boxes(
                    f"det_{tp}",
                    grid=f"g_{tp}",
                    boxes=[[[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]],
                    class_ids=[3],
                    task="detection",
                )
        with medh5.open(path) as sample:
            assert not carries_instance_ids(sample.annotations["det_tp0"])
            assert len(sample.tracks()) == 0

    def test_declared_ids_on_boxes_do_join(self, tmp_path, label_set):
        path = tmp_path / "boxes_ids.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_timepoint("tp1")
            w.label_set(label_set)
            for tp in ("tp0", "tp1"):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(SHAPE, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
                w.add_boxes(
                    f"det_{tp}",
                    grid=f"g_{tp}",
                    boxes=[[[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]],
                    class_ids=[3],
                    instance_ids=[42],
                    task="detection",
                )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking[42].timepoints == ("tp0", "tp1")
            assert tracking[42].volume("tp0") == pytest.approx(64.0)


class TestAgreement:
    def test_dice_and_iou_are_undefined_on_two_empty_masks(self):
        empty = np.zeros((4, 4), dtype=bool)
        assert dice(empty, empty) is None
        assert iou(empty, empty) is None
        full = np.ones((4, 4), dtype=bool)
        assert dice(full, full) == 1.0
        assert iou(full, empty) == 0.0

    def test_box_iou(self):
        a = np.array([[0.0, 2.0], [0.0, 2.0]])
        assert box_iou(a, a) == pytest.approx(1.0)
        far = np.array([[9.0, 10.0], [9.0, 10.0]])
        assert box_iou(a, far) == 0.0

    def test_S11_2_per_class_dice_between_two_raters(self, tmp_path, label_set):
        path = tmp_path / "raters.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "r1",
                grid="g",
                masks={1: block(SHAPE, (2, 2, 2), 8), 2: block(SHAPE, (8, 8, 8), 4)},
                annotated_classes=[1, 2, 3],
            )
            w.add_segmentation(
                "r2",
                grid="g",
                masks={1: block(SHAPE, (2, 2, 2), 8), 2: block(SHAPE, (8, 8, 8), 2)},
                annotated_classes=[1, 2],
            )
        with medh5.open(path) as sample:
            result = compare_voxel(sample.annotations["r1"], sample.annotations["r2"])
            assert result.per_class["liver"] == pytest.approx(1.0)
            assert 0.0 < result.per_class["spleen"] < 1.0
            assert result.value == pytest.approx(
                np.mean(list(result.per_class.values()))
            )
            record = result.to_record()
            assert record.metric == "dice"
            assert record.against == "annotations/r2"
            # lesion: r1 examined it, r2 did not --- not a disagreement (§11.3).
            assert result.skipped == ("lesion (not examined by both)",)
            assert "lesion" not in result.per_class

            by_iou = compare_voxel(
                sample.annotations["r1"], sample.annotations["r2"], metric="iou"
            )
            assert by_iou.metric == "iou"
            assert by_iou.value <= result.value
            assert compare(sample.annotations["r1"], sample.annotations["r2"]).metric

    def test_an_unknown_metric_is_refused(self, sample_path):
        with (
            medh5.open(sample_path) as sample,
            pytest.raises(MEDH5ValidationError),
        ):
            compare_voxel(
                sample.annotations["organs_tp0"],
                sample.annotations["organs_tp0"],
                metric="vibes",
            )

    def test_annotations_on_different_grids_are_refused(self, series):
        with (
            medh5.open(series) as sample,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            compare_voxel(sample.annotations["les_tp0"], sample.annotations["les_tp1"])
        assert exc.value.code == "E101"

    def test_instances_match_on_declared_ids_first(self, series):
        with medh5.open(series) as sample:
            result = compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
            assert result.matched_by == "instance_id"
            assert [m[0] for m in result.matched] == [0]
            assert result.only_in_a == (1,)
            assert result.only_in_b == (1,)
            assert 0.0 < result.value < 1.0
            assert result.to_record().metric == "object_f1"
            assert result.to_json()["matched_by"] == "instance_id"

    def test_class_mismatches_are_reported_on_matched_objects(
        self, tmp_path, label_set
    ):
        path = write_series(
            tmp_path / "mismatch.medh5",
            label_set,
            follow_up=[
                InstanceInput(class_id=1, instance_id=7, mask=lesion(8, 8, 8, 2))
            ],
            annotated_tp1=[1, 3],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
            assert result.class_mismatches == ((7, 3, 1),)

    def test_iou_matching_is_the_fallback_without_shared_ids(self, tmp_path, label_set):
        path = write_series(
            tmp_path / "noshare.medh5",
            label_set,
            follow_up=[
                InstanceInput(class_id=3, instance_id=99, mask=lesion(8, 8, 8, 2))
            ],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
            assert result.matched_by == "iou"
            assert result.matched and result.matched[0][2] == pytest.approx(1.0)
            assert result.mean_iou == pytest.approx(1.0)

    def test_no_matches_scores_zero(self, tmp_path, label_set):
        path = write_series(
            tmp_path / "nomatch.medh5",
            label_set,
            follow_up=[
                InstanceInput(class_id=3, instance_id=99, mask=lesion(2, 21, 21, 1))
            ],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
            assert result.value == 0.0
            assert result.mean_iou == 0.0


class TestSplitAudit:
    def _write(self, path, label_set, masks, **claim):
        write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
        with medh5.amend(path) as w:
            w.identity(subject_id=claim.pop("subject_id", "subj-A"))
            w.split(**claim)
        return path

    def test_S12_3_a_consistent_cohort_is_clean(self, tmp_path, label_set, masks):
        paths = [
            self._write(
                tmp_path / f"c{i}.medh5",
                label_set,
                masks,
                subject_id=f"subj-{i}",
                set_id="cv5",
                partition="train" if i < 2 else "test",
                manifest_sha256="a" * 64,
            )
            for i in range(3)
        ]
        audit = audit_splits(paths)
        assert audit.ok
        assert audit.set_ids == ("cv5",)
        assert audit.counts() == {"cv5": {"test": 1, "train": 2}}
        assert audit.partitions("cv5")["test"] == ("c2",)
        assert audit.to_json()["ok"] is True

    def test_W906_conflicting_manifests_across_files(self, tmp_path, label_set, masks):
        paths = [
            self._write(
                tmp_path / f"c{i}.medh5",
                label_set,
                masks,
                subject_id=f"subj-{i}",
                set_id="cv5",
                partition="train",
                manifest_sha256=("a" if i == 0 else "b") * 64,
            )
            for i in range(2)
        ]
        audit = audit_splits(paths)
        assert not audit.ok
        assert len(audit.conflicts) == 1
        assert "2 different manifest hashes" in str(audit.conflicts[0])
        assert len(audit.conflicts[0].paths_by_manifest) == 2

    def test_S12_2_subject_leakage_is_its_own_finding(self, tmp_path, label_set, masks):
        """One subject in train and test --- invisible in either file alone."""
        paths = [
            self._write(
                tmp_path / f"visit{i}.medh5",
                label_set,
                masks,
                subject_id="subj-shared",
                set_id="cv5",
                partition="train" if i == 0 else "test",
                manifest_sha256="a" * 64,
            )
            for i in range(2)
        ]
        audit = audit_splits(paths)
        assert not audit.ok
        assert not audit.conflicts
        assert len(audit.leaks) == 1
        leak = audit.leaks[0]
        assert leak.group_id == "subj-shared"
        assert leak.partitions == ("test", "train")
        assert "is in test, train" in str(leak)

    def test_files_without_claims_are_listed_not_failed(
        self, tmp_path, label_set, masks
    ):
        path = write_sample(tmp_path / "bare.medh5", label_set=label_set, masks=masks)
        audit = audit_splits([path])
        assert audit.ok
        assert audit.unclaimed == (str(path),)

    def test_an_unreadable_file_does_not_stop_the_audit(
        self, tmp_path, label_set, masks
    ):
        good = self._write(
            tmp_path / "good.medh5",
            label_set,
            masks,
            set_id="cv5",
            partition="train",
        )
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        audit = audit_splits([good, bad])
        assert not audit.ok
        assert len(audit.unreadable) == 1
        assert audit.set_ids == ("cv5",)

    def test_a_collection_contributes_every_member(self, tmp_path, label_set, masks):
        from medh5.collection import pack

        paths = [
            self._write(
                tmp_path / f"m{i}.medh5",
                label_set,
                masks,
                subject_id="subj-shared",
                set_id="cv5",
                partition="train" if i == 0 else "val",
            )
            for i in range(2)
        ]
        shard = pack(paths, tmp_path / "shard.medh5c")
        audit = audit_splits([shard])
        assert len(audit.memberships) == 2
        assert audit.leaks and audit.leaks[0].group_id == "subj-shared"
        assert all("::" in m.path for m in audit.memberships)

    def test_cohort_group_id_overrides_the_subject(self, tmp_path, label_set, masks):
        paths = []
        for i in range(2):
            path = tmp_path / f"g{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id=f"subj-{i}")
                w.cohort(group_id="family-3")
                w.split(set_id="cv5", partition="train" if i == 0 else "test")
            paths.append(path)
        audit = audit_splits(paths)
        assert audit.leaks[0].group_id == "family-3"
