"""Longitudinal joins on ``instance_id`` (spec §7.4, §11.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.curation import PRESENT, RESOLVED, UNEXAMINED, carries_instance_ids
from medh5.errors import MEDH5ValidationError
from medh5.validate import validate_file
from tests.helpers import SHAPE, lesion, write_series
from tests.kits import Hierarchy


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


class TestF01InstanceIdsBeyondUint32:
    """§7.4, §8.2: `instance_ids` is `uint32` **or** `uint64`, on both sides."""

    def test_S7_4_instances_round_trip_a_64_bit_id(self, tmp_path: Path) -> None:
        path = tmp_path / "wide.medh5"
        with medh5.create(path, sample_id="wide") as w:
            w.label_set(Hierarchy.label_set())
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_segmentation(
                "lesions",
                grid="g",
                instances=[
                    InstanceInput(
                        class_id=3, instance_id=Hierarchy.BIG, mask=Hierarchy.mask()
                    )
                ],
                annotated_classes=[3],
            )
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["lesions"]
            assert ann.group["instance_ids"].dtype == np.uint64
            assert ann.instance_ids.dtype == np.uint64
            assert [int(v) for v in ann.instance_ids] == [Hierarchy.BIG]
            assert [obj.instance_id for obj in ann.instances()] == [Hierarchy.BIG]
            assert ann.tracking() == {Hierarchy.BIG: 3}
            assert ann.instance(Hierarchy.BIG).class_id == 3

    def test_S8_2_geometric_kinds_round_trip_a_64_bit_id(self, tmp_path: Path) -> None:
        path = tmp_path / "geom.medh5"
        rotation = np.eye(3, dtype=np.float32)[None]
        with medh5.create(path, sample_id="geom") as w:
            w.label_set(Hierarchy.label_set())
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_boxes(
                "boxes",
                np.array([[[1.5, 4.5], [1.5, 4.5], [1.5, 4.5]]], dtype=np.float32),
                [3],
                grid="g",
                instance_ids=[Hierarchy.BIG],
            )
            w.add_obb(
                "obb",
                centers=np.array([[3.0, 3.0, 3.0]], dtype=np.float32),
                sizes=np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
                rotations=rotation,
                class_ids=[3],
                grid="g",
                instance_ids=[Hierarchy.BIG],
            )
            w.add_keypoints(
                "kp",
                points=np.array([[[2.0, 2.0, 2.0]]], dtype=np.float32),
                keypoint_classes=[1],
                class_ids=[3],
                grid="g",
                instance_ids=[Hierarchy.BIG],
            )
        with medh5.open(path) as sample:
            for name in ("boxes", "obb", "kp"):
                ann: Any = sample.annotations[name]
                assert ann.group["instance_ids"].dtype == np.uint64, name
                assert [int(v) for v in ann.instance_ids] == [Hierarchy.BIG], name
            boxes: Any = sample.annotations["boxes"]
            assert [obj.instance_id for obj in boxes] == [Hierarchy.BIG]

    def test_S8_2_narrow_ids_keep_the_narrow_storage(self, tmp_path: Path) -> None:
        """The width follows the data: ids that fit stay `uint32` on disk."""
        path = tmp_path / "narrow.medh5"
        with medh5.create(path, sample_id="narrow") as w:
            w.label_set(Hierarchy.label_set())
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_boxes(
                "boxes",
                np.array([[[1.5, 4.5], [1.5, 4.5], [1.5, 4.5]]], dtype=np.float32),
                [3],
                grid="g",
                instance_ids=[7],
            )
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["boxes"]
            assert ann.group["instance_ids"].dtype == np.uint32
            assert [int(v) for v in ann.instance_ids] == [7]

    def test_S7_4_tracking_joins_on_the_wide_id(self, tmp_path: Path) -> None:
        """The join is the whole point of the column, so it is what is tested."""
        path = tmp_path / "long.medh5"
        with medh5.create(path, sample_id="long", subject_id="s") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(Hierarchy.label_set())
            for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=Hierarchy.SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}", Hierarchy.image(), grid=f"g_{tp}", modality="CT"
                )
                w.add_segmentation(
                    f"lesions_{tp}",
                    grid=f"g_{tp}",
                    instances=[
                        InstanceInput(
                            class_id=3, instance_id=Hierarchy.BIG, mask=Hierarchy.mask()
                        )
                    ],
                    annotated_classes=[3],
                )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert list(tracking) == [Hierarchy.BIG]
            assert tracking[Hierarchy.BIG].timepoints == ("tp0", "tp1")
            assert tracking.is_persistent(Hierarchy.BIG)


class TestU04OneAnswerPerVisit:
    """One object seen by several annotations at one visit (U04 of the 2.0 audit)."""

    def test_U04_two_raters_masks_of_a_visit_are_refused_by_every_accessor(
        self, series
    ):
        """`at` and `volume` answered with the first rater, `volumes` with the
        last: one track reported two volumes for one visit."""
        with medh5.amend(series) as w:
            w.add_segmentation(
                "les_tp0_r2",
                grid="g_tp0",
                instances=[
                    InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 3))
                ],
                annotated_classes=[3],
            )
        with medh5.open(series) as sample:
            tracking = sample.tracks()
            track = tracking[7]
            seen = track.observations_at("tp0")
            assert sorted(o.annotation for o in seen) == ["les_tp0", "les_tp0_r2"]
            assert sorted(o.volume for o in seen) == [4 * 4 * 4 * 2.0, 6 * 6 * 6 * 2.0]
            assert len(track.measurements_at("tp0")) == 2
            for ask in (
                lambda: track.at("tp0"),
                lambda: track.volume("tp0"),
                lambda: track.volumes,
                lambda: track.relative_change("tp0", "tp1"),
            ):
                with pytest.raises(MEDH5ValidationError, match="by 2 annotations"):
                    ask()
            # The visit one rater measured still answers, and the object is
            # still present where both saw it.
            assert track.volume("tp1") == pytest.approx(6 * 6 * 6 * 2.0)
            assert tracking.state_at(7, "tp0") == PRESENT

    def test_U04_a_box_beside_a_mask_defers_to_the_mask(self, series):
        """A detection of the lesion beside its segmentation is one answer, the
        mask's: a box's volume is its bounding box's."""
        with medh5.amend(series) as w:
            w.add_boxes(
                "det_tp0",
                np.array([[[3.5, 12.5], [3.5, 12.5], [3.5, 12.5]]], np.float32),
                class_ids=["lesion"],
                grid="g_tp0",
                space="index",
                instance_ids=[7],
            )
        with medh5.open(series) as sample:
            track = sample.tracks()[7]
            kinds = sorted(o.kind for o in track.observations_at("tp0"))
            assert kinds == ["boxes", "instances"]
            assert [o.kind for o in track.measurements_at("tp0")] == ["instances"]
            assert track.at("tp0").annotation == "les_tp0"
            assert track.volumes == pytest.approx(
                {"tp0": 4 * 4 * 4 * 2.0, "tp1": 6 * 6 * 6 * 2.0}
            )
            assert track.relative_change("tp0", "tp1") == pytest.approx((216 - 64) / 64)

    def test_U04_the_command_line_names_an_ambiguous_visit(self, series, capsys):
        from medh5.cli import main

        with medh5.amend(series) as w:
            w.add_segmentation(
                "les_tp0_r2",
                grid="g_tp0",
                instances=[
                    InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 3))
                ],
                annotated_classes=[3],
            )
        assert main(["track", str(series)]) == 0
        row = next(
            line for line in capsys.readouterr().out.splitlines() if "2 raters" in line
        )
        assert row.split()[0] == "7"
