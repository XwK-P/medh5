"""Longitudinal joins on ``instance_id`` (spec §7.4, §11.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput, instances_from_masks
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

    def test_a_key_no_track_has_is_a_KeyError(self, series):
        """`Tracking` is a mapping, and a key that is no instance id raised
        `OverflowError` or `TypeError` where `get` and `in` answered absent."""
        with medh5.open(series) as sample:
            tracking = sample.tracks()
            for key in (-1, "7", 2**64, 3.5, None, 999):
                with pytest.raises(KeyError):
                    tracking[key]
                assert key not in tracking
                assert tracking.get(key) is None
            assert tracking[np.uint64(7)].timepoints == ("tp0", "tp1")


class TestIgnoredRegions:
    """§7.7, §11.3: an ignore region says "not examined here", so an object
    absent where one is declared was not looked for there.  The follow-up
    ignored as a whole still resolved the baseline's lesion, and a baseline
    that ignored a place made the lesion found there later `new`."""

    BEFORE = lesion(8, 8, 8, 1)
    AFTER = lesion(4, 16, 16, 1)

    @staticmethod
    def _visits(
        path: Path,
        label_set,
        *,
        baseline: list[InstanceInput],
        follow_up: list[InstanceInput],
        ignore: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Two visits registered by the identity, one lesion annotation each."""
        ignore = ignore or {}
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, found in (("tp0", baseline), ("tp1", follow_up)):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}", np.zeros(SHAPE, np.int16), grid=f"g_{tp}", modality="CT"
                )
                w.add_segmentation(
                    f"les_{tp}",
                    grid=f"g_{tp}",
                    instances=found,
                    annotated_classes=[3],
                    encoding="instances",
                    ignore=ignore.get(tp),
                )
            w.add_transform(
                "tp0_to_tp1",
                kind="affine",
                matrix=np.eye(4),
                from_frame="pseudo:tp0",
                to_frame="pseudo:tp1",
            )
        return path

    def _gone(self, path: Path, label_set, ignored: np.ndarray | None) -> Path:
        """Lesion 7 at baseline, nothing at the follow-up."""
        return self._visits(
            path,
            label_set,
            baseline=[InstanceInput(class_id=3, instance_id=7, mask=self.BEFORE)],
            follow_up=[],
            ignore=None if ignored is None else {"tp1": ignored},
        )

    @pytest.mark.parametrize("region", ["the lesion", "the whole volume"])
    def test_S7_7_absence_inside_an_ignore_region_is_unexamined(
        self, tmp_path, label_set, region
    ):
        ignored = self.BEFORE if region == "the lesion" else np.ones(SHAPE, bool)
        path = self._gone(tmp_path / "ignored.medh5", label_set, ignored)
        with medh5.open(path) as sample:
            assert sample.annotations["les_tp1"].has_ignore_region
            tracking = sample.tracks()
            assert tracking.states(7) == {"tp0": PRESENT, "tp1": UNEXAMINED}
            assert not tracking.is_resolved(7)
            assert tracking.unexamined()["tp1"] == (7,)
            assert tracking.coverage["tp1"] == frozenset()
            assert tracking.summary()["resolved"] == []

    def test_S11_3_examined_everywhere_the_absence_is_a_finding(
        self, tmp_path, label_set
    ):
        """The controls: no ignore region, and one that ignores no voxel."""
        for name, ignored in (("none", None), ("empty", np.zeros(SHAPE, bool))):
            path = self._gone(tmp_path / f"{name}.medh5", label_set, ignored)
            with medh5.open(path) as sample:
                tracking = sample.tracks()
                assert tracking.state_at(7, "tp1") == RESOLVED, name
                assert tracking.is_resolved(7), name
                assert tracking.coverage["tp1"] == frozenset({3}), name

    def test_S11_3_another_annotation_that_examined_the_whole_volume_resolves(
        self, tmp_path, label_set
    ):
        path = self._gone(tmp_path / "second.medh5", label_set, np.ones(SHAPE, bool))
        with medh5.amend(path) as w:
            w.add_segmentation(
                "les_tp1_r2",
                grid="g_tp1",
                instances=[],
                annotated_classes=[3],
                encoding="instances",
            )
        with medh5.open(path) as sample:
            assert sample.tracks().state_at(7, "tp1") == RESOLVED

    def test_S7_7_a_baseline_that_ignored_the_place_finds_no_new_lesion(
        self, tmp_path, label_set
    ):
        found = [InstanceInput(class_id=3, instance_id=9, mask=self.AFTER)]
        new = self._visits(
            tmp_path / "new.medh5", label_set, baseline=[], follow_up=found
        )
        with medh5.open(new) as sample:
            assert sample.tracks().is_new(9), "the control: looked for, not found"
        path = self._visits(
            tmp_path / "ignored.medh5",
            label_set,
            baseline=[],
            follow_up=found,
            ignore={"tp0": self.AFTER},
        )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking.states(9) == {"tp0": UNEXAMINED, "tp1": PRESENT}
            assert not tracking.is_new(9)
            assert tracking.summary()["new"] == []

    def test_the_command_line_says_what_resolved_means(
        self, tmp_path, label_set, capsys
    ):
        from medh5.cli import main

        path = self._gone(tmp_path / "ignored.medh5", label_set, self.BEFORE)
        assert main(["track", str(path)]) == 0
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if line.split()[:1] == ["7"])
        assert row.split()[1:4] == ["lesion", "8", "unexamined"]
        legend = " ".join(out.split())
        assert "examined at that timepoint, outside any ignore region, and" in legend


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

    PLACES = {"tp0": lesion(4, 4, 4, 1), "tp1": lesion(12, 18, 18, 1)}
    """One lesion per visit, nowhere near each other."""

    @staticmethod
    def _visits(path: Path, label_set, annotate) -> Path:
        """Two visits in their own frames; ``annotate(w, tp)`` writes each."""
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
                annotate(w, tp)
        return path

    def test_S7_4_objects_minted_from_masks_join_nothing(self, tmp_path, label_set):
        """Masks carry no identity, and stored as `instances` every
        annotation's objects were minted ids from 1: the lesion of one visit
        tracked as the other visit's, centimetres away."""

        def annotate(w, tp):
            w.add_segmentation(
                f"les_{tp}",
                grid=f"g_{tp}",
                masks={3: self.PLACES[tp]},
                encoding="instances",
            )

        path = self._visits(tmp_path / "minted.medh5", label_set, annotate)
        with medh5.open(path) as sample:
            assert list(sample.annotations["les_tp0"].instance_ids) == [1]
            assert list(sample.annotations["les_tp1"].instance_ids) == [2]
            tracking = sample.tracks()
            assert tracking.summary()["persistent"] == []
            assert [t.timepoints for t in tracking.values()] == [("tp0",), ("tp1",)]

    def test_S7_4_ids_the_caller_gives_still_join(self, tmp_path, label_set):
        """An explicit ``start_id``, like an explicit id, asserts the objects
        are one; the writer keeps it."""

        def annotate(w, tp):
            w.add_segmentation(
                f"les_{tp}",
                grid=f"g_{tp}",
                instances=instances_from_masks({3: self.PLACES[tp]}, start_id=1),
            )

        path = self._visits(tmp_path / "asserted.medh5", label_set, annotate)
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking.summary()["persistent"] == [1]
            assert tracking[1].timepoints == ("tp0", "tp1")

    def test_S7_4_an_amendment_mints_above_every_id_the_file_holds(
        self, tmp_path, label_set
    ):
        """Ids are sample-scoped, whatever kind carries them: a box's 42 is
        taken, so the lesion masks added later become object 43."""

        def annotate(w, tp):
            if tp == "tp0":
                w.add_boxes(
                    "det_tp0",
                    grid="g_tp0",
                    boxes=[[[3.0, 5.0], [3.0, 5.0], [3.0, 5.0]]],
                    class_ids=[3],
                    instance_ids=[42],
                    task="detection",
                )

        path = self._visits(tmp_path / "amended.medh5", label_set, annotate)
        with medh5.amend(path) as w:
            w.add_segmentation(
                "les_tp1",
                grid="g_tp1",
                masks={3: self.PLACES["tp1"]},
                encoding="instances",
            )
        with medh5.open(path) as sample:
            assert list(sample.annotations["les_tp1"].instance_ids) == [43]
            assert sample.tracks().summary()["persistent"] == []

    def test_S7_4_no_id_is_minted_past_the_largest_uint64(self, tmp_path, label_set):
        top = 2**64 - 1

        def annotate(w, tp):
            if tp == "tp0":
                w.add_segmentation(
                    "les_tp0",
                    grid="g_tp0",
                    instances=[
                        InstanceInput(class_id=3, instance_id=top, mask=self.PLACES[tp])
                    ],
                )

        path = self._visits(tmp_path / "full.medh5", label_set, annotate)
        with pytest.raises(MEDH5ValidationError) as exc, medh5.amend(path) as w:
            w.add_segmentation(
                "les_tp1",
                grid="g_tp1",
                masks={3: self.PLACES["tp1"]},
                encoding="instances",
            )
        assert exc.value.code == "E404"
        assert str(top) in str(exc.value)


class TestInstanceIdsBeyondUint32:
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


class TestOneAnswerPerVisit:
    """One object seen by several annotations at one visit."""

    def test_S7_4_two_raters_masks_of_a_visit_are_refused_by_every_accessor(
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

    def test_S7_4_a_box_beside_a_mask_defers_to_the_mask(self, series):
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

    def test_the_command_line_names_an_ambiguous_visit(self, series, capsys):
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


class TestVolumeUnits:
    """An observation's volume is in its grid's units cubed, and two visits
    are compared in millimetres: a visit in metres beside one in millimetres
    read as a millionfold change, and an uncalibrated one beside either as a
    change at all."""

    SPACING = {
        "mm": (2.0, 1.0, 1.0),
        "m": (0.002, 0.001, 0.001),
        "um": (2000.0, 1000.0, 1000.0),
        "px": (1.0, 1.0, 1.0),
    }

    def _visits(
        self, path: Path, label_set, units: tuple[str, str], radii: tuple[int, int]
    ) -> Path:
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, unit, radius in zip(("tp0", "tp1"), units, radii, strict=True):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=self.SPACING[unit],
                    units=unit,
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}", np.zeros(SHAPE, np.int16), grid=f"g_{tp}", modality="CT"
                )
                w.add_segmentation(
                    f"les_{tp}",
                    grid=f"g_{tp}",
                    instances=[
                        InstanceInput(
                            class_id=3, instance_id=7, mask=lesion(8, 8, 8, radius)
                        )
                    ],
                    annotated_classes=[3],
                )
        return path

    @pytest.mark.parametrize("unit", ["m", "um"])
    def test_S3_5_the_same_lesion_in_other_units_has_not_changed(
        self, tmp_path, label_set, unit
    ):
        path = self._visits(tmp_path / "s.medh5", label_set, ("mm", unit), (2, 2))
        with medh5.open(path) as sample:
            track = sample.tracks()[7]
            first, second = track.at("tp0"), track.at("tp1")
            assert (first.units, second.units) == ("mm", unit)
            assert first.volume == pytest.approx(64 * 2.0), "its own units"
            assert second.volume_mm == pytest.approx(64 * 2.0)
            assert track.relative_change("tp0", "tp1") == pytest.approx(0.0, abs=1e-9)

    def test_S7_4_an_observation_in_json_says_its_kind_and_its_millimetres(
        self, tmp_path, label_set
    ):
        """`medh5 track --json` gave each observation's volume in its own
        grid's units and neither its kind nor its volume in millimetres, so
        a reader of the JSON could not compare two visits the table did."""
        path = self._visits(tmp_path / "s.medh5", label_set, ("mm", "m"), (2, 2))
        with medh5.open(path) as sample:
            observations = sample.tracks().to_json()["tracks"][0]["observations"]
        assert [o["kind"] for o in observations] == ["instances", "instances"]
        assert [o["units"] for o in observations] == ["mm", "m"]
        assert observations[1]["volume"] == pytest.approx(64 * 2.0e-9)
        assert [o["volume_mm"] for o in observations] == [
            pytest.approx(64 * 2.0),
            pytest.approx(64 * 2.0),
        ]

    def test_S3_5_growth_across_a_change_of_units_is_measured(
        self, tmp_path, label_set
    ):
        path = self._visits(tmp_path / "s.medh5", label_set, ("mm", "m"), (2, 3))
        with medh5.open(path) as sample:
            change = sample.tracks()[7].relative_change("tp0", "tp1")
            assert change == pytest.approx((216 - 64) / 64)

    def test_S3_5_an_uncalibrated_visit_beside_a_calibrated_one_has_no_change(
        self, tmp_path, label_set
    ):
        path = self._visits(tmp_path / "a.medh5", label_set, ("mm", "px"), (2, 3))
        with medh5.open(path) as sample:
            track = sample.tracks()[7]
            assert track.at("tp1").volume_mm is None
            assert track.relative_change("tp0", "tp1") is None
        # Two visits in pixels share a measure, and are compared in it.
        path = self._visits(tmp_path / "b.medh5", label_set, ("px", "px"), (2, 3))
        with medh5.open(path) as sample:
            change = sample.tracks()[7].relative_change("tp0", "tp1")
            assert change == pytest.approx((216 - 64) / 64)

    def test_S3_5_an_area_beside_a_volume_has_no_change(self, tmp_path, label_set):
        path = tmp_path / "dims.medh5"
        plane = SHAPE[1:]
        with medh5.create(path, sample_id="dims", codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            w.add_grid("g_tp0", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_grid("g_tp1", shape=plane, spacing=(1.0, 1.0), timepoint="tp1")
            w.add_image(
                "CT_tp0", np.zeros(SHAPE, np.int16), grid="g_tp0", modality="CT"
            )
            w.add_image(
                "CT_tp1", np.zeros(plane, np.int16), grid="g_tp1", modality="CT"
            )
            w.add_boxes(
                "det_tp0",
                np.array([[[1.5, 5.5], [1.5, 5.5], [1.5, 5.5]]], np.float32),
                ["lesion"],
                grid="g_tp0",
                instance_ids=[7],
            )
            w.add_boxes(
                "det_tp1",
                np.array([[[1.5, 5.5], [1.5, 5.5]]], np.float32),
                ["lesion"],
                grid="g_tp1",
                instance_ids=[7],
            )
        with medh5.open(path) as sample:
            track = sample.tracks()[7]
            assert track.volume("tp0") == pytest.approx(64.0)
            assert track.volume("tp1") == pytest.approx(16.0), "an area, in mm²"
            assert track.relative_change("tp0", "tp1") is None

    def test_the_command_line_shows_millimetres(self, tmp_path, label_set, capsys):
        from medh5.cli import main

        path = self._visits(tmp_path / "s.medh5", label_set, ("mm", "m"), (2, 2))
        assert main(["track", str(path)]) == 0
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if line.split()[:1] == ["7"])
        assert row.split()[2:] == ["128", "128", "+0.0%"]
        assert "mm³" in out


class TestTrackedKinds:
    """Every annotation that carries ``instance_ids`` was joined, and only
    masks and axis-aligned boxes could be read as objects: an oriented box or
    a landmark carrying the ids §10.6 asks for aborted the join, and ``medh5
    track`` with it."""

    @staticmethod
    def _rotation(degrees: float) -> np.ndarray:
        """A proper rotation about the first axis."""
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _visits(self, path: Path, label_set, *, mask_first: bool = False) -> Path:
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp in ("tp0", "tp1"):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(2.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=f"pseudo:{tp}",
                )
                w.add_image(
                    f"CT_{tp}", np.zeros(SHAPE, np.int16), grid=f"g_{tp}", modality="CT"
                )
            if mask_first:
                w.add_segmentation(
                    "les_tp0",
                    grid="g_tp0",
                    instances=[
                        InstanceInput(
                            class_id=3, instance_id=7, mask=lesion(8, 8, 8, 2)
                        )
                    ],
                    annotated_classes=[3],
                )
            else:
                w.add_obb(
                    "obb_tp0",
                    centers=np.array([[8.0, 8.0, 8.0]]),
                    sizes=np.array([[4.0, 4.0, 4.0]]),
                    rotations=self._rotation(30.0)[None],
                    class_ids=["lesion"],
                    grid="g_tp0",
                    instance_ids=[7],
                )
            w.add_obb(
                "obb_tp1",
                centers=np.array([[8.0, 9.0, 9.0], [4.0, 4.0, 4.0]]),
                sizes=np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]),
                rotations=np.stack([self._rotation(45.0), np.eye(3)]),
                class_ids=["lesion", "lesion"],
                grid="g_tp1",
                instance_ids=[7, 9],
            )
        return path

    def test_S7_4_oriented_boxes_join_and_measure(self, tmp_path, label_set):
        path = self._visits(tmp_path / "obb.medh5", label_set)
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert sorted(tracking) == [7, 9]
            track = tracking[7]
            assert track.timepoints == ("tp0", "tp1")
            assert {o.kind for o in track} == {"obb"}
            # Edge lengths times the voxel volume: a rotation moves a box and
            # leaves its volume alone.
            assert track.volume("tp0") == pytest.approx(4 * 4 * 4 * 2.0)
            assert track.volume("tp1") == pytest.approx(6 * 6 * 6 * 2.0)
            assert track.relative_change("tp0", "tp1") == pytest.approx((216 - 64) / 64)
            # The join compares the enclosing axis-aligned box.
            box = track.at("tp1").box
            assert box.shape == (3, 2) and np.all(box[:, 0] < box[:, 1])
            assert tracking.is_new(9) and tracking.skipped == {}

    def test_S7_4_masks_and_oriented_boxes_join_one_track(self, tmp_path, label_set):
        path = self._visits(tmp_path / "mixed.medh5", label_set, mask_first=True)
        with medh5.open(path) as sample:
            track = sample.tracks()[7]
            assert [o.kind for o in track] == ["instances", "obb"]
            assert track.relative_change("tp0", "tp1") == pytest.approx((216 - 64) / 64)

    def test_S10_6_points_mark_presence_and_measure_nothing(self, tmp_path, label_set):
        path = self._visits(tmp_path / "obb.medh5", label_set)
        with medh5.amend(path) as w:
            w.add_points(
                "landmark_tp0",
                np.array([[8.0, 8.0, 8.0], [3.0, 3.0, 3.0]]),
                grid="g_tp0",
                class_ids=["lesion", "lesion"],
                instance_ids=[7, 11],
            )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            seen = [o.kind for o in tracking[7].observations_at("tp0")]
            assert sorted(seen) == ["obb", "points"]
            # The box answers for the visit; the point only says it was there.
            assert tracking[7].at("tp0").kind == "obb"
            point = tracking[11].at("tp0")
            assert point.kind == "points" and point.volume is None
            assert np.array_equal(point.box, [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
            assert tracking.state_at(11, "tp0") == PRESENT

    def test_a_kind_no_track_reads_is_reported_not_fatal(
        self, tmp_path, label_set, capsys
    ):
        from medh5.cli import main

        path = self._visits(tmp_path / "obb.medh5", label_set)
        with medh5.amend(path) as w:
            w.add_keypoints(
                "kp_tp0",
                np.zeros((1, 2, 3)),
                keypoint_classes=["lesion", "lesion"],
                class_ids=["lesion"],
                grid="g_tp0",
                instance_ids=[7],
            )
            w.add_points(
                "unclassed_tp1",
                np.array([[8.0, 9.0, 9.0]]),
                grid="g_tp1",
                instance_ids=[7],
            )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert sorted(tracking) == [7, 9], "the joinable annotations still join"
            assert set(tracking.skipped) == {"kp_tp0", "unclassed_tp1"}
            assert "'keypoints'" in tracking.skipped["kp_tp0"]
            assert "class_ids" in tracking.skipped["unclassed_tp1"]
            assert set(tracking.to_json()["skipped"]) == {"kp_tp0", "unclassed_tp1"}
        assert main(["track", str(path)]) == 0
        out = capsys.readouterr().out
        assert "skipped 'kp_tp0'" in out and "skipped 'unclassed_tp1'" in out
