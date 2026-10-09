"""Agreement between two raters' annotations (spec §11.2, §11.3)."""

from __future__ import annotations

import json
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
from medh5.errors import MEDH5ValidationError
from medh5.validate import validate_file
from tests.helpers import SHAPE, block, lesion, write_raters
from tests.kits import Framed


class TestL29Agreement:
    """L-29, L-38: agreement scores what was measured, and its record is storable."""

    def _raters(self, path: Path) -> Path:
        empty = np.empty((0, 3, 2), np.float32)
        with Framed.writer(path) as w:
            w.add_boxes("r1", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes("r2", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes(
                "r3",
                np.array([Framed.box(0, 2), Framed.box(4, 6)], np.float32),
                ["c1", "c2"],
                grid="g",
                annotated_classes=["c1", "c2"],
            )
            w.add_boxes(
                "r4",
                np.array([Framed.box(0, 2)], np.float32),
                ["c1"],
                grid="g",
                annotated_classes=["c1"],
            )
        return path

    def test_L29_two_raters_who_found_nothing_are_undefined_not_zero(
        self, tmp_path: Path
    ):
        """Object F1 was 0.0 --- and so was the record --- when both found nothing."""
        from medh5.curation.agreement import compare_instances

        with medh5.open(self._raters(tmp_path / "a.medh5")) as s:
            result = compare_instances(s.annotations["r1"], s.annotations["r2"])
            assert result.value is None
            assert result.mean_iou is None
            with pytest.raises(MEDH5ValidationError, match="nothing was comparable"):
                result.to_record()

    def test_L29_S11_3_a_class_one_rater_never_examined_is_skipped(
        self, tmp_path: Path
    ):
        """Identical work scored 0.667 when the other rater never looked for c2."""
        from medh5.curation.agreement import compare_instances

        with medh5.open(self._raters(tmp_path / "a.medh5")) as s:
            result = compare_instances(s.annotations["r3"], s.annotations["r4"])
            assert result.value == pytest.approx(1.0)
            assert result.only_in_a == ()
            assert result.skipped == ("c2 (not examined by both)",)

    def test_L29_index_boxes_on_different_grids_are_refused(self, tmp_path: Path):
        """Boxes on a 4 mm and a 1 mm grid were compared in raw index units."""
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "grids.medh5"
        with Framed.writer(path) as w:
            w.add_grid("fine", shape=(16, 32, 32), spacing=(1.0, 0.5, 0.5))
            w.add_boxes("a", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes(
                "b", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="fine"
            )
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E101"

    def test_L29_boxes_in_different_spaces_are_refused(self, tmp_path: Path):
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "spaces.medh5"
        with Framed.writer(path) as w:
            w.add_boxes("a", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes(
                "b",
                np.array([Framed.box(0, 4)], np.float32),
                ["c1"],
                grid="g",
                space="world",
                frame_uid="1.2.3.4",
            )
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E414"

    def test_L29_world_boxes_compare_within_one_frame(self, tmp_path: Path):
        """Two gridless world annotations passed a grid test as `None == None`."""
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "frames.medh5"
        box = np.array([Framed.box(0, 2)], np.float32)
        with Framed.writer(path) as w:
            w.add_boxes("a", box, ["c1"], space="world", frame_uid="F")
            w.add_boxes("b", box, ["c1"], space="world", frame_uid="M")
            w.add_boxes("c", box, ["c1"], space="world", frame_uid="F")
        with medh5.open(path) as s:
            same = compare_instances(s.annotations["a"], s.annotations["c"])
            assert same.value == pytest.approx(1.0)
            with pytest.raises(MEDH5ValidationError) as caught:
                compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E414"

    def test_L29_compare_refuses_an_argument_it_cannot_use(self, tmp_path: Path):
        from medh5.curation.agreement import compare

        path = self._raters(tmp_path / "a.medh5")
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with medh5.amend(path) as w:
            w.add_segmentation("s1", grid="g", masks={1: mask})
            w.add_segmentation("s2", grid="g", masks={1: mask})
        with medh5.open(path) as s:
            assert compare(s.annotations["r3"], s.annotations["r4"]).value == 1.0
            assert compare(s.annotations["s1"], s.annotations["s2"]).metric == "dice"
            with pytest.raises(MEDH5ValidationError, match="scores voxels"):
                compare(s.annotations["r3"], s.annotations["r4"], metric="iou")
            with pytest.raises(MEDH5ValidationError, match="matches objects"):
                compare(s.annotations["s1"], s.annotations["s2"], threshold=0.5)
            with pytest.raises(MEDH5ValidationError, match="common kind"):
                compare(s.annotations["r3"], s.annotations["s1"])

    def test_L29_medh5_agree_on_two_boxes_annotations(self, tmp_path: Path, capsys):
        """It exited with `'BoxesAnnotation' object has no attribute 'dense'`."""
        from medh5.cli import main

        path = self._raters(tmp_path / "a.medh5")
        assert main(["agree", str(path), "r3", "r4"]) == 0
        out = capsys.readouterr().out
        assert "object_f1 = 1.0000" in out and "not scored: c2" in out
        assert main(["agree", str(path), "r1", "r2", "--record"]) == 1
        assert "nothing was comparable" in capsys.readouterr().err

    def test_L29_voxel_agreement_with_nothing_comparable_is_undefined(
        self, tmp_path: Path
    ):
        from medh5.curation.agreement import compare_voxel

        path = tmp_path / "vox.medh5"
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with Framed.writer(path) as w:
            w.add_segmentation("a", grid="g", masks={1: mask}, annotated_classes=[1])
            w.add_segmentation("b", grid="g", masks={2: mask}, annotated_classes=[2])
        with medh5.open(path) as s:
            result = compare_voxel(s.annotations["a"], s.annotations["b"])
            assert result.value is None
            with pytest.raises(MEDH5ValidationError):
                result.to_record()

    def test_L29_an_undefined_voxel_comparison_is_reported_not_refused(
        self, tmp_path: Path, capsys
    ):
        """The report went through `to_record()`, so it refused as the record does."""
        from medh5.cli import main
        from medh5.curation.agreement import compare_voxel

        path = tmp_path / "undefined.medh5"
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with Framed.writer(path) as w:
            w.add_segmentation("a", grid="g", masks={1: mask}, annotated_classes=[1])
            w.add_segmentation("b", grid="g", masks={2: mask}, annotated_classes=[2])
        with medh5.open(path) as s:
            report = compare_voxel(s.annotations["a"], s.annotations["b"]).to_json()
        assert report["value"] is None and report["compared"] == 0
        assert main(["agree", str(path), "a", "b"]) == 0
        assert "undefined (nothing comparable)" in capsys.readouterr().out
        assert main(["agree", str(path), "a", "b", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["value"] is None
        assert main(["agree", str(path), "a", "b", "--record"]) == 1
        assert "nothing was comparable" in capsys.readouterr().err

    def test_L38_S11_2_an_agreement_record_can_be_stored_in_its_file(
        self, tmp_path: Path
    ):
        """Keyed by class name, and by `mean_iou`, no record passed the schema."""
        from medh5.curation.agreement import compare_instances, compare_voxel

        path = self._raters(tmp_path / "a.medh5")
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with medh5.amend(path) as w:
            w.add_segmentation("s1", grid="g", masks={1: mask})
            w.add_segmentation("s2", grid="g", masks={1: mask})
        with medh5.open(path) as s:
            voxel = compare_voxel(s.annotations["s1"], s.annotations["s2"]).to_record()
            boxes = compare_instances(
                s.annotations["r3"], s.annotations["r4"]
            ).to_record()
        assert voxel.per_class == {"1": 1.0}
        assert boxes.per_class == {}
        with medh5.amend(path) as w:
            w.set_quality(
                "q", status="reviewed", agreement=[voxel.to_json(), boxes.to_json()]
            )
        assert "E005" not in validate_file(path).codes


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

    def test_instances_on_two_visits_grids_are_refused(self, series):
        # Index boxes on two grids count different voxels; comparing them
        # number for number measured nothing (L-29).
        with (
            medh5.open(series) as sample,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
        assert exc.value.code == "E101"

    def test_instances_match_on_declared_ids_first(self, tmp_path, label_set):
        path = write_raters(tmp_path / "raters.medh5", label_set)
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
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
        path = write_raters(
            tmp_path / "mismatch.medh5",
            label_set,
            second=[InstanceInput(class_id=1, instance_id=7, mask=lesion(8, 8, 8, 2))],
            annotated_second=[1, 3],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.class_mismatches == ((7, 3, 1),)
            # r1 never looked for class 1, so it is skipped, not scored --- but
            # the matched pair counts, because both raters found the object.
            assert result.matched and result.skipped

    def test_iou_matching_is_the_fallback_without_shared_ids(self, tmp_path, label_set):
        path = write_raters(
            tmp_path / "noshare.medh5",
            label_set,
            second=[InstanceInput(class_id=3, instance_id=99, mask=lesion(8, 8, 8, 2))],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.matched_by == "iou"
            assert result.matched and result.matched[0][2] == pytest.approx(1.0)
            assert result.mean_iou == pytest.approx(1.0)

    def test_no_matches_scores_zero(self, tmp_path, label_set):
        path = write_raters(
            tmp_path / "nomatch.medh5",
            label_set,
            second=[
                InstanceInput(class_id=3, instance_id=99, mask=lesion(2, 21, 21, 1))
            ],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.value == 0.0
            # A mean over no matched pairs is undefined, not zero overlap.
            assert result.mean_iou is None


class TestU03IgnoredVoxels:
    """§7.7: a voxel either rater declared unexamined is evidence neither way
    (U03 of the 2.0 audit)."""

    @staticmethod
    def _slab(z0: int, z1: int) -> np.ndarray:
        mask = np.zeros(Framed.SHAPE, bool)
        mask[z0:z1, 2:5, 2:5] = True
        return mask

    @staticmethod
    def _ignore(z0: int, z1: int) -> np.ndarray:
        region = np.zeros(Framed.SHAPE, bool)
        region[z0:z1] = True
        return region

    @pytest.mark.parametrize("encoding", ["labelmap", "bitmask"])
    def test_U03_what_one_rater_never_examined_is_not_scored(
        self, tmp_path: Path, encoding: str
    ):
        """Identical work scored 0.75: rater b's finding where rater a declared
        the slab unexamined counted against a.  In band (`labelmap`) and as the
        sibling `mask` its `ignore_mask` names (`bitmask`) alike."""
        path = tmp_path / "ignore.medh5"
        with Framed.writer(path) as w:
            w.add_segmentation(
                "a",
                grid="g",
                masks={1: self._slab(2, 5)},
                annotated_classes=[1],
                ignore=self._ignore(6, 8),
                encoding=encoding,
            )
            w.add_segmentation(
                "b",
                grid="g",
                masks={1: self._slab(2, 5) | self._slab(6, 8)},
                annotated_classes=[1],
            )
        with medh5.open(path) as s:
            a, b = s.annotations["a"], s.annotations["b"]
            assert compare_voxel(a, b).value == pytest.approx(1.0)
            assert compare_voxel(b, a).value == pytest.approx(1.0)
            assert compare_voxel(a, b, metric="iou").value == pytest.approx(1.0)

    def test_U03_nothing_outside_the_ignore_regions_is_nothing_comparable(
        self, tmp_path: Path
    ):
        """Each rater found the class only where the other never looked."""
        path = tmp_path / "disjoint.medh5"
        with Framed.writer(path) as w:
            w.add_segmentation(
                "a",
                grid="g",
                masks={1: self._slab(2, 5)},
                annotated_classes=[1],
                ignore=self._ignore(6, 8),
            )
            w.add_segmentation(
                "b",
                grid="g",
                masks={1: self._slab(6, 8)},
                annotated_classes=[1],
                ignore=self._ignore(2, 5),
            )
        with medh5.open(path) as s:
            result = compare_voxel(s.annotations["a"], s.annotations["b"])
            assert result.value is None
            assert result.skipped == ("c1 (empty in both outside the ignore region)",)
