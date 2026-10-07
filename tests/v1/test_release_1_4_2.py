"""What 1.4.2 changed, held to by the reproductions that found it.

The second half of the third audit's plan: the cohort, curation and
registration tools (W20), performance (W21) and hygiene (W22).  Each test is a
finding written as the shortest program that shows it, and the ones about
behaviour fail on 1.4.1.  Test names cite the finding as well as the clause,
because the audit page is where the reasoning lives.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import LabelClass, LabelSet
from medh5.validate import validate_file

SHAPE = (8, 16, 16)
ROOT = Path(__file__).resolve().parents[2]


def _label_set(n: int = 3) -> LabelSet:
    return LabelSet(
        "rel-1.4.2",
        version="1.0.0",
        classes=[LabelClass(i, f"c{i}", f"C{i}") for i in range(1, n + 1)],
    )


def _writer(path: Path, *, shape: tuple[int, ...] = SHAPE, **options: Any) -> Any:
    options.setdefault("codec", "portable")
    w = medh5.create(path, sample_id=path.stem, subject_id="subj-1", **options)
    w.add_grid("g", shape=shape, spacing=(2.0, 1.0, 1.0), frame_uid="1.2.3.4")
    w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
    w.label_set(_label_set())
    return w


def _box(lo: float, hi: float) -> list[list[float]]:
    return [[lo, hi]] * 3


# --------------------------------------------------------------------------
# W20 --- cohort, curation and registration tools
# --------------------------------------------------------------------------


class TestL29Agreement:
    """L-29, L-38: agreement scores what was measured, and its record is storable."""

    def _raters(self, path: Path) -> Path:
        empty = np.empty((0, 3, 2), np.float32)
        with _writer(path) as w:
            w.add_boxes("r1", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes("r2", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes(
                "r3",
                np.array([_box(0, 2), _box(4, 6)], np.float32),
                ["c1", "c2"],
                grid="g",
                annotated_classes=["c1", "c2"],
            )
            w.add_boxes(
                "r4",
                np.array([_box(0, 2)], np.float32),
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
        with _writer(path) as w:
            w.add_grid("fine", shape=(16, 32, 32), spacing=(1.0, 0.5, 0.5))
            w.add_boxes("a", np.array([_box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes("b", np.array([_box(0, 2)], np.float32), ["c1"], grid="fine")
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E101"

    def test_L29_boxes_in_different_spaces_are_refused(self, tmp_path: Path):
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "spaces.medh5"
        with _writer(path) as w:
            w.add_boxes("a", np.array([_box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes(
                "b",
                np.array([_box(0, 4)], np.float32),
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
        box = np.array([_box(0, 2)], np.float32)
        with _writer(path) as w:
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
        mask = np.zeros(SHAPE, bool)
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
        mask = np.zeros(SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with _writer(path) as w:
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
        mask = np.zeros(SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with _writer(path) as w:
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
        mask = np.zeros(SHAPE, bool)
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


class TestL30DuplicateIds:
    """L-30: two nodes with one id are refused, not merged."""

    def _duplicated(self, path: Path, section: str) -> Path:
        with _writer(path) as w:
            alice = w.person("Alice")
            w.activity("annotate", agent=alice)
        with h5py.File(path, "r+") as handle:
            doc = json.loads(handle["meta"][()])
            first = doc["provenance"][section][0]
            doc["provenance"][section].append({**first})
            del handle["meta"]
            handle.create_dataset(
                "meta", data=json.dumps(doc), dtype=h5py.string_dtype()
            )
        return path

    @pytest.mark.parametrize("section", ["agents", "activities"])
    def test_L30_a_duplicated_provenance_id_is_refused_on_read(
        self, tmp_path: Path, section: str
    ):
        """The reader kept the last, and a no-op amend deleted the other."""
        path = self._duplicated(tmp_path / f"{section}.medh5", section)
        before = path.read_bytes()
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError):
            _ = s.document
        with pytest.raises(MEDH5ValidationError, match="more than once"):
            medh5.amend(path).commit()
        assert path.read_bytes() == before
        assert validate_file(path).errors

    def test_L30_S7_4_the_writer_refuses_two_objects_sharing_an_id(
        self, tmp_path: Path
    ):
        first = np.zeros(SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        with (
            _writer(tmp_path / "dup.medh5") as w,
            pytest.raises(MEDH5ValidationError) as caught,
        ):
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 5, mask=first),
                    InstanceInput(1, 5, mask=second),
                ],
            )
        assert caught.value.code == "E404"

    def test_L30_S7_4_the_validator_reports_a_shared_id_as_E404(self, tmp_path: Path):
        first = np.zeros(SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        path = tmp_path / "crafted.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 5, mask=first),
                    InstanceInput(1, 6, mask=second),
                ],
            )
        with h5py.File(path, "r+") as handle:
            ids = handle["annotations/les/instance_ids"]
            ids[...] = np.array([5, 5], dtype=ids.dtype)
        assert "E404" in validate_file(path).codes

    def test_L30_the_W909_corpus_case_is_the_sample_scoped_conflict(
        self, tmp_path: Path
    ):
        """One id classed apart across two annotations: a warning, and no E404."""
        from medh5.conformance import build_corpus, case_by_name

        case = case_by_name("W909-instance-id-two-classes")
        build_corpus(tmp_path, names=[case.name])
        path = tmp_path / f"{case.name}.medh5"
        with h5py.File(path, "r") as handle:
            groups = [g for g in handle["annotations"].values() if "instance_ids" in g]
            assert len(groups) == 2
            assert all(len(g["instance_ids"]) == 1 for g in groups)
        report = validate_file(path, level=case.level)
        assert set(report.codes) == {*case.errors, *case.warnings}


class TestL33SplitLeaks:
    """L-33: a leak is found over subjects and grouping keys together."""

    def _cohort(self, root: Path) -> list[Path]:
        paths = []
        for i, (group, part) in enumerate((("fam-A", "train"), ("fam-B", "test"))):
            path = root / f"visit{i}.medh5"
            with _writer(path) as w:
                w.identity(sample_id=f"s{i}", subject_id="patient-7")
                w.cohort(group_id=group)
                w.split(set_id="v1", partition=part)
            paths.append(path)
        return paths

    def test_L33_S12_3_one_subject_under_two_group_ids_is_a_leak(self, tmp_path: Path):
        """It reported `leaks = 0, ok = True` --- the case the splitter refuses."""
        from medh5.curation.splits import audit_splits

        audit = audit_splits(self._cohort(tmp_path))
        assert not audit.ok
        (leak,) = audit.leaks
        assert leak.groups == ("fam-A", "fam-B")
        assert leak.subjects == ("patient-7",)
        assert leak.partitions == ("test", "train")
        assert "fam-B" in str(leak)

    def test_L33_C202_finds_the_same_leak(self, tmp_path: Path):
        from medh5.dataset.check import check
        from medh5.dataset.manifest import scan

        root = tmp_path / "cohort"
        root.mkdir()
        self._cohort(root)
        manifest, _ = scan(root)
        finding = next(f for f in check(manifest).errors if f.code == "C202")
        assert "fam-A" in finding.where and "fam-B" in finding.where
        assert "patient-7" in finding.message

    def test_L33_units_join_through_subjects_transitively(self):
        from medh5.curation.splits import anatomy_units

        units = anatomy_units(
            [("s1", "A"), ("s1", "B"), ("s2", "B"), ("s2", "C"), ("s3", "D")]
        )
        assert units["A"] == units["C"] == ("A", "B", "C")
        assert units["D"] == ("D",)


class TestL35FrameGraph:
    """L-35, S-12: inverses are one route, composites end, and typos are errors."""

    def _visits(self, path: Path) -> Any:
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_timepoint("tp0", index=0)
        w.add_timepoint("tp1", index=1)
        for tp, frame in (("tp0", "F"), ("tp1", "M")):
            w.add_grid(
                f"g_{tp}",
                shape=(8, 8, 8),
                spacing=(1, 1, 1),
                frame_uid=frame,
                timepoint=tp,
            )
            w.add_image(
                f"CT_{tp}", np.zeros((8, 8, 8), np.int16), grid=f"g_{tp}", modality="CT"
            )
        return w

    def test_L35_S12_mutually_declared_inverses_resolve_both_ways(self, tmp_path: Path):
        """What E505 checks as "mutually consistent" raised E501 in both directions."""
        path = tmp_path / "mutual.medh5"
        field = np.ones((3, 8, 8, 8), np.float32)
        with self._visits(path) as w:
            for name, source, target, grid, sign, inverse in (
                ("fwd", "F", "M", "g_tp0", 1.0, "bwd"),
                ("bwd", "M", "F", "g_tp1", -1.0, "fwd"),
            ):
                w.add_transform(
                    name,
                    kind="displacement",
                    from_frame=source,
                    to_frame=target,
                    field=sign * field,
                    field_grid=grid,
                    vector_space="world",
                    inverse_id=inverse,
                )
        assert not validate_file(path).errors
        with medh5.open(path) as s:
            forward = s.transform_between("tp0", "tp1")
            backward = s.transform_between("tp1", "tp0")
            assert forward.transform_id == "fwd"
            assert backward.transform_id == "bwd"
            point = [[1.0, 1.0, 1.0]]
            assert forward.transform_points(point).tolist() == [[2.0, 2.0, 2.0]]

    def test_L35_S12_a_one_sided_declaration_is_one_route(self, tmp_path: Path):
        """An affine's analytic inverse and the stored transform naming it are one."""
        path = tmp_path / "one-sided.medh5"
        shift = np.eye(4)
        shift[:3, 3] = 2.0
        back = np.eye(4)
        back[:3, 3] = -2.0
        with self._visits(path) as w:
            w.add_transform(
                "fwd", kind="affine", from_frame="F", to_frame="M", matrix=shift,
                inverse_id="bwd",
            )  # fmt: skip
            w.add_transform(
                "bwd", kind="affine", from_frame="M", to_frame="F", matrix=back
            )
        with medh5.open(path) as s:
            assert s.transform_between("tp0", "tp1").transform_id == "fwd"
            assert s.transform_between("tp1", "tp0").transform_id == "bwd"

    def test_L35_a_composite_that_contains_itself_is_E501(self, tmp_path: Path):
        """It validated clean and evaluating it raised RecursionError."""
        path = tmp_path / "cycle.medh5"
        with self._visits(path) as w:
            w.add_transform(
                "a", kind="affine", from_frame="F", to_frame="X", matrix=np.eye(4)
            )
            w.add_transform(
                "b", kind="affine", from_frame="X", to_frame="M", matrix=np.eye(4)
            )
            w.add_transform(
                "b2", kind="affine", from_frame="X", to_frame="F", matrix=np.eye(4)
            )
            w.add_transform(
                "c1",
                kind="composite",
                from_frame="F",
                to_frame="M",
                components=["a", "b"],
            )
            w.add_transform(
                "c2", kind="composite", from_frame="X", to_frame="M",
                components=["b2", "c1"],
            )  # fmt: skip
        with h5py.File(path, "r+") as handle:
            handle["transforms/c1"].attrs["components"] = np.array(
                ["a", "c2"], dtype=h5py.string_dtype()
            )
        assert "E501" in validate_file(path).codes
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            s.transforms["c1"].transform_points([[0.0, 0.0, 0.0]])
        assert caught.value.code == "E501"
        assert "contains itself" in str(caught.value)

    def test_L35_a_mistyped_key_is_a_KeyError_not_no_registration(self, tmp_path: Path):
        path = tmp_path / "typo.medh5"
        with self._visits(path) as w:
            w.add_transform(
                "t", kind="affine", from_frame="F", to_frame="M", matrix=np.eye(4)
            )
        with medh5.open(path) as s:
            assert s.transform_between("tp0", "tp1").transform_id == "t"
            assert s.transform_between("F", "M").transform_id == "t"
            with pytest.raises(KeyError, match="TP1"):
                s.transform_between("tp0", "TP1")

    def test_L35_the_registration_guide_no_longer_warns_against_linking(self):
        guide = (ROOT / "docs/guides/registration.md").read_text(encoding="utf-8")
        assert "Do not link them with `inverse_id`" not in guide


class TestL26Verify:
    """L-26: an undigested dataset inside an attested object fails `verify`."""

    def _boxes(self, path: Path) -> Path:
        with _writer(path) as w:
            w.add_boxes("det", np.array([_box(1, 3)], np.float32), ["c1"], grid="g")
        return path

    def test_L26_S13_2_an_undigested_dataset_in_an_annotation_fails_verify(
        self, tmp_path: Path
    ):
        """`instance_ids` added without a digest verified, and `tracks()` used it."""
        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        with medh5.open(path) as s:
            result = s.verify()
            assert not result.ok
            assert result.unattested == ("annotations/det/instance_ids",)
            assert result.summary()["unattested"] == ["annotations/det/instance_ids"]

    def test_L26_recompress_names_the_unattested_dataset(self, tmp_path: Path, capsys):
        """It failed with `FAILED (0)` and no path, in the table and the JSON."""
        from medh5.cli import main
        from medh5.storage.recompress import recompress

        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        result = recompress(path, "portable")
        assert not result.ok and result.mismatched == []
        assert result.unattested == ["annotations/det/instance_ids"]
        assert result.to_json()["unattested"] == ["annotations/det/instance_ids"]
        assert main(["recompress", str(path), "--profile", "portable"]) == 1
        out = capsys.readouterr().out
        assert "FAILED (1)" in out
        assert "UNSIGNED  annotations/det/instance_ids" in out

    def test_L26_fix_counts_it_as_needing_digests(self, tmp_path: Path):
        from medh5.integrity.repair import diagnose

        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        diagnosis = diagnose(path)
        assert diagnosis.needs_digests
        assert diagnosis.unattested == ("annotations/det/instance_ids",)

    def test_L26_the_writers_files_have_none_and_an_unaddressed_file_is_not_judged(
        self, tmp_path: Path
    ):
        path = self._boxes(tmp_path / "det.medh5")
        with medh5.open(path) as s:
            assert s.verify().ok and s.verify().unattested == ()
        with h5py.File(path, "r+") as handle:
            del handle.attrs["content_id"]
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        with medh5.open(path) as s:
            # No `content_id` makes no claim to check (§13.2 is a SHOULD).
            assert s.verify().unattested == ()


class TestL32GeometricGrids:
    """L-32: index coordinates belong to the annotation's own grid."""

    def _two_grids(self, path: Path, *, frame: str | None = "F") -> Path:
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("lo", shape=(8, 16, 16), spacing=(4.0, 2.0, 2.0), frame_uid="F")
        w.add_grid("hi", shape=(32, 64, 64), spacing=(1.0, 0.5, 0.5), frame_uid=frame)
        w.add_image("CT", np.zeros((32, 64, 64), np.int16), grid="hi", modality="CT")
        w.add_image("CTlo", np.zeros((8, 16, 16), np.int16), grid="lo", modality="CT")
        w.add_boxes(
            "det",
            np.array([[[1.5, 3.5], [3.5, 7.5], [3.5, 7.5]]], np.float32),
            [1],
            grid="lo",
        )
        w.commit()
        return path

    def test_L32_S3_3_another_grid_of_the_frame_converts_through_world(
        self, tmp_path: Path
    ):
        """A box on the 4 mm grid came back unchanged when asked for the 1 mm grid."""
        with medh5.open(self._two_grids(tmp_path / "g.medh5")) as s:
            boxes = s.annotations["det"]
            assert boxes.as_slices()[0] == (slice(2, 4), slice(4, 8), slice(4, 8))
            # World [6, 14] x [7, 15] x [7, 15] mm, on a grid of 1 x 0.5 x 0.5 mm.
            assert boxes.as_slices(grid="hi")[0] == (
                slice(6, 14),
                slice(14, 30),
                slice(14, 30),
            )
            corner = boxes.to_index([[1.5, 3.5, 3.5]], grid="hi")
            assert corner.tolist() == [[6.0, 14.0, 14.0]]
            assert boxes.to_world([[1.5, 3.5, 3.5]], grid="hi").tolist() == [
                [6.0, 7.0, 7.0]
            ]
            assert boxes.to_world([[1.5, 3.5, 3.5]]).tolist() == [[6.0, 7.0, 7.0]]

    @pytest.mark.parametrize("frame", ["OTHER", None])
    def test_L32_a_grid_of_another_or_no_frame_is_refused(
        self, tmp_path: Path, frame: str | None
    ):
        with medh5.open(self._two_grids(tmp_path / "g.medh5", frame=frame)) as s:
            boxes = s.annotations["det"]
            for call in (
                lambda: boxes.as_slices(grid="hi"),
                lambda: boxes.to_index([[0.0, 0.0, 0.0]], grid="hi"),
                lambda: boxes.to_world([[0.0, 0.0, 0.0]], grid="hi"),
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    call()
                assert caught.value.code == "E414"

    def test_L32_boxes_on_a_slice_stay_on_their_own_grid(self, tmp_path: Path):
        path = tmp_path / "slices.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("lo", shape=(8, 16, 16), spacing=(4.0, 2.0, 2.0), frame_uid="F")
        w.add_grid("hi", shape=(32, 64, 64), spacing=(1.0, 0.5, 0.5), frame_uid="F")
        w.add_image("CT", np.zeros((32, 64, 64), np.int16), grid="hi", modality="CT")
        w.add_boxes(
            "key",
            np.array([[[3.0, 3.0], [3.5, 7.5], [3.5, 7.5]]], np.float32),
            [1],
            grid="lo",
            slice_index=[3],
        )
        w.commit()
        with medh5.open(path) as s:
            boxes = s.annotations["key"]
            assert boxes.as_slices()[0][0] == slice(3, 4)
            with pytest.raises(MEDH5ValidationError, match="slice_index"):
                boxes.as_slices(grid="hi")


# --------------------------------------------------------------------------
# W21 --- performance
# --------------------------------------------------------------------------


def _many_classes(path: Path, classes: int = 63, shape=(16, 32, 32)) -> Path:
    rng = np.random.default_rng(0)
    masks = {}
    for c in range(1, classes + 1):
        mask = np.zeros(shape, bool)
        z, y, x = (int(rng.integers(0, n - 2)) for n in shape)
        mask[z : z + 2, y : y + 2, x : x + 2] = True
        masks[c] = mask
    w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
    w.add_grid("g", shape=shape, spacing=(1, 1, 1))
    w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
    w.label_set(_label_set(classes))
    w.add_segmentation("organs", grid="g", masks=masks)
    w.build_index()
    w.commit()
    return path


class TestW21Performance:
    """P-08, P-09, P-11, P-12, L-28, T-11."""

    def test_P08_an_indexed_draw_reads_at_most_two_datasets_at_63_classes(
        self, tmp_path: Path, monkeypatch
    ):
        """190 HDF5 reads and 7.6 ms per draw: the class table, re-read per class."""
        from medh5.sampling import PatchSampler

        path = _many_classes(tmp_path / "many.medh5")
        reads = {"n": 0}
        original = h5py.Dataset.__getitem__

        def counting(self: Any, key: Any) -> Any:
            reads["n"] += 1
            return original(self, key)

        sampler = PatchSampler(8, strategy="foreground")
        with medh5.open(path) as s:
            sampler.draw(s, "organs", np.random.default_rng(0))
            monkeypatch.setattr(h5py.Dataset, "__getitem__", counting)
            draws = 20
            for i in range(draws):
                patch = sampler.draw(s, "organs", np.random.default_rng(i))
            monkeypatch.undo()
        assert patch.used_index is True
        assert reads["n"] <= 2 * draws

    def test_P08_a_draw_picks_what_it_picked_before(self, tmp_path: Path):
        """One row is read instead of the pool, and the same row: runs reproduce."""
        path = _many_classes(tmp_path / "many.medh5", classes=4)
        with medh5.open(path) as s:
            index = s.index["organs"]
            for class_id in index.class_ids:
                pool = index.coords(class_id)
                for seed in range(5):
                    rng = np.random.default_rng(seed)
                    expected = pool[
                        np.random.default_rng(seed).integers(0, len(pool), 1)
                    ]
                    assert index.sample_foreground(class_id, 1, rng).tolist() == (
                        expected.tolist()
                    )
            assert index.has_class(1) and not index.has_class(99)
            assert index.voxel_counts == dict(index.voxel_counts)

    @pytest.mark.parametrize("encoding", ["layers", "bitmask"])
    def test_P09_S14_plane_counts_read_within_the_slab_budget(
        self, tmp_path: Path, encoding: str
    ):
        """`data[layer]` read the whole plane before the slab loop bounded nothing.

        The read budget is the engine's: its Rust tests
        (`annotations::payload::tests::p09_s14_*`) hold every planned read to a
        4 KiB budget, tiling the dataset exactly, and the counts to the
        full-budget ones.  Here: the counts a reader gets are exact."""
        shape = (16, 32, 32)
        first = np.zeros(shape, bool)
        first[2:10, 2:20, 2:20] = True
        second = first.copy()
        second[:, :, 10:30] = True  # overlaps: two layers, or two bits
        path = tmp_path / f"{encoding}.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1))
        w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        w.label_set(_label_set(2))
        w.add_segmentation(
            "seg", grid="g", masks={1: first, 2: second}, encoding=encoding
        )
        w.commit()

        with medh5.open(path) as s:
            annotation = s.annotations["seg"]
            assert annotation.kind == encoding
            counts = annotation.voxel_counts()
        assert counts[1] == int(first.sum()) and counts[2] == int(second.sum())

    def test_P11_one_open_per_file_per_epoch(self, tmp_path: Path):
        """`shuffle=True` re-opened 76 % of items' files at 100 files."""
        pytest.importorskip("torch")
        from medh5.sampling import PatchSampler
        from medh5.torch import FileGroupedSampler, PatchDataset
        from medh5.torch.handles import CACHE

        paths = []
        for i in range(40):
            path = tmp_path / f"c{i:02d}.medh5"
            with _writer(path, shape=(8, 16, 16)):
                pass
            paths.append(path)
        dataset = PatchDataset(
            paths, PatchSampler(4, strategy="uniform"), images=["CT"],
            samples_per_volume=3, seed=0,
        )  # fmt: skip
        sampler = FileGroupedSampler(dataset, seed=0)
        orders = []
        for epoch in range(2):
            dataset.set_epoch(epoch)
            CACHE.close_all()
            CACHE.opens = 0
            order = list(sampler)
            for index in order:
                dataset[index]
            assert CACHE.opens == len(paths)
            assert sorted(order) == list(range(len(dataset)))
            orders.append(order)
        assert orders[0] != orders[1]
        assert len(sampler) == len(dataset)

    def test_P11_every_dataset_says_which_items_read_which_file(self, tmp_path: Path):
        pytest.importorskip("torch")
        from medh5.sampling import PatchSampler
        from medh5.torch import (
            FileGroupedSampler,
            GridPatchDataset,
            PatchDataset,
            VolumeDataset,
        )

        paths = []
        for i in range(3):
            path = tmp_path / f"f{i}.medh5"
            with _writer(path):
                pass
            paths.append(path)
        volume = VolumeDataset(paths, images=["CT"])
        patches = PatchDataset(
            paths, PatchSampler(4), images=["CT"], samples_per_volume=2
        )
        grid = GridPatchDataset(paths, patch_size=(8, 8, 8), images=["CT"])
        assert [list(g) for g in volume.file_groups()] == [[0], [1], [2]]
        assert [list(g) for g in patches.file_groups()] == [[0, 1], [2, 3], [4, 5]]
        per_file = len(grid) // 3
        assert [len(g) for g in grid.file_groups()] == [per_file] * 3
        unshuffled = list(FileGroupedSampler(grid, shuffle=False))
        assert unshuffled == list(range(len(grid)))
        with pytest.raises(MEDH5ValidationError, match="file_groups"):
            FileGroupedSampler(list(range(4)))

    def test_P11_T10_spawned_persistent_workers_follow_the_grouped_order(
        self, tmp_path: Path
    ):
        """The spawn start method, explicitly, on every platform CI runs."""
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from medh5.sampling import PatchSampler
        from medh5.torch import FileGroupedSampler, PatchDataset, collate

        paths = []
        for i in range(2):
            path = tmp_path / f"w{i}.medh5"
            with _writer(path, shape=(16, 32, 32)):
                pass
            paths.append(path)
        dataset = PatchDataset(
            paths, PatchSampler((8, 8, 8), strategy="uniform"), images=["CT"],
            samples_per_volume=4,
        )  # fmt: skip
        loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=FileGroupedSampler(dataset),
            num_workers=2,
            collate_fn=collate,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

        def epoch(n: int) -> list[tuple[str, tuple[int, ...]]]:
            dataset.set_epoch(n)
            seen = []
            for batch in loader:
                assert len(set(batch["meta"]["path"])) == 1  # one file per batch
                seen += [
                    (p, tuple(s))
                    for p, s in zip(
                        batch["meta"]["path"],
                        batch["meta"]["patch"]["start"],
                        strict=True,
                    )
                ]
            return seen

        first, second = epoch(0), epoch(1)
        assert len(first) == len(second) == len(dataset)
        assert sorted(first) != sorted(second)

    def test_P11_set_cache_size_is_public_and_shrinks_the_cache(self, tmp_path: Path):
        from medh5.torch import CACHE, set_cache_size
        from medh5.torch.handles import DEFAULT_MAXSIZE

        paths = []
        for i in range(4):
            path = tmp_path / f"s{i}.medh5"
            with _writer(path):
                pass
            paths.append(path)
        try:
            CACHE.close_all()
            for path in paths:
                CACHE.get(path)
            assert len(CACHE) == 4
            set_cache_size(2)
            assert len(CACHE) == 2
            with pytest.raises(ValueError):
                set_cache_size(0)
        finally:
            set_cache_size(DEFAULT_MAXSIZE)
            CACHE.close_all()

    def test_P12_S14_3_the_occupancy_map_is_compressed_one_class_per_chunk(
        self, tmp_path: Path
    ):
        """52 MB of mostly `False` at 200 classes and 512³, stored contiguous."""
        # 40 classes x (64/8, 128/8, 128/8): 80 KiB, over the compression floor.
        path = _many_classes(tmp_path / "occ.medh5", classes=40, shape=(64, 128, 128))
        with h5py.File(path, "r") as handle:
            occupancy = handle["index/organs/occupancy"]
            assert occupancy.chunks == (1, *occupancy.shape[1:])
            assert occupancy.compression == "gzip"
            assert occupancy.id.get_storage_size() < occupancy.nbytes / 10
        with medh5.open(path) as s:
            assert s.verify().ok and "organs" in s.fresh_indices

    def test_L28_S14_1_rechunk_chunks_the_way_the_writer_does(self, tmp_path: Path):
        """A `layers` dataset went to h5py's (2, 16, 24, 48), and drew W902."""
        from medh5.storage.chunking import fit_chunks, grid_chunks
        from medh5.storage.recompress import recompress

        shape = (32, 48, 48)
        rng = np.random.default_rng(0)
        masks = {}
        for c in range(1, 5):
            mask = np.zeros(shape, bool)
            mask[int(rng.integers(0, 16)) :, int(rng.integers(0, 24)) :, :] = True
            masks[c] = mask
        path = tmp_path / "rc.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(16, 16, 16))
        w.add_image(
            "CT", rng.integers(0, 1000, shape).astype(np.int16), grid="g", modality="CT"
        )
        w.label_set(_label_set(4))
        w.add_segmentation("organs", grid="g", masks=masks, encoding="layers")
        w.commit()
        # Re-chunk it the way some other tool might have, then ask for the rule.
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs/data"][...]
            attrs = dict(handle["annotations/organs/data"].attrs)
            del handle["annotations/organs/data"]
            node = handle["annotations/organs"].create_dataset(
                "data", data=data, chunks=(2, 8, 12, 24), compression="gzip"
            )
            node.attrs.update(attrs)
        result = recompress(path, "portable", rechunk=True)
        assert result.ok
        with medh5.open(path) as s:
            grid = s.grids["g"]
        with h5py.File(path, "r") as handle:
            data = handle["annotations/organs/data"]
            assert data.chunks == fit_chunks(
                grid_chunks(grid, data.dtype.itemsize, leading=1), data.shape
            )
            assert data.chunks[0] == 1
            image = handle["images/CT"]
            assert image.chunks == grid_chunks(grid, image.dtype.itemsize)
        assert "W902" not in validate_file(path, level="strict").codes

    def test_L28_rechunk_finds_each_sample_root_in_a_collection(self, tmp_path: Path):
        from medh5.storage.recompress import recompress

        shape = (32, 64, 64)  # 256 KiB of layers: chunked
        paths = []
        for i in range(2):
            path = tmp_path / f"m{i}.medh5"
            first = np.zeros(shape, bool)
            first[2:20] = True
            second = np.zeros(shape, bool)
            second[10:28] = True
            w = medh5.create(
                path, sample_id=f"m{i}", subject_id=f"p{i}", codec="portable"
            )
            w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(8, 8, 8))
            w.add_image("CT", np.ones(shape, np.int16), grid="g", modality="CT")
            w.label_set(_label_set(2))
            w.add_segmentation(
                "seg", grid="g", masks={1: first, 2: second}, encoding="layers"
            )
            w.commit()
            paths.append(path)
        shard = tmp_path / "shard.medh5c"
        medh5.pack(paths, shard)
        # Verifying the output read the collection root as a sample, and raised
        # a KeyError for its `/meta` after the file had been replaced.
        result = recompress(shard, "balanced", rechunk=True)
        assert result.ok and result.verified and result.content_id_preserved
        with h5py.File(shard, "r") as handle:
            for key in ("m0", "m1"):
                data = handle[f"samples/{key}/annotations/seg/data"]
                assert data.chunks is not None and data.chunks[0] == 1
        with medh5.open_collection(shard) as collection:
            assert all(collection[key].verify().ok for key in collection)

    def test_T11_the_bench_has_a_many_class_row(self, tmp_path: Path):
        from medh5.bench import (
            MANY_CLASSES,
            TARGETS,
            many_class_measurement,
            synthetic_sample,
        )

        path = synthetic_sample(
            tmp_path, shape=(16, 32, 32), classes=MANY_CLASSES, name="many.medh5"
        )
        measured = many_class_measurement(path, patch=8, repeats=3)
        assert measured.name == "foreground_sample_many_ms"
        assert measured.target == TARGETS["foreground_sample_many_ms"][0]
        assert measured.detail == {"classes": MANY_CLASSES, "used_index": True}

    def test_Q19_bench_json_is_only_the_document(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Progress lines on stdout ahead of the JSON made `--json` unparsable."""
        import medh5.bench as bench
        from medh5.cli import main

        small = bench.synthetic_sample
        pair = bench.synthetic_pair
        monkeypatch.setattr(
            bench,
            "synthetic_sample",
            lambda d, **kw: small(d, shape=(16, 32, 32), classes=3, **kw),
        )
        monkeypatch.setattr(
            bench, "synthetic_pair", lambda d, **kw: pair(d, shape=(8, 16, 16))
        )
        monkeypatch.setattr(
            bench,
            "synthetic_many_class_sample",
            lambda d: small(d, shape=(8, 16, 16), classes=5, name="many.medh5"),
        )
        code = main(
            ["bench", "--no-throughput", "--json", "--repeats", "1", "--patch", "8"]
        )
        out, err = capsys.readouterr()
        payload = json.loads(out)
        names = {m["name"] for m in payload["measurements"]}
        assert "foreground_sample_many_ms" in names
        assert "writing a synthetic" in err
        assert code in (0, 1)


# --------------------------------------------------------------------------
# W22 --- hygiene
# --------------------------------------------------------------------------


class TestW22Hygiene:
    """Q-11, Q-12, Q-13, Q-16, Q-17, Q-18, K-03, S-12."""

    def test_Q11_a_deidentification_record_without_a_method_is_refused(
        self, tmp_path: Path
    ):
        """An `assert` stood here, and `python -O` cleared the record instead."""
        with _writer(tmp_path / "d.medh5") as w:
            with pytest.raises(MEDH5ValidationError, match="method"):
                w.deidentification()
            w.deidentification(method="synthetic")

    def test_Q11_the_dead_names_are_gone(self):
        import medh5.geometry.grid as grid
        import medh5.transforms.apply as apply

        assert not hasattr(grid, "iter_spatial_slices")
        assert not hasattr(grid, "KNOWN_COORD_SYSTEMS")
        assert not hasattr(apply, "_refuse_outside")

    def test_Q12_a_missing_name_is_a_handled_cli_error(self, tmp_path: Path, capsys):
        """`convert to-nifti FILE <unknown image>` ended in a KeyError traceback."""
        pytest.importorskip("nibabel")
        from medh5.cli import main

        path = tmp_path / "a.medh5"
        with _writer(path):
            pass
        code = main(
            ["convert", "to-nifti", str(path), "PET", str(tmp_path / "x.nii.gz")]
        )
        assert code == 1
        assert capsys.readouterr().err.startswith("medh5: no image 'PET'")

    def test_Q12_a_bare_key_is_named(self):
        from medh5.cli import _what

        assert _what(KeyError("PET")) == "no such entry: 'PET'"
        assert _what(KeyError("no image 'PET'; available: []")).startswith("no image")

    def test_Q13_S16_a_transcode_carries_attributes_it_does_not_know(
        self, tmp_path: Path
    ):
        """`AnnotationHeader.read` dropped them; `amend` carries them elsewhere."""
        path = tmp_path / "t.medh5"
        mask = np.zeros(SHAPE, bool)
        mask[2:5, 2:6, 2:6] = True
        with _writer(path) as w:
            w.add_segmentation("seg", grid="g", masks={1: mask}, encoding="labelmap")
        with h5py.File(path, "r+") as handle:
            group = handle["annotations/seg"]
            group.attrs["x_site_review"] = "second read pending"
            group.attrs["x_site_level"] = np.int32(3)
            group.attrs["x_site_tag"] = np.bytes_(b"ABC")
            group.attrs["x_site_weights"] = np.array([0.5, 0.25], np.float32)
        with medh5.amend(path) as w:
            w.transcode_annotation("seg", "bitmask")
        with h5py.File(path, "r") as handle:
            group = handle["annotations/seg"]
            assert group.attrs["kind"] == "bitmask"
            assert group.attrs["x_site_review"] == "second read pending"
            assert int(group.attrs["x_site_level"]) == 3
            assert bytes(group.attrs["x_site_tag"]) == b"ABC"
            assert group.attrs["x_site_weights"].tolist() == [0.5, 0.25]
        assert not validate_file(path).errors

    def test_Q16_a_leased_handle_outlives_eviction(self, tmp_path: Path):
        from medh5.torch.handles import HandleCache

        paths = []
        for i in range(3):
            path = tmp_path / f"h{i}.medh5"
            with _writer(path):
                pass
            paths.append(path)
        cache = HandleCache(maxsize=1)
        with cache.lease(paths[0]) as held:
            cache.get(paths[1])
            cache.get(paths[2])
            assert held.is_open  # still open: it is in use
            assert held.images["CT"].read().shape == SHAPE
        assert not held.is_open  # released, then evicted and closed
        assert len(cache) == 1
        cache.close_all()

    def test_Q16_threads_share_the_cache_without_closing_each_others_files(
        self, tmp_path: Path
    ):
        from medh5.torch.handles import HandleCache

        paths = []
        for i in range(12):
            path = tmp_path / f"t{i}.medh5"
            with _writer(path) as w:
                w.add_segmentation(
                    "seg",
                    grid="g",
                    masks={1: np.full(SHAPE, True)},
                    encoding="labelmap",
                )
            paths.append(path)
        cache = HandleCache(maxsize=2)
        failures: list[BaseException] = []

        def work(seed: int) -> None:
            rng = np.random.default_rng(seed)
            try:
                for _ in range(40):
                    path = paths[int(rng.integers(0, len(paths)))]
                    with cache.lease(path) as sample:
                        assert sample.annotations["seg"].dense([1])[0].all()
            except BaseException as exc:  # pragma: no cover - the failure itself
                failures.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        cache.close_all()
        assert not failures, failures[:1]

    def test_Q17_original_channel_dim_follows_the_grid(self, tmp_path: Path):
        """`no_channel` on an RGB grid made `EnsureChannelFirst` add a second axis."""
        from medh5.monai import meta_dict

        path = tmp_path / "rgb.medh5"
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.add_grid(
                "rgb",
                shape=(3, 4, 5, 6),
                spacing=(1, 1, 1),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
            )
            w.add_image(
                "RGB",
                np.zeros((3, 4, 5, 6), np.uint8),
                grid="rgb",
                modality="OT",
                channel_names=["r", "g", "b"],
            )
            w.add_grid("g", shape=(4, 5, 6), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 5, 6), np.int16), grid="g", modality="CT")
        with medh5.open(path) as s:
            assert meta_dict(s, "RGB")["original_channel_dim"] == 0
            assert meta_dict(s, "CT")["original_channel_dim"] == "no_channel"

    def test_Q18_S8_6_an_island_inside_a_hole_survives_rasterisation(
        self, tmp_path: Path
    ):
        """Unioning outers and subtracting every hole erased the island."""
        from medh5.io.report import ConversionReport
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

    def test_Q18_two_overlapping_outlines_do_not_cancel(self, tmp_path: Path):
        from medh5.io.report import ConversionReport
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

    def test_K03_actions_are_pinned_by_commit(self):
        uses = []
        for name in ("ci.yml", "release.yml"):
            text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            uses += re.findall(r"uses:\s*(\S+)(.*)", text)
        external = [(u, rest) for u, rest in uses if not u.startswith("./")]
        assert external
        for action, rest in external:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
            assert re.search(r"#\s*v\d", rest), f"{action} carries no version comment"
        dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        assert "package-ecosystem: github-actions" in dependabot

    def test_K03_the_release_runs_ci_on_the_tagged_commit(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "workflow_call:" in ci and "concurrency:" in ci
        assert "uses: ./.github/workflows/ci.yml" in release
        build = release[release.index("  build:") :]
        assert "needs: ci" in build.split("\n  publish:")[0]
        # The two builds upload in one run when CI is called; names must differ.
        assert "name: ci-dist" in ci and "name: dist" in release

    def test_S12_appendix_C_states_how_many_clauses_it_lists(self):
        spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
        section = spec[spec.index("### C.1") : spec.index("### C.2")]
        table = section[section.index("| Clause | Correction |") :]
        rows = [
            line
            for line in table.splitlines()[2:]
            if line.startswith("| §") or line.startswith("| ")
        ]
        rows = rows[: next((i for i, r in enumerate(rows) if not r.strip()), len(rows))]
        words = {
            20: "Twenty",
            21: "Twenty-one",
            22: "Twenty-two",
            23: "Twenty-three",
            24: "Twenty-four",
            25: "Twenty-five",
        }
        assert f"{words[len(rows)]} clauses have been corrected" in section
        # The 1.4.2 correction is recorded; 2.0's three follow it.
        assert any(r.startswith("| §10.1 |") for r in rows)
