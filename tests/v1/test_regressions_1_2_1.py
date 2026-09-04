"""Regressions closed in 1.2.1, one class per finding of the 1.2.0 audit.

Every test here fails against 1.2.0.  They live together rather than beside the
module each one exercises because the audit found them as a set --- silent
wrong answers on the read side of contracts the write side already honoured ---
and a release that claims to have closed a set should be able to prove it in
one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5._hdf5 import open_h5
from medh5.annotations.voxel import InstanceInput
from medh5.cli import main
from medh5.errors import MEDH5Error
from medh5.labels.labelset import LabelClass, LabelSet

SHAPE = (8, 10, 10)
BIG = 2**32 + 7
"""An id §7.4 and §8.2 permit and a `uint32` cannot hold."""


def _label_set(*extra: LabelClass) -> LabelSet:
    return LabelSet(
        "reg-1.2.1",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver"),
            LabelClass(3, "lesion", "Lesion", parents=[1]),
            *extra,
        ],
    )


def _mask(origin: tuple[int, int, int] = (2, 2, 2), size: int = 3) -> Any:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[tuple(slice(o, o + size) for o in origin)] = True
    return mask


def _image() -> Any:
    return np.random.default_rng(1).integers(-1000, 1500, SHAPE).astype(np.int16)


class TestF01InstanceIdsBeyondUint32:
    """§7.4, §8.2: `instance_ids` is `uint32` **or** `uint64`, on both sides."""

    def test_S7_4_instances_round_trip_a_64_bit_id(self, tmp_path: Path) -> None:
        path = tmp_path / "wide.medh5"
        with medh5.create(path, sample_id="wide") as w:
            w.label_set(_label_set())
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT")
            w.add_segmentation(
                "lesions",
                grid="g",
                instances=[InstanceInput(class_id=3, instance_id=BIG, mask=_mask())],
                annotated_classes=[3],
            )
        with medh5.open(path) as sample:
            ann: Any = sample.annotations["lesions"]
            assert ann.group["instance_ids"].dtype == np.uint64
            assert ann.instance_ids.dtype == np.uint64
            assert [int(v) for v in ann.instance_ids] == [BIG]
            assert [obj.instance_id for obj in ann.instances()] == [BIG]
            assert ann.tracking() == {BIG: 3}
            assert ann.instance(BIG).class_id == 3

    def test_S8_2_geometric_kinds_round_trip_a_64_bit_id(self, tmp_path: Path) -> None:
        path = tmp_path / "geom.medh5"
        rotation = np.eye(3, dtype=np.float32)[None]
        with medh5.create(path, sample_id="geom") as w:
            w.label_set(_label_set())
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT")
            w.add_boxes(
                "boxes",
                np.array([[[1.5, 4.5], [1.5, 4.5], [1.5, 4.5]]], dtype=np.float32),
                [3],
                grid="g",
                instance_ids=[BIG],
            )
            w.add_obb(
                "obb",
                centers=np.array([[3.0, 3.0, 3.0]], dtype=np.float32),
                sizes=np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
                rotations=rotation,
                class_ids=[3],
                grid="g",
                instance_ids=[BIG],
            )
            w.add_keypoints(
                "kp",
                points=np.array([[[2.0, 2.0, 2.0]]], dtype=np.float32),
                keypoint_classes=[1],
                class_ids=[3],
                grid="g",
                instance_ids=[BIG],
            )
        with medh5.open(path) as sample:
            for name in ("boxes", "obb", "kp"):
                ann: Any = sample.annotations[name]
                assert ann.group["instance_ids"].dtype == np.uint64, name
                assert [int(v) for v in ann.instance_ids] == [BIG], name
            boxes: Any = sample.annotations["boxes"]
            assert [obj.instance_id for obj in boxes] == [BIG]

    def test_S8_2_narrow_ids_keep_the_narrow_storage(self, tmp_path: Path) -> None:
        """The width follows the data: ids that fit stay `uint32` on disk."""
        path = tmp_path / "narrow.medh5"
        with medh5.create(path, sample_id="narrow") as w:
            w.label_set(_label_set())
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT")
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
            w.label_set(_label_set())
            for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(f"CT_{tp}", _image(), grid=f"g_{tp}", modality="CT")
                w.add_segmentation(
                    f"lesions_{tp}",
                    grid=f"g_{tp}",
                    instances=[
                        InstanceInput(class_id=3, instance_id=BIG, mask=_mask())
                    ],
                    annotated_classes=[3],
                )
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert list(tracking) == [BIG]
            assert tracking[BIG].timepoints == ("tp0", "tp1")
            assert tracking.is_persistent(BIG)


class TestF02StatisticsArePhysical:
    """§4.2: the loaders read `stored × slope + intercept`; so do the statistics."""

    SLOPE, INTERCEPT = 2.0, -1024.0

    def _write(self, path: Path) -> Any:
        stored = np.full(SHAPE, 100, dtype=np.int16)
        stored[0] = 90  # a spread, so std is not trivially zero
        with medh5.create(path, sample_id=path.stem) as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image(
                "CT",
                stored,
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=self.SLOPE,
                rescale_intercept=self.INTERCEPT,
            )
        return stored

    def test_S4_2_normalization_matches_what_the_loader_reads(
        self, tmp_path: Path
    ) -> None:
        from medh5.dataset.stats import DatasetStats, stats_for

        path = tmp_path / "ct.medh5"
        stored = self._write(path)
        physical = stored.astype(np.float64) * self.SLOPE + self.INTERCEPT
        stats = stats_for(path)
        assert stats.physical is True
        mean, std = stats.normalization("CT")
        assert mean == pytest.approx(float(physical.mean()))
        assert std == pytest.approx(float(physical.std()))
        moments = stats.images["CT"]
        assert moments.minimum == pytest.approx(float(physical.min()))
        assert moments.maximum == pytest.approx(float(physical.max()))
        with medh5.open(path) as sample:
            loaded = sample.images["CT"].read(physical=True)
        assert mean == pytest.approx(float(loaded.mean()))
        # The convention travels with the numbers.
        back = DatasetStats.from_json(stats.to_json())
        assert back.physical is True
        assert back.normalization("CT")[0] == pytest.approx(mean)

    def test_S4_2_stored_values_stay_available(self, tmp_path: Path) -> None:
        from medh5.dataset.stats import compute_stats, stats_for

        path = tmp_path / "ct.medh5"
        stored = self._write(path)
        one = stats_for(path, physical=False)
        assert one.physical is False
        assert one.normalization("CT")[0] == pytest.approx(float(stored.mean()))
        many = compute_stats([path], physical=False)
        assert many.physical is False
        assert many.normalization("CT")[0] == pytest.approx(float(stored.mean()))

    def test_merging_the_two_conventions_is_refused(self) -> None:
        from medh5.dataset.stats import DatasetStats

        left = DatasetStats(samples=1, physical=True)
        with pytest.raises(MEDH5Error, match="physical"):
            left.merge(DatasetStats(samples=1, physical=False))
        # A failure record carries no samples and merges into either.
        left.merge(DatasetStats(failures=("x: broken",), physical=False))
        assert left.physical is True

    def test_cli_stored_flag(self, tmp_path: Path, capsys: Any) -> None:
        from medh5.dataset.manifest import scan

        self._write(tmp_path / "ct.medh5")
        manifest, _ = scan(tmp_path)
        manifest.save(tmp_path / "cohort.json")
        assert main(["dataset", "stats", str(tmp_path / "cohort.json")]) == 0
        assert "physical" in capsys.readouterr().out
        assert (
            main(["dataset", "stats", str(tmp_path / "cohort.json"), "--stored"]) == 0
        )
        assert "stored" in capsys.readouterr().out


class TestF03ImplicitTimepoint:
    """§3.7 rule 2: with one declared timepoint a grid may omit the attribute."""

    def _write(self, path: Path) -> None:
        with medh5.create(path, sample_id=path.stem) as w:
            w.label_set(_label_set())
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))  # no timepoint
            w.add_image("CT", _image(), grid="g", modality="CT")
            w.add_segmentation(
                "lesions",
                grid="g",
                instances=[InstanceInput(class_id=3, instance_id=7, mask=_mask())],
                annotated_classes=[3],
            )
            w.add_classification("finding", {3: 1.0}, scope="sample")

    def test_S3_7_a_grid_without_timepoint_belongs_to_the_only_one(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "single.medh5"
        self._write(path)
        with medh5.open(path) as sample:
            assert sample.grids["g"].timepoint == "tp0"
            assert sample.images["CT"].timepoint == "tp0"
            assert sample.annotations["lesions"].timepoints == ("tp0",)
            view = sample.at("tp0")
            assert set(view.grids) == {"g"}
            assert set(view.images) == {"CT"}
            assert "lesions" in view.annotations
            assert sample.grids["g"].summary()["timepoint"] == "tp0"

    def test_S7_4_tracking_sees_the_only_visit(self, tmp_path: Path) -> None:
        path = tmp_path / "single.medh5"
        self._write(path)
        with medh5.open(path) as sample:
            tracking = sample.tracks()
            assert tracking.states(7) == {"tp0": "present"}
            assert tracking.coverage == {"tp0": frozenset({3})}
            assert tracking.is_persistent(7)
            assert tracking.unexamined() == {}

    def test_the_stored_attribute_is_untouched(self, tmp_path: Path) -> None:
        """Resolution is a reader interpretation, not an edit."""
        path = tmp_path / "single.medh5"
        self._write(path)
        with medh5.open(path) as sample:
            before = sample.content_id
        with medh5.amend(path) as w:
            assert w.grids["g"].timepoint is None
            w.extra("note", {"amended": True})
        with open_h5(path, "r") as handle:
            assert "timepoint" not in handle["grids/g"].attrs
        with medh5.open(path) as sample:
            assert sample.grids["g"].timepoint == "tp0"
            assert sample.verify().ok
            assert sample.content_id != before  # the extra changed it, not this

    def test_S3_7_declared_timepoints_are_left_as_stored(self, tmp_path: Path) -> None:
        path = tmp_path / "explicit.medh5"
        with medh5.create(path, sample_id="explicit") as w:
            w.add_timepoint("baseline")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="baseline")
            w.add_image("CT", _image(), grid="g", modality="CT")
        with medh5.open(path) as sample:
            assert sample.grids["g"].timepoint == "baseline"
            assert set(sample.at("baseline").images) == {"CT"}

    def test_medh5_timeline_lists_the_visit(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "single.medh5"
        self._write(path)
        assert main(["timeline", str(path)]) == 0
        out = capsys.readouterr().out
        assert "CT" in out and "lesions" in out
        assert main(["track", str(path), "--json"]) == 0
        assert '"tp0": "present"' in capsys.readouterr().out

    def test_converters_name_the_visit_they_declare(self, tmp_path: Path) -> None:
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti

        volume = np.random.default_rng(0).integers(0, 200, (10, 10, 8)).astype(np.int16)
        source = tmp_path / "ct.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.diag([1.0, 1.0, 2.0, 1.0])), str(source))
        from_nifti({"CT": source}, tmp_path / "out.medh5")
        with open_h5(tmp_path / "out.medh5", "r") as handle:
            assert handle["grids/ref"].attrs["timepoint"] == "tp0"


class TestMonaiF04LabelDtype:
    """`to_dict` labels are `int64`: the full id range, and the ignore id intact."""

    def test_S5_3_ids_beyond_int16_and_the_ignore_id_survive(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("monai")
        from medh5.monai import to_dict

        wide = LabelClass(40000, "wide", "A class beyond int16")
        path = tmp_path / "wide.medh5"
        ignore = np.zeros(SHAPE, dtype=bool)
        ignore[-1] = True
        with medh5.create(path, sample_id="wide") as w:
            w.label_set(_label_set(wide))
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT")
            w.add_segmentation(
                "organs",
                grid="g",
                masks={40000: _mask()},
                encoding="labelmap",
                ignore=ignore,
            )
        with medh5.open(path) as sample:
            item = to_dict(sample, ["CT"], ["organs"])
        label = item["organs"]
        assert str(label.dtype) == "torch.int64"
        values = set(np.unique(label.numpy()).tolist())
        assert 40000 in values
        assert 65535 in values
        assert -1 not in values


class TestF05NiftiRescale:
    """§4.2: `scl_slope`/`scl_inter` are the file's rescale, stored, not applied."""

    SLOPE, INTERCEPT = 2.0, -1024.0

    def _scaled_nifti(
        self, path: Path, stored: Any, slope: float, inter: float
    ) -> None:
        """A NIfTI whose header scales its `int16` voxels.

        nibabel consumes the header fields on load --- they move onto the
        array proxy and `scl_slope` reads back NaN --- so the check that the
        file carries the scale asks the proxy, as the converter does.
        """
        nib = pytest.importorskip("nibabel")
        image = nib.Nifti1Image(stored, np.eye(4))
        image.header.set_slope_inter(slope, inter)
        nib.save(image, str(path))
        loaded = nib.load(str(path))
        assert float(loaded.dataobj.slope) == slope
        assert float(loaded.dataobj.inter) == inter

    def test_S4_2_the_scale_is_recorded_and_the_dtype_kept(
        self, tmp_path: Path
    ) -> None:
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti

        stored = (
            np.random.default_rng(2).integers(0, 2000, (12, 10, 8)).astype(np.int16)
        )
        source = tmp_path / "ct.nii"
        self._scaled_nifti(source, stored, self.SLOPE, self.INTERCEPT)
        report = from_nifti({"CT": source}, tmp_path / "ct.medh5")
        assert report.of_kind("value_scale"), "the decision is recorded"
        expected = nib.load(str(source)).get_fdata()
        with medh5.open(tmp_path / "ct.medh5") as sample:
            image = sample.images["CT"]
            assert image.dtype == np.int16
            assert image.rescale == (self.SLOPE, self.INTERCEPT)
            physical = image.read(physical=True)
        # Back to NIfTI (x, y, z) order for the comparison.
        assert np.allclose(np.transpose(physical, (2, 1, 0)), expected)

    def test_S4_2_an_unscaled_file_records_no_rescale(self, tmp_path: Path) -> None:
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti

        stored = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        source = tmp_path / "plain.nii.gz"
        nib.save(nib.Nifti1Image(stored, np.eye(4)), str(source))
        report = from_nifti({"CT": source}, tmp_path / "plain.medh5")
        assert not report.of_kind("value_scale")
        with medh5.open(tmp_path / "plain.medh5") as sample:
            assert sample.images["CT"].rescale == (1.0, 0.0)
            assert not sample.images["CT"].is_rescaled

    def test_S4_2_a_scaled_mask_is_thresholded_after_the_scale(
        self, tmp_path: Path
    ) -> None:
        from medh5.io.nifti import from_nifti

        stored = np.zeros((6, 6, 6), dtype=np.int16)
        stored[:3] = 1  # physical 0 under intercept -1: not in the mask
        stored[3:] = 2  # physical 1: in the mask
        image = tmp_path / "ct.nii"
        mask = tmp_path / "mask.nii"
        self._scaled_nifti(image, np.ones((6, 6, 6), dtype=np.int16), 1.0, 0.0)
        self._scaled_nifti(mask, stored, 1.0, -1.0)
        from_nifti({"CT": image}, tmp_path / "m.medh5", masks={"liver": mask})
        with medh5.open(tmp_path / "m.medh5") as sample:
            liver = sample.annotations["seg"].dense(["liver"])[0]
        # NIfTI x becomes the trailing medh5 axis, so the split is along x.
        assert not liver[..., :3].any()
        assert liver[..., 3:].all()


class TestF06DicomSameModality:
    """A study with several series of one modality imports, named by series."""

    def test_two_series_of_one_modality_import_side_by_side(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("pydicom")
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from tests.v1.conftest import write_dicom_series

        study = generate_uid()
        root = tmp_path / "dicom"
        first = write_dicom_series(
            root / "a",
            patient_id="P1",
            study_uid=study,
            study_date="20260101",
            modality="MR",
            seed=1,
        )
        second = write_dicom_series(
            root / "b",
            patient_id="P1",
            study_uid=study,
            study_date="20260101",
            modality="MR",
            seed=2,
        )
        report = from_dicom(root, tmp_path / "out.medh5")
        assert report.of_kind("image_ids"), "the numbering is recorded"
        by_uid = {first["series_uid"]: first, second["series_uid"]: second}
        ordered = sorted(by_uid)  # SeriesInstanceUID order
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["MR_1_tp0", "MR_2_tp0"]
            assert sorted(sample.grids) == ["mr_1_tp0", "mr_2_tp0"]
            uids = sample.timepoints["tp0"].series_uids
            assert uids == {"MR_1_tp0": ordered[0], "MR_2_tp0": ordered[1]}
            assert sample.images["MR_1_tp0"].grid_id == "mr_1_tp0"

    def test_a_single_series_keeps_its_short_name(self, tmp_path: Path) -> None:
        pytest.importorskip("pydicom")
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from tests.v1.conftest import write_dicom_series

        series = write_dicom_series(
            tmp_path / "dicom",
            patient_id="P1",
            study_uid=generate_uid(),
            study_date="20260101",
        )
        report = from_dicom(tmp_path / "dicom", tmp_path / "out.medh5")
        assert not report.of_kind("image_ids")
        with medh5.open(tmp_path / "out.medh5") as sample:
            assert sorted(sample.images) == ["CT_tp0"]
            assert sample.timepoints["tp0"].series_uids == {
                "CT_tp0": series["series_uid"]
            }


class TestOneLiners:
    def test_the_published_readme_counts_its_collections(self) -> None:
        from medh5.conformance import CASES
        from medh5.conformance.suite import _readme

        shards = sum(1 for c in CASES if c.suffix == ".medh5c")
        assert shards == 4
        assert f"{len(CASES) - shards} samples and {shards} collections" in _readme(
            CASES
        )

    def test_verify_names_the_content_id_state(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        path = tmp_path / "v.medh5"
        with medh5.create(path, sample_id="v") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", _image(), grid="g", modality="CT")
        assert main(["verify", str(path)]) == 0
        assert "content_id ok" in capsys.readouterr().out
        assert main(["verify", str(path), "--partial", "images/CT"]) == 0
        assert "content_id not verified" in capsys.readouterr().out

    def test_the_torch_docstring_agrees_with_the_docs(self) -> None:
        import medh5.torch as torch_pkg

        assert "not optional" not in (torch_pkg.__doc__ or "")
        assert "recommended" in (torch_pkg.__doc__ or "")
