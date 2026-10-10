"""nnU-Net v2 datasets, both ways (``medh5.io.nnunetv2``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from tests.kits import Numbered, Organs

nib = pytest.importorskip("nibabel")


class TestNnunet:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> Path:
        root = tmp_path / "Dataset001_Test"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (12, 10, 6)
        for case in ("CASE_001", "CASE_002"):
            rng = np.random.default_rng(len(case))
            for channel in range(2):
                nib.save(
                    nib.Nifti1Image(
                        rng.integers(0, 500, shape).astype(np.int16), np.eye(4)
                    ),
                    str(root / "imagesTr" / f"{case}_{channel:04d}.nii.gz"),
                )
            labels = np.zeros(shape, np.uint8)
            labels[2:8, 2:6, 1:4] = 1
            labels[4:6, 3:5, 2:3] = 2
            nib.save(
                nib.Nifti1Image(labels, np.eye(4)),
                str(root / "labelsTr" / f"{case}.nii.gz"),
            )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "T1", "1": "FLAIR"},
                    "labels": {
                        "background": 0,
                        "edema": 1,
                        "enhancing": 2,
                        "whole_tumor": [1, 2],
                    },
                    "numTraining": 2,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_S5_2_nnunet_ids_are_kept(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        report = from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            ids = {c.key: c.id for c in sample.label_set}
            assert ids["edema"] == 1
            assert ids["enhancing"] == 2
        assert report.of_kind("label_ids")
        assert report.of_kind("background"), "background 0 is reserved (§5.3)"

    def test_S7_regions_become_classes_over_their_components(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            seg = sample.annotations["seg"]
            edema = seg.dense(["edema"])[0]
            enhancing = seg.dense(["enhancing"])[0]
            whole = seg.dense(["whole_tumor"])[0]
            assert np.array_equal(whole, edema | enhancing)
            assert sample.label_set["edema"].parents == (
                sample.label_set["whole_tumor"].id,
            )

    def test_S11_3_an_nnunet_label_volume_is_exhaustive(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            assert sample.annotations["seg"].is_fully_covered

    def test_the_round_trip_reproduces_the_dataset(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2, read_dataset_json, to_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        cases = sorted((tmp_path / "out").glob("*.medh5"))
        to_nnunetv2(cases, tmp_path / "back")
        exported = read_dataset_json(tmp_path / "back" / "Dataset001_medh5")
        assert exported["labels"] == {
            "background": 0,
            "edema": 1,
            "enhancing": 2,
            "whole_tumor": [1, 2],
        }
        assert exported["channel_names"] == {"0": "T1", "1": "FLAIR"}
        for name in ("labelsTr/CASE_001.nii.gz", "imagesTr/CASE_001_0000.nii.gz"):
            original = nib.load(str(dataset / name))
            back = nib.load(str(tmp_path / "back" / "Dataset001_medh5" / name))
            assert np.array_equal(
                np.asanyarray(original.dataobj), np.asanyarray(back.dataobj)
            ), name
            assert np.allclose(original.affine, back.affine), name

    def test_dataset_json_is_stashed_verbatim(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        from_nnunetv2(dataset, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE_001.medh5") as sample:
            stashed = sample.document.extra["nnunetv2"]
            assert stashed == json.loads(
                (dataset / "dataset.json").read_text(encoding="utf-8")
            )

    def test_a_malformed_dataset_json_is_named(self, tmp_path):
        from medh5.io.nnunetv2 import read_dataset_json

        (tmp_path / "dataset.json").write_text(
            json.dumps({"labels": {}}), encoding="utf-8"
        )
        with pytest.raises(MEDH5ValidationError, match="missing"):
            read_dataset_json(tmp_path)
        (tmp_path / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"1": "T1"},
                    "labels": {"a": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MEDH5ValidationError, match="0\\.\\.0"):
            read_dataset_json(tmp_path)
        with pytest.raises(MEDH5ValidationError, match="not found"):
            read_dataset_json(tmp_path / "nope")

    def test_S3_2_a_channel_on_another_grid_is_refused(self, dataset, tmp_path):
        """A second channel is written onto the first one's grid, so it has to
        share it.  Same shape, different spacing: nothing in the array reveals
        that these are different volumes of the patient."""
        from medh5.io.nnunetv2 import from_nnunetv2

        odd = np.diag([2.0, 2.0, 2.0, 1.0])
        odd[:3, 3] = [50.0, 0.0, 0.0]
        nib.save(
            nib.Nifti1Image(np.zeros((12, 10, 6), np.int16), odd),
            str(dataset / "imagesTr" / "CASE_001_0001.nii.gz"),
        )
        with pytest.raises(MEDH5ValidationError, match="spacing"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])

    def test_S3_2_a_label_volume_on_another_grid_is_refused(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        shifted = np.eye(4)
        shifted[:3, 3] = [0.0, 0.0, 9.0]
        nib.save(
            nib.Nifti1Image(np.zeros((12, 10, 6), np.uint8), shifted),
            str(dataset / "labelsTr" / "CASE_001.nii.gz"),
        )
        with pytest.raises(MEDH5ValidationError, match="origin"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])

    def test_a_label_named_with_spaces_survives_the_round_trip(self, tmp_path):
        """`dataset.json` names are free text; the label set key is sanitised
        from them.  Matching classes back by name therefore finds nothing for
        any dataset that capitalises, and the export silently wrote an
        all-background volume.  Classes are matched by id instead."""
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        root = tmp_path / "Dataset002_Named"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (8, 8, 4)
        nib.save(
            nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
            str(root / "imagesTr" / "CASE_0000.nii.gz"),
        )
        volume = np.zeros(shape, np.uint8)
        volume[1:4, 1:4, 1:3] = 1
        volume[5:7, 5:7, 1:3] = 2
        nib.save(
            nib.Nifti1Image(volume, np.eye(4)),
            str(root / "labelsTr" / "CASE.nii.gz"),
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "Tumour Core": 1, "GTV": 2},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        imported = tmp_path / "imported"
        from_nnunetv2(root, imported)
        to_nnunetv2([imported / "CASE.medh5"], tmp_path / "back")
        written = np.asarray(
            nib.load(
                str(tmp_path / "back" / "Dataset001_medh5" / "labelsTr" / "CASE.nii.gz")
            ).dataobj
        )
        assert int((written == 1).sum()) == int((volume == 1).sum())
        assert int((written == 2).sum()) == int((volume == 2).sum())

    def test_a_class_the_sample_lacks_is_refused_not_dropped(self, tmp_path):
        from medh5.io.nnunetv2 import _labelmap_for

        class _Stub:
            ann_id = "seg"
            class_ids = (1,)
            spatial_shape = (2, 2, 2)

            def dense(self, ids):
                return np.zeros((1, 2, 2, 2), bool)

            def resolve_class(self, key):
                raise KeyError(key)

        with pytest.raises(MEDH5ValidationError, match="carries no class"):
            _labelmap_for(_Stub(), {"background": 0, "kidney": 1, "spleen": 7})

    def test_a_missing_channel_is_named(self, dataset, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        (dataset / "imagesTr" / "CASE_001_0001.nii.gz").unlink()
        with pytest.raises(MEDH5ValidationError, match="channel 1"):
            from_nnunetv2(dataset, tmp_path / "out", case_ids=["CASE_001"])

    def test_F10_an_nnunet_export_keeps_the_physical_values(self, tmp_path: Path):
        nib = pytest.importorskip("nibabel")
        from medh5.io.nnunetv2 import to_nnunetv2

        path = tmp_path / "ct.medh5"
        with medh5.create(path, sample_id="ct", codec="portable") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(2.0, 0.8, 0.9))
            w.add_image(
                "CT",
                np.full(Organs.SHAPE, 1000, np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
            )
        to_nnunetv2([path], tmp_path / "nn", dataset_name="D1", unlabeled="test")
        image = nib.load(str(tmp_path / "nn/D1/imagesTs/ct_0000.nii.gz"))
        assert image.get_data_dtype() == np.int16
        assert image.get_fdata().flat[0] == pytest.approx(-24.0)

    def test_F11_a_2D_nnunet_dataset_round_trips(self, tmp_path: Path):
        nib = pytest.importorskip("nibabel")
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        affine = np.diag([0.8, 0.9, 1.0, 1.0])
        data = np.arange(32 * 40, dtype=np.int16).reshape(32, 40)
        labels = np.zeros((32, 40), np.uint8)
        labels[4:8, 4:8] = 1
        root = tmp_path / "D2"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        nib.save(nib.Nifti1Image(data, affine), str(root / "imagesTr/C1_0000.nii.gz"))
        nib.save(nib.Nifti1Image(labels, affine), str(root / "labelsTr/C1.nii.gz"))
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "XR"},
                    "labels": {"background": 0, "lesion": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        from_nnunetv2(root, tmp_path / "out")
        report = to_nnunetv2(
            sorted((tmp_path / "out").glob("*.medh5")),
            tmp_path / "back",
            dataset_name="D2",
        )
        back = nib.load(str(tmp_path / "back/D2/imagesTr/C1_0000.nii.gz"))
        assert np.array_equal(np.asanyarray(back.dataobj), data)
        assert report.ok

    def test_L18_the_report_lists_only_files_that_were_written(self, tmp_path: Path):
        pytest.importorskip("nibabel")
        from medh5.io.nnunetv2 import to_nnunetv2

        path = tmp_path / "bare.medh5"
        with medh5.create(path, sample_id="bare", codec="portable") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
        report = to_nnunetv2(
            [path], tmp_path / "nn", dataset_name="D3", unlabeled="test"
        )
        for output in report.outputs:
            assert Path(output).exists(), output
        assert not any("labelsTr" in o for o in report.outputs)
        assert any("imagesTs" in o for o in report.outputs)


class TestNnunetScaling:
    """A NIfTI's `scl_slope`/`scl_inter` are what its voxels mean, to nnU-Net
    as to any reader (L03 of the 2.0 audit): an image's were dropped, and a
    label volume's ids were matched before them."""

    @staticmethod
    def _dataset(
        tmp_path: Path,
        *,
        image_scale: tuple[float, float] | None = None,
        label_scale: tuple[float, float] | None = None,
    ) -> tuple[Path, np.ndarray, np.ndarray]:
        root = tmp_path / "Dataset002_Scaled"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (6, 5, 4)
        stored = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        labels = np.zeros(shape, np.uint8)
        labels[1:3, 1:3, 1:3] = 1
        for array, scale, name in (
            (stored, image_scale, "imagesTr/CASE_0000.nii.gz"),
            (labels, label_scale, "labelsTr/CASE.nii.gz"),
        ):
            image = nib.Nifti1Image(array, np.eye(4))
            if scale is not None:
                image.header.set_slope_inter(*scale)
            nib.save(image, str(root / name))
            if scale is not None:
                assert float(nib.load(str(root / name)).dataobj.slope) == scale[0]
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "organ": 32},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        return root, stored, labels

    def test_L03_an_image_keeps_its_scale_as_its_rescale(self, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        root, stored, _ = self._dataset(tmp_path, image_scale=(1.0, -1024.0))
        report = from_nnunetv2(root, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE.medh5") as sample:
            image = sample.images["CT"]
            assert image.rescale == (1.0, -1024.0)
            physical = image.read(physical=True)
            assert np.allclose(physical, np.transpose(stored, (2, 1, 0)) - 1024.0)
        assert report.of_kind("value_scale")

    def test_L03_labels_are_matched_after_their_scale(self, tmp_path):
        """Stored 1 with slope 32 is class 32 to every reader of the file."""
        from medh5.io.nnunetv2 import from_nnunetv2

        root, _, labels = self._dataset(tmp_path, label_scale=(32.0, 0.0))
        from_nnunetv2(root, tmp_path / "out")
        with medh5.open(tmp_path / "out" / "CASE.medh5") as sample:
            organ = sample.annotations["seg"].dense(["organ"])[0]
            assert int(organ.sum()) == int(labels.sum()) == 8

    def test_L03_a_scale_that_makes_fractional_labels_is_refused(self, tmp_path):
        from medh5.io.nnunetv2 import from_nnunetv2

        root, _, _ = self._dataset(tmp_path, label_scale=(0.5, 0.0))
        with pytest.raises(MEDH5ValidationError, match="not integers"):
            from_nnunetv2(root, tmp_path / "out")

    def test_L04_labels_on_another_grid_than_the_channels_are_refused(self, tmp_path):
        """The export wrote the labels with the reference grid's affine,
        whatever grid they were on."""
        from medh5.io.nnunetv2 import to_nnunetv2

        path = tmp_path / "s.medh5"
        with Organs.writer(path) as w:
            w.add_grid("pet", shape=(4, 6, 6), spacing=(4.0, 1.6, 1.6), timepoint="tp0")
            mask = np.zeros((4, 6, 6), bool)
            mask[1:3, 2:4, 2:4] = True
            w.add_segmentation("seg", grid="pet", masks={1: mask})
        with pytest.raises(MEDH5ValidationError, match="channels' grid"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")


class TestNnunetIgnore:
    """nnU-Net's declared ignore label is the annotation's §7.7 ignore region,
    both ways (N02 of the 2.0 re-audit): the export wrote ignored voxels as 0
    --- verified background for every class --- and declared no ignore label,
    and the import read nnU-Net's ignore label as a class named ``ignore``."""

    @pytest.mark.parametrize("overlap", [False, True])
    def test_N02_ignored_voxels_round_trip_as_the_ignore_label(self, tmp_path, overlap):
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        liver, spleen, region = Numbered.two_classes()
        if overlap:  # a region over a class is stored as the sibling mask (§7.7)
            region = region.copy()
            region[1:3, 1:3, 1:3] = True
        path = tmp_path / "s1.medh5"
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: liver, 2: spleen},
                annotated_classes="all",
                ignore=region,
            )
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D9")
        labels = json.loads((tmp_path / "nn/D9/dataset.json").read_text())["labels"]
        assert labels["ignore"] == 4  # one above every other value
        (decision,) = report.of_kind("ignore")
        assert ("over a class" in decision.message) is overlap

        back = from_nnunetv2(tmp_path / "nn/D9", tmp_path / "back")
        assert back.of_kind("ignore")
        with medh5.open(tmp_path / "back/s1.medh5") as sample:
            assert "ignore" not in [c.key for c in sample.label_set]
            assert np.array_equal(sample.ignore_region("seg"), region)
            seg = sample.annotations["seg"]
            for class_id, mask in ((1, liver), (2, spleen)):
                assert np.array_equal(seg.dense([class_id])[0], mask & ~region)

    def test_N02_without_an_ignore_region_no_label_is_declared(self, tmp_path):
        from medh5.io.nnunetv2 import to_nnunetv2

        liver, spleen, _ = Numbered.two_classes()
        path = tmp_path / "s1.medh5"
        with Numbered.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks={1: liver, 2: spleen})
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D9")
        labels = json.loads((tmp_path / "nn/D9/dataset.json").read_text())["labels"]
        assert "ignore" not in labels and not report.of_kind("ignore")

    def test_N02_an_ignore_label_that_cannot_be_highest_is_refused(self):
        """nnU-Net's ignore label must be the highest value, and the label
        volume is uint16: above a class 65535 there is no room for it."""
        from medh5.io.nnunetv2 import _ignore_value

        assert _ignore_value({"background": 0, "a": 3, "r": [1, 7]}) == 8
        assert _ignore_value({"background": 0, "a": 3, "ignore": 9}) == 9
        with pytest.raises(MEDH5ValidationError, match="does not fit"):
            _ignore_value({"background": 0, "top": 65535})


class TestNnunetDatasetLabels:
    """The labels are the dataset's, each case is held to them, and every label
    volume gives back what it exports (N13 and N14 of the round-3 audit); every
    channel and the labels are one physical space (N17)."""

    SHAPE = (6, 8, 8)

    @classmethod
    def _case(
        cls,
        path: Path,
        masks: dict[int, np.ndarray],
        *,
        searched: list[int],
        label_grid: dict | None = None,
    ) -> Path:
        from medh5.labels import LabelClass, LabelSet

        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid("g", shape=cls.SHAPE, spacing=(2.0, 1.0, 1.0), frame_uid="F")
            w.add_image("CT", np.zeros(cls.SHAPE, np.int16), grid="g", modality="CT")
            w.label_set(
                LabelSet(
                    "organs",
                    version="1.0.0",
                    classes=[
                        LabelClass(1, "organ", "Organ"),
                        LabelClass(2, "lesion", "Lesion"),
                    ],
                )
            )
            grid = "g"
            if label_grid is not None:
                options = {"spacing": (2.0, 1.0, 1.0), **label_grid}
                w.add_grid("h", shape=cls.SHAPE, **options)
                grid = "h"
            w.add_segmentation(
                "seg", grid=grid, masks=masks, annotated_classes=searched
            )
        return path

    @classmethod
    def _blocks(cls) -> tuple[np.ndarray, np.ndarray]:
        organ = np.zeros(cls.SHAPE, bool)
        organ[1:3, 1:4, 1:4] = True
        lesion = np.zeros(cls.SHAPE, bool)
        lesion[3:5, 5:7, 5:7] = True
        return organ, lesion

    def test_N14_a_case_not_searched_for_a_class_is_refused(self, tmp_path: Path):
        """The labels were the first case's: a lesion only the second case
        carried became background, and the first case's unsearched lesion
        was written as its verified absence."""
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, lesion = self._blocks()
        a = self._case(tmp_path / "A.medh5", {1: organ}, searched=[1])
        b = self._case(tmp_path / "B.medh5", {1: organ, 2: lesion}, searched=[1, 2])
        for order in ([a, b], [b, a]):
            with pytest.raises(MEDH5ValidationError, match="not searched") as caught:
                to_nnunetv2(order, tmp_path / "nn", dataset_name="D1")
            # The sample is valid; what fails is nnU-Net's exhaustive volume,
            # which no §15.2 code describes.
            assert caught.value.code is None and "lesion" in str(caught.value)
            assert not (tmp_path / "nn").exists(), "refused before anything is written"
        # The subset every case examined, on request.
        report = to_nnunetv2(
            [a, b], tmp_path / "nn", dataset_name="D1", classes=["organ"]
        )
        assert report.of_kind("classes")
        labels = json.loads((tmp_path / "nn/D1/dataset.json").read_text())["labels"]
        assert labels == {"background": 0, "organ": 1}
        written = np.asarray(
            nib.load(str(tmp_path / "nn/D1/labelsTr/B.nii.gz")).dataobj
        )
        assert int((written == 1).sum()) == int(organ.sum()) and written.max() == 1

    def test_N14_the_command_line_names_the_classes_to_export(self, tmp_path, capsys):
        """The refusal says to name the classes every case examined; the
        command line had no way to."""
        from medh5.cli import EXIT_OK, main

        organ, lesion = self._blocks()
        a = self._case(tmp_path / "A.medh5", {1: organ}, searched=[1])
        b = self._case(tmp_path / "B.medh5", {1: organ, 2: lesion}, searched=[1, 2])
        argv = ["convert", "to-nnunet", str(tmp_path / "nn"), str(a), str(b)]
        assert main([*argv, "--dataset-name", "D1"]) != EXIT_OK
        assert "not searched" in capsys.readouterr().err
        assert main([*argv, "--dataset-name", "D2", "--class", "organ"]) == EXIT_OK
        labels = json.loads((tmp_path / "nn/D2/dataset.json").read_text())["labels"]
        assert labels == {"background": 0, "organ": 1}
        assert (
            main([*argv, "--dataset-name", "D3", "--class", "1", "--class", "2"])
            != EXIT_OK
        )

    def test_N14_complete_coverage_exports_in_either_order(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, lesion = self._blocks()
        a = self._case(tmp_path / "A.medh5", {1: organ}, searched=[1, 2])
        b = self._case(tmp_path / "B.medh5", {1: organ, 2: lesion}, searched=[1, 2])
        for name, order in (("D1", [a, b]), ("D2", [b, a])):
            to_nnunetv2(order, tmp_path / "nn", dataset_name=name)
            labels = json.loads((tmp_path / f"nn/{name}/dataset.json").read_text())
            assert labels["labels"] == {"background": 0, "organ": 1, "lesion": 2}
            written = np.asarray(
                nib.load(str(tmp_path / f"nn/{name}/labelsTr/B.nii.gz")).dataobj
            )
            assert int((written == 2).sum()) == int(lesion.sum())

    def test_N14_overlapping_classes_are_refused_not_overwritten(self, tmp_path):
        """One value per voxel: an organ under a lesion lost those voxels."""
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, _ = self._blocks()
        lesion = np.zeros_like(organ)
        lesion[1:2, 1:3, 1:3] = True  # inside the organ
        path = self._case(tmp_path / "A.medh5", {1: organ, 2: lesion}, searched=[1, 2])
        with pytest.raises(MEDH5ValidationError, match="does not give back") as caught:
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        assert caught.value.code is None and "'organ'" in str(caught.value)
        # "Nothing was written for this case": its channels included.
        assert not [p for p in (tmp_path / "nn").rglob("*") if p.is_file()]

    @pytest.mark.parametrize(
        "label_grid",
        [
            {"coord_system": "RAS", "frame_uid": "F"},
            {"units": "m", "frame_uid": "F"},
            {"frame_uid": "G"},
        ],
        ids=["convention", "units", "frame"],
    )
    def test_N17_labels_in_another_space_are_refused(self, tmp_path, label_grid):
        """`is_congruent` compares numbers: a label grid with the channels'
        numbers in RAS, in metres, or in another known frame exported a pair
        48 mm, or a thousandfold, apart."""
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, _ = self._blocks()
        path = self._case(
            tmp_path / "A.medh5", {1: organ}, searched=[1], label_grid=label_grid
        )
        with pytest.raises(MEDH5ValidationError, match="channels' grid"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        assert not (tmp_path / "nn").exists()

    def test_N17_labels_on_another_lattice_of_the_space_are_refused(self, tmp_path):
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, _ = self._blocks()
        path = self._case(
            tmp_path / "A.medh5",
            {1: organ},
            searched=[1],
            label_grid={"spacing": (1.0, 1.0, 1.0), "frame_uid": "F"},
        )
        with pytest.raises(MEDH5ValidationError, match="another lattice"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")

    def test_N17_a_channel_in_another_space_is_refused(self, tmp_path):
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, _ = self._blocks()
        path = self._case(tmp_path / "A.medh5", {1: organ}, searched=[1])
        with medh5.amend(path) as w:
            w.add_grid(
                "r", shape=self.SHAPE, spacing=(2.0, 1.0, 1.0), coord_system="RAS"
            )
            w.add_image("MR", np.zeros(self.SHAPE, np.int16), grid="r", modality="MR")
        with pytest.raises(MEDH5ValidationError, match="channel 'MR'"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")

    def test_N17_one_space_on_two_grids_exports(self, tmp_path):
        """The control: a second grid with the channels' lattice, convention,
        units and frame is their space; both files state one affine."""
        from medh5.io.nnunetv2 import to_nnunetv2

        organ, _ = self._blocks()
        path = self._case(
            tmp_path / "A.medh5",
            {1: organ},
            searched=[1],
            label_grid={"frame_uid": "F"},
        )
        to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        image = nib.load(str(tmp_path / "nn/D1/imagesTr/A_0000.nii.gz"))
        labels = nib.load(str(tmp_path / "nn/D1/labelsTr/A.nii.gz"))
        assert np.allclose(image.affine, labels.affine)

    @staticmethod
    def _regions(root: Path, *, order: bool) -> np.ndarray:
        """BraTS's shape: three nested regions over components 1-3, an ignore
        label 4, `regions_class_order` [1, 2, 3]."""
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (10, 10, 8)
        nib.save(
            nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
            str(root / "imagesTr/CASE_0000.nii.gz"),
        )
        volume = np.zeros(shape, np.uint8)
        volume[0, 0, :8] = 1
        volume[1, 0, :8] = 2
        volume[2, 0, :8] = 3
        volume.reshape(-1)[200:242] = 4
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(root / "labelsTr/CASE.nii.gz"))
        document: dict = {
            "channel_names": {"0": "FLAIR"},
            "labels": {
                "background": 0,
                "whole_tumor": [1, 2, 3],
                "tumor_core": [2, 3],
                "enhancing_tumor": 3,
                "ignore": 4,
            },
            "numTraining": 1,
            "file_ending": ".nii.gz",
        }
        if order:
            document["regions_class_order"] = [1, 2, 3]
        (root / "dataset.json").write_text(json.dumps(document), encoding="utf-8")
        return volume

    def test_N13_a_region_dataset_round_trips_every_voxel(self, tmp_path: Path):
        """Components 1 and 2 exist only inside regions: the export wrote the
        scalar labels and left 16 tumour voxels as background.  Painted in
        `regions_class_order`, as nnU-Net converts regions back."""
        from medh5.io.nnunetv2 import from_nnunetv2, read_dataset_json, to_nnunetv2

        volume = self._regions(tmp_path / "D", order=True)
        from_nnunetv2(tmp_path / "D", tmp_path / "out")
        to_nnunetv2([tmp_path / "out/CASE.medh5"], tmp_path / "back")
        back = tmp_path / "back/Dataset001_medh5"
        written = np.asarray(nib.load(str(back / "labelsTr/CASE.nii.gz")).dataobj)
        assert np.array_equal(written, volume)
        assert read_dataset_json(back) == read_dataset_json(tmp_path / "D")

    def test_N13_regions_no_order_reconstructs_are_refused(self, tmp_path: Path):
        """Without `regions_class_order` the components cannot be painted, and
        the regions are refused rather than written as background."""
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        self._regions(tmp_path / "D", order=False)
        from_nnunetv2(tmp_path / "D", tmp_path / "out")
        with pytest.raises(MEDH5ValidationError, match="does not give back") as caught:
            to_nnunetv2([tmp_path / "out/CASE.medh5"], tmp_path / "back")
        assert "whole_tumor" in str(caught.value)


def nnunet_checks(root: Path) -> dict:
    """What nnU-Net's dataset check holds an export to: ``numTraining`` is the
    number of training cases, each with every channel and a label file, the
    label values are consecutive (the ignore label, if any, the highest), and
    every label file holds only declared values."""
    document = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    ending = document["file_ending"]
    channels = len(document["channel_names"])
    labels = document["labels"]
    scalar = sorted(
        int(v) for k, v in labels.items() if not isinstance(v, list) and k != "ignore"
    )
    assert scalar == list(range(len(scalar))), labels
    if "ignore" in labels:
        assert int(labels["ignore"]) == max(scalar) + 1, labels
    allowed = {*scalar, *([int(labels["ignore"])] if "ignore" in labels else [])}
    cases = sorted(
        p.name[: -len(ending)] for p in (root / "labelsTr").glob(f"*{ending}")
    )
    assert document["numTraining"] == len(cases), (document["numTraining"], cases)
    images = sorted((root / "imagesTr").glob(f"*{ending}"))
    assert len(images) == channels * len(cases)
    for case in cases:
        for channel in range(channels):
            assert (root / "imagesTr" / f"{case}_{channel:04d}{ending}").exists()
        values = np.unique(
            np.asarray(nib.load(str(root / "labelsTr" / f"{case}{ending}")).dataobj)
        )
        assert set(int(v) for v in values) <= allowed, (case, values, allowed)
    return document


class TestF05CaseNames:
    """F05 of the round-4 audit: a case's ``sample_id`` was joined onto the
    export as a path, so ``../../victim`` --- or an absolute id --- wrote
    outside it, and two ids differing only in case overwrote each other on
    macOS and Windows while ``numTraining`` counted both."""

    @staticmethod
    def _case(path: Path, sample_id: str) -> Path:
        with medh5.create(path, sample_id=sample_id, codec="portable") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
            mask = np.zeros(Organs.SHAPE, bool)
            mask[1:3, 1:3, 1:3] = True
            w.add_segmentation("seg", grid="g", masks={1: mask})
        return path

    @pytest.mark.parametrize("sample_id", ["../../victim", "/tmp/victim", "..", "a b"])
    def test_F05_a_case_name_that_is_no_file_name_is_refused(
        self, tmp_path: Path, sample_id: str
    ):
        from medh5.io.nnunetv2 import to_nnunetv2

        path = self._case(tmp_path / "bad.medh5", sample_id)
        with pytest.raises(MEDH5ValidationError) as caught:
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        # A sample_id is free text in a valid sample: the refusal is the
        # export's, so it borrows no §15.2 code.
        assert caught.value.code is None and "sample_id" in str(caught.value)
        assert not (tmp_path / "nn").exists(), "refused before anything is written"
        assert not list(tmp_path.rglob("victim*"))

    def test_F05_names_that_differ_only_in_case_are_refused(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        a = self._case(tmp_path / "a.medh5", "Case1")
        b = self._case(tmp_path / "b.medh5", "case1")
        with pytest.raises(MEDH5ValidationError, match="'Case1'"):
            to_nnunetv2([a, b], tmp_path / "nn", dataset_name="D1")
        assert not (tmp_path / "nn").exists()

    @pytest.mark.parametrize("name", ["../D1", "/abs", ".."])
    def test_F05_the_dataset_name_is_a_name(self, tmp_path: Path, name: str):
        from medh5.io.nnunetv2 import to_nnunetv2

        path = self._case(tmp_path / "a.medh5", "case1")
        with pytest.raises(MEDH5ValidationError, match="dataset_name"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name=name)
        assert not (tmp_path / "nn").exists()

    def test_F05_valid_names_export(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        paths = [self._case(tmp_path / f"{n}.medh5", n) for n in ("c-1", "c.2", "C_3")]
        to_nnunetv2(paths, tmp_path / "nn", dataset_name="D1")
        nnunet_checks(tmp_path / "nn" / "D1")


class TestF22Unlabeled:
    """F22 of the round-4 audit: a case without the annotation was written to
    ``imagesTr`` with no label file, and counted in ``numTraining``."""

    def test_F22_an_unlabeled_case_is_refused_or_a_test_case(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import EXIT_OK, main
        from medh5.io.nnunetv2 import to_nnunetv2

        labeled = TestF05CaseNames._case(tmp_path / "a.medh5", "labeled")
        bare = tmp_path / "bare.medh5"
        with medh5.create(bare, sample_id="bare", codec="portable") as w:
            w.add_grid("g", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
        with pytest.raises(MEDH5ValidationError, match="bare.medh5") as caught:
            to_nnunetv2([labeled, bare], tmp_path / "nn", dataset_name="D1")
        assert "imagesTs" in str(caught.value)
        assert not (tmp_path / "nn").exists()
        report = to_nnunetv2(
            [labeled, bare], tmp_path / "nn", dataset_name="D1", unlabeled="test"
        )
        root = tmp_path / "nn" / "D1"
        document = nnunet_checks(root)
        assert document["numTraining"] == 1
        assert (root / "imagesTs" / "bare_0000.nii.gz").exists()
        assert not (root / "imagesTr" / "bare_0000.nii.gz").exists()
        assert report.of_kind("unlabeled")
        argv = ["convert", "to-nnunet", str(tmp_path / "cli"), str(labeled), str(bare)]
        assert main([*argv, "--dataset-name", "D2"]) != EXIT_OK
        assert "unlabeled" in capsys.readouterr().err
        assert main([*argv, "--dataset-name", "D2", "--unlabeled", "test"]) == EXIT_OK
        assert nnunet_checks(tmp_path / "cli" / "D2")["numTraining"] == 1


class TestF23ConsecutiveLabels:
    """F23 of the round-4 audit: class ids with a gap were written as label
    values with one, and nnU-Net's dataset check refuses them.  They are
    written ``1..K``, and the ids come back on import."""

    SHAPE = (6, 8, 8)

    @classmethod
    def _case(cls, path: Path, classes: dict[int, str]) -> Path:
        from medh5.labels import LabelClass, LabelSet

        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid("g", shape=cls.SHAPE, spacing=(2.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(cls.SHAPE, np.int16), grid="g", modality="CT")
            w.label_set(
                LabelSet(
                    "gapped",
                    version="1.0.0",
                    classes=[LabelClass(c, k, k.title()) for c, k in classes.items()],
                )
            )
            masks = {}
            for i, class_id in enumerate(sorted(classes)):
                mask = np.zeros(cls.SHAPE, bool)
                mask[i, 1:3, 1:3] = True
                masks[class_id] = mask
            w.add_segmentation("seg", grid="g", masks=masks)
        return path

    def test_F23_gapped_ids_are_written_consecutive_and_come_back(self, tmp_path: Path):
        from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

        path = self._case(tmp_path / "case1.medh5", {1: "organ", 3: "lesion"})
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        root = tmp_path / "nn" / "D1"
        document = nnunet_checks(root)
        assert document["labels"] == {"background": 0, "organ": 1, "lesion": 2}
        assert document["medh5_class_ids"] == {"2": 3}
        assert report.of_kind("label_values")
        volume = np.asarray(nib.load(str(root / "labelsTr/case1.nii.gz")).dataobj)
        assert int((volume == 2).sum()) == 4 and not (volume == 3).any()
        back = from_nnunetv2(root, tmp_path / "back")
        assert back.ok
        with medh5.open(tmp_path / "back" / "case1.medh5") as s, medh5.open(path) as o:
            ann, original = s.annotations["seg"], o.annotations["seg"]
            assert sorted(c.id for c in s.label_set) == [1, 3]
            for class_id in (1, 3):
                assert np.array_equal(ann.dense([class_id]), original.dense([class_id]))
        # Exported again, from the stash: the same dataset.
        to_nnunetv2(
            [tmp_path / "back" / "case1.medh5"], tmp_path / "again", dataset_name="D1"
        )
        assert nnunet_checks(tmp_path / "again" / "D1") == document

    def test_F23_a_subset_is_renumbered_too(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        path = self._case(tmp_path / "case1.medh5", {1: "organ", 3: "lesion"})
        to_nnunetv2([path], tmp_path / "nn", dataset_name="D1", classes=["lesion"])
        document = nnunet_checks(tmp_path / "nn" / "D1")
        assert document["labels"] == {"background": 0, "lesion": 1}
        assert document["medh5_class_ids"] == {"1": 3}

    def test_F23_consecutive_ids_record_nothing(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        path = self._case(tmp_path / "case1.medh5", {1: "organ", 2: "lesion"})
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        document = nnunet_checks(tmp_path / "nn" / "D1")
        assert "medh5_class_ids" not in document and not report.of_kind("label_values")


class TestF04StashedLabels:
    """F04 of the round-4 audit: a reused ``dataset.json`` that did not name a
    class the cases carried wrote it as background in every case, and left it
    out of the labels.  A scalar stash is extended; a region-based one is
    refused; and every exported class is read back."""

    SHAPE = (6, 8, 8)

    @classmethod
    def _case(cls, path: Path, stash: dict) -> Path:
        from medh5.labels import LabelClass, LabelSet

        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid("g", shape=cls.SHAPE, spacing=(2.0, 1.0, 1.0))
            w.add_image("c1", np.zeros(cls.SHAPE, np.int16), grid="g", modality="CT")
            w.label_set(
                LabelSet(
                    "stashed",
                    version="1.0.0",
                    classes=[LabelClass(1, "c1", "C1"), LabelClass(2, "c2", "C2")],
                )
            )
            first = np.zeros(cls.SHAPE, bool)
            first[1:3, 1:3, 1:3] = True
            second = np.zeros(cls.SHAPE, bool)
            second[4:5, 5:7, 5:7] = True
            w.add_segmentation("seg", grid="g", masks={1: first, 2: second})
            w.extra("nnunetv2", stash)
        return path

    def test_F04_a_scalar_stash_is_extended(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        stash = {
            "channel_names": {"0": "c1"},
            "labels": {"background": 0, "c1": 1},
            "numTraining": 1,
            "file_ending": ".nii.gz",
            "description": "kept",
        }
        path = self._case(tmp_path / "case1.medh5", stash)
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        root = tmp_path / "nn" / "D1"
        document = nnunet_checks(root)
        assert document["labels"] == {"background": 0, "c1": 1, "c2": 2}
        assert document["description"] == "kept"
        assert report.of_kind("labels")
        volume = np.asarray(nib.load(str(root / "labelsTr/case1.nii.gz")).dataobj)
        assert int((volume == 2).sum()) == 4

    def test_F04_a_region_stash_is_refused(self, tmp_path: Path):
        from medh5.io.nnunetv2 import to_nnunetv2

        stash = {
            "channel_names": {"0": "c1"},
            "labels": {"background": 0, "c1": 1, "whole": [1]},
            "regions_class_order": [1, 1],
            "numTraining": 1,
            "file_ending": ".nii.gz",
        }
        path = self._case(tmp_path / "case1.medh5", stash)
        with pytest.raises(MEDH5ValidationError, match="region-based"):
            to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        assert not (tmp_path / "nn").exists()

    def test_F04_a_class_no_label_reads_back_is_lost_and_refused(self):
        """The backstop, on its own: the read-back covers every exported
        class, not only the labels'."""
        from medh5.io.nnunetv2 import _not_given_back

        class Ann:
            class_ids = (1, 2)

            def dense(self, ids):
                out = np.zeros((1, 4), bool)
                out[0, ids[0] - 1] = True
                return out

        volume = np.array([1, 0, 0, 0])
        ignored = np.zeros(4, bool)
        lost = _not_given_back(
            Ann(), volume, {"c1": 1}, ignored, {"c1": 1}, frozenset({1, 2})
        )
        assert lost == ["class 2 (1 voxel(s) no label names)"]
