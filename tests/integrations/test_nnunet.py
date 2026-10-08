"""nnU-Net v2 datasets, both ways (``medh5.io.nnunetv2``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from tests.kits import Organs

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
        to_nnunetv2([path], tmp_path / "nn", dataset_name="D1")
        image = nib.load(str(tmp_path / "nn/D1/imagesTr/ct_0000.nii.gz"))
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
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D3")
        for output in report.outputs:
            assert Path(output).exists(), output
        assert not any("labelsTr" in o for o in report.outputs)
