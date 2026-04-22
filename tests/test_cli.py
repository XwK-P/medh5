"""Tests for the CLI."""

import numpy as np
import pytest

from medh5 import MEDH5File
from medh5.cli import main


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "cli_test.medh5"
    images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
    MEDH5File.write(
        path,
        images=images,
        label=1,
        label_name="positive",
        spacing=[1.0, 1.0, 2.0],
        extra={"modality": "CT"},
        seg={"tumor": np.zeros((8, 8, 8), dtype=bool)},
    )
    return path


class TestCLI:
    def test_info(self, sample_file, capsys):
        ret = main(["info", str(sample_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "CT" in captured.out
        assert "float32" in captured.out
        assert "Spacing" in captured.out

    def test_validate_valid(self, sample_file, capsys):
        ret = main(["validate", str(sample_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "VALID" in captured.out

    def test_validate_invalid(self, tmp_path, capsys):
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        ret = main(["validate", str(bad)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "INVALID" in captured.err or len(captured.err) > 0

    def test_no_command(self, capsys):
        ret = main([])
        assert ret == 2
        err = capsys.readouterr().err
        assert "usage:" in err.lower()

    def test_info_missing_file(self, tmp_path, capsys):
        ret = main(["info", str(tmp_path / "missing.medh5")])
        assert ret == 1

    def test_import_without_subcommand(self, capsys):
        ret = main(["import"])
        assert ret == 2
        err = capsys.readouterr().err
        assert "missing subcommand" in err

    def test_info_json(self, sample_file, capsys):
        ret = main(["info", str(sample_file), "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        assert '"shape"' in captured.out
        assert '"review_status"' in captured.out

    def test_validate_json(self, sample_file, capsys):
        ret = main(["validate", str(sample_file), "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        assert '"errors"' in captured.out

    def test_validate_strict_requires_no_warnings(self, tmp_path, capsys):
        path = tmp_path / "strict.medh5"
        MEDH5File.write(path, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})
        ret = main(["validate", str(path), "--strict"])
        assert ret == 1


def _make_files(tmp_path, n=3, label_cycle=(0, 1)):
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i:02d}.medh5"
        MEDH5File.write(
            p,
            images={"CT": np.zeros((2, 4, 4), dtype=np.float32) + i},
            seg={"tumor": np.zeros((2, 4, 4), dtype=bool)},
            label=label_cycle[i % len(label_cycle)],
            extra={"patient_id": f"P{i // 2:02d}"},
            checksum=True,
        )
        paths.append(p)
    return paths


class TestBatchCLI:
    def test_validate_all_clean(self, tmp_path, capsys):
        _make_files(tmp_path)
        ret = main(["validate-all", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Invalid: 0" in out

    def test_validate_all_with_corrupt(self, tmp_path, capsys):
        _make_files(tmp_path, n=2)
        (tmp_path / "bad.medh5").write_bytes(b"not hdf5")
        ret = main(["validate-all", str(tmp_path)])
        assert ret == 1

    def test_audit(self, tmp_path, capsys):
        _make_files(tmp_path)
        ret = main(["audit", str(tmp_path)])
        assert ret == 0
        assert "Failed: 0" in capsys.readouterr().out

    def test_recompress_in_place(self, tmp_path, capsys):
        paths = _make_files(tmp_path, n=2)
        ret = main(["recompress", str(tmp_path), "--compression", "max", "--checksum"])
        assert ret == 0
        for p in paths:
            assert MEDH5File.verify(p)

    def test_recompress_out_dir(self, tmp_path, capsys):
        paths = _make_files(tmp_path, n=2)
        out_dir = tmp_path / "out"
        ret = main(
            [
                "recompress",
                str(paths[0]),
                "--compression",
                "fast",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert ret == 0
        assert (out_dir / paths[0].name).exists()


class TestIndexSplitStatsCLI:
    def test_index_and_split(self, tmp_path, capsys):
        _make_files(tmp_path, n=4)
        manifest = tmp_path / "m.json"
        ret = main(["index", str(tmp_path), "-o", str(manifest)])
        assert ret == 0
        assert manifest.exists()

        out_dir = tmp_path / "splits"
        ret = main(
            [
                "split",
                str(manifest),
                "--ratios",
                "0.5,0.5",
                "--stratify",
                "label",
                "-o",
                str(out_dir),
            ]
        )
        assert ret == 0
        assert (out_dir / "train.json").exists()
        assert (out_dir / "val.json").exists()

    def test_split_kfold(self, tmp_path):
        _make_files(tmp_path, n=4)
        manifest = tmp_path / "m.json"
        main(["index", str(tmp_path), "-o", str(manifest)])
        out_dir = tmp_path / "folds"
        ret = main(["split", str(manifest), "--k-folds", "2", "-o", str(out_dir)])
        assert ret == 0
        assert (out_dir / "fold0_train.json").exists()
        assert (out_dir / "fold1_val.json").exists()

    def test_stats_to_json(self, tmp_path, capsys):
        _make_files(tmp_path, n=2)
        out = tmp_path / "stats.json"
        ret = main(["stats", str(tmp_path), "-o", str(out)])
        assert ret == 0
        assert out.exists()
        import json

        payload = json.loads(out.read_text())
        assert "modalities" in payload
        assert "CT" in payload["modalities"]


class TestImportExportCLI:
    def test_import_export_nifti_roundtrip(self, tmp_path, capsys):
        nib = pytest.importorskip("nibabel")
        ct = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        nii = tmp_path / "ct.nii.gz"
        nib.save(nib.Nifti1Image(ct, np.eye(4)), str(nii))

        out_medh5 = tmp_path / "ct.medh5"
        ret = main(
            [
                "import",
                "nifti",
                "--image",
                "CT",
                str(nii),
                "-o",
                str(out_medh5),
            ]
        )
        assert ret == 0
        assert out_medh5.exists()

        export_dir = tmp_path / "export"
        ret = main(["export", "nifti", str(out_medh5), "-o", str(export_dir)])
        assert ret == 0
        assert (export_dir / "image_CT.nii.gz").exists()

    def test_import_export_nnunetv2_roundtrip(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        import json as _json

        src = tmp_path / "Dataset099_Cli"
        (src / "imagesTr").mkdir(parents=True)
        (src / "labelsTr").mkdir()
        aff = np.diag([1.0, 1.0, 1.0, 1.0])
        for cid in ("c1", "c2"):
            for idx in (0, 1):
                arr = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) + idx
                nib.save(
                    nib.Nifti1Image(arr, aff),
                    str(src / "imagesTr" / f"{cid}_{idx:04d}.nii.gz"),
                )
            lab = np.zeros((4, 4, 4), dtype=np.uint8)
            lab[1, 1, 1] = 1
            lab[2, 2, 2] = 2
            nib.save(nib.Nifti1Image(lab, aff), str(src / "labelsTr" / f"{cid}.nii.gz"))
        (src / "dataset.json").write_text(
            _json.dumps(
                {
                    "channel_names": {"0": "T1", "1": "T2"},
                    "labels": {"background": 0, "lesion": 1, "edema": 2},
                    "numTraining": 2,
                    "file_ending": ".nii.gz",
                }
            )
        )

        medh5_out = tmp_path / "medh5_out"
        ret = main(["import", "nnunetv2", str(src), "-o", str(medh5_out)])
        assert ret == 0
        assert (medh5_out / "imagesTr" / "c1.medh5").is_file()
        assert (medh5_out / "imagesTr" / "c2.medh5").is_file()

        sample = MEDH5File.read(medh5_out / "imagesTr" / "c1.medh5")
        assert set(sample.images.keys()) == {"T1", "T2"}
        assert sample.seg is not None
        assert sample.seg["lesion"][1, 1, 1]
        assert sample.seg["edema"][2, 2, 2]

        exported = tmp_path / "exported"
        ret = main(["export", "nnunetv2", str(medh5_out), "-o", str(exported)])
        assert ret == 0
        assert (exported / "dataset.json").is_file()
        payload = _json.loads((exported / "dataset.json").read_text())
        assert payload["channel_names"] == {"0": "T1", "1": "T2"}
        assert payload["labels"] == {"background": 0, "lesion": 1, "edema": 2}
        assert payload["numTraining"] == 2

        back = nib.load(str(exported / "labelsTr" / "c1.nii.gz"))
        back_arr = np.asarray(back.dataobj)
        assert back_arr[1, 1, 1] == 1
        assert back_arr[2, 2, 2] == 2

    def test_import_dicom(self, tmp_path, capsys):
        pytest.importorskip("pydicom")
        from test_io import _make_dicom_series

        dicom_dir = tmp_path / "series"
        _make_dicom_series(dicom_dir, n_slices=3)
        out_medh5 = tmp_path / "ct.medh5"
        ret = main(
            [
                "import",
                "dicom",
                str(dicom_dir),
                "-o",
                str(out_medh5),
                "--modality",
                "CT",
                "--checksum",
            ]
        )
        assert ret == 0
        sample = MEDH5File.read(out_medh5)
        assert sample.images["CT"].shape[0] == 3
        assert MEDH5File.verify(out_medh5)

    def test_import_nifti_with_resampling(self, tmp_path):
        pytest.importorskip("SimpleITK")
        nib = pytest.importorskip("nibabel")
        ct = np.ones((8, 8, 8), dtype=np.float32)
        pet = np.ones((4, 4, 4), dtype=np.float32) * 3.0
        ct_path = tmp_path / "ct.nii.gz"
        pet_path = tmp_path / "pet.nii.gz"
        aff_ct = np.eye(4)
        aff_pet = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(ct, aff_ct), str(ct_path))
        nib.save(nib.Nifti1Image(pet, aff_pet), str(pet_path))

        out_medh5 = tmp_path / "resampled.medh5"
        ret = main(
            [
                "import",
                "nifti",
                "--image",
                "CT",
                str(ct_path),
                "--image",
                "PET",
                str(pet_path),
                "--resample-to",
                "CT",
                "-o",
                str(out_medh5),
            ]
        )
        assert ret == 0
        sample = MEDH5File.read(out_medh5)
        assert sample.images["PET"].shape == (8, 8, 8)


class TestReviewCLI:
    def test_review_set_get(self, tmp_path, capsys):
        p = tmp_path / "s.medh5"
        MEDH5File.write(p, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})
        ret = main(
            [
                "review",
                "set",
                str(p),
                "--status",
                "reviewed",
                "--annotator",
                "puyang",
            ]
        )
        assert ret == 0
        ret = main(["review", "get", str(p)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "reviewed" in out
        assert "puyang" in out

    def test_review_list_filter(self, tmp_path, capsys):
        for i in range(3):
            p = tmp_path / f"s{i}.medh5"
            MEDH5File.write(p, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})
            if i == 1:
                MEDH5File.set_review_status(p, status="reviewed", annotator="x")
        ret = main(["review", "list", str(tmp_path), "--status", "reviewed"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "s1.medh5" in out
        assert "s0.medh5" not in out

    def test_review_get_json(self, tmp_path, capsys):
        p = tmp_path / "s.medh5"
        MEDH5File.write(p, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})
        MEDH5File.set_review_status(p, status="reviewed", annotator="x")
        ret = main(["review", "get", str(p), "--json"])
        assert ret == 0
        out = capsys.readouterr().out
        assert '"status": "reviewed"' in out

    def test_review_import_seg(self, tmp_path, capsys):
        nib = pytest.importorskip("nibabel")
        p = tmp_path / "s.medh5"
        MEDH5File.write(p, images={"CT": np.zeros((2, 4, 4), dtype=np.float32)})

        mask = np.zeros((2, 4, 4), dtype=np.uint8)
        mask[1, 2, 2] = 1
        nii = tmp_path / "tumor.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(nii))

        ret = main(
            [
                "review",
                "import-seg",
                str(p),
                "--name",
                "tumor",
                "--from",
                str(nii),
            ]
        )
        assert ret == 0
        sample = MEDH5File.read(p)
        assert sample.seg is not None
        assert sample.seg["tumor"].sum() == 1

    def test_review_import_seg_with_resample(self, tmp_path):
        pytest.importorskip("SimpleITK")
        nib = pytest.importorskip("nibabel")
        p = tmp_path / "s.medh5"
        MEDH5File.write(
            p,
            images={"CT": np.zeros((8, 8, 8), dtype=np.float32)},
            spacing=[1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        nii = tmp_path / "tumor_resampled.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0])), str(nii))

        ret = main(
            [
                "review",
                "import-seg",
                str(p),
                "--name",
                "tumor",
                "--from",
                str(nii),
                "--resample",
            ]
        )
        assert ret == 0
        sample = MEDH5File.read(p)
        assert sample.seg is not None
        assert sample.seg["tumor"].shape == (8, 8, 8)
        assert sample.seg["tumor"].any()
