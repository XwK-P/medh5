"""Write a .medh5 file with every field populated, read it back, verify."""

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5SchemaError, MEDH5ValidationError


@pytest.fixture
def tmp_path_medh5(tmp_path):
    return tmp_path / "sample.medh5"


def _make_sample():
    rng = np.random.default_rng(42)
    images = {
        "CT": rng.random((32, 64, 64), dtype=np.float32),
        "PET": rng.random((32, 64, 64), dtype=np.float32),
    }
    seg = {
        "tumor": rng.random((32, 64, 64)) > 0.8,
        "liver": rng.random((32, 64, 64)) > 0.5,
    }
    bboxes = np.array(
        [
            [[2, 10], [5, 20], [5, 20]],
            [[12, 28], [40, 55], [30, 50]],
        ],
        dtype=np.float64,
    )
    bbox_scores = np.array([0.95, 0.73], dtype=np.float64)
    bbox_labels = ["tumor", "cyst"]
    return images, seg, bboxes, bbox_scores, bbox_labels


class TestFullRoundtrip:
    def test_all_fields(self, tmp_path_medh5):
        images, seg, bboxes, bbox_scores, bbox_labels = _make_sample()
        MEDH5File.write(
            tmp_path_medh5,
            images=images,
            seg=seg,
            bboxes=bboxes,
            bbox_scores=bbox_scores,
            bbox_labels=bbox_labels,
            label=2,
            label_name="malignant",
            spacing=[1.0, 0.5, 0.5],
            origin=[0.0, 0.0, 0.0],
            direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            axis_labels=["spatial_z", "spatial_y", "spatial_x"],
            coord_system="RAS",
            patch_size=64,
            extra={"modality": "multi", "patient_id": "P001"},
        )

        sample = MEDH5File.read(tmp_path_medh5)

        assert set(sample.images.keys()) == {"CT", "PET"}
        for name in images:
            np.testing.assert_array_equal(sample.images[name], images[name])
        assert set(sample.seg.keys()) == set(seg.keys())
        for name in seg:
            np.testing.assert_array_equal(sample.seg[name], seg[name])
            assert sample.seg[name].dtype == bool
        np.testing.assert_array_almost_equal(sample.bboxes, bboxes)
        np.testing.assert_array_almost_equal(sample.bbox_scores, bbox_scores)
        assert sample.bbox_labels == bbox_labels

        m = sample.meta
        assert m.label == 2
        assert m.label_name == "malignant"
        assert m.image_names == ["CT", "PET"]
        assert m.has_seg is True
        assert m.seg_names == ["liver", "tumor"]
        assert m.has_bbox is True
        assert m.spatial.spacing == [1.0, 0.5, 0.5]
        assert m.spatial.origin == [0.0, 0.0, 0.0]
        assert m.spatial.direction == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert m.spatial.axis_labels == ["spatial_z", "spatial_y", "spatial_x"]
        assert m.spatial.coord_system == "RAS"
        assert m.extra == {"modality": "multi", "patient_id": "P001"}
        assert m.schema_version == "1"

    def test_single_image(self, tmp_path_medh5):
        rng = np.random.default_rng(0)
        images = {"CT": rng.random((16, 32, 32), dtype=np.float32)}
        MEDH5File.write(tmp_path_medh5, images=images)

        sample = MEDH5File.read(tmp_path_medh5)
        np.testing.assert_array_equal(sample.images["CT"], images["CT"])
        assert sample.seg is None
        assert sample.bboxes is None
        assert sample.bbox_scores is None
        assert sample.bbox_labels is None
        assert sample.meta.label is None
        assert sample.meta.image_names == ["CT"]
        assert sample.meta.has_seg is False
        assert sample.meta.seg_names is None
        assert sample.meta.has_bbox is False

    def test_string_label(self, tmp_path_medh5):
        images = {"MRI": np.zeros((8, 8, 8), dtype=np.float32)}
        MEDH5File.write(tmp_path_medh5, images=images, label="benign")

        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.meta.label == "benign"

    def test_meta_only_read(self, tmp_path_medh5):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        MEDH5File.write(
            tmp_path_medh5,
            images=images,
            label=5,
            spacing=[1.0, 1.0, 2.0],
            extra={"key": "value"},
        )

        meta = MEDH5File.read_meta(tmp_path_medh5)
        assert meta.label == 5
        assert meta.spatial.spacing == [1.0, 1.0, 2.0]
        assert meta.extra == {"key": "value"}
        assert meta.image_names == ["CT"]

    def test_2d_images(self, tmp_path_medh5):
        rng = np.random.default_rng(7)
        images = {
            "xray": rng.random((256, 256), dtype=np.float32),
            "mask_overlay": rng.random((256, 256), dtype=np.float32),
        }
        MEDH5File.write(
            tmp_path_medh5,
            images=images,
            spacing=[0.5, 0.5],
            patch_size=128,
        )
        sample = MEDH5File.read(tmp_path_medh5)
        assert set(sample.images.keys()) == {"xray", "mask_overlay"}
        for name in images:
            np.testing.assert_array_equal(sample.images[name], images[name])
        assert sample.meta.spatial.spacing == [0.5, 0.5]

    def test_multi_dtype_float64(self, tmp_path_medh5):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float64)}
        MEDH5File.write(tmp_path_medh5, images=images)
        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.images["CT"].dtype == np.float64

    def test_multi_dtype_int16(self, tmp_path_medh5):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.int16)}
        MEDH5File.write(tmp_path_medh5, images=images)
        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.images["CT"].dtype == np.int16

    def test_multi_dtype_uint8(self, tmp_path_medh5):
        images = {"photo": np.zeros((64, 64), dtype=np.uint8)}
        MEDH5File.write(tmp_path_medh5, images=images)
        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.images["photo"].dtype == np.uint8

    def test_nested_extra_json(self, tmp_path_medh5):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        extra = {
            "patient": {"id": "P001", "age": 42},
            "tags": ["urgent", "follow-up"],
            "scores": [0.1, 0.9],
        }
        MEDH5File.write(tmp_path_medh5, images=images, extra=extra)
        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.meta.extra == extra

    def test_schema_version_future_raises(self, tmp_path_medh5):
        import h5py

        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        MEDH5File.write(tmp_path_medh5, images=images)

        with h5py.File(str(tmp_path_medh5), "a") as f:
            f.attrs["schema_version"] = "10"

        with pytest.raises(MEDH5SchemaError, match="schema version"):
            MEDH5File.read(tmp_path_medh5)


class TestValidation:
    def test_bad_extension_raises(self, tmp_path):
        images = {"CT": np.zeros((4, 4), dtype=np.float32)}
        with pytest.raises(MEDH5ValidationError, match="extension"):
            MEDH5File.write(tmp_path / "bad.hdf5", images=images)
        with pytest.raises(MEDH5ValidationError, match="extension"):
            MEDH5File.read(tmp_path / "bad.hdf5")

    def test_empty_images_raises(self, tmp_path):
        with pytest.raises(MEDH5ValidationError, match="at least one"):
            MEDH5File.write(tmp_path / "x.medh5", images={})

    def test_shape_mismatch_raises(self, tmp_path):
        images = {
            "CT": np.zeros((8, 8, 8), dtype=np.float32),
            "PET": np.zeros((8, 8, 16), dtype=np.float32),
        }
        with pytest.raises(MEDH5ValidationError, match="same shape"):
            MEDH5File.write(tmp_path / "x.medh5", images=images)

    def test_seg_shape_mismatch_raises(self, tmp_path):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        seg = {"tumor": np.zeros((8, 8, 16), dtype=bool)}
        with pytest.raises(MEDH5ValidationError, match="Segmentation mask"):
            MEDH5File.write(tmp_path / "x.medh5", images=images, seg=seg)

    def test_bbox_scores_count_mismatch_raises(self, tmp_path):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        bboxes = np.zeros((2, 3, 2))
        bbox_scores = np.array([0.5])
        with pytest.raises(MEDH5ValidationError, match="bbox_scores length"):
            MEDH5File.write(
                tmp_path / "x.medh5",
                images=images,
                bboxes=bboxes,
                bbox_scores=bbox_scores,
            )

    def test_bbox_labels_count_mismatch_raises(self, tmp_path):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        bboxes = np.zeros((2, 3, 2))
        bbox_labels = ["a"]
        with pytest.raises(MEDH5ValidationError, match="bbox_labels length"):
            MEDH5File.write(
                tmp_path / "x.medh5",
                images=images,
                bboxes=bboxes,
                bbox_labels=bbox_labels,
            )

    def test_bad_clevel_raises(self, tmp_path):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        with pytest.raises(MEDH5ValidationError, match="clevel"):
            MEDH5File.write(tmp_path / "x.medh5", images=images, clevel=15)

    def test_bad_bbox_shape_raises(self, tmp_path):
        images = {"CT": np.zeros((8, 8, 8), dtype=np.float32)}
        bboxes = np.zeros((2, 4, 2))
        with pytest.raises(MEDH5ValidationError, match="bboxes must have shape"):
            MEDH5File.write(tmp_path / "x.medh5", images=images, bboxes=bboxes)
