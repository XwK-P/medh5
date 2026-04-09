"""Write a .medh5 file with every field populated, read it back, verify."""

import numpy as np
import pytest

from medh5 import MEDH5File, MEDH5Sample


@pytest.fixture
def tmp_path_medh5(tmp_path):
    return tmp_path / "sample.medh5"


def _make_sample():
    rng = np.random.default_rng(42)
    image = rng.random((32, 64, 64), dtype=np.float32)
    seg = rng.integers(0, 4, size=(32, 64, 64), dtype=np.uint8)
    bboxes = np.array([
        [[2, 10], [5, 20], [5, 20]],
        [[12, 28], [40, 55], [30, 50]],
    ], dtype=np.float64)
    bbox_scores = np.array([0.95, 0.73], dtype=np.float64)
    bbox_labels = ["tumor", "cyst"]
    return image, seg, bboxes, bbox_scores, bbox_labels


class TestFullRoundtrip:
    def test_all_fields(self, tmp_path_medh5):
        image, seg, bboxes, bbox_scores, bbox_labels = _make_sample()
        MEDH5File.write(
            tmp_path_medh5,
            image=image,
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
            extra={"modality": "CT", "patient_id": "P001"},
        )

        sample = MEDH5File.read(tmp_path_medh5)

        np.testing.assert_array_equal(sample.image, image)
        np.testing.assert_array_equal(sample.seg, seg)
        np.testing.assert_array_almost_equal(sample.bboxes, bboxes)
        np.testing.assert_array_almost_equal(sample.bbox_scores, bbox_scores)
        assert sample.bbox_labels == bbox_labels

        m = sample.meta
        assert m.label == 2
        assert m.label_name == "malignant"
        assert m.has_seg is True
        assert m.has_bbox is True
        assert m.spatial.spacing == [1.0, 0.5, 0.5]
        assert m.spatial.origin == [0.0, 0.0, 0.0]
        assert m.spatial.direction == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert m.spatial.axis_labels == ["spatial_z", "spatial_y", "spatial_x"]
        assert m.spatial.coord_system == "RAS"
        assert m.extra == {"modality": "CT", "patient_id": "P001"}
        assert m.schema_version == "1"

    def test_image_only(self, tmp_path_medh5):
        rng = np.random.default_rng(0)
        image = rng.random((16, 32, 32), dtype=np.float32)
        MEDH5File.write(tmp_path_medh5, image=image)

        sample = MEDH5File.read(tmp_path_medh5)
        np.testing.assert_array_equal(sample.image, image)
        assert sample.seg is None
        assert sample.bboxes is None
        assert sample.bbox_scores is None
        assert sample.bbox_labels is None
        assert sample.meta.label is None
        assert sample.meta.has_seg is False
        assert sample.meta.has_bbox is False

    def test_string_label(self, tmp_path_medh5):
        image = np.zeros((8, 8, 8), dtype=np.float32)
        MEDH5File.write(tmp_path_medh5, image=image, label="benign")

        sample = MEDH5File.read(tmp_path_medh5)
        assert sample.meta.label == "benign"

    def test_meta_only_read(self, tmp_path_medh5):
        image = np.zeros((8, 8, 8), dtype=np.float32)
        MEDH5File.write(
            tmp_path_medh5,
            image=image,
            label=5,
            spacing=[1.0, 1.0, 2.0],
            extra={"key": "value"},
        )

        meta = MEDH5File.read_meta(tmp_path_medh5)
        assert meta.label == 5
        assert meta.spatial.spacing == [1.0, 1.0, 2.0]
        assert meta.extra == {"key": "value"}

    def test_bad_extension_raises(self, tmp_path):
        image = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="extension"):
            MEDH5File.write(tmp_path / "bad.hdf5", image=image)
        with pytest.raises(ValueError, match="extension"):
            MEDH5File.read(tmp_path / "bad.hdf5")
