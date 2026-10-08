"""The MONAI bridge (``medh5.monai``): what MONAI receives, and the geometry and
metadata computed without it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from tests.helpers import SHAPE
from tests.kits import Hierarchy


class TestMonaiF04LabelDtype:
    """`to_dict` labels are `int64`: the full id range, and the ignore id intact."""

    def test_S5_3_ids_beyond_int16_and_the_ignore_id_survive(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("monai")
        from medh5.monai import to_dict

        wide = LabelClass(40000, "wide", "A class beyond int16")
        path = tmp_path / "wide.medh5"
        ignore = np.zeros(Hierarchy.SHAPE, dtype=bool)
        ignore[-1] = True
        with medh5.create(path, sample_id="wide") as w:
            w.label_set(Hierarchy.label_set(wide))
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_segmentation(
                "organs",
                grid="g",
                masks={40000: Hierarchy.mask()},
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


class TestMonai:
    def test_available_reports_whether_monai_imports(self):
        """Both branches are real; neither may be excluded from coverage.

        This carried a blanket `pragma: no cover`, including on the `return
        True` that runs wherever the extra is installed -- the same shape as
        the untested-because-skipped problem the MONAI CI job was added for.
        """
        from medh5.monai import available

        pytest.importorskip("monai")
        assert available() is True

    def test_an_annotation_carries_its_own_grids_affine_not_an_images(self, tmp_path):
        """`to_dict` binds each label tensor to the grid it was drawn on.

        A sample holding CT and PET on different grids is the case this format
        exists for.  Taking the affine from the first *image* handed the
        annotation geometry belonging to something else --- here a 70 mm origin
        shift and a 2x spacing error --- so `Spacingd` resampled the label
        against the wrong frame and the ground truth moved.
        """
        pytest.importorskip("monai")
        from medh5.monai import affine_for, to_dict

        ct = np.zeros((16, 32, 32), np.int16)
        pet = np.zeros((8, 16, 16), np.int16)
        seg = np.zeros((8, 16, 16), bool)
        seg[2:6, 4:12, 4:12] = True
        path = tmp_path / "twogrid.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(
                LabelSet(
                    "v", version="1.0.0", classes=[LabelClass(1, "tumour", "Tumour")]
                )
            )
            w.add_timepoint("tp0")
            w.add_grid(
                "gct",
                shape=ct.shape,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp0",
            )
            w.add_grid(
                "gpet",
                shape=pet.shape,
                spacing=(2.0, 2.0, 2.0),
                origin=(50.0, 60.0, 70.0),
                timepoint="tp0",
            )
            w.add_image("CT", ct, grid="gct", modality="CT")
            w.add_image("PET", pet, grid="gpet", modality="PT")
            w.add_segmentation(
                "tumour",
                grid="gpet",
                masks={"tumour": seg},
                annotated_classes=["tumour"],
            )

        with medh5.open(path) as sample:
            item = to_dict(sample, images=["CT", "PET"], annotations=["tumour"])
            got = np.asarray(item["tumour"].affine)
            assert np.allclose(got, affine_for(sample, "PET"))
            assert not np.allclose(got, affine_for(sample, "CT"))
            assert item["tumour"].meta["medh5"]["grid_id"] == "gpet"
            # No image requested must not mean an invented identity affine.
            only = to_dict(sample, images=[], annotations=["tumour"])
            assert np.allclose(
                np.asarray(only["tumour"].affine), affine_for(sample, "PET")
            )

    def test_the_affine_is_the_grid_affine(self, indexed_cohort):
        from medh5.monai import affine_for, meta_dict

        with medh5.open(indexed_cohort[0]) as sample:
            affine = affine_for(sample, "CT_tp0")
            assert np.allclose(affine, sample.grids["ct_tp0"].affine)
            meta = meta_dict(sample, "CT_tp0")
            assert meta["space"] == "LPS"
            assert meta["medh5"]["modality"] == "CT"
            assert list(meta["spatial_shape"]) == list(SHAPE)

    def test_S3_1_LPS_to_RAS_flips_only_the_first_two_world_axes(self, indexed_cohort):
        from medh5.monai import affine_for, convert_affine

        with medh5.open(indexed_cohort[0]) as sample:
            lps = affine_for(sample, "CT_tp0")
            ras = affine_for(sample, "CT_tp0", space="RAS")
        assert np.allclose(ras[:2], -lps[:2])
        assert np.allclose(ras[2:], lps[2:])
        assert np.allclose(convert_affine(lps, source="LPS", target="LPS"), lps)

    def test_a_conversion_it_cannot_justify_is_refused(self):
        from medh5.monai import convert_affine

        with pytest.raises(MEDH5ValidationError, match="3-D"):
            convert_affine(np.eye(3), source="LPS", target="RAS")
        with pytest.raises(MEDH5ValidationError, match="world convention"):
            convert_affine(np.eye(4), source="LPS", target="talairach")

    def test_an_roi_moves_the_origin(self, indexed_cohort):
        from medh5.monai import _shift_origin, affine_for

        with medh5.open(indexed_cohort[0]) as sample:
            grid = sample.grids["ct_tp0"]
            affine = affine_for(sample, "CT_tp0")
            roi = (slice(2, 6), slice(4, 8), slice(0, 4))
            shifted = _shift_origin(affine, roi)
        assert np.allclose(shifted[:3, 3], grid.index_to_world([[2, 4, 0]])[0])
        assert np.allclose(shifted[:3, :3], affine[:3, :3])

    def test_S4_3_a_MetaTensor_reads_the_level_its_affine_describes(self, tmp_path):
        """Level-0 voxels under a level-1 affine misplace every saved prediction.

        `meta_dict` already selects the requested level's grid and affine, so the
        array has to come from that level too; nothing about the MetaTensor
        would say the two disagree.
        """
        pytest.importorskip("monai")
        from medh5.geometry import derive_level_grid
        from medh5.monai import to_metatensor

        shape = (16, 32, 32)
        path = tmp_path / "pyr.medh5"
        with medh5.create(path, codec="portable") as w:
            base = w.add_grid(
                "l0", shape=shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
            )
            level1 = derive_level_grid(base, (2, 2, 2), "l1")
            w.add_grid(
                "l1",
                shape=level1.shape,
                spacing=level1.spacing,
                origin=level1.origin,
                direction=level1.direction,
            )
            w.add_pyramid(
                "CT",
                [
                    np.zeros(shape, dtype=np.int16),
                    np.full((8, 16, 16), 7, dtype=np.int16),
                ],
                grid_levels=["l0", "l1"],
                modality="CT",
            )

        with medh5.open(path) as sample:
            tensor = to_metatensor(sample, "CT", level=1)
            assert tuple(tensor.shape) == (8, 16, 16), "the level-1 array, not level 0"
            assert float(np.asarray(tensor).flat[0]) == 7.0
            assert list(tensor.meta["spatial_shape"]) == [8, 16, 16]
            assert np.allclose(
                np.asarray(tensor.meta["affine"]),
                sample.images["CT"].level(1).grid.affine,
            )

    def test_medh5_metadata_is_json_safe(self, indexed_cohort):
        from medh5.monai import meta_dict

        with medh5.open(indexed_cohort[0]) as sample:
            json.dumps(meta_dict(sample, "CT_tp0")["medh5"])


class TestWithoutMonai:
    """The parts of the adapter that do not need MONAI installed."""

    def test_from_metatensor_recovers_the_geometry(self):
        from medh5.monai import from_metatensor

        class Duck:
            """Anything with `.meta` and array semantics --- MONAI's own shape."""

            def __init__(self, array, meta):
                self._array = array
                self.meta = meta

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._array, dtype=dtype)

        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [5.0, 6.0, 7.0]
        array, geometry = from_metatensor(
            Duck(np.zeros((4, 4, 4)), {"affine": affine, "space": "RAS"})
        )
        assert array.shape == (4, 4, 4)
        assert geometry["spacing"] == [2.0, 1.0, 1.0]
        assert geometry["origin"] == [5.0, 6.0, 7.0]
        assert geometry["coord_system"] == "RAS"

    def test_a_tensor_without_metadata_still_decomposes(self):
        from medh5.monai import from_metatensor

        array, geometry = from_metatensor(np.zeros((2, 2)))
        assert geometry["spacing"] == [1.0, 1.0]
        assert geometry["coord_system"] == "LPS"

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
