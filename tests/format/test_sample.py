"""Writing, reading, amending and the timepoint model (spec §2, §3.7, §14.4)."""

from __future__ import annotations

import dataclasses
import errno
import inspect
import json
import os
import pickle
import stat
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.cli import main
from medh5.curation import Timeline, Timepoint
from medh5.errors import (
    MEDH5Error,
    MEDH5FileError,
    MEDH5FileNotFoundError,
    MEDH5IsADirectoryError,
    MEDH5PermissionError,
    MEDH5ValidationError,
    MEDH5VersionError,
)
from medh5.validate import validate_file
from tests.helpers import (
    LATER_MINOR,
    SHAPE,
    as_str,
    encode_attr,
    str_dtype,
    write_sample,
)
from tests.kits import Flat, Framed, Hierarchy, Organs

# Windows keeps only a read-only bit, so a mode round trip cannot be asserted
# there. The Linux matrix holds these; the Windows job exists for the replace
# ordering in `TestAtomicRewrite`, which runs everywhere.
posix_modes = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits are not a Windows concept"
)


def minimal(writer, *, grid: str = "g") -> None:
    """The two calls every sample needs, so a test can add just its own point."""
    writer.add_grid(grid, shape=SHAPE, spacing=(1.0, 1.0, 1.0))
    writer.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid=grid, modality="CT")


def restamped_in(path: Path, algo: str) -> str:
    """Every digest of *path*, and its ``content_id``, restamped in *algo* as
    a writer digesting in it writes them; returns the ``content_id``."""
    from medh5.integrity import collect_digests, dataset_digest

    with medh5.open(path) as sample:
        fresh = {
            name: dataset_digest(sample.root[name], name, algo)
            for name in collect_digests(sample.root)
        }
    with h5py.File(path, "r+") as handle:
        handle.attrs["digest_algo"] = algo
        for name, digest in fresh.items():
            handle[name].attrs["digest"] = digest
    with medh5.open(path) as sample:
        content_id = sample.compute_content_id()
    with h5py.File(path, "r+") as handle:
        handle.attrs["content_id"] = content_id
    return content_id


class TestWriteRead:
    def test_round_trip(self, sample_path, masks):
        with medh5.open(sample_path) as sample:
            assert sample.identity.sample_id == "case"
            assert sample.identity.subject_id == "subj-A"
            # Imaging alone needs 1.0, and a writer writes the lowest version
            # its content needs (1.1 §2.3); FORMAT_VERSION is the newest.
            assert sample.version == "1.0"
            assert medh5.FORMAT_VERSION == "1.1"
            assert sample.kind == "sample"
            image = sample.images["CT_tp0"]
            assert image.shape == SHAPE
            assert image.modality == "CT"
            assert image.value_units == "HU"
            seg = sample.annotations["organs_tp0"]
            for class_id, mask in masks.items():
                assert np.array_equal(seg.dense([class_id])[0], mask)

    def test_S14_4_create_is_atomic(self, tmp_path):
        """§14.4: a failed create leaves no file and no temporary behind."""
        path = tmp_path / "boom.medh5"
        with pytest.raises(RuntimeError), medh5.create(path) as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            raise RuntimeError("boom")
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_commit_requires_an_image(self, tmp_path):
        """Refused at the end of a ``with`` block, a commit has no one left to
        mend it: the writer is aborted, and nothing is left behind."""
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "empty.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        assert exc.value.code == "E201"
        assert w.closed
        assert list(tmp_path.iterdir()) == []

    def test_S14_4_a_commit_refused_before_writing_leaves_the_writer_open(
        self, tmp_path
    ):
        """A refused commit aborted the writer, and the next ``commit()``
        returned ``None`` --- *already committed* --- for a file it never
        wrote.  Refused before it changed anything, the writer stays open, to
        mend and commit again."""
        path = tmp_path / "mended.medh5"
        writer = medh5.create(path, sample_id="mended")
        writer.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        with pytest.raises(MEDH5ValidationError) as exc:
            writer.commit()
        assert exc.value.code == "E201"
        assert not writer.closed
        assert not path.exists()

        writer.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        content_id = writer.commit()
        assert writer.closed
        assert content_id is not None
        assert writer.commit() is None, "committed: there is nothing more to do"
        with medh5.open(path) as sample:
            assert sample.content_id == content_id
        assert validate_file(path).ok

    def test_S14_4_an_aborted_writer_refuses_to_commit(self, sample_path):
        """Aborted by its caller, or by a commit refused once it had begun
        writing (E701 is met while stamping), a writer has nothing to commit
        and says so, and the target is as it was, with no temporary file."""
        writer = medh5.create(sample_path.parent / "aborted.medh5")
        minimal(writer)
        writer.abort()
        with pytest.raises(MEDH5ValidationError, match="aborted"):
            writer.commit()

        with h5py.File(sample_path, "r+") as handle:
            node = handle["images/CT_tp0"]
            data = node[...]
            data[0, 0, 0] += 7
            node[...] = data
        before = sample_path.read_bytes()
        writer = medh5.amend(sample_path)
        with pytest.raises(MEDH5ValidationError) as exc:
            writer.commit()
        assert exc.value.code == "E701"
        assert writer.closed
        with pytest.raises(MEDH5ValidationError, match="aborted"):
            writer.commit()
        assert sample_path.read_bytes() == before
        assert [p.name for p in sample_path.parent.iterdir()] == [sample_path.name]

    def test_commit_stamps_a_content_id(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert sample.content_id is not None

    def test_abort_discards(self, tmp_path):
        path = tmp_path / "aborted.medh5"
        writer = medh5.create(path)
        writer.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        writer.abort()
        assert not path.exists()

    def test_S14_4_a_writer_that_fails_to_open_leaves_no_temp_file(self, sample_path):
        """Nothing will ever call `abort()` for a writer that never returned.

        `amend` on a file whose `/meta` cannot be read raises out of
        `SampleWriter.__init__`, so the `with` statement that would have closed
        the ExitStack was never entered.  The sibling temp file and its open
        HDF5 handle then survived for as long as the traceback did --- which is
        as long as anything holds the exception, and in a notebook that is the
        rest of the session.
        """
        directory = sample_path.parent
        with h5py.File(sample_path, "r+") as handle:
            del handle["meta"]
            handle.create_dataset("meta", data=np.bytes_(b"{ not json"))
        before = sample_path.read_bytes()

        held: BaseException | None = None
        try:
            medh5.amend(sample_path)
        except MEDH5Error as exc:
            held = exc

        assert held is not None, "the unreadable document must still be reported"
        leftovers = [p.name for p in directory.iterdir() if p.name.startswith(".")]
        assert leftovers == [], "the temp file outlived the writer"
        assert sample_path.read_bytes() == before, "and the original is untouched"

    def test_S4_1_image_shape_must_match_its_grid(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros((2, 2, 2)), grid="g", modality="CT")
        assert exc.value.code == "E202"

    def test_unknown_grid_reference_is_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_image("CT", np.zeros(SHAPE), grid="nope", modality="CT")
        assert exc.value.code == "E101"

    def test_duplicate_ids_are_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError),
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))

    def test_S2_3_bad_identifiers_are_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("bad id", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        assert exc.value.code == "E003"

    def test_S4_2_physical_values_are_never_silent(self, tmp_path):
        path = tmp_path / "rescaled.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image(
                "CT",
                np.full(SHAPE, 10, dtype=np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=2.0,
                rescale_intercept=-5.0,
            )
        with medh5.open(path) as sample:
            image = sample.images["CT"]
            assert image.read()[0, 0, 0] == 10
            assert image.read(physical=True)[0, 0, 0] == 15.0
            assert image.is_rescaled

    def test_S4_2_an_image_without_a_rescale_is_read_as_stored(self, tmp_path):
        """``physical=True`` rescales into ``float32`` an image that declares a
        rescale; one that declares none already holds physical values, and is
        read in its own dtype --- the docstring promised ``float32``.  A
        ``dtype`` reads either as asked."""
        path = tmp_path / "plain.medh5"
        with medh5.create(path, codec="portable") as w:
            minimal(w)
            w.add_image(
                "PT",
                np.full(SHAPE, 10, np.int16),
                grid="g",
                modality="PT",
                rescale_slope=0.5,
            )
        with medh5.open(path) as sample:
            ct, pt = sample.images["CT"], sample.images["PT"]
            assert not ct.is_rescaled and pt.is_rescaled
            assert ct.read(physical=True).dtype == np.int16
            assert pt.read(physical=True).dtype == np.float32
            for image in (ct, pt):
                found = image.read(physical=True, dtype=np.float32)
                assert found.dtype == np.float32
            np.testing.assert_array_equal(
                ct.read(physical=True), ct.read(), strict=True
            )

    def test_S4_2_a_physical_read_without_a_rescale_is_the_stored_values_in_their_type(
        self, tmp_path
    ):
        """D27: a physical read is float32 only where a rescale applies."""
        path = tmp_path / "stored.medh5"
        values = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("MR", values, grid="g", modality="MR")
        with medh5.open(path) as sample:
            found = sample.images["MR"].read(physical=True)
        np.testing.assert_array_equal(found, values, strict=True)

    def test_S4_2_an_identity_rescale_declared_explicitly_reads_as_stored(
        self, tmp_path
    ):
        path = tmp_path / "identity.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                np.full(SHAPE, -7, np.int16),
                grid="g",
                modality="CT",
                rescale_slope=1.0,
                rescale_intercept=0.0,
            )
        with medh5.open(path) as sample:
            image = sample.images["CT"]
            assert not image.is_rescaled
            assert image.read(physical=True).dtype == np.int16

    def test_S4_2_dtype_gives_one_type_across_a_cohort(self, tmp_path):
        """Two files, one rescaled and one not: ``dtype`` reads both alike."""
        read = []
        for name, slope in (("a", None), ("b", 0.5)):
            path = tmp_path / f"{name}.medh5"
            with medh5.create(path, codec="portable") as w:
                w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
                extra = {} if slope is None else {"rescale_slope": slope}
                w.add_image(
                    "CT", np.full(SHAPE, 8, np.int16), grid="g", modality="CT", **extra
                )
            with medh5.open(path) as sample:
                read.append(sample.images["CT"].read(physical=True, dtype=np.float32))
        assert [r.dtype for r in read] == [np.float32, np.float32]
        assert (read[0][0, 0, 0], read[1][0, 0, 0]) == (8.0, 4.0)

    @pytest.mark.parametrize("odd", [None, np.nan, np.inf, 0.5])
    def test_S4_2_W907_advises_int16_only_where_it_holds_every_value(
        self, tmp_path, odd
    ):
        """``int16`` holds no NaN and no infinity.  Every *finite* value was
        tested, so a float image with NaN outside its field of view was
        advised into a type that cannot store it."""
        from medh5.image import lossless_as_int16

        values = np.full(SHAPE, -1000.0, np.float32)
        if odd is not None:
            values[0, 0, 0] = odd
        path = tmp_path / "float.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", values, grid="g", modality="CT")
        advised = "W907" in validate_file(path).codes
        assert advised == lossless_as_int16(values) == (odd is None)

    def test_roi_accepts_spatial_or_full_rank(self, sample_path):
        with medh5.open(sample_path) as sample:
            image = sample.images["CT_tp0"]
            assert image.read(np.s_[0:4, 0:4, 0:4]).shape == (4, 4, 4)
            with pytest.raises(MEDH5ValidationError):
                image.read([slice(0, 4)])

    def test_S14_5_an_ellipsis_fills_the_axes_it_stands_for(self, tmp_path):
        """`read((..., slice(0, 2)))` raised `TypeError` where the dataset's own
        `[..., 0:2]` worked."""
        with medh5.open(write_sample(tmp_path / "s.medh5")) as sample:
            image = sample.images["CT_tp0"]
            whole = image.read()
            for roi in ((..., slice(0, 2)), (slice(1, 3), ...), ...):
                np.testing.assert_array_equal(image.read(roi), whole[roi])
            assert image.read((slice(0, 2), ..., slice(0, 3))).shape == (
                2,
                SHAPE[1],
                3,
            )
        path = tmp_path / "4d.medh5"
        values = np.arange(2 * 4 * 5 * 6, dtype=np.int16).reshape(2, 4, 5, 6)
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=(2, 4, 5, 6),
                spacing=(1.0, 1.0, 1.0),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
            )
            w.add_image("MR", values, grid="g", modality="MR", channel_names=["a", "b"])
        with medh5.open(path) as sample:
            image = sample.images["MR"]
            assert image.read((..., slice(0, 2))).shape == (2, 4, 5, 2)
            np.testing.assert_array_equal(
                image.read((0, ..., slice(1, 3))), values[0, ..., 1:3]
            )

    def test_S14_5_two_ellipses_are_refused(self, sample_path):
        with medh5.open(sample_path) as sample:  # noqa: SIM117
            with pytest.raises(IndexError, match="single ellipsis"):
                sample.images["CT_tp0"].read((..., slice(0, 2), ...))

    @pytest.mark.parametrize("masked", [False, True])
    def test_S4_4_a_valid_region_resolves_its_window_as_an_annotation_does(
        self, tmp_path, masked
    ):
        """A negative start was taken as written: with no `valid_mask` the
        region came back larger than the grid, and with one it disagreed with
        the mask it was read from.  A step was ignored."""
        fov = np.zeros(SHAPE, dtype=bool)
        fov[-2:, 1:5, :] = True
        path = tmp_path / "fov.medh5"
        with medh5.create(path, codec="portable") as w:
            minimal(w)
            w.add_mask("fov", fov, grid="g")
            if masked:
                w.add_image(
                    "PET",
                    np.zeros(SHAPE, np.float32),
                    grid="g",
                    modality="PT",
                    valid_mask="fov",
                )
        image = "PET" if masked else "CT"
        expected = fov if masked else np.ones(SHAPE, dtype=bool)
        with medh5.open(path) as sample:
            for roi in (
                (slice(-2, None), slice(None), slice(None)),
                (slice(None, -3), slice(1, 100), slice(-4, -1)),
            ):
                region = sample.valid_region(image, roi)
                assert region.shape == sample.images[image].read(roi).shape
                np.testing.assert_array_equal(region, expected[roi])
            with pytest.raises(ValueError, match="step 2"):
                sample.valid_region(image, (slice(0, 4, 2), slice(None), slice(None)))
            with pytest.raises(MEDH5ValidationError, match="spatial axes"):
                sample.valid_region(image, (slice(0, 4),))

    def test_S4_3_pyramid_levels_are_addressable(self, tmp_path):
        from medh5.geometry import derive_level_grid

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
                    np.zeros((8, 16, 16), dtype=np.int16),
                ],
                grid_levels=["l0", "l1"],
                modality="CT",
            )
        with medh5.open(path) as sample:
            image = sample.images["CT"]
            assert image.is_multiscale
            assert image.levels == 2
            assert image.level(1).shape == (8, 16, 16)
            assert image.pyramid is not None
            assert "multiscale" in sample.profiles

    def test_S4_3_inconsistent_pyramid_is_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_grid("l0", shape=(16, 32, 32), spacing=(1.0, 1.0, 1.0))
            w.add_grid("l1", shape=(8, 16, 16), spacing=(2.0, 2.0, 2.0))
            w.add_pyramid(
                "CT",
                [np.zeros((16, 32, 32)), np.zeros((8, 16, 16))],
                grid_levels=["l0", "l1"],
                modality="CT",
            )
        assert exc.value.code == "E105"

    def test_S4_3_a_pyramid_without_levels_is_refused(self, tmp_path):
        with medh5.create(tmp_path / "x.medh5") as w:
            minimal(w)
            with pytest.raises(MEDH5ValidationError) as exc:
                w.add_pyramid("P", [], grid_levels=[], modality="CT")
        assert exc.value.code == "E105"

    def test_S4_3_a_refused_pyramid_leaves_nothing_behind(self, tmp_path):
        """A level of the wrong shape, or an unknown downsampling method, was
        refused after the image's group was made: the corrected call then
        failed with "name already exists", and the commit with it."""
        from medh5.geometry import derive_level_grid

        path = tmp_path / "pyr.medh5"
        shape = (16, 32, 32)
        with medh5.create(path, codec="portable") as w:
            base = w.add_grid("l0", shape=shape, spacing=(1.0, 1.0, 1.0))
            half = derive_level_grid(base, (2, 2, 2), "l1")
            w.add_grid("l1", shape=half.shape, spacing=half.spacing, origin=half.origin)
            levels = [np.zeros(shape, np.int16), np.zeros(half.shape, np.int16)]
            with pytest.raises(MEDH5ValidationError) as exc:
                w.add_pyramid(
                    "CT",
                    [levels[0], levels[0]],
                    grid_levels=["l0", "l1"],
                    modality="CT",
                )
            assert exc.value.code == "E202"
            with pytest.raises(MEDH5ValidationError) as exc:
                w.add_pyramid(
                    "CT",
                    levels,
                    grid_levels=["l0", "l1"],
                    modality="CT",
                    downsample_method="nearest-ish",
                )
            assert exc.value.code == "E105"
            w.add_pyramid("CT", levels, grid_levels=["l0", "l1"], modality="CT")
        with medh5.open(path) as sample:
            assert sample.images["CT"].levels == 2
        assert validate_file(path).ok


class TestProfiles:
    def test_profiles_are_inferred_from_content(self, sample_path, longitudinal_path):
        with medh5.open(sample_path) as sample:
            assert sample.profiles == frozenset({"core", "seg", "curation"})
        with medh5.open(longitudinal_path) as sample:
            assert {"longitudinal", "training"} <= sample.profiles

    def test_unknown_profile_is_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5", profiles=["quantum"]) as w,
        ):
            minimal(w)
        assert exc.value.code == "E007"


class TestTimepoints:
    def test_S3_7_indices_must_be_dense(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Timeline([Timepoint("a", 0), Timepoint("c", 2)])
        assert exc.value.code == "E108"

    def test_S3_7_at_least_one_timepoint(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Timeline([])
        assert exc.value.code == "E108"

    def test_S3_7_days_must_increase_with_index(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Timeline(
                [
                    Timepoint("a", 0, days_from_baseline=90),
                    Timepoint("b", 1, days_from_baseline=0),
                ]
            )
        assert exc.value.code == "E108"

    def test_S3_7_grid_must_declare_a_timepoint_when_several_exist(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_timepoint("tp0")
            w.add_timepoint("tp1")
            minimal(w)
        assert exc.value.code == "E106"

    def test_S3_7_undeclared_timepoint_is_refused(self, tmp_path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_timepoint("tp0")
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp9")
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        assert exc.value.code == "E107"

    def test_S3_7_timepoint_is_inherited_never_repeated(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            assert sample.images["CT_tp1"].timepoint == "tp1"
            assert sample.annotations["organs_tp1"].timepoints == ("tp1",)

    def test_timepoint_view_filters_everything(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            view = sample.at("tp1")
            assert set(view.images) == {"CT_tp1"}
            assert set(view.annotations) == {"organs_tp1"}
            assert set(view.grids) == {"ct_tp1"}
            assert sample.at(0).id == "tp0"
            assert "tp1" in repr(view)

    def test_interval_days(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            assert sample.timepoints.interval_days("tp0", "tp1") == 90.0
            assert sample.is_longitudinal

    def test_single_timepoint_needs_no_ceremony(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert sample.timepoints.ids == ("tp0",)
            assert not sample.is_longitudinal

    def test_undeclared_timepoint_lookup_is_helpful(self, sample_path):
        with medh5.open(sample_path) as sample:  # noqa: SIM117
            with pytest.raises(KeyError, match="undeclared"):
                sample.timepoints["tp7"]


class TestCoverage:
    def test_S11_3_annotated_subset_is_the_contract(self, tmp_path, label_set, masks):
        path = write_sample(
            tmp_path / "partial.medh5",
            label_set=label_set,
            masks=masks,
            annotated=[1],
        )
        with medh5.open(path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.is_annotated("liver")
            assert not seg.is_annotated("spleen")
            assert not seg.is_fully_covered
            assert {c.key for c in seg.annotated_classes} == {"liver"}

    def test_S6_2_annotated_must_be_a_subset(self):
        """The invariant itself: no header may claim coverage it cannot index."""
        from medh5.annotations.base import AnnotationHeader

        with pytest.raises(MEDH5ValidationError) as exc:
            AnnotationHeader(
                kind="layers",
                task="segmentation",
                class_ids=(1, 2),
                annotated_class_ids=(1, 4),
            )
        assert exc.value.code == "E403"

    def test_S11_3_a_class_examined_and_not_found_is_declared(
        self, tmp_path, label_set, masks
    ):
        """ "Verified absent" must not collapse into "never looked for" (§11.3)."""
        path = tmp_path / "absent.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "s", grid="g", masks=masks, annotated_classes=[1, 2, 3, 4]
            )
        with medh5.open(path) as sample:
            seg = sample.annotations["s"]
            assert 4 in seg.class_ids
            assert seg.is_annotated("vessel")
            assert not seg.dense(["vessel"])[0].any()
            assert seg.is_fully_covered
        assert not validate_file(path).errors

    def test_annotated_all_uses_the_label_set(self, tmp_path, label_set, masks):
        path = tmp_path / "all.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation("s", grid="g", masks=masks, annotated_classes="all")
        with medh5.open(path) as sample:
            assert sample.annotations["s"].is_fully_covered


class TestAnnotatedClassesAll:
    def test_S11_3_all_claims_the_whole_label_set_not_just_what_was_given(
        self, tmp_path, label_set
    ):
        """`"all"` and `"all_given"` were byte-identical; only one is documented so.

        `_resolve_annotated` intersected the label set back down to `class_ids`,
        so a class the annotator searched for and did not find never reached the
        file -- the "examined and absent" negative became "never looked for",
        which is the one distinction the coverage contract exists to keep. No
        validator fires either, because W904 only warns when
        `annotated_class_ids` is a strict subset, and this made them equal.
        """
        shape = SHAPE
        liver = np.zeros(shape, bool)
        liver[2:6, 4:12, 4:12] = True
        made = {}
        for mode in ("all", "all_given"):
            path = tmp_path / f"{mode}.medh5"
            with medh5.create(path, codec="portable") as w:
                w.label_set(label_set)
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
                w.add_segmentation(
                    "seg", grid="g", masks={1: liver}, annotated_classes=mode
                )
            with medh5.open(path) as sample:
                annotation = sample.annotations["seg"]
                made[mode] = (
                    annotation.class_ids,
                    annotation.annotated_class_ids,
                )

        assert made["all_given"] == ((1,), (1,))
        assert set(made["all"][1]) == set(label_set.ids)
        assert made["all"] != made["all_given"], (
            "'all' must not collapse onto 'all_given'"
        )

    def test_S11_3_all_works_for_a_partial_probability_map(self, tmp_path, label_set):
        """The probability branch ignored `examined`, so `"all"` could not be used.

        `class_ids` held only the planes supplied while `annotated_class_ids`
        named the whole label set, and `AnnotationHeader` rejected the write
        with E403 -- so the corrected `"all"` was unusable for exactly the
        ordinary case of a partial probability map. Absent classes need a
        zero-valued plane, as absent masks get an empty mask.
        """
        shape = SHAPE
        probabilities = {1: np.full(shape, 0.9, np.float32)}
        path = tmp_path / "probs.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "seg", grid="g", probabilities=probabilities, annotated_classes="all"
            )
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert set(annotation.class_ids) == set(label_set.ids)
            assert set(annotation.annotated_class_ids) == set(label_set.ids)
            # The supplied plane is untouched; the injected ones are empty.
            assert annotation.dense([1])[0].all()
            absent = next(c for c in label_set.ids if c != 1)
            assert not annotation.dense([absent])[0].any()


class TestTracking:
    def test_S7_4_instance_ids_join_across_timepoints(self, tmp_path, label_set):
        path = tmp_path / "track.medh5"

        def lesion(origin):
            mask = np.zeros(SHAPE, dtype=bool)
            mask[tuple(slice(o, o + 3) for o in origin)] = True
            return mask

        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
                )
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(SHAPE, dtype=np.int16),
                    grid=f"g_{tp}",
                    modality="CT",
                )
            w.add_segmentation(
                "les_tp0",
                grid="g_tp0",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 3, mask=lesion((5, 5, 5))),
                ],
            )
            w.add_segmentation(
                "les_tp1",
                grid="g_tp1",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 8, mask=lesion((9, 9, 9))),
                ],
            )
        with medh5.open(path) as sample:
            tracking = sample.tracks(3)
            assert sorted(tracking) == [1, 3, 8]
            assert tracking.is_persistent(1)
            assert tracking.is_resolved(3)
            assert tracking.is_new(8)
            assert tracking[1].timepoints == ("tp0", "tp1")
            assert sample.annotations["les_tp0"].instance(3).class_id == 3


class TestAmend:
    def test_S14_4_amend_is_copy_on_write(self, sample_path, masks):
        before = sample_path.stat().st_mtime_ns
        with medh5.amend(sample_path) as w:
            w.set_quality("organs_tp0", status="reviewed", reviewed_by=["r2"])
        with medh5.open(sample_path) as sample:
            assert sample.document.quality["organs_tp0"].status == "reviewed"
            seg = sample.annotations["organs_tp0"]
            assert np.array_equal(seg.dense([1])[0], masks[1])
        assert sample_path.stat().st_mtime_ns != before

    def test_S14_4_amend_preserves_unknown_objects(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            group = handle.create_group("x_acme_scratch")
            group.create_dataset("payload", data=np.arange(4))
        with medh5.amend(sample_path) as w:
            w.set_quality("organs_tp0", status="reviewed")
        with h5py.File(sample_path, "r") as handle:
            assert handle["x_acme_scratch/payload"][2] == 2

    def test_transcode_preserves_contains(self, sample_path, masks):
        with medh5.amend(sample_path) as w:
            w.transcode_annotation("organs_tp0", "bitmask")
        with medh5.open(sample_path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "bitmask"
            for class_id, mask in masks.items():
                assert np.array_equal(seg.dense([class_id])[0], mask)

    def test_transcode_to_the_same_kind_is_a_no_op(self, sample_path):
        with medh5.amend(sample_path) as w:
            assert w.transcode_annotation("organs_tp0", "layers") == "layers"

    def test_remove_annotation_drops_its_index(self, longitudinal_path):
        with medh5.amend(longitudinal_path) as w:
            w.remove_annotation("organs_tp1")
        with medh5.open(longitudinal_path) as sample:
            assert "organs_tp1" not in sample.annotations
            assert "organs_tp1" not in sample.index

    def test_S14_4_a_removed_annotation_leaves_no_bytes(self, sample_path):
        """HDF5 never reclaims an unlinked group's space: a removed annotation
        stayed in the amended file's bytes.  The
        commit rewrites the file from its root when something was removed."""
        marker = "SYNTHETIC-RATER-NOTE-41d9"
        with h5py.File(sample_path, "r+") as f:
            f["annotations/organs_tp0"].attrs["x_note"] = marker
        assert marker.encode() in sample_path.read_bytes(), "the control"
        with medh5.amend(sample_path) as w:
            w.remove_annotation("organs_tp0")
        assert marker.encode() not in sample_path.read_bytes()
        with medh5.open(sample_path) as sample:
            assert "organs_tp0" not in sample.annotations
            assert sample.verify().ok

    @pytest.mark.parametrize(
        ("alias", "link"),
        [("x_acme/meta", "hard"), ("a_first/meta", "hard"), ("x_acme/meta", "soft")],
    )
    def test_S2_4_an_alias_of_meta_names_the_document_the_amendment_wrote(
        self, tmp_path, alias, link
    ):
        """`/meta` was copied by path and written anew, so a hard link to it
        under another name kept the old document as an object of its own: an
        identifier the amendment replaced --- the record number a scrub
        pseudonymised --- stayed under the alias, and in the file's bytes.
        A soft link names `/meta` by path, and always followed it."""
        record = "MRN-0042-7731"
        path = tmp_path / "alias.medh5"
        with medh5.create(path, subject_id=record, codec="portable") as w:
            minimal(w)
        with h5py.File(path, "r+") as handle:
            handle.create_group(alias.split("/")[0])
            handle[alias] = handle["meta"] if link == "hard" else h5py.SoftLink("/meta")
        with medh5.amend(path) as w:
            w.identity(subject_id="pseudo-1")
        with h5py.File(path, "r") as handle:
            assert handle[alias] == handle["meta"]
            document = json.loads(handle[alias][()])
        assert document["identity"]["subject_id"] == "pseudo-1"
        assert record.encode() not in path.read_bytes()
        with medh5.open(path) as sample:
            assert sample.identity.subject_id == "pseudo-1"
            assert sample.verify().ok

    def test_S3_4_a_remapped_frame_uid_leaves_no_bytes(self, tmp_path):
        """HDF5 keeps an overwritten attribute's old value in the file's free
        space, so every frame UID `remap_frame_uids` replaced --- what a scrub
        pseudonymises them for --- stayed in the committed bytes."""
        uid = "1.2.826.0.1.3680043.8.498.4242"
        path = tmp_path / "frames.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid=uid)
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        assert uid.encode() in path.read_bytes(), "the control"
        with medh5.amend(path) as w:
            assert w.remap_frame_uids({uid: "pseudo-frame-1"})
        assert uid.encode() not in path.read_bytes()
        with medh5.open(path) as sample:
            assert sample.grids["g"].frame_uid == "pseudo-frame-1"
            assert sample.verify().ok

    @pytest.mark.parametrize("algo", ["sha512", "blake2b"])
    def test_S13_1_an_amendment_keeps_the_files_digest_algorithm(self, tmp_path, algo):
        """An amendment restamped every digest in sha256 under a root it
        relabelled --- a new `content_id` for an unchanged sample --- and with
        ``digests=False`` kept the old digests under a root declaring sha256:
        two algorithms in one file, which validated.  `fix --rewrite-digests`
        restamps an edited file in its own algorithm too, rather than in one
        its commit then finds changed from what the file failed (E701)."""
        from medh5.integrity import collect_digests, fix

        path = tmp_path / "algo.medh5"
        with medh5.create(path, codec="portable") as w:
            minimal(w)
        address = restamped_in(path, algo)

        def algorithms() -> set[str]:
            with medh5.open(path) as sample:
                assert sample.verify().ok
                found = {d.split(":")[0] for d in collect_digests(sample.root).values()}
            with h5py.File(path, "r") as handle:
                return found | {handle.attrs["digest_algo"]}

        for digests in (True, False):
            with medh5.amend(path) as w:
                w.commit(digests=digests)
            assert algorithms() == {algo}
            with medh5.open(path) as sample:
                assert sample.content_id == address, "an unchanged sample moved"
        with h5py.File(path, "r+") as handle:
            handle["images/CT"][0, 0, 0] = 7
        with medh5.open(path) as sample:
            assert sample.verify().mismatched == ("images/CT",)
        assert fix(path, rewrite_digests=True, reason="edited").rewrote_digests
        assert algorithms() == {algo}
        assert validate_file(path, level="integrity").ok

    def test_removing_a_missing_annotation_is_an_error(self, sample_path):
        with pytest.raises(MEDH5ValidationError), medh5.amend(sample_path) as w:
            w.remove_annotation("nope")

    def test_amend_does_not_inherit_profile_claims(self, longitudinal_path):
        """A claim the amended content no longer justifies must be dropped."""
        with medh5.amend(longitudinal_path) as w:
            w.remove_annotation("organs_tp0")
            w.remove_annotation("organs_tp1")
        with medh5.open(longitudinal_path) as sample:
            assert "training" not in sample.profiles
            assert "seg" not in sample.profiles


class TestAmendPreservesWhatItDoesNotOwn:
    def test_S16_unknown_root_attributes_survive_an_amend(self, tmp_path):
        """§16 lets a minor version add attributes; a 1.0 amend must keep them.

        Unknown attributes on grids, images, annotations and unknown groups all
        survived already -- the root was the one level that dropped them, so a
        1.0 tool amending a 1.1 file silently discarded whatever 1.1 put there.
        """
        path = tmp_path / "roots.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        with h5py.File(path, "r+") as handle:
            handle.attrs["x_future_root"] = "keep-me"
            handle.attrs["medh5_future_thing"] = "also-keep"

        with medh5.amend(path) as writer:
            writer.add_image("CT2", np.zeros(SHAPE, np.int16), grid="g", modality="MR")

        with h5py.File(path) as handle:
            assert handle.attrs["x_future_root"] == "keep-me"
            assert handle.attrs["medh5_future_thing"] == "also-keep"
            # The ones `commit` owns are still rewritten from the amended state,
            # and `profiles` is still derived rather than inherited: imaging
            # alone is written as 1.0, whatever the newest version is.
            assert handle.attrs["medh5_version"] == "1.0"
            assert "images" in handle and "CT2" in handle["images"]

    @posix_modes
    def test_S14_4_permissions_survive_the_copy_on_write_replace(self, tmp_path):
        """The mode is the access control on a shared research filesystem.

        Every copy-on-write command -- amend, scrub --apply, fix, recompress --
        replaces the file, and the replacement was created with the process
        umask, so a 0o600 sample came back 0o644 and world-readable.
        """
        path = tmp_path / "perm.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        for mode in (0o600, 0o640, 0o660):
            path.chmod(mode)
            with medh5.amend(path) as writer:
                writer.add_image(
                    f"X{mode:o}", np.zeros(SHAPE, np.int16), grid="g", modality="MR"
                )
            assert stat.S_IMODE(path.stat().st_mode) == mode


class TestAtomicRewrite:
    def test_the_source_is_closed_before_the_replace(self, tmp_path):
        """`with open_h5(...) as src, atomic_h5(...) as dst:` exits right-to-left.

        The replace therefore ran while the source was still open. POSIX allows
        that, so it passed here and on CI; Windows refuses to replace an open
        file, so every rewrite-in-place -- `repack`, and `recompress` without
        `out=` -- failed there. The ordering is asserted directly by the engine
        (`h5::file::tests::s14_4_the_source_is_closed_before_the_replace`);
        here, its outcome: a rewrite in place succeeds and leaves this process
        holding nothing of the file it replaced.
        """
        from medh5.storage import recompress

        path = tmp_path / "rw.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        before = path.stat().st_ino

        recompress(path, "archive")  # no `out=`: the rewrite in place

        assert path.stat().st_ino != before
        with medh5.open(path) as sample:
            assert "CT" in sample.images
            assert sample.verify().ok
        fds = Path("/proc/self/fd")
        if fds.is_dir():  # Linux: what this process still has open
            held = []
            for fd in fds.iterdir():
                try:
                    held.append(os.readlink(fd))
                except OSError:
                    continue  # closed while listing
            assert not [link for link in held if link.startswith(str(path))]

    @posix_modes
    def test_repack_preserves_content_and_permissions(self, tmp_path):
        """A repack rewrites storage, not content.

        The repack is the engine's (`sample::reader::tests::repack_preserves_
        content_and_permissions` holds the content id, the values and the mode
        in Rust); here it runs as scrub's last step, the one place the package
        repacks a file.
        """
        from medh5.curation import scrub

        path = tmp_path / "pack.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE),
                grid="g",
                modality="CT",
            )
        path.chmod(0o600)
        with medh5.open(path) as sample:
            data = sample.images["CT"].read()

        report = scrub.apply(path)

        assert report.applied
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with medh5.open(path) as sample:
            assert sample.verify().ok, "every digest survives the rewrite"
            assert np.array_equal(sample.images["CT"].read(), data)

    def test_S14_4_temporary_names_are_unique_within_a_process(self, tmp_path: Path):
        """Writers of one target, open at once, each get their own temporary.

        The name itself is the engine's (`h5::file::tests::
        s14_4_temporary_names_are_unique_within_a_process`); this holds what it
        is for: concurrent writers in one process never write the same file.
        """
        target = tmp_path / "x.medh5"
        writers = [Flat.open_writer(target) for _ in range(50)]
        names = {p.name for p in tmp_path.iterdir()}
        assert len(names) == 50
        assert all(n.startswith(".x.medh5.tmp-") for n in names)
        for w in writers:
            w.commit()
        assert [p.name for p in tmp_path.iterdir()] == ["x.medh5"]
        with medh5.open(target) as sample:
            assert sample.verify().ok


# Arguments for the view methods that take some, by method name.
POINT = np.array([[1.0, 2.0, 3.0]])
VIEW_ARGS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    "at": (("tp0",), {}),
    "bbox": ((1,), {}),
    "class_key": ((1,), {}),
    "classes_at": (((2, 2, 2),), {}),
    "contains": ((1, (2, 2, 2)), {}),
    "coords": ((1,), {}),
    "crop": ((0,), {}),
    "displacement_at": ((POINT,), {}),
    "document_text": (("x",), {}),
    "get": (("x",), {}),
    "grid_in": (("F0",), {}),
    "has_class": ((1,), {}),
    "ignore_region": (("lm",), {}),
    "instance": ((7,), {}),
    "inverse_points": ((POINT,), {}),
    "is_annotated": ((1,), {}),
    "is_current": (("x",), {}),
    "level": ((0,), {}),
    "occupancy_plane": ((0,), {}),
    "read_layer": ((0,), {}),
    "resolve_class": ((1,), {}),
    "resolve_classes": (([1],), {}),
    "resolve_frames": (("F0", "F1"), {}),
    "sample_foreground": ((1,), {"rng": 0}),
    "sample_indices": ((POINT,), {}),
    "scheme": (("x",), {}),
    "state": (("liver",), {}),
    "to_index": ((POINT,), {}),
    "to_world": ((POINT,), {}),
    "transform_between": (("F0", "F1"), {}),
    "transform_points": ((POINT,), {}),
    "valid_region": (("CT",), {}),
    "value": (("liver",), {}),
}


def every_view(path: Path) -> Path:
    """A sample holding one of each kind of view a reader hands out."""
    from medh5.geometry import derive_level_grid

    shape = Organs.SHAPE
    organs = Organs.blocks()
    spleen = np.zeros(shape, dtype=bool)
    spleen[4:6, 6:10, 6:10] = True
    matrix = np.eye(4)
    matrix[:3, 3] = (1.0, 0.0, 0.0)
    with medh5.create(path, sample_id="views", codec="portable") as w:
        w.label_set(Organs.label_set())
        spacing = (2.0, 0.8, 0.8)
        g = w.add_grid("g", shape=shape, spacing=spacing, frame_uid="F0")
        w.add_grid("f", shape=shape, spacing=spacing, frame_uid="F1")
        half = derive_level_grid(g, (2, 2, 2), "h")
        w.add_grid(
            "h",
            shape=half.shape,
            spacing=half.spacing,
            origin=half.origin,
            direction=half.direction,
            frame_uid="F0",
        )
        ct = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        w.add_image(
            "CT", ct, grid="g", modality="CT", window_center=[40], window_width=[400]
        )
        w.add_pyramid(
            "P",
            [ct, np.zeros(half.shape, np.int16)],
            grid_levels=["g", "h"],
            modality="CT",
        )
        w.add_segmentation(
            "lm",
            grid="g",
            masks={1: organs[1], 2: spleen},
            encoding="labelmap",
            ignore=Organs.ignore(),
        )
        w.add_segmentation("ly", grid="g", masks=organs, encoding="layers")
        w.add_segmentation("bm", grid="g", masks=organs, encoding="bitmask")
        w.add_segmentation(
            "pm", grid="g", probabilities={1: organs[1].astype(np.float32)}
        )
        w.add_segmentation(
            "in",
            grid="g",
            instances=[InstanceInput(class_id=3, instance_id=7, mask=organs[3])],
        )
        w.add_mask("fov", np.ones(shape, dtype=bool), grid="g")
        w.add_boxes("bx", [[[1, 4], [1, 6], [1, 6]]], ["liver"], grid="g")
        w.add_points("pt", POINT, grid="g", names=["apex"])
        w.add_classification("cl", {"liver": 1.0, "lesion": 0.0})
        w.add_transform(
            "af", kind="affine", from_frame="F0", to_frame="F1", matrix=matrix
        )
        w.add_transform(
            "dp",
            kind="displacement",
            from_frame="F1",
            to_frame="F2",
            field=np.zeros((3, *shape), dtype=np.float32),
            field_grid="f",
        )
        w.add_transform(
            "ch",
            kind="composite",
            from_frame="F0",
            to_frame="F2",
            components=["af", "dp"],
        )
        w.build_index(["lm"], occupancy=2)
    return path


def answer(view: Any, name: str) -> tuple[Any, ...]:
    """What one public attribute of a view gives: its value (a method called
    with its `VIEW_ARGS`, an iterator run out), or the exception it raised."""
    static = inspect.getattr_static(view, name)
    args, kwargs = VIEW_ARGS.get(name, ((), {}))
    try:
        value = getattr(view, name)
        if callable(static):
            value = value(*args, **kwargs)
        if isinstance(value, Iterator):
            value = list(value)
    except Exception as exc:  # the exception is the answer
        return ("raised", type(exc), str(exc))
    return ("value", value)


def walk(view: Any) -> dict[str, tuple[Any, ...]]:
    """Every public method and property of a view, and `len` and `iter`."""
    found = {}
    for name in dir(view):
        if name.startswith("_") and name not in ("__len__", "__iter__"):
            continue
        if name in ("close", "is_open"):  # the lifecycle itself
            continue
        static = inspect.getattr_static(view, name)
        if isinstance(static, staticmethod | classmethod):
            continue
        found[name] = answer(view, name)
    return found


def same(a: Any, b: Any) -> bool:
    """Equal values: arrays by content, containers item by item, a view (an
    object without equality of its own) by its type."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(same(a[k], b[k]) for k in a)
        )
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b, strict=True))
        )
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, float) and isinstance(b, float) and a != a:
        return b != b
    if type(a).__eq__ is object.__eq__:
        return type(a) is type(b)
    return bool(a == b)


class TestClose:
    """§14.4: a closed sample holds nothing open.

    1.x closed the h5py file and everything opened through it.  HDF5 keeps a
    file open while any object in it is, so an image or group a caller kept
    past the ``with`` block held the file open --- and locked against the next
    writer --- until garbage collection got to it.
    """

    def test_S14_4_close_releases_the_file_with_everything_read_from_it(
        self, sample_path
    ):
        with medh5.open(sample_path) as sample:
            image = sample.images["CT_tp0"]
            stored = sample.root["images/CT_tp0"]
        with h5py.File(sample_path, "r+") as handle:  # not locked
            handle.attrs["x_touched"] = encode_attr("yes")
        with pytest.raises(MEDH5FileError, match="closed"):
            image.read()
        with pytest.raises(MEDH5FileError, match="closed"):
            stored.shape  # noqa: B018 - the access is the assertion
        assert repr(stored) == "<closed medh5 Dataset>"

    @pytest.mark.filterwarnings("ignore:labelmap\\(\\) flattened")
    def test_S14_4_every_view_refuses_a_closed_file_or_answers_as_before(
        self, tmp_path
    ):
        """Every public method and property of every view, after ``close()``:
        ``MEDH5FileError``, or exactly the answer it gave while the file was
        open.  A closed file read as an empty one answered shape ``()``, name
        ``''``, no occupancy map, or "requires a data dataset" (E410/E502)."""
        sample = medh5.open(every_view(tmp_path / "views.medh5"))
        ct = sample.images["CT"]
        views = [
            sample,
            *sample.images.values(),
            sample.images["P"].level(1),
            *sample.annotations.values(),
            *sample.transforms.values(),
            *sample.index.values(),
            sample.root,
            sample.root.attrs,
            ct.dataset,
            ct.dataset.attrs,
            sample.annotations["lm"].group,
        ]
        before = [walk(view) for view in views]
        sample.close()
        after = [walk(view) for view in views]

        problems = []
        for view, was, now in zip(views, before, after, strict=True):
            for name, (kind, *found) in now.items():
                where = f"{type(view).__name__}.{name}"
                assert not (was[name][0] == "raised" and was[name][1] is TypeError), (
                    f"{where} was not called: {was[name][2]}"
                )
                if kind == "raised" and issubclass(found[0], MEDH5FileError):
                    continue
                if kind == was[name][0] == "value" and same(was[name][1], found[0]):
                    continue
                if kind == was[name][0] == "raised" and found[0] is was[name][1]:
                    continue
                problems.append(f"{where}: {was[name][:2]} then {(kind, *found)}")
        assert problems == [], "\n".join(problems)
        assert sum(map(len, after)) > 300

    def test_S14_4_a_closed_sample_or_image_says_so(self, sample_path):
        """`repr()` of a closed sample or image raised, as a closed dataset's
        does not: debuggers, loggers and test reports ask for one after the
        ``with`` block."""
        with medh5.open(sample_path) as sample:
            image = sample.images["CT_tp0"]
            while_open = repr(sample), repr(image)
        closed = f"Sample({str(sample_path)!r}, closed)", "Image('CT_tp0', closed)"
        assert (repr(sample), repr(image)) == closed != while_open

    def test_S14_4_commit_closes_what_the_writer_handed_out(self, tmp_path):
        path = tmp_path / "w.medh5"
        with medh5.create(path, sample_id="w") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            kept = w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        with h5py.File(path, "r+"):
            pass  # the written file is closed and in place
        with pytest.raises(MEDH5FileError, match="closed"):
            kept.shape  # noqa: B018
        with medh5.open(path) as sample:
            assert sample.verify().ok


class TestOpen:
    def test_a_0x_file_is_refused_loudly(self, tmp_path):
        path = tmp_path / "old.medh5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = "1"
        with pytest.raises(MEDH5VersionError, match="medh5 migrate"):
            medh5.open(path)

    def test_a_future_major_is_refused(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("2.0")
        with pytest.raises(MEDH5VersionError):
            medh5.open(sample_path)

    def test_a_collection_is_not_a_sample(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle.attrs["medh5_kind"] = encode_attr("collection")
        with pytest.raises(MEDH5FileError, match="collection"):
            medh5.open(sample_path)

    def test_write_modes_are_refused(self, sample_path):
        with pytest.raises(MEDH5ValidationError, match="create"):
            medh5.open(sample_path, "w")

    def test_missing_file(self, tmp_path):
        with pytest.raises(MEDH5FileError):
            medh5.open(tmp_path / "nope.medh5")

    def test_a_missing_file_is_a_file_not_found_error(self, tmp_path):
        """One missing file gave three kinds of error, and the one `open` gave
        carried no `errno`, so ``except FileNotFoundError`` missed it."""
        missing = tmp_path / "nope.medh5"
        with pytest.raises(FileNotFoundError) as raised:
            medh5.open(missing)
        error = raised.value
        assert isinstance(error, MEDH5FileNotFoundError)
        assert isinstance(error, MEDH5FileError)
        assert error.errno == errno.ENOENT
        assert error.filename == str(missing)
        assert str(error) == (
            f"failed to open {str(missing)!r}: no such file or directory"
        )
        for door in (medh5.amend, medh5.open_collection):
            with pytest.raises(MEDH5FileNotFoundError):
                door(missing)

    def test_an_os_error_from_the_engine_carries_its_errno(self, tmp_path):
        """The engine wrote "[Errno 2] ..." into a plain `OSError` whose
        `errno` was `None`."""
        from medh5 import conformance

        with pytest.raises(FileNotFoundError) as raised:
            conformance.check_checksums(tmp_path / "no-suite")
        assert raised.value.errno == errno.ENOENT
        assert Path(raised.value.filename) == tmp_path / "no-suite" / "SHA256SUMS"
        # Python's own words: Windows numbers the error as it does (a missing
        # directory is its error 3), and Python maps it to ENOENT.
        expected = "[WinError 3] " if sys.platform == "win32" else "[Errno 2] "
        assert str(raised.value).startswith(expected)

    def test_a_directory_is_an_is_a_directory_error(self, tmp_path):
        with pytest.raises(IsADirectoryError) as raised:
            medh5.open(tmp_path)
        assert isinstance(raised.value, MEDH5IsADirectoryError)
        assert raised.value.errno == errno.EISDIR
        assert raised.value.filename == str(tmp_path)
        assert "it is a directory, not a file" in str(raised.value)

    @posix_modes
    @pytest.mark.skipif(
        getattr(os, "geteuid", lambda: 1)() == 0, reason="root reads any file"
    )
    def test_an_unreadable_file_is_a_permission_error(self, sample_path):
        sample_path.chmod(0)
        try:
            with pytest.raises(PermissionError) as raised:
                medh5.open(sample_path)
        finally:
            sample_path.chmod(0o644)
        assert isinstance(raised.value, MEDH5PermissionError)
        assert raised.value.errno == errno.EACCES

    def test_a_file_error_pickles_with_its_errno(self, tmp_path):
        with pytest.raises(MEDH5FileNotFoundError) as raised:
            medh5.open(tmp_path / "nope.medh5")
        back = pickle.loads(pickle.dumps(raised.value))
        assert type(back) is MEDH5FileNotFoundError
        assert (back.errno, back.filename, str(back)) == (
            raised.value.errno,
            raised.value.filename,
            str(raised.value),
        )
        # The defaults keep the one-argument form.
        plain = MEDH5FileError("x")
        assert (str(plain), plain.errno, plain.filename) == ("x", None, None)
        assert str(pickle.loads(pickle.dumps(plain))) == "x"

    def test_reference_grid_falls_back_to_the_first_image(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert sample.reference_grid.grid_id == "ct_tp0"

    def test_collections_report_helpful_key_errors(self, sample_path):
        with medh5.open(sample_path) as sample:  # noqa: SIM117
            with pytest.raises(KeyError, match="available"):
                sample.images["nope"]

    def test_collection_views(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            assert sample.images.by_modality("CT") == ("CT_tp0", "CT_tp1")
            assert sample.images.on_grid("ct_tp0") == ("CT_tp0",)
            assert len(sample.annotations.by_task("segmentation")) == 2
            assert sample.annotations.by_kind("layers")
            assert sample.annotations.spanning() == ()
            assert "image" in repr(sample.images)

    def test_open_is_read_only(self, tmp_path: Path):
        path = tmp_path / "ro.medh5"
        with Flat.open_writer(path):
            pass
        with pytest.raises(MEDH5ValidationError, match="read-only"):
            medh5.open(path, "r+")
        with medh5.open(path, "r") as sample:
            assert sample.identity.sample_id == "ro"

    def test_S3_2_a_file_with_no_grids_raises_E111(self, tmp_path: Path):
        """`reference_grid` ended in `grids[sorted(grids)[0]]` --- a bare IndexError."""
        import h5py

        path = tmp_path / "nogrid.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        with h5py.File(path, "r+") as handle:
            del handle["images"]["CT"]
            del handle["grids"]["g"]
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                _ = sample.reference_grid
            assert exc.value.code == "E111"

    @staticmethod
    def _found(path: Path, location: str) -> list[str]:
        return [
            d.code for d in validate_file(path).diagnostics if d.location == location
        ]

    def test_S2_3_an_image_that_leads_nowhere_is_refused_as_E001(self, sample_path):
        """Listing the images raised a bare `KeyError` for a member that is a
        link to nothing: a defect of the file, read as a name the caller got
        wrong.  The reader now refuses it under the validator's code."""
        with h5py.File(sample_path, "r+") as handle:
            handle["images"]["PET"] = h5py.SoftLink("/nowhere")
        with medh5.open(sample_path) as sample:
            with pytest.raises(MEDH5ValidationError, match="'PET'") as exc:
                list(sample.images)
            assert exc.value.code == "E001"
            with pytest.raises(MEDH5ValidationError):
                sample.summary()
        assert self._found(sample_path, "/images/PET") == [exc.value.code]

    def test_S4_2_an_image_of_strings_is_refused_as_E001(self, tmp_path: Path):
        """An image is numbers; one of strings raised a bare `TypeError` from
        `summary()`, `dtype` and `read()`, as for an argument of the wrong
        type, where the validator reports the file (E001)."""
        path = tmp_path / "text.medh5"
        with Organs.writer(path):
            pass
        with h5py.File(path, "r+") as handle:
            old = handle["images/CT"]
            text = handle["images"].create_dataset("CT.text", old.shape, dtype="S4")
            for key, value in old.attrs.items():
                text.attrs[key] = value
            del handle["images/CT"]
            handle["images"].move("CT.text", "CT")
        with medh5.open(path) as sample:
            image = sample.images["CT"]
            for call in (image.summary, image.read, lambda: image.dtype):
                with pytest.raises(MEDH5ValidationError, match="'CT'") as exc:
                    call()
                assert exc.value.code == "E001"
        assert self._found(path, "/images/CT") == ["E001"]


class TestDocument:
    def test_meta_is_a_scalar_utf8_string(self, sample_path):
        with h5py.File(sample_path, "r") as handle:
            node = handle["meta"]
            assert node.shape == ()
            assert h5py.check_string_dtype(node.dtype) is not None
            json.loads(as_str(node[()]))

    def test_S2_4_no_value_is_mirrored(self, sample_path):
        """§2.4: `/meta` MUST NOT duplicate what lives in an HDF5 attribute."""
        with h5py.File(sample_path, "r") as handle:
            document = json.loads(as_str(handle["meta"][()]))
        flat = json.dumps(document)
        for banned in ("spacing", "origin", "direction", "class_ids", "shape"):
            assert f'"{banned}"' not in flat

    def test_document_schema_failure_is_reported(self, tmp_path):
        path = tmp_path / "bad.medh5"
        with medh5.create(path, codec="portable") as w:
            minimal(w)
        with h5py.File(path, "r+") as handle:
            del handle["meta"]
            handle.create_dataset("meta", data="{}", dtype=str_dtype())
        assert "E005" in validate_file(path, level="structural").codes

    def test_document_accessors(self, sample_path):
        with medh5.open(sample_path) as sample:
            document = sample.document
            assert document.subject_id == "subj-A"
            assert document.group_id == "subj-A"
            assert document.quality_of("organs_tp0") is not None
            assert document.quality_of(None) is None
            assert document.summary()["label_set"]["classes"] == 4

    @staticmethod
    def _rewritten(path: Path, version: str, edit: Any) -> Path:
        """`/meta` edited by *edit*, and the root declaring *version*."""
        with h5py.File(path, "r+") as handle:
            document = json.loads(as_str(handle["meta"][()]))
            edit(document)
            del handle["meta"]
            handle.create_dataset("meta", data=json.dumps(document), dtype=str_dtype())
            handle.attrs["medh5_version"] = encode_attr(version)
        return path

    def test_S2_2_a_later_minors_document_is_read_as_its_projection(self, sample_path):
        """1.1 §2.2: a reader MUST accept a higher minor for reading.  The
        validator read past a value only a later minor defines (W913), and the
        reader refused the same document (E005)."""
        from medh5.document import read_document

        def later(document: dict[str, Any]) -> None:
            document["identity"]["sex"] = "X"
            document["provenance"]["activities"][0]["type"] = "synthesize"

        self._rewritten(sample_path, LATER_MINOR, later)
        with medh5.open(sample_path) as sample:
            assert sample.identity.sex is None
            assert sample.document.provenance.activities[0].type == "other"
            assert read_document(sample.root).identity.sex is None
        # A version this reader implements is held to the schema as written.
        self._rewritten(sample_path, "1.0", lambda _: None)
        with medh5.open(sample_path) as sample:
            with pytest.raises(MEDH5ValidationError) as caught:
                _ = sample.document
            assert caught.value.code == "E005"

    def test_S2_2_a_value_nothing_known_stands_in_for_is_named(self, sample_path):
        """An agent's `type` is required and its vocabulary has no `other`, so a
        later minor's value leaves the document unread, and says where --- under
        W913, as the validator reports the value, not E005, which says the
        document is malformed."""

        def robot(document: dict[str, Any]) -> None:
            document["provenance"]["agents"][0]["type"] = "robot"

        self._rewritten(sample_path, LATER_MINOR, robot)
        with medh5.open(sample_path) as sample:
            with pytest.raises(
                MEDH5ValidationError, match="provenance/agents/0/type"
            ) as caught:
                _ = sample.document
            assert caught.value.code == "W913"


class TestProjectionRead:
    """1.1 §2.2: a later minor's file is read as this engine's projection.  An
    object holding a value only that minor defines failed the whole listing it
    was in --- one grid's axis kind every image and annotation, one
    annotation's task every annotation --- where the validator read past it
    (W913).  It is left out, with what depends on it, and a lookup of it is
    W913, saying why."""

    LISTINGS = ("grids", "images", "annotations", "transforms", "index")

    @staticmethod
    def _sample(path: Path) -> Path:
        """On grid `g` (frame 1.2.3.4): image `CT`, the indexed segmentation
        `seg`, boxes `found`, classification `graded` and the field `warp`; on
        grid `h` (frame 5.6): image `MR` and segmentation `segh`; and the affine
        `rigid` from `h`'s frame to `g`'s."""
        mask = np.zeros(Framed.SHAPE, bool)
        mask[1:4, 2:6, 2:6] = True
        with Framed.writer(path) as w:
            w.add_grid(
                "h", shape=Framed.SHAPE, spacing=(2.0, 1.0, 1.0), frame_uid="5.6"
            )
            w.add_image("MR", np.zeros(Framed.SHAPE, np.int16), grid="h", modality="MR")
            w.add_segmentation("seg", grid="g", masks={1: mask})
            w.add_segmentation("segh", grid="h", masks={1: mask})
            w.add_boxes("found", [Framed.box(1.5, 2.5)], [1], grid="g")
            w.add_classification("graded", {1: 1.0}, grid="g")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="1.2.3.4",
                to_frame="5.6",
                field=np.zeros((3, *Framed.SHAPE), np.float32),
                field_grid="g",
            )
            w.add_transform(
                "rigid",
                kind="affine",
                from_frame="5.6",
                to_frame="1.2.3.4",
                matrix=np.eye(4),
            )
            w.build_index(["seg"])
        return path

    @staticmethod
    def _edited(
        path: Path, node: str, attribute: str, value: Any, version: str = LATER_MINOR
    ) -> Path:
        """*node*'s *attribute* set to *value*, and the root at *version*."""
        with h5py.File(path, "r+") as handle:
            handle[node].attrs[attribute] = encode_attr(value)
            handle.attrs["medh5_version"] = encode_attr(version)
        return path

    def _listed(self, sample: Any) -> dict[str, list[str]]:
        return {name: sorted(getattr(sample, name)) for name in self.LISTINGS}

    def test_S2_2_a_grid_of_a_later_axis_kind_is_left_out_with_what_lies_on_it(
        self, tmp_path
    ):
        path = self._edited(
            self._sample(tmp_path / "s.medh5"),
            "grids/g",
            "axis_kinds",
            ["depth", "spatial", "spatial"],
        )
        with medh5.open(path) as sample:
            assert self._listed(sample) == {
                "grids": ["h"],
                "images": ["MR"],
                "annotations": ["segh"],
                "transforms": ["rigid"],
                "index": [],
            }
            assert sample.images["MR"].read().shape == Framed.SHAPE
            assert sample.annotations["segh"].dense().any()
            assert "g" not in sample.grids and sample.grids.get("g") is None
            assert "CT" not in sample.images and sample.images.get("CT") is None
            with pytest.raises(MEDH5ValidationError, match="axis kind 'depth'") as e:
                _ = sample.grids["g"]
            assert e.value.code == "W913" and "1.1 §2.2" in str(e.value)
            lookups = [
                (sample.images, "CT"),
                (sample.annotations, "seg"),
                (sample.annotations, "found"),
                (sample.transforms, "warp"),
                (sample.index, "seg"),
            ]
            for mapping, key in lookups:
                with pytest.raises(MEDH5ValidationError, match="grid 'g'") as e:
                    _ = mapping[key]
                assert e.value.code == "W913", key
            # Any other name is still the `KeyError` it was.
            with pytest.raises(KeyError):
                _ = sample.grids["nope"]
            assert sorted(sample.summary()["left_out"]) == [
                "annotations/found",
                "annotations/graded",
                "annotations/seg",
                "grids/g",
                "images/CT",
                "index/seg",
                "transforms/warp",
            ]
        assert validate_file(path).ok

    @pytest.mark.parametrize(
        ("ann_id", "attribute", "value"),
        [
            ("segh", "kind", "sdf"),
            ("segh", "kind", "rle"),
            ("segh", "task", "tracking"),
            ("segh", "closure", "transitive"),
            ("found", "space", "patient"),
            ("graded", "scope", "lesion_pair"),
        ],
    )
    def test_S2_2_an_annotation_of_a_later_value_is_left_out_alone(
        self, tmp_path, ann_id, attribute, value
    ):
        path = self._edited(
            self._sample(tmp_path / "s.medh5"),
            f"annotations/{ann_id}",
            attribute,
            value,
        )
        with medh5.open(path) as sample:
            known = {"seg", "segh", "found", "graded"} - {ann_id}
            assert sorted(sample.annotations) == sorted(known)
            assert sample.annotations["seg"].dense().any()
            assert sorted(sample.index) == ["seg"]
            with pytest.raises(MEDH5ValidationError, match=f"'{value}'") as e:
                _ = sample.annotations[ann_id]
            assert e.value.code == "W913"
        assert validate_file(path).ok

    @pytest.mark.parametrize(
        ("transform_id", "attribute", "value"),
        [("rigid", "kind", "similarity"), ("warp", "vector_space", "material")],
    )
    def test_S2_2_a_transform_of_a_later_value_is_left_out_alone(
        self, tmp_path, transform_id, attribute, value
    ):
        path = self._edited(
            self._sample(tmp_path / "s.medh5"),
            f"transforms/{transform_id}",
            attribute,
            value,
        )
        with medh5.open(path) as sample:
            (other,) = {"rigid", "warp"} - {transform_id}
            assert sorted(sample.transforms) == [other]
            moved = sample.transforms[other].transform_points(np.zeros((1, 3)))
            assert moved.shape == (1, 3)
            with pytest.raises(MEDH5ValidationError, match=f"'{value}'") as e:
                _ = sample.transforms[transform_id]
            assert e.value.code == "W913"
        assert validate_file(path).ok

    @staticmethod
    def _chained(path: Path) -> Path:
        """Frames 1, 2 and 3 on grids `a`, `b` and `c`; affines `ab` and `bc`
        with their composite `ac`, and `ba`, the inverse `ab` names."""
        with medh5.create(path, codec="portable") as w:
            for gid, frame in (("a", "1"), ("b", "2"), ("c", "3")):
                w.add_grid(
                    gid, shape=(4, 6, 6), spacing=(1.0, 1.0, 1.0), frame_uid=frame
                )
            w.add_image("CT", np.zeros((4, 6, 6), np.int16), grid="a", modality="CT")
            for tid, source, target in (
                ("ab", "1", "2"),
                ("ba", "2", "1"),
                ("bc", "2", "3"),
            ):
                inverse = {"ab": "ba", "ba": "ab"}.get(tid)
                w.add_transform(
                    tid,
                    kind="affine",
                    from_frame=source,
                    to_frame=target,
                    matrix=np.eye(4),
                    inverse_id=inverse,
                )
            w.add_transform(
                "ac",
                kind="composite",
                from_frame="1",
                to_frame="3",
                components=["ab", "bc"],
            )
        return path

    def test_S2_2_a_composite_of_a_step_left_out_is_left_out(self, tmp_path):
        path = self._edited(
            self._chained(tmp_path / "c.medh5"), "transforms/bc", "kind", "similarity"
        )
        with medh5.open(path) as sample:
            assert sorted(sample.transforms) == ["ab", "ba"]
            with pytest.raises(MEDH5ValidationError, match="transform 'bc'") as e:
                _ = sample.transforms["ac"]
            assert e.value.code == "W913"
        assert validate_file(path).ok

    def test_S2_2_a_stored_inverse_left_out_is_no_stored_inverse(self, tmp_path):
        """`inverse_id` resolves among the transforms the projection reads: the
        partner left out, `ab` has no stored inverse, as with an `inverse_id`
        naming nothing, and is inverted as an affine with none is."""
        path = self._edited(
            self._chained(tmp_path / "c.medh5"), "transforms/ba", "kind", "similarity"
        )
        with medh5.open(path) as sample:
            assert sorted(sample.transforms) == ["ab", "ac", "bc"]
            assert sample.transforms["ab"].inverse() is None
            back = sample.transform_between("b", "a")
            assert back is not None and back.to_frame == "1"
        found = validate_file(path).diagnostics
        assert ("W913", "/transforms/ba") in {(d.code, d.location) for d in found}

    @pytest.mark.parametrize(
        ("version", "code"), [("1.1", "E502"), (LATER_MINOR, "W913")]
    )
    def test_S10_1_an_inverse_of_a_kind_not_known_is_its_own_finding(
        self, tmp_path, version, code
    ):
        """The validator reported `ab`'s `inverse_id` as naming a transform that
        does not exist (E505) beside `ba`'s own finding, which failed a later
        minor's file the projection reads."""
        path = self._edited(
            self._chained(tmp_path / "c.medh5"),
            "transforms/ba",
            "kind",
            "similarity",
            version=version,
        )
        report = validate_file(path)
        found = {(d.code, d.location) for d in report.diagnostics}
        assert (code, "/transforms/ba") in found
        assert not [d for d in report.diagnostics if d.code == "E505"], found
        assert report.ok is (version == LATER_MINOR)

    @pytest.mark.parametrize(
        ("node", "attribute", "value", "code", "listing"),
        [
            ("grids/g", "axis_kinds", ["depth", "spatial", "spatial"], "E110", "grids"),
            ("annotations/segh", "kind", "sdf", "E401", "annotations"),
            ("annotations/segh", "task", "tracking", "E412", "annotations"),
            ("annotations/segh", "closure", "transitive", "E412", "annotations"),
            ("transforms/rigid", "kind", "similarity", "E502", "transforms"),
        ],
    )
    def test_S2_2_a_version_this_engine_implements_is_read_as_before(
        self, tmp_path, node, attribute, value, code, listing
    ):
        path = self._edited(
            self._sample(tmp_path / "s.medh5"), node, attribute, value, version="1.1"
        )
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as e:
                _ = getattr(sample, listing)
            assert e.value.code == code

    @pytest.mark.parametrize(
        ("node", "edits", "code"),
        [
            # A grid's shape is read before its axis kinds,
            ("grids/g", {"axis_kinds": ["depth", "spatial"]}, "E109"),
            # boxes' classes before the space they are in is used,
            (
                "annotations/found",
                {"space": "patient", "annotated_class_ids": [1, 2]},
                "E403",
            ),
            # and a field's frames before its vector space is.
            (
                "transforms/warp",
                {"vector_space": "material", "from_frame": None},
                "E502",
            ),
        ],
    )
    def test_S2_2_a_defect_met_before_the_later_value_is_the_error_it_was(
        self, tmp_path, node, edits, code
    ):
        """An object is read as far as a file of 1.1 is read before the value
        is met: what is wrong with it before then fails the listing, with the
        code it has in a file of any version."""
        path = self._sample(tmp_path / "s.medh5")
        listing = node.split("/")[0]
        for version in ("1.1", LATER_MINOR):
            with h5py.File(path, "r+") as handle:
                for name, value in edits.items():
                    if value is None:
                        handle[node].attrs.pop(name, None)
                    else:
                        handle[node].attrs[name] = encode_attr(value)
                handle.attrs["medh5_version"] = encode_attr(version)
            with medh5.open(path) as sample:
                with pytest.raises(MEDH5ValidationError) as e:
                    _ = getattr(sample, listing)
                assert e.value.code == code, version

    @pytest.mark.parametrize(
        ("node", "attribute", "value"),
        [
            ("grids/g", "units", "cm"),
            ("grids/g", "time_units", "h"),
            ("images/MR", "value_type", "label"),
        ],
    )
    def test_S2_2_a_value_the_reader_carries_as_written_leaves_nothing_out(
        self, tmp_path, node, attribute, value
    ):
        path = self._edited(self._sample(tmp_path / "s.medh5"), node, attribute, value)
        with medh5.open(path) as sample:
            assert self._listed(sample)["images"] == ["CT", "MR"]
            assert len(sample.annotations) == 4
            assert "left_out" not in sample.summary()

    def test_S2_2_a_member_that_lost_its_version_is_read_as_before(self, tmp_path):
        """A collection member declaring no version is read as the oldest
        (1.1 §8), no later minor's: its listings are a member's with its
        version, and a missing key is the `KeyError` it was."""
        from medh5.collection import open_collection, pack

        members = [self._sample(tmp_path / f"{name}.medh5") for name in ("a", "b")]
        shard = tmp_path / "s.medh5c"
        pack(members, shard)
        with h5py.File(shard, "r+") as handle:
            del handle["samples/a"].attrs["medh5_version"]
        with open_collection(shard) as collection:
            lost = collection["a"]
            assert self._listed(lost) == self._listed(collection["b"])
            assert type(lost.grids) is dict
            with pytest.raises(KeyError):
                _ = lost.annotations["nope"]


class TestDefaultIds:
    """§12.1: the default `sample_id` and pack key are the file name up to its
    first dot; 3.0 takes the stem, and 2.1 warns wherever the two differ."""

    @staticmethod
    def _write(path: Path, **ids: str) -> Path:
        with medh5.create(path, codec="portable", **ids) as w:
            w.add_grid("g", shape=Flat.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", Flat.image(), grid="g", modality="CT")
        return path

    def test_S12_1_a_dotted_name_warns_that_its_default_id_will_change(
        self, tmp_path: Path
    ):
        with pytest.warns(
            FutureWarning, match=r"^medh5 3\.0 names this sample 'case\.v2'"
        ) as caught:
            path = self._write(tmp_path / "case.v2.medh5")
        assert "sample_id=" in str(caught[0].message)
        assert caught[0].filename == __file__
        with medh5.open(path) as sample:
            assert (sample.identity.sample_id, sample.identity.subject_id) == (
                "case",
                "case",
            )

    def test_S12_1_an_undotted_name_or_explicit_id_does_not_warn(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._write(tmp_path / "case.medh5")
            self._write(tmp_path / "P001.ct.medh5", sample_id="P001.ct")
            self._write(
                tmp_path / "P001.mr.medh5", sample_id="P001.mr", subject_id="P001"
            )
            # An amendment takes its identity from the file.
            with medh5.amend(tmp_path / "P001.mr.medh5") as w:
                w.set_quality("CT", status="approved")
        with medh5.open(tmp_path / "P001.mr.medh5") as sample:
            assert sample.identity.subject_id == "P001"

    def test_S12_1_a_name_that_gives_no_id_is_refused_by_name(self, tmp_path: Path):
        """It failed "[E005] identity requires both sample_id and subject_id",
        naming neither the file nor the cause."""
        path = tmp_path / ".hidden.medh5"
        with pytest.raises(MEDH5ValidationError, match=r"\.hidden\.medh5.*first dot"):
            self._write(path)
        assert not path.exists()
        self._write(path, sample_id="hidden")
        with pytest.raises(MEDH5ValidationError, match="first dot"):
            medh5.pack([path], tmp_path / "s.medh5c")

    def test_S2_2_pack_warns_where_a_default_key_will_change(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        plain = self._write(tmp_path / "a.medh5")
        dotted = self._write(tmp_path / "b.v2.medh5", sample_id="b")
        with pytest.warns(
            FutureWarning, match=r"keys .*b\.v2\.medh5.* as 'b\.v2'"
        ) as caught:
            shard = medh5.pack([plain, dotted], tmp_path / "s.medh5c")
        assert len(caught) == 1 and caught[0].filename == __file__
        with medh5.open_collection(shard) as collection:
            assert sorted(collection.keys()) == ["a", "b"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            medh5.pack([plain, dotted], tmp_path / "t.medh5c", keys=["a", "b"])
        assert (
            main(["pack", str(plain), str(dotted), "-o", str(tmp_path / "u.medh5c")])
            == 0
        )
        note = capsys.readouterr().err
        assert "medh5 3.0 keys this sample 'b.v2'" in note and "a.medh5" not in note


class TestImplicitTimepoint:
    """§3.7 rule 2: with one declared timepoint a grid may omit the attribute."""

    def _write(self, path: Path) -> None:
        with medh5.create(path, sample_id=path.stem) as w:
            w.label_set(Hierarchy.label_set())
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0)
            )  # no timepoint
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_segmentation(
                "lesions",
                grid="g",
                instances=[
                    InstanceInput(class_id=3, instance_id=7, mask=Hierarchy.mask())
                ],
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
        with medh5.open(path) as stored:
            assert "timepoint" not in stored.root["grids/g"].attrs
        with medh5.open(path) as sample:
            assert sample.grids["g"].timepoint == "tp0"
            assert sample.verify().ok
            assert sample.content_id != before  # the extra changed it, not this

    def test_S3_7_declared_timepoints_are_left_as_stored(self, tmp_path: Path) -> None:
        path = tmp_path / "explicit.medh5"
        with medh5.create(path, sample_id="explicit") as w:
            w.add_timepoint("baseline")
            w.add_grid(
                "g",
                shape=Hierarchy.SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint="baseline",
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
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
        with medh5.open(tmp_path / "out.medh5") as stored:
            assert stored.root["grids/ref"].attrs["timepoint"] == "tp0"


class TestThe1xNamesStillAnswer:
    """1.x read helpers that took an ``h5py`` group take the package's own view
    of one (``sample.root[...]``) and answer as they did."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        path = tmp_path / "a.medh5"
        with medh5.create(path, sample_id="a", subject_id="s") as w:
            w.add_grid("g", shape=(4, 4, 4), spacing=(1.0, 1.0, 2.0))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.add_segmentation("seg", grid="g", masks={1: np.ones((4, 4, 4), bool)})
            w.add_transform("t", kind="identity", from_frame="a", to_frame="b")
            w.build_index()
        return path

    def test_documents_grids_and_digests_read_from_a_view(self, path: Path) -> None:
        from medh5.document import read_document, read_document_text
        from medh5.geometry import read_grid, read_grids
        from medh5.integrity import collect_digests

        with medh5.open(path) as sample:
            assert read_document(sample.root).identity.subject_id == "s"
            assert (
                json.loads(read_document_text(sample.root))["identity"]["sample_id"]
                == "a"
            )
            stored = read_grid(sample.root["grids/g"])
            resolved = sample.grids["g"]
            assert (stored.grid_id, stored.shape, stored.spacing) == (
                resolved.grid_id,
                resolved.shape,
                resolved.spacing,
            )
            # The stored group, not the reader's interpretation of it: the one
            # timepoint is resolved by `Sample`, never written back (1.2.1).
            assert stored.timepoint is None and resolved.timepoint == "tp0"
            assert list(read_grids(sample.root)) == ["g"]
            digests = collect_digests(sample.root)
            assert digests["images/CT"] == sample.root["images/CT"].attrs["digest"]
            assert not any(k.startswith("index/") for k in digests)

    def test_headers_read_from_a_group_and_the_index_names_its_own(
        self, path: Path
    ) -> None:
        from medh5.annotations.base import AnnotationHeader
        from medh5.transforms import TransformHeader

        with medh5.open(path) as sample:
            header = AnnotationHeader.read(sample.root["annotations/seg"])
            assert header == sample.annotations["seg"].header
            assert TransformHeader.read(sample.root["transforms/t"]).kind == "identity"
            assert sample.index["seg"].group.name == "/index/seg"

    def test_a_timeline_is_a_sequence_and_a_record_lists_its_fields(
        self, path: Path
    ) -> None:
        from collections.abc import Sequence

        from medh5.curation import Identity

        with medh5.open(path) as sample:
            timeline = sample.timepoints
            assert isinstance(timeline, Sequence)
            assert timeline.index(timeline[0]) == 0 and timeline.count(timeline[0]) == 1
            with pytest.raises(ValueError):
                timeline.index("not a timepoint")
        assert {"sample_id", "subject_id", "extra"} <= set(dir(Identity("x", "y")))

    def test_a_codec_still_gives_h5py_its_keywords(self) -> None:
        from medh5.annotations.voxel import AnnotationPayload
        from medh5.storage import PROFILES

        assert AnnotationPayload.__name__ == "AnnotationPayload"
        assert PROFILES["portable"].image.kwargs() == {
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
        }
        # Blosc2's keywords are hdf5plugin's to give.
        pytest.importorskip("hdf5plugin")
        assert PROFILES["training"].image.kwargs()["compression"] == 32026
