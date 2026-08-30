"""Writing, reading, amending and the timepoint model (spec §2, §3.7, §14.4)."""

from __future__ import annotations

import json
import stat

import h5py
import numpy as np
import pytest

import medh5
from medh5._hdf5 import as_str, encode_attr, str_dtype
from medh5.annotations.voxel import InstanceInput
from medh5.curation.timeline import Timeline, Timepoint
from medh5.errors import (
    MEDH5Error,
    MEDH5FileError,
    MEDH5ValidationError,
    MEDH5VersionError,
)
from medh5.validate import validate_file
from tests.v1.conftest import SHAPE, write_sample


def minimal(writer, *, grid: str = "g") -> None:
    """The two calls every sample needs, so a test can add just its own point."""
    writer.add_grid(grid, shape=SHAPE, spacing=(1.0, 1.0, 1.0))
    writer.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid=grid, modality="CT")


class TestWriteRead:
    def test_round_trip(self, sample_path, masks):
        with medh5.open(sample_path) as sample:
            assert sample.identity.sample_id == "case"
            assert sample.identity.subject_id == "subj-A"
            assert sample.version == medh5.FORMAT_VERSION
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
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "empty.medh5") as w,
        ):
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        assert exc.value.code == "E201"

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

    def test_roi_accepts_spatial_or_full_rank(self, sample_path):
        with medh5.open(sample_path) as sample:
            image = sample.images["CT_tp0"]
            assert image.read(np.s_[0:4, 0:4, 0:4]).shape == (4, 4, 4)
            with pytest.raises(MEDH5ValidationError):
                image.read([slice(0, 4)])

    def test_S4_3_pyramid_levels_are_addressable(self, tmp_path):
        from medh5.geometry.multiscale import derive_level_grid

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
            # and `profiles` is still derived rather than inherited.
            assert handle.attrs["medh5_version"] == medh5.__format_version__
            assert "images" in handle and "CT2" in handle["images"]

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
