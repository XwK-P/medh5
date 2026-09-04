"""What 1.4.0 changed, held to by the reproductions that found it.

Every test here is one of the second audit's findings, written as the shortest
program that shows it, and every one of them fails on 1.3.0.  The audit that
produced them went after what the *writer* was asked for rather than what it
produced --- an ignore region under an encoding chosen by measurement, a
provenance id a caller picked, an export of a 2-D sample, a SEG written the way
highdicom writes them by default --- and found that the suite covered the
outputs well and the inputs less well.  So the inputs get a module.

Test names cite the finding as well as the clause, because the audit page is
where the reasoning lives and the id is what joins the two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import LabelClass, LabelSet

SHAPE = (8, 12, 12)


def _label_set() -> LabelSet:
    return LabelSet(
        "rel-1.4.0",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver"),
            LabelClass(2, "spleen", "Spleen"),
            LabelClass(3, "lesion", "Lesion"),
        ],
    )


def _blocks() -> dict[int, Any]:
    liver = np.zeros(SHAPE, dtype=bool)
    liver[1:4, 1:6, 1:6] = True
    lesion = np.zeros(SHAPE, dtype=bool)
    lesion[2:3, 2:4, 2:4] = True  # inside the liver, so the classes overlap
    return {1: liver, 3: lesion}


def _ignore() -> Any:
    """The bottom two slices: nobody looked at them."""
    region = np.zeros(SHAPE, dtype=bool)
    region[6:, :, :] = True
    return region


def _writer(path: Path) -> Any:
    w = medh5.create(path, sample_id=path.stem, codec="portable")
    w.label_set(_label_set())
    w.add_grid("g", shape=SHAPE, spacing=(2.0, 0.8, 0.8), timepoint="tp0")
    w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
    return w


class TestW7WriterContracts:
    """F-07, F-09, F-13: what the writer is handed reaches the file, or is refused."""

    @pytest.mark.parametrize(
        "encoding", ["labelmap", "layers", "bitmask", "instances", "probmap", "auto"]
    )
    def test_F07_S7_7_an_ignore_region_survives_every_encoding(
        self, tmp_path: Path, encoding: str
    ):
        """`ignore=` was forwarded to the encoder only under two of six kinds.

        Under `bitmask`, `instances` and `probmap` the array was dropped with
        no in-band value, no sibling mask, no `ignore_mask` attribute and
        `has_ignore_region` False --- and with `encoding="auto"` the caller
        cannot know which branch they are on, so the same call kept the region
        for one cohort and lost it for the next.  Every ignored voxel became a
        verified negative for every annotated class, and W904 could not fire
        because coverage read as complete.
        """
        region = _ignore()
        path = tmp_path / f"{encoding}.medh5"
        with _writer(path) as w:
            act = w.activity("annotate", agent=w.software("t"))
            if encoding == "probmap":
                payload = {
                    "probabilities": {
                        c: m.astype(np.float32) for c, m in _blocks().items()
                    }
                }
            elif encoding == "instances":
                payload = {
                    "instances": [
                        InstanceInput(class_id=c, instance_id=i + 1, mask=m)
                        for i, (c, m) in enumerate(_blocks().items())
                    ]
                }
            elif encoding == "labelmap":
                payload = {"masks": {1: _blocks()[1]}}  # labelmap needs no overlap
            else:
                payload = {"masks": _blocks()}
            kind, _ = w.add_segmentation(
                "seg",
                grid="g",
                encoding="auto" if encoding in ("probmap", "instances") else encoding,
                annotated_classes=[1],
                ignore=region,
                prov=act,
                **payload,
            )

        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.has_ignore_region
            referenced = annotation.header.ignore_mask
            if referenced is None:
                assert kind in ("labelmap", "layers")
                read_back = annotation.ignore_mask()
            else:
                # §7.7's separate-mask form, written by the writer rather than
                # left to the caller: same grid, same provenance, `task="other"`.
                assert referenced == "seg_ignore"
                sibling = sample.annotations[referenced]
                assert sibling.kind == "mask"
                assert sibling.grid_id == "g"
                assert sibling.prov == annotation.prov
                read_back = sibling.read()
            assert np.array_equal(read_back, region)

    def test_F07_S7_7_the_sibling_mask_validates_and_silences_W904(
        self, tmp_path: Path
    ):
        from medh5.validate import validate_file

        path = tmp_path / "cover.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks=_blocks(),
                encoding="bitmask",
                annotated_classes=[1],
                ignore=_ignore(),
            )
        report = validate_file(path, level="strict")
        assert "W904" not in report.codes
        assert not [c for c in report.codes if c.startswith("E")]

    def test_F07_ignore_and_ignore_mask_together_are_refused(self, tmp_path: Path):
        """Two sources of one fact cannot be kept in agreement."""
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            _writer(tmp_path / "both.medh5") as w,
        ):
            w.add_mask("mine", _ignore(), grid="g")
            w.add_segmentation(
                "seg",
                grid="g",
                masks=_blocks(),
                encoding="bitmask",
                ignore=_ignore(),
                ignore_mask="mine",
            )
        assert exc.value.code == "E404"

    def test_F07_encode_voxels_refuses_what_one_payload_cannot_carry(self):
        """The payload-level entry point had the identical silent branch.

        It returns one payload and so cannot create the sibling mask; refusing
        is the same answer `transcode` already gave in the other direction.
        """
        from medh5.annotations.voxel import encode_voxels

        with pytest.raises(MEDH5ValidationError) as exc:
            encode_voxels(_blocks(), SHAPE, encoding="bitmask", ignore=_ignore())
        assert exc.value.code == "E404"
        payload, _ = encode_voxels(
            _blocks(), SHAPE, encoding="layers", ignore=_ignore()
        )
        assert payload.kind == "layers"

    def test_F07_S7_6_selection_costs_the_widening_an_ignore_forces(self):
        """An in-band ignore forces `uint16` planes, so the choice must see it."""
        from medh5.annotations.voxel import select_encoding

        masks = {c: np.zeros(SHAPE, dtype=bool) for c in range(1, 4)}
        for c, mask in masks.items():
            mask[c] = True
        _, plain = select_encoding(masks, SHAPE)
        _, widened = select_encoding(masks, SHAPE, ignore=True)
        from medh5.annotations.voxel.select import cost_model

        assert cost_model(widened, ignore=True).layers > cost_model(plain).layers

    def test_F09_S11_1_a_provenance_id_is_never_silently_overwritten(
        self, tmp_path: Path
    ):
        """`person("Alice", agent_id="s2")` then `software("tool")` left one agent.

        The automatic ids are `<type initial><n>` and `act_<type>_<n>`, which
        is exactly what a caller who named a node explicitly is likely to have
        used.  Assigning into the dict meant the second write won, every
        reference resolved, and it resolved to the wrong node.
        """
        path = tmp_path / "prov.medh5"
        with _writer(path) as w:
            alice = w.person("Alice", agent_id="s2")
            tool = w.software("tool")
            assert tool.id != alice.id
            imported = w.activity("import", activity_id="act_annotate_2", agent=tool)
            annotated = w.activity("annotate", agent=tool)
            assert annotated.id != imported.id
            w.add_segmentation("seg", grid="g", masks=_blocks(), prov=imported)

        with medh5.open(path) as sample:
            provenance = sample.document.provenance
            assert {a.id for a in provenance.agents} == {"s2", "s3"}
            assert provenance.agent("s2").name == "Alice"
            assert (
                provenance.activity(sample.annotations["seg"].prov or "").type
                == "import"
            )

    def test_F09_an_explicit_duplicate_id_raises(self, tmp_path: Path):
        with _writer(tmp_path / "dup.medh5") as w:
            alice = w.person("Alice", agent_id="s2")
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.person("Bob", agent_id="s2")
            w.activity("import", activity_id="act1", agent=alice)
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.activity("annotate", activity_id="act1", agent=alice)
            w.add_segmentation("seg", grid="g", masks=_blocks())

    def test_F09_replace_is_available_for_the_one_legitimate_rewrite(self):
        from medh5.curation.provenance import Agent, Provenance

        graph = Provenance(agents=[Agent("a1", "person", "Alice")])
        graph.add_agent(Agent("a1", "person", "pseudo:abc"), replace=True)
        assert graph.agent("a1").name == "pseudo:abc"

    def test_F13_S7_6_transcoding_to_mask_is_refused(self, tmp_path: Path):
        """`mask` has no classes, so the conversion erased the coverage contract.

        Every class was OR-ed into one volume, `class_ids` and
        `annotated_class_ids` came out empty under `task="segmentation"`, and
        the result validated clean.  The CLI was safe because `seg convert`
        restricts its choices; the Python API was not.
        """
        path = tmp_path / "mask.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks=_blocks(), annotated_classes=[1, 3]
            )
        with (
            medh5.amend(path) as w,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            w.transcode_annotation("seg", "mask")
        assert exc.value.code == "E404"
        assert "encode_mask" in str(exc.value)

        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.class_ids == (1, 3)
            assert annotation.annotated_class_ids == (1, 3)

    def test_F13_TRANSCODABLE_is_the_whole_truth(self):
        from medh5.annotations.voxel import encode_masks
        from medh5.annotations.voxel.transcode import TRANSCODABLE, transcode_payload

        assert "mask" not in TRANSCODABLE
        payload = encode_masks(_blocks(), "layers", SHAPE)
        with pytest.raises(MEDH5ValidationError):
            transcode_payload(payload, "mask", spatial_shape=SHAPE)


class TestW8Deidentification:
    """F-08: the attestation, the scope it covers, and the exit code."""

    def _dirty(self, path: Path) -> Path:
        from medh5.curation.quality import Issue

        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.identity(PatientName="Doe^Jane")
            w.cohort(
                dataset_id="d",
                InstitutionName="St Elsewhere",
                ReferringPhysicianName="Smith^John",
            )
            w.add_timepoint("tp0", date="2026-02-03", study_uid="1.2.840.113619.2.55")
            w.add_grid(
                "g",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                frame_uid="1.2.840.10008.5.1.4.1.1.2",
            )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            tool = w.software("conv")
            w.person("Brown^Ann")
            w.activity(
                "import", agent=tool, tool="ctp", params={"OperatorsName": "Lee^Bo"}
            )
            w.acquisition("CT", PatientID="MRN-1")
            w.set_quality(
                "q", status="draft", issues=[Issue(code="x", note="Green^Sam")]
            )
            w.extra("dicom", {"PatientID": "MRN-1"})
        return path

    def test_F08_S11_4_apply_acts_on_everything_it_flags(self, tmp_path: Path):
        """A strict apply used to leave three identifiers it had itself flagged.

        `apply` cleaned `extra` and `acquisition` only, then wrote a record
        whose profile string reads "quasi-identifiers removed" and exited 0 ---
        over a file still carrying a patient name, an institution and a
        referring physician, in `identity.extra` and `cohort`.
        """
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "dirty.medh5")
        before = scrub.scan(path, profile="strict")
        flagged = {f.where for f in before.actionable}
        assert {
            "identity.extra.PatientName",
            "cohort.InstitutionName",
            "cohort.ReferringPhysicianName",
            "provenance.activities[act_import_1].params.OperatorsName",
            "quality.q.issues[0].note",
            "provenance.agents[p2]",
        } <= flagged

        report = scrub.apply(path, profile="strict", date_shift_days=-30, salt="pep")
        assert report.applied
        assert not report.remaining_actionable, report.remaining
        assert report.ok

        # The independent re-scan, which is the check the tool now makes itself.
        assert not scrub.scan(path, profile="strict").actionable

    def test_F08_the_record_states_what_the_run_achieved(self, tmp_path: Path):
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "record.medh5")
        scrub.apply(path, profile="strict", date_shift_days=-30)
        with medh5.open(path) as sample:
            activity = sample.document.provenance.activities_by_type("deidentify")[0]
            assert activity.params["remaining_actionable"] == 0
            assert activity.params["changes"] >= activity.params["remaining"]

    def test_F08_apply_is_idempotent_and_stays_green(self, tmp_path: Path):
        """A rule that fires on its own output can never go green in a pipeline."""
        from medh5.curation import scrub

        path = self._dirty(tmp_path / "twice.medh5")
        first = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        second = scrub.apply(path, profile="strict", date_shift_days=-30, salt="p")
        assert first.ok and second.ok
        assert not second.remaining_actionable

    def test_F08_the_ids_a_join_needs_are_reported_and_never_rewritten(
        self, tmp_path: Path
    ):
        from medh5.curation import scrub
        from medh5.curation.scrub import UNFIXABLE_LOCATIONS

        path = tmp_path / "named.medh5"
        with medh5.create(path, sample_id="s1", subject_id="Doe^Jane") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        report = scrub.scan(path, profile="strict")
        named = [f for f in report.findings if f.where == "identity.subject_id"]
        assert named and not named[0].actionable
        assert "identity.subject_id" in UNFIXABLE_LOCATIONS
        scrub.apply(path, profile="strict")
        with medh5.open(path) as sample:
            assert sample.identity.subject_id == "Doe^Jane"

    def test_F08_the_cli_exit_code_follows_the_re_scan(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._dirty(tmp_path / "cli.medh5")
        assert main(["scrub", str(path)]) == 1
        assert (
            main(["scrub", str(path), "--apply", "--profile", "strict", "--by", "R7"])
            == 0
        )
        capsys.readouterr()
        assert main(["scrub", str(path), "--profile", "strict"]) == 0


class TestW9Converters:
    """F-10, F-11, F-12, L-18: value- and dimension-preserving, both ways."""

    def test_F10_S4_2_a_rescaled_image_exports_with_its_scale(self, tmp_path: Path):
        """`image.read()` returns stored values; the exporters wrote them bare.

        A CT imported from DICOM with intercept −1024 exported every voxel 1024
        HU too high --- in the nnU-Net exporter's only path, and in
        `to_nifti --stored` --- with nothing in `dataset.json` or the report to
        say so.
        """
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import to_nifti

        path = tmp_path / "ct.medh5"
        with medh5.create(path, sample_id="ct", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(2.0, 0.8, 0.9))
            w.add_image(
                "CT",
                np.full(SHAPE, 1000, np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
            )
        out = to_nifti(path, "CT", tmp_path / "stored.nii.gz", physical=False)
        image = nib.load(str(out))
        assert image.get_data_dtype() == np.int16
        assert np.asanyarray(image.dataobj.get_unscaled()).flat[0] == 1000
        assert image.get_fdata().flat[0] == pytest.approx(-24.0)

    def test_F10_an_nnunet_export_keeps_the_physical_values(self, tmp_path: Path):
        nib = pytest.importorskip("nibabel")
        from medh5.io.nnunetv2 import to_nnunetv2

        path = tmp_path / "ct.medh5"
        with medh5.create(path, sample_id="ct", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(2.0, 0.8, 0.9))
            w.add_image(
                "CT",
                np.full(SHAPE, 1000, np.int16),
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

    def test_F11_S3_6_a_2D_sample_round_trips_through_NIfTI(self, tmp_path: Path):
        """Both importers take 2-D input; both exporters stopped on a 3x3 affine.

        `convert_world` was defined for 4x4 only, so an imported radiograph
        could never be written back out --- the round trip the converters page
        promises was broken for every 2-D sample.
        """
        nib = pytest.importorskip("nibabel")
        from medh5.io.nifti import from_nifti, to_nifti

        affine = np.array(
            [[0.8, 0, 0, -1.0], [0, 0.9, 0, -2.0], [0, 0, 3.0, 5.0], [0, 0, 0, 1]]
        )
        data = np.arange(32 * 40, dtype=np.int16).reshape(32, 40)
        source = tmp_path / "xray.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(source))

        from_nifti({"XR": source}, tmp_path / "xray.medh5")
        with medh5.open(tmp_path / "xray.medh5") as sample:
            first_affine = np.asarray(sample.grids["ref"].affine)
            first_voxels = sample.images["XR"].read()
        assert first_affine.shape == (3, 3)

        exported = to_nifti(tmp_path / "xray.medh5", "XR", tmp_path / "back.nii.gz")
        from_nifti({"XR": exported}, tmp_path / "again.medh5")
        with medh5.open(tmp_path / "again.medh5") as sample:
            assert np.array_equal(sample.images["XR"].read(), first_voxels)
            assert np.allclose(np.asarray(sample.grids["ref"].affine), first_affine)
            assert sample.grids["ref"].shape == (32, 40)

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

    def test_F12_S3_3_a_seg_with_omitted_frames_imports(self, tmp_path: Path):
        """highdicom's default omits empty frames, and the import refused it.

        The reader assembled a volume from the planes present --- two slices,
        spaced by the distance between them --- and `from_dicom_seg` compared
        that shape with the grid's and reported E405 "drawn on a different
        reconstruction".  Frames carry their own position; they are placed.
        """
        hd = pytest.importorskip("highdicom")
        pydicom = pytest.importorskip("pydicom")
        from pydicom.uid import generate_uid

        from medh5.io.dicom import from_dicom
        from medh5.io.dicom_seg import from_dicom_seg
        from tests.v1.conftest import write_dicom_series

        root = tmp_path / "dcm"
        series = write_dicom_series(
            root,
            patient_id="p",
            study_uid=generate_uid(),
            study_date="20260101",
            shape=(8, 16, 20),
        )
        path = tmp_path / "case.medh5"
        from_dicom(root, path, group_by="study")
        with medh5.open(path) as sample:
            shape = sample.grids[sorted(sample.grids)[0]].spatial_shape

        datasets = [pydicom.dcmread(p) for p in series["paths"]]
        normal = np.cross([0, 0, 1], [0, 1, 0])
        datasets.sort(
            key=lambda d: float(
                np.dot([float(v) for v in d.ImagePositionPatient], normal)
            )
        )
        mask = np.zeros((*shape, 1), np.uint8)
        mask[3, 4:8, 5:9, 0] = 1
        mask[4, 4:8, 5:9, 0] = 1  # two non-adjacent-from-the-edges slices
        description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label="liver",
            segmented_property_category=hd.sr.CodedConcept(
                "91723000", "SCT", "Anatomical Structure"
            ),
            segmented_property_type=hd.sr.CodedConcept("10200004", "SCT", "Liver"),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=mask,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[description],
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="t",
            manufacturer_model_name="t",
            software_versions="1",
            device_serial_number="0",
            omit_empty_frames=True,
        )
        out = tmp_path / "omit.dcm"
        segmentation.save_as(str(out))

        report = from_dicom_seg(out, path, ann_id="imported")
        assert report.of_kind("frames")
        with medh5.open(path) as sample:
            dense = sample.annotations["imported"].dense(["liver"])[0]
            assert dense.shape == shape
            counts = [int(dense[k].sum()) for k in range(shape[0])]
            assert counts == [int(mask[k, ..., 0].sum()) for k in range(shape[0])]

    def test_F12_a_frame_that_is_not_a_slice_is_refused_by_name(self, tmp_path: Path):
        """The refusal stays, and says which frame and why."""
        from medh5.io.dicom_seg import place_frames

        path = tmp_path / "grid.medh5"
        with _writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=_blocks())
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            geometry = {
                "rows": SHAPE[1],
                "columns": SHAPE[2],
                "segments": {1: {"label": "liver"}},
                "fractional": False,
            }
            good = [
                {
                    "index": 0,
                    "segment": 1,
                    "position": grid.index_to_world([2, 0, 0]),
                    "data": np.ones(SHAPE[1:], bool),
                }
            ]
            placed = place_frames(good, geometry, grid)
            assert placed[1][2].all() and not placed[1][3].any()

            off = [
                {
                    "index": 7,
                    "segment": 1,
                    "position": grid.index_to_world([2.5, 0, 0]),
                    "data": np.ones(SHAPE[1:], bool),
                }
            ]
            with pytest.raises(MEDH5ValidationError) as exc:
                place_frames(off, geometry, grid)
            assert exc.value.code == "E405"
            assert "frame 7" in str(exc.value)

    def test_L18_the_report_lists_only_files_that_were_written(self, tmp_path: Path):
        pytest.importorskip("nibabel")
        from medh5.io.nnunetv2 import to_nnunetv2

        path = tmp_path / "bare.medh5"
        with medh5.create(path, sample_id="bare", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        report = to_nnunetv2([path], tmp_path / "nn", dataset_name="D3")
        for output in report.outputs:
            assert Path(output).exists(), output
        assert not any("labelsTr" in o for o in report.outputs)


class TestW10Integrity:
    """L-12, L-17: a check that checks, and an amend that stays addressed."""

    def _sample(self, path: Path) -> Path:
        with _writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=_blocks())
        return path

    def test_L12_S13_1_recompress_verifies_what_it_wrote(self, tmp_path: Path):
        """`content_id_preserved` compared the attribute it had just copied.

        True by construction --- including on a file whose bytes were corrupted
        before the run, which reported "content_id yes", exited 0, and failed
        `verify()` immediately afterwards.
        """
        import h5py

        from medh5.storage.recompress import recompress

        clean = self._sample(tmp_path / "clean.medh5")
        result = recompress(clean, "archive")
        assert result.verified and result.content_id_preserved and result.ok

        broken = self._sample(tmp_path / "broken.medh5")
        with h5py.File(broken, "r+") as handle:
            node = handle["images/CT"]
            data = node[...]
            data[0, 0, 0] += 7
            node[...] = data
        result = recompress(broken, "archive")
        assert not result.verified
        assert not result.ok
        assert "images/CT" in result.mismatched
        with medh5.open(broken) as sample:
            assert not sample.verify().ok

    def test_L12_the_cli_exit_code_follows_the_verification(
        self, tmp_path: Path, capsys
    ):
        import h5py

        from medh5.cli import main

        broken = self._sample(tmp_path / "cli.medh5")
        with h5py.File(broken, "r+") as handle:
            node = handle["images/CT"]
            data = node[...]
            data[1, 1, 1] += 3
            node[...] = data
        assert main(["recompress", str(broken), "--profile", "portable"]) == 1
        assert "FAILED" in capsys.readouterr().out

    def test_L17_S13_2_an_amend_without_digests_stays_addressed(self, tmp_path: Path):
        """`digests=False` dropped `content_id` and left the file unaddressed.

        The flag exists to skip re-reading what copy-on-write already brought
        across with its digest intact; it was never meant to un-address the
        sample, and every cache keyed on the root missed afterwards.
        """
        path = self._sample(tmp_path / "amend.medh5")
        with medh5.amend(path) as w:
            w.set_quality("seg", status="approved")
            w.commit(digests=False)
        with medh5.open(path) as sample:
            assert sample.content_id is not None
            assert sample.content_id == sample.compute_content_id()
            result = sample.verify()
            assert result.ok
            assert not result.undigested

    def test_L17_a_fresh_file_is_fully_stamped_either_way(self, tmp_path: Path):
        path = tmp_path / "fresh.medh5"
        writer = _writer(path)
        writer.add_segmentation("seg", grid="g", masks=_blocks())
        writer.commit(digests=False)
        with medh5.open(path) as sample:
            assert not sample.verify().undigested
            assert sample.content_id is not None


class TestW11Performance:
    """P-05, P-06, P-07: the same answers, without the cliff."""

    def test_P05_the_class_table_is_read_once_per_annotation(self, tmp_path: Path):
        """`dense()` asked the table once per class: 63 reads for 63 classes."""
        import h5py

        masks = {c: np.zeros(SHAPE, dtype=bool) for c in range(1, 4)}
        for c, mask in masks.items():
            mask[c, c:, c:] = True
        for encoding, table in (
            ("layers", "layer_class_ids"),
            ("bitmask", "bit_class_ids"),
        ):
            path = tmp_path / f"{encoding}.medh5"
            with _writer(path) as w:
                w.add_segmentation("seg", grid="g", masks=masks, encoding=encoding)

            reads = 0
            original = h5py.Dataset.__getitem__

            def counting(
                self: Any, key: Any, _name: str = table, _original: Any = original
            ) -> Any:
                nonlocal reads
                if self.name.endswith(_name):
                    reads += 1
                return _original(self, key)

            with medh5.open(path) as sample:
                annotation = sample.annotations["seg"]
                h5py.Dataset.__getitem__ = counting  # type: ignore[method-assign]
                try:
                    annotation.dense(list(annotation.class_ids))
                finally:
                    h5py.Dataset.__getitem__ = original  # type: ignore[method-assign]
            assert reads <= 1, f"{encoding}: {reads} reads of {table}"

    def test_P06_the_occupancy_map_equals_the_loop_it_replaced(self):
        from medh5.storage.index import _occupancy

        rng = np.random.default_rng(0)
        for shape in [(9, 7, 5), (16, 16, 16), (33, 17, 8)]:
            mask = rng.random(shape) > 0.9
            coarse = tuple(max(1, -(-n // 8)) for n in shape)
            expected = np.zeros(coarse, dtype=bool)
            for block in np.ndindex(*coarse):
                window = tuple(
                    slice(i * 8, min((i + 1) * 8, n))
                    for i, n in zip(block, shape, strict=True)
                )
                expected[block] = bool(mask[window].any())
            assert np.array_equal(_occupancy(mask, 8), expected), shape

    @pytest.mark.parametrize(
        "encoding", ["labelmap", "layers", "bitmask", "instances", "probmap"]
    )
    def test_P07_voxel_counts_agree_with_the_generic_path(
        self, tmp_path: Path, encoding: str
    ):
        path = tmp_path / f"{encoding}.medh5"
        blocks = _blocks()
        with _writer(path) as w:
            if encoding == "probmap":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    probabilities={c: m.astype(np.float32) for c, m in blocks.items()},
                )
            elif encoding == "instances":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    instances=[
                        InstanceInput(class_id=c, instance_id=i + 1, mask=m)
                        for i, (c, m) in enumerate(blocks.items())
                    ],
                )
            elif encoding == "labelmap":
                w.add_segmentation(
                    "seg", grid="g", masks={1: blocks[1]}, encoding="labelmap"
                )
            else:
                w.add_segmentation("seg", grid="g", masks=blocks, encoding=encoding)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            fast = annotation.voxel_counts()
            slow = {
                c: int(annotation.dense([c])[0].sum()) for c in annotation.class_ids
            }
            assert fast == slow

    def test_P07_an_empty_class_still_counts_zero(self, tmp_path: Path):
        """The coverage contract: examined and absent is not the same as missing."""
        path = tmp_path / "empty.medh5"
        with _writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks=_blocks(),
                encoding="layers",
                annotated_classes=[1, 2, 3],
            )
        with medh5.open(path) as sample:
            counts = sample.annotations["seg"].voxel_counts()
            assert counts[2] == 0
            assert set(counts) == {1, 2, 3}


class TestW12ApiAndCli:
    """L-13, L-14, L-15, L-16, Q-10: the loose ends."""

    def test_L13_open_collection_takes_no_mode(self, tmp_path: Path):
        import inspect

        from medh5.collection import open_collection, pack

        assert "mode" not in inspect.signature(open_collection).parameters
        path = tmp_path / "one.medh5"
        with _writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=_blocks())
        shard = tmp_path / "s.medh5c"
        pack([path], shard)
        with open_collection(shard) as collection:
            assert sorted(collection) == ["one"]

    @pytest.mark.parametrize(
        ("build", "what"),
        [
            (
                lambda: __import__(
                    "medh5.curation.timeline", fromlist=["Timepoint"]
                ).Timepoint.from_json({"id": "tp0", "index": 0, "x": 1}),
                "timepoint",
            ),
            (
                lambda: __import__(
                    "medh5.curation.quality", fromlist=["QualityRecord"]
                ).QualityRecord.from_json({"status": "draft", "x": 1}),
                "quality record",
            ),
            (
                lambda: __import__(
                    "medh5.curation.provenance", fromlist=["Activity"]
                ).Activity.from_json({"id": "a", "type": "import", "x": 1}),
                "activity",
            ),
            (
                lambda: __import__(
                    "medh5.curation.provenance", fromlist=["Agent"]
                ).Agent.from_json({"id": "a", "type": "software", "name": "m", "x": 1}),
                "agent",
            ),
        ],
    )
    def test_L14_S2_4_a_closed_object_refuses_an_unknown_key(
        self, build: Any, what: str
    ):
        """Four objects are `additionalProperties: false`, and three had `extra`.

        Anything placed there failed E005 at `commit()`, so the mapping was an
        extension point that could not reach a file.  The refusal moves to the
        call that introduces the value, with the field named.
        """
        with pytest.raises(MEDH5ValidationError) as exc:
            build()
        assert exc.value.code == "E005"
        assert what in str(exc.value)

    def test_L14_an_agent_carries_its_organization_as_a_field(self):
        from medh5.curation.provenance import Agent

        agent = Agent.from_json(
            {"id": "p1", "type": "person", "name": "R7", "organization": "org1"}
        )
        assert agent.organization == "org1"
        assert agent.to_json()["organization"] == "org1"

    def test_L15_the_inspection_commands_take_a_shard_with_a_key(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import main
        from medh5.collection import pack

        members = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.medh5"
            with _writer(path) as w:
                w.add_segmentation("seg", grid="g", masks=_blocks())
            members.append(path)
        shard = tmp_path / "s.medh5c"
        pack(members, shard)

        assert main(["info", str(shard)]) == 0
        assert "per-sample detail" in capsys.readouterr().out
        for command in ("info", "tree", "verify", "timeline", "track"):
            assert main([command, str(shard), "--key", "a"]) == 0, command
            capsys.readouterr()
        # A per-sample command with no key says which keys there are.
        assert main(["timeline", str(shard)]) == 1
        assert "--key" in capsys.readouterr().err

    def test_L16_C202_groups_by_the_grouping_key(self, tmp_path: Path):
        """`dataset check` grouped by subject; `medh5 splits` by the §12.2 key.

        A family or longitudinal group straddling two partitions was a LEAK in
        one tool and clean in the other, on the same files.
        """
        from medh5.dataset.check import check
        from medh5.dataset.manifest import scan

        root = tmp_path / "cohort"
        root.mkdir()
        for name, subject, partition in (
            ("a", "subj-A", "train"),
            ("b", "subj-B", "test"),
        ):
            path = root / f"{name}.medh5"
            with medh5.create(path, sample_id=name, subject_id=subject) as w:
                w.cohort(group_id="family-7")
                w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
                w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
                w.split(set_id="cv5", partition=partition)
        manifest, _ = scan(root)
        report = check(manifest)
        finding = next(f for f in report.errors if f.code == "C202")
        assert "family-7" in finding.where
        assert "subj-A" in finding.message and "subj-B" in finding.message

    def test_Q10_S3_2_a_file_with_no_grids_raises_E111(self, tmp_path: Path):
        """`reference_grid` ended in `grids[sorted(grids)[0]]` --- a bare IndexError."""
        import h5py

        path = tmp_path / "nogrid.medh5"
        with _writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=_blocks())
        with h5py.File(path, "r+") as handle:
            del handle["images"]["CT"]
            del handle["grids"]["g"]
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                _ = sample.reference_grid
            assert exc.value.code == "E111"
