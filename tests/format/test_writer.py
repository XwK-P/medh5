"""What the writer accepts, refuses and checks at every door (spec §14.4, §15).

The writer is held to the validator: a rule the validator enforces is refused
at the call or at commit, and every tool that rewrites a file --- amend, scrub,
fix, recompress, unpack --- passes the same gate.  A test is named for the
clause it holds; the changelog says which release made the writer hold it.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.geometric import Polygon
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5FileError, MEDH5ValidationError, MEDH5VersionError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file
from tests.helpers import LATER_MINOR, attested_as_written
from tests.kits import Flat, Framed, Numbered, Organs

# Each door a boolean volume comes in by, writing it into annotation `a`, and
# what reads it back.
_BOOLEAN_VOLUMES: dict[str, tuple[Any, Any]] = {
    "add_mask": (
        lambda w, volume: w.add_mask("a", volume, grid="g"),
        lambda sample: sample.annotations["a"].read(),
    ),
    "masks": (
        lambda w, volume: w.add_segmentation("a", grid="g", masks={1: volume}),
        lambda sample: sample.annotations["a"].dense([1])[0],
    ),
    "ignore": (
        lambda w, volume: w.add_segmentation(
            "a", grid="g", masks={3: Flat.mask((5, 5, 5))}, ignore=volume
        ),
        lambda sample: sample.ignore_region("a"),
    ),
    "instance": (
        lambda w, volume: w.add_segmentation(
            "a", grid="g", instances=[InstanceInput(1, 1, mask=volume)]
        ),
        lambda sample: sample.annotations["a"].dense([1])[0],
    ),
}


class TestWriterSideRefusals:
    """Every new validator rule is refused by the writer at commit as well."""

    def _refused(self, tmp_path: Path, code: str, build: Any) -> None:
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Flat.open_writer(tmp_path / f"{code}.medh5") as w,
        ):
            build(w)
        assert exc.value.code == code, exc.value

    def test_S7_7_ignore_mask_must_exist(self, tmp_path: Path):
        self._refused(
            tmp_path,
            "E413",
            lambda w: w.add_segmentation(
                "s", grid="g", masks={3: Flat.mask()}, ignore_mask="nope"
            ),
        )

    def test_S7_7_ignore_mask_must_be_a_mask_on_the_same_grid(self, tmp_path: Path):
        def not_a_mask(w: Any) -> None:
            w.add_segmentation("other", grid="g", masks={1: Flat.mask()})
            w.add_segmentation(
                "s", grid="g", masks={3: Flat.mask()}, ignore_mask="other"
            )

        self._refused(tmp_path, "E413", not_a_mask)

        def other_grid(w: Any) -> None:
            w.add_grid("h", shape=Flat.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_mask("fov", np.ones(Flat.SHAPE, dtype=bool), grid="h")
            w.add_segmentation("s", grid="g", masks={3: Flat.mask()}, ignore_mask="fov")

        self._refused(tmp_path, "E413", other_grid)

    def test_S7_7_a_correct_ignore_mask_is_accepted(self, tmp_path: Path):
        path = tmp_path / "ok.medh5"
        with Flat.open_writer(path) as w:
            w.add_mask("uncertain", Flat.mask((5, 5, 5), 2), grid="g")
            w.add_segmentation(
                "s", grid="g", masks={3: Flat.mask()}, ignore_mask="uncertain"
            )
        from medh5.validate import validate_file

        assert "E413" not in validate_file(path, level="strict").codes

    def test_S6_2_derived_from_must_exist(self, tmp_path: Path):
        self._refused(
            tmp_path,
            "E413",
            lambda w: w.add_segmentation(
                "s", grid="g", masks={3: Flat.mask()}, derived_from=["ghost"]
            ),
        )

    def test_S6_2_derived_from_accepts_the_path_spelling(self, tmp_path: Path):
        path = tmp_path / "paths.medh5"
        with Flat.open_writer(path) as w:
            w.add_segmentation("a", grid="g", masks={3: Flat.mask()})
            w.add_segmentation(
                "b", grid="g", masks={3: Flat.mask()}, derived_from=["annotations/a"]
            )
        from medh5.validate import validate_file

        assert "E413" not in validate_file(path).codes

    def test_S4_4_valid_mask_must_be_a_mask(self, tmp_path: Path):
        path = tmp_path / "vm.medh5"
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(path, sample_id="vm") as w,
        ):
            w.add_grid("g", shape=Flat.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", Flat.image(), grid="g", modality="CT", valid_mask="nope")
        assert exc.value.code == "E413"

    @pytest.mark.parametrize("door", sorted(_BOOLEAN_VOLUMES))
    def test_S4_4_a_float_mask_holds_only_0_and_1(self, tmp_path: Path, door: str):
        """Every boolean volume a writer takes was read as set wherever it was
        non-zero, so a soft value --- 0.2 --- became `True`: a threshold
        nobody chose, stored as which voxels are in the mask.  A float array
        of 0s and 1s is still a mask."""
        write, read = _BOOLEAN_VOLUMES[door]
        soft = Flat.mask().astype(np.float32) * np.float32(0.2)
        self._refused(tmp_path, "E411", lambda w: write(w, soft))

        path = tmp_path / "binary.medh5"
        with Flat.open_writer(path) as w:
            write(w, Flat.mask().astype(np.float64))
        with medh5.open(path) as sample:
            np.testing.assert_array_equal(read(sample), Flat.mask())
        assert validate_file(path).ok

    def test_S10_1_transform_links_are_checked(self, tmp_path: Path):
        def dangling_prov(w: Any) -> None:
            w.add_transform(
                "t",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=np.eye(4),
                prov="act_nope",
            )

        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Flat.open_writer(tmp_path / "p.medh5", frames=True) as w,
        ):
            dangling_prov(w)
        assert exc.value.code == "E601"

        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Flat.open_writer(tmp_path / "m.medh5", frames=True) as w,
        ):
            w.add_transform(
                "t",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=np.eye(4),
                metrics="nope",
            )
        assert exc.value.code == "E602"

    def test_S12_3_and_S11_4_timestamps_are_RFC_3339(self):
        from medh5.curation import Deidentification, SplitClaim

        with pytest.raises(MEDH5ValidationError) as exc:
            SplitClaim(set_id="cv", partition="train", assigned_at="yesterday")
        assert exc.value.code == "E604"
        with pytest.raises(MEDH5ValidationError) as exc:
            Deidentification(method="m", date="yesterday")
        assert exc.value.code == "E604"
        assert SplitClaim(
            set_id="cv", partition="train", assigned_at="2026-09-04T10:00:00Z"
        )

    def test_S3_2_a_time_axis_carries_its_timings(self, tmp_path: Path):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "t.medh5", sample_id="t") as w,
        ):
            w.add_grid(
                "dce",
                shape=(3, *Flat.SHAPE),
                spacing=(1.0, 1.0, 1.0),
                axis_kinds=("time", "spatial", "spatial", "spatial"),
                axis_names=("t", "z", "y", "x"),
                timepoint="tp0",
            )
        assert exc.value.code == "E109"


class TestImageAndGridEdits:
    """Images and grids are removed, and an image's attributes edited, as
    annotations and transforms are: what still names them, and the last one
    going, are the commit's to refuse."""

    @staticmethod
    def _two_images(path: Path) -> Path:
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_timepoint("tp0", series_uids={"CT": "1.2.3.1", "PT": "1.2.3.2"})
            w.add_grid("g", shape=Flat.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_grid("p", shape=(4, 5, 5), spacing=(2.0, 2.0, 2.0), timepoint="tp0")
            w.add_image("CT", Flat.image(), grid="g", modality="CT")
            w.add_image(
                "PT",
                np.zeros((4, 5, 5), np.float32),
                grid="p",
                modality="PT",
                value_units="marker-units-7d1c",
            )
            w.acquisition("PT", tracer="FDG")
        return path

    def test_S14_4_remove_image_leaves_no_bytes_behind(self, tmp_path: Path):
        path = self._two_images(tmp_path / "s.medh5")
        assert b"marker-units-7d1c" in path.read_bytes(), "the control"
        with medh5.amend(path) as w:
            w.remove_image("PT")
            w.remove_grid("p")
        assert b"marker-units-7d1c" not in path.read_bytes()
        with medh5.open(path) as sample:
            assert sorted(sample.images) == ["CT"]
            assert sorted(sample.grids) == ["g"]
            assert "PT" not in sample.document.acquisition
            assert dict(sample.timepoints["tp0"].series_uids) == {"CT": "1.2.3.1"}
        assert validate_file(path, level="integrity").ok

    def test_S14_4_removals_may_come_in_any_order(self, tmp_path: Path):
        """The grid goes before the image that names it, and the last image
        goes before its replacement arrives: only the commit judges."""
        path = self._two_images(tmp_path / "s.medh5")
        with medh5.amend(path) as w:
            w.remove_grid("p")
            w.remove_image("PT")
            w.remove_image("CT")
            w.add_image("CT", Flat.image(), grid="g", modality="MR")
        with medh5.open(path) as sample:
            assert [(i, sample.images[i].modality) for i in sample.images] == [
                ("CT", "MR")
            ]
        for remove, missing in (("remove_image", "PT"), ("remove_grid", "q")):
            with (
                pytest.raises(MEDH5ValidationError, match=missing),
                medh5.amend(path) as w,
            ):
                getattr(w, remove)(missing)

    def test_S3_2_remove_grid_is_refused_at_commit_while_named(self, tmp_path: Path):
        path = self._two_images(tmp_path / "s.medh5")
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError) as exc, medh5.amend(path) as w:
            w.remove_grid("p")
        assert exc.value.code == "E101"
        assert path.read_bytes() == before
        with pytest.raises(MEDH5ValidationError) as exc, medh5.amend(path) as w:
            w.remove_image("CT")
            w.remove_image("PT")
        assert exc.value.code == "E201"
        assert path.read_bytes() == before

    def test_S4_1_set_image_attrs_edits_and_refuses_grid(self, tmp_path: Path):
        path = self._two_images(tmp_path / "s.medh5")
        with medh5.amend(path) as w:
            w.set_image_attrs(
                "PT", modality="NM", value_units=None, window_center=[1.0]
            )
            w.set_image_attrs("CT", window_width=[400.0])
        assert b"marker-units-7d1c" not in path.read_bytes()
        with h5py.File(path, "r") as handle:
            pt = handle["images/PT"].attrs
            assert pt["modality"] in ("NM", b"NM")
            assert "value_units" not in pt
            assert list(pt["window_center"]) == [1.0]
        with medh5.open(path) as sample:
            assert sample.images["PT"].modality == "NM"
            assert sample.verify().ok
        refusals = [
            ({"grid": "g"}, None, "grid"),
            ({"digest": "sha256:" + "0" * 64}, None, "digest"),
            ({"levels": 2}, None, "levels"),
            ({"colour": "red"}, None, "colour"),
            ({"modality": None}, "E205", "modality"),
            ({"value_type": "loudness"}, "E203", "loudness"),
            ({"channel_names": ["a", "b"]}, "E204", "channel_names"),
            ({"rescale_slope": float("inf")}, None, "rescale_slope"),
        ]
        for attrs, code, named in refusals:
            with (
                pytest.raises(MEDH5ValidationError, match=named) as exc,
                medh5.amend(path) as w,
            ):
                w.set_image_attrs("PT", **attrs)
            assert exc.value.code == code, attrs
        with pytest.raises(MEDH5ValidationError, match="XR"), medh5.amend(path) as w:
            w.set_image_attrs("XR", modality="DX")

    def test_S6_3_add_mask_returns_its_group(self, tmp_path: Path):
        with Flat.open_writer(tmp_path / "m.medh5") as w:
            group = w.add_mask("fov", Flat.mask(), grid="g")
            assert group.name == "/annotations/fov"
            assert group.attrs["kind"] in ("mask", b"mask")


class TestWriterContracts:
    """What the writer is handed reaches the file, or is refused."""

    @pytest.mark.parametrize(
        "encoding", ["labelmap", "layers", "bitmask", "instances", "probmap", "auto"]
    )
    def test_S7_7_an_ignore_region_survives_every_encoding(
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
        region = Organs.ignore()
        path = tmp_path / f"{encoding}.medh5"
        with Organs.writer(path) as w:
            act = w.activity("annotate", agent=w.software("t"))
            if encoding == "probmap":
                payload = {
                    "probabilities": {
                        c: m.astype(np.float32) for c, m in Organs.blocks().items()
                    }
                }
            elif encoding == "instances":
                payload = {
                    "instances": [
                        InstanceInput(class_id=c, instance_id=i + 1, mask=m)
                        for i, (c, m) in enumerate(Organs.blocks().items())
                    ]
                }
            elif encoding == "labelmap":
                payload = {
                    "masks": {1: Organs.blocks()[1]}
                }  # labelmap needs no overlap
            else:
                payload = {"masks": Organs.blocks()}
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

    def test_S7_7_the_sibling_mask_validates_and_silences_W904(self, tmp_path: Path):
        from medh5.validate import validate_file

        path = tmp_path / "cover.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation(
                "seg",
                grid="g",
                masks=Organs.blocks(),
                encoding="bitmask",
                annotated_classes=[1],
                ignore=Organs.ignore(),
            )
        report = validate_file(path, level="strict")
        assert "W904" not in report.codes
        assert not [c for c in report.codes if c.startswith("E")]

    def test_S7_7_ignore_and_ignore_mask_together_are_refused(self, tmp_path: Path):
        """Two sources of one fact cannot be kept in agreement."""
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            Organs.writer(tmp_path / "both.medh5") as w,
        ):
            w.add_mask("mine", Organs.ignore(), grid="g")
            w.add_segmentation(
                "seg",
                grid="g",
                masks=Organs.blocks(),
                encoding="bitmask",
                ignore=Organs.ignore(),
                ignore_mask="mine",
            )
        assert exc.value.code == "E404"

    def test_S7_7_encode_voxels_refuses_what_one_payload_cannot_carry(self):
        """The payload-level entry point had the identical silent branch.

        It returns one payload and so cannot create the sibling mask; refusing
        is the same answer `transcode` already gave in the other direction.
        """
        from medh5.annotations.voxel import encode_voxels

        with pytest.raises(MEDH5ValidationError) as exc:
            encode_voxels(
                Organs.blocks(),
                Organs.SHAPE,
                encoding="bitmask",
                ignore=Organs.ignore(),
            )
        assert exc.value.code == "E404"
        payload, _ = encode_voxels(
            Organs.blocks(), Organs.SHAPE, encoding="layers", ignore=Organs.ignore()
        )
        assert payload.kind == "layers"

    def test_S7_6_selection_costs_the_widening_an_ignore_forces(self):
        """An in-band ignore forces `uint16` planes, so the choice must see it."""
        from medh5.annotations.voxel import select_encoding

        masks = {c: np.zeros(Organs.SHAPE, dtype=bool) for c in range(1, 4)}
        for c, mask in masks.items():
            mask[c] = True
        _, plain = select_encoding(masks, Organs.SHAPE)
        _, widened = select_encoding(masks, Organs.SHAPE, ignore=True)
        from medh5.annotations.voxel import cost_model

        assert cost_model(widened, ignore=True).layers > cost_model(plain).layers

    def test_S11_1_a_provenance_id_is_never_silently_overwritten(self, tmp_path: Path):
        """`person("Alice", agent_id="s2")` then `software("tool")` left one agent.

        The automatic ids are `<type initial><n>` and `act_<type>_<n>`, which
        is exactly what a caller who named a node explicitly is likely to have
        used.  Assigning into the dict meant the second write won, every
        reference resolved, and it resolved to the wrong node.
        """
        path = tmp_path / "prov.medh5"
        with Organs.writer(path) as w:
            alice = w.person("Alice", agent_id="s2")
            tool = w.software("tool")
            assert tool.id != alice.id
            imported = w.activity("import", activity_id="act_annotate_2", agent=tool)
            annotated = w.activity("annotate", agent=tool)
            assert annotated.id != imported.id
            w.add_segmentation("seg", grid="g", masks=Organs.blocks(), prov=imported)

        with medh5.open(path) as sample:
            provenance = sample.document.provenance
            assert {a.id for a in provenance.agents} == {"s2", "s3"}
            assert provenance.agent("s2").name == "Alice"
            assert (
                provenance.activity(sample.annotations["seg"].prov or "").type
                == "import"
            )

    def test_S11_1_an_explicit_duplicate_id_raises(self, tmp_path: Path):
        with Organs.writer(tmp_path / "dup.medh5") as w:
            alice = w.person("Alice", agent_id="s2")
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.person("Bob", agent_id="s2")
            w.activity("import", activity_id="act1", agent=alice)
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.activity("annotate", activity_id="act1", agent=alice)
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())

    def test_S11_1_replace_is_available_for_the_one_legitimate_rewrite(self):
        from medh5.curation import Agent, Provenance

        graph = Provenance(agents=[Agent("a1", "person", "Alice")])
        graph.add_agent(Agent("a1", "person", "pseudo:abc"), replace=True)
        assert graph.agent("a1").name == "pseudo:abc"

    def test_S7_6_transcoding_to_mask_is_refused(self, tmp_path: Path):
        """`mask` has no classes, so the conversion erased the coverage contract.

        Every class was OR-ed into one volume, `class_ids` and
        `annotated_class_ids` came out empty under `task="segmentation"`, and
        the result validated clean.  The CLI was safe because `seg convert`
        restricts its choices; the Python API was not.
        """
        path = tmp_path / "mask.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation(
                "seg", grid="g", masks=Organs.blocks(), annotated_classes=[1, 3]
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

    def test_S7_6_TRANSCODABLE_is_the_whole_truth(self):
        from medh5.annotations.voxel import (
            TRANSCODABLE,
            encode_masks,
            transcode_payload,
        )

        assert "mask" not in TRANSCODABLE
        payload = encode_masks(Organs.blocks(), "layers", Organs.SHAPE)
        with pytest.raises(MEDH5ValidationError):
            transcode_payload(payload, "mask", spatial_shape=Organs.SHAPE)

    @pytest.mark.parametrize(("encoding", "code"), [("mask", "E404"), ("nope", "E401")])
    def test_S7_6_a_write_refuses_the_encodings_a_transcode_refuses(
        self, tmp_path: Path, encoding: str, code: str
    ):
        """`add_segmentation(encoding="mask")` OR-ed every class into one
        volume and wrote it with empty `class_ids` and `annotated_class_ids`
        under `task="segmentation"` --- the coverage contract a transcode to
        `mask` refuses to erase --- and the file validated clean."""
        from medh5.annotations.voxel import encode_masks, transcode_payload

        payload = encode_masks(Organs.blocks(), "layers", Organs.SHAPE)
        with pytest.raises(MEDH5ValidationError) as transcoded:
            transcode_payload(payload, encoding, spatial_shape=Organs.SHAPE)
        path = tmp_path / "encoding.medh5"
        with Organs.writer(path) as w:
            with pytest.raises(MEDH5ValidationError) as written:
                w.add_segmentation(
                    "seg", grid="g", masks=Organs.blocks(), encoding=encoding
                )
            assert written.value.code == transcoded.value.code == code
            assert list(w.handle["annotations"].keys()) == []
            w.add_segmentation(
                "seg", grid="g", masks=Organs.blocks(), annotated_classes=[1, 3]
            )
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.kind != "mask"
            assert annotation.class_ids == annotation.annotated_class_ids == (1, 3)

    def test_S7_2_the_writer_stores_the_layers_the_encoder_builds(self, tmp_path: Path):
        """The writer measured the overlap graph to choose an encoding, then
        measured and coloured it again to write `layers`.  It now writes the
        colouring it measured, which must be the one the encoder builds."""
        from medh5.annotations.voxel import encode_masks

        rng = np.random.default_rng(7)
        masks = {}
        for class_id in range(1, 9):
            mask = np.zeros(Numbered.SHAPE, bool)
            z, y, x = (int(rng.integers(0, n - 4)) for n in Numbered.SHAPE)
            mask[z : z + 4, y : y + 4, x : x + 4] = True
            masks[class_id] = mask
        expected = encode_masks(masks, "layers", Numbered.SHAPE)
        assert expected.datasets["data"].shape[0] > 1, "the classes overlap"

        path = tmp_path / "layers.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
            w.label_set(Numbered.label_set(8))
            kind, _ = w.add_segmentation("seg", grid="g", masks=masks)
        assert kind == "layers"
        with medh5.open(path) as sample:
            group = sample.annotations["seg"].group
            for name, array in expected.datasets.items():
                np.testing.assert_array_equal(group[name][...], array, strict=True)


class TestRewriteGate:
    """What every door into a file checks."""

    def test_S16_amend_refuses_a_future_major(self, tmp_path: Path):
        """`open` refused a 2.0 file; `amend` restamped it 1.0 and carried on."""
        path = Numbered.plain(tmp_path / "v2.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        before = path.read_bytes()
        with pytest.raises(MEDH5VersionError):
            medh5.amend(path)
        assert path.read_bytes() == before

    def test_S2_3_an_amendment_never_lowers_the_version(self, tmp_path: Path):
        """1.1 §2.1, §2.3: imaging alone needs 1.0, but a file that declares
        1.1 stays a 1.1 file, with what it carries.  (1.1 is this engine's own
        version: a *later* minor is refused, which the test of the gate
        below holds.)"""
        path = Numbered.plain(tmp_path / "v11.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "1.1"
            handle.create_group("x_future")
        attested_as_written(path)
        with medh5.amend(path):
            pass
        with h5py.File(path, "r") as handle:
            assert handle.attrs["medh5_version"] == "1.1"
            assert "x_future" in handle

    def test_S2_1_dropping_a_clinical_profile_the_file_lacks_keeps_its_version(
        self, tmp_path: Path
    ):
        """Only an amendment that removes the clinical profile lowers the
        version (1.1 §2.1).  `drop_clinical` on a file with none removed
        nothing, and wrote a 1.1 file as 1.0."""
        path = Numbered.plain(tmp_path / "v11.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "1.1"
        attested_as_written(path)
        with medh5.amend(path) as w:
            w.drop_clinical()
        with h5py.File(path, "r") as handle:
            assert handle.attrs["medh5_version"] == "1.1"

    @pytest.mark.parametrize("version", ["2.0", LATER_MINOR])
    def test_S16_the_scrub_and_fix_doors_are_the_same_gate(
        self, tmp_path: Path, version: str
    ):
        """Every tool that rewrites a file refuses what `amend` refuses ---
        another major, and a later minor read only as its projection (1.1
        §2.3) --- and leaves it as it was.  The test was named for the fix
        door and knocked only on scrub's."""
        from medh5.curation import scrub
        from medh5.integrity import fix

        path = Numbered.plain(tmp_path / "gate.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = version
        before = path.read_bytes()
        doors = {
            "amend": lambda: medh5.amend(path),
            "scrub": lambda: scrub.apply(path),
            "fix": lambda: fix(path, rewrite_digests=True, reason="restamp"),
        }
        for door, knock in doors.items():
            with pytest.raises(MEDH5VersionError):
                knock()
            assert path.read_bytes() == before, door
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    @pytest.fixture
    def secret(self, tmp_path: Path) -> Path:
        path = tmp_path / "id_rsa"
        path.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----hunter2" + b"A" * 4000)
        return path

    def _external_storage(self, tmp_path: Path, secret: Path) -> Path:
        path = Numbered.plain(tmp_path / "ext.medh5")
        size = secret.stat().st_size
        with h5py.File(path, "r+") as handle:
            handle.create_dataset(
                "x_vendor_notes",
                shape=(size,),
                dtype="u1",
                external=[(secret, 0, size)],
            )
        return path

    def test_S2_external_storage_is_refused_by_every_tool(
        self, tmp_path: Path, secret: Path
    ):
        """A crafted file made `recompress` copy a local file into its output."""
        from medh5.storage import recompress

        path = self._external_storage(tmp_path, secret)
        out = tmp_path / "shared.medh5"
        for door in (
            lambda: medh5.open(path),
            lambda: medh5.amend(path),
            lambda: recompress(path, "portable", out=out),
        ):
            with pytest.raises(MEDH5FileError, match="not self-contained"):
                door()
        assert not out.exists()
        assert validate_file(path).codes == ("E001",)

    def test_S2_an_external_link_is_refused(self, tmp_path: Path):
        other = Numbered.plain(tmp_path / "other.medh5")
        path = Numbered.plain(tmp_path / "linked.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_link"] = h5py.ExternalLink(str(other), "/images")
        with pytest.raises(MEDH5FileError, match="external link"):
            medh5.open(path)

    def test_S2_a_virtual_dataset_is_refused(self, tmp_path: Path):
        source = tmp_path / "source.h5"
        with h5py.File(source, "w") as handle:
            handle["data"] = np.arange(16, dtype="u1")
        path = Numbered.plain(tmp_path / "virtual.medh5")
        layout = h5py.VirtualLayout(shape=(16,), dtype="u1")
        layout[:] = h5py.VirtualSource(str(source), "data", shape=(16,))
        with h5py.File(path, "r+") as handle:
            handle.create_virtual_dataset("x_view", layout)
        with pytest.raises(MEDH5FileError, match="virtual dataset"):
            medh5.open(path)

    @pytest.mark.skipif(os.name == "nt", reason="Windows cannot replace an open file")
    def test_S2_the_check_is_remembered_for_the_file_it_read(
        self, tmp_path: Path, secret: Path
    ):
        """A path replaced between the open and the check must not inherit it.

        The memo was keyed by a stat of the *path*, taken after the open.  A
        replacement landing in between was recorded as checked while the handle
        scanned the old file, and its next open skipped the check.  The race
        itself is the engine's to hold (`h5::ops::tests::
        s2_the_check_is_remembered_for_the_file_it_read`); through the public
        door, a file checked and remembered does not vouch for its replacement.
        """
        target = Numbered.plain(tmp_path / "target.medh5")
        bad = self._external_storage(tmp_path, secret)
        with medh5.open(target) as sample:  # checked, and remembered
            os.replace(bad, target)  # the path now names another file
            assert "CT" in sample.images  # the file read is clean
        with pytest.raises(MEDH5FileError, match="not self-contained"):
            medh5.open(target)

    def test_S2_a_file_of_a_checked_files_length_and_time_is_still_checked(
        self, tmp_path: Path
    ):
        """The memo is keyed by the opened file's identity, never by its length
        and modification time alone.

        Off POSIX the key was ``(0, 0, size, mtime, 0)``: a crafted file padded
        to a checked file's length, with its time set to match, skipped the
        check on open.  (On POSIX the inode tells them apart; on Windows this
        holds only because nothing is remembered there.)
        """
        checked = Numbered.plain(tmp_path / "checked.medh5")
        other = Numbered.plain(tmp_path / "other.medh5")
        crafted = Numbered.plain(tmp_path / "crafted.medh5")
        with h5py.File(crafted, "r+") as handle:
            handle["x_link"] = h5py.ExternalLink(str(other), "/images")
        small, big = sorted((checked, crafted), key=lambda p: p.stat().st_size)
        with open(small, "ab") as handle:  # HDF5 ignores bytes past its end
            handle.write(b"\0" * (big.stat().st_size - small.stat().st_size))
        times = checked.stat()
        os.utime(crafted, ns=(times.st_atime_ns, times.st_mtime_ns))
        assert checked.stat().st_size == crafted.stat().st_size
        with medh5.open(checked) as sample:  # checked, and remembered if it can be
            assert "CT" in sample.images
        with pytest.raises(MEDH5FileError, match="external link"):
            medh5.open(crafted)

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
        reason="POSIX file modes, which root ignores",
    )
    def test_S14_4_a_read_only_sample_can_still_be_amended(self, tmp_path: Path):
        """The temporary took the target's `0o444` and HDF5 could not write it."""
        path = Numbered.plain(tmp_path / "read-only.medh5")
        os.chmod(path, 0o444)
        writer = medh5.amend(path)
        try:
            temporary = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
            assert temporary
            assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in temporary)
        finally:
            writer.commit()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444

    def test_S2_2_unpack_refuses_a_member_name_that_is_a_path(self, tmp_path: Path):
        from medh5.collection import pack, unpack

        shard = tmp_path / "s.medh5c"
        pack([Numbered.plain(tmp_path / "a.medh5")], shard, keys=["good"])
        with h5py.File(shard, "r+") as handle:
            handle.copy("samples/good", handle["samples"], name="..\\..\\evil")
        with pytest.raises(MEDH5ValidationError, match="evil"):
            unpack(shard, tmp_path / "out")
        assert not list(tmp_path.rglob("evil*"))

    def test_S14_amend_keeps_a_portable_file_portable(self, tmp_path: Path):
        """New datasets in an amended `portable` file were Blosc2."""
        from medh5.storage import describe_filters

        path = Numbered.plain(tmp_path / "port.medh5", codec="portable")
        big = np.random.default_rng(0).random((64, 64, 64)) > 0.5
        with medh5.amend(path) as w:
            w.add_grid("g2", shape=big.shape, spacing=(1.0, 1.0, 1.0))
            w.add_mask("new", big, grid="g2")
        with medh5.open(path) as sample:
            stored = sample.root["annotations/new/data"]
            assert describe_filters(stored).startswith("gzip")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_S14_4_the_temporary_file_is_never_more_permissive(self, tmp_path: Path):
        path = Numbered.plain(tmp_path / "perm.medh5")
        os.chmod(path, 0o600)
        writer = medh5.amend(path)
        try:
            temporary = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
            assert temporary
            assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in temporary)
        finally:
            writer.commit()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_S14_4_a_read_only_file_is_amended_and_stays_read_only(
        self, tmp_path: Path
    ):
        """On every platform, a read-only file is amended, and stays read-only.

        Windows refused both halves of it: the temporary was made read-only
        and then opened for writing to be flushed, and a rename does not
        replace a read-only file.
        """
        path = Numbered.plain(tmp_path / "read-only.medh5")
        os.chmod(path, stat.S_IREAD)
        try:
            with medh5.amend(path) as w:
                w.add_grid("g2", shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
            assert not path.stat().st_mode & stat.S_IWRITE
            with medh5.open(path) as sample:
                assert "g2" in sample.grids
        finally:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        assert [p.name for p in tmp_path.iterdir()] == [path.name]


def _amend(path: Path) -> None:
    with medh5.amend(path) as w:
        w.add_grid("g2", shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))


def _scrub(path: Path) -> None:
    from medh5.curation import scrub

    assert scrub.apply(path).ok


def _recompress(path: Path) -> None:
    from medh5.storage import recompress

    assert recompress(path, "archive").changed


def _fix(path: Path) -> None:
    from medh5.integrity import fix

    with h5py.File(path, "r+") as handle:  # an edit behind the writer's back
        handle["images/CT"][0, 0, 0] = 7
    assert fix(path, rewrite_digests=True, reason="edited by another tool").changed


class TestLostUpdates:
    """§14.4: an amendment refuses to replace a file that changed after it read
    it.  Two amendments, or an amendment and an edit, both went through, and
    the first one's work was lost without an error."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    @staticmethod
    def _refused(writer: Any, path: Path) -> None:
        with pytest.raises(MEDH5FileError) as err:
            writer.commit()
        assert str(err.value) == (
            f"{str(path)!r} changed after this amendment read it; nothing was "
            "written: amend it again"
        )
        assert writer.closed

    @staticmethod
    def _note(path: Path, note: str) -> None:
        with h5py.File(path, "r+") as handle:
            handle.attrs["x_note"] = note

    def test_S14_4_a_commit_refuses_a_source_another_writer_replaced(
        self, tmp_path: Path
    ):
        path = self._sample(tmp_path / "case.medh5")
        writer = medh5.amend(path)
        writer.set_quality("seg", status="approved")
        other = tmp_path / "other.medh5"
        other.write_bytes(path.read_bytes())
        self._note(other, "another writer")
        os.replace(other, path)
        self._refused(writer, path)
        with h5py.File(path, "r") as handle:
            assert handle.attrs["x_note"] == "another writer"
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    def test_S14_4_a_commit_refuses_a_source_modified_in_place(self, tmp_path: Path):
        path = self._sample(tmp_path / "case.medh5")
        writer = medh5.amend(path)
        writer.set_quality("seg", status="approved")
        self._note(path, "edited in place")
        self._refused(writer, path)
        with h5py.File(path, "r") as handle:
            assert handle.attrs["x_note"] == "edited in place"
        with medh5.open(path) as sample:
            assert "seg" not in sample.document.quality
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    def test_S14_4_an_amendment_that_read_before_another_committed_is_refused(
        self, tmp_path: Path
    ):
        path = self._sample(tmp_path / "case.medh5")
        first = medh5.amend(path)
        first.set_quality("seg", status="approved")
        with medh5.amend(path) as second:
            second.set_quality("seg", status="rejected")
        self._refused(first, path)
        with medh5.open(path) as sample:
            assert sample.document.quality["seg"].status == "rejected"
        # Amended again, from what is there now, it commits.
        with medh5.amend(path) as again:
            again.set_quality("seg", status="approved")
        with medh5.open(path) as sample:
            assert sample.document.quality["seg"].status == "approved"


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privilege")
class TestReplaceThroughLinks:
    """§14.4: a rewrite through a symbolic link replaces the file it names.

    The temporary file was made beside the link and renamed over it: the link
    became an independent copy, and the file it named was left as it was ---
    after ``scrub --apply``, the identifying original, beside a scrubbed copy
    the user took for it.
    """

    @pytest.mark.parametrize(
        "rewrite",
        [_amend, _scrub, _recompress, _fix],
        ids=["amend", "scrub", "recompress", "fix"],
    )
    @pytest.mark.parametrize("link", ["absolute", "relative", "chain"])
    def test_S14_4_the_file_a_link_names_is_rewritten_and_the_link_kept(
        self, tmp_path: Path, rewrite: Any, link: str
    ):
        data, farm = tmp_path / "data", tmp_path / "farm"
        data.mkdir()
        farm.mkdir()
        # Large enough that recompression has a bulk dataset to re-encode.
        with Numbered.writer(data / "real.medh5", shape=(32, 64, 64)):
            pass
        real = data / "real.medh5"
        before = real.read_bytes()
        given = farm / "given.medh5"
        if link == "absolute":
            given.symlink_to(real)
        elif link == "relative":
            given.symlink_to(Path("..", "data", "real.medh5"))
        else:
            (farm / "middle.medh5").symlink_to(Path("..", "data", "real.medh5"))
            given.symlink_to("middle.medh5")
        named = os.readlink(given)

        rewrite(given)

        assert given.is_symlink() and os.readlink(given) == named
        assert given.resolve() == real.resolve()
        assert real.read_bytes() != before
        assert validate_file(real).ok
        assert sorted(p.name for p in data.iterdir()) == ["real.medh5"]
        assert all(p.is_symlink() for p in farm.iterdir())


class TestWriterEqualsValidator:
    """One definition of a valid file."""

    def test_S10_3_a_field_off_its_grid_is_refused_at_the_call(self, tmp_path: Path):
        """A (3,4,4,4) field on a 16³ grid was written, then read as T(x) = x."""
        with medh5.create(tmp_path / "disp.medh5", sample_id="s") as w:
            w.add_timepoint("tp0", index=0)
            w.add_timepoint("tp1", index=1)
            for n in range(2):
                w.add_grid(
                    f"g{n}",
                    shape=(16, 16, 16),
                    spacing=(1.0, 1.0, 1.0),
                    frame_uid=f"F{n}",
                    timepoint=f"tp{n}",
                )
                w.add_image(
                    f"CT{n}", np.zeros((16,) * 3, np.int16), grid=f"g{n}", modality="CT"
                )
            with pytest.raises(MEDH5ValidationError, match="lattice") as caught:
                w.add_transform(
                    "t",
                    kind="displacement",
                    from_frame="F0",
                    to_frame="F1",
                    field=np.ones((3, 4, 4, 4), np.float32),
                    field_grid="g0",
                )
            assert caught.value.code == "E503"
            with pytest.raises(MEDH5ValidationError, match="components"):
                w.add_transform(
                    "b",
                    kind="bspline",
                    from_frame="F0",
                    to_frame="F1",
                    control_points=np.zeros((2, 5, 5), np.float32),
                    cp_grid="g0",
                )
            w.abort()

    def test_S15_commit_refuses_what_the_validator_rejects(self, tmp_path: Path):
        """Anything written through the handle is held to the same rules."""
        path = Numbered.plain(tmp_path / "keep.medh5")
        before = path.read_bytes()
        writer = medh5.amend(path)
        del writer.handle["images/CT"].attrs["grid"]
        with pytest.raises(MEDH5ValidationError, match="fail validation") as caught:
            writer.commit()
        assert caught.value.code.startswith("E")
        assert path.read_bytes() == before, "the refused commit replaced the file"
        assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name]

    def test_S5_1_a_ref_label_set_is_written(self, tmp_path: Path):
        """The writer refused what the validator accepts: E402 under form=ref."""
        path = tmp_path / "ref.medh5"
        mask = np.zeros(Numbered.SHAPE, bool)
        mask[1, 2, 2] = True
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=Numbered.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Numbered.SHAPE, np.int16), grid="g", modality="CT"
            )
            w.label_set(Numbered.label_set().as_ref("https://example.org/vocab.json"))
            w.add_segmentation("seg", grid="g", masks={1: mask})
        assert validate_file(path).ok

    def test_the_schema_check_needs_no_optional_dependency(self):
        """1.4.1 made jsonschema a core dependency so E005 was checked
        everywhere.  2.0 compiles the validator into the engine: NumPy is the
        only runtime dependency, and the check is always there."""
        from importlib.metadata import requires

        from medh5.document import validate_against_schema

        core = [r for r in requires("medh5") or () if "extra ==" not in r]
        assert [re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0] for r in core] == ["numpy"]
        assert validate_against_schema({"identity": {}})

    def test_S2_4_commit_checks_the_schema(self, tmp_path: Path):
        """A class key like `Left-Kidney` failed E005 only where jsonschema was."""
        with Numbered.writer(tmp_path / "schema.medh5") as w:
            w.extra("x", {"fine": 1})
            w.document.label_set = LabelSet(
                "bad",
                version="1",
                classes=[LabelClass(1, "Left-Kidney", "Left kidney")],
            )
            with pytest.raises(MEDH5ValidationError) as caught:
                w.commit()
            assert caught.value.code == "E005"
            w.abort()

    @pytest.mark.parametrize(
        ("field", "value"), [("units", "cm"), ("units", "inch"), ("time_units", "min")]
    )
    def test_S3_2_units_outside_the_list_are_refused(
        self, tmp_path: Path, field: str, value: str
    ):
        with medh5.create(tmp_path / "u.medh5", sample_id="s") as w:
            options: dict[str, Any] = {field: value}
            if field == "time_units":
                options.update(
                    axis_kinds=("time", "spatial", "spatial", "spatial"),
                    time_values=[0.0, 1.0],
                )
                shape: tuple[int, ...] = (2, *Numbered.SHAPE)
            else:
                shape = Numbered.SHAPE
            with pytest.raises(MEDH5ValidationError, match=value) as caught:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), **options)
            assert caught.value.code == "E109"
            w.abort()

    def test_S7_6_one_source_and_a_matching_encoding(self, tmp_path: Path):
        """`masks=` was dropped when `probabilities=` came too."""
        from medh5.annotations.voxel import InstanceInput

        liver, spleen, _ = Numbered.two_classes()
        with Numbered.writer(tmp_path / "l23.medh5") as w:
            for call in (
                {"masks": {1: liver}, "probabilities": {2: spleen.astype(float)}},
                {"probabilities": {1: liver.astype(float)}, "encoding": "layers"},
                {"instances": [InstanceInput(1, 1, mask=liver)], "encoding": "bitmask"},
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    w.add_segmentation("s", grid="g", **call)
                assert caught.value.code == "E404"
            kind, _ = w.add_segmentation(
                "p",
                grid="g",
                probabilities={1: liver.astype(float)},
                encoding="probmap",
            )
            assert kind == "probmap"

    @pytest.mark.parametrize("encoding", Numbered.ENCODINGS)
    def test_S7_7_an_overlapping_region_reads_back_whole(
        self, tmp_path: Path, encoding: str
    ):
        """Under `labelmap` 56 of 64 ignored voxels came back; `bitmask` gave 64."""
        path = tmp_path / f"ovl-{encoding}.medh5"
        region = Numbered.encoded(path, encoding, overlap=True)
        liver, spleen, _ = Numbered.two_classes()
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            np.testing.assert_array_equal(sample.ignore_region("seg"), region)
            np.testing.assert_array_equal(annotation.dense([1])[0], liver)
            np.testing.assert_array_equal(annotation.dense([2])[0], spleen)
            assert annotation.header.ignore_mask == "seg_ignore"
        assert validate_file(path).ok

    def test_S7_7_a_region_clear_of_the_classes_stays_in_band(self, tmp_path: Path):
        path = tmp_path / "band.medh5"
        Numbered.encoded(path, "labelmap")
        with medh5.open(path) as sample:
            assert sample.annotations["seg"].header.ignore_mask is None
            assert "seg_ignore" not in sample.annotations

    def _lesions(self, path: Path) -> Path:
        from medh5.annotations.voxel import InstanceInput

        first = np.zeros(Numbered.SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(Numbered.SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 11, mask=first),
                    InstanceInput(1, 22, mask=second),
                ],
            )
        return path

    def test_S7_4_a_transcode_does_not_drop_identity_silently(self, tmp_path: Path):
        """After instances -> labelmap, tracks() was empty and nothing said so."""
        path = self._lesions(tmp_path / "tc.medh5")
        with medh5.amend(path) as w:
            with pytest.raises(MEDH5ValidationError, match="drop_identity") as caught:
                w.transcode_annotation("les", "labelmap")
            assert caught.value.code == "E404"
        with medh5.amend(path) as w:
            w.transcode_annotation("les", "labelmap", drop_identity=True)
        with medh5.open(path) as sample:
            assert sample.annotations["les"].kind == "labelmap"
            (activity,) = sample.document.provenance.activities_by_type("transcode")
            assert activity.params["dropped"] == "instance identity"
            assert activity.params["objects"] == 2
            assert activity.outputs == ("annotations/les",)
        assert validate_file(path).ok

    def test_the_cli_flag(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._lesions(tmp_path / "cli.medh5")
        args = ["seg", "convert", str(path), "les", "--to", "layers"]
        assert main(args) != 0
        assert main([*args, "--drop-identity"]) == 0
        capsys.readouterr()
        with medh5.open(path) as sample:
            assert sample.annotations["les"].kind == "layers"

    def test_S7_4_examined_and_none_found(self, tmp_path: Path):
        """`instances=[]` was refused (E410): a resolved lesion read unexamined."""
        from medh5.annotations.voxel import InstanceInput

        lesion = np.zeros(Numbered.SHAPE, bool)
        lesion[2:4, 2:4, 2:4] = True
        path = tmp_path / "resolved.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(Numbered.label_set())
            for n in range(2):
                w.add_timepoint(f"tp{n}", index=n)
                w.add_grid(
                    f"g{n}",
                    shape=Numbered.SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=f"tp{n}",
                )
                w.add_image(
                    f"CT{n}",
                    np.zeros(Numbered.SHAPE, np.int16),
                    grid=f"g{n}",
                    modality="CT",
                )
            w.add_segmentation(
                "les0", grid="g0", instances=[InstanceInput(1, 7, mask=lesion)]
            )
            w.add_segmentation("les1", grid="g1", instances=[], annotated_classes=[1])
        report = validate_file(path, level="strict")
        assert not [c for c in report.codes if c.startswith("E")], report.codes
        with medh5.open(path) as sample:
            empty = sample.annotations["les1"]
            assert (empty.class_ids, empty.annotated_class_ids) == ((1,), (1,))
            assert not empty.dense().any()
            assert list(empty.instances()) == []
            tracking = sample.tracks()
            assert tracking.state_at(7, "tp1") == "resolved"
            assert tracking.is_resolved(7)

    def test_S7_4_an_empty_list_alone_says_nothing_and_is_refused(self, tmp_path: Path):
        with Numbered.writer(tmp_path / "bare.medh5") as w:
            with pytest.raises(
                MEDH5ValidationError, match="annotated_classes"
            ) as caught:
                w.add_segmentation("les", grid="g", instances=[])
            assert caught.value.code == "E410"
            w.abort()

    def test_property_12_no_tool_writes_what_the_validator_rejects(
        self, tmp_path: Path
    ):
        """Every corpus file, amended as it is: refused, or written valid.

        The writer and the validator each enforced a hand-kept list of rules,
        and the two drifted.  With ``commit()`` running the
        validator there is one list --- which this checks from the outside, over
        the conformance corpus: a no-op amend of each of its files either
        refuses the file or writes one the validator passes, and a valid case
        comes back valid.  A valid higher-minor file is one valid sample that
        must be refused: 1.1 §2.3 forbids amending a projection.  A case valid
        at its level whose `content_id` the integrity level fails (E702) is
        another: §14.4 forbids carrying attestation past a break.
        """
        from medh5.conformance import CASES
        from medh5.errors import MEDH5Error

        refused = written = 0
        for case in CASES:
            path = tmp_path / f"{case.name}{case.suffix}"
            case.build(path)
            try:
                with medh5.amend(path):
                    pass
            except (MEDH5Error, OSError) as err:
                refused += 1
                projection = "W913" in case.warnings
                broken = getattr(err, "code", None) == "E702" and "E702" in (
                    validate_file(path, level="integrity").codes
                )
                assert (
                    not case.valid or case.suffix == ".medh5c" or projection or broken
                ), case.name
                continue
            written += 1
            errors = [c for c in validate_file(path).codes if c.startswith("E")]
            assert errors == [], (case.name, errors)
        assert refused and written


class TestDuplicateIds:
    """Two nodes with one id are refused, not merged."""

    def _duplicated(self, path: Path, section: str) -> Path:
        with Framed.writer(path) as w:
            alice = w.person("Alice")
            w.activity("annotate", agent=alice)
        with h5py.File(path, "r+") as handle:
            doc = json.loads(handle["meta"][()])
            first = doc["provenance"][section][0]
            doc["provenance"][section].append({**first})
            del handle["meta"]
            handle.create_dataset(
                "meta", data=json.dumps(doc), dtype=h5py.string_dtype()
            )
        attested_as_written(path)
        return path

    @pytest.mark.parametrize("section", ["agents", "activities"])
    def test_S11_1_a_duplicated_provenance_id_is_refused_on_read(
        self, tmp_path: Path, section: str
    ):
        """The reader kept the last, and a no-op amend deleted the other."""
        path = self._duplicated(tmp_path / f"{section}.medh5", section)
        before = path.read_bytes()
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError):
            _ = s.document
        with pytest.raises(MEDH5ValidationError, match="more than once"):
            medh5.amend(path).commit()
        assert path.read_bytes() == before
        assert validate_file(path).errors

    def test_S7_4_the_writer_refuses_two_objects_sharing_an_id(self, tmp_path: Path):
        first = np.zeros(Framed.SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(Framed.SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        with (
            Framed.writer(tmp_path / "dup.medh5") as w,
            pytest.raises(MEDH5ValidationError) as caught,
        ):
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 5, mask=first),
                    InstanceInput(1, 5, mask=second),
                ],
            )
        assert caught.value.code == "E404"

    def test_S7_4_the_validator_reports_a_shared_id_as_E404(self, tmp_path: Path):
        first = np.zeros(Framed.SHAPE, bool)
        first[1:3, 1:3, 1:3] = True
        second = np.zeros(Framed.SHAPE, bool)
        second[5:7, 5:7, 5:7] = True
        path = tmp_path / "crafted.medh5"
        with Framed.writer(path) as w:
            w.add_segmentation(
                "les",
                grid="g",
                instances=[
                    InstanceInput(1, 5, mask=first),
                    InstanceInput(1, 6, mask=second),
                ],
            )
        with h5py.File(path, "r+") as handle:
            ids = handle["annotations/les/instance_ids"]
            ids[...] = np.array([5, 5], dtype=ids.dtype)
        assert "E404" in validate_file(path).codes

    def test_S7_4_the_W909_corpus_case_is_the_sample_scoped_conflict(
        self, tmp_path: Path
    ):
        """One id classed apart across two annotations: a warning, and no E404."""
        from medh5.conformance import build_corpus, case_by_name

        case = case_by_name("W909-instance-id-two-classes")
        build_corpus(tmp_path, names=[case.name])
        path = tmp_path / f"{case.name}.medh5"
        with h5py.File(path, "r") as handle:
            groups = [g for g in handle["annotations"].values() if "instance_ids" in g]
            assert len(groups) == 2
            assert all(len(g["instance_ids"]) == 1 for g in groups)
        report = validate_file(path, level=case.level)
        assert set(report.codes) == {*case.errors, *case.warnings}


def _laid_out(values: np.ndarray, layout: str) -> np.ndarray:
    """*values* in another memory layout: NumPy hands out each of these."""
    if layout == "fortran":  # nibabel's get_fdata()
        return np.asfortranarray(values)
    if layout == "transposed":
        return np.ascontiguousarray(values.T).T
    if layout == "reversed":  # np.flip: negative strides
        return np.flip(np.flip(values, 0).copy(), 0)
    wider = np.repeat(values, 2, axis=-1)  # every other element: a strided view
    return wider[..., ::2]


class TestAnyArrayLayout:
    """§14.4: what the writer stores is an array's values, whatever the layout
    NumPy keeps them in --- the writer reads them in C order."""

    @pytest.mark.parametrize("layout", ["fortran", "transposed", "reversed", "strided"])
    def test_S14_4_every_layout_is_written_as_its_values(
        self, tmp_path: Path, layout: str
    ):
        image = Flat.image()
        mask = Flat.mask()
        boxes = np.array(
            [[[0, 2], [1, 3], [2, 4]], [[1, 4], [2, 5], [0, 9]]], np.float32
        )
        matrix = np.eye(4)
        matrix[0, 3] = 5.0
        vector = np.arange(6.0).reshape(2, 3)
        laid = {
            name: _laid_out(a, layout)
            for name, a in [
                ("image", image),
                ("mask", mask),
                ("boxes", boxes),
                ("matrix", matrix),
                ("vector", vector),
            ]
        }
        assert not laid["image"].flags.c_contiguous
        path = tmp_path / f"{layout}.medh5"
        with Flat.open_writer(path, frames=True) as w:
            w.add_image("CT2", laid["image"], grid="g", modality="CT")
            w.add_segmentation("seg", grid="g", masks={1: laid["mask"]})
            w.add_boxes("det", laid["boxes"], [1, 1], grid="g")
            w.add_transform(
                "shift",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=laid["matrix"],
            )
            w.handle.attrs["vector"] = laid["vector"]
        with medh5.open(path) as sample:
            assert np.array_equal(sample.images["CT2"].read(), image)
            assert np.array_equal(sample.annotations["seg"].dense([1])[0], mask)
            assert np.array_equal(sample.annotations["det"].boxes, boxes)
            assert np.array_equal(sample.transforms["shift"].matrix, matrix)
            assert np.array_equal(sample.root.attrs["vector"], vector)
        assert validate_file(path).ok


class TestJsonValues:
    """§5.1: a document value is JSON, hashed as canonical JSON, so what has no
    single JSON spelling is refused rather than spelled one way of several."""

    @pytest.mark.parametrize(
        ("value", "error", "match"),
        [
            ({"tags": {"a", "b"}}, TypeError, "set is not JSON serializable"),
            ({"tags": frozenset({1})}, TypeError, "frozenset"),
            ({"rows": (i for i in range(3))}, TypeError, "generator"),
            ({"v": Decimal("NaN")}, MEDH5ValidationError, "not a JSON number"),
            ({"v": Decimal("-Infinity")}, MEDH5ValidationError, "not a JSON number"),
            ({"v": 10**30}, MEDH5ValidationError, "64-bit integers"),
            ({"v": -(2**63) - 1}, MEDH5ValidationError, "64-bit integers"),
        ],
    )
    def test_S5_1_a_value_without_one_JSON_spelling_is_refused(
        self, tmp_path: Path, value: Any, error: type[Exception], match: str
    ):
        """A set's order is its hash seed's, so ``content_id`` changed from one
        run to the next; a NaN ``Decimal`` became ``null`` and a 100-bit
        integer a rounded float."""
        with (
            Flat.open_writer(tmp_path / "x.medh5") as w,
            pytest.raises(error, match=match),
        ):
            w.extra("tool", value)
        with (
            Flat.open_writer(tmp_path / "y.medh5") as w,
            pytest.raises(error, match=match),
        ):
            w.document.extra["tool"] = value

    def test_S5_1_every_64_bit_integer_is_kept_exactly(self, tmp_path: Path):
        path = tmp_path / "ints.medh5"
        edges = [2**64 - 1, -(2**63), 2**53 + 1]
        with Flat.open_writer(path) as w:
            w.extra("tool", {"edges": edges, "decimal": Decimal("1.5")})
        with medh5.open(path) as sample:
            assert sample.document.extra["tool"] == {"edges": edges, "decimal": 1.5}


class TestDocumentEdits:
    """§2.4: ``SampleWriter.document`` is the document being written, and its
    free-form members --- ``extra``, ``acquisition``, ``quality`` --- and the
    dicts of the records it hands out are dicts whose edits reach the file,
    nested ones too, as 1.x's fields did."""

    def test_S2_4_an_edit_in_place_reaches_the_file(self, tmp_path: Path):
        path = tmp_path / "edits.medh5"
        with Flat.open_writer(path) as w:
            w.document.extra["mytool"] = {"a": 1}
            w.document.extra["mytool"]["b"] = [1, 2]
            w.document.extra["mytool"]["b"].append(3)
            w.document.extra.setdefault("other", {})["k"] = "v"
            w.document.extra.update(gone=1)
            del w.document.extra["gone"]
            w.document.acquisition["CT"] = {"kvp": 120}
            w.document.acquisition["CT"]["mA"] = 300
            w.document.quality["CT"] = {"status": "draft"}
            assert json.loads(json.dumps(w.document.extra))["mytool"]["b"] == [1, 2, 3]
        with medh5.open(path) as sample:
            document = sample.document
            assert document.extra == {
                "mytool": {"a": 1, "b": [1, 2, 3]},
                "other": {"k": "v"},
            }
            assert document.acquisition == {"CT": {"kvp": 120, "mA": 300}}
            assert document.quality["CT"].status == "draft"

    def test_S2_4_an_edit_to_a_member_changed_since_is_refused(self, tmp_path: Path):
        """Writing the stale copy back would undo the other change."""
        with Flat.open_writer(tmp_path / "stale.medh5") as w:
            w.document.acquisition["CT"] = {"kvp": 120}
            held = w.document.acquisition["CT"]
            w.acquisition("CT", mA=300)
            with pytest.raises(RuntimeError, match="has changed since it was read"):
                held["ms"] = 5
            with pytest.raises(RuntimeError, match="read it again"):
                held["kvp"] = 100
            w.document.acquisition["CT"]["ms"] = 5  # read again: accepted
            assert w.document.acquisition["CT"] == {"kvp": 120, "mA": 300, "ms": 5}

    def test_S2_4_every_mapping_edit_reaches_the_file(self, tmp_path: Path):
        """A member mapping's every edit writes through, not only ``[k] = v``
        and ``del``: what a removal hands back is detached."""
        path = tmp_path / "mapping.medh5"
        with Flat.open_writer(path) as w:
            extra = w.document.extra
            extra.update({"a": 1, "b": {"x": [1]}}, c=3)
            extra |= {"d": 4}
            assert extra.setdefault("d", 9) == 4
            extra.setdefault("e", {"y": 1})["z"] = 2
            popped = extra.pop("b")
            popped["x"].append(2)
            assert extra.pop("b", None) is None
            with pytest.raises(KeyError):
                extra.pop("b")
            assert extra.popitem() == ("e", {"y": 1, "z": 2})
            assert extra == w.document.extra == {"a": 1, "c": 3, "d": 4}
            w.document.acquisition["CT"] = {"kvp": 120}
            w.document.acquisition.clear()
            with pytest.raises(KeyError):
                w.document.acquisition.popitem()
        with medh5.open(path) as sample:
            assert sample.document.extra == {"a": 1, "c": 3, "d": 4}
            assert sample.document.acquisition == {}

    def test_S2_4_every_nested_edit_reaches_the_file(self, tmp_path: Path):
        """Every edit of a dict or a list inside a member rewrites the member."""
        path = tmp_path / "nested.medh5"
        with Flat.open_writer(path) as w:
            w.document.extra["t"] = {"d": {"a": 1, "b": 2}, "l": [3, 1, 2]}
            member = w.document.extra["t"]
            d, items = member["d"], member["l"]
            d |= {"c": 3}
            d.update(e=5)
            d.setdefault("g", [0]).append(1)
            assert d.pop("a") == 1 and d.pop("absent", None) is None
            assert d.popitem() == ("g", [0, 1])
            del d["b"]
            items.insert(0, 4)
            items.remove(1)
            items.sort()
            items += [7]
            items *= 2
            items.extend([8])
            items.reverse()
            assert items.pop() == 2
            items[0] = 9
            items[1:3] = [10, 11]
            del items[-1]
            expected = {"d": {"c": 3, "e": 5}, "l": [9, 10, 11, 3, 2, 7, 4]}
            assert w.document.extra["t"] == expected
            w.document.extra["u"] = {"d": {"a": 1}, "l": [1]}
            w.document.extra["u"]["d"].clear()
            w.document.extra["u"]["l"].clear()
        with medh5.open(path) as sample:
            assert sample.document.extra == {"t": expected, "u": {"d": {}, "l": []}}

    @pytest.mark.parametrize(
        "edit",
        [
            lambda d, items: d.__setitem__("c", 3),
            lambda d, items: d.__delitem__("a"),
            lambda d, items: d.update(c=3),
            lambda d, items: d.__ior__({"c": 3}),
            lambda d, items: d.setdefault(f"k{len(d)}", 3),
            lambda d, items: d.pop(next(iter(d))),
            lambda d, items: d.popitem(),
            lambda d, items: d.clear(),
            lambda d, items: items.__setitem__(0, 9),
            lambda d, items: items.__setitem__(slice(0, 2), []),
            lambda d, items: items.__delitem__(0),
            lambda d, items: items.__iadd__([9]),
            lambda d, items: items.__imul__(2),
            lambda d, items: items.append(9),
            lambda d, items: items.extend([9]),
            lambda d, items: items.insert(0, 9),
            lambda d, items: items.pop(),
            lambda d, items: items.remove(1),
            lambda d, items: items.clear(),
            lambda d, items: items.sort(),
            lambda d, items: items.reverse(),
        ],
    )
    def test_S2_4_every_nested_edit_to_a_member_changed_since_is_refused(
        self, tmp_path: Path, edit
    ):
        """Each edit refuses to write a stale member back; once refused, the
        member's containers refuse every edit before changing anything."""
        with Flat.open_writer(tmp_path / "stale-every.medh5") as w:
            w.document.extra["t"] = {"d": {"a": 1, "b": 2}, "l": [2, 1, 3]}
            member = w.document.extra["t"]
            d, items = member["d"], member["l"]
            changed = {"d": {"a": 5}, "l": [5]}
            w.document.extra["t"] = changed
            with pytest.raises(RuntimeError, match="has changed since it was read"):
                edit(d, items)
            held = (dict(d), list(items))
            with pytest.raises(RuntimeError, match="read it again"):
                edit(d, items)
            assert (dict(d), list(items)) == held
            assert w.document.extra["t"] == changed

    def test_S2_4_an_update_refused_partway_shows_what_was_written(
        self, tmp_path: Path
    ):
        """The members before the refused value are in the document, and the
        view showed none of them."""
        with Flat.open_writer(tmp_path / "partway.medh5") as w:
            extra = w.document.extra
            with pytest.raises(TypeError, match="not JSON serializable"):
                extra.update({"a": 1, "b": {"x", "y"}})
            assert extra == w.document.extra == {"a": 1}

    def test_S2_4_a_sort_that_fails_writes_nothing_back(self, tmp_path: Path):
        """A sort that fails partway leaves a list part-sorted, and the next
        edit wrote that order into the document."""
        with Flat.open_writer(tmp_path / "sort.medh5") as w:
            w.document.extra["t"] = [1, 3, 2, 4, "a"]
            items = w.document.extra["t"]
            with pytest.raises(TypeError):
                items.sort()
            items.append(0)
            assert w.document.extra["t"] == items == [1, 3, 2, 4, "a", 0]

    def test_S2_4_a_copy_of_a_member_is_a_plain_detached_dict(self, tmp_path: Path):
        import copy
        import pickle

        with Flat.open_writer(tmp_path / "copy.medh5") as w:
            w.document.extra["mytool"] = {"a": [1]}
            for copied in (
                pickle.loads(pickle.dumps(w.document.extra)),
                copy.copy(w.document.extra),
                copy.deepcopy(w.document.extra),
            ):
                assert type(copied) is dict
                assert copied == {"mytool": {"a": [1]}}
            detached = copy.deepcopy(w.document.extra)
            detached["mytool"]["a"].append(2)
            assert w.document.extra == {"mytool": {"a": [1]}}

    def test_S2_4_a_value_set_is_held_as_converted(self, tmp_path: Path):
        """The view holds what the document holds, not the object handed in."""
        with Flat.open_writer(tmp_path / "held.medh5") as w:
            given = {"shape": (2, 3), "scale": np.float32(0.5)}
            w.document.extra["mytool"] = given
            given["shape"] = (9, 9)
            assert w.document.extra["mytool"] == {"shape": [2, 3], "scale": 0.5}

    def test_S2_4_an_edit_to_a_records_dict_reaches_the_file(self, tmp_path: Path):
        """What the writer returns, and what its document hands out, is the
        document's record: 1.x returned the document's own object."""
        from medh5.curation import Activity, Agent

        path = tmp_path / "records.medh5"
        with Flat.open_writer(path) as w:
            w.identity(sex="F").extra["consent"] = "research-only"
            w.document.identity.extra["trial"] = {"arm": "A"}
            w.document.identity.extra["trial"]["visits"] = [1]
            w.document.identity.extra["trial"]["visits"].append(2)
            w.cohort(site_id="site-B").extra["batch"] = 7
            review = w.activity("review", agent=w.person("RAD-08"), activity_id="r1")
            review.params["window"] = [1, 2]
            w.document.provenance.activity("r1").params["level"] = 3
            graph = w.document.provenance
            graph.add_agent(Agent("p9", "person", "Second reader"))
            added = graph.add_activity(Activity("r2", "review", agent="p9"))
            added.params["tool"] = "brush"
            w.document.timepoints["tp0"].series_uids["CT"] = "1.2.840.99"
            quality = w.set_quality(
                "CT", agreement=[{"metric": "dice", "value": 0.8, "against": "r2"}]
            )
            quality.agreement[0].per_class["1"] = 0.8
            w.document.quality["CT"].agreement[0].per_class["2"] = 0.7
            w.acquisition("CT", kvp=120)["mA"] = 300
        with medh5.open(path) as sample:
            document = sample.document
            assert document.identity.extra == {
                "consent": "research-only",
                "trial": {"arm": "A", "visits": [1, 2]},
            }
            assert document.cohort.extra == {"batch": 7}
            assert document.provenance.activity("r1").params == {
                "window": [1, 2],
                "level": 3,
            }
            assert document.provenance.activity("r2").params == {"tool": "brush"}
            assert document.provenance.has_agent("p9")
            assert document.timepoints["tp0"].series_uids == {"CT": "1.2.840.99"}
            assert document.quality["CT"].agreement[0].per_class == {"1": 0.8, "2": 0.7}
            assert document.acquisition == {"CT": {"kvp": 120, "mA": 300}}

    def test_S2_4_a_record_given_to_the_writer_carries_its_edits(self, tmp_path: Path):
        from medh5.curation import Activity, Identity, Timeline, Timepoint

        path = tmp_path / "given.medh5"
        with Flat.open_writer(path) as w:
            identity = Identity(sample_id="given", subject_id="p")
            identity.extra["consent"] = "research-only"
            w.document.identity = identity
            point = Timepoint("tp0", 0)
            point.series_uids["CT"] = "1.2.840.99"
            w.document.timepoints = Timeline([point])
            activity = Activity("r1", "import", agent=w.person("RAD-08").id)
            activity.params["source"] = "PACS"
            w.document.provenance.add_activity(activity)
        with medh5.open(path) as sample:
            document = sample.document
            assert document.identity.extra == {"consent": "research-only"}
            assert document.timepoints["tp0"].series_uids == {"CT": "1.2.840.99"}
            assert document.provenance.activity("r1").params == {"source": "PACS"}

    def test_S2_4_an_edit_to_a_record_changed_since_is_refused(self, tmp_path: Path):
        """Writing the stale record back would undo the other change."""
        with Flat.open_writer(tmp_path / "stale.medh5") as w:
            held = w.document.identity
            w.identity(sex="F")
            with pytest.raises(RuntimeError, match="`document.identity` has changed"):
                held.extra["consent"] = "research-only"
            with pytest.raises(RuntimeError, match="was read before an edit"):
                held.extra["consent"] = "research-only"
            w.document.identity.extra["consent"] = "research-only"  # read again
            assert w.document.identity.sex == "F"
            assert w.document.identity.extra == {"consent": "research-only"}

    def test_S5_2_an_edit_to_a_class_property_reaches_the_file(self, tmp_path: Path):
        """The label set the writer returns, and the one its document hands
        out, is the document's: 1.x stored and returned the set it was given."""
        path = tmp_path / "labels.medh5"
        with Flat.open_writer(path) as w:
            given = Flat.label_set()
            given["liver"].properties["paired"] = False
            returned = w.label_set(given)
            assert returned is not given
            returned["liver"].properties["window"] = [-150, 250]
            lesion = w.document.label_set["lesion"]
            lesion.properties["is_lesion"] = True
            lesion.properties["grading"] = {"scale": "LI-RADS"}
            lesion.properties["grading"]["levels"] = 5
        with medh5.open(path) as sample:
            labels = sample.label_set
            assert labels["liver"].properties == {
                "paired": False,
                "window": [-150, 250],
            }
            assert labels["lesion"].properties == {
                "is_lesion": True,
                "grading": {"scale": "LI-RADS", "levels": 5},
            }
            assert labels.sha256 == sample.document.to_json()["label_set"]["sha256"]

    def test_S2_4_an_edit_to_a_class_changed_since_is_refused(self, tmp_path: Path):
        """Writing the stale class back would undo the other change."""
        with Flat.open_writer(tmp_path / "stale-class.medh5") as w:
            held = w.document.label_set["liver"]
            w.document.label_set["liver"].properties["window"] = [-150, 250]
            with pytest.raises(
                RuntimeError, match=r"`document.label_set\['liver'\]` has changed"
            ):
                held.properties["paired"] = False
            w.document.label_set["liver"].properties["paired"] = False  # read again
            assert w.document.label_set["liver"].properties == {
                "window": [-150, 250],
                "paired": False,
            }

    def test_S2_4_a_record_read_and_dropped_is_collected(self, tmp_path: Path):
        """A record, or a label class, and the dicts that write through it
        refer to each other; the cycle is the collector's to free."""
        import gc

        from medh5._live import _Anchor

        def anchors() -> int:
            gc.collect()
            return sum(isinstance(o, _Anchor) for o in gc.get_objects())

        with Flat.open_writer(tmp_path / "gc.medh5") as w:
            before = anchors()
            for n in range(50):
                w.document.identity.extra["n"] = n
                w.document.timepoints["tp0"].series_uids["CT"] = str(n)
                w.document.label_set["liver"].properties["n"] = n
            assert anchors() == before


_BOXES = np.array([[[0, 2], [1, 3], [2, 4]]], float)

# One call per annotation writer, each taking the keyword that refuses it.
_ANNOTATION_WRITES: dict[str, Any] = {
    "boxes": lambda w, **kw: w.add_boxes("a", _BOXES, [1], grid="g", **kw),
    "obb": lambda w, **kw: w.add_obb(
        "a", np.ones((1, 3)), np.ones((1, 3)), np.eye(3)[None], [1], grid="g", **kw
    ),
    "keypoints": lambda w, **kw: w.add_keypoints(
        "a", np.ones((1, 2, 3)), [1, 3], [1], grid="g", **kw
    ),
    "points": lambda w, **kw: w.add_points("a", np.ones((2, 3)), grid="g", **kw),
    "contours": lambda w, **kw: w.add_contours(
        "a", [Polygon(np.ones((3, 3)), 1, (0, 1))], grid="g", **kw
    ),
    "mesh": lambda w, **kw: w.add_mesh(
        "a", np.eye(3), np.array([[0, 1, 2]]), grid="g", space="index", **kw
    ),
    "classification": lambda w, **kw: w.add_classification("a", {1: 1.0}, **kw),
    "segmentation": lambda w, **kw: w.add_segmentation(
        "a", grid="g", masks={1: Flat.mask()}, **kw
    ),
    "mask": lambda w, **kw: w.add_mask("a", Flat.mask(), grid="g", **kw),
}

# Refused before anything is written, and refused partway: a string HDF5
# cannot store fails the attribute write after the datasets are written.
_REFUSALS = {"codec": {"codec": "nope"}, "attribute": {"prov": "act\x00ivity"}}


class TestRefusedWrites:
    """§14.4: a write the builder refuses leaves nothing behind.  Refused once
    its group was made, an annotation, transform or grid stayed half-written:
    a corrected retry found its name taken, and the commit refused the file
    (E412) --- or kept the object without the attributes it failed to write."""

    @pytest.mark.parametrize("refusal", sorted(_REFUSALS))
    @pytest.mark.parametrize("kind", sorted(_ANNOTATION_WRITES))
    def test_S6_2_a_refused_annotation_leaves_nothing_behind(
        self, tmp_path: Path, kind: str, refusal: str
    ):
        path = tmp_path / "ann.medh5"
        write = _ANNOTATION_WRITES[kind]
        with Flat.open_writer(path) as w:
            with pytest.raises(ValueError):
                write(w, **_REFUSALS[refusal])
            assert list(w.handle["annotations"].keys()) == []
            write(w)
        with medh5.open(path) as sample:
            assert list(sample.annotations) == ["a"]
        assert validate_file(path).ok

    @pytest.mark.parametrize("refusal", sorted(_REFUSALS))
    def test_S10_1_a_refused_transform_leaves_nothing_behind(
        self, tmp_path: Path, refusal: str
    ):
        """The `transforms` group the refused write made goes with it."""
        path = tmp_path / "tr.medh5"

        def write(w: Any, **kw: Any) -> None:
            w.add_transform(
                "t",
                kind="affine",
                from_frame="f0",
                to_frame="f1",
                matrix=np.eye(4),
                **kw,
            )

        with Flat.open_writer(path) as w:
            with pytest.raises(ValueError):
                write(w, **_REFUSALS[refusal])
            assert "transforms" not in w.handle
            write(w)
        with medh5.open(path) as sample:
            assert list(sample.transforms) == ["t"]
        assert validate_file(path).ok

    def test_S3_2_a_grid_that_cannot_be_written_whole_is_not_written(
        self, tmp_path: Path
    ):
        path = tmp_path / "grid.medh5"
        with Flat.open_writer(path) as w:
            with pytest.raises(ValueError):
                w.add_grid("h", shape=Flat.SHAPE, spacing=(1, 1, 1), frame_uid="f\x00")
            w.add_grid("h", shape=Flat.SHAPE, spacing=(1, 1, 1), frame_uid="f1")
        with medh5.open(path) as sample:
            assert sample.grids["h"].frame_uid == "f1"
        assert validate_file(path).ok

    def test_S11_2_a_refused_write_files_no_quality_record(self, tmp_path: Path):
        """A record given inline is filed with what it describes: filed first,
        a refused write left it behind, and a refused duplicate replaced the
        record of the object already written under its id."""
        path = tmp_path / "quality.medh5"
        affine = {"kind": "affine", "from_frame": "f0", "to_frame": "f1"}
        with Flat.open_writer(path) as w:
            with pytest.raises(MEDH5ValidationError) as exc:
                w.add_boxes(
                    "a",
                    _BOXES,
                    [1],
                    grid="g",
                    closure="nope",
                    quality={"status": "draft"},
                )
            assert exc.value.code == "E412"
            assert "a" not in w.document.quality
            w.add_boxes("a", _BOXES, [1], grid="g", quality={"status": "approved"})
            w.add_transform(
                "t", matrix=np.eye(4), metrics={"status": "approved"}, **affine
            )
            with pytest.raises(MEDH5ValidationError, match="already exists"):
                w.add_boxes("a", _BOXES, [1], grid="g", quality={"status": "rejected"})
            with pytest.raises(MEDH5ValidationError, match="already exists"):
                w.add_transform(
                    "t", matrix=np.eye(4), metrics={"status": "rejected"}, **affine
                )
        with medh5.open(path) as sample:
            quality = sample.document.quality
            assert {k: quality[k].status for k in quality} == {
                "a": "approved",
                "t": "approved",
            }

    def test_S7_6_a_refused_transcode_keeps_the_annotation(self, tmp_path: Path):
        """The annotation was removed before its new encoding was written, so
        a refused transcode lost it, and the commit refused the half-written
        group in its place (E412)."""
        path = tmp_path / "transcode.medh5"
        masks = {1: Flat.mask(), 3: Flat.mask((5, 5, 5))}
        with Flat.open_writer(path) as w:
            w.add_segmentation("a", grid="g", masks=masks, encoding="labelmap")
            w.build_index()
        with medh5.open(path) as sample:
            before = sample.annotations["a"].labelmap()
        with (
            medh5.amend(path) as w,
            pytest.raises(MEDH5ValidationError, match="unknown codec"),
        ):
            w.transcode_annotation("a", "layers", codec="nope")
        with medh5.open(path) as sample:
            annotation = sample.annotations["a"]
            assert annotation.kind == "labelmap"
            np.testing.assert_array_equal(annotation.labelmap(), before)
            assert list(sample.index) == ["a"]
        assert validate_file(path).ok
