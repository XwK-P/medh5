"""What the writer accepts, refuses and checks at every door (spec §14.4, §15).

The writer is held to the validator: a rule the validator enforces is refused
at the call or at commit, and every tool that rewrites a file --- amend, scrub,
fix, recompress, unpack --- passes the same gate.  Class names cite the audit
finding a test reproduces (``W14``, ``L30``, ...), as the changelog entry of the
release that fixed it does.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.errors import MEDH5FileError, MEDH5ValidationError, MEDH5VersionError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file
from tests.kits import Flat, Framed, Numbered, Organs


class TestW2WriterSideRefusals:
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

    def test_F07_S7_7_the_sibling_mask_validates_and_silences_W904(
        self, tmp_path: Path
    ):
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

    def test_F07_ignore_and_ignore_mask_together_are_refused(self, tmp_path: Path):
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

    def test_F07_encode_voxels_refuses_what_one_payload_cannot_carry(self):
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

    def test_F07_S7_6_selection_costs_the_widening_an_ignore_forces(self):
        """An in-band ignore forces `uint16` planes, so the choice must see it."""
        from medh5.annotations.voxel import select_encoding

        masks = {c: np.zeros(Organs.SHAPE, dtype=bool) for c in range(1, 4)}
        for c, mask in masks.items():
            mask[c] = True
        _, plain = select_encoding(masks, Organs.SHAPE)
        _, widened = select_encoding(masks, Organs.SHAPE, ignore=True)
        from medh5.annotations.voxel import cost_model

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

    def test_F09_an_explicit_duplicate_id_raises(self, tmp_path: Path):
        with Organs.writer(tmp_path / "dup.medh5") as w:
            alice = w.person("Alice", agent_id="s2")
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.person("Bob", agent_id="s2")
            w.activity("import", activity_id="act1", agent=alice)
            with pytest.raises(MEDH5ValidationError, match="already declared"):
                w.activity("annotate", activity_id="act1", agent=alice)
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())

    def test_F09_replace_is_available_for_the_one_legitimate_rewrite(self):
        from medh5.curation import Agent, Provenance

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

    def test_F13_TRANSCODABLE_is_the_whole_truth(self):
        from medh5.annotations.voxel import (
            TRANSCODABLE,
            encode_masks,
            transcode_payload,
        )

        assert "mask" not in TRANSCODABLE
        payload = encode_masks(Organs.blocks(), "layers", Organs.SHAPE)
        with pytest.raises(MEDH5ValidationError):
            transcode_payload(payload, "mask", spatial_shape=Organs.SHAPE)


class TestW14RewriteGate:
    """F-20, F-22, L-27, L-31, Q-15: what every door into a file checks."""

    def test_F20_S16_amend_refuses_a_future_major(self, tmp_path: Path):
        """`open` refused a 2.0 file; `amend` restamped it 1.0 and carried on."""
        path = Numbered.plain(tmp_path / "v2.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        before = path.read_bytes()
        with pytest.raises(MEDH5VersionError):
            medh5.amend(path)
        assert path.read_bytes() == before

    def test_F20_S16_amend_keeps_a_later_minor(self, tmp_path: Path):
        """A 1.1 file's objects are carried through, so it stays a 1.1 file."""
        path = Numbered.plain(tmp_path / "v11.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "1.1"
            handle.create_group("x_future")
        with medh5.amend(path):
            pass
        with h5py.File(path, "r") as handle:
            assert handle.attrs["medh5_version"] == "1.1"
            assert "x_future" in handle

    def test_F20_the_scrub_and_fix_doors_are_the_same_gate(self, tmp_path: Path):
        from medh5.curation import scrub

        path = Numbered.plain(tmp_path / "v2s.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            scrub.apply(path)

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

    def test_F22_external_storage_is_refused_by_every_tool(
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

    def test_F22_an_external_link_is_refused(self, tmp_path: Path):
        other = Numbered.plain(tmp_path / "other.medh5")
        path = Numbered.plain(tmp_path / "linked.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_link"] = h5py.ExternalLink(str(other), "/images")
        with pytest.raises(MEDH5FileError, match="external link"):
            medh5.open(path)

    def test_F22_a_virtual_dataset_is_refused(self, tmp_path: Path):
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
    def test_F22_the_check_is_remembered_for_the_file_it_read(
        self, tmp_path: Path, secret: Path
    ):
        """A path replaced between the open and the check must not inherit it.

        The memo was keyed by a stat of the *path*, taken after the open.  A
        replacement landing in between was recorded as checked while the handle
        scanned the old file, and its next open skipped the check.  The race
        itself is the engine's to hold (`h5::ops::tests::
        f22_the_check_is_remembered_for_the_file_it_read`); through the public
        door, a file checked and remembered does not vouch for its replacement.
        """
        target = Numbered.plain(tmp_path / "target.medh5")
        bad = self._external_storage(tmp_path, secret)
        with medh5.open(target) as sample:  # checked, and remembered
            os.replace(bad, target)  # the path now names another file
            assert "CT" in sample.images  # the file read is clean
        with pytest.raises(MEDH5FileError, match="not self-contained"):
            medh5.open(target)

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
        reason="POSIX file modes, which root ignores",
    )
    def test_Q15_a_read_only_sample_can_still_be_amended(self, tmp_path: Path):
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

    def test_L31_unpack_refuses_a_member_name_that_is_a_path(self, tmp_path: Path):
        from medh5.collection import pack, unpack

        shard = tmp_path / "s.medh5c"
        pack([Numbered.plain(tmp_path / "a.medh5")], shard, keys=["good"])
        with h5py.File(shard, "r+") as handle:
            handle.copy("samples/good", handle["samples"], name="..\\..\\evil")
        with pytest.raises(MEDH5ValidationError, match="evil"):
            unpack(shard, tmp_path / "out")
        assert not list(tmp_path.rglob("evil*"))

    def test_L27_S14_amend_keeps_a_portable_file_portable(self, tmp_path: Path):
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
    def test_Q15_the_temporary_file_is_never_more_permissive(self, tmp_path: Path):
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


class TestW19WriterEqualsValidator:
    """L-19...L-25, L-37, S-13: one definition of a valid file."""

    def test_L19_S10_3_a_field_off_its_grid_is_refused_at_the_call(
        self, tmp_path: Path
    ):
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

    def test_L19_commit_refuses_what_the_validator_rejects(self, tmp_path: Path):
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

    def test_L20_S5_1_a_ref_label_set_is_written(self, tmp_path: Path):
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

    def test_L21_the_schema_check_needs_no_optional_dependency(self):
        """1.4.1 made jsonschema a core dependency so E005 was checked
        everywhere.  2.0 compiles the validator into the engine: NumPy is the
        only runtime dependency, and the check is always there."""
        from importlib.metadata import requires

        from medh5.document import validate_against_schema

        core = [r for r in requires("medh5") or () if "extra ==" not in r]
        assert [re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0] for r in core] == ["numpy"]
        assert validate_against_schema({"identity": {}})

    def test_L21_S2_4_commit_checks_the_schema(self, tmp_path: Path):
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
    def test_L22_S3_2_units_outside_the_list_are_refused(
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

    def test_L23_one_source_and_a_matching_encoding(self, tmp_path: Path):
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
    def test_L24_S7_7_an_overlapping_region_reads_back_whole(
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

    def test_L24_a_region_clear_of_the_classes_stays_in_band(self, tmp_path: Path):
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

    def test_L25_S7_4_a_transcode_does_not_drop_identity_silently(self, tmp_path: Path):
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

    def test_L25_the_cli_flag(self, tmp_path: Path, capsys):
        from medh5.cli import main

        path = self._lesions(tmp_path / "cli.medh5")
        args = ["seg", "convert", str(path), "les", "--to", "layers"]
        assert main(args) != 0
        assert main([*args, "--drop-identity"]) == 0
        capsys.readouterr()
        with medh5.open(path) as sample:
            assert sample.annotations["les"].kind == "layers"

    def test_L37_S7_4_examined_and_none_found(self, tmp_path: Path):
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

    def test_L37_an_empty_list_alone_says_nothing_and_is_refused(self, tmp_path: Path):
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
        and the two drifted (L-19...L-22).  With ``commit()`` running the
        validator there is one list --- which this checks from the outside, over
        the conformance corpus: a no-op amend of each of its 117 files either
        refuses the file or writes one the validator passes, and a valid case
        comes back valid.
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
            except (MEDH5Error, OSError):
                refused += 1
                assert not case.valid or case.suffix == ".medh5c", case.name
                continue
            written += 1
            errors = [c for c in validate_file(path).codes if c.startswith("E")]
            assert errors == [], (case.name, errors)
        assert refused and written


class TestL30DuplicateIds:
    """L-30: two nodes with one id are refused, not merged."""

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
        return path

    @pytest.mark.parametrize("section", ["agents", "activities"])
    def test_L30_a_duplicated_provenance_id_is_refused_on_read(
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

    def test_L30_S7_4_the_writer_refuses_two_objects_sharing_an_id(
        self, tmp_path: Path
    ):
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

    def test_L30_S7_4_the_validator_reports_a_shared_id_as_E404(self, tmp_path: Path):
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

    def test_L30_the_W909_corpus_case_is_the_sample_scoped_conflict(
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
