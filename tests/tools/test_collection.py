"""Collections: packing, unpacking and the containment claim (spec §2.2)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.collection import (
    SUFFIX,
    Collection,
    default_key,
    extract,
    is_collection,
    open_any,
    open_collection,
    pack,
    unpack,
)
from medh5.errors import MEDH5FileError, MEDH5ValidationError, MEDH5VersionError
from medh5.integrity import raw_chunks, subtrees_identical
from medh5.validate import validate_file
from tests.helpers import encode_attr, write_sample
from tests.kits import Flat, Organs


@pytest.fixture
def members(tmp_path: Path, label_set, masks) -> list[Path]:
    return [
        write_sample(
            tmp_path / f"case_{i}.medh5",
            label_set=label_set,
            masks=masks,
            sample_id=f"case_{i}",
            codec="balanced",
        )
        for i in range(3)
    ]


@pytest.fixture
def shard(tmp_path: Path, members: list[Path]) -> Path:
    return pack(members, tmp_path / f"cohort{SUFFIX}")


class TestPack:
    def test_S2_2_keys_default_to_the_file_stem(self, shard):
        with open_collection(shard) as collection:
            assert list(collection) == ["case_0", "case_1", "case_2"]
            assert collection.kind == "collection"
            # A shard declares its newest member's version (1.1 §8).
            assert collection.version == "1.0"
            assert "3 samples" in repr(collection)

    def test_explicit_keys_are_used_in_order(self, tmp_path, members):
        path = pack(members, tmp_path / f"k{SUFFIX}", keys=["a", "b.1", "c-2"])
        with open_collection(path) as collection:
            assert list(collection) == ["a", "b.1", "c-2"]

    def test_duplicate_keys_are_refused(self, tmp_path, members):
        with pytest.raises(MEDH5ValidationError) as exc:
            pack(members, tmp_path / f"d{SUFFIX}", keys=["a", "a", "b"])
        assert exc.value.code == "E003"
        assert "not unique" in str(exc.value)

    def test_key_count_must_match(self, tmp_path, members):
        with pytest.raises(MEDH5ValidationError):
            pack(members, tmp_path / f"d{SUFFIX}", keys=["only-one"])

    def test_packing_nothing_is_refused(self, tmp_path):
        with pytest.raises(MEDH5ValidationError):
            pack([], tmp_path / f"empty{SUFFIX}")

    def test_a_collection_cannot_be_packed_again(self, tmp_path, shard, members):
        with pytest.raises(MEDH5ValidationError) as exc:
            pack([shard, *members], tmp_path / f"nested{SUFFIX}")
        assert exc.value.code == "E006"

    def test_bad_sample_keys_are_refused(self, tmp_path, members):
        with pytest.raises(MEDH5ValidationError):
            pack(members[:1], tmp_path / f"x{SUFFIX}", keys=["not a key"])
        assert default_key("/x/case_0.medh5") == "case_0"


class TestContainment:
    """§2.2: a sample root in a shard *is* a sample root."""

    def test_members_read_like_standalone_samples(self, shard, masks):
        with open_collection(shard) as collection:
            sample = collection["case_1"]
            assert sample.identity.sample_id == "case_1"
            assert sample.images["CT_tp0"].shape == (16, 24, 24)
            seg = sample.annotations["organs_tp0"]
            for class_id, mask in masks.items():
                assert np.array_equal(seg.dense([class_id])[0], mask)
            assert sample.verify().ok

    def test_S2_2_round_trip_is_byte_identical(self, tmp_path, members, shard):
        """The claim that makes packing safe: chunks move, nothing re-encodes."""
        written = unpack(shard, tmp_path / "out")
        assert [p.name for p in written] == [p.name for p in members]
        for original, extracted in zip(members, written, strict=True):
            with medh5.open(original) as a, medh5.open(extracted) as b:
                assert subtrees_identical(a.root, b.root) == ()

    def test_compressed_chunks_are_copied_not_recompressed(self, tmp_path):
        """A raw-byte check, so a silently dropped filter cannot pass."""
        big = tmp_path / "big.medh5"
        shape = (64, 96, 96)
        rng = np.random.default_rng(7)
        with medh5.create(big, sample_id="big", codec="balanced") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid="g",
                modality="CT",
            )
        with medh5.open(big) as sample:
            source = raw_chunks(sample.images["CT"].dataset)
        assert source, "the fixture must be chunked for this to prove anything"

        shard = pack([big], tmp_path / f"one{SUFFIX}")
        with open_collection(shard) as collection:
            assert raw_chunks(collection["big"].images["CT"].dataset) == source
        written = unpack(shard, tmp_path / "out")
        with medh5.open(written[0]) as sample:
            assert raw_chunks(sample.images["CT"].dataset) == source

    def test_content_id_survives_the_round_trip(self, tmp_path, members, shard):
        with medh5.open(members[0]) as before:
            original = before.content_id
        with open_collection(shard) as collection:
            assert collection["case_0"].content_id == original
        written = unpack(shard, tmp_path / "out", keys=["case_0"])
        with medh5.open(written[0]) as after:
            assert after.content_id == original
            assert after.kind == "sample"

    def test_extracted_members_validate_as_samples(self, tmp_path, shard):
        written = unpack(shard, tmp_path / "out")
        for path in written:
            assert validate_file(path, level="integrity").ok


class TestUnpack:
    def test_selecting_keys(self, tmp_path, shard):
        written = unpack(shard, tmp_path / "some", keys=["case_2"])
        assert [p.stem for p in written] == ["case_2"]

    def test_unknown_keys_are_named(self, tmp_path, shard):
        with pytest.raises(MEDH5ValidationError) as exc:
            unpack(shard, tmp_path / "out", keys=["ghost"])
        assert "ghost" in str(exc.value)

    def test_extract_writes_one_file_where_asked(self, tmp_path, shard):
        target = tmp_path / "just_one.medh5"
        assert extract(shard, "case_1", target) == target
        with medh5.open(target) as sample:
            assert sample.identity.sample_id == "case_1"

    def test_S2_2_extract_touches_nothing_it_was_not_asked_to_write(
        self, tmp_path, shard
    ):
        """Extracting to a new name must not go via the member's own name.

        It used to unpack into the destination's parent --- where `unpack` names
        its output after the key --- and move the result into place afterwards,
        destroying any unrelated file already sitting under that name on the way
        past.  There is no way to get it back.
        """
        outdir = tmp_path / "out"
        outdir.mkdir()
        bystander = outdir / "case_1.medh5"
        bystander.write_bytes(b"someone else's data")

        extract(shard, "case_1", outdir / "renamed.medh5")

        assert bystander.read_bytes() == b"someone else's data"
        with medh5.open(outdir / "renamed.medh5") as sample:
            assert sample.identity.sample_id == "case_1"

    def test_extract_creates_the_directory_it_was_given(self, tmp_path, shard):
        target = tmp_path / "deep" / "nested" / "one.medh5"
        assert extract(shard, "case_1", target).exists()

    def test_extract_names_an_unknown_key(self, tmp_path, shard):
        with pytest.raises(MEDH5ValidationError) as exc:
            extract(shard, "ghost", tmp_path / "x.medh5")
        assert exc.value.code == "E003"


class TestOpening:
    def test_open_any_dispatches_on_kind(self, tmp_path, members, shard):
        with open_any(members[0]) as sample:
            assert sample.identity.sample_id == "case_0"
        with open_any(shard) as collection:
            assert isinstance(collection, Collection)
        with open_any(shard, key="case_1") as member:
            assert member.identity.sample_id == "case_1"

    def test_a_key_on_a_sample_file_is_an_error(self, members):
        with pytest.raises(MEDH5FileError):
            open_any(members[0], key="case_0")

    def test_open_collection_refuses_a_sample(self, members):
        with pytest.raises(MEDH5ValidationError) as exc:
            open_collection(members[0])
        assert exc.value.code == "E006"

    def test_medh5_open_still_reads_a_plain_sample(self, members):
        with medh5.open(members[0]) as sample:
            assert not is_collection(sample.root)

    def test_unknown_member_raises_with_the_known_keys(self, shard):
        with (
            open_collection(shard) as collection,
            pytest.raises(KeyError, match="case_0"),
        ):
            collection["nope"]

    def test_subject_ids_are_what_a_split_groups_by(self, shard):
        with open_collection(shard) as collection:
            assert set(collection.subject_ids().values()) == {"subj-A"}

    def test_summary_is_json_safe(self, shard):

        with open_collection(shard) as collection:
            json.dumps(collection.summary())

    def test_S2_1_a_collection_from_another_major_is_refused(self, tmp_path: Path):
        from medh5.collection import open_any, open_collection, pack

        member = tmp_path / "m.medh5"
        with Flat.open_writer(member):
            pass
        shard = tmp_path / "s.medh5c"
        pack([member], shard)
        with h5py.File(shard, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            open_collection(shard)
        with pytest.raises(MEDH5VersionError):
            open_any(shard)
        with h5py.File(member, "r+") as handle:
            handle.attrs["medh5_version"] = "2.0"
        with pytest.raises(MEDH5VersionError):
            pack([member], tmp_path / "t.medh5c")

    def test_L13_open_collection_takes_no_mode(self, tmp_path: Path):
        import inspect

        from medh5.collection import open_collection, pack

        assert "mode" not in inspect.signature(open_collection).parameters
        path = tmp_path / "one.medh5"
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        shard = tmp_path / "s.medh5c"
        pack([path], shard)
        with open_collection(shard) as collection:
            assert sorted(collection) == ["one"]


class TestValidation:
    def test_a_clean_shard_validates(self, shard):
        report = validate_file(shard, level="integrity")
        assert report.ok, report.format()
        assert report.checked["samples"] == [
            "/samples/case_0",
            "/samples/case_1",
            "/samples/case_2",
        ]

    def test_member_diagnostics_name_the_member(self, tmp_path, shard):
        with h5py.File(shard, "r+") as handle:
            del handle["samples/case_1"].attrs["medh5_profiles"]
        report = validate_file(shard)
        assert not report.ok
        assert any("case_1" in d.location for d in report.errors)
        assert all("case_0" not in d.location for d in report.errors)

    def test_S2_2_a_member_without_content_id_is_E010(self, shard):
        with h5py.File(shard, "r+") as handle:
            del handle["samples/case_0"].attrs["content_id"]
        report = validate_file(shard)
        assert "E010" in report.codes

    def test_a_collection_without_samples_is_E008(self, tmp_path, shard):
        with h5py.File(shard, "r+") as handle:
            del handle["samples"]
        assert "E008" in validate_file(shard).codes
        with pytest.raises(MEDH5ValidationError):
            open_collection(shard)

    def test_an_empty_samples_group_is_E008(self, tmp_path, shard):
        with h5py.File(shard, "r+") as handle:
            for key in list(handle["samples"]):
                del handle["samples"][key]
        assert "E008" in validate_file(shard).codes

    def test_a_bad_sample_key_is_E003(self, shard):
        with h5py.File(shard, "r+") as handle:
            handle["samples"].move("case_0", "not a key")
        assert "E003" in validate_file(shard).codes

    def test_a_shard_missing_its_version_is_E001(self, shard):
        with h5py.File(shard, "r+") as handle:
            del handle.attrs["medh5_version"]
        assert "E001" in validate_file(shard).codes

    def test_a_shard_of_a_future_major_is_E002(self, shard):

        with h5py.File(shard, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("2.0")
        assert "E002" in validate_file(shard).codes


class TestF11GraphCopy:
    """Pack, unpack and amend copy a sample as a graph (F11 of the round-4
    audit).  Copied member by member, a hard link between two members became a
    second dataset --- a new line under ``content_id`` --- and a soft link was
    expanded into one: packed and unpacked, the sample failed ``verify``, and
    a no-op amend changed its address."""

    @staticmethod
    def _linked(path: Path) -> Path:
        write_sample(path, codec="balanced")
        with h5py.File(path, "r+") as f:
            f["zzz_alias"] = f["images/CT_tp0"]  # a later path to one dataset
            f.create_group("x_ext")
            f["x_ext/loop"] = f["/"]  # a cycle through the root
            f["x_soft"] = h5py.SoftLink("/images/CT_tp0")
            f["x_ext/relative"] = h5py.SoftLink("loop")
        return path

    @staticmethod
    def _check(group: h5py.Group, root: str) -> None:
        """The graph `_linked` made, under a sample root at `root`."""
        assert group["zzz_alias"].id == group["images/CT_tp0"].id
        assert group["x_ext/loop"].id == group.id
        assert group.get("x_soft", getlink=True).path == f"{root}/images/CT_tp0"
        assert group["x_ext"].get("relative", getlink=True).path == "loop"

    def test_F11_S2_2_pack_and_unpack_keep_aliases_cycles_and_soft_links(
        self, tmp_path: Path
    ):
        source = self._linked(tmp_path / "linked.medh5")
        with medh5.open(source) as s:
            before = s.content_id
            assert before == s.compute_content_id()
        shard = pack([source], tmp_path / f"one{SUFFIX}")
        with h5py.File(shard, "r") as f:
            self._check(f["samples/linked"], "/samples/linked")
        with open_collection(shard) as collection:
            sample = collection["linked"]
            assert sample.content_id == before
            assert sample.verify().ok
        (unpacked,) = unpack(shard, tmp_path / "out")
        with h5py.File(unpacked, "r") as f:
            self._check(f, "")
        with medh5.open(unpacked) as s:
            assert s.content_id == before == s.compute_content_id()
            assert s.verify().ok
        with medh5.open(source) as a, medh5.open(unpacked) as b:
            assert subtrees_identical(a.root, b.root) == ()

    def test_F11_S14_4_a_no_op_amend_keeps_the_graph_and_the_address(
        self, tmp_path: Path
    ):
        path = self._linked(tmp_path / "amended.medh5")
        with medh5.open(path) as s:
            before = s.content_id
        with medh5.amend(path) as w:
            w.commit()
        with h5py.File(path, "r") as f:
            self._check(f, "")
        with medh5.open(path) as s:
            assert s.content_id == before
            assert s.verify().ok

    def test_F11_a_soft_link_out_of_its_member_is_not_unpacked(self, tmp_path: Path):
        """A member's soft link to another member's object has no meaning in
        a file of its own; the unpack is refused rather than repointed."""
        a = write_sample(tmp_path / "a.medh5")
        b = write_sample(tmp_path / "b.medh5")
        shard = pack([a, b], tmp_path / f"two{SUFFIX}")
        with h5py.File(shard, "r+") as f:
            f["samples/b/x_elsewhere"] = h5py.SoftLink("/samples/a/images/CT_tp0")
        with pytest.raises(MEDH5FileError, match="outside its sample root"):
            extract(shard, "b", tmp_path / "b_out.medh5")
        assert not (tmp_path / "b_out.medh5").exists()
