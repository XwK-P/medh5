"""Digests, content ids and index currency (spec §13)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.integrity import (
    array_digest,
    canonical_attrs,
    dataset_digest,
    digest_bytes,
    group_digest,
    parse_digest,
    relative_path,
    stale_index_entries,
    verify_object,
    verify_root,
)
from tests.v1._h5 import encode_attr
from tests.v1.conftest import SHAPE, write_sample


class TestDigests:
    def test_S13_1_covers_path_dtype_shape_and_bytes(self):
        data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        base = array_digest("images/CT", data)
        assert base != array_digest("images/MR", data)
        assert base != array_digest("images/CT", data.astype(np.int32))
        assert base != array_digest("images/CT", data.reshape(4, 3, 2))
        changed = data.copy()
        changed[0, 0, 0] += 1
        assert base != array_digest("images/CT", changed)
        assert base == array_digest("images/CT", data.copy())

    def test_S13_1_is_byte_order_independent(self):
        little = np.arange(8, dtype="<i2")
        big = little.astype(">i2")
        assert array_digest("x", little) == array_digest("x", big)

    def test_S13_1_covers_decompressed_content(self, tmp_path, label_set, masks):
        """Recompression must not invalidate a digest."""
        fast = write_sample(
            tmp_path / "a.medh5",
            label_set=label_set,
            masks=masks,
            codec="training",
            sample_id="same",
        )
        archive = write_sample(
            tmp_path / "b.medh5",
            label_set=label_set,
            masks=masks,
            codec="archive",
            sample_id="same",
        )
        with h5py.File(fast) as a, h5py.File(archive) as b:
            assert (
                a["images/CT_tp0"].attrs["digest"] == b["images/CT_tp0"].attrs["digest"]
            )
            assert a.attrs["content_id"] == b.attrs["content_id"]

    def test_parse_digest_rejects_junk(self):
        assert parse_digest("sha256:ab12") == ("sha256", "ab12")
        for junk in ("sha256:", "md5:abcd", "nonsense", "sha256:zz"):
            with pytest.raises(MEDH5ValidationError) as exc:
                parse_digest(junk)
            assert exc.value.code == "E703"

    def test_unsupported_algorithm(self):
        with pytest.raises(MEDH5ValidationError):
            digest_bytes(b"x", "md5")

    def test_streaming_matches_whole_array(self, sample_path):
        """A stored dataset digests as its in-memory array does.

        The stream budget is the engine's; its Rust test
        (`integrity::digest::tests::s13_1_streaming_matches_the_whole_array`)
        digests at a 128-byte and a 1-byte budget.
        """
        data = np.arange(4096, dtype=np.int16).reshape(64, 64)
        with h5py.File(sample_path, "r+") as handle:
            handle.create_dataset("x_test/d", data=data)
        with medh5.open(sample_path) as sample:
            streamed = dataset_digest(sample.root["x_test/d"], "d")
        assert streamed == array_digest("d", data)

    def test_vlen_and_scalar_datasets(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            group = handle.create_group("x_test")
            group.create_dataset("s", data="hello", dtype=h5py.string_dtype())
            group.create_dataset("n", data=np.int32(7))
            group.create_dataset("e", data=np.zeros((0, 3)))
        with medh5.open(sample_path) as sample:
            group = sample.root["x_test"]
            assert dataset_digest(group["s"], "s").startswith("sha256:")
            assert dataset_digest(group["n"], "n").startswith("sha256:")
            assert dataset_digest(group["e"], "e").startswith("sha256:")

    def test_canonical_attrs_excludes_unlisted(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            group = handle.create_group("x_test")
            group.attrs["kept"] = encode_attr("yes")
            group.attrs["ignored"] = encode_attr("no")
        with medh5.open(sample_path) as sample:
            rendered = canonical_attrs(sample.root["x_test"], ["kept", "absent"])
        assert rendered == '{"kept":"yes"}'

    def test_relative_path_strips_the_sample_root(self, sample_path):
        with medh5.open(sample_path) as sample:
            image = sample.root["images/CT_tp0"]
            assert relative_path(image) == "images/CT_tp0"
            assert relative_path(image, sample.root) == "images/CT_tp0"


class TestVerification:
    def test_a_clean_file_verifies(self, sample_path):
        with medh5.open(sample_path) as sample:
            result = sample.verify()
            assert result.ok
            assert result.content_id_ok is True
            assert not result.undigested
            assert result.summary()["checked"] == len(result.checked)

    def test_S13_2_a_mismatch_names_the_object(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(0 for _ in block.shape)] = 4
            data[...] = block
        with medh5.open(sample_path) as sample:
            result = sample.verify()
            assert result.mismatched == ("annotations/organs_tp0/data",)
            assert not result.ok
            assert result.content_id_ok is True  # the digest *list* is intact

    def test_S13_2_partial_verification(self, sample_path):
        with medh5.open(sample_path) as sample:
            result = sample.verify(partial=["images/CT_tp0"])
            assert result.checked == ("images/CT_tp0",)
            assert result.content_id_computed is None

    def test_content_id_detects_a_rewritten_digest(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle["images/CT_tp0"].attrs["digest"] = encode_attr("sha256:" + "0" * 64)
        with medh5.open(sample_path) as sample:
            result = sample.verify()
            assert result.content_id_ok is False

    def test_verify_object(self, sample_path):
        with medh5.open(sample_path) as sample:
            assert verify_object(sample.root, "images/CT_tp0")
            assert not verify_object(sample.root, "meta")

    def test_S13_2_content_id_is_a_content_address(self, tmp_path, label_set, masks):
        """Two identical samples written separately share a content_id."""
        a = write_sample(
            tmp_path / "a.medh5", label_set=label_set, masks=masks, sample_id="same"
        )
        b = write_sample(
            tmp_path / "b.medh5", label_set=label_set, masks=masks, sample_id="same"
        )
        with h5py.File(a) as fa, h5py.File(b) as fb:
            assert fa.attrs["content_id"] == fb.attrs["content_id"]

    def test_content_id_covers_the_document(self, tmp_path, label_set, masks):
        """A different sample_id is different content, so a different address."""
        a = write_sample(
            tmp_path / "a.medh5", label_set=label_set, masks=masks, sample_id="one"
        )
        b = write_sample(
            tmp_path / "b.medh5", label_set=label_set, masks=masks, sample_id="two"
        )
        with h5py.File(a) as fa, h5py.File(b) as fb:
            assert fa.attrs["content_id"] != fb.attrs["content_id"]

    def test_content_id_changes_with_content(self, tmp_path, label_set, masks):
        a = write_sample(
            tmp_path / "a.medh5", label_set=label_set, masks=masks, sample_id="same"
        )
        other = {k: v.copy() for k, v in masks.items()}
        other[1][0, 0, 0] = True
        b = write_sample(
            tmp_path / "b.medh5", label_set=label_set, masks=other, sample_id="same"
        )
        with h5py.File(a) as fa, h5py.File(b) as fb:
            assert fa.attrs["content_id"] != fb.attrs["content_id"]


class TestIndexCurrency:
    def test_S13_3_a_fresh_index_is_current(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            assert stale_index_entries(sample.root) == ()

    def test_S13_3_a_changed_annotation_makes_it_stale(self, longitudinal_path):
        with h5py.File(longitudinal_path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        with medh5.open(longitudinal_path) as sample:
            assert stale_index_entries(sample.root) == ("organs_tp0",)

    def test_group_digest_tracks_every_dataset(self, sample_path):
        def digest() -> str:
            with medh5.open(sample_path) as sample:
                group = sample.root["annotations/organs_tp0"]
                return str(group_digest(group, root=sample.root))

        before = digest()
        with h5py.File(sample_path, "r+") as handle:
            handle["annotations/organs_tp0"].create_dataset(
                "scratch", data=np.arange(3)
            )
        assert digest() != before

    def test_index_without_a_source_digest_is_stale(self, longitudinal_path):
        with h5py.File(longitudinal_path, "r+") as handle:
            del handle["index/organs_tp0"].attrs["source_digest"]
        with medh5.open(longitudinal_path) as sample:
            assert "organs_tp0" in stale_index_entries(sample.root)


class TestVerifyRoot:
    def test_undigested_datasets_are_listed(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle["images"].create_dataset("scratch", data=np.zeros(SHAPE))
        with medh5.open(sample_path) as sample:
            result = verify_root(sample.root)
        assert "images/scratch" in result.undigested

    def test_malformed_digest_is_reported(self, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            handle["images/CT_tp0"].attrs["digest"] = encode_attr("garbage")
        with medh5.open(sample_path) as sample:
            result = verify_root(sample.root, check_content_id=False)
        assert "images/CT_tp0" in result.malformed
