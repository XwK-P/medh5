"""Digests, content ids and index currency (spec §13)."""

from __future__ import annotations

from pathlib import Path

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
from medh5.validate import validate_file
from tests.helpers import SHAPE, encode_attr, write_sample
from tests.kits import Flat, Framed, Organs


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

    def test_S2_1_an_unknown_digest_algo_is_E703_everywhere(self, tmp_path: Path):
        path = tmp_path / "algo.medh5"
        with Flat.open_writer(path):
            pass
        with h5py.File(path, "r+") as handle:
            handle.attrs["digest_algo"] = "blake3"
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.compute_content_id()
            assert exc.value.code == "E703"
            with pytest.raises(MEDH5ValidationError) as exc:
                sample.verify()
            assert exc.value.code == "E703"

        report = validate_file(path, level="integrity")
        assert "E703" in report.codes
        # The pass still reports, rather than dying inside the recompute.
        assert "E001" not in report.codes


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


class TestW10Integrity:
    """L-12, L-17: a check that checks, and an amend that stays addressed."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    def test_L12_S13_1_recompress_verifies_what_it_wrote(self, tmp_path: Path):
        """`content_id_preserved` compared the attribute it had just copied.

        True by construction --- including on a file whose bytes were corrupted
        before the run, which reported "content_id yes", exited 0, and failed
        `verify()` immediately afterwards.
        """
        import h5py

        from medh5.storage import recompress

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
        writer = Organs.writer(path)
        writer.add_segmentation("seg", grid="g", masks=Organs.blocks())
        writer.commit(digests=False)
        with medh5.open(path) as sample:
            assert not sample.verify().undigested
            assert sample.content_id is not None


class TestL26Verify:
    """L-26: an undigested dataset inside an attested object fails `verify`."""

    def _boxes(self, path: Path) -> Path:
        with Framed.writer(path) as w:
            w.add_boxes(
                "det", np.array([Framed.box(1, 3)], np.float32), ["c1"], grid="g"
            )
        return path

    def test_L26_S13_2_an_undigested_dataset_in_an_annotation_fails_verify(
        self, tmp_path: Path
    ):
        """`instance_ids` added without a digest verified, and `tracks()` used it."""
        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        with medh5.open(path) as s:
            result = s.verify()
            assert not result.ok
            assert result.unattested == ("annotations/det/instance_ids",)
            assert result.summary()["unattested"] == ["annotations/det/instance_ids"]

    def test_L26_recompress_names_the_unattested_dataset(self, tmp_path: Path, capsys):
        """It failed with `FAILED (0)` and no path, in the table and the JSON."""
        from medh5.cli import main
        from medh5.storage import recompress

        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        result = recompress(path, "portable")
        assert not result.ok and result.mismatched == []
        assert result.unattested == ["annotations/det/instance_ids"]
        assert result.to_json()["unattested"] == ["annotations/det/instance_ids"]
        assert main(["recompress", str(path), "--profile", "portable"]) == 1
        out = capsys.readouterr().out
        assert "FAILED (1)" in out
        assert "UNSIGNED  annotations/det/instance_ids" in out

    def test_L26_fix_counts_it_as_needing_digests(self, tmp_path: Path):
        from medh5.integrity import diagnose

        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        diagnosis = diagnose(path)
        assert diagnosis.needs_digests
        assert diagnosis.unattested == ("annotations/det/instance_ids",)

    def test_L26_the_writers_files_have_none_and_an_unaddressed_file_is_not_judged(
        self, tmp_path: Path
    ):
        path = self._boxes(tmp_path / "det.medh5")
        with medh5.open(path) as s:
            assert s.verify().ok and s.verify().unattested == ()
        with h5py.File(path, "r+") as handle:
            del handle.attrs["content_id"]
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        with medh5.open(path) as s:
            # No `content_id` makes no claim to check (§13.2 is a SHOULD).
            assert s.verify().unattested == ()


class TestF13CyclicCompare:
    """F13 of the round-4 audit: comparing a tree that links back to an
    ancestor recursed without end, and a self-cycle overflowed the stack ---
    a segfault, not an exception.  Trees are compared as graphs now."""

    def test_F13_S14_4_a_cycle_compares_once_at_the_default_stack_size(
        self, tmp_path: Path
    ):
        import subprocess
        import sys

        path = write_sample(tmp_path / "cyclic.medh5")
        with h5py.File(path, "r+") as handle:
            handle["x_loop"] = handle["/"]
            handle["images/x_self"] = handle["images"]
        code = (
            "import sys, medh5\n"
            "from medh5.integrity import subtrees_identical\n"
            "with medh5.open(sys.argv[1]) as s:\n"
            "    print(subtrees_identical(s.root, s.root))\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert run.returncode == 0, (run.returncode, run.stderr[-2000:])
        assert run.stdout.strip() == "()"

    def test_F13_an_alias_differs_from_a_duplicate(self, tmp_path: Path):
        from medh5.integrity import subtrees_identical

        alias = write_sample(tmp_path / "alias.medh5")
        copy = write_sample(tmp_path / "copy.medh5")
        with h5py.File(alias, "r+") as handle:
            handle["zzz_alias"] = handle["images/CT_tp0"]
        with h5py.File(copy, "r+") as handle:
            handle.copy("images/CT_tp0", "zzz_alias")
        with medh5.open(alias) as a, medh5.open(copy) as b:
            found = subtrees_identical(a.root, b.root)
        assert any(line.startswith("/zzz_alias") for line in found), found
