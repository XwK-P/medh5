"""Digests, content ids and index currency (spec §13)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.integrity import (
    array_digest,
    canonical_attrs,
    dataset_digest,
    diagnose,
    digest_bytes,
    fix,
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


def at_location(report, location: str) -> list[str]:
    """The codes reported at *location*."""
    return [d.code for d in report.diagnostics if d.location == location]


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

    def test_a_digest_lets_other_threads_run(self):
        """A digest is a pass over a whole payload, run with the GIL released
        as every long engine call is: held, another Python thread stood still
        for the whole of it.  The other thread stamps the time as it runs; a
        gap as long as the call is the GIL held."""
        payload = bytes(4 << 20)
        start = time.perf_counter()
        digest_bytes(payload, "sha512")
        rate = len(payload) / max(time.perf_counter() - start, 1e-6)
        # About 0.4 s of hashing, within 256 MiB (of untouched zero pages).
        payload = bytes(min(int(rate * 0.4), 256 << 20))
        stamps: list[float] = []
        done = threading.Event()

        def tick() -> None:
            while not done.is_set():
                stamps.append(time.perf_counter())
                time.sleep(0.0005)

        ticker = threading.Thread(target=tick)
        ticker.start()
        try:
            time.sleep(0.01)
            t0 = time.perf_counter()
            digest_bytes(payload, "sha512")
            t1 = time.perf_counter()
        finally:
            done.set()
            ticker.join()
        inside = [t for t in stamps if t0 < t < t1]
        longest = float(np.diff([t0, *inside, t1]).max())
        assert longest < (t1 - t0) / 2, f"no tick for {longest:.3f} of {t1 - t0:.3f} s"

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

    def test_S13_2_meta_is_hashed_as_stored(self, tmp_path):
        """Bytes that are not UTF-8 were hashed as the text they decode to with
        replacement, so two different documents --- and one holding U+FFFD
        itself --- shared one `content_id`."""
        base = write_sample(tmp_path / "base.medh5")
        computed = set()
        for i, bad in enumerate((b"\xff", b"\xfe", "\ufffd".encode())):
            path = tmp_path / f"m{i}.medh5"
            path.write_bytes(base.read_bytes())
            with h5py.File(path, "r+") as handle:
                raw = handle["meta"][()]
                del handle["meta"]
                handle.create_dataset(
                    "meta",
                    data=np.array(raw.replace(b"subj-A", b"subj-" + bad), dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
            with medh5.open(path) as sample:
                computed.add(sample.verify().content_id_computed)
        assert len(computed) == 3

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


def _digested(handle: h5py.File) -> list[str]:
    """Every dataset that carries a `digest`."""
    found: list[str] = []

    def visit(name: str, node: object) -> None:
        if isinstance(node, h5py.Dataset) and "digest" in node.attrs:
            found.append(name)

    handle.visititems(visit)
    return found


def _plant(handle: h5py.File, case: str) -> None:
    unaddressed = case.endswith(", no content_id")
    case = case.removesuffix(", no content_id")
    if case == "a digest removed":
        del handle["images/CT_tp0"].attrs["digest"]
    elif case == "a digest that does not parse":
        handle["images/CT_tp0"].attrs["digest"] = encode_attr("garbage")
    elif case.startswith("no digest parses"):
        if case.endswith("data changed"):
            image = handle["images/CT_tp0"]
            image[0, 0, 0] = image[0, 0, 0] + 1
        for name in _digested(handle):
            handle[name].attrs["digest"] = encode_attr("md5:" + "0" * 32)
    elif case == "data changed":
        image = handle["images/CT_tp0"]
        image[0, 0, 0] = image[0, 0, 0] + 1
    elif case == "a wrong content_id":
        handle.attrs["content_id"] = encode_attr("sha256:" + "1" * 64)
    elif case == "every digest removed":
        for name in _digested(handle):
            del handle[name].attrs["digest"]
    if unaddressed:
        del handle.attrs["content_id"]


class TestValidationAgreesWithVerify:
    """§13, §15.1: an integrity or strict pass fails exactly the files `verify`
    fails, and `diagnose` sees what both see.  A file in which no digest parsed
    passed both levels --- its E703, its E702 and its stale index dropped ---
    while `verify` failed it, and `fix` called it clean."""

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            ("untouched", set()),
            ("a digest removed", {"E702"}),
            ("a digest removed, no content_id", set()),
            ("a digest that does not parse", {"E702", "E703"}),
            ("a digest that does not parse, no content_id", {"E703"}),
            ("no digest parses", {"E702", "E703"}),
            ("no digest parses, no content_id", {"E703"}),
            ("no digest parses, data changed", {"E702", "E703"}),
            ("data changed", {"E701"}),
            ("a wrong content_id", {"E702"}),
            ("every digest removed", {"E702"}),
        ],
    )
    def test_S13_1_validation_fails_what_verify_fails(
        self, tmp_path, label_set, masks, case, expected
    ):
        from medh5.integrity import diagnose

        path = write_sample(
            tmp_path / "case.medh5", label_set=label_set, masks=masks, index=True
        )
        with h5py.File(path, "r+") as handle:
            _plant(handle, case)
        with medh5.open(path) as sample:
            verified = sample.verify()
        integrity = validate_file(path, level="integrity")
        strict = validate_file(path, level="strict")

        def attested(report) -> set[str]:
            return {d.code for d in report.diagnostics if d.code.startswith("E7")}

        assert attested(integrity) == attested(strict) == expected
        assert integrity.ok is verified.ok
        assert strict.ok is False or verified.ok
        assert diagnose(path).needs_digests is not verified.ok

    def test_S13_3_a_stale_index_is_reported_when_no_digest_parses(
        self, tmp_path, label_set, masks
    ):
        path = write_sample(
            tmp_path / "case.medh5", label_set=label_set, masks=masks, index=True
        )
        with h5py.File(path, "r+") as handle:
            _plant(handle, "no digest parses, no content_id")
        assert "W905" in validate_file(path, level="integrity").codes

    def test_S15_1_the_command_line_fails_a_file_no_digest_of_which_parses(
        self, tmp_path, label_set, masks, capsys
    ):
        from medh5.cli import main

        path = write_sample(tmp_path / "case.medh5", label_set=label_set, masks=masks)
        with h5py.File(path, "r+") as handle:
            _plant(handle, "no digest parses, data changed")
        for level in ("integrity", "strict"):
            assert main(["validate", str(path), "--level", level]) == 1, level
        assert main(["verify", str(path)]) == 1
        capsys.readouterr()


class TestUnreadableData:
    """§13.1: one chunk that no longer decodes stopped the integrity pass, so
    every other E701 and E702 went unreported, and so did its dataset."""

    def test_S13_1_an_unreadable_dataset_is_named_and_the_others_checked(
        self, tmp_path
    ):
        path = tmp_path / "two.medh5"
        shape = (8, 64, 64)  # large enough to be chunked and compressed
        rng = np.random.default_rng(0)
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            for name in ("CT", "MR"):
                data = rng.integers(0, 100, shape).astype(np.int16)
                w.add_image(name, data, grid="g", modality=name)
        with h5py.File(path, "r+") as handle:
            info = handle["images/CT"].id.get_chunk_info(0)
            image = handle["images/MR"]
            image[0, 0, 0] = image[0, 0, 0] + 1
        with path.open("r+b") as raw:
            raw.seek(info.byte_offset + info.size // 2)
            raw.write(b"\xde\xad\xbe\xef" * 8)
        report = validate_file(path, level="integrity")
        assert at_location(report, "/images/CT") == ["E001"]
        assert at_location(report, "/images/MR") == ["E701"]
        with medh5.open(path) as sample, pytest.raises(OSError, match="images/CT"):
            sample.verify()


class TestIntegrity:
    """A check that checks, and an amend that stays addressed."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    def test_S13_1_recompress_verifies_what_it_wrote(self, tmp_path: Path):
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

    def test_the_cli_exit_code_follows_the_verification(self, tmp_path: Path, capsys):
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

    def test_S13_2_an_amend_without_digests_stays_addressed(self, tmp_path: Path):
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

    def test_S13_2_a_fresh_file_is_fully_stamped_either_way(self, tmp_path: Path):
        path = tmp_path / "fresh.medh5"
        writer = Organs.writer(path)
        writer.add_segmentation("seg", grid="g", masks=Organs.blocks())
        writer.commit(digests=False)
        with medh5.open(path) as sample:
            assert not sample.verify().undigested
            assert sample.content_id is not None


class TestAmendStamping:
    """An amendment stamps what it wrote and never restamps what it inherited."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    def test_S13_1_an_amendment_refuses_to_restamp_a_digest_its_source_failed(
        self, tmp_path: Path
    ):
        """Every amend recomputed every digest: a no-op amend of a file edited
        behind the library's back made it verify, and left no record."""
        from medh5.integrity import diagnose, fix

        path = self._sample(tmp_path / "edited.medh5")
        with h5py.File(path, "r+") as handle:
            node = handle["images/CT"]
            data = node[...]
            data[0, 0, 0] += 7
            node[...] = data
        before = path.read_bytes()
        with (
            pytest.raises(MEDH5ValidationError, match="fix --rewrite-digests") as err,
            medh5.amend(path),
        ):
            pass
        assert err.value.code == "E701"
        assert "images/CT" in str(err.value)
        assert path.read_bytes() == before, "nothing was written"
        assert diagnose(path).mismatched == ("images/CT",)

        # Stamping only what carries no digest leaves the evidence in place.
        with medh5.amend(path) as w:
            w.set_quality("seg", status="approved")
            w.commit(digests=False)
        assert diagnose(path).mismatched == ("images/CT",)

        # The deliberate restamp still works, and says it verified nothing.
        repair = fix(path, rewrite_digests=True, reason="the edit was intended")
        assert repair.rewrote_digests
        assert diagnose(path).clean
        with medh5.amend(path) as w:
            w.set_quality("seg", status="draft")
        with medh5.open(path) as sample:
            assert sample.verify().ok


class TestAttestationCarriedForward:
    """§14.4, §2.1: an amendment carries attestation forward, never past a
    break.  A commit recomputes the `content_id` from whatever is stored, so a
    no-op amendment of a file edited outside medh5 made it validate, with no
    record of the edit; and an algorithm it could not compute gave way to
    `sha256`."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    @staticmethod
    def _stale(path: Path) -> None:
        """An edit of a covered attribute, made outside medh5 (E702 at /)."""
        with h5py.File(path, "r+") as handle:
            handle["grids/g"].attrs["spacing"] = np.array([2.0, 2.0, 2.0])
        assert at_location(validate_file(path, level="integrity"), "/") == ["E702"]

    def test_S14_4_an_amendment_does_not_reattest_a_stale_root(self, tmp_path: Path):
        from medh5.curation import scrub
        from medh5.dataset.manifest import scan
        from medh5.dataset.split import make_splits, write_claims

        cohort = tmp_path / "cohort"
        cohort.mkdir()
        path = self._sample(cohort / "case.medh5")
        self._stale(path)
        before = path.read_bytes()

        def amended() -> None:
            with medh5.amend(path) as w:
                w.set_quality("seg", status="approved")

        def scrubbed() -> None:
            scrub.apply(path)

        def claimed() -> None:
            manifest, failures = scan(cohort)
            assert not failures
            write_claims(make_splits(manifest), manifest)

        for amend in (amended, scrubbed, claimed):
            with pytest.raises(MEDH5ValidationError) as err:
                amend()
            assert err.value.code == "E702", amend.__name__
            assert "medh5 fix --rewrite-digests --reason" in str(err.value)
            assert path.read_bytes() == before, amend.__name__
            assert sorted(p.name for p in cohort.iterdir()) == ["case.medh5"]

    def test_S14_4_fix_reattests_a_stale_root_deliberately(self, tmp_path: Path):
        from medh5.integrity import diagnose, fix

        path = self._sample(tmp_path / "case.medh5")
        self._stale(path)
        assert diagnose(path).content_id_ok is False
        repair = fix(path, rewrite_digests=True, reason="spacing corrected by hand")
        assert repair.rewrote_digests
        assert validate_file(path, level="integrity").ok
        with medh5.open(path) as sample:
            (activity,) = sample.document.provenance.activities
            assert activity.params["reason"] == "spacing corrected by hand"
            assert activity.params["verified_content"] is False
            assert "digest_algo" not in activity.params
        # Re-attested, it amends as any file does.
        with medh5.amend(path) as w:
            w.set_quality("seg", status="approved")
        assert validate_file(path, level="integrity").ok

    @staticmethod
    def _unknown_algo(path: Path) -> None:
        with h5py.File(path, "r+") as handle:
            handle.attrs["digest_algo"] = "blake3"

    def test_S2_1_an_amendment_refuses_an_algorithm_it_cannot_compute(
        self, tmp_path: Path
    ):
        path = self._sample(tmp_path / "case.medh5")
        self._unknown_algo(path)
        before = path.read_bytes()
        for digests in (True, False):
            with pytest.raises(MEDH5ValidationError) as err:
                w = medh5.amend(path)
                w.set_quality("seg", status="approved")
                w.commit(digests=digests)
            assert err.value.code == "E703"
            assert "'blake3'" in str(err.value)
            assert "medh5 fix --rewrite-digests --reason" in str(err.value)
            assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["case.medh5"]

    def test_S2_1_fix_restamps_an_unknown_algorithm_and_records_it(
        self, tmp_path: Path
    ):
        from medh5.integrity import diagnose, fix

        path = self._sample(tmp_path / "case.medh5")
        self._unknown_algo(path)

        def verified() -> None:
            with medh5.open(path) as sample:
                sample.verify()

        for refused in (lambda: diagnose(path), lambda: fix(path), verified):
            with pytest.raises(MEDH5ValidationError, match="--rewrite-digests") as err:
                refused()
            assert err.value.code == "E703"
        repair = fix(path, rewrite_digests=True, reason="imported from a blake3 tool")
        assert repair.rewrote_digests
        assert any("'blake3'" in note and "sha256" in note for note in repair.notes)
        assert validate_file(path, level="integrity").ok
        with h5py.File(path, "r") as handle:
            assert handle.attrs["digest_algo"] == "sha256"
        with medh5.open(path) as sample:
            (activity,) = sample.document.provenance.activities
            assert activity.params["digest_algo"] == {"from": "blake3", "to": "sha256"}
            assert activity.params["verified_content"] is False

    def test_S13_1_an_unreadable_digest_refuses_an_amendment_as_E703(
        self, tmp_path: Path
    ):
        path = self._sample(tmp_path / "case.medh5")
        with h5py.File(path, "r+") as handle:
            handle["images/CT"].attrs["digest"] = encode_attr("sha256:not-hex")
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError) as err, medh5.amend(path):
            pass
        assert err.value.code == "E703"
        assert "/images/CT" in str(err.value)
        assert path.read_bytes() == before


class TestDigestedTypes:
    """What each stored type hashes as (§13.1), as 1.x hashed it through h5py."""

    def _sample(self, path: Path) -> Path:
        with Organs.writer(path) as w:
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        return path

    @staticmethod
    def _digest(path: str, dtype: str, shape: str, payload: bytes) -> str:
        import hashlib

        head = f"{path}\0{dtype}\0{shape}\0".encode()
        return "sha256:" + hashlib.sha256(head + payload).hexdigest()

    def test_S13_1_a_compound_is_digested_and_a_sequence_is_left_undigested(
        self, tmp_path: Path
    ):
        """A compound or a variable-length sequence in an extension group is
        valid, and every amend of such a file failed (TypeError); a 1.x digest
        on a compound failed `verify`, integrity validation, and `recompress`
        after it had replaced the file."""
        from medh5.storage import recompress

        path = self._sample(tmp_path / "ext.medh5")
        table = np.array([(1, 0.5), (2, -1.0)], dtype=[("id", "<i4"), ("v", "<f8")])
        with h5py.File(path, "r+") as handle:
            group = handle.create_group("x_acme")
            group["table"] = table
            seq = group.create_dataset(
                "seq", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            seq[0] = np.array([1, 2, 3], dtype=np.uint8)
            seq[1] = np.array([4], dtype=np.uint8)
        with medh5.amend(path):
            pass
        with medh5.open(path) as sample:
            result = sample.verify()
            assert result.ok, result.summary()
            assert "x_acme/seq" in result.undigested
            stored = sample.root["x_acme/table"].attrs["digest"]
        assert stored == self._digest("x_acme/table", "|V12", "2", table.tobytes())

        # A digest 1.x stamped on a sequence hashed pointers: it cannot be
        # checked, and is not a mismatch.  1.x covered it in the root too.
        with h5py.File(path, "r+") as handle:
            handle["x_acme/seq"].attrs["digest"] = "sha256:" + "0" * 64
        with medh5.open(path) as sample:
            root = sample.compute_content_id()
        with h5py.File(path, "r+") as handle:
            handle.attrs["content_id"] = root
        with medh5.open(path) as sample:
            result = sample.verify()
            assert result.unverifiable == ("x_acme/seq",)
            assert result.ok and not result.mismatched
        codes = validate_file(path, level="integrity").codes
        assert "E001" not in codes and "E701" not in codes, codes
        assert recompress(path, "archive").ok

    def test_S13_1_a_string_hashes_its_stored_bytes(self, tmp_path: Path):
        """Strings were hashed as text decoded with replacement, so different
        invalid bytes shared a digest and an edit to them passed `verify`."""
        path = self._sample(tmp_path / "strings.medh5")
        with h5py.File(path, "r+") as handle:
            group = handle.create_group("x_site")
            fixed = np.array([b"caf\xe9", b"ok"], dtype="S4")
            group.create_dataset("fixed", data=fixed)
            group.create_dataset(
                "vlen", data=[b"caf\xe9", b"ok"], dtype=h5py.string_dtype("ascii")
            )
        with medh5.amend(path):
            pass
        with medh5.open(path) as sample:
            assert sample.verify().ok
            for name in ("fixed", "vlen"):
                stored = sample.root[f"x_site/{name}"].attrs["digest"]
                expected = self._digest(f"x_site/{name}", "|O", "2", b"caf\xe9\0ok\0")
                assert stored == expected, name
        with h5py.File(path, "r+") as handle:
            handle["x_site/fixed"][0] = b"caf\xe8"
            handle["x_site/vlen"][0] = b"caf\xfe"
        with medh5.open(path) as sample:
            assert sample.verify().mismatched == ("x_site/fixed", "x_site/vlen")


class TestVerify:
    """An undigested dataset inside an attested object fails `verify`."""

    def _boxes(self, path: Path) -> Path:
        with Framed.writer(path) as w:
            w.add_boxes(
                "det", np.array([Framed.box(1, 3)], np.float32), ["c1"], grid="g"
            )
        return path

    def test_S13_2_an_undigested_dataset_in_an_annotation_fails_verify(
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

    def test_S13_2_recompress_names_the_unattested_dataset(
        self, tmp_path: Path, capsys
    ):
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

    def test_S13_2_fix_counts_it_as_needing_digests(self, tmp_path: Path):
        from medh5.integrity import diagnose

        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )
        diagnosis = diagnose(path)
        assert diagnosis.needs_digests
        assert diagnosis.unattested == ("annotations/det/instance_ids",)

    def test_S13_2_the_writers_files_have_none_and_an_unaddressed_file_is_not_judged(
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


class TestUndigestedDatasets:
    """D6, §13.1: a dataset without a `digest`, inside an object a declared
    `content_id` covers, is W915 --- `--level strict` passed the file `verify`
    failed as unattested."""

    def _boxes(self, path: Path) -> Path:
        with Framed.writer(path) as w:
            w.add_boxes(
                "det", np.array([Framed.box(1, 3)], np.float32), ["c1"], grid="g"
            )
        return path

    def _plant(self, path: Path) -> None:
        with h5py.File(path, "r+") as handle:
            handle["annotations/det"].create_dataset(
                "instance_ids", data=np.array([7], np.uint32)
            )

    def test_S13_1_an_undigested_dataset_under_a_content_id_is_W915(
        self, tmp_path: Path
    ):
        path = self._boxes(tmp_path / "det.medh5")
        self._plant(path)
        integrity = validate_file(path, level="integrity")
        assert [
            (d.code, d.location) for d in integrity.diagnostics if d.code == "W915"
        ] == [("W915", "/annotations/det/instance_ids")]
        assert "W915" not in validate_file(path, level="semantic").codes
        assert "W915" in validate_file(path, level="strict").codes
        # Pinned to revision 1, 2.0's verdict.
        assert "W915" not in validate_file(path, level="strict", codes_revision=1).codes
        with medh5.open(path) as sample:
            assert not sample.verify().ok

    def test_S13_1_W915_is_not_reported_without_a_content_id(self, tmp_path: Path):
        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            del handle.attrs["content_id"]
        self._plant(path)
        assert "W915" not in validate_file(path, level="integrity").codes

    def test_S13_1_a_file_with_no_digests_is_W901_not_W915(self, tmp_path: Path):
        path = self._boxes(tmp_path / "det.medh5")
        with h5py.File(path, "r+") as handle:
            for name in list(_datasets(handle)):
                handle[name].attrs.pop("digest", None)
        codes = validate_file(path, level="integrity").codes
        assert "W901" in codes and "W915" not in codes

    def test_S13_1_an_undigested_clinical_dataset_is_E818_not_W915(
        self, tmp_path: Path
    ):
        from tests.kits import History

        path = tmp_path / "history.medh5"
        History.write(path)
        with h5py.File(path, "r+") as handle:
            events = handle["clinical/events"]
            events.create_dataset(
                "x_note", data=np.zeros(events["available_lo_us"].shape, "u1")
            )
        codes = validate_file(path, level="integrity").codes
        assert "E818" in codes and "W915" not in codes


def _datasets(handle: h5py.File) -> list[str]:
    found: list[str] = []
    handle.visititems(
        lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) else None
    )
    return found


_SPELLINGS = {
    "sign": lambda hex_: f"+{hex_}",
    "0x": lambda hex_: f"0x{hex_}",
    "underscore": lambda hex_: f"{hex_[:32]}_{hex_[32:]}",
    "whitespace": lambda hex_: f" {hex_}",
    "uppercase": str.upper,
    "short": lambda hex_: hex_[:-2],
}


class TestCanonicalDigests:
    """D4, §13.1: `<hex>` is lowercase hex of the algorithm's length.  Read as
    Python's `int(hex, 16)` reads a number, an uppercase digest of unchanged
    bytes was a mismatch (E701) and its root tampered with (E702)."""

    def _respell(self, path: Path, spell: Any, at: str = "images/CT_tp0") -> None:
        with h5py.File(path, "r+") as handle:
            algo, hex_ = str(handle[at].attrs["digest"]).split(":")
            handle[at].attrs["digest"] = f"{algo}:{spell(hex_)}"

    @pytest.mark.parametrize("spell", _SPELLINGS.values(), ids=list(_SPELLINGS))
    def test_S13_1_a_digest_not_in_canonical_hex_is_E703(
        self, tmp_path: Path, spell: Any
    ):
        path = write_sample(tmp_path / "s.medh5")
        self._respell(path, spell)
        report = validate_file(path, level="integrity")
        assert ("E703", "/images/CT_tp0") in {
            (d.code, d.location) for d in report.diagnostics
        }
        assert "E701" not in report.codes
        with medh5.open(path) as sample:
            result = sample.verify()
        assert "images/CT_tp0" in result.malformed and not result.mismatched

    def test_S13_1_fix_restamps_a_non_canonical_digest(self, tmp_path: Path):
        path = write_sample(tmp_path / "s.medh5")
        self._respell(path, str.upper)
        assert "images/CT_tp0" in diagnose(path).malformed
        fix(
            path,
            rewrite_digests=True,
            reason="digests from a tool spelling hex in capitals",
        )
        with medh5.open(path) as sample:
            assert sample.verify().ok
        assert validate_file(path, level="integrity").ok

    def test_S13_1_a_content_id_not_in_canonical_hex_is_E703_not_E702(
        self, tmp_path: Path
    ):
        path = write_sample(tmp_path / "s.medh5")
        with h5py.File(path, "r+") as handle:
            handle.attrs["content_id"] = str(handle.attrs["content_id"]).upper()
        report = validate_file(path, level="integrity")
        root = {d.code for d in report.diagnostics if d.location == "/"}
        assert root == {"E703"}
        with medh5.open(path) as sample:
            assert not sample.verify().ok
        with (
            pytest.raises(MEDH5ValidationError, match="not a digest"),
            medh5.amend(path),
        ):
            pass


class TestCyclicCompare:
    """Comparing a tree that links back to an
    ancestor recursed without end, and a self-cycle overflowed the stack ---
    a segfault, not an exception.  Trees are compared as graphs now."""

    def test_S14_4_a_cycle_compares_once_at_the_default_stack_size(
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

    def test_S13_2_an_alias_differs_from_a_duplicate(self, tmp_path: Path):
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


class TestBoundPaths:
    """A line binds bytes to the path it names
    (§13.2, E704).  Coverage was judged by object, so a file crafted with an
    alias sorting before `images/` --- holding the image's line, digested and
    sealed there --- passed every check, and relinking `images/CT_tp0` to
    another covered image after a pin changed no line: the root recomputed,
    `verify`, the pin and the validator all held, and the reader read the
    other image."""

    @staticmethod
    def _crafted(path: Path) -> str:
        """The crafted file: an alias sorting before `images/` holding the
        image's line, and a decoy --- the image's attributes over other voxels
        --- with a line of its own, both digested and the root sealed."""
        write_sample(path)
        with h5py.File(path, "r+") as handle:
            handle["aaa_ct"] = handle["images/CT_tp0"]
            handle.copy("images/CT_tp0", "aaa_decoy")
            handle["aaa_decoy"][0, 0, 0] = handle["aaa_decoy"][0, 0, 0] + 1
        with medh5.open(path) as s:
            digests = {
                name: dataset_digest(s.root[name], name)
                for name in ("aaa_ct", "aaa_decoy")
            }
        with h5py.File(path, "r+") as handle:
            for name, digest in digests.items():
                handle[name].attrs["digest"] = digest
        with medh5.open(path) as s:
            content_id = s.compute_content_id()
        with h5py.File(path, "r+") as handle:
            handle.attrs["content_id"] = content_id
        return content_id

    def test_S13_2_a_name_an_earlier_alias_holds_is_unbound(self, tmp_path: Path):
        from medh5.errors import MEDH5IntegrityError
        from medh5.task import SourceRef

        path = tmp_path / "crafted.medh5"
        self._crafted(path)
        with medh5.open(path) as s:
            result = s.verify()
            assert result.content_id_ok is True and result.mismatched == ()
            assert not result.ok
            assert result.unattested == ("images/CT_tp0",)
        report = validate_file(path, level="integrity")
        errors = [d for d in report.diagnostics if d.severity == "error"]
        assert [(d.code, d.location) for d in errors] == [("E704", "/images/CT_tp0")]
        assert "'/aaa_ct'" in errors[0].message
        with pytest.raises(MEDH5IntegrityError, match="cannot be pinned"):
            SourceRef.pin(path)

    @pytest.mark.parametrize("link", ["hard", "soft"])
    def test_S13_2_a_relink_under_an_unchanged_root_is_refused(
        self, tmp_path: Path, link: str
    ):
        from medh5.task import SourceRef

        path = tmp_path / "crafted.medh5"
        sealed = self._crafted(path)
        earlier = SourceRef(path.name, sealed)  # as an earlier release pinned it
        with h5py.File(path, "r+") as handle:
            del handle["images/CT_tp0"]
            if link == "hard":
                handle["images/CT_tp0"] = handle["aaa_decoy"]
            else:
                handle["images/CT_tp0"] = h5py.SoftLink("/aaa_decoy")
        with medh5.open(path) as s:
            # The premise: no line changed, so the root still recomputes.
            assert s.compute_content_id() == sealed
            assert s.verify().unattested == ("images/CT_tp0",)
            assert not s.verify().ok
        for deep in (False, True):
            (finding,) = earlier.check(tmp_path, deep=deep)
            assert finding.code == "T302" and "images/CT_tp0" in str(finding)
        diagnostics = validate_file(path, level="integrity").diagnostics
        assert ("E704", "/images/CT_tp0") in {(d.code, d.location) for d in diagnostics}

    def test_S13_2_a_later_alias_changes_nothing(self, tmp_path: Path):
        """The control: a second path after every attested group is an
        extension's, and the image keeps its own path and its line."""
        from medh5.task import SourceRef

        path = write_sample(tmp_path / "later.medh5", timepoints=("tp0", "tp1"))
        with medh5.open(path) as s:
            before = s.content_id
        with h5py.File(path, "r+") as handle:
            handle["zzz_alias"] = handle["images/CT_tp0"]
            handle["x_ext/soft"] = h5py.SoftLink("/images/CT_tp1")
        with medh5.open(path) as s:
            assert s.content_id == before == s.compute_content_id()
            assert s.verify().ok and s.verify().unattested == ()
        assert validate_file(path, level="integrity").ok
        pin = SourceRef.pin(path, uri=path.name)
        assert pin.check(tmp_path) == [] and pin.check(tmp_path, deep=True) == []

    def test_S13_2_a_second_path_inside_an_attested_group_is_unbound(
        self, tmp_path: Path
    ):
        """Inside the groups every path is judged: a second link, a soft link,
        a group reached again --- each one finding, and none walked twice."""
        path = write_sample(tmp_path / "inside.medh5")
        with h5py.File(path, "r+") as handle:
            handle["images/x_second"] = handle["images/CT_tp0"]
            handle["grids/x_soft"] = h5py.SoftLink("/grids/ct_tp0")
            handle["annotations/x_self"] = handle["annotations"]
        with medh5.open(path) as s:
            assert s.verify().unattested == (
                "annotations/x_self",
                "grids/x_soft",
                "images/x_second",
            )
        codes = {
            (d.code, d.location)
            for d in validate_file(path, level="integrity").diagnostics
            if d.code == "E704"
        }
        assert codes == {
            ("E704", "/annotations/x_self"),
            ("E704", "/grids/x_soft"),
            ("E704", "/images/x_second"),
        }
