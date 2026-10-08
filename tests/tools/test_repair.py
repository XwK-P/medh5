"""``medh5 fix``: rebuilding what is derived --- stale indices and digests
(spec §13.3, §14.3).

A tool which changes a file must say what it changed, and must not claim more
than it did.  ``medh5 scrub``, which removes what should never have been
written, is held to the same principle in ``test_scrub.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.integrity import diagnose, fix
from tests.helpers import SHAPE, block, write_sample


@pytest.fixture
def indexed(tmp_path: Path, label_set, masks) -> Path:
    return write_sample(
        tmp_path / "case.medh5", label_set=label_set, masks=masks, index=True
    )


def _edit_mask_in_place(path: Path) -> None:
    """Change an annotation's bytes behind the writer's back.

    Exactly what an external tool does, and the reason a stale index and a
    mismatched digest are different problems from a malformed file.
    """
    import h5py

    with h5py.File(path, "a") as handle:
        node = handle["annotations"]["organs_tp0"]["data"]
        data = node[...]
        assert data.any(), "the fixture must have foreground to erase"
        data[data != 0] = 0
        node[...] = data


class TestFix:
    def test_a_healthy_file_needs_nothing(self, indexed):
        assert diagnose(indexed).clean
        assert not fix(indexed).changed

    def test_an_external_edit_shows_up_as_a_digest_mismatch(self, indexed):
        """The case `fix` exists for: something edited the data past the writer."""
        _edit_mask_in_place(indexed)
        diagnosis = diagnose(indexed)
        assert diagnosis.mismatched == ("annotations/organs_tp0/data",)
        assert diagnosis.needs_digests

    def test_the_content_id_alone_does_not_catch_an_edited_dataset(self, indexed):
        """`content_id` is a Merkle over stored digests, not over the bytes.

        Editing a dataset without restamping it changes the data but not the
        digest attribute the root hashes, so the root still matches while the
        object does not --- which is why `verify` checks both and why an
        integrity pass is per-object, not just a single top-level comparison.
        """
        _edit_mask_in_place(indexed)
        diagnosis = diagnose(indexed)
        assert diagnosis.content_id_ok is True
        assert diagnosis.mismatched

    def test_a_stale_index_is_found_and_rebuilt(self, indexed):
        """Restamping an external edit is what leaves the index behind."""
        _edit_mask_in_place(indexed)
        fix(indexed, rewrite_digests=True, reason="edited by an external tool")
        diagnosis = diagnose(indexed)
        assert diagnosis.stale_index == ("organs_tp0",)
        assert diagnosis.needs_index

        repair = fix(indexed, rebuild_index=True)
        assert repair.rebuilt_index == ("organs_tp0",)
        assert not diagnose(indexed).stale_index
        with medh5.open(indexed) as sample:
            counts = sample.index["organs_tp0"].voxel_counts
            direct = sample.annotations["organs_tp0"].voxel_counts()
            assert counts == direct

    def test_S13_3_rebuilding_an_index_will_not_launder_a_digest_mismatch(
        self, indexed
    ):
        """An amend restamps every digest, so the rebuild path needs the guard too.

        Recomputing a mismatched digest does not undo the edit that caused it,
        it destroys the evidence of it --- and on this path it did so with no
        reason, no provenance activity, and `rewrote_digests` reporting False.
        """
        _edit_mask_in_place(indexed)
        assert diagnose(indexed).mismatched

        with pytest.raises(MEDH5ValidationError, match="no longer match"):
            fix(indexed, rebuild_index=True)
        assert diagnose(indexed).mismatched, "the evidence survived the refusal"

        # The deliberate path still works, and still says what it did not verify.
        repair = fix(
            indexed,
            rebuild_index=True,
            rewrite_digests=True,
            reason="edited by an external tool, content confirmed",
        )
        assert repair.rewrote_digests
        assert not diagnose(indexed).mismatched
        assert any("asserts nothing" in note for note in repair.notes)

    def test_S14_3_rebuilding_does_not_index_what_was_never_indexed(
        self, tmp_path, label_set, masks
    ):
        """`fix` repairs; it does not decide a curator's storage budget for them.

        An empty rebuild list used to reach `build_index` as None, which means
        "every indexable annotation", so a file deliberately shipped without an
        index got one built for everything.
        """
        path = write_sample(
            tmp_path / "bare.medh5", label_set=label_set, masks=masks, index=False
        )
        diagnosis = diagnose(path)
        assert diagnosis.stale_index == ()
        # Reported, so a curator can see the option --- but not a defect, so it
        # does not make the file "need attention" and `fix` will not act on it.
        assert diagnosis.missing_index == ("organs_tp0",)
        assert not diagnosis.needs_index

        repair = fix(path, rebuild_index=True)
        assert repair.rebuilt_index == ()
        assert not repair.changed
        with medh5.open(path) as sample:
            assert sorted(sample.index) == []

    def test_S14_3_a_deliberately_partial_index_is_left_partial(
        self, tmp_path, label_set, masks
    ):
        """One annotation indexed and another not is a choice, not a defect.

        `present` being non-empty used to make every *other* indexable
        annotation count as missing, and `fix` then built its cache --- deciding
        the curator's storage budget for them one annotation at a time.
        """
        path = write_sample(
            tmp_path / "partial.medh5",
            label_set=label_set,
            masks=masks,
            timepoints=("tp0", "tp1"),
        )
        with medh5.amend(path) as writer:
            writer.build_index(["organs_tp0"])

        diagnosis = diagnose(path)
        assert diagnosis.stale_index == ()
        assert diagnosis.missing_index == ("organs_tp1",), "reported, not inferred away"
        assert not diagnosis.needs_index, "an absent index is not a defect"

        repair = fix(path, rebuild_index=True)
        assert repair.rebuilt_index == ()
        with medh5.open(path) as sample:
            assert sorted(sample.index) == ["organs_tp0"]

    def test_removing_an_annotation_takes_its_index_with_it(self, indexed):
        """Not stale --- gone.  A stale index is a mismatch, not an absence."""
        with medh5.amend(indexed) as writer:
            writer.remove_annotation("organs_tp0")
            writer.add_segmentation(
                "organs_tp0", grid="ct_tp0", masks={1: block(SHAPE, (2, 2, 2), 3)}
            )
        assert diagnose(indexed).stale_index == ()

    def test_diagnosing_changes_nothing(self, indexed):
        with medh5.open(indexed) as sample:
            before = sample.content_id
        diagnose(indexed)
        with medh5.open(indexed) as sample:
            assert sample.content_id == before

    def test_no_flags_means_report_only(self, indexed):
        _edit_mask_in_place(indexed)
        repair = fix(indexed)
        assert repair.diagnosis.needs_digests
        assert not repair.changed
        assert diagnose(indexed).needs_digests

    def test_rewriting_digests_without_a_reason_is_refused(self, indexed):
        """Restamping destroys evidence; it must be a decision, not a default."""
        with pytest.raises(MEDH5ValidationError, match="reason"):
            fix(indexed, rewrite_digests=True)

    def test_rewriting_digests_records_what_it_did_not_verify(self, indexed):
        repair = fix(
            indexed, rewrite_digests=True, reason="reconstructed by an external tool"
        )
        assert repair.rewrote_digests
        assert any("asserts nothing" in note for note in repair.notes)
        with medh5.open(indexed) as sample:
            activities = [a for a in sample.document.provenance.activities if a.tool]
            restamp = [a for a in activities if "rewrite-digests" in str(a.tool)]
            assert restamp, "the restamp must be in the file's own provenance"
            assert restamp[0].params["verified_content"] is False
            assert "external tool" in restamp[0].params["reason"]

    def test_a_restamped_file_verifies_again(self, indexed):
        fix(indexed, rewrite_digests=True, reason="test")
        with medh5.open(indexed) as sample:
            assert sample.verify().ok
