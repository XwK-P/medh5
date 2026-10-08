"""Auditing split claims across a cohort (spec §12.3, cohort check C202).

A leak is one subject --- or one grouping key --- in two partitions.  Making a
split is ``medh5.dataset``'s, tested in ``test_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import medh5
from medh5.curation.splits import audit_splits
from tests.helpers import write_sample
from tests.kits import Framed


class TestL33SplitLeaks:
    """L-33: a leak is found over subjects and grouping keys together."""

    def _cohort(self, root: Path) -> list[Path]:
        paths = []
        for i, (group, part) in enumerate((("fam-A", "train"), ("fam-B", "test"))):
            path = root / f"visit{i}.medh5"
            with Framed.writer(path) as w:
                w.identity(sample_id=f"s{i}", subject_id="patient-7")
                w.cohort(group_id=group)
                w.split(set_id="v1", partition=part)
            paths.append(path)
        return paths

    def test_L33_S12_3_one_subject_under_two_group_ids_is_a_leak(self, tmp_path: Path):
        """It reported `leaks = 0, ok = True` --- the case the splitter refuses."""
        from medh5.curation.splits import audit_splits

        audit = audit_splits(self._cohort(tmp_path))
        assert not audit.ok
        (leak,) = audit.leaks
        assert leak.groups == ("fam-A", "fam-B")
        assert leak.subjects == ("patient-7",)
        assert leak.partitions == ("test", "train")
        assert "fam-B" in str(leak)

    def test_L33_C202_finds_the_same_leak(self, tmp_path: Path):
        from medh5.dataset.check import check
        from medh5.dataset.manifest import scan

        root = tmp_path / "cohort"
        root.mkdir()
        self._cohort(root)
        manifest, _ = scan(root)
        finding = next(f for f in check(manifest).errors if f.code == "C202")
        assert "fam-A" in finding.where and "fam-B" in finding.where
        assert "patient-7" in finding.message

    def test_L33_units_join_through_subjects_transitively(self):
        from medh5.curation.splits import anatomy_units

        units = anatomy_units(
            [("s1", "A"), ("s1", "B"), ("s2", "B"), ("s2", "C"), ("s3", "D")]
        )
        assert units["A"] == units["C"] == ("A", "B", "C")
        assert units["D"] == ("D",)


class TestSplitAudit:
    def _write(self, path, label_set, masks, **claim):
        write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
        with medh5.amend(path) as w:
            w.identity(subject_id=claim.pop("subject_id", "subj-A"))
            w.split(**claim)
        return path

    def test_S12_3_a_consistent_cohort_is_clean(self, tmp_path, label_set, masks):
        paths = [
            self._write(
                tmp_path / f"c{i}.medh5",
                label_set,
                masks,
                subject_id=f"subj-{i}",
                set_id="cv5",
                partition="train" if i < 2 else "test",
                manifest_sha256="a" * 64,
            )
            for i in range(3)
        ]
        audit = audit_splits(paths)
        assert audit.ok
        assert audit.set_ids == ("cv5",)
        assert audit.counts() == {"cv5": {"test": 1, "train": 2}}
        assert audit.partitions("cv5")["test"] == ("c2",)
        assert audit.to_json()["ok"] is True

    def test_W906_conflicting_manifests_across_files(self, tmp_path, label_set, masks):
        paths = [
            self._write(
                tmp_path / f"c{i}.medh5",
                label_set,
                masks,
                subject_id=f"subj-{i}",
                set_id="cv5",
                partition="train",
                manifest_sha256=("a" if i == 0 else "b") * 64,
            )
            for i in range(2)
        ]
        audit = audit_splits(paths)
        assert not audit.ok
        assert len(audit.conflicts) == 1
        assert "2 different manifest hashes" in str(audit.conflicts[0])
        assert len(audit.conflicts[0].paths_by_manifest) == 2

    def test_S12_2_subject_leakage_is_its_own_finding(self, tmp_path, label_set, masks):
        """One subject in train and test --- invisible in either file alone."""
        paths = [
            self._write(
                tmp_path / f"visit{i}.medh5",
                label_set,
                masks,
                subject_id="subj-shared",
                set_id="cv5",
                partition="train" if i == 0 else "test",
                manifest_sha256="a" * 64,
            )
            for i in range(2)
        ]
        audit = audit_splits(paths)
        assert not audit.ok
        assert not audit.conflicts
        assert len(audit.leaks) == 1
        leak = audit.leaks[0]
        assert leak.group_id == "subj-shared"
        assert leak.partitions == ("test", "train")
        assert "is in test, train" in str(leak)

    def test_files_without_claims_are_listed_not_failed(
        self, tmp_path, label_set, masks
    ):
        path = write_sample(tmp_path / "bare.medh5", label_set=label_set, masks=masks)
        audit = audit_splits([path])
        assert audit.ok
        assert audit.unclaimed == (str(path),)

    def test_an_unreadable_file_does_not_stop_the_audit(
        self, tmp_path, label_set, masks
    ):
        good = self._write(
            tmp_path / "good.medh5",
            label_set,
            masks,
            set_id="cv5",
            partition="train",
        )
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        audit = audit_splits([good, bad])
        assert not audit.ok
        assert len(audit.unreadable) == 1
        assert audit.set_ids == ("cv5",)

    def test_a_collection_contributes_every_member(self, tmp_path, label_set, masks):
        from medh5.collection import pack

        paths = [
            self._write(
                tmp_path / f"m{i}.medh5",
                label_set,
                masks,
                subject_id="subj-shared",
                set_id="cv5",
                partition="train" if i == 0 else "val",
            )
            for i in range(2)
        ]
        shard = pack(paths, tmp_path / "shard.medh5c")
        audit = audit_splits([shard])
        assert len(audit.memberships) == 2
        assert audit.leaks and audit.leaks[0].group_id == "subj-shared"
        assert all("::" in m.path for m in audit.memberships)

    def test_cohort_group_id_overrides_the_subject(self, tmp_path, label_set, masks):
        paths = []
        for i in range(2):
            path = tmp_path / f"g{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id=f"subj-{i}")
                w.cohort(group_id="family-3")
                w.split(set_id="cv5", partition="train" if i == 0 else "test")
            paths.append(path)
        audit = audit_splits(paths)
        assert audit.leaks[0].group_id == "family-3"
