"""The conformance corpus is the contract (spec §15, plan phase 0).

Every case is a file plus the exact diagnostic codes a conforming validator must
emit for it.  This module runs the corpus against *this* validator; a
third-party implementation runs the same manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import medh5
from medh5.conformance import CASES, build_corpus, case_by_name, run_corpus
from medh5.errors import CODES
from medh5.validate import validate_file

# Codes owned by §10 (transforms), which phase 4 implements.  Listed explicitly
# so the corpus's coverage gap is a decision on the record rather than an
# omission nobody noticed.
DEFERRED_CODES = frozenset({"E501", "E502", "E503", "E504", "E505"})


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> list:
    return run_corpus(tmp_path_factory.mktemp("corpus"))


class TestCorpus:
    def test_every_case_reports_exactly_its_expected_codes(self, corpus):
        failures = [
            f"{r.case.name}: missing={r.missing} unexpected={r.unexpected} "
            f"{r.error or ''}"
            for r in corpus
            if not r.ok
        ]
        assert not failures, "\n".join(failures)

    def test_the_corpus_has_both_valid_and_invalid_cases(self, corpus):
        assert sum(1 for r in corpus if r.case.valid) >= 10
        assert sum(1 for r in corpus if not r.case.valid) >= 30

    def test_S15_2_every_implemented_code_has_a_case(self):
        covered = {c for case in CASES for c in (*case.errors, *case.warnings)}
        uncovered = set(CODES) - covered - DEFERRED_CODES
        assert not uncovered, (
            f"diagnostic codes with no corpus case: {sorted(uncovered)}"
        )

    def test_deferred_codes_are_genuinely_unimplemented(self):
        """The exemption list must not quietly grow to cover real gaps."""
        covered = {c for case in CASES for c in (*case.errors, *case.warnings)}
        assert not (DEFERRED_CODES & covered), (
            "a deferred code now has a case; remove it from DEFERRED_CODES"
        )

    def test_case_names_are_unique(self):
        names = [c.name for c in CASES]
        assert len(names) == len(set(names))

    def test_every_case_cites_a_spec_clause(self):
        for case in CASES:
            assert case.clause.startswith("§"), case.name

    def test_case_by_name(self):
        assert case_by_name("core-minimal").valid
        with pytest.raises(KeyError):
            case_by_name("nope")


class TestManifest:
    def test_manifest_describes_every_written_file(self, tmp_path):
        manifest_path = build_corpus(tmp_path, names=["core-minimal", "seg-layers"])
        manifest = json.loads(Path(manifest_path).read_text())
        assert manifest["format"] == medh5.FORMAT_VERSION
        assert len(manifest["cases"]) == 2
        for record in manifest["cases"]:
            assert (tmp_path / record["file"]).exists()
            assert set(record) >= {
                "name",
                "description",
                "clause",
                "level",
                "valid",
                "expect_errors",
                "expect_warnings",
                "file",
            }

    def test_valid_cases_open_with_the_public_reader(self, tmp_path):
        names = [c.name for c in CASES if c.valid]
        build_corpus(tmp_path, names=names)
        for name in names:
            with medh5.open(tmp_path / f"{name}.medh5") as sample:
                assert sample.version == medh5.FORMAT_VERSION
                assert len(sample.images) >= 1

    def test_unmutated_valid_cases_verify(self, tmp_path):
        """A file the writer produced end to end must also verify (spec §13)."""
        names = [c.name for c in CASES if c.valid and not c.mutated]
        assert len(names) >= 10
        build_corpus(tmp_path, names=names)
        for name in names:
            with medh5.open(tmp_path / f"{name}.medh5") as sample:
                assert sample.verify().ok, name

    def test_mutated_cases_declare_it(self):
        """`valid` must never be read as "this also verifies"."""
        from medh5.conformance import case_by_name

        assert case_by_name("W903-no-deidentification").mutated
        assert not case_by_name("core-minimal").mutated

    def test_rebuilding_is_deterministic(self, tmp_path):
        """A corpus that changes between runs cannot be a contract."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        build_corpus(first, names=["seg-layers"])
        build_corpus(second, names=["seg-layers"])
        import h5py

        with (
            h5py.File(first / "seg-layers.medh5") as a,
            h5py.File(second / "seg-layers.medh5") as b,
        ):
            assert a.attrs["content_id"] == b.attrs["content_id"]


class TestPublishedSchema:
    def test_the_packaged_schema_matches_the_published_copy(self):
        """`schemas/` beside the spec and the packaged copy must not drift."""
        from medh5.document import SCHEMA_PATH

        published = Path(__file__).resolve().parents[2] / "schemas" / SCHEMA_PATH.name
        assert published.exists(), f"published schema missing at {published}"
        assert json.loads(published.read_text()) == json.loads(SCHEMA_PATH.read_text())

    def test_the_schema_is_reachable_as_package_data(self):
        from medh5.document import schema

        assert schema()["$schema"].endswith("2020-12/schema")
        assert set(schema()["required"]) == {"identity", "timepoints"}


class TestValidatorHonesty:
    def test_a_missing_schema_library_is_reported_not_assumed(self, sample_path):
        """Absence of `jsonschema` must not read as a clean document."""
        report = validate_file(sample_path, level="structural")
        assert report.checked["schema_checked"] is True


class TestSpecSync:
    """The spec's code table and the implementation must not drift apart."""

    @staticmethod
    def _spec_codes() -> set[str]:
        import re

        spec = (
            Path(__file__).resolve().parents[2] / "docs" / "spec" / "medh5-1.0.md"
        ).read_text()
        start = spec.index("### 15.2 Error codes")
        table = spec[start : spec.index("## 16. Versioning")]
        return set(re.findall(r"`([EW]\d{3})`", table))

    def test_S15_2_the_table_matches_the_registry(self):
        assert self._spec_codes() == set(CODES)

    def test_every_code_has_a_summary_and_a_domain(self):
        for code in CODES.values():
            assert code.summary and code.domain
            expected = "warning" if code.code.startswith("W") else "error"
            assert code.severity == expected
