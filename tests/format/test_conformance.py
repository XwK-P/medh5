"""The conformance corpus is the contract (spec §15).

Every case is a file plus the exact diagnostic codes a conforming validator must
emit for it.  This module runs the corpus against *this* validator; a
third-party implementation runs the same manifest --- and is scored through the
same door, which is what `TestPublication` holds to.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import h5py
import pytest

import medh5
from medh5.cli import main
from medh5.collection import open_collection
from medh5.conformance import (
    CASES,
    build_corpus,
    case_by_name,
    check_checksums,
    load_manifest,
    publish,
    run_corpus,
    score,
    summarize,
)
from medh5.errors import CODES, MEDH5Error, MEDH5ValidationError
from medh5.validate import validate_file
from tests.helpers import ROOT

# Every diagnostic code in §15.2 now has a corpus case.  The set stays here so a
# future deferral has to be written down rather than silently opening a gap.
DEFERRED_CODES: frozenset[str] = frozenset()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> list:
    return run_corpus(tmp_path_factory.mktemp("corpus"))


@contextmanager
def _open_any(path: Path):
    """Open a corpus file whatever kind it is."""
    if path.suffix == ".medh5c":
        with open_collection(path) as shard:
            yield [shard[key] for key in shard]
        return
    with medh5.open(path) as sample:
        yield [sample]


def _samples_of(path: Path):
    """Every sample in a corpus file --- one, or one per collection member."""
    with _open_any(path) as samples:
        yield from samples


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

    def test_the_corpus_covers_every_code(self):
        covered = {c for case in CASES for c in (*case.errors, *case.warnings)}
        assert covered == set(CODES)

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
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
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
                "file_suffix",
            }

    def test_valid_cases_open_with_the_public_reader(self, tmp_path):
        valid = [c for c in CASES if c.valid]
        build_corpus(tmp_path, names=[c.name for c in valid])
        for case in valid:
            for sample in _samples_of(tmp_path / f"{case.name}{case.suffix}"):
                # Written at the lowest version the content needs (1.1 §2.3):
                # 1.0 for imaging, 1.1 with the clinical profile --- and the
                # higher-minor case is a projection, read as what is known.
                if "W913" in case.warnings:
                    assert sample.support == "projection", case.name
                else:
                    expected = "1.1" if "clinical" in sample.profiles else "1.0"
                    assert sample.version == expected, case.name
                    assert sample.support == "full", case.name
                assert len(sample.images) >= 1

    def test_unmutated_valid_cases_verify(self, tmp_path):
        """A file the writer produced end to end must also verify (spec §13)."""
        clean = [c for c in CASES if c.valid and not c.mutated]
        assert len(clean) >= 10
        build_corpus(tmp_path, names=[c.name for c in clean])
        for case in clean:
            for sample in _samples_of(tmp_path / f"{case.name}{case.suffix}"):
                assert sample.verify().ok, case.name

    def test_the_corpus_directory_holds_only_its_cases(self, tmp_path):
        """A shipped artifact must not carry the scaffolding that built it."""
        build_corpus(tmp_path, names=["collection-two-samples", "core-minimal"])
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "collection-two-samples.medh5c",
            "core-minimal.medh5",
            "expected.json",
        ]

    def test_S2_2_a_packed_sample_root_is_a_sample_root(self, tmp_path):
        """The containment the corpus asserts: no member is a lesser sample."""
        build_corpus(tmp_path, names=["collection-two-samples"])
        with open_collection(tmp_path / "collection-two-samples.medh5c") as shard:
            assert sorted(shard) == ["case.0", "case_1"]
            for key in shard:
                member = shard[key]
                assert member.content_id is not None
                assert member.profiles
                assert member.verify().ok, key

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

        with (
            h5py.File(first / "seg-layers.medh5") as a,
            h5py.File(second / "seg-layers.medh5") as b,
        ):
            assert a.attrs["content_id"] == b.attrs["content_id"]


class TestPublishedSchema:
    def test_S2_4_the_spec_names_the_schema_the_engine_embeds(self):
        """§2.4 names one file, and the engine embeds that file: E005 is checked
        against exactly the schema a third-party implementation reads."""
        from medh5.document import schema_text

        source = "crates/medh5/data/medh5-sample-1.0.schema.json"
        spec = (ROOT / "docs" / "spec" / "medh5-1.0.md").read_text(encoding="utf-8")
        assert f"`{source}`" in spec
        assert json.loads((ROOT / source).read_text(encoding="utf-8")) == json.loads(
            schema_text()
        )

    def test_the_schema_is_reachable_from_the_package(self):
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
        """1.0 §15.2's table, and the codes 1.1 §11.2 adds to it."""
        spec = (ROOT / "docs" / "spec" / "medh5-1.0.md").read_text(encoding="utf-8")
        start = spec.index("### 15.2 Error codes")
        table = spec[start : spec.index("## 16. Versioning")]
        minor = (ROOT / "docs" / "spec" / "medh5-1.1.md").read_text(encoding="utf-8")
        added = minor[
            minor.index("### 11.2 Diagnostic codes") : minor.index("### 11.3")
        ]
        return set(re.findall(r"`([EW]\d{3})`", table + added))

    def test_S15_2_the_table_matches_the_registry(self):
        assert self._spec_codes() == set(CODES)

    def test_S11_2_a_minor_version_adds_codes_and_redefines_none(self):
        """1.1 only adds: no code of its table is also one of 1.0's."""
        spec = (ROOT / "docs" / "spec" / "medh5-1.0.md").read_text(encoding="utf-8")
        table = spec[
            spec.index("### 15.2 Error codes") : spec.index("## 16. Versioning")
        ]
        minor = (ROOT / "docs" / "spec" / "medh5-1.1.md").read_text(encoding="utf-8")
        added = minor[
            minor.index("### 11.2 Diagnostic codes") : minor.index("### 11.3")
        ]
        base = set(re.findall(r"`([EW]\d{3})`", table))
        new = set(re.findall(r"`([EW]\d{3})`", added))
        assert new and not base & new
        assert {"E011", "W913", "W914"} <= new
        assert {f"E8{i:02d}" for i in range(1, 20)} <= new

    def test_every_code_has_a_summary_and_a_domain(self):
        for code in CODES.values():
            assert code.summary and code.domain
            expected = "warning" if code.code.startswith("W") else "error"
            assert code.severity == expected

    def test_S12_appendix_C_records_the_four_2_0_corrections(self) -> None:
        spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
        section = spec[spec.index("### C.1") : spec.index("### C.2")]
        for clause in ("| §2.5 |", "| §5.1, §13.2 |", "| §13.1 |", "| §14.3 |"):
            assert clause in section
        assert "four when the engine was written a second time, in Rust" in section

    def test_S12_appendix_C_states_how_many_clauses_it_lists(self):
        spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
        section = spec[spec.index("### C.1") : spec.index("### C.2")]
        table = section[section.index("| Clause | Correction |") :]
        rows = [
            line
            for line in table.splitlines()[2:]
            if line.startswith("| §") or line.startswith("| ")
        ]
        rows = rows[: next((i for i, r in enumerate(rows) if not r.strip()), len(rows))]
        words = {
            20: "Twenty",
            21: "Twenty-one",
            22: "Twenty-two",
            23: "Twenty-three",
            24: "Twenty-four",
            25: "Twenty-five",
            26: "Twenty-six",
            27: "Twenty-seven",
            28: "Twenty-eight",
            29: "Twenty-nine",
            30: "Thirty",
            31: "Thirty-one",
            32: "Thirty-two",
            33: "Thirty-three",
            34: "Thirty-four",
            35: "Thirty-five",
            36: "Thirty-six",
        }
        assert f"{words[len(rows)]} clauses have been corrected" in section
        # The 1.4.2 correction is recorded; 2.0's three follow it.
        assert any(r.startswith("| §10.1 |") for r in rows)


class TestPublication:
    """The suite as a shipped artifact, and scoring somebody else's validator."""

    @pytest.fixture(scope="class")
    def suite(self, tmp_path_factory) -> Path:
        return publish(tmp_path_factory.mktemp("suite"))

    def test_the_published_suite_stands_alone(self, suite):
        """Everything an implementer needs, without installing this package."""
        assert (suite / "expected.json").exists()
        assert (suite / "codes.json").exists()
        assert (suite / "medh5-sample-1.0.schema.json").exists()
        assert (suite / "medh5-clinical-1.schema.json").exists()
        assert (suite / "medh5-task-1.schema.json").exists()
        assert (suite / "medh5-cache-1.schema.json").exists()
        assert (suite / "README.md").exists()

    def test_the_task_and_cache_fixtures_score_through_the_cli(self, suite, capsys):
        """task-cache-1 §10: every fixture, through the public door."""
        from medh5.cli import main

        expected = json.loads((suite / "companion" / "expected.json").read_text())
        fixtures = expected["fixtures"]
        assert len(fixtures) == 14
        for fixture in fixtures:
            path = suite / "companion" / fixture["file"]
            main(["task", fixture["level"], str(path), "--json"])
            report = json.loads(capsys.readouterr().out)
            found = {f["code"] for f in report["findings"]}
            assert found == set(fixture["findings"]), fixture["name"]
            for row in report.get("rows", []):
                want = fixture["rows"].get(row["row_id"])
                if want is None:
                    continue
                assert row["status"] == want["status"], (fixture["name"], row["row_id"])
                admitted = (row["selection"] or {}).get("events", [])
                assert {e["event_id"] for e in admitted} == set(want["events"])
                assert row["target"]["status"] == want["target"]
        assert (suite / "SHA256SUMS").exists()
        codes = json.loads((suite / "codes.json").read_text(encoding="utf-8"))
        assert {c["code"] for c in codes["codes"]} == set(CODES)
        for case in load_manifest(suite)["cases"]:
            assert (suite / case["file"]).exists(), case["name"]

    def test_the_checksums_cover_every_published_file(self, suite):
        listed = {
            line.split("  ", 1)[1]
            for line in (suite / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        on_disk = {
            p.relative_to(suite).as_posix()
            for p in suite.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        }
        assert listed == on_disk
        assert not check_checksums(suite)

    def test_a_changed_case_is_caught(self, tmp_path):
        suite = publish(tmp_path / "s", names=["core-minimal"])
        target = suite / "core-minimal.medh5"
        target.write_bytes(target.read_bytes() + b"\x00")
        assert check_checksums(suite) == ("core-minimal.medh5",)

    def test_this_validator_scores_through_the_published_path(self, suite):
        """The reference implementation gets no private door into the suite."""
        submitted = []
        for case in load_manifest(suite)["cases"]:
            report = validate_file(suite / case["file"], level=case["level"])
            submitted.append({"file": case["file"], **report.to_json()})
        results = score(suite, submitted)
        assert summarize(results)["ok"], summarize(results)["failures"][:3]

    def test_the_minimal_submission_shape_is_enough(self, suite):
        """`{file, errors, warnings}` --- no need to reproduce our report."""
        submitted = []
        for case in load_manifest(suite)["cases"]:
            submitted.append(
                {
                    "file": case["file"],
                    "errors": case["expect_errors"],
                    "warnings": case["expect_warnings"],
                }
            )
        assert summarize(score(suite, submitted))["ok"]

    def test_a_missed_code_fails(self, suite):
        manifest = load_manifest(suite)
        broken = next(c for c in manifest["cases"] if c["expect_errors"])
        submitted = [
            {
                "file": c["file"],
                "errors": [] if c is broken else c["expect_errors"],
                "warnings": c["expect_warnings"],
            }
            for c in manifest["cases"]
        ]
        failed = [r for r in score(suite, submitted) if not r.ok]
        assert [r.case.name for r in failed] == [broken["name"]]
        assert failed[0].missing == tuple(sorted(broken["expect_errors"]))

    def test_an_invented_code_fails(self, suite):
        manifest = load_manifest(suite)
        submitted = [
            {"file": c["file"], "errors": [*c["expect_errors"], "E999"], "warnings": []}
            for c in manifest["cases"]
        ]
        results = score(suite, submitted)
        assert all("E999" in r.unexpected for r in results)

    def test_silence_about_a_case_is_a_failure_not_a_skip(self, suite):
        manifest = load_manifest(suite)
        submitted = [
            {
                "file": c["file"],
                "errors": c["expect_errors"],
                "warnings": c["expect_warnings"],
            }
            for c in manifest["cases"][1:]
        ]
        results = score(suite, submitted)
        missing = [r for r in results if r.error]
        assert len(missing) == 1
        assert missing[0].case.name == manifest["cases"][0]["name"]

    def test_a_case_this_build_does_not_know_is_still_scored(self, suite, tmp_path):
        """A suite published by a newer medh5 must not silently lose cases."""
        manifest = load_manifest(suite)
        manifest["cases"].append(
            {
                "name": "from-the-future",
                "file": "from-the-future.medh5",
                "level": "semantic",
                "expect_errors": ["E404"],
                "expect_warnings": [],
                "mutated": False,
            }
        )
        room = tmp_path / "future"
        room.mkdir()
        (room / "expected.json").write_text(json.dumps(manifest), encoding="utf-8")
        results = score(room, [{"file": "from-the-future.medh5", "errors": ["E404"]}])
        future = [r for r in results if r.case.name == "from-the-future"]
        assert len(future) == 1 and future[0].ok

    def test_scoring_without_a_manifest_says_so(self, tmp_path):
        with pytest.raises(MEDH5ValidationError, match="conformance publish"):
            score(tmp_path, [])

    def test_the_published_readme_counts_its_collections(self, tmp_path: Path) -> None:
        from medh5.conformance import CASES, publish

        shards = sum(1 for c in CASES if c.suffix == ".medh5c")
        assert shards == 6
        readme = (publish(tmp_path / "suite") / "README.md").read_text("utf-8")
        assert f"{len(CASES) - shards} samples and {shards} collections" in readme


class TestCorpusSmoke:
    """Every public read path, over every corpus case.

    The corpus checks that each case reports exactly its expected diagnostic
    codes.  It does not check that the rest of the read surface *survives* those
    files, and the surface is wide: `summary()`, `verify()`, and the grid, image,
    annotation and transform accessors are all documented entry points that no
    corpus test calls.  These hold two contracts over them --- a valid case is
    fully readable, and *any* case, including the deliberately malformed ones,
    fails only through `MEDH5Error`.  An `AttributeError` or a `KeyError` out of
    a public entry point is a bug even on invalid input, because it tells a
    caller nothing and cannot be caught by the documented exception type.

    What this is **not**: a guard against the `medh5 info` regression that
    prompted it.  That bug needed a class carrying two assertions, which is
    valid and so raises no diagnostic code and has no corpus case --- the corpus
    is organised one case per code.  Checked by re-introducing the regression:
    these pass with it in place.  It is guarded directly, in
    `TestMultiAssertionScopes`; the value here is breadth over 103 adversarial
    files, not that particular bug.
    """

    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("smoke")
        build_corpus(out)
        return Path(out)

    def _read_surface(self, sample) -> None:
        """Everything a consumer is documented to be able to call."""
        touched: list[object] = [
            sample.document.summary(),
            sample.summary(),
            sample.verify(),
            sample.identity,
            sample.profiles,
            sample.content_id,
        ]
        for grid in sample.grids.values():
            touched += [grid.affine, grid.shape, grid.spacing]
        for image in sample.images.values():
            touched += [image.grid, image.summary()]
        for ann in sample.annotations.values():
            touched += [
                ann.summary(),
                ann.class_ids,
                ann.annotated_class_ids,
                ann.classes,
            ]
        for transform in sample.transforms.values():
            touched.append(transform.summary())
        assert touched

    def test_every_valid_case_survives_the_whole_read_surface(self, built):
        cases = [c for c in CASES if c.valid]
        assert len(cases) >= 10
        for case in cases:
            path = built / f"{case.name}{case.suffix}"
            for sample in _samples_of(path):
                try:
                    self._read_surface(sample)
                except Exception as exc:
                    raise AssertionError(
                        f"{case.name}: {type(exc).__name__}: {exc}"
                    ) from exc

    def test_no_case_fails_through_an_undocumented_exception(self, built):
        """Invalid cases included: they are the adversarial inputs."""

        bad: list[str] = []
        for case in CASES:
            path = built / f"{case.name}{case.suffix}"
            try:
                for sample in _samples_of(path):
                    self._read_surface(sample)
                    validate_file(path)
            except MEDH5Error:
                continue  # a documented refusal is the correct outcome
            except Exception as exc:
                bad.append(f"{case.name}: {type(exc).__name__}: {exc}")
        assert not bad, (
            "public read paths raised undocumented exceptions:\n" + "\n".join(bad)
        )

    def test_medh5_info_succeeds_on_every_valid_case(self, built):
        """`info` is the command a user runs first; it must not exit non-zero.

        Collections are excluded because `info` refuses them deliberately, and
        says to use `open_collection()` instead.
        """

        cases = [c for c in CASES if c.valid and c.suffix == ".medh5"]
        assert len(cases) >= 10
        failed = [
            case.name
            for case in cases
            if main(["info", str(built / f"{case.name}{case.suffix}")]) != 0
        ]
        assert not failed, f"`medh5 info` exited non-zero on: {failed}"


class TestStatedCounts:
    """Every place the corpus size is written down agrees with the corpus (D-09).

    The count was maintained by hand in about ten places --- the README, the
    site's front page, the conformance page, Appendix C and the CI comments ---
    and each release that added a case had to find them all.  A documentation
    hook could render the site's copies, but not the README, the CI comments or
    the specification as GitHub shows it, so the check is a test instead.
    """

    SOURCES = (
        "README.md",
        "CLAUDE.md",
        ".github/workflows/ci.yml",
        "docs/index.md",
        "docs/spec/conformance.md",
        "docs/spec/medh5-1.0.md",
        # The crates' READMEs are their pages on crates.io; the CLI's said 117
        # for the release whose corpus had 153 (R08 of the 2.0 audit).
        *sorted(
            p.relative_to(ROOT).as_posix() for p in ROOT.glob("crates/*/README.md")
        ),
    )
    STATED = (
        r"(\d+)-case",
        r"(\d+) cases",
        r"(\d+)/\d+ cases pass",
        r"\d+/(\d+) cases pass",
        r"\(§15\) of (\d+) files",
        r'"cases": (\d+)',
        r'"passed": (\d+)',
        r"\d+ of the (\d+) cases",
    )

    def _stated(self, pattern: str) -> list[tuple[str, int]]:

        found = []
        for name in self.SOURCES:
            text = (ROOT / name).read_text(encoding="utf-8")
            found += [(name, int(n)) for n in re.findall(pattern, text)]
        return found

    def test_D09_every_stated_corpus_size_is_the_corpus_size(self):
        stated = [hit for pattern in self.STATED for hit in self._stated(pattern)]
        assert len(stated) >= 12, stated  # the count is still written down
        wrong = [(where, n) for where, n in stated if n != len(CASES)]
        assert not wrong, f"the corpus has {len(CASES)} cases; stated: {wrong}"

    def test_D09_the_stated_invalid_count_is_the_corpus_count(self):
        invalid = sum(1 for case in CASES if case.errors)
        stated = self._stated(r"(\d+) of the \d+ cases are deliberately invalid")
        assert stated and all(n == invalid for _, n in stated), (invalid, stated)

    def test_D09_the_conformance_page_breakdown_is_the_corpus_breakdown(self):
        """The page splits the total several ways, and each split drifts too.

        Only the total was checked, so the total stayed right while the page
        went on saying 39 valid cases, 111 samples and 71 mutated ones for
        several releases after all three had changed.
        """

        page = (ROOT / "docs/spec/conformance.md").read_text(encoding="utf-8")
        valid = [case for case in CASES if not case.errors]
        expected = {
            r"(\d+) are valid files": len(valid),
            r"(\d+) of them with specific warnings": sum(
                1 for case in valid if case.warnings
            ),
            r"(\d+) are invalid ones": len(CASES) - len(valid),
            r"(\d+) samples and \w+ collections": sum(
                1 for case in CASES if case.suffix == ".medh5"
            ),
            r"(\d+) of the cases prove it": sum(1 for case in CASES if case.mutated),
        }
        for pattern, count in expected.items():
            stated = [int(n) for n in re.findall(pattern, page)]
            assert stated == [count], (pattern, count, stated)
        words = ("zero", "one", "two", "three", "four", "five", "six", "seven")
        collections = sum(1 for case in CASES if case.suffix == ".medh5c")
        stated_words = re.findall(r"\d+ samples and (\w+) collections", page)
        assert stated_words == [words[collections]], (collections, stated_words)
