# W21-03 Conformance suite

| | |
|---|---|
| Branch | `w21/conformance` |
| Size | L overall: PR 1 S–M, PR 2 S, PR 3 M–L |
| Lands after | W21-00 PR 4, PR 5, PR 6 (PR 1 edits `create_truncate`); PR 2 after W21-02 PR 1 and the first revision-2 code (W21-02 PR 2, or W21-13 PR 2 if it merges first) |
| Blocks | nothing |
| Owns | `crates/medh5/src/conformance/**` except `robustness.rs` (W21-01's), shared: each new code's case is added by its own workstream; `crates/medh5-cli/src/conformance.rs`, `medh5/conformance.py`, `crates/medh5/tests/conformance.rs`, `tests/format/test_conformance.py`, `docs/spec/conformance.md`, task-cache-1 §10 |
| Decisions | D1 (scoring side), D19 (§10) |
| Backport | none |

## Session prompt

> Implement workstream W21-03 (conformance suite) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D1, D19), then `plans/2.1.0/w21-03-conformance.md`
> in full. Work from the latest `main` on `w21/conformance`; land the three pull
> requests in order. Check before PR 1 that W21-00 PR 6 has merged (it edits the
> same `create_truncate`), and before PR 2 that W21-02 PR 1 has (the code
> revisions). The corpus is a shipped artifact third parties run: every stated
> count is tested (`TestStatedCounts`), so correct each sentence it names. Write
> tests first, run the local gate from `CLAUDE.md` (both `conformance run`
> commands included), add a fragment, open `W21-03: …` pull requests, and do
> not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 03.1 A suite built twice is the same bytes | S–M |
| 2 | 03.2 Each case states its version; scoring bounded by version and revision | S |
| 3 | 03.3 Cache fixtures, a companion scorer, rows compared both ways, code lists from `CODES` | M–L |

## Items

### 03.1 A suite built twice is the same bytes (PR 1)

- **State at 2.0.2.** Two publishes about three minutes apart differ in 163 of
  185 `SHA256SUMS` lines (every case file), with the same `content_id`s. Causes:
  `created` (`utcnow()`, `sample/writer.rs`), and group and root timestamps.
  Datasets are already untimed (`h5/data.rs`), but about 25 `create_group` sites
  (for example `h5/graph.rs`) and the root (`create_truncate`, `h5/file.rs`) are
  not. `core-minimal` differs in 43 bytes.
- **Change.**
  - `h5::ops::new_group(parent, name) -> Result<Group>`, using hdf5-metno 0.15's
    `GroupBuilder` with object time tracking off, replaces every non-test
    `create_group` in `crates/`.
  - `create_truncate` sets the file creation property list's object time
    tracking off (that covers the root group), beside the `close_on_exec` call.
  - The corpus builder and the cache fixtures (PR 3) pin `created` through a
    crate-private writer field.
  - Every engine write becomes reproducible apart from `created`; no digest or
    `content_id` changes (group timestamps are not format content).
- **Tests.** Rust `a_corpus_published_twice_is_the_same_bytes`: build the suite
  twice into two directories and compare `SHA256SUMS` (no spec clause governs
  it; say so in its docstring). A test that no non-test `create_group` call
  remains in `crates/medh5/src` (a source scan, with an empty allow-list).
- **Docs.** `docs/spec/conformance.md`: the suite is reproducible, so a mirror
  can check a published suite by rebuilding it (`medh5 conformance publish` then
  compare `SHA256SUMS`).
- **Changelog.** Added, area `validation`: "**The conformance suite builds to the
  same bytes every time.**"

### 03.2 Each case states its version; scoring is bounded (PR 2, after the first revision-2 code)

- **State at 2.0.2.** `expected.json` (`conformance/mod.rs`) records only a
  top-level `"format": "1.1"` and `"generator"`; no case records a version.
  Case files declare "1.0" (125), "1.1" (33), "1.0" but expecting 1.1's answers
  (the E009 and E011 cases), "1.2" (the W913 case), "2.0" (E002), none (E001).
  A 1.0-only implementation has no defined pass criterion, and a third party's
  score drops on cases added by erratum it was never told about.
  `conformance::Case` is an exhaustive struct with all-pub fields (no new field).
- **Change.**
  - `Case::format_version(&self) -> String`: the highest `since` among the
    case's expected codes, and "1.1" for a clause marked (1.1) (expect 127 cases
    at 1.0 and 36 at 1.1 before 2.1.0's additions; recount). Written into
    `expected.json` per case as `"format_version"`, beside a top-level
    `"codes_revision"`.
  - `suite::ScoreOptions` (`#[non_exhaustive]`, `Default`, builders
    `format_version(&str)`, `codes_revision(u32)`) and
    `suite::score_with(root: &Path, submitted: &[Value], options: &ScoreOptions)
    -> Result<Vec<CaseResult>>`; `score` delegates. A case above either bound is
    skipped (reported as such); submitted codes of a later revision than the
    bound are ignored.
  - Python `medh5.conformance.score(…, *, format_version=None,
    codes_revision=None)`; CLI `medh5 conformance score SUITE RESULTS
    [--format-version V] [--codes-revision N]`.
- **Tests.**
  - `test_S15_2_a_validator_of_an_earlier_revision_is_scored_against_it`: a
    results file holding none of the table's revision-2 codes (parametrised
    over those the table holds when the test runs, at least one) fails
    unbounded and passes with `--codes-revision 1`.
  - `test_S2_2_a_1_0_implementation_is_scored_on_the_cases_addressed_to_1_0`.
  - `test_every_case_states_its_format_version`.
- **Docs.** `docs/spec/conformance.md` (the two bounds, and the per-case member),
  the suite README (`suite.rs` `readme()`), `docs/reference/cli.md`.
- **Changelog.** Added, area `validation`.

### 03.3 Cache fixtures, a companion scorer, rows both ways, code lists from `CODES` (PR 3)

- **State at 2.0.2.** `conformance/companion.rs` builds 14 task fixtures
  (`valid-two-subjects`, T101–T306), published under `companion/` with
  `companion/expected.json`. There is no T4xx cache fixture. `run_fixtures` is
  reached only from `crates/medh5/tests/conformance.rs`, and compares neither `ok`
  nor each row's reasons; `conformance score` scores only the format corpus;
  `check()` compares rows one way (rows an implementation adds go unflagged).
  `crates/medh5/tests/conformance.rs` hard-codes T101–T306 and
  `tests/format/test_conformance.py` asserts `len(fixtures) == 14`.
- **Change.**
  - Cache fixtures under `companion/caches/`: one `.medh5cache` for each of
    T401–T407, one valid `event` cache and one valid `patient` cache
    (`companion/cache.rs` levels), each built with `CacheWriter` (with `created`
    pinned) and then mutated, with a task where T404–T406 need one.
    `companion/expected.json` gains entries `{name, file, task?, level: "cache",
    findings}`, checked through `medh5 cache validate [--task]`.
  - A scorer: Rust `conformance::companion::score(root: &Path, submitted:
    &[Value]) -> Result<Vec<CaseResult>>`; Python
    `medh5.conformance.score_companion(root, submitted) -> list[CaseResult]`; CLI
    `medh5 conformance score SUITE RESULTS --companion`.
  - `check()` compares rows both ways; `run_fixtures` compares `ok` and each
    row's reason codes.
  - Both tests derive their code lists from `companion::CODES`, so every T-code
    has a fixture (a code without one fails the test).
  - If W21-11 PR 1 has landed, its `valid-self-asserted-mask` fixture and the
    row's `inputs` member are already in `companion.rs`; keep them, and make the
    scorer compare `inputs` too.
- **Tests.** `test_S8_cache_fixtures_score_their_findings` (task-cache-1 §8);
  `test_S9_every_companion_code_has_a_fixture` (§9); `test_S4_an_extra_row_fails_a_fixture`;
  `TestStatedCounts` for the fixture counts.
- **Docs.** `docs/spec/conformance.md` ("fourteen task manifests" and the new
  cache fixtures), task-cache-1 §10 (the fixture list), the suite README,
  `docs/reference/cli.md`.
- **Changelog.** Added, area `task-cache`: "**The conformance suite scores
  feature caches and task results.**"
- **Done when** `test_S8_a_published_suite_scores_a_cache_implementation`
  passes: in a temporary directory, with the working directory elsewhere,
  `medh5 conformance publish OUT` writes the nine cache fixtures and their
  entries in `companion/expected.json`; `medh5 cache validate --json` run over
  each fixture (with its task where it has one) gives a results file; and
  `medh5 conformance score OUT RESULTS --companion` scores every fixture
  passed, reading no file outside `OUT` and the results.
