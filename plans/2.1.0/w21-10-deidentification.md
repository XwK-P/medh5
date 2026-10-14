# W21-10 De-identification

| | |
|---|---|
| Branch | `w21/deidentification` |
| Size | L overall: PR 1 S–M, PR 2 M, PR 3 M–L |
| Lands after | W21-00 PR 4, PR 5; PR 3 after W21-04 PR 1 (the clinical hold) |
| Blocks | nothing |
| Owns | `crates/medh5/src/curation/{scrub.rs,identity.rs}`, `crates/medh5-cli/src/scrub.rs`, `medh5/curation/scrub.py`, `docs/guides/deidentify.md`; touches `sample/writer*.rs` for `derive_deidentified` |
| Decisions | D14, D15, D26 |
| Backport | none |

## Session prompt

> Implement workstream W21-10 (de-identification) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D14, D15, D26), then
> `plans/2.1.0/w21-10-deidentification.md` in full. Work from the latest `main`
> on `w21/deidentification`; land the three items in order as one pull request,
> and check before opening it that W21-04 PR 1 has merged (it closes the
> clinical hold that PR 3's derivation relies on). Never attest work that was
> not done: a scrub that cannot see a field reports it as not examined, and
> keeps the rule names users gate on (`clinical`, `unreadable`, `too_deep`).
> Write tests first, write the decisions' spec text with their erratum rows, run
> the local gate from `CLAUDE.md`, add a fragment per item, open the `W21-10: …`
> pull request, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 10.1 Options in one checked type, and `scrub_each`; 10.2 unsalted name pseudonyms refused; 10.3 tool timestamps; 10.4 date spellings | S–M |
| 2 | 10.5 A withheld shift, a shift per subject, scrubbing to a copy | M |
| 3 | 10.6 Scrubbing a clinical sample | M–L |

## Items

### 10.1 Options in one checked type, and `scrub_each` (PR 1)

- **State at 2.0.2.** Rust `scrub_paths` (`curation/scrub.rs`) `collect()`s an
  iterator of `Result`s, so the first refused file ends the call and the reports
  of files already applied, whose `uid_map` is the only key to the pseudonyms
  written, are lost (2.0.2 fixed this in Python; the CLI loop avoids it). The
  apply-only options are listed three times and disagree: Python 4
  (`medh5/curation/scrub.py`), the CLI 3 (`crates/medh5-cli/src/scrub.rs`, whose
  grammar ties `--pseudonymise-ids` to `--apply`). `ApplyOptions` is an
  exhaustive struct. `Deidentification::shift_days` (`curation/identity.rs`)
  reads 0 as a shift while `recorded_shift` filters 0.
- **Change.**
  - `#[non_exhaustive] #[derive(Debug, Clone, Default)] pub struct ScrubOptions`
    with today's `ApplyOptions` fields, except that `date_shift_days:
    Option<i64>` becomes `date_shift: DateShift`, a new `#[non_exhaustive]` enum
    (`None`, the `Default`, and `Days(i64)`; PR 2 adds `Range`), so one field
    holds the shift however it was asked for. PR 2 adds more fields. `impl
    From<ApplyOptions> for ScrubOptions` (`date_shift_days: Some(d)` is
    `Days(d)`), and `ScrubOptions::check(&self, apply_changes: bool) ->
    Result<()>` (apply-only options without `apply`, pseudonymisation without a
    salt, a shift of 0). `pub fn apply_with(path: &Path, options: &ScrubOptions)
    -> Result<ScrubReport>`; `apply` delegates.
  - `pub fn scrub_each<P: AsRef<Path>>(paths: &[P], apply_changes: bool, options:
    &ScrubOptions) -> Result<Vec<Result<ScrubReport>>>`: the outer error is
    `check`'s, the inner ones per file. `scrub_paths` is
    `#[deprecated(since = "2.1.0", note = "stops at the first refused file and
    loses the reports before it; use scrub_each; removed in 3.0")]`.
  - The CLI and Python's `scrub_paths` call `check` instead of their own lists.
  - `shift_days` reads 0 as no shift, and `recorded_shift` uses it.
  - **Rule names.** The rules this workstream adds are `unsalted_pseudonym`
    (10.2), `tool_timestamp` (10.3), `locale_date` and `dicom_time` (10.4) and
    `date_shift_recorded` (10.5). `STRICT_RULES: [&str; 6]` is a public array,
    so it cannot grow: add `pub fn strict_rules() -> &'static [&'static str]`
    (today's six plus `date_shift_recorded`), have the engine use it, and mark
    `STRICT_RULES` `#[deprecated(since = "2.1.0", note = "lists only the rules
    of 2.0; use strict_rules(); removed in 3.0")]`, with a ledger line.
    `IDENTITY_RULES` does not change.
- **Tests.** Rust `s11_4_scrub_each_keeps_reports_after_a_refusal` (over
  [plain, clinical, plain]: two reports, one refusal, both written files carry
  their `uid_map`); Python and the CLI give the same message for each refused
  option combination; `shift_days` and `recorded_shift` agree on 0, integers and
  floats.
- **Changelog.** Deprecated (Rust), Fixed, area `deidentification`.

### 10.2 Unsalted name pseudonyms are refused (PR 1, D26)

- **State at 2.0.2.** Verified: strict mode without `--salt` wrote "John Smith"
  as `pseudo:ef61a579…`, which is `sha256("John Smith")[:32]`
  (`scrub.rs`); 2.0.2 decided "strict without salt allowed (unsalted, recorded)".
  Unsalted UID pseudonyms can be linked across releases.
- **Change.** `apply` refuses before writing when a name would be pseudonymised
  without a salt, worded as the `pseudonymise_ids` refusal; names go through
  `self.pseudonym()` so `uid_map` lists them. Unsalted UID pseudonyms get a review
  finding `unsalted_pseudonym`; 3.0 requires a salt (ledger).
- **Tests.** `test_S11_4_strict_refuses_to_hash_a_name_without_a_salt`;
  `test_S11_4_salted_names_are_in_the_uid_map`;
  `test_S11_4_an_unsalted_uid_pseudonym_is_reported`.
- **Changelog.** Security, area `deidentification`: "**A strict scrub no longer
  hashes a name without a salt.** The pseudonym could be reversed by hashing
  candidate names…"

### 10.3 Tool timestamps (PR 1)

- **State at 2.0.2.** Root `created` and provenance `started`/`ended` reveal the
  shift when a converter ran near acquisition.
- **Change.** The tool timestamps are the root `created` attribute (1.0 §2.1)
  and each provenance activity's `started` and `ended` (§11.1); no other time
  is a tool's. Under `strict`, `--apply` truncates each to its year, written
  `YYYY-01-01T00:00:00Z` so it stays an RFC 3339 timestamp (E604 otherwise),
  and the report lists each as acted on (rule `tool_timestamp`). Under `basic`,
  once a shift is applied, each is a review finding `tool_timestamp` ("a tool's
  time, unshifted beside shifted dates: it may date the acquisition"), left as
  it is. Times the scrub itself writes (its own activity) are truncated the
  same way under `strict`.
- **Tests.** `test_S11_4_strict_truncates_tool_times` (`created`, `started`,
  `ended` become `YYYY-01-01T00:00:00Z`, and the file validates);
  `test_S11_4_basic_reports_tool_times_after_a_shift`;
  `test_S11_4_basic_leaves_tool_times_alone_without_a_shift`.
- **Changelog.** Security, area `deidentification`.

### 10.4 Date spellings (PR 1)

- **State at 2.0.2.** The date rules recognise ISO and DICOM DA dates; DICOM TM
  times and locale dates (`31.01.2020`) are not reported.
- **Change.** Recognise them as review findings, never shifted (day/month order
  would be a guess): rule `locale_date` for a value that is wholly a
  `DD.MM.YYYY`, `DD/MM/YYYY` or `MM/DD/YYYY` date (a valid calendar date under
  either order), rule `dicom_time` for a value that is wholly a DICOM TM
  (`HHMMSS` with optional `.FFFFFF`, under a key whose name ends in `Time` or
  `_time`, so that a six-digit id is not read as a time).
- **Tests.** `test_S11_4_a_locale_date_is_reported`;
  `test_S11_4_a_dicom_time_is_reported`;
  `test_S11_4_a_six_digit_id_is_not_a_time`.
- **Changelog.** Fixed, area `deidentification`.

### 10.5 A withheld shift, a shift per subject, scrubbing to a copy (PR 2, D14)

- **State at 2.0.2.** One shift per run (`ApplyOptions`), written into every file;
  a recorded shift is reused, another refused. 1.0 §11.4: "`date_shift_days`
  records that dates were consistently shifted"; its example records −117.
  `apply` rewrites the file in place. Collections are refused (and `amend_into`
  refuses them).
- **Change.**
  - `--date-shift-range MIN:MAX`: the shift is taken from HMAC-SHA256(salt,
    original `subject_id`) over the nonzero days in the range, so one subject's
    files get one shift; the amount is always withheld.
  - `--withhold-shift`: the record omits `date_shift_days`; the activity's params
    hold `"dates": "shifted"`, `"shift": "withheld"`, `"shift_commitment":
    hex(HMAC(salt, "shift:" ‖ days))` (a re-run with the same salt and shift
    verifies against it; a different shift is refused).
  - The scan reads either form of record as "dates shifted"; a new finding
    `date_shift_recorded` (review; blocking under strict) reports a record that
    states its amount.
  - 2.1 raises `FutureWarning` when a shift is recorded without an explicit
    `withhold_shift=` ("medh5 3.0 withholds the date shift by default: pass
    withhold_shift=False to record it"); 3.0 withholds by default (ledger).
  - Merging an existing de-identification record keeps a `true`
    `burned_in_annotation_checked` and writes `date`. The record holds one
    `method` (its struct is exhaustive and its schema closed), so the method it
    replaces goes into this scrub's activity params as `"prior_method"`.
  - Copies: `--out FILE` and `--out-dir DIR`, through `amend_into`; the source is
    only read. In a directory the output is `<sample_id>.medh5`, or keeps the
    input's name when ids are not pseudonymised.
  - Rust: `DateShift` gains `Range(i64, i64)`; `ScrubOptions` gains
    `withhold_shift: Option<bool>` and `out: Option<PathBuf>`. (Both types are
    new in 2.1.0, so growing them between PR 1 and PR 2 breaks nothing
    published.)
  - Python, new keywords after the existing ones: `apply(path, *,
    profile="basic", salt="", date_shift_days=None, performed_by=None,
    pseudonymise_ids=False, date_shift_range: tuple[int, int] | None = None,
    withhold_shift: bool | None = None, out: str | os.PathLike[str] | None =
    None)`; `scrub_paths(..., pseudonymise_ids=False, date_shift_range=None,
    withhold_shift=None, out_dir=None)`. `date_shift_days=d` is
    `DateShift::Days(d)` and `date_shift_range=(a, b)` is `Range(a, b)`;
    giving both is refused by `check`.
  - CLI: `--date-shift-range`, `--withhold-shift`, `--out`, `--out-dir`, all in
    `APPLY_ONLY`.
  - D14's text in 1.0 §11.4 and 1.1 §3, with their rows; the valid conformance
    case `clinical-shifted-clock-withheld-shift` (in `conformance/clinical.rs`).
- **Tests.** `test_S11_4_a_withheld_shift_leaves_no_amount_in_the_file`;
  `test_S11_4_a_withheld_shift_still_passes_the_date_scan`;
  `test_S11_4_the_same_salt_and_shift_reapply_another_is_refused`;
  `test_S11_4_one_subjects_files_share_a_shift_from_a_range`;
  `test_S11_4_two_subjects_get_different_shifts`;
  `test_S11_4_a_recorded_shift_is_reported`;
  `test_S11_4_out_leaves_the_source_byte_identical`;
  `test_S11_4_a_burned_in_check_survives_apply`.
- **Docs.** `docs/guides/deidentify.md`, `docs/reference/cli.md`.
- **Changelog.** Added, Deprecated, Security (the recorded amount), area
  `deidentification`.

### 10.6 Scrubbing a clinical sample (PR 3, D15, after W21-04 PR 1)

- **State at 2.0.2.** `scan` adds one non-actionable `clinical` finding
  (`scrub.rs`); `apply` refuses every clinical sample. 1.1 §3: `shifted_utc`
  needs a de-identification record (E802); no conversion between clocks; date
  shifts preserve one timeline. 1.1 §6: de-identification covers the body;
  changed bytes make a new document and version. 1.1 §7.3 holds re-added
  versions (the writer enforces it: `set_clock` refuses another clock, E802;
  `check_held` holds a re-added version). Text is read on demand
  (`Documents::text`, `TextColumn::scan`).
- **Change.**
  - **Scan**, reading the clock, id columns and text on demand, one document at a
    time, with spans:

    | Rule | Reports | Status |
    |---|---|---|
    | `clinical_clock` | a `utc` clock | actionable with a shift |
    | `clinical_id` | `encounter_id`, `clock.id`, and event/record/document ids that match the id rules | actionable with `--pseudonymise-ids` |
    | `clinical_text` | the 1.0 name, date, UID and record-number rules over `value_text`, `missing_reason`, `origin_description` and each document | review, blocking |
    | `clinical` (name kept) | one finding per document, plus one for the short text columns: "free text not attested as de-identified" | blocking until attested |

  - **Apply**, a derivation (D15): one shift for the sample (from
    `--date-shift-days` or 10.5's range); `utc` becomes `shifted_utc`; every `*_us`
    bound moves by days × 86,400,000,000 (nulls stay null; a `relative` clock
    keeps its times); `clock.id` re-minted from the salt, so a subject's
    fragments still share a clock (T304); event, record and document ids
    re-minted consistently, with `source_id`, `target_id`,
    `asserted_by_event_id` and `supersedes` following; `encounter_id`
    pseudonymised. Text bytes untouched (spans stay valid). Requires `--salt`;
    refused while any `clinical` finding is unattested or any `clinical_text`
    hit remains.
  - **Attestation**: `--attest-text TOOL` records a `deidentify` activity with
    `params = {"scope": "clinical_text", "tool": TOOL, "digests": {path: §13.1
    digest}}`, honoured only while those digests match; pattern hits still
    block. It also works alone, as an amendment that touches nothing under
    `clinical/` (so §7.3 is not engaged): the "attest external
    de-identification" amendment.
  - Rust (`curation/scrub.rs`): `fn clinical_findings(sample: &Sample, report:
    &mut ScrubReport) -> Result<()>`; `fn derive_clinical(records: &Clinical,
    shift_us: i64, salt: &str) -> Result<(ClinicalRecords, Vec<Replaced>)>`.
    Writer: `pub(crate) fn derive_deidentified(path: &Path, out: &Path) ->
    Result<SampleWriter>` = `amend_into(path, out, None, &["clinical"])` plus a
    fresh clock. `ScrubOptions` gains `attest_text: Option<String>`; Python
    `apply(..., attest_text: str | None = None)`; CLI `--attest-text TOOL` in
    `APPLY_ONLY`.
  - The guide's gate adds `clinical_text` (`NOT_EXAMINED` keeps `clinical`,
    `unreadable`, `too_deep`).
  - D15's 1.1 §10 bullet and an Appendix A row.
- **Tests.** `test_S6_text_is_scanned_one_document_at_a_time` (a large document is
  never decoded whole); `test_S6_a_name_in_a_report_blocks_apply`;
  `test_S6_unattested_text_refuses_apply`;
  `test_S6_an_attestation_lapses_when_the_text_changes`;
  `test_S3_utc_becomes_shifted_utc_by_the_one_shift`;
  `test_S3_fragments_scrubbed_with_one_salt_share_a_clock`;
  `test_S7_3_reminted_ids_keep_every_link_resolving` (validates clean, no E813);
  `test_S9_a_shifted_cutoff_admits_the_same_versions`;
  `test_S11_4_the_clinical_rule_still_blocks_the_guide_gate`.
- **Docs.** `docs/guides/deidentify.md`, `docs/guides/clinical.md`,
  `docs/reference/cli.md`, 1.1 §10 and Appendix A; and, once W21-12 PR 3 has
  merged, the paragraph of `docs/guides/clinical-import.md` on what `scrub`
  covers.
- **Changelog.** Added, area `deidentification`: "**`scrub --apply` de-identifies
  a clinical sample, as a new sample.**"
