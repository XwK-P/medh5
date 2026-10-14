# W21-08 Cohorts: manifests, statistics, splits, checks

| | |
|---|---|
| Branch | `w21/cohorts` |
| Size | L overall: PR 1 M, PR 2 S, PR 3 M, PR 4 S–M |
| Lands after | W21-00 PR 4, PR 5 |
| Blocks | W21-11 PR 4 (after PR 1) |
| Owns | `crates/medh5/src/dataset/**`, `crates/medh5/src/labels/mod.rs` (the vocabulary rule), `crates/medh5-cli/src/{dataset.rs,labels.rs}`, `medh5/dataset/**`, `medh5/labels.py` (`check`) |
| Decisions | D21, D22, D25 |
| Backport | none |

## Session prompt

> Implement workstream W21-08 (cohorts: manifests, statistics, splits, checks)
> of medh5 2.1.0 in this repository. First read `plans/README.md`,
> `plans/2.1.0/README.md` and `plans/2.1.0/decisions.md` (D21, D22, D25), then
> `plans/2.1.0/w21-08-cohorts.md` in full. Work from the latest `main` on
> `w21/cohorts`. PR 1 is an enabler (W21-11 waits for its path rule, and PR
> 3's split keys use it): open it as a pull request of its own first; items 2–4
> follow, in order, as a second pull request. Every default whose result
> changes in 3.0 raises a `FutureWarning` through `medh5/_deprecation.py`, only
> where the answer will differ. Write tests first, run the local gate from
> `CLAUDE.md`, add a fragment per item, open the two `W21-08: …` pull requests,
> and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 08.1 Manifest paths resolve against the manifest (`medh5.dataset/1`); 08.2 `Entry.from_json` reads through the engine | M |
| 2 | 08.3 `ClassStats` delegates; 08.4 `examined_prevalence` | S |
| 3 | 08.5 `medh5.split/2` and `previous=` | M |
| 4 | 08.6 A severity for each cohort check; 08.7 `labels check` groups as C101 does | S–M |

## Items

### 08.1 Manifest paths resolve against the manifest (PR 1, D21)

- **State at 2.0.2.** `scan_with` stores `root` and each entry's path exactly as
  the caller named them (`crates/medh5/src/dataset/manifest.rs`); `save`,
  `load`/`from_json` copy that text; `stale`, `check` (`dataset/check.rs`) and
  `open_located` (`dataset/stats.rs`) stat the paths against the working
  directory. Run `medh5 dataset index studies -o out/m.json` from one directory
  and load it from another: `stale()` lists every file, `check` reports C402 "2
  manifest file(s) no longer exist" (false), the statistics open nothing. The
  manifest's `format` key is the *sample* format version; `sha256` covers ids
  only; `from_json` ignores unknown keys. Splits key assignments by the same path
  text (`dataset/split.rs`).
- **Change.**
  - Every save writes `"schema": "medh5.dataset/1"`. A path is written relative to
    the manifest's directory only when it lies beneath it, absolute otherwise; no
    `..`.
  - In memory a relative path stays relative to the working directory: `load`
    joins each stored path onto the manifest's directory as the caller named it,
    and `save` re-bases it. A path given absolute stays absolute unless the
    caller passes `paths="relative"`.
  - A manifest without `schema` is legacy: read against the working directory,
    as today, with a `FutureWarning` only when the two readings differ (some path
    relative and the manifest not in the working directory): "medh5 3.0 reads
    this manifest's paths against its own directory: pass base=… or run `medh5
    dataset upgrade`". `base=` names the base explicitly. An unknown `schema` is
    refused. `sha256` unchanged.
  - Rust: `pub const MANIFEST_SCHEMA: &str`; `Manifest::save_with(&self, path:
    &Path, paths: PathStyle) -> Result<PathBuf>` with `#[non_exhaustive] pub enum
    PathStyle { Keep, Relative, Absolute }`; `Manifest::load_with_base(path:
    &Path, base: Option<&Path>) -> Result<Manifest>`; `save` and `load` delegate;
    `to_json` adds `schema`.
  - Python: `Manifest.save(path, *, paths="keep")`, `Manifest.load(path, *,
    base=None)` (the private binding also returns whether the manifest was
    legacy).
  - CLI: `dataset index --absolute`; a new `medh5 dataset upgrade MANIFEST -o OUT
    [--base DIR] [--json]`, refusing if any entry is missing under the base.
  - D21's text on a reference page (a short "The cohort manifest" section in
    `docs/reference/` beside the cohort checks).
  - 3.0 (ledger): a legacy manifest whose readings differ is refused unless
    `base=` is given.
- **Tests.** `test_S12_3_a_saved_manifest_opens_from_any_directory` (after
  `chdir`, `stale() == []`, no C402);
  `test_S12_3_a_manifest_moved_with_its_cohort_still_opens`;
  `test_S12_3_an_entry_outside_the_manifest_directory_is_written_absolute`;
  `test_S12_3_absolute_paths_on_request`;
  `test_S12_3_a_legacy_manifest_reads_against_the_working_directory_and_warns`
  (and no warning when the manifest is in the working directory);
  `test_S12_3_an_unknown_schema_is_refused`; `test_S12_3_rebasing_changes_no_sha256`;
  `test_dataset_upgrade_refuses_an_entry_it_cannot_find`.
- **Docs.** `docs/guides/cohorts.md`, `docs/reference/cli.md`,
  `docs/reference/python-api.md`, the reference section.
- **Changelog.** Behaviour changes, area `cohorts`: "**A cohort manifest's paths
  are relative to the manifest.** The saved text changes only where it was
  already unreadable from any directory but the scan's…"; Deprecated for the
  legacy reading.

### 08.2 `Entry.from_json` reads through the engine (PR 1)

- **State at 2.0.2.** `medh5/dataset/manifest.py` `Entry.from_json` (with
  `_integers`) coerces a raw entry its own way, while the engine's reader
  (`dataset/manifest.rs`) is bound as `_core.dataset_entry_json` and `Entry.to_json`
  uses it. `Manifest.load` is safe (the engine normalises first);
  `Manifest.from_json` and `Entry.from_json` on a caller's document are not, and
  what they coerce is what `save` writes. Verified on 12 inputs:

  | Input | Python | Engine |
  |---|---|---|
  | `deidentified: "false"` | `True` | `False` |
  | `size: "12"` / `size: -1` | `12` / `-1` | `0` / `0` |
  | `mtime: "1.5"` | `1.5` | `0.0` |
  | `quality: {"q": 3}` | `{"q": 3}` | `{"q": "3"}` |
  | `content_id: 5` | `5` | `None` |
  | `profiles: null` / `"abc"` | `TypeError` / `('a', 'b', 'c')` | `[]` / `[]` |
  | `splits: [1]` | `TypeError` | `[]` |
  | bad `class_keys` | `int()`'s `ValueError` | the engine's `ValueError` |
  | `path: None` | `'None'` | `KeyError` |
  | `sex: 3` | `3` | omitted |

  So `Entry.from_json({..., "deidentified": "false"})` then `save` records a
  member as de-identified.
- **Change.** `Entry.from_json(doc)` becomes
  `cls._from_fields(_core.dataset_entry_json(doc))`; `_integers` is deleted.
- **Tests.** The 12 rows as one parametrised test:
  `Entry.from_json(doc).to_json() == _core.dataset_entry_json(doc)`, with the same
  exception type wherever the engine refuses.
- **Changelog.** Fixed, area `cohorts`.

### 08.3 `ClassStats` delegates to the engine (PR 2)

- **State at 2.0.2.** `ClassStats::prevalence()` and its JSON live in
  `dataset/stats.rs`; Python re-implements both (`medh5/dataset/stats.py`);
  `DatasetStats.to_json` already delegates (`_core.dataset_stats_json`).
- **Change.** `ClassStats.to_json` delegates through a new
  `_core.dataset_class_stats_json(stats)` (the `report_arg` pattern W21-06 PR 3
  uses), before 08.4, so 08.4 lands once, in the engine.
- **Tests.** An engine-produced `ClassStats` round-trips through Python to the
  engine's JSON.

### 08.4 `examined_prevalence` (PR 2)

- **State at 2.0.2.** `ClassStats::prevalence()` is 0.0 when `examined_in == 0`;
  the JSON writes it and `from_json` ignores it; the CLI prints it
  (`crates/medh5-cli/src/dataset.rs`); `tests/tools/test_dataset.py` asserts 0.0.
  Verified: a class drawn with 27 voxels and examined nowhere reports 0.0, which
  reads as "never found where looked" for a class nobody looked for.
- **Change.**
  - Rust: `pub fn examined_prevalence(&self) -> Option<f64>`;
    `#[deprecated(since = "2.1.0", note = "0.0 for a class no sample examined;
    use examined_prevalence; Option<f64> in 3.0")] pub fn prevalence(&self) ->
    f64`; `to_json` writes `examined_prevalence()` under `"prevalence"` (so
    `null`). The CLI and the bindings move in the same pull request (clippy
    `-D warnings`).
  - Python: `ClassStats.examined_prevalence: float | None` (a property, not a new
    dataclass field); `prevalence` still returns 0.0 there but raises
    `FutureWarning` *only there*: "medh5 3.0 returns None for the prevalence of a
    class no sample examined: use examined_prevalence". Correct callers never see
    it.
  - The CLI prints `-`.
- **Tests.** `test_S11_3_prevalence_of_a_class_nobody_examined_is_undefined`
  (`None`, JSON null, and the warning with 0.0);
  `test_S11_3_examined_prevalence_counts_only_examining_samples`;
  `test_S11_3_prevalence_warns_only_when_undefined`;
  `test_S11_3_stats_round_trip_keeps_null_prevalence`; update the 0.0 assertion;
  Rust `s11_3_examined_prevalence_is_none_without_an_examiner`.
- **Docs.** `docs/guides/cohorts.md`, `docs/guides/partial-labels.md`,
  `docs/reference/python-api.md`.
- **Changelog.** Behaviour changes (the JSON `null`: a saved 0.0 was a recorded
  false measurement), Deprecated, area `cohorts`.

### 08.5 `medh5.split/2` and `previous=` (PR 3, D22)

- **State at 2.0.2.** `make_splits` (`dataset/split.rs`) ranks groups by hash
  within strata, interleaves strata round-robin and deals by the largest deficit
  against global shares (`deal`), so a group's partition depends on every other
  group: adding one group to 100 moves a median of 34 existing groups (max 57),
  9 of them test→train; to 1000, a median of 300. Rare strata of 3, 5 and 7 come
  out 2/1/0, 4/1/0, 6/1/0 at 70/15/15. No algorithm is recorded. `Split` and
  `SplitOptions` are structs with only pub fields (no new fields). §12.3: claims
  carry `manifest_sha256`; the manifest is authoritative.
- **Change.** D22's algorithm and `previous=`.
  - Rust:

    ```rust
    #[non_exhaustive] pub enum SplitAlgorithm { V1, V2 }   // "medh5.split/1", "medh5.split/2"
    #[non_exhaustive] #[derive(Debug, Clone, Default)]
    pub struct SplitRequest { /* SplitOptions' fields */ pub algorithm: SplitAlgorithm, pub previous: Option<SplitRecord> }
    #[non_exhaustive] pub struct Lineage { pub manifest_sha256: String, pub kept: usize, pub added: Vec<String>, pub dropped: Vec<String> }
    #[non_exhaustive] pub struct SplitRecord { pub split: Split, pub algorithm: SplitAlgorithm, pub lineage: Option<Lineage> }
    pub fn make_splits_with(manifest: &Manifest, request: &SplitRequest) -> Result<SplitRecord>;
    ```

    (`SplitAlgorithm` needs a `Default`: `V1` in 2.1.)
  - The split JSON gains `"algorithm"` and `"previous": {manifest_sha256, kept,
    added, dropped}`; a JSON without `algorithm` reads as /1. Assignments key by
    the manifest's paths as 08.1 resolves them.
  - Python: `make_splits(manifest, *, …, algorithm: Literal["medh5.split/1",
    "medh5.split/2"] | None = None, previous: Split | str | os.PathLike[str] |
    None = None)`; omitting `algorithm` raises `FutureWarning` only when /2
    would place some group differently (D22's Deprecation: the function deals
    both ways). Tests and docs that split without `algorithm=` pass it from now
    on, so the suite's `filterwarnings` stays quiet. `Split` gains appended
    fields `algorithm: str = "medh5.split/1"` and `previous: Mapping[str, Any] |
    None = None`, and `moved_from(other: Split) -> dict[str, tuple[str, str]]`.
  - CLI: `medh5 dataset split --algorithm 1|2 --previous SPLIT.json`.
  - No new code: C201 marks claims from the previous manifest stale (correct);
    re-run `write_claims`.
- **The split cohort** (`tests/kits.py`, `SplitCohortKit`): N one-file groups,
  subject ids `s000`, `s001`, …, each with a `site` attribute for
  `stratify_by`: three strata holding 60 %, 30 % and 10 % of N, and three rare
  strata of 3, 5 and 7 groups; ratios 70/15/15. Build the samples with the
  public writer (the smallest valid sample each), or the manifest's entries
  directly if `Manifest` accepts entries without files. The moving tests take
  N = 100 and add one group to the largest stratum, over seeds 0–19; measure
  /1 on the same cohort and write its median in the test's docstring (the
  research measured 34).
- **Tests.** `test_S12_3_split2_gives_a_rare_stratum_a_test_group`;
  `test_S12_3_split2_meets_the_global_totals`;
  `test_S12_3_split2_one_new_group_moves_few` (20 seeds; median ≤ 2 where /1
  moves 34); `test_S12_3_previous_moves_no_group` (100 → 300, `moved_from`
  empty); `test_S12_3_previous_keeps_a_kept_subjects_new_file_in_its_partition`;
  `test_S12_3_previous_refuses_other_parameters`;
  `test_S12_3_previous_lists_dropped_groups`;
  `test_S12_3_omitting_algorithm_warns_where_split2_differs` (and is silent on a
  cohort both deal alike; the native `dataset split` prints the same warning);
  `test_S12_3_a_2_0_split_reads_as_split1`; Rust
  `s12_3_controlled_rounding_is_deterministic`.
- **Docs.** `docs/guides/cohorts.md` (the algorithm, and why `previous=` prevents
  a leak), `docs/reference/cli.md`, `docs/explanation/design-rationale.md`.
- **Changelog.** Added and Deprecated, area `cohorts`.

### 08.6 A severity for each cohort check (PR 4)

- **State at 2.0.2.** `CHECK_CODES: [(&str, &str); 12]` (`dataset/check.rs`)
  holds codes and summaries; Python gets `_core.CHECK_CODES: dict[str, str]`;
  `CHECK_SEVERITIES` is already taken (the severity vocabulary). Each
  `report.add` chooses its severity: error C102, C201, C202, C402; warning C101,
  C103, C203, C301, C501; info C302; C401 warning, or error with `--deep`; C204 a
  refusal raised by `dataset split`. The hook parses the pairs with a regex for
  `docs/reference/cohort-checks.md`.
- **Change.** A parallel `pub const CHECK_SEVERITY: &[(&str, &str)]` (a slice,
  per the rules) with each code's base severity (C204 "error"), and `pub fn
  code_severity(code: &str) -> Option<&'static str>` used at the call sites;
  Python `CHECK_CODE_SEVERITY: dict[str, str]`; the hook renders `| Code |
  Severity | Meaning |` and states C401's promotion under `--deep` beneath.
- **Tests.** Rust `every_code_has_one_severity_from_the_vocabulary`,
  `every_finding_carries_its_codes_severity` (fixtures triggering each code),
  `c401_is_an_error_only_with_deep`; Python
  `test_check_code_severity_matches_the_findings`; a docs test that the rendered
  column equals the table.
- **Changelog.** Documentation (and Added for the constant).

### 08.7 `labels check` groups as C101 does (PR 4, D25)

- **State at 2.0.2.** The rule exists only in the CLI
  (`crates/medh5-cli/src/labels.rs` `check`): a file's key is
  `{id}@{version}#{sha256[..16]}` when its label set has a class or declares a
  digest, `<none>` otherwise; files grouped by key; JSON `{ok, vocabularies,
  unreadable, files}`; no Python API. `dataset check`'s C101/C103 group by the
  full `label_set_digest`, which the manifest records for every label set, empty
  or not.
- **Change.** In `crates/medh5/src/labels/mod.rs`:

  ```rust
  /// What decides that two files share class ids: the label set's digest; `None` when the file declares none.
  pub fn vocabulary_of(label_set: Option<&LabelSet>) -> Option<String>;
  /// How `labels check` prints a vocabulary: `id@version#` and 16 hex digits.
  pub fn vocabulary_key(label_set: &LabelSet) -> String;
  #[non_exhaustive]
  pub struct VocabularyCheck {
      pub groups: IndexMap<Option<String>, Vec<String>>,
      pub keys: IndexMap<Option<String>, String>,
      pub unreadable: IndexMap<String, Error>,
  }
  impl VocabularyCheck { pub fn ok(&self) -> bool; pub fn to_json(&self) -> Value; }
  pub fn check_vocabularies<P: AsRef<Path>>(paths: &[P]) -> VocabularyCheck;
  ```

  The CLI groups by `vocabulary_of` and keeps its printing and JSON;
  `medh5.labels.check(paths) -> dict[str, Any]` returns the CLI's JSON (binding
  `_core.labels_check`).
- **Tests.** A cohort of two empty label sets with different ids, a file with no
  label set and two equal non-empty sets gives the same grouping from `medh5
  labels check --json`, from `medh5.labels.check` and from `dataset check`'s
  C101/C103.
- **Changelog.** Behaviour changes, area `cohorts` (empty label sets); Added
  (`medh5.labels.check`).
