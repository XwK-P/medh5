# W21-11 Task and cache companion

| | |
|---|---|
| Branch | `w21/task-cache` |
| Size | L overall: PR 1 S–M, PR 2 S, PR 3 M, PR 4 L, PR 5 M |
| Lands after | W21-00 PR 4, PR 5; PR 3 after W21-05 PR 1 (`_core.write_atomic`, `stage_file`); PR 4 after W21-08 PR 1 (cohort paths) |
| Blocks | W21-12 (after PR 2 and PR 4), W21-16 PR 4 (after PR 4) |
| Owns | `crates/medh5/src/companion/**`, `crates/medh5/src/clinical/{select.rs,model.rs}` (selection, clocks, rounding), `crates/medh5-cli/src/clinical.rs` (task and cache commands), `crates/medh5-python/src/{clinical.rs,preflight.rs}`, `medh5/{task.py,cache.py,clinical.py}`, `docs/spec/task-cache-1.md`; touches `h5/file.rs` (`open_read`'s non-regular-file refusal) and `conformance/companion.rs` (one fixture) |
| Decisions | D17, D19, D24, D32, D33 |
| Backport | PR 1 |

## Session prompt

> Implement workstream W21-11 (task and cache companion) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D17, D19, D24, D32, D33), then
> `plans/2.1.0/w21-11-task-cache.md` in full. Work from the latest `main` on
> `w21/task-cache`; land the five pull requests in order. PR 1 is a security and
> correctness fix that may be backported to 2.0.3: keep it small and free of new
> API. `medh5.task/1` and `medh5.cache/1` take errata that add no field and
> never change a normal form, a fingerprint or a default under `/1`; an erratum
> may say what an implementation must refuse to open or must not report, never
> what a manifest must contain. Rows index, never copy; document text is read
> on demand. Write
> tests first, write the decisions' text into task-cache-1 (its Appendix A is
> created by PR 1) and 1.1 with their rows, run the local gate from `CLAUDE.md`,
> add fragments, open `W21-11: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 11.1 A source off its pin is read no further; a pipe is not opened; 11.2 an asserter that is its own endpoint | S–M |
| 2 | 11.3 One rounding of hours and days; 11.4 default source ids that do not collide | S |
| 3 | 11.5 Confining sources; 11.6 reconcile and save through the engine | M |
| 4 | 11.7 A task from a cohort; repinning; clocks and days; caches through their task's pins | L |
| 5 | 11.8 The follow-up note; 11.9 constants from the engine; 11.10 leftovers | M |

## Items

### 11.1 A source off its pin is read no further; a pipe is not opened (PR 1, D19 §2)

- **State at 2.0.2.** task-cache-1 §2: a `uri` is a path (absolute or relative
  to the manifest's directory) or a `file:` URI, and a reference "counts only
  while" steps 1–4 hold. `SourceRef::resolve` (`companion/source.rs`) accepts any
  path. `load_subject` (`companion/view.rs`) moves on after T301, but after
  `source.check` it keeps reading the file: T303 prints its subject id (the
  probe below printed `'MRN-0042-SECRET'`), T305 its event ids, and `--json` its whole event
  table, lab values included; the T302 for a stored `content_id` that differs
  from the pin prints that stored value (`source.rs`; `medh5/cache.py` does the
  same). `view::reconcile` opens every source and checks no pin. `open_read`
  (`h5/file.rs`) refuses only a directory, so a named pipe given as a source
  blocks `task preflight`, `task reconcile` and `cache validate` inside
  `H5Fopen`. (T301 telling an absent path from a non-HDF5 file is an existence
  oracle; only confinement, PR 3, closes it.)
- **The probe.** A manifest mapping two sources to subject `S-01`. The first
  reference pins `content_id` X and `sample_key` K, but its `uri` names a
  clinical sample whose stored `content_id` is Y ≠ X, whose
  `identity.subject_id` is `MRN-0042-SECRET` (T303 at 2.0.2), whose clock id is
  `clk-secret`, and which holds a `lab` event `ev-secret` with `value_num` 42.0
  and a document `doc-secret`. The second, pinned correctly, has clock
  `clk-01` and holds `ev-secret` with `value_num` 7.0 (T304 and T305 at
  2.0.2). Build both samples with the public writer
  in `tests/kits.py` (`HostileManifestKit`); the Rust test builds the same
  through `crate::sample::writer`.
- **Change.**
  - In `load_subject`: `let found = source.check(&sample, deep)?; if
    !found.is_empty() { history.findings.extend(found); continue; }`: a source
    that fails steps 1–4 adds no identity, clock, event, document or link, and no
    finding but its own speaks of its content.
  - `reconcile` refuses a source off its pin with `Error::coded("T302", …)`.
  - T302 names only the pin: "{locator} is not the version the reference pins
    ({pin}): `medh5 task repin` pins what is there, if the change is meant" (Rust
    and `medh5/cache.py`; `repin` arrives in PR 4, so until then end the sentence
    at the pin).
  - `open_read`, after the directory arm: `Ok(meta) if !meta.is_file() => return
    Err(failed(&"it is not a regular file"))`.
  - task-cache-1: create "Appendix A. Errata"; D19's §2 text, its last two
    sentences now (refusing non-regular files; a failing source read no further),
    the confinement sentences with PR 3.
  - `SECURITY.md`: a section "Running a task manifest you did not write": what a
    manifest can make medh5 open, what pins guarantee, and (from PR 3) how to
    confine.
- **Tests.** Rust `companion::view::tests::s2_a_source_off_its_pin_is_read_no_further`
  (only T302: no T303, no T305, no events);
  `s2_reconcile_refuses_a_source_off_its_pin`;
  `h5::file::tests::s2_a_fifo_is_refused_before_hdf5_opens_it` (Unix `mkfifo`;
  the call returns instead of blocking). Python
  `tests/tools/test_task_cache.py::test_S2_a_hostile_manifest_reads_nothing_it_does_not_pin`
  (replays the probe: the only finding is the first source's T302, and none of
  `MRN-0042-SECRET`, `clk-secret`, `doc-secret`, Y or `42.0` appears anywhere
  in `to_json()`, in the CLI's `--json`, or in its text output; the second
  source's `ev-secret` at 7.0 does, as it should).
- **Changelog.** Security, area `task-cache`: "**A task manifest reads and reports
  nothing its pins do not name, and a pipe given as a source is refused.**"

### 11.2 An asserter that is its own endpoint (PR 1, D17)

- **State at 2.0.2.** `select.rs` checks `eligible[*by]`, then `"event" =>
  event.is_some_and(|e| chosen[e])`, so when the link's source *is* its
  asserter, the asserter must be admitted after all. Run on the `History` kit
  (seg0 a `procedure` of `unknown` time, available at 2 h; row at 24 h):
  self-asserted under `effective` gives `center_fallback` and no annotations;
  the same link phrased from `ct0` gives `eligible_instances` at (3, 2, 7) and
  `('lesions_tp0',)`; under `available` both agree. The 2.0.2 test uses the `ct0`
  phrasing; both shipped examples (`conformance/clinical.rs`,
  `docs/examples/clinical_longitudinal.py`) use the self-asserted one. No
  conformance case checks payload admission (1.1 §9.1 step 5): the fixture
  checker compares status, event ids, the slot's image id and the target; the
  valid fixture's cutoffs come before seg0 is available, and its description
  states the bug as the rule ("the annotations attested by events of unknown time
  are never inputs").
- **Change.** `select.rs`: `"event" => event.is_some_and(|e| chosen[e] || e ==
  *by),`. Python and the CLI call this code. D17's text in 1.1 §7.3, §9.1 step 5
  and Appendix A, and TC §3.5 with its Appendix A row. A fixture
  `valid-self-asserted-mask` (the worked subject at `worked@h2250`, after seg0's
  2200 h and before grounding0's 2300 h) whose row gains an optional member
  `"inputs": {"ct": {"roi": …, "annotations": [...]}}` that `check()` compares
  (`["lesions_tp1"]`; under 2.0.2 it would be `[]`); correct `valid-two-subjects`'s
  description to "no annotation is an input before the procedure that drew it is
  known (hour 2200)". `docs/explanation/design-rationale.md`: "…or a link a
  version known by then asserts, about itself or between admitted events".
- **Tests.** Rust `clinical::select::tests::s7_3_an_asserter_is_its_own_endpoint`
  (admits `lesions_tp0`; `seg0` is still not an input),
  `s9_1_a_document_endpoint_needs_its_event_admitted`; Python
  `tests/format/test_clinical.py::test_S7_3_an_asserter_is_its_own_endpoint`
  (both shapes agree under both orderings),
  `tests/tools/test_task_cache.py::test_S3_5_a_self_asserted_mask_centres_the_crop`
  (`roi == "eligible_instances"`, centre (3, 2, 7)); the corpus test and
  `TestStatedCounts` (15 task fixtures).
- **Changelog.** Behaviour changes, area `clinical`: "**A procedure's assertion
  about its own mask attests the mask.** … rebuild caches whose slots read
  self-asserted masks: no fingerprint changes, so they still validate."

### 11.3 One rounding of hours and days (PR 2, D32)

- **State at 2.0.2.** Python's `hours`/`days` round half to even
  (`medh5/clinical.py`: `int(round(value * HOUR))`). `--cutoff-hours`
  (`crates/medh5-cli/src/clinical.rs`, range check in `app.rs`) and augment's
  `day_after` (`clinical/augment.rs`) round half away from zero: at exact ties
  they differ by 1 µs (`1.25e-09` hours: Python 4, CLI 5), and admission is a
  closed bound.
- **Change.** In `clinical/model.rs`: `pub fn round_micros(value: f64, unit_us:
  i64) -> Result<i64>` (`round_ties_even`; a non-finite or out-of-int64 result is
  `Error::Value` naming the value and unit), `pub fn hours(value: f64) ->
  Result<i64>`, `pub fn days(value: f64) -> Result<i64>`. `clock_hours` and
  `select_cmd` call `hours`; `day_after` calls `round_micros(days, DAY)`. Python
  keeps its arithmetic; `_core.clinical_round_micros(value, unit_us)` is bound for
  the cross-check only. The docstrings and `docs/reference/cli.md` state the
  rounding.
- **Tests.** Rust `clinical::model::tests::s3_times_round_half_to_even` (the ties
  above; ±2^63 refused); Python `test_S3_hours_and_days_are_the_engines` (a sweep
  of ties: `hours(x) == _core.clinical_round_micros(x, HOUR)`); CLI
  `test_cutoff_hours_rounds_as_hours_does` (`--cutoff-hours=1.25e-09` reports
  `cutoff_us` 4).
- **Changelog.** Behaviour changes, area `clinical`.

### 11.4 Default source ids that do not collide (PR 2, D33)

- **State at 2.0.2.** `_default_source_id` (`medh5/task.py`) builds
  `{parent.name}/{stem}`, adds `:{key}` for a collection member, replaces
  characters ids cannot hold with `_`, cuts at 128; it runs in `__post_init__`,
  and `add_subject` cannot tell a derived id from a given one. T204 is checked
  across the manifest (`task.rs`). Collisions: `a/site-a/P-01.medh5` beside
  `b/site-a/P-01.medh5`; `P-01.medh5` and `P-01.h5`; long names cut to one 128.
  It splits with the platform's `PurePath`: `..\x\y.medh5c` gives `.._x_y` on
  POSIX and `x_y` on Windows, so one script's fingerprints differ by platform.
  Rust derives no default (`SourceRef::new` leaves it empty: T101).
- **Change.** `impl SourceRef { pub fn default_id(uri: &str, sample_key:
  Option<&str>) -> String }`: Python's derivation, with a `file:` URI resolved,
  any other URI split on `/` only (a backslash is an ordinary character), the
  characters a `const ID_CHARS` checked against the schema's `$defs/id`.
  `TaskManifest::add_subject(&mut self, subject: Subject) -> &mut Self` widens
  only a *derived* id that collides: first by the directories above it, then
  `@2`, `@3`, within 128 characters; the first holder keeps its id; an explicit
  colliding id is still T204. Binding `_core.source_default_id(uri,
  sample_key=None, taken=())`. Python: `SourceRef` gains `_derived: bool =
  field(default=False, init=False, repr=False, compare=False)`;
  `_default_source_id` and its `import re` go; `add_subject` re-derives a taken
  derived id.
- **Tests.** `test_S2_colliding_default_ids_are_widened` (`a_site-a_P-01` and
  `b_site-a_P-01`, validating clean); `test_S2_default_ids_widen_then_number`
  (`site-a_P-01`, `site-a_P-01@2`); `test_S2_an_explicit_colliding_id_is_still_T204`;
  `test_S2_uncolliding_default_ids_are_unchanged` (a table of 2.0.2's outputs);
  a URI table with fixed ids, identical on every CI platform (relative, `../`,
  absolute `file:`, `s3://b/x.medh5`, backslashes, 200 characters, empty); Rust
  `companion::source::tests::s2_default_ids_are_pythons`.
- **Changelog.** Fixed, area `task-cache`.

### 11.5 Confining sources (PR 3, D19 §2)

- **State at 2.0.2.** As 11.1; confinement by default would break every manifest
  with `../data/…` or absolute URIs, and confining by canonical path would refuse
  the symlinked layouts ML data directories are full of.
- **Change.**
  - Rust `companion::source::SourceAccess { pub base: Option<PathBuf>, pub root:
    Option<PathBuf> }` (`#[non_exhaustive]`), with `SourceAccess::new(base)`,
    `SourceAccess::confined(base, root)`, and `fn locate(&self, source:
    &SourceRef) -> Result<PathBuf>`: join the URI to the base, resolve `.` and `..`
    over `Path::components` lexically, compare with the root (base and root are the
    runner's: canonicalized once). Outside the root is T301 naming the URI and
    nothing about the path, with the same message whether or not the path exists.
  - New entry points taking it: `SourceRef::open_in(&self, &SourceAccess) ->
    Result<Sample>`; `view::{preflight_in, preflight_each_in, write_preflight_in,
    reconcile_in}`; `Admitted::preflight_in`; `cache::validate_cache_in`. The 2.0
    functions delegate with `SourceAccess::new(base)`.
  - Python: `confine: bool | os.PathLike[str] | None = None` on
    `TaskManifest.preflight`, `.reconcile`, `.repin` (PR 4), the module-level
    `preflight`, `validate_cache`, `build_document_cache`, `ClinicalTaskDataset`:
    `True` the base, a path that root, `False` unconfined and silent, `None`
    (2.1's default) unconfined but a *loaded* manifest whose source lies outside
    its base raises `FutureWarning` ("medh5 3.0 confines a task's sources to its
    directory: pass confine=False to keep reading …"); a manifest built in memory
    stays unconfined and silent. `SourceRef.open(base, *, root=None)`,
    `SourceRef.resolve(base, *, root=None)`.
  - CLI: `--source-root DIR` and `--any-source-path` (conflicting) on `task
    preflight`, `task reconcile`, `task repin` (PR 4) and `cache validate`; the
    stderr note when neither is given and a source lies outside. `task reconcile
    -o`'s same-file check locates sources through `SourceAccess`, so it stats
    nothing outside the root.
  - D19's confinement sentences in TC §2; `SECURITY.md`'s section gains how to
    confine.
- **Tests.** Rust `companion::source::tests::s2_confinement_is_lexical` (`a/../b`
  inside; `../x`, `/abs`, `file:///…` outside; `link/../x` opens `root/x`); Python
  `test_S2_confine_refuses_a_path_outside_the_root`,
  `test_S2_an_outside_source_warns_until_3_0`; CLI
  `test_task_preflight_source_root_and_any_source_path`. No corpus fixture
  (confinement is the runner's option).
- **Changelog.** Security and Deprecated, area `task-cache`.

### 11.6 Reconcile and save through the engine (PR 3, D24)

- **State at 2.0.2.** The reconcile loop is written twice: the CLI's
  `task_reconcile` (`crates/medh5-cli/src/clinical.rs`: per subject, a count,
  `fingerprint = None`, save) and the binding's (`crates/medh5-python/src/clinical.rs`:
  no count, the declared fingerprint kept). Only the CLI refuses an `--out` that
  is one of the sources; `TaskManifest.load(m).reconcile().save("ct.medh5")`
  replaces a patient's sample with JSON. The engine writes a manifest with
  `json::pretty`, ASCII-escaped; Python with `json.dumps(indent=2,
  ensure_ascii=False)` (621 against 629 bytes for one document, only `"Müller"`
  differing); Python's `json.loads` accepts `NaN`, the engine refuses it (T101).
- **Change.**
  - `impl TaskManifest { pub fn reconcile_all(&mut self, access: &SourceAccess)
    -> Result<usize>; pub fn check_output(&self, out: &Path, access:
    &SourceAccess) -> Result<()>; pub fn read_document(path: &Path) ->
    Result<Value> }` (`read_document` raises T101 for anything not JSON, `NaN`
    included; `load` becomes `from_json(&read_document(path)?)`).
  - The CLI and the binding call `reconcile_all`; both call `check_output` before
    writing.
  - Bindings `task_save(doc, path, base=None)` (`from_json`, then `check_output`,
    then the engine's `save`) and `task_load_document(path)`;
    `TaskManifest.save` and `TaskManifest.load` use them (keeping the raw
    document, so errors surface where they do today). This moves the
    `TaskManifest.save` write site off W21-05's allow-list, and drops
    `medh5/task.py`'s import of `temporary_beside`; if W21-13 PR 9 has already
    moved `medh5/io/nifti.py`, delete `temporary_beside` from `medh5/_files.py`
    and its allow-list entry here.
- **Tests.** With the source reached by its name, through `..`, through a
  symbolic link and through a hard link, both `medh5 task reconcile --out` and
  `TaskManifest.save` refuse, and the source is byte-identical; the reconciled
  documents from the CLI and Python are equal;
  `test_S3_1_both_frontends_save_one_byte_form` (a non-ASCII description and
  namespace: byte-identical files); a manifest containing `NaN` is T101 from both
  `load`s.
- **Changelog.** Behaviour changes, area `task-cache`: Python writes the engine's
  bytes, refuses `NaN`, and refuses to save over one of its sources.

### 11.7 A task from a cohort; repinning; clocks and days; caches through their task's pins (PR 4)

- **State at 2.0.2.** None of it exists: no `from_cohort`, `task new`, `cache
  ls`/`info`, `repin`, `Clock.utc`/`shifted_utc` or `day_bounds`; cache entries
  resolve through their own URI (`companion/cache.rs`); the CLI has no
  `--task-base` and always preflights from the manifest's directory. Only
  `relative` clocks exist (`medh5/clinical.py`, `clinical/model.rs`);
  `docs/examples/bench_preflight.py` computes day bounds by hand;
  `docs/examples/clinical_cache.py` and `docs/guides/clinical-training.md` re-pin
  by hand. The cohort side provides `Split`, claims under `/meta.cohort.splits`
  with `manifest_sha256` (`write_claims`), `Manifest.sha256()` over sample,
  subject and group ids, `Entry.content_id`, `PARTITIONS`.
- **Change.**
  1. **From a cohort.** Python `TaskManifest.from_cohort(cohort: Manifest, split:
     Split | str, *, task_id: str, version: str, identity_namespace: str, base:
     os.PathLike[str] | None = None, fold: int | None = None,
     training_partition: str = "train", policy=None, slots=(), target=None,
     description=None) -> TaskManifest`. Rust `TaskManifest::from_cohort(cohort:
     &dataset::Manifest, partitions: Partitions<'_>, task: &CohortTask<'_>) ->
     Result<TaskManifest>`, with `#[non_exhaustive] enum Partitions<'a> { Split
     { split: &'a Split, fold: Option<u32> }, Claims { set_id: &'a str } }` and a
     `#[non_exhaustive] struct CohortTask<'a>` (`task_id`, `version`,
     `identity_namespace`, `base`, `training_partition`). One subject per cohort
     `subject_id`, its entries as sources; each entry placed by the split, else
     by its claim for `set_id` (whose `manifest_sha256` must equal
     `cohort.sha256()`); a k-fold split makes fold `n` `val` and the rest `train`
     (omitting `fold` is refused); sources pinned by opening them; URIs relative
     to `base`, absolute when `None`; ids as 11.4; `split = (set_id, [training
     partition first, …])`; no rows. Refused: an entry unplaced or
     `unassigned`, a subject in two partitions, a claim against another
     manifest, an entry whose recorded `content_id` is not the file's. Entry paths
     resolve as W21-08 PR 1 makes them.
  2. **`medh5 task new OUT --cohort C.json (--split S.json [--fold N] | --claims
     SET_ID) [--definition TASK.json] [--id ID] [--version V] [--namespace NS]
     [--description TEXT] [--training-partition P] [--rows ROWS.csv]
     [--min-follow-up-us N] [--json]`.** `--definition` supplies `task`, `policy`,
     `slots`, `target`; `--rows` is a CSV `row_id,subject_id,cutoff_us`; `OUT`'s
     directory is the base; negatives without `--min-follow-up-us` are refused
     from its first release.
  3. **Cache listing**, rendering what `FeatureCache` holds: `medh5 cache info
     CACHE [--json]` (level, encoder, preprocessing, output, task fingerprint,
     selection, `fitted_on`, counts of entries and distinct sources); `medh5 cache
     ls CACHE [--source CONTENT_ID] [--row ROW_ID] [--json]` (each entry's id,
     source pins, event and document or row and cutoff, digest).
  4. **Repin.** Python `TaskManifest.repin(base=None, *, sources: Iterable[str] |
     None = None, dry_run: bool = False, confine=None) -> RepinReport`
     (`RepinReport(changed: tuple[Repinned, ...], unchanged: tuple[str, ...],
     reconciled: bool, rows: tuple[str, ...])`, `Repinned(source_id, before,
     after)`). Rust `TaskManifest::repin(&mut self, access: &SourceAccess, only:
     Option<&BTreeSet<String>>) -> Result<RepinReport>`, on `SourceRef::state(&self,
     &Sample, deep) -> Result<SourceState>` (`#[non_exhaustive] enum SourceState
     { Pinned, Moved, Corrupt(Vec<Finding>) }`). It pins only a source whose stored
     root recomputes and whose digests hold, refusing T301 (cannot open), T302
     (corrupt), T303 (another subject), T304 (another clock), T306 (this engine
     cannot certify it); it recomputes `reconciled` and lists the rows whose
     fingerprints change (their patient-cache entries go stale, T403, as they
     should). CLI `medh5 task repin MANIFEST [--base DIR] [--source ID]...
     [--dry-run] [-o OUT] [--json]` plus 11.5's flags. Finish 11.1's T302 sentence.
  5. **Clocks and days.** Rust `Clock::utc(id)`, `Clock::shifted_utc(id)`,
     `Bounds::day_containing(t: i64) -> Bounds` (`div_euclid`). Python
     `Clock.utc(clock_id)`, `Clock.shifted_utc(clock_id)` (docstring: without a
     `/meta` de-identification it is E802), `utc_us(moment: datetime) -> int`
     (aware datetimes only), `day_bounds(day: date, tz: str | tzinfo | None) ->
     Bounds` (`None`: the zone is unknown, so the day's possible instants from
     UTC+14 to UTC−12, 50 hours, as 1.1 §5.2 asks; IANA names need `tzdata` on
     Windows), `clock_day(t_us: int) -> Bounds`.
  6. **Cache entries through the task's pins.** D19's §7.1 text. Rust
     `#[non_exhaustive] struct CacheCheck<'a> { entries: SourceAccess, task:
     Option<(&'a TaskManifest, &'a SourceAccess)>, admitted: Option<&'a
     Admitted> }` and `validate_cache_in(path: &Path, check: &CacheCheck<'_>) ->
     Result<CacheReport>`.
  7. **`medh5 cache validate CACHE --task M [--task-base DIR] [--base DIR]`**:
     `--task-base` locates the task's sources (preflight and pins); `--base` the
     entry URIs the task does not pin.
  - The two documentation examples switch to the new calls; TC §10's command
    list.
- **Tests.** `TestFromCohort` (`test_S3_3_partitions_come_from_the_split`,
  `…_from_this_manifests_claims`, `…_a_claim_of_another_manifest_is_refused`,
  `…_a_subject_in_two_partitions_is_refused`, `…_fold_n_is_val`,
  `…_the_training_partition_comes_first`); `TestRepin`
  (`test_S2_repin_pins_what_is_there`, `…_refuses_another_subject`,
  `…_refuses_a_corrupt_source`, `…_names_the_rows_it_stales`,
  `…_dry_run_writes_nothing`); `test_S7_1_a_moved_cache_validates_through_its_task`;
  `tests/format/test_clinical.py` `test_S3_utc_clocks`,
  `test_S5_2_day_bounds_span_the_possible_instants`; CLI `test_task_new_from_a_split`,
  `test_task_repin`, `test_cache_ls_and_info`, `test_cache_validate_task_base`;
  Rust tests beside each function.
- **Docs.** `docs/guides/clinical-training.md` (a task from a cohort; repinning),
  `docs/reference/cli.md`, TC §7.1 and §10.
- **Changelog.** Added, area `task-cache`.

### 11.8 The follow-up note (PR 5, D19 §5)

- **State at 2.0.2.** TC §3.1 lists `min_follow_up_us` in the normal form; §5
  sets its default to `horizon_us` and warns that nearly every row without a
  positive is then censored; the engine fills it in (`companion/task.rs`), so it
  feeds every task fingerprint and patient-cache pin. `Target.min_follow_up_us:
  int | None = None` (`medh5/task.py`), with a docstring warning. T-findings have
  no severity: any finding makes `ok` false, so a warning must be a *note*.
- **Change.** D19's §5 text. Rust `companion::Note { pub location: String, pub
  message: String }` (`#[non_exhaustive]`, a new type) and a new function
  `companion::task::notes_of(doc: &serde_json::Value) -> Vec<Note>`, read from
  the raw document because the normal form has already lost the difference: a
  note at `/target/min_follow_up_us` when `target.negative` is non-empty and the
  member is absent or null. No field is added to `TaskManifest` or `Target`
  (both exhaustive). Binding `_core.task_notes(doc)`; Python's
  `TaskManifest.notes` passes the loaded document, or `to_json()` for a
  manifest built in Python (which omits a `None` follow-up).
  Python `TaskManifest.notes -> tuple[Note, ...]`; `Target.__post_init__` raises
  `DeprecationWarning` when `negative` is set and `min_follow_up_us is None`
  ("medh5 3.0 refuses a target with negatives and no min_follow_up_us: state
  it"), suppressed in `Target.from_json` (a loaded document gets the note). CLI
  `task validate` and `task preflight` print `note /target/min_follow_up_us: …`
  after the findings; `--json` gains `"notes"`; `ok` and the exit status do not
  change.
- **Tests.** `test_S5_a_target_with_negatives_states_its_follow_up` (warns; not
  when given, not when `negative` is empty, not from `from_json`);
  `test_S3_2_the_default_follow_up_is_in_the_fingerprint` (omitting equals the
  horizon explicitly, differs from horizon/2: why the default cannot move); CLI
  `test_task_validate_notes_a_defaulted_follow_up`; Rust
  `companion::task::tests::s5_a_defaulted_follow_up_is_noted`.
- **Changelog.** Deprecated, area `task-cache`.

### 11.9 Constants from the engine (PR 5)

- **State at 2.0.2.** The preflight vocabularies are defined in the binding
  (`crates/medh5-python/src/preflight.rs`: `ROW_STATUSES`, `SELECTION_STATUSES`,
  `TARGET_STATUSES`, `ROIS`) while the engine emits literals
  (`clinical/select.rs`, `companion/view.rs`). `code()` codes an unknown word as
  −1 and three Python lookups (`medh5/task.py`) index with it unguarded, so it
  reads as the table's last word: a new engine status would read as `"error"`,
  `"provable"` or `"none"`. `medh5/task.py` re-spells `ROI`, `CENSORING`,
  `STATUSES` (= `ROW_STATUSES`) and `LABELS` (= `TARGET_STATUSES`), all in
  `__all__`; `task.STATUSES` and `clinical.STATUSES` are one name for two
  vocabularies. Also literal: `medh5/cache.py` `SUFFIX`, `LEVELS`;
  `medh5/clinical.py` `ASSESSMENT_SYSTEM`, `LESION_PRESENCE`.
  `build_document_cache` re-derives document ownership in Python.
- **Change.** Engine constants live where the words are produced, and the code
  emitting a word uses its constant: `pub const ROW_STATUSES: &[&str]`,
  `TARGET_STATUSES`, `ROI_FILLS` in `companion/view.rs`; `SELECTION_STATUSES` in
  `clinical/select.rs`. The binding re-exports them, plus `TASK_ROI`,
  `TASK_CENSORING`, `CACHE_LEVELS`, `CACHE_SUFFIX`, `ASSESSMENT_SYSTEM`,
  `LESION_PRESENCE`. `code()` returns `Result<i8>` and refuses an unknown word
  naming it (a null stays −1). Each module binds `X = _core.X`. Two ROI
  vocabularies stay apart: `task.ROI = _core.TASK_ROI`, what a slot may
  request (`center`, `eligible_instances`); `task.ROIS = _core.ROIS`, what a
  fill reports, from the engine's `ROI_FILLS` (it adds `center_fallback`).
  `task.STATUSES` and `task.LABELS` move to `_DEPRECATED` (served by
  `__getattr__`, out of `__all__`, `DeprecationWarning`: "medh5 3.0 removes
  medh5.task.STATUSES: use ROW_STATUSES"). `build_document_cache` uses the
  engine's ownership rule: `companion::cache::Ownership` and `ownership` are
  private, so factor out `#[doc(hidden)] pub fn owned_documents(source:
  &SourceRef, base: Option<&Path>) -> Result<IndexMap<String, Vec<String>>>`
  (each event version and the documents its structural `describes` link owns;
  `ownership` builds on it), bound as `_core.cache_owned_documents(source,
  base=None)`. Remove these from W21-06's vocabulary allow-list.
- **Tests.** Identity (`is`) of each constant with its `_core` object; a Rust test
  that every word `view.rs` and `select.rs` emit over the companion fixtures is in
  its table; `pytest.warns(DeprecationWarning)` on `task.STATUSES`.
- **Changelog.** Deprecated, area `task-cache`.

### 11.10 Leftovers (PR 5)

1. **An entry with no sources is T401**, not T403: `CacheWriter::add` refuses it
   with T403 (`companion/cache.rs`), but the schema requires `minItems: 1`, and
   §7.1 says a writer refuses an entry that would fail its schema (T401). Test
   `test_S7_1_an_entry_without_sources_is_T401`.
2. **`CacheReport` and `CacheEntry` read through the engine.** `CacheReport.ok`,
   `stale`, `corrupt` (`medh5/cache.py`) re-compute what the engine's `to_json`
   returns: delegate (`_core.cache_report_json`, W21-06's pattern).
   `CacheEntry.from_json` accepts entries the strict engine reader refuses (T401)
   and rebuilds each source as `SourceRef(uri, content_id, sample_key)`, dropping
   the recorded `source_id` and `local_subject_id`: build from a new
   `_core.cache_entry_json(doc)` (`CacheEntry::from_json(…)?.to_json()`). Test: a
   cache entry carrying `source_id` round-trips unchanged; malformed entries are
   refused (Fixed).
3. **Defaults held to the engine's.** `SelectionPolicy()`, `Slot()`, `Target()`
   keep their Python defaults, with a test that `X().to_json()` equals the
   engine's normalisation.
4. **One open per source.** `validate_cache` opens an event-level source in
   `ownership` and again for its pin: open once. `TaskManifest::rows_in` costs rows
   × subjects (`subject()` searches linearly): one map. A Rust test counts opens.
5. **`Selection.admits_any(kind, id)`** (Python and Rust): `admits(kind, id,
   fragment=0)` answers for fragment 0 only, so a payload in fragment 1 reads "not
   admitted"; its docstring says what the default means. Test
   `test_S9_1_admits_any_looks_in_every_fragment`.
6. **`--cutoff-hours -5.5e-07`** is read as a flag despite
   `allow_negative_numbers(true)`: document the `--cutoff-hours=-5.5e-07` form in
   `docs/reference/cli.md`; test
   `test_cutoff_hours_takes_a_negative_exponent_after_equals`.
7. **The known limit** (D19): the Appendix A paragraph on what a patient-level
   entry does not pin.
- **Changelog.** Fixed and Performance, area `task-cache`.
