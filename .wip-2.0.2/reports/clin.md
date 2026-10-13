## clin workstream: final report (branch `w202/clin`, one commit `bb5d6ea` on `a7351b6`)

### Items (all 7 DONE)
- Item 4 (F11): `clinical export --out` refuses the file being exported before reading or writing: same path, `sub/../S`, symlink, hard link, and the collection a `--key` member comes from (local `same_file`, dev/ino). `strip` and `augment -o` already refuse an existing output; now tested. (`crates/medh5-cli/src/clinical.rs`)
- Item 2: new `Clinical::select` (`clinical/table.rs`) raises `MEDH5VersionError` for a projection under both policies; CLI `select_cmd` and binding `ClinicalHandle.select` call it; free `medh5.clinical.select(...)` unchanged.
- Item 1: an asserting version need only be **eligible** (available by the cutoff, current in its record: steps 1–2); the link's event and document endpoints must still be **admitted** (`select.rs`, 1.1 §7.3, a new 1.1 Appendix A row, `design-rationale.md:243`).
- Item 3 (`check.rs`, `columns.rs`): E816 checks availability along the whole chain in one linear pass (a neighbour contradiction still reported once, with its link); a repeated `supersedes` row is one link; cycles reported once per record and only those skip the one-chain check; E805 takes the row count most columns agree on (ties to the earliest), and names the counts.
- Item 5: `medh5/clinical.py` `_bounds` accepts any integer via `operator.index` (NumPy too); floats and bools in a pair pass to the engine, which refuses them (E811); a non-integer scalar raises a clear `TypeError`.
- Item 6: `table.rs::storage_order` uses `sort_by_key` with a borrowed id; identical, stable order.
- Item 7: `select.rs` and `clinical/mod.rs` cite §3.4; Python `Selection` docstring cites 1.1 §9.

### Tests
Rust: `select::tests::s7_3_an_asserting_version_need_only_be_eligible`, `check::tests::{s7_2_availability_is_never_contradicted_along_the_chain, s7_2_a_link_given_twice_is_one_link, s7_2_a_cycle_hides_no_other_record}`, `columns::tests::s4_a_table_has_the_row_count_most_of_its_columns_agree_on`, `table::tests::s4_events_are_stored_by_known_start_then_id`.
Python (`test_clinical.py`): `test_S2_2_a_projection_is_never_selected_from[×2]`, `test_S5_2_a_numpy_integer_is_a_time`, `test_S7_2_a_chain_is_checked_whole`, `test_S7_3_an_eligible_version_attests_whatever_a_row_reads`, `TestOutputPaths::test_S10_no_output_names_the_sample_it_reads[exact|relative|symlink|hardlink]`, `TestOutputPaths::test_S10_export_refuses_the_collection_it_reads`. 9 of 10 fail against the base extension; the tenth (pure Python) fails against the old `clinical.py`.
Run: clinical:: 32, --test clinical 15, test_clinical 84, clinical_training + task_cache 128, conformance/cli -k conformance or clinical 60, docs_examples 232 passed; mkdocs --strict ok; ruff, mypy, fmt clean; clippy per crate clean.

### CHANGELOG drafts
**Fixed**
- **`medh5 clinical export S -o S` no longer destroys the sample (F11).** The export wrote its JSON over whatever `--out` named and exited 0. An output that is the file being exported is refused before anything is written: the same path, through `..` or a link, or the collection a `--key` member comes from.
- **Selection never certifies a later minor version's projection.** `Clinical.select` and `medh5 clinical select` reported `certified` for a 1.2 file read as a projection, which a task's preflight refuses (T306); both now raise `MEDH5VersionError` (1.1 §2.2).
- **An asserting event attests what was known at the cutoff, whatever a row reads (1.1 §7.3).** `asserted_by_event_id` counted only when the asserter passed the kinds, plan, unknown-time, window and limit filters, so under the default policy a segmentation of unknown time never attested its mask and `roi="eligible_instances"` fell back to the grid centre.
- **Revision chains are checked whole (E816):** a version available before an older one could have been is caught across a version of unknown availability (selection certified the newest version at a cutoff before its ancestor existed); a repeated `supersedes` link is no longer refused as a branch and a merge; a cycle no longer hides other records' problems.
- **E805 blames the column whose row count is wrong:** the row count is the one most columns agree on, not the first column's (a short `event_id` got every other column blamed and dropped).
- **`Event` accepts NumPy integer times.**

**Behaviour changes**
- Selecting from a projection raises `MEDH5VersionError` (it answered `certified`).
- More payloads are admitted: those attested by an eligible asserter that the filters leave out of the inputs.
- The writer's commit refuses a transitive availability contradiction, and accepts a duplicated `supersedes` row.
- A float or bool in an `Event` time pair is refused (E811) instead of truncated.

**Performance**
- Writing a history no longer clones every event id on each sort comparison.

### Handoffs
- comp: `task reconcile -o` writes the manifest over whatever `-o` names, including a source sample (`clinical.rs::task_reconcile` → `TaskManifest::save`); an atomic save still replaces the sample. Refuse an `-o` that is the same file as a resolved source.
- cli: three private `same_file` helpers (`cli/curation.rs`, `cli/clinical.rs`, `collection.rs`); unify in `common.rs`.
- ci: CLAUDE.md "a selected event attests it" → "an admitted event owns it, or an eligible version asserts it".

### Spec and doc wording
- spec 1.1 §9.1 step 5: "Only payloads attested at the cutoff are admitted (§7.3): the structural targets of admitted `document` and `imaging` events, and the endpoints — with any `target_annotation_id` — of links asserted by an eligible version (steps 1–2) whose event and document endpoints are themselves admitted."
- cli.md `clinical export`: "`--out` never names the file being exported, by any path or link; that is refused before anything is written."
- cli.md `clinical select`: "A later minor version, read as a projection, is refused (1.1 §2.2)."
- python-api.md at `c.select(...)`: "MEDH5VersionError for a later minor's projection".

### Decisions
1. Item 1 follows the spec's words ("eligible" asserter §7.3, "admitted" endpoints step 5): filters choose what a row reads, not withdraw evidence known by the cutoff; eligibility depends only on known availability, so no leakage. Open: in the worked example seg0's links are self-asserted, so under `order_by=effective` that mask still never centres a crop (corpus unchanged). For 2.1: should an endpoint that is the asserter itself only need to be eligible?
2. Item 2: a refusal, not a new status (new status = new API; preflight treats `provable` as eligible).
3. Item 3: the transitive E816 finding is located at `/clinical/links`; cycles one finding per record (was per file).
