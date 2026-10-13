## legacy workstream: final report (branch `w202/legacy`, 4 commits on `c3b9f0e`)

### Status (all DONE)
1. 0.x review records (`convert/legacy.rs`). 2. Appendix B classification. 3. `parse_date` panic (`convert/grouping.rs`). 4. Study-mode `sample_id` and the `output_name` "study" prefix. 5. Root `shape`, non-object `extra`, notes of failed groups (one docstring line in `medh5/io/legacy.py`). 6. W1, no paths in ids. 7. F11 (`legacy.rs`, `crates/medh5-cli/src/convert.rs`). 8. Docs (`docs/guides/migrate-0x.md`, including the `--level integrity` handoff). Converters handoff: `sanitize_stem` folds to ASCII (`grouping.rs`).

### Tests
Rust: `grouping::tests::{s3_7_a_non_ascii_date_is_unparseable_not_a_panic, s3_7_only_a_fallback_is_named_for_its_source, s2_2_a_stem_is_an_ascii_identifier}`, `legacy::tests::appendix_b_review_times_are_0x_isoformat_or_a_date`.
Python (`test_migrate.py`): `test_appendix_B_a_review_the_0x_api_wrote_migrates` (4 statuses), `test_a_review_naming_no_one_is_the_import_tools`, `test_S12_1_each_study_of_a_subject_has_its_own_sample_id`, `test_W1_no_local_path_reaches_the_sample`, `TestSampleLabel` (3), `TestOlderLayouts` (2), `TestNoOutputAliasesAnInput` (8 alias cases + report inside `-o` allowed + `migrate()` onto its source); the failed-group test now requires its notes removed.
Run: `convert::` 5 passed; test_migrate + test_conversion + test_docs_examples 297 passed / 1 failed (report inside output dir wrongly refused, fixed); after the fix test_migrate + test_cli -k migrate/report 112 passed; clippy -p medh5 -p medh5-cli clean; ruff, mypy clean.

### CHANGELOG drafts
*Fixed*
- **A review record written by the 0.x API now migrates.** `annotator` and `timestamp` were never read, so every such file failed with E005 (an activity with no agent), and a `pending` or `flagged` file became `reviewed`. Both now become `submitted`, a flag is kept as an issue, and the report calls each mapping a guess; a record naming no reviewer is attributed to the import tool.
- **A 0.x sample label survives migration.** `label`/`label_name` was kept only when its name matched a mask or box; its class is now minted with the cohort's label set (Appendix B).
- **A non-ASCII date no longer crashes grouping.** `parse_date` counted characters, then sliced bytes.
- **Each study keeps its own `sample_id`, and a subject named `Study-7` keeps its name.** A sample's id is now its output file's name, unique within the run.
- **Smaller migration fixes:** the root `shape` of 0.x ≤0.5 is read; a non-object `extra` is kept as `extra.legacy.<tp>.extra`; a failed group's notes are removed with it; a non-ASCII subject key (`Müller`) no longer fails with E003.

*Behaviour changes*
- **Migration writes no local path into a sample.** A file with no subject got `subject_id = "study:/abs/path"`, and its import activity named the path. Now `sample_id` is the file's stem and `subject_id` `study:<stem>` in both `migrate()` and `migrate_paths()`, and the import activity names only the file.
- **`migrate` refuses to write over an input (F11).** A sample whose output would be an input is reported `failed` and not written (`migrate()` raises); `--report` or `--write-labels` naming an input, and a `--report` that is a directory or in a missing directory, are refused before anything is written; a report naming an output is not written.
- **`sanitize_stem` folds to ASCII** (`Müller` becomes `Muller`).

*Documentation*
- **The 0.x migration guide checks at `--level integrity`**, explains why `--level strict` fails fresh files (W903, W912), and documents how review records and sample labels migrate, how samples are named, that outputs never replace inputs, and that a reviewed sidecar must keep its keys.

### Handoffs
- conv: `medh5/io/_common.py` `sanitize_ascii_stem` docstring says `sanitize_stem` keeps non-ASCII letters; no longer true.
- conv2: `nifti.py:1230` writes input paths into the import activity's `inputs` (the same W1 leak).
- spec, Appendix B: root `shape` up to 0.5; `label` without `label_name` is its own class; review status `pending`/`flagged` becomes `submitted`; the fallback identity is `study:<stem>`.

### Decisions
1. `pending`, `flagged`, missing and unknown statuses all become `submitted`; `flagged` adds an issue (`flagged`, warning); a pending record's person is not in `reviewed_by`.
2. The raw `label` value goes only into the report note's detail (classification values must be 0–1); label classes are minted after mask and box classes, so existing ids don't move.
3. ASCII fold is a lookup table (no new dependency): Latin-1, Latin Extended-A, combining marks, fullwidth ASCII, ª º ¹ ² ³; matches Python for U+00C0–U+017F.
4. Import activity `inputs` names the file, not the path.
5. The CLI's `same_file` is a private copy (did not touch cli-owned `curation.rs`).
Not done: `build_label_set` still skips unreadable inputs silently under `--write-labels` (low severity, unassigned).
