## polish workstream: final report (branch `w202/polish`, 7 commits on `38dc93b`, `7f902bd..e425ad6`)

### Items (all 19 DONE)
1. `task reconcile -o` refuses an output that is one of the task's sources (same path, `..`, symlink, hard link, or the collection a member comes from), via `same_file` on `SourceRef::resolve(base)`, before anything is read or written (`crates/medh5-cli/src/clinical.rs`).
2. `nifti.py`: the import activity's `inputs` is `nifti:<file name>`; dicom, dicom_seg, nnunetv2, rtstruct record UIDs, internal paths or `nnunetv2:<dir>/<case>` (no path leak).
3. Engine `compare_instances` (and `compare`) refuse a threshold that is NaN, ≤0 or >1; body only, signature unchanged (`agreement.rs`).
4. Bench bindings: negative `repeats`/`patch` reach the engine as 0 (same `MEDH5ValidationError`); negative `warmup` refused (`medh5-python/src/conformance.rs`).
5. `Manifest::save` and `write_sidecar` use `write_atomic`.
6. `sanitize_stem` / `sanitize_ascii_stem` docstrings fixed.
7. `build_label_set` reports `unreadable` for each input it can't read (private `mint_label_set(…, unreadable)`, so `migrate_paths` doesn't report twice).
8. GIL released in `ClinicalHandle.records`, `CacheWriterHandle.add`/`commit`.
9. GIL released in `storage.rs` (`build_index`, `occupancy`), `transforms.rs` (`inside_extent`, the three samplers, `encode_displacement`, `to_world_vectors`, `jacobian_determinant`, `folding_fraction`), `geometry.rs` (inside `points_through`: `Grid.index_to_world`, `world_to_index`, `world_into` and the free functions; single-box helpers left).
10. `export.md`: when an ROI is named for its key (`roi_names`); conv2 had the rest.
11. `deidentify.md`, `curation.md`: link and shared-directory wording, matching `python-api.md`.
12. `python-api.md`: `c.select` raises `MEDH5VersionError` on a projection.
13. `medh5/cli.py` docstring and `crates/medh5-cli/README.md` mention `python` on Windows (README, install.md, cli.md already did).
14. Stale-claim pattern now `medh5 validate \S+\.medh5 --level strict`; geom's two phrases added to `STALE_PHRASES` (case-insensitive).
15. Companion renderer, its regexes and docstring mention removed from `hooks/mkdocs_hooks.py`.
16–19. 1.0 Appendix B (four mappings); §2.1 version sentence; §4.3 tolerance sentence; §14.1 sizing guidance (a recommendation: no Appendix C row); three new Appendix C rows (§2.1, §4.3, Appendix B), count now 77 (twenty-three in the audit of 2.0.1); 1.1 §9.1 step 5 reworded, clin's Appendix A row cites "§7.3, §9.1".
Unrequested one-sentence docs: `cli.md` (reconcile `--out`), `curation.md` (IoU range), `migrate-0x.md` (unreadable inputs).

### Tests
New: `TestCommandLine::test_S3_3_reconcile_never_writes_over_a_source[exact|relative|symlink|hardlink]`, `…_over_a_collection_it_reads`, `test_S3_1_a_manifest_that_cannot_be_saved_is_named`; `TestAgreement::test_an_iou_threshold_lies_in_0_to_1[×5]`; bench refusals for negative counts and warmup; `TestManifest::test_a_save_replaces_the_manifest_whole`; `TestMigrate::test_a_sidecar_is_replaced_whole`, `test_a_label_set_names_the_files_it_could_not_read`; `TestErrorsNameTheFile::test_a_file_replaced_whole_is_named`; `TestNiftiInputs::test_W1_no_local_path_reaches_the_sample`. All 18 new Rust-backed cases fail against the older extension; the NIfTI test fails with the fix reverted.
Run: 255 passed, 1 skipped; 615 passed, 1 skipped; geometry/transforms/storage/sampling/stubtest 347; docs 418; conformance 46; mkdocs --strict ok; fmt, ruff, mypy clean; clippy -p medh5 -p medh5-cli -p medh5-python clean. Not run: Rust tests, corpus CLI run, GIL timing for items 8–9.

### CHANGELOG drafts
**Fixed**
- **`medh5 task reconcile -o` can no longer replace a source sample.** It saved the manifest over whatever `--out` named and exited 0; an output that is a source, by any path or link, or the collection holding one, is refused before anything is written.
- **The Python agreement API refuses an IoU threshold outside (0, 1], as `medh5 agree` does.** NaN matched no object and 0 or less matched every pair; both were scored as agreement.
- **A negative `repeats`, `patch` or `warmup` in `medh5.bench` is a `MEDH5ValidationError`**, not PyO3's `OverflowError`.
- **A cohort manifest and a label-set sidecar are replaced whole**, and a failed atomic write names its file; `task reconcile -o` and `migrate --write-labels` said only "No such file or directory (os error 2)".

**Behaviour changes**
- **`migrate --write-labels` reports a file it cannot read (`unreadable`, exit 1)** instead of silently leaving its classes out of the label set.
- **A NIfTI import records `nifti:<file name>`, not the input's path**, so `content_id` no longer depends on where the file was.
- **`Manifest.save` and `write_sidecar` replace a read-only file**, keeping its mode, as amend does.

**Performance**
- **Index building, occupancy, field sampling, displacement encoding, Jacobians, folding fractions, point mapping, clinical `records()` and cache writing run with the GIL released.**

**Documentation**
- **The spec records three more 2.0.1 corrections** (§2.1 version reading, §4.3 pyramid origin, Appendix B), updates §14.1's sizing guidance, and aligns 1.1 §9.1 step 5 with §7.3.
- **The guides cover link handling for `scrub --apply` and amend, ROI naming, `select` on a projection, and `python` on Windows.**

### Handoffs
- sec (`h5/file.rs`): I/O errors from `write_atomic` name the target; private `precreate` and `fsync_path` now return `io::Result`.
- ci: `CLAUDE.md:237–240`, `CONTRIBUTING.md:165–168` still say the hook renders the task-and-cache code table; it no longer does.

### Decisions
- NIfTI input written `nifti:<name>` (follows migrate's `medh5-0.x:`).
- Threshold check in the engine function (signature unchanged), so Rust callers get it too.
- The four Appendix B changes are one Appendix C row.
- The stale pattern matches any sample or glob; publishing paragraphs stay exempt.
