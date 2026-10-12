# Workstream `legacy`: 0.x migration (top issue 5, W2 legacy, W1 path privacy, F11 native migrate)

Worktree: `SCR/wt/legacy` (branch `w202/legacy`). It is created from the integration branch, so it already contains the merged wave-1 work.

## Files you own
- `crates/medh5/src/convert/**`: `legacy.rs`, `grouping.rs`, `report.rs`, `mod.rs`.
- `crates/medh5-cli/src/convert.rs`.
- `crates/medh5-python/src/{convert,io}.rs`.
- `medh5/io/{legacy,grouping}.py`.
- `docs/guides/migrate-0x.md`.
- Tests: `tests/integrations/test_migrate.py`.

## Items (quoted from the audit; verify each against the code first)
1. **Top issue 5:** every review record the 0.x API wrote (`annotator`/`timestamp`) fails migration with E005, and `pending`/`flagged` become `reviewed` (`convert/legacy.rs:850-885`).
   - Read 0.x review records as 0.x wrote them (`annotator`, `timestamp`).
   - Map `pending`/`flagged` explicitly, and report the mapping as a guess in the migration report.
   - Never write an activity with no agent.
   - Find out what the 0.x API wrote: the appendix's legacy report has the evidence (grep "annotator" in `SCR/medh5-audit-appendix-area-reports.md`).
2. **Top issue 5:** the sample-level `label`/`label_name` classification is dropped unless it matches a mask or box name, contrary to Appendix B. Migrate it per Appendix B (`legacy.rs:252-296, 796-826`; spec `docs/spec/medh5-1.0.md` Appendix B).
3. `parse_date` panics on non-ASCII input, reachable from `group_by_subject` (`convert/grouping.rs:108-121`).
4. Study mode with `--subject-key` gives every study the same `sample_id` (`legacy.rs:503`). Separately, `output_name`'s `starts_with("study")` misfires on real ids like `Study-7` (`grouping.rs:325-333`).
5. Smaller items:
   - read the 0.x ≤0.5 root `shape`;
   - keep a non-object `extra`;
   - don't leave decision notes for groups that failed.
6. **W1, privacy:** legacy migration must not write the input's absolute path into `identity.subject_id`/`sample_id` (`convert/grouping.rs:226-229`; `legacy.rs:439-447, 503`).
   - Local paths often contain MRNs or names, and `content_id` then depends on the machine.
   - Use the stem or a digest, with the same rule in `migrate()` and `migrate_paths`.
   - Report it as a behaviour change.
7. **F11, native migrate:** `migrate -o .` replaces the 0.x original.
   - Check the report path first, so `migrate --report DIR` is not checked only after converting.
   - Refuse a `--report`/`--write-labels` path that is an input (`crates/medh5-cli/src/convert.rs:80-90`).
   - Refuse a target that is one of its sources (`legacy.rs:501-503`), by resolved path and same-file identity.
   - Test as a relationship: no output path may alias an input.
8. **Docs:** in `migrate-0x.md:92`, the "check it" step recommends `--level strict`, which fails on every freshly converted file (W903 and W912 are promoted). Recommend `--level integrity`, or explain. Document the review-record and classification migration, if the page covers it.

## Notes
- Tests to run:
  - `wcargo legacy test -p medh5 --lib convert::`;
  - `wcargo legacy test -p medh5-cli` filtered to migrate;
  - `wext legacy`, then `wpy legacy -m pytest tests/integrations/test_migrate.py tests/project/test_docs_examples.py -q`.
- Build 0.x fixtures the way the existing migrate tests do.
