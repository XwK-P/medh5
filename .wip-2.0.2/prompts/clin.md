# Workstream `clin`: clinical 1.1 profile (W2 clinical; F11 clinical export; small items)

Worktree: `SCR/wt/clin` (branch `w202/clin`). It is created from the integration branch, so it already contains the merged wave-1 work.

## Files you own
- `crates/medh5/src/clinical/**`: `select.rs`, `check.rs`, `columns.rs`, `model.rs`, `table.rs`, `augment.rs`, `schema.rs`, `mod.rs`.
- `crates/medh5-cli/src/clinical.rs`. The comp agent may make a localized edit around the cache-URI base (`~435-447`).
- `crates/medh5-python/src/{clinical,records}.rs`. The comp agent owns the cache-binding parts of `clinical.rs`; keep your hunks to the clinical selection and records functions.
- `medh5/clinical.py`.
- Tests: `tests/format/test_clinical.py`, `crates/medh5/tests/clinical.rs`.
- Localized: 1.1 spec §7.3 in `docs/spec/medh5-1.1.md`, and `docs/explanation/design-rationale.md:243`, for item 1 only.

## Items (quoted from the audit; verify each against the code first)
1. **Asserting events:** the code requires the asserter to be *admitted* (kinds, window and limit filters), but 1.1 §7.3 says *eligible* (`clinical/select.rs:627-650`). Under the default policy, an unknown-time asserter never attests, so `roi=eligible_instances` falls back to the centre.
   - **Decide which is intended,** and align the spec (§7.3), `design-rationale.md:243` and the code.
   - Keep the invariant "Inputs only through attestation" (CLAUDE.md). Strict selection never uses unknown times.
   - Explain the decision in your report.
   - If you change the 1.1 spec text, add an Appendix A entry, following the spec's convention for recorded changes.
2. `select` on a projection (a later minor) reports `certified` (`crates/medh5-python/src/clinical.rs:138-142`; `crates/medh5-cli/src/clinical.rs:187-200`). A projection must not be reported as certified.
3. **Checks** (`clinical/check.rs:597-643, 676-679`; `columns.rs:340-370`):
   - E816 checks only neighbouring versions;
   - a duplicate `supersedes` row is reported as branch+merge;
   - one cycle suppresses the chain check for unrelated records;
   - E805 blames the wrong columns when the first column is the outlier.
   - Fix each, with a test per case.
4. **F11:** `clinical export S -o S` overwrites the sample (`crates/medh5-cli/src/clinical.rs:~237`). Refuse an output that is the input, by resolved path and same-file identity. Check any other clinical subcommand with an output path for the same problem.
5. **W2, small:** NumPy integers are rejected as clinical times (`medh5/clinical.py` and/or the bindings). Accept `numpy.integer`; keep refusing floats and bools if they are refused today.
6. **W6:** the event sort clones ids per comparison. Use `sort_by` with borrowed keys or a cached key, with an identical order.
7. **W7, code docs:** `select.rs` cites task-cache §4 for the policy; it is §3.4. Check `docs/spec/task-cache-1.md` for the right section.

## Notes
- Two times, never one (CLAUDE.md): effective and available times are separate pairs of inclusive bounds; unknown stays null.
- Tests to run:
  - `wcargo clin test -p medh5 --lib clinical::`, plus `--test clinical`;
  - `wcargo clin test -p medh5-cli` filtered to clinical;
  - `wext clin`, then `wpy clin -m pytest tests/format/test_clinical.py tests/integrations/test_clinical_training.py tests/tools/test_task_cache.py -q`.
