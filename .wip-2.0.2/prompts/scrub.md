# Workstream `scrub`: de-identification (W1, F03 scrub side, F04, W5 scrub flags)

Worktree: `SCR/wt/scrub` (branch `w202/scrub`).

## Files you own
- `crates/medh5/src/curation/scrub.rs`, plus `curation/identity.rs` and `curation/provenance.rs` if an item needs them.
- `medh5/curation/scrub.py`.
- `crates/medh5-cli/src/scrub.rs`.
- `docs/guides/deidentify.md`.
- `tests/tools/test_scrub.py`.
- Localized: the scrub bindings in `crates/medh5-python/src/curation_tools.rs`. The curation agent edits the agreement parts of the same file, so keep your hunks to the scrub functions.

## Not yours (other agents do them; don't duplicate)
- Resolving symlinks in the atomic replace: the sec agent, `h5/file.rs`.
- The writer side of F03, re-linking `/meta` aliases after commit, and `remap_frame_uids` setting `compact`: the writer agent. Once that lands, scrub's second full-file `repack` becomes droppable. **Keep the repack in this PR** and mention it in your report.
- Legacy migration writing absolute paths into ids: the legacy agent.

## Items (quoted from the audit; verify each against the code first)
1. Store the scrub key tables already normalised, and add a test that `entry == normalise(entry)` for every entry (`curation/scrub.rs:62-63, 149-152, 166-168` vs `normalise` at `350-352`).
   - Today dead underscore keys (`nhs_number`, `birth_date`, `date_of_birth`, `*_uid`) never match.
2. Route top-level `extra`/`acquisition` keys (`PatientID`, `AccessionNumber`) through `entry()`/`key()`, as `identity()` does (`scrub.rs:895-904`). `deidentify.md:52` already promises this.
3. Refuse `date_shift_days == 0` in the engine, the CLI and Python, and treat a recorded 0 as "not shifted" (`scrub.rs:880`).
   - Today 0 is recorded as a shift and silences every later date scan.
   - Where are the old records with 0? Treat them as unshifted on read; don't rewrite them.
4. Shift any value that is *wholly* a date, regardless of key, e.g. `RadiopharmaceuticalStartDateTime`. Today that survives a strict apply with `ok=True` (`scrub.rs:136-153, 789-792`).
   - Recognise UIDs embedded in text (`text()` at `765-812`; reuse `has_uid_token`).
5. **F03, scrub side:** a hard-link alias of `/meta` (e.g. `x_acme/meta`) keeps the MRN after `scrub --pseudonymise-ids`. Apply returns ok, and a fresh strict scan is clean.
   - In the scan, report any non-canonical object that *is* `/meta` (same object, by object identity or address), or that decodes to a document with an `identity`, as an actionable identity finding.
   - The writer agent separately makes amend re-link the aliases.
   - Reproductions: `SCR/reports/indep-scrub.md` and `SCR/agents/indep/scrub/`.
6. **F04: name and type coverage.** Strict scrub never sees five kinds of identifier carrier: later hard-link names, soft-link names, the attributes and names of committed datatypes, nested compound string fields, and compound fields holding string arrays. All scan clean before and after apply (`scrub.rs:1231-1245, 1345-1356, 1385-1391`; `h5/ops.rs:204-242`).
   - Scan every link name, including later hard links and soft links, with a walk that expands each group once so cycles terminate. `ops::visit` deduplicates objects, which is right for digests but wrong for names.
   - Scan the attributes and names of committed datatypes.
   - Make string-type detection recursive through compound, array and vlen types, with a depth cap that reports `unreadable` instead of passing.
   - `deidentify.md:41-44` promises "every object name, every HDF5 attribute". Make that true, or qualify it precisely.
   - If you need a new walk primitive in `h5/ops.rs` (owned by sec), write it inside `scrub.rs` instead, using the public `h5` helpers, or make a minimal localized addition and report it.
7. Report timepoint dates added *after* a recorded shift (`scrub.rs:1001-1003`). Today they are invisible to every later scan.
8. Report non-string datasets outside the imaging groups as "not examined", and list them in `not_checked` (`scrub.rs:1344-1360, 431-434`).
9. After `--pseudonymise-ids`, sweep every string for the replaced values (`scrub.rs:1847-1882`).
10. Route strict staff and organisation pseudonyms through `pseudonym()`, so they are salted and recorded in `uid_map`. Scan the `deidentification` record and agents' `organization` (`scrub.rs:1112-1122, 1737-1740, 924-925, 1124`).
11. Refuse an explicit `date_shift_days` that differs from the shift already recorded (`scrub.rs:1677-1686`).
12. Python `scrub_paths` (`medh5/curation/scrub.py:316-332`; no test today):
    - continue per file like the CLI, so the `uid_map`s of files already rewritten are never lost;
    - take explicit keyword-only arguments, so `apply=True` (a typo for `apply_changes`) cannot silently become a scan;
    - add tests.
13. **W5:** `scrub --date-shift-days/--salt/--by` without `--apply` are silently ignored; refuse them (`crates/medh5-cli/src/scrub.rs`). Check how the Python console script reaches the same parser; it is the same Rust code through `_core.cli_main`.
14. `docs/guides/deidentify.md:82-94`: the suggested publication gate passes clinical samples and open identity findings, because they are non-actionable. Block on them, or say so plainly. Keep the page's python blocks runnable; they are executed by `tests/project/test_docs_python.py`.

## Notes
- **Privacy fixes must not weaken the scan.** Every finding that was reported before must still be reported.
- The CLI flags in docs are checked against the parser by `tests/project/test_docs_examples.py`; run it if you change flags or the page.
- Tests to run:
  - `wcargo scrub test -p medh5 --lib curation::scrub`;
  - `wext scrub`, then `wpy scrub -m pytest tests/tools/test_scrub.py tests/project/test_docs_examples.py -q`;
  - the scrub part of `tests/project/test_docs_python.py`: run `-k deidentify`, or the whole module if it is quick.
