# Workstream `valid`: validator, integrity, repair and conformance (W3, F01, F09, top issues 6 and 13, parts of W2/W6/W8/W9)

Worktree: `SCR/wt/valid` (branch `w202/valid`).

## Files you own
- `crates/medh5/src/validate/**`: `mod.rs` and `rules/*`.
- `crates/medh5/src/integrity/**`: `digest.rs`, `verify.rs`, `repair.rs`.
- `crates/medh5/src/conformance/**`: `build.rs`, `suite.rs`, `mod.rs`, `clinical.rs`, `companion.rs`.
- `crates/medh5/src/codes.rs`.
- `crates/medh5/data/codes.json`, for **wording only**: no new codes.
- `crates/medh5/tests/conformance.rs`.
- `crates/medh5-python/src/{integrity,validate,conformance}.rs` and `medh5/{validate,integrity,conformance}.py`.
- `docs/spec/conformance.md`.
- Tests: `tests/format/{test_validate,test_integrity,test_conformance}.py`, `tests/tools/test_repair.py`.
- The stated corpus counts wherever the docs state them. `TestStatedCounts` in `tests/format/test_conformance.py` finds every stated size and breakdown, so update every page it checks; those are localized edits in pages you don't own.

## Items (quoted from the audit; verify each against the code first)
1. **F01** (P1, high): integrity and strict validation report ok for a file in which no digest parses.
   - An early return at `validate/rules/integrity.rs:60-62` drops the E703 (malformed digest), E702 (`content_id`) and W905 (stale index) diagnostics it has already computed.
   - `integrity/verify.rs:345-353` puts both a missing and an unparseable digest into `result.malformed` only.
   - So `medh5 validate --level strict` exits 0 on a file whose pixels changed, while `verify` fails it.
   - **Fix:** delete the early return and keep W901 gated as at `:63`.
   - Add a corpus case `E703-every-digest-malformed`. The existing E703 case also carries a valid digest, so it cannot catch this.
   - Update the stated counts that `TestStatedCounts` checks.
   - Add a relationship test: validation (integrity and strict) and `verify` agree on every digest case — missing, malformed, all malformed, mismatching, and `content_id`.
2. **W3 / F09, node kinds:**
   - E004 for a non-dataset `meta`; E008 for a non-group `grids`, `images` or `annotations` (`rules/container.rs:68-75,155-158`; `rules/images.rs:22`).
   - A dangling image member is E001 (`images.rs:28` skips it today).
   - A non-numeric image or pyramid level is E001. §4.2 says "any numeric", but no E2xx code covers dtype.
   - Make `Context::children()` report members it cannot resolve instead of dropping them (`rules/mod.rs:158-175`), so dangling annotations and transforms stop being invisible to every rule.
   - Today a dangling `/images` or `/images/<id>` passes all four levels, including strict, through F01's early return. So do string and compound images, which `read()` then refuses.
   - `fix --rewrite-digests` restamps such a file, which then passes strict with zero codes. After your fix, strict must flag it, and `fix --rewrite-digests` should refuse to restamp a structurally invalid file, if that is a small change in `repair.rs`. Decide and report.
3. **W3:** per-dataset error handling in the integrity pass (`integrity/verify.rs:355-367`). Today one corrupt chunk aborts the pass and hides every other E701/E702 and the dataset's name. Report the unreadable dataset by path and keep checking.
4. **W3:** per-object error handling in `check_images`, `annotations`, `transforms` and `probmaps`, as `rules/mod.rs:3-7` promises. Use `is_group` where grid and transform ids are resolved, and report non-group members in `check_geometry`.
5. **W3:** emit E412 for the missing §6.2 MUST attributes `task`, `closure` and `class_ids` (`rules/annotations.rs:93-127`). A missing `closure` silently defaults to `explicit` today.
6. **W3:** read `/meta` and `clinical/meta` bytes strictly as UTF-8 (E004/E801). Report a non-string `meta` as E004, not E001 (`rules/container.rs:173`; `rules/clinical.rs:254`).
7. **W3:** skip cross-reference checks when `/meta` didn't parse, which avoids E409/E413 cascades. W911 is suppressed by any external-frame transform; fix that. Use `version::parse`, and stop after E002.
8. **W3:** `check_clinical_records` clones every clinical table; use `take()` (`rules/clinical.rs:368`).
9. **Top issue 6**, validator side:
   - Range-check a stored `ignore_id`: 1..65535, and not a class id. A `0` makes all background "ignored" (`rules/annotations.rs:249,520`).
   - Check `bit_class_ids`: unique, equal as a set to `class_ids`, shape `(C,)`.
   - Check `layer_class_ids`: rows equal planes (`rules/annotations.rs:483-495`).
   - Unchecked, these give false *verified negatives*.
   - Reuse existing codes; read `codes.json` for the right ones.
   - The annot agent adds the same checks at open time in `annotations/header.rs` and `layout.rs`. You do the validator side only.
10. **W2, validator side:**
    - Refuse non-finite box coordinates in `bad_boxes` (`rules/geometric.rs:179`). Use `!(lo <= hi)` plus `is_finite`.
    - Do the same for affine matrix entries (`rules/transforms.rs:156-166`), as E406/E504 or whatever the code table says for those clauses.
    - The encoder side belongs to the annot and geom agents.
11. **W2, Python API:** `diagnose`/`fix` ignore `malformed` digests: `verify` fails while `fix` says "nothing to fix" (`integrity/repair.rs:19-46, 93-109`). It is a sibling of F01, not the same bug: `Diagnosis` has no `malformed` field. Fix both. Adding the field to the result is additive and allowed; update `_core.pyi` and the Python wrapper.
12. **W2:** `fix_paths` stops at the first failure and loses the results of files already rewritten (`integrity/repair.rs:229-231`). Return per-path outcomes, as the CLI does, or continue and report every failure, with the smallest API impact. Decide and report.
13. **W2:** `fix --rebuild-index` rebuilds with the default `max_coords`, occupancy and seed instead of the values stored on the index entry (`repair.rs:182`).
14. **Top issue 13 / W8:** the conformance scorer ignores severity, so a validator that reports every E-code as a warning scores 157/157 (`conformance/suite.rs:205-212`; `conformance/mod.rs:166-190`).
    - Make E-codes count only as errors, and W-codes as warnings, as the corpus expects.
    - In `docs/spec/conformance.md`:
      - say that severity is scored;
      - say that 1.1 reclassifies some 1.0-declared files (E009/E803 vs a pure 1.0 validator's E007);
      - cite §2.2, not §2.1, for collections.
15. **W9:** promote the corrupt-container cases into the conformance corpus, so both command lines are held to them: dangling `/images`, wrong-kind nodes, all-malformed digests. Then update every stated count.
16. **W6:** digest slabs are not chunk-aligned, so each chunk is decoded 2–4 times (`integrity/digest.rs:189-208`). Align the slabs to the chunk extent. The digest bytes must not change: test that the digests are identical before and after.
17. **W2, small:** `rules_for("bogus")` returns every rule. Refuse an unknown level, in the engine (`validate/rules/mod.rs:264`) and in Python (`medh5/validate.py:275`).
18. **W8:** E702's summary in `codes.json` should include the uncovered-digest case. This is a wording fix; the docs page is generated from it.

## Notes
- The corpus is a shipped artifact that third-party implementations run, so be careful with case names and expected codes. Read how `build.rs` names and declares cases, and keep its style.
- Both command lines must pass the corpus. Check with `wsh valid 'medh5 conformance run SCR/tmp/valid/corpus'` (the Python console script calls the engine) and `wcargo valid run -p medh5-cli -- conformance run SCR/tmp/valid/corpus-native`. Read `tests/format/test_conformance.py` and the docs for how the corpus is built.
- Tests to run:
  - `wcargo valid test -p medh5 --lib validate:: integrity:: conformance::`, plus `--test conformance`;
  - `wext valid`, then `wpy valid -m pytest tests/format/test_validate.py tests/format/test_integrity.py tests/format/test_conformance.py tests/tools/test_repair.py -q`.
- F01 and F09 reproductions: `SCR/reports/indep-integrity.md` and `SCR/agents/indep/integrity/`.
