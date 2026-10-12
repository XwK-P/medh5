# Workstream `spec`: specification errata (W8, plus the spec wording for the revision-2 fixes)

Worktree: `SCR/wt/spec` (branch `w202/spec`). This workstream is **docs only**. Run `SCR/bin/wext-prebuilt spec` once if you need Python to check a claim against the implementation, and you should check every claim.

## Files you own
- `docs/spec/medh5-1.0.md` and `docs/spec/medh5-1.1.md`.
  - **Not** 1.1 §7.3, on asserting events: the clin agent decides it in a later wave.
  - **Not** `task-cache-1.md` (the comp agent) or `conformance.md` (the valid agent).
- `tests/project/test_docs_examples.py`, localized, if a `STALE_CLAIMS` entry fits.

## How errata are made in this project
- The specs are **normative**. "W8 errata" means wording that matches what the reference implementation already does. **Verify each against the Rust engine** (`crates/medh5/src/...`) before writing it. If the implementation disagrees with the audit's proposed wording, follow the implementation and report the disagreement.
- 1.0 Appendix C records the clauses corrected because implementing them showed the text was not implementable; 1.1 Appendix A records the changes to the reviewed 1.1 draft. Read both, and follow their convention for recording errata: add entries for 2.0.2. Don't renumber sections, since anchors are linked from the docs and tests.
- **Don't change the format.** An erratum clarifies; it never makes a valid file invalid. The exceptions are the revision-2 errata below, which state MUSTs the implementation is enforcing in this same release. Mark those clearly in the appendix as corrections found by the 2.0.2 audit.
- Build check: `wsh spec 'mkdocs build --strict -d SCR/tmp/spec/site'`. Also run `wpy spec -m pytest tests/project/test_docs_examples.py tests/project/test_docs_python.py -q -k spec` and `tests/format/test_conformance.py -k Stated`. Never write `---` as a dash; write `—`.

## Items (quoted from the audit)
1. **1.0 §5.1:** the float rule's literal condition leaves out zero. Make it "when `x` is zero or `1e-4 ≤ |x| < 1e16`", and add `0.0`/`-0.0` examples. Check `crates/medh5/src/json.rs` for what is actually emitted.
2. **1.0 §13**, pinning the `content_id` bytes (check `crates/medh5/src/integrity/digest.rs` and `crates/medh5/src/digest.rs`):
   - the root attribute line's `obj` is empty;
   - `object_path` is relative to the sample root, with no leading `/`;
   - the meta line hashes `/meta`'s stored bytes, and `/meta` carries no `digest`;
   - lines sort by UTF-8 bytes;
   - `blake2b` means BLAKE2b-512;
   - define the class form that `label_set.sha256` hashes: omitted null or empty members, and the order of `parents`/`codes` (check `crates/medh5/src/labels/`).
3. **Later-minor covered attributes:** 1.0 §13.2 says "never covered", 1.1 §2.2 says "may cover". Pick the rule the implementation follows (`crates/medh5/src/version.rs`, `integrity/`) and make both documents say it.
4. **1.0 §2.1** still has the pre-erratum "MAY read a higher minor". Point it to §16 and 1.1 §2.2.
5. **Revision-2 errata**, which match fixes landing in this release:
   - **§9 (F14):** "a class in `annotated_class_ids` with no entry is a negative" conflicts with "ancestors follow from the label-set DAG" under `implicit`. Add "unless, under `closure = "implicit"`, a descendant is asserted positive (§5.4)". In §5.4, state that readers MUST answer state and membership queries for ancestors under `implicit`, while stored assertions stay literal.
   - **§7.4 (F15):** "resolved is distinguishable from unexamined only through `annotated_class_ids`" contradicts §7.7 and §11.3, where an ignore region is the official way to say "not examined here". Make it "through `annotated_class_ids` and the annotation's ignore region".
   - **§4.2 (F09):** images are numeric ("any numeric"); say MUST, so a string or compound image is an error.
   - **§3.6 (F05):** say which two world axes a 2-D grid's `coord_system` refers to; `RAS_TO_LPS_2D` assumes the first two.
6. **Other errata:**
   - `annotated_class_ids` is MUST except for `mask`, matching the validator;
   - §7.1: "no ignore region" means no voxel holds `ignore_id`;
   - §2.5: integer widths;
   - §10's "never share a frame" vs §3.4: reconcile;
   - require finite box coordinates and affine entries (E406/E504), which the validator and encoders now enforce;
   - E702's summary should include the uncovered-digest case. That is `codes.json`, the valid agent's; only adjust the spec prose if it restates it.
7. **1.1:**
   - tie groups are connected components of the overlap relation;
   - the 1.1 §8 "any context ≥ 72 h" claim holds only for `closed`;
   - 1.1's own task-cache items (§3.5 patch placement, §5, §7.1, §9) live in `task-cache-1.md`, which is the comp agent's; skip them.
8. **1.0 §14.4 (top issue 1):** the engine now resolves symlinks before the atomic replace. The temporary file goes beside the real file, the rename goes over the real path, and the link is kept. State this if §14.4 describes the replace, and verify it against `crates/medh5/src/h5/file.rs` on `main`, where the fix isn't present yet. Write the clause as the intended behaviour, and say in your report that it depends on the sec agent's fix.
9. **The 1.0 §15 code table:** if prose restates codes whose meaning the fixes change (E004/E008/E001 for wrong-kind and dangling nodes, E412 for missing §6.2 attributes), make the prose agree. Never edit the generated table.

## Notes
- The audit's original analysis of these items is in `SCR/medh5-audit-appendix-area-reports.md`; grep "spec" and "§13", and "content_id" for the normative-specs report.
- **Precision beats volume.** Each erratum is a small, exact text change, with an appendix entry.
