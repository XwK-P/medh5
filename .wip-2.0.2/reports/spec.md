## spec workstream: final report (branch `w202/spec`, 6 commits)

Every erratum was checked against the integration tree's Rust, and the pinned bytes empirically. A script rebuilt `content_id` and `label_set.sha256` from the 1.0 text alone (h5py and hashlib) and matched the engine for sha256 and blake2b (BLAKE2b-512, 128 hex). Digesting the label set "as written" (with `laterality: null`, empty `parents`) does not match. Files edited: `docs/spec/medh5-1.0.md`, `docs/spec/medh5-1.1.md`, `tests/project/test_docs_examples.py`.

### Items
1. **Revision-2 errata and merged behaviour: DONE** (`91cc491`)
   - §5.4/§9 (F14): readers answer state and membership queries with ancestors under `implicit`. A descendant asserted positive in the same scope unit makes the ancestor positive.
   - §7.4 (F15): the ignore region is part of coverage. §4.2 (F09): images must be numeric, else E001.
   - §3.6 (F05): a 2-D grid uses the first two world axes, with its plane at third coordinate 0. §3.3 rule 4: grids must also share the number of spatial axes S.
   - §8.1/§8.3/§10.3: box bounds, obb values and affine entries must be finite (E406/E504). §8.2: boxes without `instance_ids` may be numbered by row.
   - §6.2: E412 covers a missing `task`/`closure`/`class_ids`; `annotated_class_ids` is "MUST except mask".
   - §14.4: symlinks are resolved before the replace (sec's wording).
   - §15.2 prose matches the new codes.json summaries (E004/E008/E109/E406/E504/E702), plus E001 and E412.
   - Optional §11.4 (a zero date shift means no shift) and §2.1 (amend keeps `digest_algo`).
2. **§5.1/§13 byte pinning: DONE** (`ca076bf`)
   - The float rule covers zero (`0.0`, `-0.0`). The label-set class, relation and skeleton forms are defined.
   - `blake2b` = BLAKE2b-512. Paths are sample-root relative with no leading `/`, and the root line starts `@\t`.
   - The meta line hashes the stored bytes; lines sort by UTF-8 bytes.
3. **Covered attributes: DONE** (`f3e8566`). The engine reports a later minor's `content_id` it can't recompute as W913, following 1.1. 1.0 §13.2: a later minor MAY extend the lists; 1.1 §2.2 says the same.
4. **§2.1 pointer: DONE** (`df906ca`). The version is `"MAJOR.MINOR"`, with the MUST NOT refuse / supported projection rule.
5. **The rest: DONE** (`6cb49f9`): §2.5 integer widths, §7.1 "no voxel holds `ignore_id`", §10 vs §3.4, 1.1 tie groups as connected components, and the 1.1 §9.3 "≥ 72 h" claim (holds for `closed` only). Task-cache items skipped as instructed (comp owns them).
6. **One `STALE_CLAIMS` entry** (`6339ab2`) keeps "MAY read a higher minor" out of the docs.

**Appendices:** Appendix C has 20 new rows ("Seventy-four … and twenty in the audit of 2.0.1"). Each row starts "Found by the audit of 2.0.1", plus "; enforced from 2.0.2" where a new MUST is now enforced. 1.1 Appendix A has 3 new rows.

### Checks
- `test_conformance.py -k "appendix or Stated or SpecSync"`: 12 passed after each commit.
- `test_docs_examples.py`: 232 passed; with the conformance subset, 244.
- The stale-claim tests: 18 passed.
- ruff passes on the test file.
- `mkdocs build --strict` passes.
- No `---` dashes.

### CHANGELOG drafts (Documentation)
- **The specification now defines `content_id` and `label_set.sha256` to the byte.** Zero is written `0.0`, `blake2b` is BLAKE2b-512, and paths, the root line, the `/meta` line, the sort order and the digested class form are spelled out. Every one matches what the engine always wrote, so no digest changes.
- **1.0 and 1.1 now agree on later minor versions.** A later minor may extend the covered attributes, and a reader must not refuse a file only because its minor is higher (§2.1, §13.2, 1.1 §2.2).
- **The specification states the rules 2.0.2 enforces.** These are implicit closure in readers, ignore regions in tracking coverage, numeric images, the 2-D world axes, finite boxes and affines, E412 for missing annotation attributes, and replacing through symlinks. Each is recorded in Appendix C as a correction from the 2.0.1 audit.

### Disagreements and handoffs
- **`ignore_id` width:** the writer stores `int64`; the table says `uint16`. §2.5 now says readers accept any width and the table's type bounds the value, which keeps the writer conforming.
- **obb `centers`:** the spec now requires them finite (E406). The writer refuses non-finite centers, but the validator checks only `sizes`. Handoff to valid (coordinator).
- **`/meta` digest:** "writers MUST NOT stamp one"; a digest found there still counts as an ordinary dataset line, no error.
- **§8.7:** no change needed; E408 already states the rule.
- **Open (from writer):** the index `source_digest` is always sha256; §13.3 doesn't say whether it should follow the file's algorithm. Left open.
