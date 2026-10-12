# Workstream `dataset`: cohort stats, manifests, splits, sampling, collections (top issue 8, F10, F16 stats, W2/W5/W6 items)

Worktree: `SCR/wt/dataset` (branch `w202/dataset`).

## Files you own
- `crates/medh5/src/dataset/**`: `stats.rs`, `manifest.rs`, `check.rs`, `split.rs`, `mod.rs`.
- `crates/medh5/src/sampling.rs`, `crates/medh5/src/collection.rs`.
- `crates/medh5/src/json.rs`, localized: the stats `null` item.
- `crates/medh5-cli/src/dataset.rs`.
- `crates/medh5-python/src/{dataset,sampling}.rs`.
- `medh5/{collection,sampling}.py`, `medh5/dataset/**`.
- `docs/guides/cohorts.md`, and `docs/reference/cohort-checks.md`, whose table is generated from `CHECK_CODES`: edit the source, not the page.
- Tests: `tests/tools/{test_dataset,test_sampling,test_collection}.py`.

## Items (quoted from the audit; verify each against the code first)
1. **Top issue 8:** cohort stats with a NaN voxel (PET, resampled data) (`dataset/stats.rs:45-126`; `json.rs:410`).
   - min and max depend on file order and can drop a finite extreme.
   - The stats JSON writes `null` and reloads as `mean 0, std 0`, so normalisation silently becomes (0, 1).
   - **Fix:** count or exclude non-finite values, or refuse naming the file; choose, and keep the stats honest. Make min and max order-independent. Refuse `null` moments on reload.
   - Test: shuffled file order gives identical stats, and a NaN-containing image is handled as decided.
2. **F16, stats:** cohort stats bin by numeric `class_id` alone, so two label sets where id 1 means "liver" and "spleen" merge into one class that feeds `class_weights` (`dataset/stats.rs:193-219, 407-419`).
   - Record each class's key in `stats_for`; the manifest already holds `class_keys`.
   - When one id carries two keys, refuse, or report it in `failures` and make `class_weights` refuse.
   - Add `key` to `ClassStats` and its JSON. This is additive.
   - Reproductions: `SCR/reports/indep-agreement.md`, `SCR/agents/indep/agreement/`.
3. **W6:** `dataset stats` reads plane by plane, decompressing each chunk once per plane: 0.60 s vs 0.03 s. Step by the chunk extent (`dataset/stats.rs:443-453`). Results must be identical (bit-identical where the summation order allows; otherwise within float tolerance, documented).
4. Manifest paths depend on the CWD, and `root` is recorded but unused (`dataset/manifest.rs:433, 521, 549`). Resolve against `root` or the manifest's directory.
5. Scan silently skips symlinked directories and `*.MEDH5`, and picks up `._*` files (`manifest.rs:490-500`). Follow symlinked directories with cycle protection, accept the extension case-insensitively, and skip AppleDouble `._*` files.
6. C201 counts claims, not samples (`check.rs:353-358`).
   - `write_claims` should refuse a manifest the split wasn't made from (`dataset/split.rs:468`, `medh5/dataset/split.py:210`).
   - Its docstring describes a fixed bug as current (W7); fix the docstring.
7. Sampling accepts a misspelled annotation name for uniform draws (`sampling.rs:289-292`). Refuse it.
8. **F10:** `extract(shard, key, out=shard)` replaces the whole collection with one member (`collection.rs:444-461`).
   - Refuse when the output is the same file. The same happens through a relative or symlinked alias, and in `unpack` when a member's output name is the collection itself (`out/a.medh5` unpacked into `out/`), from Python and from `medh5 unpack`.
   - Compute every `unpack` destination after the case-collision loop (`:431`), and check each with `same_file` *before the first write*.
   - Reproductions: `SCR/reports/indep-dataloss.md`, `SCR/agents/indep/dataloss/`.
9. **W6:** `Split.partition_of`/`fold_of` rebuild the whole split per call, so the documented loop takes about 34 s at 3k subjects. Cache a map, and point the docs at `split.paths()` (`medh5/dataset/split.py:85-93`).
10. **W6:** check and pack use `Vec` lookups in loops (`dataset/check.rs`, the collection pack). Use maps, with identical results.
11. **W5:** `dataset stats` exits 0 with part of the cohort unreadable, and prints the failures on stdout (`crates/medh5-cli/src/dataset.rs:251-255`). Exit nonzero, and send the failures to stderr. Keep `--json` well-formed.
12. **W2, small:**
    - `PatchSampler` revalidates only in `__init__` (`medh5/sampling.py`);
    - `scan(on_error="warn")` never warns;
    - `sample_stride=0` is accepted (`crates/medh5-python/src/dataset.rs:589`); refuse it.

## Notes
- Tests to run:
  - `wcargo dataset test -p medh5 --lib dataset:: sampling collection json`;
  - `wext dataset`, then `wpy dataset -m pytest tests/tools/test_dataset.py tests/tools/test_sampling.py tests/tools/test_collection.py tests/tools/test_cli.py -q -k "dataset or stats or split or unpack or extract or pack or sampl"` (adjust the filter);
  - the CLI crate tests for `dataset`.
