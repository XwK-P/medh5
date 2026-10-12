# dataset report (w202/dataset 6 commits to b972663) — 11/12 DONE, item 4 SKIPPED (design decision)
Item 4 (manifest paths relative to CWD, root unused) SKIPPED: making paths absolute breaks portable relative manifests; relative-to-manifest needs a marker field (e.g. "paths_relative_to": "manifest") because legacy paths are relative to the scan-time CWD. -> defer to 2.1.0 (format of the manifest JSON).
Tests: Rust 23+1; pytest dataset/sampling/collection 220 (1 skip); cli subset 53; typing 34; docs cohort 57; fmt/ruff/mypy/clippy clean.
## CHANGELOG drafts
Fixed:
- compute_stats: a NaN or inf voxel no longer turns the moments into NaN; such values are counted in Moments.nonfinite, and the merge does not depend on order (top issue 8).
- Cohort stats no longer merge two meanings of one class id (F16): ClassStats.key added; a class id with two keys is refused naming the file; CLI class table gets a key column.
- extract/unpack no longer replace a collection with one of its members (F10): same path, sub/../, new/../, a symlink, or a shard named for a member; every unpack destination checked before the first write (Python and medh5 unpack).
- C201 counts samples; C202 sees one subject in two folds.
- The scan follows symlinked directories (each once), accepts .MEDH5, and skips ._* files.
Behaviour changes:
- medh5 dataset stats exits 1 over unreadable files and names them on stderr (--json stays well-formed).
- Reloading stats with a null moment is refused.
- A class id with two keys is refused.
- write_claims refuses a manifest the split does not place (filtered manifests still accepted).
- An unknown annotation name is refused for sampling; so is a non-voxel annotation named for foreground draws.
- scan(on_error="warn") warns; PatchSampler validates later attribute changes; sample_stride=0 refused.
Performance:
- Stats read chunk-sized slabs (bit-identical).
- partition_of/fold_of are cached.
- Pack and check use hash lookups.
## Handoffs
- docs cli.md: dataset stats "Exits 1 if any file would not open, naming each on stderr; the class table shows each class's label-set key."; dataset index "Follows symlinked directories (each once), matches suffixes in any case, skips AppleDouble ._* files."
- docs: guides/partial-labels.md keys files by e.path, which loses collection members; use split.paths().
