## geom workstream: final report (branch `w202/geom`, base `a7351b6`; e13d638 items 1–8, 543ac50 items 9–10, 996fe5d docs handoff, 4d08cfb clippy + content_id test, 5b5aceb cubic cache budget)

### Items (all DONE)
1. F19 chain frames: `Transform::chain`/`ChainTransform` refuse steps whose frames don't meet (E501), message shared with `check_chain` via `frame_breaks` (`transforms/model.rs`, `medh5/transforms.py`).
2. Pyramid origin held to 1e-3 of the level's own voxel in its index space, not 1e-3 of |origin| (`geometry/multiscale.rs`).
3. F19 Jacobian ROI: `jacobian_determinant`/`folding_fraction` resolve the ROI as `read_field` (NumPy negatives, clipping, missing axes padded), read only the window, refuse a step, an empty axis, < 2 voxels, more slices than spatial axes (ValueError); `cropped_grid` public, same signature, same resolution.
4. Box helpers: `box_corners`/`apply_affine_to_box` check `(S, 2)`; `slices_to_box` refuses an open stop, a negative bound, a step ≠ 1, a stop before its start (`crates/medh5-python/src/geometry.rs`).
5. Storage: `chunk_hint` reaches stacked encodings and displacement fields (`[1]*leading + hint`; writer and `--rechunk`); `portable` recompression rewrites contiguous any small dataset under a plugin filter; `describe_filters` names `zstd:<level>`, `filter:<id>`.
6. Non-finite refused when written: box bounds (E406), affine entries (E504), pyramid factors (E105).
7. `recompress_paths` checks the profile first, tries every file; one failure raises that file's error, several restate the first with counts (the `fix_paths` pattern).
8. `dataset_kwargs` honours a custom `CodecProfile`.
9. W6: cubic coefficients made once per `Transform` handle and shared by clones, **within a 256 MiB process-wide budget** (`CUBIC_CACHE_BYTES`; a `Kept` holder returns its bytes on drop; past the budget coefficients serve one evaluation, as 2.0.1; a lost `OnceLock::set` race returns the loser's bytes); padding and tap loop no longer allocate per voxel/tap. Chunk ceiling: a start chunk past 4 MiB starts at the patch and halves until it fits; growth stops at 4 MiB. L3 = sysfs size / `shared_cpu_list` count (macOS `hw.cacheconfig` untested). content_id unchanged by a test (a 2.0.1-style 128³ re-chunked copy verifies under the same content_id).
10. W9: the dead monkeypatch half of the L3 test replaced by a test of the real probe against sysfs.
Docs handoff: `storage.md` L3 paragraph, commit = fsync + rename + directory fsync, link following (target replaced, link kept; hard links keep old content; sticky world-writable directory refusal, `MEDH5FileError`), sec's recompress wording; `performance.md` chunk paragraph rewritten.

### Tests
Python: test_transforms (`TestComposite::test_S10_5_an_in_memory_chain_is_held_to_the_composites_rule`, `TestDisplacement::test_S10_4_a_jacobian_roi_resolves_like_read_field`, `…_is_the_same_region_of_the_field`, `…_it_cannot_difference_is_refused`, `…_a_cubic_field_is_prefiltered_once_and_answers_alike`, `TestAffine::test_S10_3_a_matrix_entry_that_is_not_finite_is_E504`); test_geometry (`TestPyramidChecks::test_S4_3_the_origin_is_held_to_a_level_voxel_wherever_it_lies`, `TestBoxConvention::test_S8_1_a_box_is_made_only_from_a_bounded_block`, `…_a_bound_that_is_not_finite_is_E406`, `TestMisshapenArguments::test_S8_1_every_box_helper_refuses_what_box_to_slices_refuses`, `TestMultiscale::test_S4_3_a_factor_that_is_not_finite_and_positive_is_E105`); test_storage (`TestChunking::test_S14_1_every_array_on_a_hinted_grid_takes_its_spatial_chunk`, `…test_S13_1_a_new_chunk_shape_changes_no_content_id`, `…test_S14_1_l3_is_one_cpus_share_of_what_sysfs_reports`, `TestCodecs::test_S14_2_a_profile_of_ones_own_is_honoured`, `…_every_plugin_filter_is_named`, `TestRecompress::test_S14_2_a_portable_recompression_leaves_nothing_needing_a_plugin`, `…test_S14_4_recompress_paths_attempts_every_file`). Rust: `s8_1_a_bound_that_is_not_finite_is_e406`, `s14_1_no_computed_chunk_is_past_the_ceiling`, `s14_1_l3_is_read_per_sharing_cpu`, `transforms::apply::tests::s10_4_coefficients_past_the_budget_are_not_kept_and_answer_alike`.
Run: geometry/transforms/storage 248 passed; writer/validate/sample/geometric/readers + dataset-stats 559 passed, 1 skipped; cli recompress/bench + dataset 9; conformance 46; docs_examples 232; docs_python storage/performance 7; mkdocs --strict clean; ruff, mypy clean; cargo geometry/transforms/storage 32, transforms 14 (after budget); fmt clean; clippy medh5 and medh5-python clean; test_transforms 126 and integrations pair/transform 24 after the budget change.

### CHANGELOG drafts
**Fixed**
- **A transform chain whose steps do not meet is refused (E501).** `ChainTransform`/`Transform::chain` checked only units, so steps out of order were chained and points mapped along a path that does not exist.
- **The pyramid check finds a missing half-voxel shift wherever the grid lies.** The origin was held to 1e-3 of |origin|, so about 500 voxels from zero (whole-body CT, a µm stage) the shift passed the writer and the validator; it is now held to 1e-3 of the level's voxel.
- **`jacobian_determinant(roi=…)` and `folding_fraction(roi=…)` resolve the region as `read_field` does.** A negative stop was E109, a short ROI raised a points error, and a strided ROI was differenced as if contiguous: `::2` reported 100 % folding on a field that folds nowhere.
- **Every box helper checks the box's shape.** `box_corners` and `apply_affine_to_box` read a `(2, 3)` min/max-corner box as three made-up axes.
- **Stacked annotations and displacement fields follow the grid's `chunk_hint`** (§14.1); the writer and `recompress --rechunk` chunked them apart from the images they label.
- **`recompress --profile portable` leaves nothing that needs a plugin.** A small dataset another writer stored under Blosc2 or Zstandard was copied as it was; it is now rewritten contiguous.
- **`describe_filters` and `medh5 info` name Zstandard and unknown plugin filters** (`zstd:<level>`, `filter:<id>`).
- **`recompress_paths` tries every file;** one failure stopped the batch and the files already rewritten went unreported; the error now names each failure, as `fix_paths` does.
- **`storage.dataset_kwargs` honours a `CodecProfile` of your own;** only its name reached the engine, which refused it.

**Behaviour changes**
- **`jacobian_determinant`/`folding_fraction` refuse a strided, empty or one-voxel region, or more slices than spatial axes (ValueError);** `cropped_grid` refuses a step.
- **Non-finite values are refused where they are written:** box bounds (E406), affine entries (E504), pyramid factors (E105); the validator already reported each.
- **`slices_to_box` refuses an open stop, a negative bound, a step and a stop before its start (ValueError).** An open stop was a bare `TypeError`; the rest made a box of other voxels.
- **A pyramid level whose origin is off by more than 1e-3 of a level voxel is E105,** where a large |origin| let it pass.
- **New files may be chunked differently; no `content_id` changes.** L3 is read as one CPU's share, and no computed chunk exceeds 4 MiB (a 96³ float32 patch made 8 MiB chunks, a 160³ int16 one 32 MiB). Digests cover decompressed content, so only stored bytes change.

**Performance**
- **A cubic displacement field is prefiltered once per transform handle, within a 256 MiB process-wide budget.** It was read whole and prefiltered at every evaluation, which in a paired loader meant once per item; past the budget, coefficients serve one evaluation and are made again, as before. Results are identical.

**Documentation**
- **`storage.dataset_kwargs` changed meaning in 2.0, and now says so.** 1.x returned `h5py` keywords; 2.0 returns `{"chunks", "codec"}` or `{}`; for `h5py`, use `resolve_profile(p).codec(role).kwargs()`.
- **The storage pages state the chunk budget and ceiling, and how commit replaces a file through a link.**

### Handoffs
- Merge conflict in `docs/guides/performance.md` (lines 82–89) with the docs merge: take geom's side ("one CPU's share", not "one core's": `shared_cpu_list` counts hardware threads).
- GIL pass: geom's hunks in `crates/medh5-python/src/geometry.rs` (553–630) sit after `world_to_index` (546).
- 2.1.0: a composite re-opens its components per evaluation, so a cubic field inside a chain does not reuse the cache.
- docs (optional): add "detected per-core" and "chunks are held between 512 KiB and 4 MiB" to `STALE_PHRASES`.

### Spec wording
- §4.3: "A validator **MUST** check this to a tolerance of 1e-3: each spacing relative to the derived spacing, and the origin to 1e-3 of the level's voxel along each of its axes (its offset from the derived origin, in the level's index coordinates)."
- §14.1 recommended sizing: "≈ 80 % of one CPU's share of the L3 (its size over the CPUs sharing it)"; "start from `2^ceil(log2(patch_hint))`, or, where that passes 4 MiB, from the patch, halved until within it"; "stop when the mean ratio exceeds 1.5 or the next step passes 4 MiB".

### Decisions
- Cubic cache memory: f64, 3 components, padded 12 per side under `nearest` (~84 MB for 128³); bounded by the 256 MiB process-wide budget.
- No new codes for the ROI refusals (ValueError, argument errors).
- Shared-CPU count unreadable → whole L3, as before.
- `dataset_kwargs` keeps its 2.0 return shape; unknown plugin filters print as `filter:<id>`.
- No public Rust signature changed; additions are `pub(crate)`.
