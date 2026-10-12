# Workstream `curation`: tracking, agreement and splits (F15, F16, F17, F18; parts of W2/W6/W7)

Worktree: `SCR/wt/curation` (branch `w202/curation`).

## Files you own
- `crates/medh5/src/curation/{tracking,agreement,splits,quality,timeline,mod}.rs`.
- `crates/medh5-python/src/{tracking,curation}.rs`.
- The agreement and box-IoU parts of `crates/medh5-python/src/curation_tools.rs`. The scrub agent edits the scrub functions in the same file.
- `medh5/curation/{__init__,agreement,splits}.py`.
- `docs/reference/curation.md`.
- Tests: `tests/tools/{test_agreement,test_tracking,test_splits}.py`.

## Localized edits you may make
- `crates/medh5/src/geometry/grid.rs` `comparable_with` (F18). The geom agent runs later, from your merged result.
- The `medh5 track` legend text in `crates/medh5-cli/src/curation.rs`.
- **Do not** edit `docs/guides/longitudinal.md`. The docs agent updates its lines 124-125 with the wording you report.

## Items (quoted from the audit; verify each against the code first)
1. **F15** (medium-high): tracking reports a lesion `resolved` where the follow-up's ignore region covers it, even when the ignore region covers the whole volume.
   - Coverage is class-level only: it never consults the ignore region or `valid_mask`.
   - The same root cause produces false `new` lesions when the baseline ignores the region.
   - Where: `curation/tracking.rs:255-262, 269-275, 477-499`.
   - Spec §7.4 says resolved and unexamined are told apart "only through `annotated_class_ids`", while §7.7 and §11.3 make an ignore region a way to say "not examined here".
   - **Fix:** an absent object is `resolved` only when some annotation at that visit examined its class *and* either has no ignore region, or has one that provably misses the object's last observed location. "Provably misses" means the location is mapped through the shared frame or `transform_between`, then through `box_to_slices`. Otherwise the object is `unexamined`, an existing state, so no API is added.
   - The simplest safe patch is to treat any ignore region on the covering annotation as `unexamined`. Implement the precise rule only if it is clean and well tested; otherwise ship the simple safe patch, and say which you chose.
   - Apply the same rule to `is_new` (`:269-275`).
   - Update the `medh5 track` legend: "examined at this visit, outside any ignore region, and not found".
   - Report the wording for `longitudinal.md:124-125` and for the spec §7.4 erratum ("through `annotated_class_ids` and the annotation's ignore region").
   - Field-of-view (`valid_mask`) checks and an `indeterminate` state are 2.1.0; don't do them.
   - Reproductions: `SCR/reports/indep-labels-tracking.md`, `SCR/agents/indep/labels-tracking/`.
2. **F16, agreement across files:** cross-file agreement treats label-set-local class ids as global, so "liver" in A is scored against "spleen" in B (Dice 1).
   - Before scoring, check that each compared id has the same class *key* on both sides. This is the rule C102 uses (`dataset/check.rs:270-276`); equal label-set digests short-circuit the check.
   - Resolve string keys on each side, and refuse when the ids differ.
   - Make a mismatch an error, not a `skipped` entry (`agreement.rs:265-342`, `:490`).
   - Keep numeric comparison when neither side has a label set.
   - `medh5 agree` compares within one sample and cannot reach this; Python and Rust can.
   - Reproductions: `SCR/reports/indep-agreement.md`, `SCR/agents/indep/agreement/`.
3. **F17** (medium-high): object agreement matches by `instance_id` whenever any id is shared, so two raters who each number from 1 get F1 = 1 and `mean_iou` = 0 for disjoint lesions.
   - **Fix:** use `instance_id` matching only when `one_sample(a, b)` (`agreement.rs:494-501`). Don't widen it to an equal `sample_id`: raters' copies of one case share it and still allocate ids independently.
   - Today ids {1,2} vs {1,3} on identical boxes give F1 = 0.5.
   - Report `threshold: null`, or apply the threshold, for id-matched pairs (`:539-566`).
   - Update `docs/reference/curation.md:116`: id matching applies within one sample only. Also fix `curation.md:114`: `matched` holds indices, not ids.
4. **F18:** `box_iou` of a 2-D box against a 3-D box returns about 10, and `compare_instances` reports it as a match. A box with zero spatial axes scores 1.0.
   - Make `box_iou_f64` and the Python `agreement_box_iou` fallible: refuse unequal or zero row counts, `!(lo <= hi)` and non-finite bounds, and never clamp (`agreement.rs:66-84`; `crates/medh5-python/src/curation_tools.rs:270-283`).
   - `compare_instances` raises E414 when the two sides have different spatial dimensions.
   - `Grid::comparable_with` requires equal S (`geometry/grid.rs:424-426`, also exposed at `crates/medh5-python/src/geometry.rs:339`).
5. The `splits` audit ignores `fold`, so a subject in two k-fold folds is not reported as a leak (`curation/splits.rs:340-364`).
6. **W6:** `compare_voxel` decodes per class (`agreement.rs:362-364`). Decode once.
7. **W6:** `anatomy_units` is quadratic: 2.5 s at 40k groups. It appears in `curation/splits.rs`, `dataset/check.rs`, the Python bindings and `medh5/curation/splits.py`. Fix the quadratic step where it lives (probably `splits.rs`). Results must be identical.
8. **W2, small:** `Tracking[...]` raises something other than `KeyError` for a missing key. Make it raise `KeyError` (`crates/medh5-python/src/tracking.rs`).

## Notes
- Tests to run:
  - `wcargo curation test -p medh5 --lib curation:: geometry::grid`;
  - `wext curation`, then `wpy curation -m pytest tests/tools/test_agreement.py tests/tools/test_tracking.py tests/tools/test_splits.py tests/format/test_geometry.py -q`;
  - if you touch the CLI legend: `wcargo curation test -p medh5-cli` filtered to track.
