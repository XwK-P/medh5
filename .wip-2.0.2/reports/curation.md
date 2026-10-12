# curation report (w202/curation 48329c2) — all 8 items DONE
F15: simple safe patch (annotation whose ignore region holds >=1 voxel or can't be read adds nothing to coverage -> absent = unexamined; is_new same). Legend lives in medh5-cli/src/inspect.rs (edited there).
F16: ids both sides use checked key against key; equal label-set digests skip; clash raises C102 naming both files; names resolved per side; no label set -> numeric.
F17: id matching only when one_sample(a,b); id matches report threshold: null in JSON (Python attr still float). curation.md: matched holds row indices; matched_by per sample.
F18: box_iou/box_iou_f64 return Result; refuse unequal/zero axis counts (ValueError), non-finite or lo>hi (E406); never clamp; compare_instances E414 for different dims, E406 naming bad object. Grid::comparable_with requires equal S (localized edit geometry/grid.rs).
Folds: leaks() keys on (partition, fold), prints "train (fold 0)".
compare_voxel: all classes read together via dense() in whole-chunk slabs (<=64M voxels per side). anatomy_units HashMap: 40k units 2.43s -> 0.28s. Tracking[key] KeyError.
Tests: 21 Rust; 136 targeted pytest; 16 new fail on main; cli/sample 423; docs 390; typing 34; clippy/fmt/ruff/mypy clean. Perf 40 classes 1M voxels: layers 1.15->0.38s, bitmask 5.12->0.38s, labelmap 1.00->0.94s.
## CHANGELOG drafts
Fixed:
- Tracking no longer resolves a lesion an ignore region may hide. An annotation that ignores part of its grid no longer counts its classes as examined, so the object is unexamined, and a baseline that ignored the place no longer makes a lesion new.
- Cross-file agreement refuses class ids with two meanings ("liver" against "spleen" scored a Dice of 1; now C102).
- Instance ids are matched only within one sample (two raters numbering from 1 scored F1 = 1 for disjoint lesions).
- Box IoU refuses boxes of two spaces (2-D vs 3-D scored about 10 and matched).
- medh5 splits reports a subject in two folds.
- Tracking[key] raises KeyError.
Behaviour changes: tracking coverage/states for annotations with ignore regions; C102 refusal; IoU matching across files; threshold: null for id matches; box_iou refusals (ValueError/E406) and E414; Grid.comparable_with false across S; fold-qualified leak partition names; medh5 track legend.
Performance: compare_voxel reads layers/bitmask once per slab, not once per class; anatomy_units linear.
## Handoffs
- dataset: C202 in dataset/check.rs splits() ignores folds -> same (partition, fold) fix.
- annot: labelmap dense() single-pass; once has_ignore_region agrees with ignore_region, tracking's ignores_part can drop the extra read.
- docs: longitudinal.md:124-125 resolved = "the class was examined at this visit, outside any ignore region, and the object was not found"; unexamined = "nobody looked for that class at this visit, or an ignore region (§7.7) may hide it"; :137-140 add "and the annotation's ignore region"; matching wording python-api.md:202 and cli.md track.
- spec: §7.4 "…only through annotated_class_ids and the annotation's ignore region (§7.7, §11.3): absence inside, or possibly inside, an ignore region is unexamined."; §3.3 rule 4 add "and the same number of spatial axes S"; E406 extend to non-finite bounds.
## Decisions: simple F15 patch over-reports unexamined for slab-limited reviews, never wrong; reuse C102/E414/E406; Rust box_iou signature changed (semver note); no label set -> by number; a name one label set lacks falls through to the id check.
