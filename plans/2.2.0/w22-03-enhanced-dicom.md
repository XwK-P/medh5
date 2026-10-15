# W22-03 Enhanced multi-frame DICOM import

| | |
|---|---|
| Branch | `w22/enhanced-dicom` |
| Size | L |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `medh5/io/_frames.py` (W21-13 PR 1), `per_slice_rescale` (W21-13 PR 5) |
| Owns | `medh5/io/dicom.py` (`scan_dicom`, `_plan_series`), `medh5/io/_frames.py` |
| Decisions | none beyond the rules |

## Session prompt

> Implement workstream W22-03 (enhanced multi-frame DICOM import) of medh5 2.2.0
> in this repository. First read `plans/README.md` and `plans/2.2.0/README.md`,
> then `plans/2.2.0/w22-03-enhanced-dicom.md` in full. Check that 2.1.0's
> `medh5/io/_frames.py` and `per_slice_rescale` are on `main`. Work from the
> latest `main` on `w22/enhanced-dicom`; land the three pull requests in order.
> Geometry is never invented: frames are placed from their functional groups or
> refused, never sorted by guess. No new runtime dependency (pydicom ≥ 3.0.1 is
> the floor; highdicom builds the fixtures and is a test dependency). Reproduce at
> the dependency floors before pushing. Write tests first, run the local gate from
> `CLAUDE.md`, add fragments, open `W22-03: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 03.1 Enhanced CT: one object's frames into stacks | M |
| 2 | 03.2 Per-frame rescale and acquisition attributes | S–M |
| 3 | 03.3 MR, PET, legacy-converted objects, concatenations | M |

## Items

### 03.1 Enhanced CT: frames into stacks (PR 1)

- **State at 2.1.0 (planned).** Enhanced and legacy-converted multi-frame objects
  (`_ENHANCED_SOP_CLASSES`, `medh5/io/dicom.py`) are skipped in `scan_dicom` with
  an "unsupported" warning.
- **Change.** Split each object's frames into stacks: by `StackID` and
  `InStackPositionNumber` through `DimensionIndexSequence`, else by plane
  position; place frames with `_frames.py` (`_frame_placement` and its helpers,
  as SEG import does); decode one frame at a time with pydicom 3's
  `pixel_array(index=)`; each stack becomes a series, as a classic series would.
  A stack whose frames do not form a regular grid is refused with the reason.
- **Tests.** `test_enhanced_ct_imports_as_the_classic_series`: a
  `highdicom.legacy.LegacyConvertedEnhancedCTImage` built from the classic
  fixture gives the same grid and image digests as the classic import;
  `test_enhanced_two_stacks_are_two_series`;
  `test_an_irregular_stack_is_refused`.

### 03.2 Per-frame rescale and acquisition attributes (PR 2)

- **Change.** Read the rescale per frame (from the per-frame and shared
  functional groups), feeding 2.1.0's `per_slice_rescale` (refuse unless
  `"apply"` when frames disagree); acquisition attributes from the shared and
  per-frame groups.
- **Tests.** `test_enhanced_per_frame_rescale_follows_per_slice_rescale`.

### 03.3 MR, PET, legacy-converted objects, concatenations (PR 3)

- **Change.** Enhanced MR and PET, legacy-converted enhanced objects, and
  concatenations (one logical object split across instances, joined by
  `ConcatenationUID` and `InConcatenationNumber`).
- **Tests.** Variants of 03.1's test for MR and PET; `test_a_concatenation_is_one_object`.
- **Docs.** `docs/guides/import-dicom.md`, `docs/reference/converters.md`.
- **Changelog.** Added, area `converters`.
