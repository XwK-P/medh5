# conv1 report (w202/conv1: e712d1d 535e562 62d15a6 38e5d76) — all 9 items DONE
Valid input output bytes unchanged (4-series tree + ordinary RTSTRUCT give same content_id as 2.0.1).
Tests: test_dicom 46, test_dicom_seg 46, test_rtstruct 59, test_conversion 14 passed; 13/8/24 new items fail on base. docs tests ok, mkdocs strict ok, ruff/mypy clean.
## CHANGELOG drafts
Fixed:
- A DICOM SEG with several segments of one label imports all of them. Each segment replaced the one before, so every lesion but the last was lost with no warning. Segments of one class are now combined, and two segments with one label no longer fail E302 when classes are minted.
- from_dicom imports a subject whose PatientID is not ASCII. The sample id kept Unicode letters and failed E003; it is folded to ASCII (Muller_Hans), and the subject id keeps the original.
- to_rtstruct writes non-ASCII text under a stated character set, and keeps the label set's names and colours. Text went out as Latin-1 under the ASCII default; every ROI was named for its key and coloured red.
- A pixel decoder's failure is a refusal naming the series and file (was a raw traceback).
Behaviour changes:
- DICOM slices are held to a per-slice bound: a tenth of the spacing, at least 0.01 mm. The 1% per-gap rule passed 4.5 mm of drift and refused positions rounded to two decimals. SPACING_TOLERANCE is now 0.1, a fraction of the spacing; read_dicom_seg uses the same bound.
- One unreadable series no longer aborts from_dicom: left out with a `series` warning, the rest import. RT Dose skipped with a note, multi-frame objects with a warning; refused only when nothing can be read.
- to_dicom_seg writes only the classes the annotation examined (unexamined became empty segments = verified negatives on re-read).
- read_dicom_seg no longer assumes a 1 mm step for a one-plane SEG (SpacingBetweenSlices, else SliceThickness as a decision, else refused).
- RTSTRUCT import reads ContourGeometricType: POINT and OPEN_* ROIs left out with a warning; crossing XOR contours, ROIs mixing types, contours between/across slices or outside the grid are refused.
- RTSTRUCT import uses each ROI's own frame of reference; ROIs in several frames refused (E414). Used to depend on the order of the file's frame list.
- to_rtstruct warns when contours of one class cross on a plane (medh5 = union; XOR/even-odd readers cancel overlap). ROINames are now display names.
Performance:
- read_series checks every header before decoding, then decodes one slice at a time (held the series ~3x before the first check).
Documentation:
- The DICOM guide states the slice tolerance and checks imports with --level integrity (strict fails every fresh import on W903, W912 for minted classes).
## Handoffs
- docs: refusals.md + export.md wording (below).
- conv2: from_nifti default sample_id=stem has the same non-ASCII E003 failure; to_nifti(class_key=unexamined) coverage (already conv2 item 4).
- legacy (Rust): sanitize_stem in convert/grouping.rs keeps Unicode letters (Python folds via _safe).
- 2.1.0: valid check contour_plane vs vertices; cli to-dicom-seg --class; spec §8.6 open/closed flag + union/nesting-parity rule.
## Doc wording
- refusals.md DICOM: replace "If the gaps disagree by more than a tolerance…" with "Every slice must lie within a tenth of the slice spacing, and at least 0.01 mm, of where the regular stack puts it; drift that every gap hides is refused, rounding is not." Add: "A refused series is left out with a warning; the rest of the tree imports."
- refusals.md SEG: "Segments of one class are combined (union or maximum). Export writes only examined classes; naming an unexamined one is refused. A one-plane SEG stating no step is refused."
- refusals.md RTSTRUCT: "Only closed contours import; POINT and OPEN_* ROIs are left out with a warning. Crossing XOR contours, mixed types, and contours off the grid's slices are refused. ROIs in several frames are refused (E414)."
- export.md: examined-only SEG rule; RTSTRUCT names, colours and crossing warning.
## Decisions
1. Spacing: median gap first, fallback least squares from first origin (keeps output byte-identical).
2. SPACING_TOLERANCE value+meaning changed; LATTICE_TOLERANCE kept importable but unused.
3. RT Dose = info note (exit 0); any per-series warning makes CLI exit 1.
4. Off-slice/cross-slice/out-of-grid contours refused (no code).
5. Touching contours count as crossing; unknown contour types skipped like OPEN; POINT mixed with closed refused.
6. ROI with no frame in a multi-frame file refused E414; file listing none -> assumed other ROIs' frame (recorded guess).
7. Display names only if all distinct and <=64 chars; ISO_IR 192 only when non-ASCII.
8. Extra: warning when a class rasterises to no voxels; read_rtstruct meta keys types/dropped/roi_frames/frames; frame_uid = shared frame; SEG geometry spacing_source; empty timepoint kept when all series refused.
9. Open: two RTSTRUCT ROIs with the same name still fail E302 when minting (unchanged).
