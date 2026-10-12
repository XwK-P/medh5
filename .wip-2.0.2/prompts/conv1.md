# Workstream `conv1`: DICOM, DICOM SEG and RTSTRUCT converters (top issue 4, F02, F06, F07, W2 converters)

Worktree: `SCR/wt/conv1` (branch `w202/conv1`). This workstream is **Python only**. Run `SCR/bin/wext-prebuilt conv1` once and never build Rust.

## Files you own
- `medh5/io/{dicom,dicom_seg,rtstruct,report}.py`.
- Tests: `tests/integrations/{test_dicom,test_dicom_seg,test_rtstruct}.py`.
- `docs/guides/import-dicom.md`.
- The DICOM, DICOM SEG and RTSTRUCT sections of `docs/reference/converters.md`. The conv2 agent edits its NIfTI and nnU-Net sections, so keep your hunks in your sections.

## Items (quoted from the audit; verify each against the code first)
1. **Top issue 4:** DICOM SEG import keeps only the last of several segments that share a label. One "Lesion" segment per lesion is common, and half the voxels are lost with 0 warnings (`medh5/io/dicom_seg.py:690, 775-794`).
   - **Fix:** OR (binary) or max (fractional) the segments that map to one class, or refuse; record a decision note in the report, as the converters do.
   - Test with two same-label segments in different places: both survive.
2. `to_dicom_seg` (`dicom_seg.py:924,935`): export `annotated_class_ids` by default, and refuse an unexamined class. Today unexamined classes become empty segments, i.e. verified negatives on re-import.
3. **F02** (medium-high): RTSTRUCT import ignores `ContourGeometricType` (`medh5/io/rtstruct.py:65-74, 189-203, 293-306, 396-404, 618`).
   - What goes wrong today:
     - crossing `CLOSEDPLANAR_XOR` contours are unioned (63 voxels instead of 54); nested ones are already right;
     - `OPEN_PLANAR` lines are filled as areas;
     - `OPEN_NONPLANAR` applicators are flattened onto one plane, and the class is marked examined with 0 voxels, a fabricated negative;
     - POINT ROIs vanish without a note;
     - export writes `CLOSED_PLANAR` for everything.
   - **Fix:**
     - Read `ContourGeometricType` per contour, and treat a missing one as `CLOSED_PLANAR`, with a recorded guess.
     - Skip ROIs made only of POINT, `OPEN_PLANAR` or `OPEN_NONPLANAR` contours, with one warning per ROI and type, so no class is marked examined with 0 voxels.
     - Refuse an ROI that mixes open and closed contours.
     - Report the drop of contours with fewer than 3 points at `:73`.
     - For `CLOSEDPLANAR_XOR`, refuse a mix with other types, as the standard does. Accept contours that are nested or disjoint, which import correctly today. Refuse crossing ones, naming the ROI and plane. Use a segment-intersection test, since vertex-in-polygon misses a "plus" shape.
     - In `to_rtstruct`, warn when two contours of one class cross on a plane: medh5 means their union, while XOR and even-odd readers cancel the overlap.
   - Also report or refuse contours that are off-slice, out of the grid or non-axial (`rtstruct.py:294-299, 383-386`).
   - Representing open curves faithfully needs an open/closed flag in §8.6, which is 2.1.0; don't add it.
   - Reproductions: `SCR/reports/indep-rtstruct.md`, `SCR/agents/indep/rtstruct/`.
4. **F06:** RTSTRUCT applies the first file-level frame to every ROI.
   - **Fix:** read each ROI's `ReferencedFrameOfReferenceUID` (`rtstruct.py:54-57`). Use the one frame the contoured ROIs share, and fall back to the file-level list only for ROIs that state none.
   - Refuse with E414, before opening the sample, when ROIs span frames, listing which ROI is in which frame.
   - Warn when an ROI's frame is missing from the file-level list.
   - Match grids on the ROI frame (`_match_grid`, `:432`), and keep the salted comparison (`_require_same_frame`).
   - Today the first file-level frame is applied to every ROI (`:83-88, 133-137, 230-241`), so the outcome depends on list order.
5. **F07:** DICOM slice stacking (`medh5/io/dicom.py:52, 366-388`).
   - Estimate the spacing as `(last − first)/(n − 1)` or by a least-squares fit, not by the median of gaps.
   - Bound the through-plane residual of every slice against `origin + k·spacing·normal`, with `tol_z = max(0.1·spacing, 0.01 mm)`, and keep the half-pixel in-plane bound.
   - Keep a per-gap test only to name a missing slice (a gap above 1.5× the spacing).
   - Report `max_through_plane_error_mm` in the `slice_spacing` decision.
   - This refuses accumulated drift (the 401-slice case, 4.5 mm out) and accepts 2-decimal rounding, which today's 1% per-gap rule refuses.
   - **Refuse; never resample:** converters never invent geometry.
   - Align with the SEG reader's absolute per-plane test, and document one policy in `import-dicom.md`: state the slice-position tolerance as a physical bound.
   - Reproductions: `SCR/reports/indep-nifti-dicom.md`, `SCR/agents/indep/nifti-dicom/`.
6. `from_dicom` (`dicom.py:185-213, 757-760`):
   - skip non-image SOP classes and multi-frame objects;
   - record per-series refusals as warnings, so one RTDOSE or localizer series doesn't abort the whole tree;
   - wrap pixel-decoder errors.
7. `read_dicom_seg`: don't invent a 1 mm through-plane step for a one-plane SEG (`dicom_seg.py:587-588`). Geometry is never invented: refuse, or use what the SEG states.
8. `to_rtstruct` (`rtstruct.py:568-631`):
   - write `SpecificCharacterSet`, the label-set colours and display names;
   - a non-ASCII PatientID aborts `from_dicom` (`sanitize_stem`); fix it.
9. **Docs:**
   - `import-dicom.md`: the tolerance (item 5), and the "check it" step, which today recommends `--level strict`. That level fails on every freshly converted file, because W903 and W912 are promoted. Recommend `--level integrity`, or explain (`import-dicom.md:127`).
   - `converters.md`: which RTSTRUCT contour types import, and that crossing contours mean union on export (F02).

## Notes
- **Synthetic DICOM:** build test inputs with pydicom/highdicom the way the existing tests do; reuse their builders.
- Every refusal names the ROI, series, file or plane.
- Tests to run: `wpy conv1 -m pytest tests/integrations/test_dicom.py tests/integrations/test_dicom_seg.py tests/integrations/test_rtstruct.py tests/integrations/test_conversion.py -q`, plus `tests/project/test_docs_examples.py` and `test_docs_python.py` `-k "dicom or converters"` if you change those pages.
- Lint: `wsh conv1 'ruff check . && ruff format --check . && mypy medh5'`.
