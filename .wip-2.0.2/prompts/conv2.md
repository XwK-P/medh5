# Workstream `conv2`: NIfTI, nnU-Net and the converter CLI (F05, F08, F11, W2 converters)

Worktree: `SCR/wt/conv2` (branch `w202/conv2`). This workstream is **Python only**. Run `SCR/bin/wext-prebuilt conv2` once and never build Rust.

## Files you own
- `medh5/io/{nifti,nnunetv2,_common,__init__}.py`.
- `medh5/cli.py`: the Python converter commands.
- Tests: `tests/integrations/{test_nifti,test_nnunet,test_conversion}.py`, plus localized additions to `tests/tools/test_cli.py` for F11.
- `docs/guides/{import-nifti,export}.md`.
- The NIfTI and nnU-Net sections of `docs/reference/converters.md`. The conv1 agent edits the DICOM, SEG and RTSTRUCT sections.

## Items (quoted from the audit; verify each against the code first)
1. **F05:** a 2-D NIfTI plane loses its z offset (z=50 exports as z=0), and two planes at different z, or an image and a mask from different slices, are accepted onto one grid (`medh5/io/nifti.py:848-854, 101-102`).
   - **Fix:** in `_reduce_plane`, refuse a plane whose discarded world coordinate is nonzero (> 1e-3 mm after unit scaling). Tell the user to save the image as `(x, y, 1)`, to import it as a one-slice volume, which already keeps z.
   - Change `tests/integrations/test_nifti.py:594`, which asserts the lossy round trip, to use z=0.
   - Reword the "tilted in 3-D" refusal of sagittal planes to "not parallel to the x-y world plane".
   - In `import-nifti.md:25-35`, say that a 2-D plane must lie at z = 0, and otherwise must be saved as `(x, y, 1)`.
   - Report the spec §3.6 wording for the spec agent: a 2-D grid's `coord_system` refers to the first two world axes.
   - Reproductions: `SCR/reports/indep-nifti-dicom.md`, `SCR/agents/indep/nifti-dicom/`.
2. 5-D NIfTI `(x,y,z,1,n)`: never take the timing from the squeezed axis (`nifti.py:536-538, 829-830, 899-924`).
3. `import_seg_nifti`:
   - replay the `read_nifti` notes;
   - accept `assume_geometry`;
   - fix the provenance tool name;
   - `read_nifti` on Analyze or MGH files should give a clean refusal.
4. `to_nifti(class_key=)`: export `annotated_class_ids` by default, and refuse an unexamined class (`nifti.py:1446`). Today unexamined classes become empty masks, i.e. verified negatives on re-import.
5. **F08:** re-exporting to nnU-Net into the same directory leaves the old files.
   - What survives today: removed cases; dropped channels; a case moved to `imagesTs`, which stays in `imagesTr` (train/test leakage); both file endings.
   - `dataset.json` then disagrees with the inventory, and `from_nnunetv2` brings the removed case back.
   - **Fix:** before the first `mkdir`, list the NIfTI files already under `imagesTr`/`labelsTr`/`imagesTs`, and refuse, writing nothing, when any is not in the new plan (`medh5/io/nnunetv2.py:572-666`).
   - An identical re-export stays allowed, and unrelated files are untouched.
   - A `--replace` staging mode is 2.1.0; don't add it.
   - `docs/guides/export.md:111-112` promises "no half-written dataset behind". Say what an existing destination does.
   - Test it as a relationship: a second export has exactly the second plan's inventory, or is refused with nothing written.
   - Reproductions: `SCR/reports/indep-dataloss.md`, `SCR/agents/indep/dataloss/`.
6. **F11:** `--report` or `--out` naming an input or an output overwrites it, and the command exits 0. This applies to every converter. `from-dicom-seg`/`from-rtstruct --report case.medh5` destroys the only copy of the sample just amended.
   - Before any handler runs (`medh5/cli.py:129-131`):
     - gather each command's input and output files;
     - refuse a `--report` or `--out` that is the same file as any of them, by resolved path and by `samefile`;
     - refuse a report inside an input root or the nnU-Net dataset directory.
   - In `_finish` (`:184-193`), refuse a report that aliases `report.outputs`, and write the report through a temp file and `os.replace`.
   - The native `migrate` and `clinical export` sides are other agents' (legacy, clin).
   - Test it as a relationship: no command's report or output path may alias an input, for every converter subcommand.
7. **Docs:**
   - `import-nifti.md:106`: the "check it" step recommends `--level strict`, which fails on every freshly converted file (W903 and W912 are promoted). Recommend `--level integrity`, or explain.
   - `export.md:118-122` promises "lossless round trips to DICOM", but there is no DICOM image exporter. Fix the claim, and list what has no exporter: images to DICOM, boxes, keypoints, meshes, transforms, clinical.

## Notes
- **Converters never invent geometry;** refusals name the file and axis.
- Tests to run:
  - `wpy conv2 -m pytest tests/integrations/test_nifti.py tests/integrations/test_nnunet.py tests/integrations/test_conversion.py -q`;
  - the converter tests in `tests/tools/test_cli.py` (`-k "report or out or convert or nifti or nnunet"`);
  - `tests/project/test_docs_examples.py`;
  - `tests/project/test_docs_python.py -k "nifti or export"`.
- Lint: `wsh conv2 'ruff check . && ruff format --check . && mypy medh5'`.
