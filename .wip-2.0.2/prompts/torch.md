# Workstream `torch`: PyTorch and MONAI loaders (top issue 10, F12, F13, F20, F21; W2 torch items, W6, W9)

Worktree: `SCR/wt/torch` (branch `w202/torch`). This workstream is **Python only**. Run `SCR/bin/wext-prebuilt torch` once and never build Rust. torch 2.14.1 (CPU) and MONAI 1.6.1 are installed in the venv.

## Files you own
- `medh5/torch/**`: `datasets.py`, `clinical.py`, `handles.py`, `collate.py`, `samplers.py`, `_compat.py`, `__init__.py`.
- `medh5/monai.py`.
- Tests: `tests/integrations/{test_loaders,test_monai,test_clinical_training}.py`.
- `docs/reference/torch.md`.
- Localized: the `PairedPatchDataset` example in `docs/guides/longitudinal.md:163-176` only. The docs agent edits other parts of that page.

## Items (quoted from the audit; verify each against the code first)
1. `PairedPatchDataset(label=)` serves one pair's change label as the target of every pair. Attach the label only when `ann.timepoints == (first, second)`, and refuse non-classification annotations (`datasets.py:674-678`).
2. `PairedPatchDataset(annotation=)` fails on every pair not starting at that visit; the longitudinal guide's own example hits this. Use the annotation only at its own visit, and auto-select elsewhere (`datasets.py:693-694, 739-741`). Fix `longitudinal.md:163-176` to match; its python blocks are executed by the docs test.
3. **F20:** refuse non-voxel annotations by name. Today they crash with `AttributeError` in *every* label format: `onehot` fails on `.dense` and `labelmap` on `.labelmap`.
   - That includes `medh5.monai.to_dict`, where the `isinstance` check comes after `labelmap()` (`datasets.py:304-306, 324, 349`; `monai.py:347`).
   - Add the named refusal right after the annotation lookup (`datasets.py:288`), and before `labelmap()` in `to_dict`.
   - Document the voxel-only contract in `torch.md`.
4. **F13:** clinical optional slots. A mixed present/empty optional slot cannot be batched, even with fixed patches, when the slot is multichannel (`torch/clinical.py:547-578`).
   - Take the placeholder's channel extent from the task, not from the row. A modality string does not fix a channel count, and "the first observed row" depends on read order.
   - Derive the extent once in `ClinicalTaskDataset.__init__`, from the preflight fills of *all* rows of the task, reading grid metadata only.
   - Refuse inconsistent extents, naming the slot and the images. Use 1 when no row fills the slot.
   - Only shapes cross rows, never values.
   - For `patch=None`, the collate error should say "give the slot a `patch` or use batch_size=1", not "sample fixed-size patches".
   - Reproductions: `SCR/reports/indep-torch.md`, `SCR/agents/indep/torch/`.
5. **F12:** a clinical task slot takes the ignore regions of every voxel annotation on the grid, including annotations that supply and examine none of the requested classes. An unrelated organ annotation with a whole-grid ignore region removed all 8 lesion positives from the loss mask.
   - **Fix:** in `_slot`, skip the ignore region of any annotation that neither labels nor examined a requested class (`torch/clinical.py:590-607`).
   - With several requested classes, the union over contributors is still conservative and can discard independent supervision. Document that in `docs/reference/torch.md:298`.
   - Per-class ignore is 2.1.0.
   - Test it as a relationship: adding an unrelated annotation cannot erase requested supervision.
   - The comp agent may also filter `label_annotations` by class in preflight. Your fix must not depend on that.
6. **F21:** handle-cache leases. A warm lease keeps serving a replaced file's old, pinned bytes, while a cold one raises T302, so in one DataLoader epoch worker 0 served the old snapshot and worker 1 refused the same row.
   - Record `(st_dev, st_ino, st_size, st_mtime_ns)` when `_acquire` opens a key; stat before and after `_open`, and reopen on a difference (`handles.py:65-75`).
   - In `lease`, when the cached handle matches the pin, stat the path again. If its identity changed, take the existing stale path, so T302 is raised, or a repack with the same `content_id` is served. That costs one `stat` per lease (`handles.py:190-207`).
   - Document in `docs/reference/torch.md`:
     - a lease in progress finishes on the bytes it opened, and every new lease re-checks the source;
     - the non-clinical datasets lease without a pin, so after an amend one epoch can mix versions of a file.
   - Report the task-cache-1 §6 wording for the comp agent.
7. Smaller items:
   - `HandleCache.clear()` in the owning process leaks every handle (`handles.py:388-394`);
   - `_compat` turns real torch import errors into "not installed";
   - `timepoint=` is ignored when `images=` is given;
   - MONAI meta lacks `filename_or_obj`;
   - `collate_clinical` raises a bare `ValueError`. Use the package's error type, as the surrounding code does.
8. **W6:** the torch `instances` format decodes every object mask per patch. Decode only the objects that intersect the patch, or decode once, with identical output.
9. **W9:** the fork test for `HandleCache.get` passes even if the inherited handles were closed. Make it fail if the child closes rather than abandons the parent's handles; see CLAUDE.md, "HDF5 handles must not cross fork".

## Notes
- **Behaviour must stay identical** for valid uses, except the fixes.
- Tests to run:
  - `wpy torch -m pytest tests/integrations/test_loaders.py tests/integrations/test_monai.py tests/integrations/test_clinical_training.py -q`;
  - `tests/project/test_docs_python.py -k "longitudinal or torch"`.
  - The DataLoader multi-worker tests run here.
- Lint: `wsh torch 'ruff check . && ruff format --check . && mypy medh5'`.
