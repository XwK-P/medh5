# torch report (w202/torch, 8 commits on 1959938) — all items DONE (Python only)
Tests: 3 area modules 156 passed; docs tests 390; ruff/mypy clean; each new test fails with the change stashed. Not run: mkdocs (text-only edits), full suite.
## CHANGELOG drafts
Fixed:
- An unrelated annotation's ignore region no longer erases a clinical slot's supervision: only annotations that label or examined one of the slot's classes contribute to ignore[slot]; the mask is still one union across classes (F12).
- PairedPatchDataset(label=) labels only the pair it spans, and annotation= is drawn from at its own visit, so pairs not starting there no longer fail.
- A missing multichannel optional slot now batches beside a present one: the placeholder takes the channel count of the slot's images across the task (F13).
- Warm and cold handle caches agree after a pinned source is replaced: each new pinned lease re-checks the file with one stat, so a replaced file is refused (T302) or reread if it holds the same version (F21).
- HandleCache.clear() and worker_init_fn no longer leak handles in the process that opened them.
- A broken torch install is reported with its cause; MONAI metadata names the source file (filename_or_obj), so SaveImage names its output after the sample.
Behaviour changes:
- Non-voxel annotations are refused by name in every label format and in to_dict (used to fail with AttributeError) (F20).
- label= naming a non-classification annotation is refused.
- images= that are not at timepoint= are refused (were read silently).
- A slot empty for some rows whose fills have different channel counts is refused at construction (T102).
- collate_clinical raises MEDH5ValidationError (still a ValueError) for a shape mismatch.
Performance:
- label_format="instances" no longer decodes every object mask per patch.
Documentation: torch.md voxel-only contract, pair label=/annotation= scope, ignore[slot] conservative union, images[slot] placeholder and patch=None advice, lease re-check, unpinned datasets mixing versions after an amend, clear(), timepoint=, filename_or_obj.
## Handoffs
- spec/comp: task-cache-1 §6 after "…is refused (T302), never read in its place": "A lease already in progress finishes on the bytes it opened; every new lease re-checks the source."
- comp (optional): filter label_annotations by class in preflight (view.rs:662-664).
## Decisions: T102 for disagreeing channel counts only when a placeholder is needed; error rows skipped when deriving extent; filename_or_obj = sample.path; stat failure falls through to the reopen error (not T302); private cross-module imports handles._open and collate._shared_keys in clinical.py.
