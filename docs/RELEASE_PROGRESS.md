# Release progress log — 0.5.0

Tracks phase-by-phase progress on `docs/RELEASE_PLAN.md`. Each checkpoint
entry records what changed, what was verified, and any deviations from
the plan.

## Status at a glance

| Phase | Status | Checkpoint commit |
|---|---|---|
| 1. Release blockers (data safety + packaging) | complete | 14a8a8a |
| 2. Correctness bugs | complete | see Phase 2 checkpoint |
| 3. Stability hardening | pending | — |
| 4. Essential features | pending | — |
| 5. New tests | pending | — |
| 6. CI workflow | pending | — |
| 7. Documentation | pending | — |

## Phase 1 — Release blockers

### 1.1 Atomic write
- [x] `MEDH5File.write` writes to `.<name>.tmp-<pid>` → fsync → `os.replace`
- [x] tmp file unlinked on any exception (try/finally with `done` flag)
- [x] `test_interrupted_write_leaves_no_file` + `test_interrupted_write_preserves_existing_file`

### 1.2 Checksum verification before `update()`
- [x] `update()` calls `verify_checksum` before mutating when the attr exists
- [x] `force=False` kwarg as escape hatch for intentional repairs
- [x] `TestUpdateVerifiesChecksum` class in `tests/test_integrity.py`

### 1.3 Torch fork/spawn safety
- [x] `_HandleCache` PID-keyed (auto-resets on fork via `_ensure_owner`)
- [x] `medh5.torch.worker_init_fn` exported
- [x] `_HandleCache.clear()` abandons entries without closing (safe when fds are shared with parent)
- [x] `TestDataLoaderMultiprocessing::test_dataloader_workers[spawn]` + `test_patch_dataloader_spawn` (real DataLoader with `multiprocessing_context="spawn"`, `num_workers=2`)
- [x] `test_pid_reset_on_fork` and `test_worker_init_fn_clears_cache` unit tests
- Note: skipped `__getstate__` — module-level cache means spawned workers re-import and get a fresh cache automatically; dataset attrs (paths, sampler, transform) are already pickleable.

### 1.4 PyPI packaging
- [x] `LICENSE` file (MIT, Puyang Wang, 2026)
- [x] `pyproject.toml`: authors, project.urls, classifiers (Development Status :: 4 - Beta, Medical Science Apps., Typing :: Typed), `package-data = {medh5 = ["py.typed"]}`, `license = {file = "LICENSE"}`, tightened `h5py>=3.10`, `hdf5plugin>=4.1`, bumped to `0.5.0`, added `build`/`twine` to dev extras
- [x] `medh5/__init__.py` exports `__version__ = "0.5.0"`
- [x] Verified `python -m build` produces wheel+sdist containing `LICENSE` and `py.typed`
- [x] Verified `twine check dist/*` passes (required upgrading local `packaging>=24.2` because the wheel uses core metadata 2.4)
- Note: `medh5.egg-info/` was not actually tracked in git (audit agent was wrong); no `git rm --cached` needed.

### 1.5 Spatial metadata validation
- [x] `SampleMeta.validate` checks `direction` is `ndim × ndim`
- [x] `SampleMeta.validate` checks `axis_labels` length matches `ndim`
- [x] `read_meta` raises `MEDH5SchemaError` on malformed `direction` (was a silent `warnings.warn`)
- [x] Tests: `test_direction_dim_mismatch_raises`, `test_direction_non_square_raises`, `test_axis_labels_length_mismatch_raises`, `test_malformed_direction_on_read_raises`

### Phase 1 checkpoint results
- [x] `ruff check .` — all checks passed (fixed 2 SIM102 along the way)
- [x] `ruff format --check .` — clean after `ruff format` on the two touched files
- [x] `mypy medh5` — 0 issues across 18 source files, strict mode
- [x] `pytest tests/` — **181 passed, 1 skipped, coverage 90.44%** (fail-under 90 honored)
- [x] `python -m build && twine check dist/*` — both artifacts PASSED

---

## Checkpoints

### Phase 1 checkpoint — 2026-04-12

**Scope:** Data-safety + packaging release blockers from `docs/RELEASE_PLAN.md` §1.

**Files changed:**
- `medh5/core.py` — atomic-write pipeline; `update(force=…)`; pre-mutation checksum verify; `MEDH5FileError` pass-through clause; `contextlib`/`os` imports.
- `medh5/meta.py` — `direction`/`axis_labels` ndim validation; top-level `MEDH5SchemaError` import; malformed-direction on read now raises (was a warning).
- `medh5/torch.py` — PID-scoped `_HandleCache`; `_ensure_owner`/`clear`; module-level `worker_init_fn`; docstring rewritten to reflect the DataLoader story.
- `medh5/__init__.py` — `__version__ = "0.5.0"`.
- `pyproject.toml` — full PyPI metadata (authors, urls, classifiers, package-data, license file ref, keywords, bumped pins, added `build`+`twine` to dev extras).
- `LICENSE` — new, MIT.
- `docs/RELEASE_PLAN.md` — copied from the approved plan.
- `docs/RELEASE_PROGRESS.md` — this file.
- `tests/test_roundtrip.py` — `TestAtomicWrite` + 4 validation tests + `h5py` import.
- `tests/test_integrity.py` — `TestUpdateVerifiesChecksum`.
- `tests/test_torch.py` — `TestDataLoaderMultiprocessing`, cache pid-reset test, `worker_init_fn` test, top-level `_identity_collate` for spawn-pickleability.

**Deviations from plan:**
- Dropped `MEDH5TorchDataset.__getstate__` hook — module-level cache means spawned workers re-import `medh5.torch` and get a fresh empty cache for free. `__getstate__` would only matter for fork, where the PID-reset path already handles it.
- Did not remove `medh5.egg-info/` from git — it was never tracked to begin with.
- Left `SampleMeta.validate(strict=…)` removal for Phase 2 cleanup; no behavior bug, just dead code.

**Verification:** see "Phase 1 checkpoint results" above.

### Phase 2 checkpoint — 2026-04-13

**Scope:** Correctness bugs from `docs/RELEASE_PLAN.md` §2.

**Files changed:**
- `medh5/transforms.py` — `RandomFlip` updates `meta.spatial.direction` (negates
  column `k` for each flipped axis) via `dataclasses.replace`, and flips
  bboxes in voxel coords using the sample's actual image shape.
- `medh5/sampling.py` — `PatchSampler(include_bboxes=True)` returns
  patch-local bboxes filtered by intersection with the patch window, plus
  the matching `bbox_scores` / `bbox_labels`. Off by default.
- `medh5/core.py` — `read()` returns `seg=None` when the HDF5 `seg` group
  has zero members; `bboxes` dataset only uses Blosc2 compression when
  `len(bboxes) > 64`.
- `medh5/meta.py` — `read_meta` derives `has_seg` from the actual group
  state so `seg is None ↔ has_seg is False` always holds.
- `tests/test_transforms.py` — new RandomFlip tests for direction, bbox,
  multi-axis bbox, and no-meta compatibility.
- `tests/test_sampling.py` — new `TestIncludeBboxes` class.
- `tests/test_roundtrip.py` — `test_many_bboxes_roundtrip`,
  `test_empty_seg_group_returns_none`.

**Deviations from plan:**
- `read_meta` now cross-checks the HDF5 group state for `has_seg`, which
  wasn't in the plan but is the minimum required for consistency once
  `seg=None` is returned for an empty group.
- Did not add the `strict=…` kwarg removal from `SampleMeta.validate` —
  deferred to a future phase since it's dead code, not a bug.

---

## Phase 2 — Correctness bugs

### 2.1 `RandomFlip` geometry sync + bbox flip
- [x] `RandomFlip` negates column `k` of `meta.spatial.direction` for each flipped axis
- [x] Uses `dataclasses.replace` so the dataset's cached `SampleMeta` is not mutated
- [x] Flips bboxes symmetrically in voxel coords: `[lo, hi] → [S-1-hi, S-1-lo]`
- [x] Bbox flip uses the actual image shape in the sample (works for patches)
- [x] Tests: `test_direction_column_negated_on_flip`, `test_flip_without_meta_ok`, `test_bbox_mirrored`, `test_bbox_flip_multiple_axes`
- Note: origin shift is RAS/coord-system dependent; left for a future phase per the plan.

### 2.2 `PatchSampler` bbox support
- [x] `PatchSampler(include_bboxes=True)` returns patch-local bboxes
- [x] Boxes are translated by `-start` and filtered to those intersecting the patch
- [x] `bbox_scores` and `bbox_labels` are filtered to match the surviving boxes
- [x] Default is `include_bboxes=False` — existing users see no behavior change
- [x] Tests: `test_patch_local_bboxes_filter_and_translate`, `test_include_bboxes_default_off`, `test_include_bboxes_no_bboxes_in_file`
- Note: Boxes are *not* clipped to patch extent — downstream detection losses
  generally prefer raw (possibly-truncated) box bounds over clipped ones.

### 2.3 Empty seg group handling
- [x] `MEDH5File.read` returns `sample.seg = None` when the `seg` group is empty
- [x] `read_meta` overrides `has_seg`/`seg_names` to reflect the actual group state
  so the invariant `seg is None ↔ has_seg is False` holds
- [x] `test_empty_seg_group_returns_none` — simulates an empty seg group by
  deleting its sole dataset after write

### 2.4 Bbox compression threshold
- [x] `MEDH5File.write` only applies the Blosc2 filter to `bboxes` when the
  first-axis length exceeds 64; tiny arrays stay raw to avoid chunk overhead
- [x] `test_many_bboxes_roundtrip` exercises the compressed path with N=128

### Phase 2 checkpoint results
- [x] `ruff check .` — all checks passed
- [x] `ruff format --check .` — clean after `ruff format` on `medh5/sampling.py` and `tests/test_transforms.py`
- [x] `mypy medh5` — 0 issues across 18 source files, strict mode
- [x] `pytest tests/` — **190 passed, 1 skipped, coverage 90.56%** (fail-under 90 honored)
