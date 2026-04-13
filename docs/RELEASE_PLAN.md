# Plan: Take `medh5` from WIP → release-ready on PyPI

## Context

`medh5` is a pre-release Python package (currently v0.4.0) that stores a single
medical-imaging sample (multi-modality images + seg + bboxes + label +
spatial metadata) in a `.medh5` file backed by HDF5 + Blosc2. CLAUDE.md
states "WIP … backward compatibility is not needed" — we're allowed to make
breaking changes.

A deep audit (3 parallel Explore agents + manual verification of key
findings) surfaced release blockers in four categories:

1. **Data-corruption hazards** — non-atomic writes, a checksum-update path
   that can silently bake in corrupted data, a torch DataLoader integration
   that is documented as forked-worker-safe but actually crashes under
   `spawn` (macOS/Windows default) and deadlocks under `fork` with h5py.
2. **Medical-imaging correctness** — `RandomFlip` silently desynchronizes
   image data from the `direction` matrix; `SampleMeta.validate` never
   checks that `direction` / `axis_labels` match `ndim`; malformed
   `direction` attrs are dropped with a warning on read.
3. **PyPI packaging gaps** — no `LICENSE` file on disk despite
   `pyproject.toml` declaring MIT; `py.typed` never declared as package
   data; no authors, no project URLs, no "Typing :: Typed" / Medical
   classifiers; `medh5.egg-info/` checked in; README's `Development Status`
   missing.
4. **Test coverage gaps** on the *exact* paths users hit first — parallel
   `compute_stats(workers=N)`, torch `DataLoader(num_workers>0)`,
   spatial-metadata dimension mismatches, flip-under-asymmetric-geometry.

The intended outcome: after this plan is executed, `medh5` can be
`python -m build && twine upload`-ed with confidence that a medical-imaging
researcher using it on macOS with `DataLoader` workers won't (a) get silently
wrong training data, (b) get a crashed file from a Ctrl-C during write, or
(c) land on a PyPI page with "Author Unknown".

---

## Critical files to be modified

| File | Purpose of change |
|---|---|
| `pyproject.toml` | Authors, project URLs, classifiers, `package-data = {medh5 = ["py.typed"]}`, bump deps, `readme` content-type |
| `LICENSE` (new) | MIT license text |
| `medh5/core.py` | Atomic write via tempfile+rename; verify checksum before `update()`; default `recompute_checksum=False`; remove unused `strict` kwarg on `validate` |
| `medh5/meta.py` | Add `direction` and `axis_labels` dimension checks to `validate()`; promote malformed-direction on read from warning → `MEDH5SchemaError`; populate `image_names` from file when attr missing |
| `medh5/torch.py` | Make `_HandleCache` PID-keyed; add `worker_init_fn` helper; close handles in `__getstate__`/`__setstate__` so dataset survives `spawn`; update docstring |
| `medh5/transforms.py` | `RandomFlip` updates `sample["meta"].spatial.direction` and origin to keep geometry consistent; bbox flip support |
| `medh5/sampling.py` | Return bboxes clipped/translated to patch coords; optional `include_bboxes` flag |
| `medh5/stats.py` | Kahan-compensated sum for float64 precision on huge uint16 CT volumes; NaN/Inf guard |
| `medh5/cli.py` | Exit code 2 for no-command / bad-args; 1 for runtime errors; dispatch table for subcommands (remove the manual `if cmd ==` chain) |
| `medh5/__init__.py` | Bump `__version__` to 0.5.0; confirm public surface matches README |
| `.github/workflows/ci.yml` | Add macOS job (catches fork/spawn bugs); add `python -m build` smoke job; add `twine check dist/*` step |
| `.gitignore` + `git rm --cached medh5.egg-info` | Remove tracked artifact |
| `tests/` | New tests listed in §8 below |
| `README.md` + `CHANGELOG.md` | Document the DataLoader-worker story; add upgrade notes; bump version |

---

## Phase 1 — Release blockers (data safety + packaging)

### 1.1 Atomic write
`MEDH5File.write()` at `medh5/core.py:568-605` opens the destination
directly with `h5py.File(str(path), "w")`. An interrupted write (Ctrl-C,
OOM, crash) leaves a truncated file at the exact path users expect to find
their data. Fix: write to `path.with_suffix(".medh5.tmp-<pid>")`, `fsync`,
then `os.replace()` into place. `os.replace` is atomic on POSIX and Windows.
Wrap in `try/except` so the temp file is unlinked on failure.

### 1.2 Checksum verification before `update()`
`core.py:854` defaults `recompute_checksum=True`. If a file's data has been
corrupted externally between the original `write(checksum=True)` and a later
`update_meta(label=…)`, the update silently re-hashes the corrupted data and
stores the new digest — destroying the ability to detect corruption.

Fix: **before** mutating, if `_CHECKSUM_ATTR` is present, call
`verify_checksum(f)`; raise `MEDH5FileError("checksum mismatch — refusing to
update")` if it fails. Add `force=False` escape hatch. Flip the default to
`recompute_checksum=True` *only after* verification succeeds (current
default is still sensible once we verify first).

### 1.3 PyTorch `DataLoader` + h5py safety
`medh5/torch.py:13-18` documents the forked-worker assumption
(`_HANDLE_CACHE` at module level, persisted across fork). Reality:

- macOS and Windows default to `spawn`. With `spawn`, the dataset object is
  pickled; the open `h5py.File` inside cached handles is not pickleable.
- With `fork`, h5py's C-level thread/file state is copied into workers and
  can deadlock or corrupt reads.

Fix (minimal, no behavior change for single-process users):

1. Add `_owner_pid` attribute to `_HandleCache`; on `get()`, if
   `os.getpid() != _owner_pid`, reset the cache dict and update the PID.
   This turns a fork into a cold cache (safe) instead of shared state
   (unsafe). Cost: one extra open per worker per file, which is what we
   want.
2. Provide `medh5.torch.worker_init_fn(worker_id)` that clears
   `_HANDLE_CACHE`. Document it in the docstring and README as the
   supported pattern:
   ```python
   DataLoader(ds, num_workers=4, worker_init_fn=medh5.torch.worker_init_fn)
   ```
3. Implement `MEDH5TorchDataset.__getstate__` → strip the handle cache /
   transform cache so the dataset pickles cleanly for `spawn`.
4. Keep `MEDH5TorchDataset.__getitem__` using `MEDH5File.read()` (it already
   does) — only `MEDH5PatchDataset` relies on the cache.

### 1.4 Packaging for PyPI
All of these are in `pyproject.toml`; none require code changes.

```toml
[project]
version = "0.5.0"
authors = [{ name = "Puyang Wang", email = "<fill in>" }]
readme = { file = "README.md", content-type = "text/markdown" }
license = { file = "LICENSE" }
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: MIT License",
  "Topic :: Scientific/Engineering :: Medical Science Apps.",
  "Typing :: Typed",
  # keep existing Programming Language / OS classifiers
]

[project.urls]
Homepage = "https://github.com/<owner>/medh5"
Repository = "https://github.com/<owner>/medh5"
Issues = "https://github.com/<owner>/medh5/issues"
Changelog = "https://github.com/<owner>/medh5/blob/main/CHANGELOG.md"

[tool.setuptools.package-data]
medh5 = ["py.typed"]
```

Also:
- Create `/Users/puyangwang/medh5/LICENSE` with standard MIT text (author =
  Puyang Wang, year 2026).
- `git rm --cached -r medh5.egg-info` (file is already in `.gitignore`, it
  was tracked before the ignore was added).
- Tighten minimum pins: `h5py>=3.10`, `hdf5plugin>=4.1`, `numpy>=1.24`.

### 1.5 Spatial metadata validation
`meta.py:100-107` checks that `direction` contains numbers but never
verifies it's `ndim × ndim`, and never checks `axis_labels` length. Add:

```python
if s.direction is not None and ndim is not None:
    if len(s.direction) != ndim or any(len(r) != ndim for r in s.direction):
        raise ValueError(f"direction must be {ndim}x{ndim}, got "
                         f"{len(s.direction)}x{len(s.direction[0]) if s.direction else 0}")
if s.axis_labels is not None and ndim is not None and len(s.axis_labels) != ndim:
    raise ValueError(f"axis_labels length {len(s.axis_labels)} != ndim {ndim}")
```

`read_meta` at `meta.py:186-191` silently drops malformed direction with a
warning. Change to raise `MEDH5SchemaError` — a malformed direction in a
medical-imaging file is an error, not a warning we pretend didn't happen.

---

## Phase 2 — Correctness bugs

### 2.1 `RandomFlip` geometry sync (`transforms.py:112-139`)
After flipping axis `k`, the direction-matrix column `k` should negate and
the origin should shift to the opposite corner along that axis. Without
this, any downstream code that uses `meta.spatial` (e.g., NIfTI export,
physical-space metrics, `to_nifti`) silently produces wrong geometry.

Implementation:
```python
def __call__(self, sample):
    flip_axes = [ax for ax in self.axes if self._rng.random() < self.p]
    if not flip_axes: return sample
    # flip images + seg (existing logic)
    meta = sample.get("meta")
    if meta is not None and meta.spatial.direction is not None:
        dir_arr = np.asarray(meta.spatial.direction, dtype=np.float64)
        for ax in flip_axes:
            dir_arr[:, ax] *= -1.0
        # new meta so we don't mutate the dataset's cached copy
        new_spatial = replace(meta.spatial, direction=dir_arr.tolist())
        sample["meta"] = replace(meta, spatial=new_spatial)
    # bbox flip: box[..., ax, :] = [shape[ax]-1-box[..., ax, 1], shape[ax]-1-box[..., ax, 0]]
    return sample
```

Origin update is optional and RAS-dependent; document and leave for later
unless we find a concrete need.

### 2.2 `PatchSampler` + bboxes (`sampling.py:179-195`)
`PatchSampler.sample()` returns `images` + `seg` for the patch but drops
bboxes entirely. For a detection-training user this silently ignores their
labels. Add:

```python
if "bboxes" in f._h5:
    boxes = np.asarray(f._h5["bboxes"][...])
    # shift to patch-local coords
    shift = np.array(starts)
    boxes_local = boxes - shift[None, :, None]
    # keep only boxes whose patch intersection is non-empty
    mask = np.all(
        (boxes_local[..., 1] >= 0) & (boxes_local[..., 0] < np.array(patch)),
        axis=1,
    )
    out["bboxes"] = boxes_local[mask]
    if "bbox_scores" in f._h5:
        out["bbox_scores"] = f._h5["bbox_scores"][...][mask]
```

Make this behaviour opt-in via a `PatchSampler(..., include_bboxes=True)`
flag so existing users aren't surprised.

### 2.3 Empty seg group handling (`core.py:670-671`)
If a file has `seg/` with zero members, `read()` returns `sample.seg = {}`
but `sample.meta.has_seg = False`. Fix: return `None` when the group is
empty.

### 2.4 Bboxes stored uncompressed (`core.py:596-605`)
Minor — bboxes are typically small so this is a low-priority fix. Only
apply the Blosc2 filter when `len(bboxes) > 64`, otherwise leave as-is to
avoid per-dataset chunk overhead for tiny arrays.

---

## Phase 3 — Stability & correctness hardening

### 3.1 `compute_stats` numerical precision
`stats.py:141-146` casts a whole volume to float64 and sums in one shot.
For a 512³ uint16 CT (134M voxels), naive float64 summation loses ~6 bits.
Switch the per-file sum to Welford's online algorithm (or at minimum
`math.fsum` on a generator of row sums). Welford is already the *merge*
strategy but not the *per-file* accumulation — that's the inconsistency.

### 3.2 `compute_stats(workers=N)` untested
The parallel path uses `ProcessPoolExecutor` which defaults to `spawn` on
macOS. No existing test exercises it. Add a tiny fixture (3 files, workers=2)
to catch pickling/import regressions — mandatory CI coverage.

### 3.3 CLI exit codes (`cli.py:674-729`)
`main([])` currently returns 0. Shell automation (`medh5 validate … || exit
1`) is broken. Return `2` for "no command / bad args", `1` for runtime
errors, `0` on success. Replace the manual `if cmd == "import" …` chain
with a single handlers dict that covers all subcommands.

---

## Phase 4 — Essential new feature surface

These are not invented; they are things the README/tests imply exist but
don't, or that any ML researcher will reach for on day one.

### 4.1 Bbox-aware transforms + sampling (see 2.1, 2.2 above).
### 4.2 `worker_init_fn` export (see 1.3).
### 4.3 `MEDH5File.is_valid(path)` convenience
Thin wrapper: `validate(path).is_valid`. Users currently have to construct
and inspect a `ValidationReport`, which is surprisingly verbose for the
common "is this file OK?" check.

---

## Phase 5 — Tests to add (explicit coverage gaps)

All new tests go under `tests/`; none of these exist today.

| Test | File | Why |
|---|---|---|
| `test_write_atomic_interrupted` | `tests/test_roundtrip.py` | Monkeypatch h5py to raise mid-write; assert no file at target path |
| `test_update_verifies_checksum` | `tests/test_integrity.py` | Corrupt a dataset, call `update_meta`, expect `MEDH5FileError` |
| `test_validate_direction_dim_mismatch` | `tests/test_core.py` (or new) | Writing `direction=[[1,0],[0,1]]` with 3D image must raise |
| `test_validate_axis_labels_length` | same | `axis_labels=["x","y"]` with 3D must raise |
| `test_randomflip_updates_direction` | `tests/test_transforms.py` | After flip axis 0, `direction[:,0]` is negated |
| `test_randomflip_bbox` | same | Bbox coords flip symmetrically |
| `test_patchsampler_bboxes` | `tests/test_sampling.py` | Bboxes returned in patch-local coords |
| `test_compute_stats_parallel` | `tests/test_stats.py` | `workers=2`, 3 files, assert same numerics as `workers=1` |
| `test_compute_stats_uint16_precision` | same | Known sum for a 256³ uint16 volume |
| `test_torch_dataloader_spawn` | `tests/test_torch.py` | `DataLoader(ds, num_workers=2, worker_init_fn=medh5.torch.worker_init_fn, multiprocessing_context="spawn")`, iterate fully |
| `test_cli_exit_codes` | `tests/test_cli.py` | `main([])` returns 2; `main(["validate","missing.medh5"])` returns 1 |

Also add `pytest --cov-fail-under=90` is already set; after these
additions the fail-under can move to 92.

---

## Phase 6 — CI and release workflow

Additions to `.github/workflows/ci.yml`:

```yaml
  test-macos:
    runs-on: macos-latest
    strategy: { matrix: { python-version: ["3.12"] } }
    steps: <same as test job>   # catches spawn-default issues

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }
```

(Actual publish on tag push is an optional follow-up; do not automate upload
until the user reviews the dist locally.)

---

## Phase 7 — Documentation

- Add "PyTorch DataLoader with `num_workers>0`" section to README — show
  `worker_init_fn` usage, explain the h5py fork-safety issue in one
  paragraph, link to h5py's documentation.
- CHANGELOG.md entry for 0.5.0 covering: atomic write, checksum-verify on
  update, worker-safe cache, dim-checked metadata, bbox sampling, CLI exit
  codes, PyPI packaging.
- README "Status" line: change from silent to explicit "Beta — WIP, no
  backcompat guarantees yet."

---

## Phase 8 — What we deliberately are NOT doing

Called out so the user can veto or add back:

- **No `to_dicom()` export.** DICOM write is a multi-week project; scope
  creep for a v0.5 release.
- **No SWMR / concurrent-writer support.** Not requested and not trivial.
- **No seg label remapping.** Users can `np.vectorize` one-liner.
- **No `seg_ops` API redesign** (compression params mixed with action dict
  — ugly but works; fixing would be churn without user-visible win).
- **No upper-bound deps.** Medical-imaging users have enough dependency
  hell already; we pin lower bounds only.

---

## Verification plan

After each phase, run in this order:

```bash
ruff check . && ruff format --check .
mypy medh5
pytest tests/ -v --cov=medh5 --cov-fail-under=90
```

End-to-end release smoke test (manual, before tagging):

```bash
# 1. Build the wheel + sdist
python -m build

# 2. Verify metadata, LICENSE, py.typed all present
twine check dist/*
unzip -l dist/medh5-0.5.0-py3-none-any.whl | grep -E "LICENSE|py.typed"

# 3. Install in a fresh venv and run the README quickstart verbatim
python -m venv /tmp/medh5-release && source /tmp/medh5-release/bin/activate
pip install dist/medh5-0.5.0-py3-none-any.whl'[torch,nifti,dicom]'
python -c "
import numpy as np, medh5
medh5.MEDH5File.write('/tmp/x.medh5',
    images={'CT': np.zeros((32,64,64), np.float32)},
    spacing=[1,1,1], direction=[[1,0,0],[0,1,0],[0,0,1]],
    coord_system='RAS', checksum=True)
print(medh5.MEDH5File.read('/tmp/x.medh5'))
print('verify:', medh5.MEDH5File.verify('/tmp/x.medh5'))
"

# 4. Torch DataLoader worker smoke test (macOS default = spawn)
python -c "
import torch, medh5, medh5.torch
ds = medh5.torch.MEDH5TorchDataset(['/tmp/x.medh5'])
loader = torch.utils.data.DataLoader(ds, num_workers=2,
    worker_init_fn=medh5.torch.worker_init_fn,
    collate_fn=lambda x: x)
for batch in loader: print(batch[0]['images']['CT'].shape)
"

# 5. Interrupted-write recovery
python -c "
import os, signal, threading, medh5, numpy as np
# write starts, SIGINT partway, assert no stale file
... (see test_write_atomic_interrupted)
"
```

If all five steps pass and CI is green on Linux + macOS, the package is
ready for `twine upload --repository testpypi dist/*`, then (after
downloading + installing from TestPyPI works), `twine upload`.

---

## Rough effort estimate

| Phase | Effort |
|---|---|
| 1. Release blockers | ~1 day |
| 2. Correctness bugs | ~1 day |
| 3. Stability | ~0.5 day |
| 4. Features | ~0.5 day (mostly covered by 2.2) |
| 5. Tests | ~1 day |
| 6. CI | ~2 hrs |
| 7. Docs | ~2 hrs |
| **Total** | **~4 working days** to a tagged 0.5.0 release |
