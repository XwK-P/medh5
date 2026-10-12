# Workstream `pyapi`: Python API surface (F22, top issue 14 `gil_used`, W9 read counts and `_live`, small items)

Worktree: `SCR/wt/pyapi` (branch `w202/pyapi`). It is created from the integration branch, so it already contains the merged wave-1 work.

## Files you own
- `crates/medh5-python/src/{nodes,document,lib,misc,values,errors,rng}.rs`.
- `crates/medh5/src/{document,pyval}.rs`.
- `medh5/{__init__,nodes,document,errors,_live}.py`.
- Tests: `tests/format/test_nodes.py`, and tests for `_live` in the module that tests the live metadata views (find it; probably `tests/format/test_sample.py`). Those are localized additions.
- **Read-count instrumentation:** localized edits in the h5 read paths (`crates/medh5/src/h5/data.rs` and `window.rs`), plus `tests/tools/test_sampling.py:326-349` and `tests/format/test_readers.py:533-564`.

## Items (quoted from the audit; verify each against the code first)
1. **F22:** string datasets break the generic `Dataset` interface (`crates/medh5-python/src/nodes.rs:262-305`).
   - What goes wrong today:
     - `read()` and `np.asarray(ds)` raise "holds strings, not numbers", although the `medh5/nodes.py` docstring, `docs/reference/python-api.md:248` and the stub (`read() -> NDArray`) promise them;
     - `ds[...]` and `ds[:,]` raise;
     - on a 2-D string dataset, `ds[0]` returns one element, not the row;
     - a scalar string dataset ignores its key.
   - **Fix:** fill `read_strings` into an object array of the stored shape (0-d for a scalar), and return it from `read()`, which also fixes `__array__`. Index that array in `__getitem__`.
   - Reproductions: `SCR/reports/indep-transforms-nodes.md`, `SCR/agents/indep/transforms-nodes/`.
2. **Top issue 14:** free-threaded builds claim to be GIL-free without an audit. PyO3 0.29.3 emits `Py_MOD_GIL_NOT_USED` when `gil_used` is absent, so a from-source 3.14t build runs `_core` without the GIL. Write `#[pymodule(gil_used = true)]` explicitly (`crates/medh5-python/src/lib.rs:44`), with a comment on why.
3. **W2, small:** `SampleDocument.loads` decodes invalid UTF-8 lossily; refuse it (`crates/medh5-python/src/document.rs` and/or `crates/medh5/src/document.rs`).
4. **W9, read counts:** two read-count regression tests cannot fail. They count `h5py.Dataset.__getitem__` calls, but the package no longer uses h5py, so the 1.x "190 reads per draw" regression could return unseen (`tests/tools/test_sampling.py:326-349`; `tests/format/test_readers.py:533-564`).
   - Replace them with engine-level counts: a Rust test, or a debug counter in `_core`.
   - A private, underscore-named counter is acceptable, e.g. `_core._read_stats()` that returns and resets the number of hyperslab/window reads. It must be cheap (a relaxed atomic) and not public API: no docs, and only a `_core.pyi` entry if stubtest requires it.
   - Prove the new tests would catch the regression: temporarily force extra reads, and watch them fail.
5. **W9, `_live.py`:** test the mutation methods; they are 60% covered today.
   - Dict: `pop`, `popitem`, `clear`, `update`, `setdefault`, `|=`.
   - List: `insert`, `remove`, `sort`, `+=`, etc.
   - Each must write through, or refuse a stale anchor. Fix any that don't.
6. **W7, code docs:**
   - `medh5/__init__.py:20-21` still says "1.0 ships a reader"; fix it;
   - `MEDH5Error` is "base of every error", but HDF5 I/O errors are plain `OSError`. Make the docstrings in `medh5/errors.py` accurate. Don't change the exception hierarchy.
7. **W6:** release the GIL in the remaining long native calls. **Only in the binding files you own** (`nodes.rs`, `document.rs`, …). List the long calls in *other* binding files that still hold the GIL; the coordinator handles those after all merges:
   - annotation encoders, `transcode`, `annotation_to_masks`;
   - `add_mask`, `tracks(measure=True)`, `IndexHandle` reads;
   - storage `build_index`/`occupancy`;
   - transform sampling and Jacobians;
   - integrity helpers;
   - dataset `find`/`manifest_stale`;
   - `CacheWriterHandle.add`/`commit`, `ClinicalHandle.records`.

## Notes
- `medh5/_core.pyi` must match the bindings; `tests/project/test_typing.py` runs stubtest. Run it.
- Tests to run:
  - `wext pyapi`, then `wpy pyapi -m pytest tests/format/test_nodes.py tests/format/test_sample.py tests/tools/test_sampling.py tests/format/test_readers.py tests/project/test_typing.py -q`;
  - plus whatever module holds the `_live` tests.
