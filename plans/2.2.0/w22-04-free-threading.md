# W22-04 Free-threading

| | |
|---|---|
| Branch | `w22/free-threading` |
| Size | L |
| Lands after | v2.1.0 released; PR 4 after PRs 1–3 |
| Builds on (2.1.0) | cp314t wheels and the `test-free-threaded` and `test-no-gil` jobs (W21-01 PR 3); the GIL releases (W21-15 PR 1) |
| Owns | `crates/medh5-python/src/lib.rs` (`gil_used`), the mutable pyclasses, `medh5/torch/handles.py`, `medh5/labels.py` and `medh5/_live.py` registries, `tests/format/test_threads.py` |
| Decisions | E4 |

## Session prompt

> Implement workstream W22-04 (free-threading) of medh5 2.2.0 in this
> repository. First read `plans/README.md`, `plans/2.2.0/README.md` and
> `plans/2.2.0/decisions.md` (E4), then `plans/2.2.0/w22-04-free-threading.md`
> in full. Work from the latest `main` on `w22/free-threading`; land the four
> items in order, as one pull request. `gil_used = false` is the last change,
> made only when every audit item is done and the free-threaded stress job
> passes. A concurrent misuse must be refused deterministically, never deadlock
> and never race. HDF5 handles must not cross `fork`. Write tests first, run the
> local gate from `CLAUDE.md` and the suite on a free-threaded 3.14t interpreter
> with `PYTHON_GIL=0`, add fragments, open a `W22-04: …` pull request, and arm
> auto-merge only as plans/README.md's "Review before the queue" says.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 04.1 Mutable pyclasses: one thread at a time, refused, not raced | M |
| 2 | 04.2 Python-side shared state | S–M |
| 3 | 04.3 The engine's statics, mutexes and the HDF5 lock | M |
| 4 | 04.4 The stress job, and `gil_used = false` | S |

## Items

### 04.1 Mutable pyclasses (PR 1)

- **State at 2.1.0** (checked at v2.1.0). `#[pymodule(gil_used = true)]`
  (`lib.rs`, held by a `test_packaging` test). Seven pyclasses are not
  `frozen`: `SampleDocument` (`document.rs`), `AnnotationPayload`
  (`annotations.rs`), `Provenance` (`curation.rs`), `SampleWriter`
  (`writer.rs`), `CacheWriterHandle` (`clinical.rs`), `StagedFileHandle`
  (`files.rs`) and `PatchDraws` (`sampling.rs`, whose `__next__` takes
  `PyRefMut`, so `rg "&mut self"` misses it). Recount with `rg "#\[pyclass"
  crates/medh5-python/src | rg -v frozen` before starting. Beside the
  methods, the `try_borrow_mut` sites (`SampleWriter`'s setters in
  `writer.rs`, `curation.rs`'s provenance edit) change too. A `SampleWriter`
  used by a second thread raises `RuntimeError: Already borrowed` (2.0.2
  documents it).
- **Change.** Each becomes `frozen` with an inner `Mutex`; every method takes the
  lock with `try_lock`, and a second thread gets a `MEDH5Error` ("a writer is used
  by one thread at a time"). Never hold the lock across a call into Python.
- **Tests.** `tests/format/test_threads.py::test_two_threads_on_one_writer_get_a_medh5_error`
  (and one per class).
- **Changelog.** Behaviour changes, area `python-api` (the error type).

### 04.2 Python-side shared state (PR 2)

- **Change.** `medh5/torch/handles.py`: `_ensure_owner` checks then acts outside
  the lock; use `os.register_at_fork(after_in_child=...)` plus the lock.
  `medh5/labels.py` `_REGISTERED`: a lock. `_live.py` has no registry; its
  shared state is each anchor's token, checked and then replaced
  (`_Anchor.rewrite`), which needs the document's lock around both. The lazy
  `if self._x is None` caches in `sample.py` and `task.py`: benign duplicate work;
  document it in a comment.
- **Tests.** `test_handle_cache_survives_parallel_leases` (under
  `pytest-run-parallel` in the free-threaded job); `test_registries_are_safe_across_threads`.

### 04.3 The engine's statics, mutexes and the HDF5 lock (PR 3)

- **Change.** Check every `PyOnceLock` static, the `checked_files` `Mutex`
  (`h5/ops.rs`) and the atomic counters (fine as they are), and that no Rust mutex
  is held across a call into Python. There are 38 frozen pyclasses (28
  declared directly, 10 through `record_class!` in `curation.rs`); the records
  hold values and JSON, the handles an `Arc`, and `SampleHandle` a `Mutex`.
  Detach before `locked()` (`h5/mod.rs`): a thread blocked
  there while attached stalls stop-the-world under free-threading; confirm no
  Python callback (`__fspath__`, `__array__`, a progress hook) runs while the HDF5
  lock is held. c-blosc2 contexts in `b2nd_slice.c` must be per call with
  `blosc2_init` called once. Verify rust-numpy's borrow tracking on 3.14t. Write
  the audit's findings into the pull request as a checklist.
- **Tests.** A Rust test that `locked()` is never entered with the GIL attached
  (a debug assertion under a test feature); a stress test reading one sample from
  eight threads.

### 04.4 The stress job, and `gil_used = false` (PR 4)

- **Change.** The `test-free-threaded` job runs with `PYTHON_GIL=0` and
  `pytest-run-parallel` over `tests/format` and `tests/tools`; `test-no-gil` folds
  into it. Then `#[pymodule(gil_used = false)]`, and the packaging assertion
  becomes `sys._is_gil_enabled() is False` after `import medh5` on a free-threaded
  interpreter.
- **Docs.** `docs/install.md` (free-threaded wheels), the concurrency page.
- **Changelog.** Added, area `platform`: "**medh5 runs on free-threaded CPython
  without re-enabling the GIL.**"
