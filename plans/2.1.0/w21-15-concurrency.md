# W21-15 Concurrency: the GIL and `--workers`

| | |
|---|---|
| Branch | `w21/concurrency` |
| Size | M overall: PR 1 S–M, PR 2 M–L |
| Lands after | W21-00 PR 4, PR 5; PR 2 after W21-00 PR 6 (children start under the HDF5 lock) |
| Blocks | W21-16 PR 5 |
| Owns | `crates/medh5-cli/src/workers.rs` (new), the `Host` trait (`crates/medh5-cli/src/common.rs`) and `PyHost` (`crates/medh5-python/src/cli.rs`), `tests/format/test_threads.py` (new); touches the binding files that hold the GIL (small, local edits) |
| Decisions | none beyond the rules |
| Backport | none |

## Session prompt

> Implement workstream W21-15 (concurrency: the GIL and `--workers`) of medh5
> 2.1.0 in this repository. First read `plans/README.md` and
> `plans/2.1.0/README.md`, then `plans/2.1.0/w21-15-concurrency.md` in full. Work
> from the latest `main` on `w21/concurrency`; land the two pull requests in
> order, and check before PR 2 that W21-00 PR 6 has merged. One HDF5 lock covers
> each process, filters included, so parallelism belongs to processes; output
> must not depend on the worker count, byte for byte. Measure before and after
> each GIL change and put the numbers in the pull request. Write tests first, run
> the local gate from `CLAUDE.md`, add fragments, open `W21-15: …` pull requests,
> and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 15.1 The remaining GIL holders | S–M |
| 2 | 15.2 `--workers` for validate, verify, fix, index build, recompress | M–L |

## Items

### 15.1 The remaining GIL holders (PR 1)

- **State at 2.0.2.** The CHANGELOG says transcoding, digests, index building and
  field sampling release the GIL. These still hold it:

  | Holder | Location | Work under the GIL |
  |---|---|---|
  | `typed()` | `crates/medh5-python/src/convert.rs` | the copy behind `bool_array`, `f64_array`, `masks_arg`: every large array argument |
  | `normalize_masks` | `annotations.rs` | copy, shape checks, `masks_to_py` |
  | `encode_contours` | `annotations.rs` | packing |
  | `polygon_arg` | `annotations.rs` | walks Python objects |
  | `encode_voxels` | `annotations.rs` | everything but `analyse` |
  | `mask_pair`, `agreement_dice`, `agreement_iou` | `curation_tools.rs` | broadcast copy and count |
  | encoders for boxes, OBBs, keypoints, points; the writer's `add_*` for them | `annotations.rs`, `writer.rs` | O(N) |
  | `to_world`, `to_index` | `reader.rs` | O(N) |
  | `refuse_outside` | `transforms.rs` | O(N) |
  | single-box helpers | `geometry.rs` | O(1) |

  Measured on 300×512² masks: `normalize_masks` (bool) 0.148 s, longest pause
  another thread saw 0.131 s; (uint8) 0.526 s, 0.062 s (NumPy's cast releases);
  `agreement_dice` 0.612 s, 0.612 s.
- **Change.** Release the GIL where work grows with the data or waits on HDF5;
  keep it for O(1) work and anything walking Python objects.
  1. `typed()`: for arrays of 1 MiB or more, keep the `PyReadonlyArray` guard and
     copy with `py.detach(|| view.as_standard_layout().into_owned())` (the view is
     `Send` when `T: Sync`). A Python thread writing the array during the call may
     see a torn copy, as NumPy's own GIL-free copies allow; the docs say not to
     mutate an argument mid-call.
  2. `normalize_masks`: check the shape on the view and return
     `np.ascontiguousarray(masks, dtype=bool)`, no copy for C-contiguous bool
     input.
  3. `agreement_*`: count on views while detached, broadcasting lazily
     (`broadcast_with`).
  4. `encode_contours`: traverse the polygons under the GIL, detach the packing.
  5. The O(N) encoders and writer `add_*`, `to_world`, `to_index`,
     `refuse_outside`: detach after argument conversion when N ≥ 4096 rows.
  6. Keep `polygon_arg`, the single-box helpers and `basis` (`transforms.rs`,
     O(S²)).
- **Tests.** New `tests/format/test_threads.py`, using the ticker of
  `tests/format/test_integrity.py`: `test_normalize_masks_does_not_copy_contiguous_bool`
  (`np.shares_memory`); `test_agreement_releases_the_gil` (longest pause under
  25 % of the call); `test_large_array_arguments_copy_without_the_gil` (through
  `add_segmentation`); `test_encode_contours_releases_while_packing`.
- **Changelog.** Performance, area `python-api` (name the calls that now release
  the GIL).

### 15.2 `--workers` for validate, verify, fix, index build, recompress (PR 2)

- **State at 2.0.2.** All five run sequentially (`validate` through
  `validate_paths`, a per-file map; `verify`, `fix`, `index build`, `recompress`
  in `crates/medh5-cli/src/{inspect.rs,seg.rs,perf.rs}`). Only `stats --workers`
  exists, with threads, merged in path order (`dataset/stats.rs`). One HDF5 lock
  per process covers the filters too (Blosc2 decodes outside it only on the
  partial-window path); measured: `verify` of 8 small files 1.22× on 2 threads,
  1.06× on 4, 1.37–1.42× with 2 processes including start-up; `validate --level
  semantic` *slower* on 2 threads (0.8–0.9×); `verify` of 4 × 75 MB `training`
  files (window path) 1.96× and 2.14×. `docs/explanation/design-rationale.md`
  says parallelism belongs to processes.
- **Change.**
  - The flag: `count("workers", "workers", "processes to run files in",
    0).default_value("1")` on the five commands; 0 and 1 both mean sequential
    (as for `stats`, which keeps its threads).
  - New `crates/medh5-cli/src/workers.rs`: each command's loop body moves into
    `fn one(path, m, ctx) -> FileOutcome`, which the sequential path calls
    directly:

    ```rust
    pub struct FileOutcome { text: Vec<u8>, json: Option<Value>, ok: bool, error: Option<(String, String)> }
    pub fn run_files(m: &ArgMatches, ctx: &mut Ctx, paths: &[String],
                     one: fn(&str, &ArgMatches, &mut Ctx) -> FileOutcome) -> CmdResult
    ```

  - Protocol: the parent starts `min(N, files)` children through `medh5::h5::spawn`
    (W21-00 PR 6), each running `<worker_command> __worker <command> <args
    without paths and --workers>` (a hidden subcommand); it writes
    `{"index","path"}` JSON lines to each child's stdin and reads outcome lines
    back; a reorder buffer prints outcomes in input order, so text, `--json` and
    the exit code match a sequential run byte for byte; an error crosses as
    `(kind, message)`, is re-raised at the same file, and stops the run where a
    sequential run would.
  - Writing commands (`fix`, `index build`, `recompress`): paths naming one file
    (`h5::file::same_file`) go to one worker, in order.
  - A worker that dies fails only the file it held; its queue moves to the others.
  - `Host` (a public trait of `medh5-cli`) gains a method with a default body,
    so no implementor breaks: `fn worker_command(&self) -> Option<Vec<OsString>>
    { None }`. `NativeHost` returns `current_exe()`; `PyHost` returns
    `sys.executable -m medh5.cli`. With `None`, `--workers N` runs in one
    process and says so once on standard error ("this host cannot start
    workers; running in one process").
  - The parent opens no file before spawning.
  - Optional, measure first: route whole-dataset Blosc2 reads through
    `read_window` so their decode leaves the lock (today `h5/data.rs` returns
    before reaching it), tested against `read_hyperslab` as the window invariant
    requires.
- **Tests.** `tests/tools/test_cli.py::TestWorkers` (main, macOS and Windows jobs):
  `test_output_does_not_depend_on_the_worker_count[validate|verify|fix|index-build|recompress]`
  (1, 2 and 3 workers over five files, one defective; text and JSON compared byte
  for byte); `test_the_exit_code_does_not_depend_on_the_worker_count`;
  `test_a_file_named_twice_is_written_by_one_worker` (a hard link);
  `test_a_dying_worker_fails_only_its_file`;
  `test_an_error_stops_where_sequential_stops`. Rust
  `workers::tests::the_reorder_buffer_emits_in_input_order`, and a native-binary
  run in the `rust` job.
- **Docs.** `docs/reference/cli.md` (flags are checked by `test_docs_examples.py`),
  `docs/guides/performance.md`.
- **Changelog.** Added and Performance, area `cli`.
