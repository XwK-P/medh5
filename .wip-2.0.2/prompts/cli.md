# Workstream `cli`: the command line (W5, plus bench and CLI-doc items)

Worktree: `SCR/wt/cli` (branch `w202/cli`). It is created from the integration branch, so it already contains the merged wave-1 work.

## Files you own
- `crates/medh5-cli/src/{app,bench,common,conformance,curation,inspect,labels,lib,main,perf,seg}.rs`.
- `crates/medh5-python/src/cli.rs`.
- `docs/reference/cli.md`.
- Tests: the native CLI tests in `crates/medh5-cli` and `tests/tools/test_cli.py`. The conv2 agent added F11 tests to `test_cli.py`; keep them.
- Bench: `crates/medh5/src/bench.rs`, `medh5/bench.py`, `tests/tools/test_bench.py`.

## Not yours (done by other agents; don't redo them)
- `dataset.rs`: `dataset stats` exit code and stderr.
- `convert.rs`: migrate report checks.
- `scrub.rs`: the scrub partner flags.
- `clinical.rs`: the clinical export guard.
- The `medh5 track` legend, if the curation agent changed it in `curation.rs`; keep it.

## Items (quoted from the audit; verify each against the code first)
1. `index build`/`seg convert` (`seg.rs:163-185, 226-250`):
   - both rewrite the file, with a new inode, even when nothing changes; make them leave the file untouched when nothing would change;
   - `seg convert` transcodes twice; short-circuit `from == to`;
   - `index build` with nothing to index exits 1 silently; make it exit 0 and say so, or fail with a message. Decide and report.
2. **One failure shape** for batch `--json` output across `fix`, `index build`, `labels` and `splits`: `ok`, `error` (`inspect.rs:528`; `seg.rs:237`; `labels.rs:96`; `curation.rs:350`). Send unreadables to stderr. Keep the existing keys for compatibility and add what's missing; don't remove keys.
3. `verify`'s text output omits unverifiable datasets. `fix` exits 0 with an index still stale, and drops notes (`inspect.rs:433-458, 489-532`). The valid agent changed `integrity/repair.rs` (a `malformed` field and per-path outcomes); build on what is in your worktree.
4. **Partner flags** that are silently ignored without their partner. Refuse them with a clear message:
   - `fix --reason/--by` without `--rewrite-digests`;
   - `split --fold/--assigned-by` without `--write-claims`.
5. Other CLI fixes:
   - `agree --threshold` accepts NaN, negatives and values above 1; refuse them;
   - help goes to stdout for bare `medh5` but to stderr for groups; make it consistent, and match common practice: help requested explicitly goes to stdout, usage errors to stderr;
   - Python-API wording appears in CLI messages; find the instances and give CLI wording;
   - the NativeHost probe hides Python's stderr and accepts a 1.x `medh5`. Show the stderr on failure, and require a compatible version;
   - `PyStream` is unbuffered (`crates/medh5-python/src/cli.rs`); buffer it, and flush correctly.
6. **W7:** prefix abbreviation of long flags is undocumented. Document it in `docs/reference/cli.md`, or turn it off if it is accidental and risky. Decide, report, and keep the docs test of flags passing.
7. **W2, small:** `bench` accepts `repeats=0`; refuse it in the engine, the Python API and the CLI.
8. **W9, speed:** `tests/tools/test_bench.py:102` takes a large share of the run. Make it cheaper without losing what it checks.

## Notes
- Every documented `medh5 …` flag is checked against the parser (`tests/project/test_docs_examples.py`); run it.
- Tests to run:
  - `wcargo cli test -p medh5-cli`;
  - `wext cli`, then `wpy cli -m pytest tests/tools/test_cli.py tests/tools/test_bench.py tests/project/test_docs_examples.py -q`.
