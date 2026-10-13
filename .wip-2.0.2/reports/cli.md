## cli workstream: final report (branch `w202/cli`, 4 commits on `a7351b6`: 4b5a33b, cd789e8, a943342, bfc2878)

### Items (all DONE)
1. `index build` keeps an entry already holding what the build would write (digest, `max_coords`, `seed`, occupancy presence, tables compared against an in-memory build); a file whose entries all hold is not rewritten. `seg convert` to the stored encoding writes nothing; a real conversion transcodes once; a kind refusal comes before the copy. (`seg.rs`, `test_cli.py`)
2. One failure shape in batch `--json`: `{path, ok:false, error}` in `fix`, `index build` (was `null`), `labels check` (new `files` list), `splits` (unreadables on stderr in CLI words; stdout `UNREADABLE` lines gone; not counted "without a claim"). No keys removed. (`common.rs`, `inspect.rs`, `seg.rs`, `labels.rs`, `curation.rs`)
3. `verify` text prints `UNCHECKED` lines; `fix` exits 0 only when nothing is left, including the index (re-read after a rewrite); notes printed whatever `fix` did; a "malformed digest" line (valid's `malformed`). (`inspect.rs`)
4. Partner flags: `fix --reason/--by` need `--rewrite-digests`, `dataset split --fold/--assigned-by` need `--write-claims` (clap `requires`, exit 2). (`app.rs`)
5. `agree --threshold` in (0, 1]; bare `medh5` prints help on stderr, `--help` on stdout; Python-API wording mapped to CLI wording (`in_cli_terms` in `Ctx::fail`/`file_failed`); NativeHost probe shows Python's last stderr line, requires a medh5 of the same major, says why; `PyStream` line-at-a-time where Python's stream flushes that often (stderr, `line_buffering`, `write_through`, `isatty`), else 64 KiB blocks, stdout flushed before every stderr write. (`app.rs`, `lib.rs`, `common.rs`, `medh5-python/src/cli.rs`)
6. Long-option abbreviation kept (1.x argparse compatibility) and documented in `cli.md`, with the advice to spell options in full in scripts; a Rust test holds `--dry` and refuses ambiguous `--dr`.
7. Bench refuses `repeats=0` and `patch=0` (`check_repeats`/`check_run`, `MEDH5ValidationError`; engine, Python and CLI). (`crates/medh5/src/bench.rs`)
8. Slow bench test: the Python test measures a file; the synthetic path is tested in Rust (`bench_with` at small shapes).
Handoffs done in `cli.md`: scrub coverage and `--apply` partner flags; `dataset stats`/`dataset index`; the F11 sentence; the `track` legend; sec's rewrites through links and the sticky-directory refusal; clin's `clinical export`/`clinical select` sentences; `app.rs` "with --apply" on three scrub flags; "lossless but one" corrected to name both lossy conversions (cli.md and `seg convert` help).

### Tests
Rust: `a_partner_flag_is_refused_without_its_partner`, `an_iou_threshold_lies_in_0_to_1`, `a_long_option_shortens_to_a_prefix_that_names_it_alone`, `a_python_is_handed_a_command_only_by_a_package_of_this_major`, `s14_bench_json_is_only_the_document`, `s14_a_run_that_measures_nothing_is_refused`. Python: ~20 in `test_cli.py` (`TestBatches::test_a_file_that_failed_has_one_shape_of_entry`, `TestFixAndScrub::test_S13_3_fix_succeeds_exactly_when_nothing_is_left_to_fix`, `TestOutput` (3 buffering modes), `TestSegAndIndex`, help streams, partner flags) + bench refusal in `test_bench.py`.
Run: medh5-cli tests 15 passed; `bench::` 3 passed; clippy (medh5-cli, medh5, medh5-python) clean; fmt clean; test_cli + test_bench + test_docs_examples 500 passed; other CLI-driving test files 1274 passed, 2 skipped; ruff, mypy clean; mkdocs --strict ok; corpus 163/163 (console script and native debug binary); cli.md vs 25 STALE_CLAIMS + 7 STALE_PHRASES: 0 hits.

### CHANGELOG drafts
**Fixed**
- **`medh5 fix` exits 0 only when nothing is left to fix.** It exited 0 with an index entry its `--rewrite-digests` left stale, or digests `--rebuild-index` could not touch; it now always prints its notes and counts malformed digests.
- **`medh5 verify`'s text output names the datasets it could not check** on `UNCHECKED` lines, as `--json` already listed them.
- **`index build` and `seg convert` leave files they would not change untouched;** they rewrote them, copying every image and putting a new inode under every open handle.
- **In batch `--json` output, a file that failed is `{path, ok: false, error}`** in `fix`, `index build`, `labels check` and `splits`; `splits` reports such files on stderr instead of `UNREADABLE  path: MEDH5FileError: …` on stdout.
- **CLI messages name the flags,** not Python keyword arguments (`rewrite_digests`, `rebuild_index`, `` `key` ``).
- **The standalone binary says why it cannot hand a command to Python** (Python's last line, or the version found), and no longer hands commands to a 1.x package.
- **`medh5.bench` refuses `repeats=0` (NaN rows) and `patch=0` (timed reads of nothing).**
- **`seg convert --help` names both lossy conversions.**

**Behaviour changes**
- `fix --reason/--by` without `--rewrite-digests`, and `dataset split --fold/--assigned-by` without `--write-claims`, are usage errors.
- `agree --threshold` must lie in (0, 1].
- `fix` exits 1 while an index entry is left stale.
- `index build` with nothing to index exits 0 (it was 1, after rewriting the file); its failure entry is an object, not `null`.
- `splits --json` `unreadable` entries gain `ok: false` and carry the CLI's message.
- Bare `medh5` prints its help on stderr.
- The standalone binary requires a Python `medh5` of its own major version.

**Performance**
- `index build` and `seg convert` skip the copy-on-write rewrite when nothing changes, and `seg convert` transcodes once.
- The Python console script hands Python its output by line or by 64 KiB block, not one call per fragment.

**Documentation**
- The CLI reference covers help and usage streams, long-option prefixes, the batch failure entry, rewrites through links, `verify`'s `UNCHECKED`, `fix`'s exit code and partner flags, `seg convert`'s no-op and both lossy conversions, `index build`'s no-op and JSON, `agree`'s threshold range, `bench`'s floors, and the scrub, dataset, track, F11 and clinical sentences.

### Handoffs
- curation (engine): `compare`/`compare_instances` should refuse an out-of-range threshold in the Python API too (only the CLI does now).
- pyapi: a negative `repeats` or `patch` in `medh5.bench` raises pyo3's `OverflowError`, not `MEDH5ValidationError`.
- docs: README and `install.md` say `python3` only; Windows uses `python`.
- conv2: `medh5/cli.py` docstring says `python3` only.
- clin: unify the three private `same_file` helpers in `common.rs` (post-merge refactor).

### Decisions
- Nothing to index is done (exit 0, said on stdout): an index is optional per annotation (§14.3).
- "Would change nothing" decided exactly by an in-memory build (the occupancy block size is not stored).
- Prefix abbreviation kept and documented.
- Python-API wording translated by a 4-row phrase table in `Ctx`; the engine keeps its messages for the Python API.
- `fix`'s "ok" = no stale entry left and digests rewritten or not needed.
