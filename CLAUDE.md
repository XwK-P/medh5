# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# The Python package, with the extension built from crates/ (maturin, release
# profile) and the extras the test suite needs.  Re-run after changing Rust.
pip install -e ".[dev,torch,nifti,dicom,dicomseg,itk]"

# The engine and the native CLI
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace            # unit tests, the corpus, the README doctest

# Full Python suite (90% coverage floor)
pytest tests/ --cov=medh5 --cov-report=term-missing --cov-fail-under=90

# A single file or test
pytest tests/v1/test_sample.py -v
pytest tests/v1/test_dataset.py::TestSplits -v
cargo test -p medh5 --lib h5::file

# Lint, format, types
ruff check . && ruff format --check . && mypy medh5

# The conformance corpus must stay green --- through both command lines
medh5 conformance run /tmp/corpus
cargo run --release -p medh5-cli -- conformance run /tmp/corpus-native

# MONAI stays out of the line above --- it is heavy, and only a handful of
# tests need it. Those tests `importorskip`, so they skip silently without it
# and CI runs them in a dedicated job.
pip install -e ".[monai]" && pytest tests/ -k "Monai or monai"

# The documentation site. `--strict` makes an unresolved cross-reference a
# build failure, which is the gate CI runs; `serve` live-reloads on :8000.
pip install -r docs/requirements.txt
mkdocs build --strict
mkdocs serve
```

## Pre-commit checks

All of these must pass before committing:

```bash
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test --workspace \
  && ruff check . && ruff format --check . && mypy medh5 \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus \
  && mkdocs build --strict
```

## The model

**Format 1.0, package 2.0.** A `.medh5` file is **one subject at one or more
timepoints**, with every image, annotation, transform and curation record about
them. Not one scan — one subject. Most of the design follows from that: splitting
by file is subject-safe, a change annotation has a referent, and registration
between visits is an object in the file rather than a convention between
filenames.

`docs/spec/medh5-1.0.md` is **normative**. Code implements it; when they
disagree, one of them is a bug. Appendix C records the clauses corrected
because implementing them showed the text was not implementable — including
the four the Rust re-implementation found.

## Architecture

**One format engine, three frontends.** The engine is Rust; the Python package
and the CLI are layers over it, and all three read and write the same bytes.

- **`crates/medh5`** — the engine (crates.io `medh5`). Modules map onto
  specification sections, so a spec change has one obvious home: `sample`
  (§2, §4, §14.4), `geometry` (§3), `annotations` (§5–§9), `transforms` (§10),
  `labels`, `curation` (§11–§12), `integrity` (§13), `storage` (§14),
  `validate` (§15), `collection`, `dataset`, `sampling`, `conformance`.
  `data/codes.json` is the §15.2 diagnostic code table (a test asserts the
  table and the spec agree); `data/` also holds the schema and vocabularies,
  embedded at compile time.
- **`crates/medh5-sys`** — HDF5 (static, built from source), C-Blosc2
  (vendored) and the Blosc2 (32026) and Zstandard (32015) HDF5 filters.
- **`crates/medh5-cli`** — the `medh5` command line (crates.io `medh5-cli`).
  The Python console script runs the same code through `_core.cli_main`;
  converters are Python and the standalone binary spawns `python3 -m medh5.cli`
  for them (`Host`).
- **`crates/medh5-python`** — the PyO3 extension `medh5._core` (not published;
  maturin builds it into the wheel). Engine handles live in a facade's
  `._handle`; result types are rebuilt as the 1.x Python classes.
- **`medh5/`** — the Python package: thin modules re-exporting `_core` under
  the 1.x import paths, plus what is Python's — `torch`, `monai`, `io`
  converters. `medh5/_core.pyi` types the extension; keep it in step with the
  bindings (`tests/v1/test_typing.py` runs `stubtest` against the build).
- **`conformance`** — the corpus is a *shipped artifact*, not a test fixture:
  third-party implementations run it. The engine builds it
  (`crates/medh5/src/conformance/build.rs`).

The version has one source, `[workspace.package] version` in `Cargo.toml`:
every crate inherits it, maturin stamps it on the wheel, and the engine writes
it into every file's `generator`.

## Invariants that are easy to break

- **Coverage.** `class_ids` is what an annotation contains;
  `annotated_class_ids` is what was *looked for*. A class examined and absent is
  a usable negative; a class nobody examined is not. Never collapse the two.
- **Boxes sit at voxel edges**, indices at voxel centres. `[a, b]` is the slice
  `a+0.5 : b+0.5`. Every off-by-one in detection lives here.
- **Digests cover decompressed content**, so recompression changes every stored
  byte and no digest. `content_id` is a Merkle root over *stored digests*, so an
  edited dataset breaks its object digest and leaves the root matching — verify
  per object, never only the root.
- **Canonical JSON is defined to the byte** (§5.1): sorted keys, no
  whitespace, UTF-8, Python's float spelling. A NaN in an *attribute* hashes as
  `NaN`, as 1.x hashed it; a NaN in a *document* is refused (JSON has none).
  `content_id` must not depend on which version wrote a file.
- **Geometry is never invented.** Converters refuse rather than resample, guess a
  grid, or fabricate a transform. `transform_between` returns `None` when no path
  exists.
- **HDF5 handles must not cross `fork`.** The torch handle cache is PID-keyed and
  a forked child *abandons* the parent's handles (`SampleHandle.abandon`) rather
  than closing them.
- **Closing a file closes what was opened through it** (`close_everything`), as
  `h5py`'s `File.close()` does; a view used afterwards raises `MEDH5FileError`.
- **`amend` is copy-on-write** and replaces the file, so anything holding an open
  handle across it keeps reading the old inode. A rewrite in place closes its
  source before the rename (Windows cannot replace an open file).

## Linting & style

- **ruff** is pinned in the `dev` extra and in `.pre-commit-config.yaml` (bump
  both). **rustfmt** (`rustfmt.toml`) and **clippy** with `-D warnings` for Rust.

## Testing patterns

- Python tests live in `tests/v1/`; Rust tests sit beside the code
  (`#[cfg(test)]`) and in `crates/medh5/tests/`. Optional Python deps are
  guarded with `pytest.importorskip`.
- Test names cite the clause they hold: `test_S8_1_boxes_shift_by_half_a_voxel`,
  `s14_4_the_source_is_closed_before_the_replace`.
- Fixtures are built by the **public writer**, so every reader test is also a
  writer test. Tests plant defects with `h5py` (a test dependency only).
- Engine internals with no Python door (slab budgets, temporary names, the
  rename order) are held by Rust tests; the Python test keeps the observable
  outcome.
- **A skipped test is not a passing test.** Anything guarded by
  `pytest.importorskip` needs a CI job that installs the dependency, or it
  reports coverage it does not have.

## Documentation

`docs/` is the MkDocs site: `tutorials/`, `guides/` (how-to), `reference/`,
`explanation/`, `spec/`, and `examples/` (standalone scripts). `CHANGELOG.md`
stays at the root and `hooks/mkdocs_hooks.py` pulls it in; the same hook renders
the diagnostic-code, cohort-code and schema tables from their sources (edit
`crates/medh5/data/codes.json`, the `CHECK_CODES` table in
`crates/medh5/src/dataset/check.rs`, or the schema — never the page) and emits
redirects for moved pages (`REDIRECTS`). There are no separate design records:
the reasoning lives in `docs/explanation/design-rationale.md`.

The docs are tested, not proofread:

- `tests/v1/test_docs_python.py` executes every `python` block against a real
  sample. A block that cannot run gets `<!-- illustrative -->` on the line above
  its fence; fewer than half may.
- `tests/v1/test_docs_examples.py` checks every documented `medh5 …` flag
  against the parser, and keeps `STALE_CLAIMS` — statements once wrong in the
  docs that must not come back.
- `TestStatedCounts` in `tests/v1/test_conformance.py` compares every stated
  corpus size and breakdown with the corpus.
- The engine crate's README is its crate documentation, so its Rust example is a
  doctest; `docs/reference/rust.md` shows the same block (a test compares them).

`CONTRIBUTING.md` is the human-facing version of this file.
