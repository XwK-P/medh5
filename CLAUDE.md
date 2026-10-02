# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install with the extras the test suite needs
pip install -e ".[dev,torch,nifti,dicom,dicomseg,itk,interp]"

# Full suite (90% coverage floor)
pytest tests/ --cov=medh5 --cov-report=term-missing --cov-fail-under=90

# A single file or test
pytest tests/v1/test_sample.py -v
pytest tests/v1/test_dataset.py::TestSplits -v

# Lint, format, types
ruff check . && ruff format --check . && mypy medh5

# The conformance corpus must stay green
medh5 conformance run /tmp/corpus

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
ruff check . && ruff format --check . && mypy medh5 \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus \
  && mkdocs build --strict
```

## The model

**Format 1.0.** A `.medh5` file is **one subject at one or more timepoints**,
with every image, annotation, transform and curation record about them. Not one
scan — one subject. Most of the design follows from that: splitting by file is
subject-safe, a change annotation has a referent, and registration between
visits is an object in the file rather than a convention between filenames.

`docs/spec/medh5-1.0.md` is **normative**. Code implements it; when they
disagree, one of them is a bug. Appendix C records the clauses corrected
because implementing them showed the text was not implementable.

## Architecture

Sub-packages map onto specification sections, so a spec change has one obvious
home.

- **`errors.py`** holds the §15.2 diagnostic code table; a test asserts the
  table and the spec agree.
- **`sampling.py`** depends on no deep-learning framework, because where to read
  is geometry.
- **`conformance/`** — the corpus is a *shipped artifact*, not a test fixture:
  third-party implementations run it.

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
- **Geometry is never invented.** Converters refuse rather than resample, guess a
  grid, or fabricate a transform. `transform_between` returns `None` when no path
  exists.
- **HDF5 handles must not cross `fork`.** The torch handle cache is PID-keyed and
  a forked child *abandons* the parent's handles rather than closing them.
- **`amend` is copy-on-write** and replaces the file, so anything holding an open
  handle across it keeps reading the old inode.

## Linting & style

- **ruff** is pinned in the `dev` extra and in `.pre-commit-config.yaml` (bump
  both).

## Testing patterns

- Tests live in `tests/v1/`. Optional deps are guarded with
  `pytest.importorskip`.
- Test names cite the clause they hold: `test_S8_1_boxes_shift_by_half_a_voxel`.
- Fixtures are built by the **public writer**, so every reader test is also a
  writer test.
- **A skipped test is not a passing test.** Anything guarded by
  `pytest.importorskip` needs a CI job that installs the dependency, or it
  reports coverage it does not have.

## Documentation

`docs/` is the MkDocs site: `tutorials/`, `guides/` (how-to), `reference/`,
`explanation/`, `spec/`, and `examples/` (standalone scripts). `CHANGELOG.md`
stays at the root and `hooks/mkdocs_hooks.py` pulls it in; the same hook renders
the diagnostic-code, cohort-code and schema tables from their sources (edit
`errors.py`, `dataset/check.py` or the schema, never the page) and emits
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

`CONTRIBUTING.md` is the human-facing version of this file.
