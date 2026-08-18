# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install with the extras the test suite needs
pip install -e ".[dev,schema,torch,nifti,dicom,dicomseg,itk,interp]"

# Full suite (90% coverage floor)
pytest tests/ --cov=medh5 --cov-report=term-missing --cov-fail-under=90

# A single file or test
pytest tests/v1/test_sample.py -v
pytest tests/v1/test_dataset.py::TestSplits -v

# Lint, format, types
ruff check . && ruff format --check . && mypy medh5

# The conformance corpus must stay green
medh5 conformance run /tmp/corpus
```

## Pre-commit checks

All of these must pass before committing:

```bash
ruff check . && ruff format --check . && mypy medh5 \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus
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

- **`_hdf5.py`** — attribute codecs, identifier rules, atomic create, CoW amend.
- **`sample.py`** — `Sample` (read) and `SampleWriter` (write); `medh5.open`,
  `create`, `amend`. The central API.
- **`document.py`** — the `/meta` sample document and its JSON Schema.
- **`image.py`** — lazy reads, rescale, pyramid levels.
- **`errors.py`** — the exception hierarchy and the §15.2 diagnostic code table.
  A test asserts the table and the spec agree.
- **`geometry/`** (§3) — `Grid`, index↔world affines, the half-voxel box rules,
  multiscale derivation.
- **`labels/`** (§5) — `LabelSet` as a DAG, canonical digests, bundled vocabularies.
- **`annotations/`** (§6–§9) — `base.py` defines the
  `contains`/`dense`/`labelmap`/`instances` contract; `voxel/` holds the five
  encodings plus `select.py` (auto-selection by overlap graph) and
  `transcode.py`; `geometric.py` and `classification.py` hold the rest.
- **`transforms/`** (§10) — affine, displacement, B-spline, composite, plus
  frame-graph resolution in `resolve.py`.
- **`curation/`** (§11–§12) — provenance, quality, agreement, identity, splits,
  tracking, timeline, and `scrub.py` (de-identification).
- **`integrity/`** (§13) — per-object digests, `content_id`, verification, repair.
- **`storage/`** (§14) — codec profiles, chunking, sampling index, recompression.
- **`dataset/`** — cohort tools: manifests, splits, streaming stats, `C1xx` checks.
- **`io/`** — converters, each lazily imported: NIfTI, DICOM, DICOM SEG,
  RTSTRUCT, nnU-Net v2, and `legacy.py` (0.x → 1.0 migration).
- **`torch/`**, **`sampling.py`**, **`monai.py`** — loaders. `sampling.py`
  depends on no deep-learning framework, because where to read is geometry.
- **`conformance/`** — the corpus is a *shipped artifact*, not a test fixture:
  third-party implementations run it.
- **`cli/`** — one module per command group, each exposing `register(sub)` and
  `dispatch(cmd, args)`; `cli/__init__.py::main` composes them.

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

## Codec profiles

`training` (lz4:1), `balanced` (zstd:3, default), `archive` (zstd:9),
`portable` (gzip:4, readable without hdf5plugin). Labels get `bitshuffle` where
images get `shuffle`. Chunks are sized by `optimize_chunks()` from the patch
hint toward an L3-cache budget; stacked encodings chunk per plane so one layer
reads without the others.

## Linting & style

- **ruff** with `E, F, I, UP, B, SIM`, `target-version = "py310"`.
- **mypy --strict**, with `ignore_missing_imports` for h5py, hdf5plugin, torch,
  nibabel, pydicom, SimpleITK, jsonschema, scipy, monai, highdicom.

## Testing patterns

- Tests live in `tests/v1/`. Optional deps are guarded with
  `pytest.importorskip`.
- Test names cite the clause they hold: `test_S8_1_boxes_shift_by_half_a_voxel`.
- Fixtures are built by the **public writer**, so every reader test is also a
  writer test.
- CI matrix: Python 3.10–3.12, plus a macOS job specifically for the `spawn`
  start method, plus a conformance job that publishes the suite and scores this
  validator through the public `score` path.

## 0.x

Deleted at 1.0. `io/_legacy_reader.py` is a read-only reader of the old layout
so `medh5 migrate` works; there is no 0.x writer, deliberately. 0.x files are
converted once, and the migration reports every non-mechanical decision.
