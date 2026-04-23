# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a WIP project that has not been released yet — backward compatibility is not needed.

## Build & Test Commands

```bash
# Install with all extras (dev + optional deps)
pip install -e ".[dev,torch,nifti,dicom]"

# Run full test suite (90% coverage required)
pytest tests/ --cov=medh5 --cov-report=term-missing --cov-fail-under=90

# Run a single test file or test
pytest tests/test_core.py -v
pytest tests/test_core.py::TestMEDH5FileWrite::test_basic_write -v

# Lint and format
ruff check .
ruff format --check .
ruff format .          # auto-fix formatting

# Type checking (strict mode)
mypy medh5
```

## Architecture

**Single-file-per-sample HDF5 format** with Blosc2 compression. Each `.medh5` file stores one patient/scan with co-registered multi-modality images, segmentation masks, bounding boxes, labels, and spatial metadata as HDF5 attributes.

### Core module relationships

- **`core.py`** — `MEDH5File` is the central API. Dual interface: static methods (`write`/`read`/`read_meta`/`verify`/`validate`/`update`/`update_meta`/`add_seg`) for one-shot operations, and context-manager instance (`with MEDH5File(path) as f`) for lazy typed access via `f.images`, `f.seg`, `f.meta`. Also defines `ValidationReport`/`ValidationIssue` for structured validation results. `update()` is the unified entry point for in-place mutations (metadata, seg add/replace/remove, bbox); `update_meta()` and `add_seg()` delegate to it.
- **`chunks.py`** — L3 cache-aware chunk optimizer (from DKFZ mlarray). Called during `write()` to size HDF5 chunks for efficient patch reads.
- **`meta.py`** — `SampleMeta` and `SpatialMeta` dataclasses that serialize to/from HDF5 attributes. Root attrs hold scalar/label/flag metadata; `shape` and spatial metadata are stored on the `images` group. Schema version is `"1"`. Exports `_ROOT_META_ATTRS` and `_IMAGE_META_ATTRS` tuples that canonically list which HDF5 attributes belong to the schema (also used by `integrity.py` for checksum hashing).
- **`integrity.py`** — SHA-256 checksum computation/verification over image datasets, segmentation masks, bounding boxes, and critical metadata attributes. Reuses attribute-name tuples from `meta.py`.
- **`torch.py`** — Two PyTorch datasets: `MEDH5TorchDataset` (eager full-volume) and `MEDH5PatchDataset` (lazy patch-based via `PatchSampler`). Both share a module-level `_HandleCache` (LRU, 32 handles) so repeated `__getitem__` calls reuse open h5py files.
- **`sampling.py`** — `PatchSampler` with uniform/foreground/balanced strategies. Works with open `MEDH5File` instances, slicing lazily from h5py datasets.
- **`transforms.py`** — Pure-numpy transforms (`Compose`, `Clip`, `Normalize`, `ZScore`, `RandomFlip`). Operate on sample dicts with `images` and `seg` keys.
- **`stats.py`** — Streaming dataset statistics via Welford merge. Multi-process via `ProcessPoolExecutor`.
- **`dataset/`** — `Dataset` (metadata-only manifest from `read_meta`, JSON persistence, staleness detection) and `make_splits` (stratified/grouped/k-fold splitting).
- **`io/`** — NIfTI round-trip (`from_nifti`/`to_nifti`), `import_seg_nifti` for adding NIfTI masks to existing files, DICOM ingestion (`from_dicom`) with series selection, geometry validation, and modality LUT support, and nnU-Net v2 dataset round-trip (`from_nnunetv2`/`to_nnunetv2`) that bundles each case's channels + per-class seg masks into one `.medh5` file and stashes the parsed `dataset.json` in `extra["nnunetv2"]` for lossless export. Optional SimpleITK resampling for multi-resolution data. Lazy-imported via `__getattr__` so importing `medh5` doesn't require nibabel/pydicom.
- **`cli/`** — Argparse-based CLI split across submodules by command group: `cli/inspect.py` (info/validate/validate-all/audit/recompress), `cli/dataset.py` (index/split/stats), `cli/convert.py` (import/export subgroups for NIfTI/DICOM/nnU-Net v2), `cli/review.py` (review subgroup). Each submodule exposes `register(sub)` and `dispatch(cmd, args) -> int | None`; `cli/__init__.py::main` composes them. Several commands support `--json` for machine-readable output.

### Compression pipeline

Three presets: `"fast"` (lz4/3), `"balanced"` (lz4hc/8, default), `"max"` (zstd/9). Applied via hdf5plugin Blosc2 filter on `h5py.create_dataset()`. Chunk sizes are computed by `optimize_chunks()` targeting ~1.4 MiB per L3 cache slice.

### Lazy vs eager reads

- **Eager:** `MEDH5File.read()` → loads everything into `MEDH5Sample` dataclass
- **Lazy:** `MEDH5File(path)` context manager → `f.images["CT"][z0:z1, y0:y1, x0:x1]` slices directly from HDF5 chunks
- **Metadata-only:** `MEDH5File.read_meta()` → reads HDF5 attributes without touching arrays

## Pre-commit checks

All CI checks must pass before committing and pushing:

```bash
ruff check . && ruff format --check . && mypy medh5 && pytest tests/ --cov=medh5 --cov-fail-under=90
```

## Linting & Style

- **ruff** with rules `E, F, I, UP, B, SIM` and `target-version = "py310"`
- **mypy --strict** with `ignore_missing_imports` for h5py, hdf5plugin, torch, nibabel, pydicom, SimpleITK
- Both `ruff check` and `ruff format` must pass in CI

## Testing Patterns

- Optional deps guarded with `pytest.importorskip("nibabel")` / `pytest.importorskip("torch")`
- CI matrix: Python 3.10, 3.11, 3.12 with `[dev,torch,nifti,dicom]` extras installed
- Exception hierarchy: `MEDH5Error` → `MEDH5FileError`, `MEDH5SchemaError`, `MEDH5ValidationError`
