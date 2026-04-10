# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - Unreleased

### Added

- **NIfTI converter** (`medh5.io.nifti`): `from_nifti()` and `to_nifti()` for
  round-trip conversion between NIfTI and `.medh5`. Automatically extracts
  spacing, origin, direction, and coordinate system from the NIfTI affine.
  Requires optional `nibabel` dependency (`pip install medh5[nifti]`).
- **DICOM converter** (`medh5.io.dicom`): `from_dicom()` ingests a DICOM
  series directory into `.medh5`, extracting spatial metadata from standard
  tags and storing selected DICOM attributes under `extra["dicom"]`. Requires
  optional `pydicom` dependency (`pip install medh5[dicom]`).
- **Dataset manifest** (`medh5.dataset`): `Dataset.from_directory()` scans a
  directory tree for `.medh5` files and builds a lightweight manifest (no
  array reads). Supports `filter()`, `save()`/`load()` (JSON), and staleness
  detection via file mtime/size.
- **Dataset splitting** (`medh5.dataset.make_splits`): reproducible
  train/val/test splitting with stratification (`stratify_by`), patient-level
  grouping (`group_by` with dotted-path support into `extra`), and k-fold
  cross-validation.
- **Dataset statistics** (`medh5.stats.compute_stats`): streaming
  per-modality mean, std, min, max, and percentiles (p01/p99) using Welford
  merge across files. Supports foreground-restricted stats via a named
  segmentation mask, label distribution counts, shape histograms, and
  segmentation coverage fractions. Multi-process via `ProcessPoolExecutor`.
- **Patch sampler** (`medh5.sampling.PatchSampler`): lazy, chunk-aligned
  patch extraction with three strategies: `uniform`, `foreground` (biased
  toward a named seg mask), and `balanced` (alternating). Caches foreground
  voxel coordinates per file for efficiency.
- **Pure-numpy transforms** (`medh5.transforms`): `Compose`, `Clip`,
  `Normalize`, `ZScore`, and `RandomFlip`. No torch or PIL dependency.
- **Patch-based PyTorch dataset** (`medh5.torch.MEDH5PatchDataset`): uses
  `PatchSampler` for lazy patch reads instead of full-volume eager loads.
  Configurable `samples_per_volume` for virtual dataset length.
- **Per-worker file handle cache** (`medh5.torch._HandleCache`): LRU cache
  (default 32 handles) shared by both `MEDH5TorchDataset` and
  `MEDH5PatchDataset`. Each DataLoader worker gets its own cache (forked
  process). Eliminates redundant `h5py.File()` opens across epochs.
- **Review/QA workflow**: `MEDH5File.set_review_status()` and
  `MEDH5File.get_review_status()` for tracking annotation review state
  (`pending`/`reviewed`/`flagged`/`rejected`), annotator, timestamp, and
  notes. Prior states are appended to a `history` list. Stored under
  `extra["review"]` (no schema change). `ReviewStatus` dataclass exported
  from `medh5`.
- **Batch CLI commands**:
  - `medh5 validate-all <dir>` — parallel validation of all `.medh5` files.
  - `medh5 audit <dir>` — parallel SHA-256 checksum verification.
  - `medh5 recompress <dir|file> --compression <preset>` — rewrite files with
    a different compression preset. Supports `--out-dir` or atomic in-place
    rewrite via tempfile + rename. Optional `--checksum` flag.
- **Dataset CLI commands**:
  - `medh5 index <dir> -o manifest.json` — build a manifest.
  - `medh5 split <manifest> --ratios 0.7,0.15,0.15 -o splits/` — split with
    optional `--stratify`, `--group`, `--k-folds`, `--seed`.
  - `medh5 stats <dir|manifest> -o stats.json` — compute dataset statistics.
- **Import/export CLI commands**:
  - `medh5 import nifti --image <name> <path> -o out.medh5`
  - `medh5 import dicom <dir> -o out.medh5`
  - `medh5 export nifti <file> -o <dir>`
- **Review CLI commands**:
  - `medh5 review set <file> --status <status> --annotator <name>`
  - `medh5 review get <file>`
  - `medh5 review list <dir> --status <status>`
  - `medh5 review import-seg <file> --name <mask> --from <nifti>`
- **Optional dependency extras** in `pyproject.toml`: `nifti`, `dicom`, `itk`.
- **Tests**: expanded from 62 to 135 tests (91% coverage).

## [0.2.0]

### Breaking Changes

- **Multi-modality images**: The `image` parameter in `MEDH5File.write()` is
  replaced by `images: dict[str, np.ndarray]`.  Each key is a modality name
  (e.g. `"CT"`, `"MRI_T1"`, `"PET"`).  All arrays must share the same shape.
- **On-disk layout**: Image data is stored under an `images/` HDF5 group
  instead of a top-level `image` dataset.
- **`MEDH5Sample.image`** is replaced by `MEDH5Sample.images` (a dict).
- **Schema version** remains `"1"` for the current multi-image layout.
- `SampleMeta` gains `image_names: list[str]`.

### Added

- **Compression presets**: `compression="fast"`, `"balanced"`, or `"max"` as
  a shorthand for `cname`/`clevel` pairs.
- **Context-manager protocol**: `MEDH5File` is now instantiable and supports
  `with MEDH5File("file.medh5") as f:` for typed lazy access via `f.images`,
  `f.seg`, `f.meta`.
- **Custom exceptions**: `MEDH5Error`, `MEDH5ValidationError`,
  `MEDH5FileError`, `MEDH5SchemaError`.
- **Write-time validation**: seg shape vs image shape, bbox count vs
  scores/labels, bboxes shape, clevel range, empty images dict.
- **Schema version checking**: reading a file with a future schema version
  raises `MEDH5SchemaError`.
- **`MEDH5File.update_meta()`**: update label, label_name, or extra metadata
  without rewriting arrays.
- **`MEDH5File.add_seg()`**: add a segmentation mask to an existing file.
- **`MEDH5File.verify()`**: verify SHA-256 checksum of image data.
- **`checksum=True`** parameter on `write()` to store a SHA-256 digest.
- **CLI**: `medh5 info <file>` and `medh5 validate <file>` commands.
- **PyTorch integration**: `MEDH5TorchDataset` in `medh5.torch` (optional
  dependency via `pip install medh5[torch]`).
- **`__repr__`** for `MEDH5Sample` and `SampleMeta`.
- **`py.typed`** marker for downstream type checking.
- **Chunk optimizer**: named `_CHUNK_OVERSHOOT_LIMIT` constant, optional
  L3 cache auto-detection.
- **CI**: GitHub Actions workflow (lint, typecheck, test on Python 3.10-3.12).
- **Tooling**: ruff linting/formatting, pre-commit hooks.
- **Tests**: expanded from 12 to 62 tests with pytest-cov.

### Fixed

- Removed unused `from copy import deepcopy` import in `chunks.py`.
- Malformed `direction` attribute now emits a warning instead of crashing.

## [0.1.0]

Initial release with single-image `.medh5` format, HDF5 + Blosc2 compression,
segmentation masks, bounding boxes, labels, spatial metadata, and chunk
optimization.
