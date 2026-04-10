# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - Unreleased

### Added

- **Structured validation** (`ValidationReport`, `ValidationIssue`): `MEDH5File.validate()`
  returns a report with typed error/warning codes instead of plain strings.
  Supports `strict` mode where warnings are treated as failures. `ValidationReport`
  is exported from `medh5`.
- **Unified update API** (`MEDH5File.update()`): single entry point for in-place
  metadata, segmentation (add/replace/remove), and bounding-box mutations.
  Automatically resyncs `image_names`, `shape`, `has_seg`, `seg_names`, `has_bbox`
  from file state and recomputes checksums when present.
- **DICOM series selection**: `from_dicom()` now accepts `series_uid` to select
  a specific series when multiple exist. Without it, the largest series is chosen
  deterministically. Available series UIDs are recorded in `extra["dicom"]`.
- **DICOM geometry validation**: strict checks for consistent
  `ImageOrientationPatient`, `PixelSpacing`, and uniform slice spacing across
  the selected series. Multi-frame and non-grayscale DICOM are rejected with
  clear errors.
- **DICOM modality LUT**: `apply_modality_lut` parameter (default `True`) applies
  RescaleSlope/RescaleIntercept before writing via `pydicom.pixels`.
  Disable with `apply_modality_lut=False` or `--no-modality-lut` on the CLI.
- **SimpleITK resampling** for NIfTI imports: `from_nifti(resample_to=...)` resamples
  all images and masks onto a shared reference grid. Supports `"linear"`,
  `"nearest"`, and `"bspline"` interpolators. Masks always use nearest-neighbor.
- **`import_seg_nifti()`** (`medh5.io`): import a NIfTI segmentation mask into
  an existing `.medh5` file with optional resampling and replace semantics.
- **Expanded checksum coverage**: SHA-256 now covers segmentation masks, bounding
  boxes, and critical metadata attributes — not just image datasets. Review status
  updates also recompute the checksum when one is stored.
- **JSON output on CLI**: `--json` flag on `info`, `validate`, `stats`, and
  `review get` commands for machine-readable output.
- **CLI flags**: `--strict` on `validate`, `--fail-fast` on `validate-all`,
  `--resample-to`/`--interpolator` on `import nifti`, `--series-uid`/`--no-modality-lut`
  on `import dicom`, `--resample`/`--replace` on `review import-seg`.
- **Dataset record fields**: `DatasetRecord` now includes `shape`, `spacing`,
  `coord_system`, `patch_size`, and `review_status`.
- **Metadata validation**: `SampleMeta.validate()` now checks `patch_size`
  length and element types.
- **`meta.py` attribute lists**: `_ROOT_META_ATTRS` and `_IMAGE_META_ATTRS`
  tuples canonically define which HDF5 attributes belong to the schema.
  `write_meta()` clears stale attributes before writing.

### Changed

- `MEDH5File.update_meta()` now delegates to `MEDH5File.update()` internally.
- `MEDH5File.add_seg()` now delegates to `MEDH5File.update()` internally.
- `_validate_file()` in `cli.py` replaced by `MEDH5File.validate()`.
- DICOM `_read_series()` returns provenance metadata (selected UID, available
  UIDs, instance count, LUT application status).
- `from_dicom()` now raises on missing `ImageOrientationPatient`,
  `ImagePositionPatient`, or `PixelSpacing` instead of falling back to defaults.
- CLI `main()` wraps all command handlers in a top-level
  `except (ImportError, MEDH5Error, ValueError)` for consistent error reporting.
- **Tests**: expanded from 135 to 167 tests (90% coverage).

### Fixed

- `ValidationPayload` type alias was defined after `if __name__ == "__main____"`
  in `cli.py`, making it unreachable during normal imports. Moved to module top.
- `_build_info_payload()` opened the file twice (once via `MEDH5File` context
  manager, once via `get_review_status()`). Now extracts review status from
  `meta.extra` inline.
- `_validate_open_file()` loaded the entire `bboxes` dataset into memory just
  to check its shape. Now reads only HDF5 dataset metadata.
- Duplicate attribute-name tuples in `integrity.py` (`_HASHED_ROOT_ATTRS`,
  `_HASHED_IMAGE_ATTRS`) now reuse the canonical tuples from `meta.py`.

## [0.3.0]

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
