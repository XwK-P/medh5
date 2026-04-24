# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.5.0]

First PyPI release. Bundles the 0.4.0 work (never released) with a
dedicated release-hardening pass covering data-safety, PyTorch
multiprocessing, spatial-metadata validation, statistics numerics, CLI
exit codes, packaging, and adds the nnU-Net v2 dataset converter and a
post-review refactor round that split the CLI into a package and
consolidated duplicated helpers.

### Added

- **Atomic writes**: `MEDH5File.write()` now writes to a sibling temp file,
  `fsync`s, and `os.replace`s into place. An interrupted write (Ctrl-C,
  OOM, crash) can no longer leave a truncated `.medh5` file at the
  destination path. Any pre-existing file at the target path is preserved
  on failure.
- **Checksum verification before in-place updates**: `MEDH5File.update()`
  (and by extension `update_meta`, `add_seg`, `set_review_status`) now
  verifies any stored SHA-256 *before* mutating, so an externally
  corrupted file cannot silently have a fresh checksum baked in over top
  of the corruption. New `force=True` escape hatch for intentional
  repairs.
- **Fork/spawn-safe PyTorch handle cache**: `medh5.torch._HandleCache` is
  now PID-scoped — a forked worker observes the PID mismatch and resets
  to a cold cache instead of inheriting parent h5py state. Works
  transparently with `multiprocessing_context="spawn"` (default on macOS
  / Windows / Python 3.14+).
- **`medh5.torch.worker_init_fn`**: the supported `DataLoader(
  worker_init_fn=…)` helper for `num_workers > 0`. Documented in
  README.
- **`PatchSampler(include_bboxes=True)`**: opt-in bbox return from
  `PatchSampler.sample()`. Bboxes are translated into patch-local
  coordinates and filtered to the ones intersecting the patch;
  `bbox_scores` / `bbox_labels` are filtered consistently.
- **`RandomFlip` geometry sync**: flipping now negates the corresponding
  column of `meta.spatial.direction` (via `dataclasses.replace`, so the
  file's cached `SampleMeta` is not mutated) and mirrors any bboxes in
  the sample dict, keeping physical-space metadata consistent with the
  flipped voxel data.
- **`MEDH5File.is_valid(path)`**: thin convenience wrapper returning a
  plain `bool` for the common "is this file OK?" check (swallows
  `MEDH5ValidationError`).
- **Dimension checks in `SampleMeta.validate()`**: `direction` must be
  `ndim × ndim` and `axis_labels` length must equal `ndim`. A malformed
  `direction` attribute on read now raises `MEDH5SchemaError` instead of
  emitting a warning.
- **Numerically-stable parallel stats**: `compute_stats` now accumulates
  per-file `(n, mean, M2)` via Welford and merges with Chan's parallel
  algorithm. Large uint16 CT volumes no longer suffer catastrophic
  cancellation on variance.
- **CLI exit codes**: `medh5 <no args>` and unknown subcommands return
  exit code 2; runtime errors (`MEDH5Error`, `ValueError`, `ImportError`)
  return 1; success returns 0. Replaced the `if cmd == …` ladder with a
  typed dispatch table (`_TOP_HANDLERS`, `_SUB_DISPATCH`).
- **macOS CI job**: `test-macos` on `macos-latest` + Python 3.12
  exercises the `spawn` multiprocessing path that the Linux matrix does
  not cover.
- **Release-build CI job**: runs `python -m build`, `twine check dist/*`,
  inspects the wheel for `medh5/py.typed` + `LICENSE`, and uploads the
  dist/ artifact.
- **PyPI packaging metadata**: authors, project URLs (Homepage,
  Repository, Issues, Changelog), classifiers (Development Status ::
  4 - Beta, Topic :: Scientific/Engineering :: Medical Science Apps.,
  Typing :: Typed), `package-data = {medh5 = ["py.typed"]}`,
  `license = {file = "LICENSE"}`. `LICENSE` file (MIT, Puyang Wang,
  2026) added to the repo root and bundled in both wheel and sdist.
- **Tightened lower bounds**: `h5py >= 3.10`, `hdf5plugin >= 4.1`,
  `numpy >= 1.24`. No upper bounds.
- **nnU-Net v2 dataset converters** (`medh5.io.nnunetv2`): `from_nnunetv2()`
  converts a raw nnU-Net v2 dataset folder (`imagesTr/`, `labelsTr/`,
  optional `imagesTs/`, `dataset.json`) into a directory of per-case
  `.medh5` files, bundling every channel and splitting the integer label
  volume into one boolean mask per foreground class declared in
  `dataset.json`. `to_nnunetv2()` is the reverse: it emits a raw nnU-Net
  v2 layout from a directory of `.medh5` files. The parsed `dataset.json`
  payload is stashed in each file's `extra["nnunetv2"]` so export is
  lossless — channel order, label integer values, and optional fields
  (`overwrite_image_reader_writer`, `regions_class_order`, `name`) all
  round-trip. Region-based (list-valued) labels are rejected with a clear
  error. Requires the `nifti` extra. Lazy-imported from `medh5.io`.
- **CLI nnU-Net v2 subcommands**: `medh5 import nnunetv2 <src> -o <dst>`
  and `medh5 export nnunetv2 <src> -o <dst>` with `--no-test`,
  `--compression`, `--checksum`, `--dataset-name`, and `--file-ending`
  flags.
- **`MEDH5File.is_valid(strict=...)`**: `is_valid()` now forwards a
  `strict` kwarg to `ValidationReport.ok()`, so callers that want the
  one-call "did this file pass cleanly, warnings included?" check can
  get it without building a report object themselves.
- **Deterministic `stats.compute_stats` sampling**: per-file percentile
  sample seeds now derive from a stable BLAKE2b digest of the file path
  instead of Python's hash-randomized `hash()`, so percentile estimates
  are reproducible across runs and across Python invocations.
- **Tests**: expanded to 217 passing (91% coverage), including
  `test_dataloader_workers[spawn]`, `test_patch_dataloader_spawn`,
  `test_interrupted_write_*`, `test_update_verifies_checksum`,
  `test_include_bboxes_*`, `test_randomflip_direction_sync`,
  `test_compute_stats_parallel_matches_serial`, `test_is_valid_*`,
  CLI exit-code tests, end-to-end `medh5 import dicom` CLI coverage,
  `TestFromNnunetv2`/`TestToNnunetv2` happy-path and silent-data-loss
  guards, and a `medh5 import/export nnunetv2` CLI round-trip test.

### Changed

- `MEDH5File.read()` returns `sample.seg = None` when the `seg/` group
  exists but is empty, and `read_meta()` reports `has_seg = False` in
  the same case — previously both could be inconsistent with file
  state.
- Bounding-box datasets are only Blosc2-compressed when `n > 64`;
  tiny bbox arrays are written raw to avoid per-chunk filter overhead.
- **`MEDH5File.validate()` no longer takes `strict`**: strictness is
  applied on the returned `ValidationReport` via `report.ok(strict=...)`,
  keeping the report layer policy-free. The one-call `is_valid()`
  shortcut accepts `strict` as described above.
- **CLI split into `medh5.cli` package**: the 819-line flat
  `medh5/cli.py` is now a package grouped by command —
  `cli/inspect.py` (`info`/`validate`/`validate-all`/`audit`/`recompress`),
  `cli/dataset.py` (`index`/`split`/`stats`), `cli/convert.py`
  (`import`/`export` subgroups), `cli/review.py` (`review set`/`get`/
  `list`/`import-seg`), and `cli/_common.py` for shared helpers. Each
  submodule exposes `register(sub)` and `dispatch(cmd, args) -> int | None`;
  `cli/__init__.py::main()` composes them. Public surface
  (`medh5.cli:main`, `python -m medh5.cli`) is unchanged.
- **`.medh5` suffix helper consolidated**: the duplicate
  `_validate_suffix` / `_SUFFIX` pair in `core.py` and `review.py` was
  hoisted into `medh5.meta` and re-used from both modules.

### Fixed

- `MEDH5File.write()` no longer leaves partial output when interrupted
  mid-write (see "Atomic writes" above).
- `MEDH5File.update()` no longer silently re-hashes corrupted data
  (see "Checksum verification" above).
- `MEDH5PatchDataset` + `DataLoader(num_workers > 0)` no longer
  deadlocks under `fork` or crashes pickling under `spawn`.
- `RandomFlip` no longer silently desynchronizes `meta.spatial.direction`
  from the flipped voxel grid; downstream NIfTI export and
  physical-space metrics now see consistent geometry.
- `compute_stats(workers > 1)` no longer suffers precision loss on
  large integer volumes.
- `medh5 <no args>` now returns exit code 2 instead of 0, unbreaking
  shell automation like `medh5 validate … || exit 1`.
- nnU-Net v2 import no longer silently drops voxels whose integer label
  is not declared in `dataset.json` — `_split_label_volume` raises
  `MEDH5ValidationError` listing the offending values, and rejects
  float label volumes that contain genuinely non-integer voxels while
  still accepting integer-valued floats (`0.0`, `1.0`, …).
- nnU-Net v2 export no longer silently drops seg masks whose names are
  not declared in the nnU-Net label map when merging back to an integer
  label volume; it raises `MEDH5ValidationError` and asks the caller to
  update `extra["nnunetv2"]["labels"]` or remove the extra mask.
- nnU-Net v2 export no longer silently omits per-file image channels
  that disagree with the dataset-wide channel set resolved from the
  first file's metadata; channel mismatches raise
  `MEDH5ValidationError` with a clear missing/extra report.

## [0.4.0]

Bundled into 0.5.0 — never released on PyPI. Entries below describe
work landed under the 0.4 development branch.

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
