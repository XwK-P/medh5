# medh5 documentation

`medh5` stores one medical-imaging sample — multi-modality images, segmentation
masks, bounding boxes, image-level label, and spatial metadata — in a single
HDF5 + Blosc2 file optimized for patch-based ML training.

This documentation covers the on-disk format, Python and CLI APIs, converters
for NIfTI/DICOM/nnU-Net v2, PyTorch integration, dataset manifests and
statistics, and the review/QA workflow.

## Contents

- **[Getting started](getting-started.md)** — install and write your first `.medh5` file.
- **[File format](file-format.md)** — on-disk HDF5 layout, metadata schema, compression presets, chunk optimizer.
- **[Python API](python-api.md)** — `MEDH5File` (read / write / validate / verify / update), metadata, exceptions.
- **[CLI reference](cli.md)** — every `medh5 ...` subcommand, flags, exit codes, JSON output.
- **[PyTorch integration](pytorch.md)** — eager and patch-based datasets, sampling strategies, transforms, fork/spawn-safe `DataLoader` setup.
- **[Converters](converters.md)** — NIfTI, DICOM, nnU-Net v2 round-trip, SimpleITK resampling.
- **[Datasets and statistics](dataset-and-stats.md)** — manifest scanning, filtering, reproducible splitting, streaming stats.
- **[Review / QA workflow](review.md)** — tracking annotation review state in-file.

## What makes medh5 different

- **One file per sample.** Each `.medh5` is self-contained — images, masks,
  bboxes, label, and spatial metadata live in one HDF5 file. No sidecar JSON,
  no coupling to a dataset-wide schema.
- **Plain HDF5.** Inspectable with `h5ls` / HDFView / h5py / MATLAB / Julia.
- **Atomic writes, checksums, structured validation.** Writes go to a sibling
  temp file and are `os.replace`'d into place; SHA-256 covers images, seg
  masks, bboxes, and metadata; `validate()` returns typed issues, not strings.
- **Fork/spawn-safe PyTorch datasets.** The per-worker handle cache is
  PID-scoped, with a supported `worker_init_fn` for `num_workers > 0`.
- **No multi-sample lock-in.** You can keep the files on disk, move them
  between machines, or re-group them without re-encoding arrays.

## Versioning

The file format carries a `schema_version` attribute (currently `"1"`). See
[File format](file-format.md) for details and forward-compatibility rules.

The Python package follows semver from 1.0 onwards. During 0.x, minor
versions may break API (see [CHANGELOG](../CHANGELOG.md)).
