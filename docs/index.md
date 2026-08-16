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

## Format v1.0 — in development

A clean-slate redesign of the format. It is **not** backward compatible with 0.x and targets
classification, detection, segmentation and registration workflows at scale.

In 1.0 a **sample is one subject at one or more timepoints**, each with one or more images — so
longitudinal work (change detection, response assessment, lesion tracking, follow-up registration)
lives inside one file, and assigning whole files to train/val/test cannot leak a patient.

**Implemented so far** (plan phases 0–3): the container and its geometry, timepoints, images and
multiscale pyramids, label sets, all five voxel-annotation encodings with lossless transcoding between
them, every geometric annotation (boxes, oriented boxes, keypoints, landmarks, contours, meshes),
classification including change labels across timepoints, per-object digests and content addressing,
the sampling index, the validator, an 87-case conformance corpus, and a CLI.

```python
import medh5

with medh5.open("case_0001.medh5") as s:
    s.at("tp1").images["CT_tp1"].read(physical=True)      # HU, not raw counts
    s.annotations["organs"].dense(["liver", "spleen"])    # any encoding, one API
    s.annotations["lesions"].as_slices()                  # boxes -> numpy slices
    s.annotations["response"].labels                      # change label across visits
    s.track(class_key="lesion")                           # instance ids across visits
```

```
$ medh5 info case_0001.medh5      # grids, images, annotations, coverage, codecs
$ medh5 validate case_0001.medh5 --level strict
$ medh5 seg convert case_0001.medh5 organs --to bitmask   # lossless re-encoding
```

**Not yet implemented:** transforms (§10), collections (§2.2), converters, and the PyTorch/MONAI
loaders. Until those land, `medh5.legacy` holds the 0.6.0 implementation unchanged — the docs below
describe it, and its import paths are `medh5.legacy.*`.

- **[Specification (v1.0)](spec/medh5-1.0.md)** — normative on-disk schema: grids, geometry and
  timepoints, label sets, the voxel-annotation encodings, geometric annotations, transforms,
  provenance, integrity, conformance profiles and the diagnostic-code table.
- **[Design proposal](design/medh5-1.0-proposal.md)** — what breaks in 0.6.0 (with measurements),
  design principles, alternatives considered, benchmark results, costs, risks, decisions taken.
- **[Implementation plan](design/medh5-1.0-implementation-plan.md)** — package layout, public API,
  phased delivery, test strategy, CLI, migration from 0.x.
- **[JSON Schema](../schemas/medh5-sample-1.0.schema.json)** — machine validation of the `/meta`
  document; the package ships an identical copy and a test asserts the two match.
- **[Benchmarks and reference prototype](design/benchmarks/README.md)** — reproducible scripts behind
  every number in the proposal.

The documents below describe the **0.6.0 format**, now reached through `medh5.legacy`.

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
