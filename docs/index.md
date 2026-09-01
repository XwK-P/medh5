# medh5

One medical imaging **sample** — a subject, at one or more timepoints, with
every image, annotation, registration and curation record about them — in a
single self-describing HDF5 file.

```python
import medh5

with medh5.open("case_0001.medh5") as s:
    s.identity.subject_id                              # "BRATS-GLI-01234"
    s.at("tp1").images["CT_tp1"].read(physical=True)   # HU, not raw counts
    s.annotations["organs"].dense(["liver", "spleen"]) # any encoding, one API
    s.transform_between("tp0", "tp1")                  # resolved via frames
    s.tracks("lesion")                                 # lesions joined across visits
```

## Start here

| | |
|---|---|
| **[Getting started](getting-started.md)** | Install, write a sample, read it back. |
| **[Concepts](concepts.md)** | The data model in ten minutes: samples, grids, timepoints, coverage. |
| **[Specification](spec/medh5-1.0.md)** | The normative format. Everything else is commentary. |

## Reference

| | |
|---|---|
| **[Python API](python-api.md)** | Reading, writing, amending. |
| **[CLI](cli.md)** | Every `medh5` command. |
| **[Annotations](annotations.md)** | The five voxel encodings, boxes, contours, keypoints, meshes, labels. |
| **[Longitudinal](longitudinal.md)** | Timepoints, registration, change, instance tracking. |
| **[Training](training.md)** | PyTorch datasets, samplers, MONAI, performance. |
| **[Converters](converters.md)** | NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net v2, and 0.x migration. |
| **[Curation](curation.md)** | Provenance, quality, agreement, de-identification, collections. |
| **[Cohorts](cohorts.md)** | Manifests, leakage-free splits, streaming statistics, cross-file checks. |
| **[Storage](file-format.md)** | On-disk layout, codecs, chunking, integrity. |
| **[Conformance](conformance.md)** | The suite, and how to run it against your own implementation. |

## What the format is for

**One file per subject, not per scan.** A sample is a *subject*, and a subject
has visits. Longitudinal work — change detection, response assessment, lesion
tracking, follow-up registration — lives inside one file, which also means
assigning whole files to train and test cannot leak a patient between them.

**Geometry is stated once and never guessed.** Every array is bound to a
declared grid with spacing, origin and direction; a box is at voxel edges and a
voxel index is a voxel centre, both written down. Converting to NIfTI or DICOM
moves numbers between conventions explicitly, and refuses when it cannot.

**Absence is not silence.** `class_ids` says what an annotation contains;
`annotated_class_ids` says what was *looked for*. A class searched for and not
found is recorded as searched for and not found, which is a different training
signal from a class nobody examined.

**Every claim is checkable.** Per-object SHA-256 over decompressed content and
a Merkle `content_id` that survives recompression; a validator with a stable
diagnostic-code table; and a 103-case conformance corpus, one case per code,
that any implementation can run.

**Reading a patch is fast.** A 64³ multi-class patch reads in ~4 ms, against
117 ms measured on 0.x, because chunks are sized for it and the sampling index
makes foreground sampling O(1) in the volume — 0.09 ms at 1 Mvox and at 20
Mvox. The index is written by `build_index()` and is not automatic: without one
the same draw scans the labels and costs 1.4 ms and 21 ms. Run `medh5 bench` on
your own hardware.

## The command line

```
$ medh5 info case_0001.medh5             # grids, images, annotations, coverage
$ medh5 validate case_0001.medh5 --level strict
$ medh5 track case_0001.medh5            # per-lesion volumes across visits
$ medh5 dataset index studies/ -o cohort.json
$ medh5 dataset split cohort.json --group-by group_id --stratify-by site_id
$ medh5 convert from-dicom /studies out/ # one sample per patient, all visits
$ medh5 scrub out/*.medh5 --apply --date-shift-days -117
$ medh5 bench                            # reproduce the performance targets
```

## Going deeper

- **[Runnable examples](examples/index.md)** — a complete two-timepoint sample
  written by following the specification literally, and the benchmark scripts
  behind every number quoted here.
- **[Sample document schema](reference/schema.md)** — every field of `/meta`,
  with the machine-readable schema itself.
- **[Design records](https://github.com/XwK-P/medh5/tree/main/design)** — how
  1.0 was arrived at: what broke in 0.6.0, the alternatives weighed, the plan
  delivered against. Historical, and kept in the repository rather than here.

## Versioning

The **format** is 1.0 and versioned by `medh5_version`. A minor version may add
optional objects, profiles, encodings and diagnostic codes; it may not change
what an existing one means. See spec §16.

The **package** follows semantic versioning from 1.0.0. `medh5.__version__` is
the package; `medh5.FORMAT_VERSION` is the format.

0.x files are not readable by 1.0 and are not meant to be: `medh5 migrate`
converts them once, reporting every decision it took. See
[Converters](converters.md#migrating-from-0x).
