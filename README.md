# medh5

[![PyPI version](https://img.shields.io/pypi/v/medh5.svg)](https://pypi.org/project/medh5/)
[![Python versions](https://img.shields.io/pypi/pyversions/medh5.svg)](https://pypi.org/project/medh5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/XwK-P/medh5/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XwK-P/medh5/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](#)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-informational.svg)](medh5/py.typed)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**One medical imaging sample — a subject, at every timepoint, with all of its
ground truth — in a single self-describing HDF5 file.**

Multi-modality images, segmentation in five encodings, detection boxes,
keypoints, contours, meshes, classification, registration between visits,
provenance and quality records, and per-object integrity digests. Format
version **1.0**, with a [normative specification](docs/spec/medh5-1.0.md) and a
[103-case conformance suite](docs/conformance.md) any implementation can run.

```python
import medh5

with medh5.open("case_0001.medh5") as s:
    s.identity.subject_id                              # "BRATS-GLI-01234"
    s.at("tp1").images["CT_tp1"].read(physical=True)   # HU, not raw counts
    s.annotations["organs"].dense(["liver", "spleen"]) # any encoding, one API
    s.transform_between("tp0", "tp1")                  # resolved via frames
    s.tracks("lesion")                                 # lesions joined across visits
```

## Install

```bash
pip install medh5
pip install "medh5[torch,nifti,dicom]"
```

Reading and writing needs only `h5py`, `hdf5plugin` and `numpy`. Extras:
`torch`, `monai`, `nifti`, `dicom`, `dicomseg`, `itk`, `schema`, `interp`.

## Documentation

[**docs/**](docs/index.md) · [Getting started](docs/getting-started.md) ·
[Concepts](docs/concepts.md) · [Specification](docs/spec/medh5-1.0.md)

[Python API](docs/python-api.md) · [CLI](docs/cli.md) ·
[Annotations](docs/annotations.md) · [Longitudinal](docs/longitudinal.md) ·
[Training](docs/training.md) · [Converters](docs/converters.md) ·
[Curation](docs/curation.md) · [Cohorts](docs/cohorts.md) ·
[Storage](docs/file-format.md) · [Conformance](docs/conformance.md)

## What the format is for

**One file per subject, not per scan.** A sample is a *subject*, and a subject
has visits. Longitudinal work — change detection, response assessment, lesion
tracking, follow-up registration — lives inside one file, which also means
assigning whole files to train and test cannot leak a patient between them.

**Geometry is stated once and never guessed.** Every array is bound to a
declared grid with spacing, origin and direction. A box sits at voxel edges and
an integer index is a voxel centre, both written down. Converting to NIfTI or
DICOM moves numbers between conventions explicitly, and refuses when it cannot.

**Absence is not silence.** `class_ids` says what an annotation contains;
`annotated_class_ids` says what was *looked for*. A class searched for and not
found is a usable negative example; a class nobody examined is not. Collapsing
the two is how a model learns a site's scans have no spleens.

**Every claim is checkable.** Per-object SHA-256 over decompressed content and
a Merkle `content_id` that survives recompression; a validator with a stable
diagnostic-code table; and a conformance corpus with one case per code.

**Reading a patch is fast.** A 64³ multi-class patch reads in 4 ms against
117 ms in 0.x, and foreground sampling is O(1) in the volume. Reproduce it with
`medh5 bench`.

## Write a sample

```python
import numpy as np
import medh5
from medh5 import LabelClass, LabelSet

labels = LabelSet("demo-v1", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "spleen", "Spleen", category="organ"),
    LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
])

with medh5.create("case_0001.medh5", sample_id="case_0001",
                  subject_id="DEMO-0001") as w:
    w.label_set(labels)
    w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
    w.add_grid("ct", shape=ct.shape, spacing=(2.0, 0.8, 0.8),
               origin=(-64.0, -38.4, -38.4), timepoint="tp0")
    w.add_image("CT", ct, grid="ct", modality="CT",
                value_type="quantitative", value_units="HU")
    w.add_segmentation("organs", grid="ct",
                       masks={"liver": liver, "lesion": lesion},
                       annotated_classes=["liver", "spleen", "lesion"])
```

`annotated_classes` names the spleen although there is no spleen mask: that
records "we looked and found none". The encoding is chosen by measuring the
class overlap graph — liver and lesion overlap, so it picks one that can
represent that — and the write is atomic.

## Train on it

```python
from torch.utils.data import DataLoader
from medh5.torch import PatchDataset, collate, worker_init_fn
from medh5.sampling import PatchSampler

sampler = PatchSampler((96, 96, 96), strategy="balanced",
                       foreground_classes=["liver", "lesion"])
dataset = PatchDataset(paths, sampler, images=["CT"],
                       annotations={"organs": ["liver", "lesion"]},
                       samples_per_volume=8)

loader = DataLoader(dataset, batch_size=2, num_workers=8,
                    worker_init_fn=worker_init_fn, collate_fn=collate)
```

`worker_init_fn` is required for `num_workers > 0`: HDF5 handles must not cross
a `fork`, so the cache is PID-keyed and a forked child abandons the parent's
handles rather than closing descriptors the parent still owns.

## Command line

```bash
medh5 info case.medh5                  # grids, images, annotations, coverage
medh5 validate case.medh5 --level strict
medh5 verify case.medh5                # digests and content_id
medh5 timeline case.medh5              # visits and intervals
medh5 track case.medh5 --class lesion  # per-lesion volumes across visits

medh5 dataset index studies/ -o cohort.json
medh5 dataset split cohort.json --group-by group_id --stratify-by site_id
medh5 dataset stats cohort.json --partition train --workers 8
medh5 dataset check cohort.json --deep

medh5 convert from-dicom /studies out/     # one sample per patient, all visits
medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz
medh5 convert from-rtstruct plan.dcm case.medh5 --rasterize
medh5 migrate old/*.medh5 -o new/ --group-by subject

medh5 scrub out/*.medh5 --apply --date-shift-days -117
medh5 pack cohort/*.medh5 -o shard.medh5c
medh5 recompress cohort/*.medh5 --profile training
medh5 bench                                # reproduce the performance targets
medh5 conformance publish suite/           # the suite, for another implementation
```

## Interoperability

| Format | |
|---|---|
| **NIfTI** | affine and voxels bit-identical on round trip; RAS↔LPS is a sign flip, never a resample |
| **DICOM** | slices ordered by geometry, spacing measured between origins, modality LUT stored not applied, tags on an explicit allow-list |
| **DICOM SEG** | frames placed by geometry; overlap and `FRACTIONAL` survive; segments matched by label, not number |
| **RTSTRUCT** | contours stay contours; rasterisation is opt-in and recorded in provenance |
| **nnU-Net v2** | class ids kept; region labels become label-set DAG parents; `dataset.json` round-trips |
| **MONAI** | `to_metatensor` gives a `MetaTensor` with the correct affine |
| **0.x** | `medh5 migrate`, reporting every decision and every guess |

Every conversion returns a report distinguishing what it **decided** from the
data and where it **guessed** — the encoding chosen, the class ids minted, a
half-voxel convention changed, a timepoint order inferred rather than read.

COCO is deliberately unsupported: it has no world geometry, spacing or frame of
reference, so importing means inventing a grid and exporting means discarding
the geometry that makes a medical annotation reproducible.

## Reading it without medh5

```python
import h5py, json, hdf5plugin       # hdf5plugin only for blosc2 profiles

with h5py.File("case_0001.medh5") as f:
    doc = json.loads(f["meta"][()])
    doc["identity"]["subject_id"]
    dict(f["grids"]["ct"].attrs)     # spacing, origin, direction
    f["images"]["CT"][10:20]
```

`medh5 recompress --profile portable` writes gzip, readable by any HDF5 build.

## Versioning

The **format** is 1.0. A minor version may add optional objects, profiles,
encodings and diagnostic codes; it may not change what an existing one means
(spec §16). The **package** follows semantic versioning from 1.0.0.

0.x files are not readable by 1.0 and are not meant to be — `medh5 migrate`
converts them once. See [Converters](docs/converters.md#migrating-from-0x).

## License

MIT
