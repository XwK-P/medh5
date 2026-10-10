# medh5

[![PyPI version](https://img.shields.io/pypi/v/medh5.svg)](https://pypi.org/project/medh5/)
[![Python versions](https://img.shields.io/pypi/pyversions/medh5.svg)](https://pypi.org/project/medh5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/XwK-P/medh5/blob/main/LICENSE)
[![CI](https://github.com/XwK-P/medh5/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XwK-P/medh5/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/readthedocs/medh5/latest.svg)](https://medh5.readthedocs.io/en/latest/)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-informational.svg)](https://github.com/XwK-P/medh5/blob/main/medh5/py.typed)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**One medical imaging sample — a subject, at every timepoint, with all of its
ground truth — in a single self-describing HDF5 file.**

Multi-modality images, segmentation in five encodings, detection boxes,
keypoints, contours, meshes, classification, registration between visits,
provenance and quality records, and per-object integrity digests --- and, from
format **1.1**, the clinical history around the images, with *when each fact
became known*, so a model trained at a cutoff reads only what was available
then. Format versions [**1.0**](https://medh5.readthedocs.io/en/latest/spec/medh5-1.0/) and
[**1.1**](https://medh5.readthedocs.io/en/latest/spec/medh5-1.1/), with normative specifications and a
[157-case conformance suite](https://medh5.readthedocs.io/en/latest/spec/conformance/) any implementation can run.

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

One format engine, written in Rust, with three frontends that read and write
the same bytes:

```bash
pip install medh5                      # the Python package
pip install "medh5[torch,nifti,dicom]"
cargo add medh5                        # the Rust crate
cargo install medh5-cli                # the `medh5` command line, natively
```

The wheels carry the engine, HDF5 included, so reading and writing need only
NumPy. Extras: `torch`, `monai`, `nifti`, `dicom`, `dicomseg`, `itk`, and
`h5py` for opening files with `h5py` directly. The `medh5` binary is also attached to every
[GitHub Release](https://github.com/XwK-P/medh5/releases), and installs with
`brew install XwK-P/medh5/medh5`; building it or the crate from source needs a
C compiler and CMake, for HDF5.

## Documentation

**[medh5.readthedocs.io](https://medh5.readthedocs.io/)** — tutorials, how-to
guides, the Python and CLI reference, and the normative specification.

[Write your first sample](https://medh5.readthedocs.io/en/latest/tutorials/first-sample/) ·
[How-to guides](https://medh5.readthedocs.io/en/latest/guides/) ·
[Python API](https://medh5.readthedocs.io/en/latest/reference/python-api/) ·
[CLI](https://medh5.readthedocs.io/en/latest/reference/cli/) ·
[Specification](https://medh5.readthedocs.io/en/latest/spec/medh5-1.0/)

## What the format is for

- **One file per subject, not per scan** — every visit in one place, so
  longitudinal work has a referent and splitting by file cannot leak a patient.
- **Geometry is stated once and never guessed** — declared grids, boxes at voxel
  edges, and converters that refuse rather than invent.
- **Absence is not silence** — a class examined and not found is recorded as
  such, which is a different training signal from one nobody examined.
- **Every claim is checkable** — per-object digests, a Merkle `content_id` that
  survives recompression, a stable diagnostic-code table, and a 157-case
  conformance corpus.
- **Reading a patch is fast** — a 64³ multi-class patch in ~3.5 ms, and O(1)
  foreground sampling once `build_index()` has run.
- **What was known, when** — labs, reports and their revisions, diagnoses and
  assessments on one subject clock, each with when it happened and when it
  became available; strict prospective selection decides what a row may read,
  and tasks and feature caches are pinned to the sample versions they read.

[The reasoning behind each](https://medh5.readthedocs.io/en/latest/explanation/design-rationale/).

## Write a sample

```python
import numpy as np
import medh5
from medh5 import LabelClass, LabelSet

# Stand-ins for a real CT and its masks.
ct = np.random.default_rng(0).integers(-1000, 1500, (64, 96, 96)).astype(np.int16)
liver = np.zeros(ct.shape, bool); liver[10:40, 20:70, 20:70] = True
lesion = np.zeros(ct.shape, bool); lesion[20:26, 35:45, 35:45] = True

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
    w.build_index()   # optional; foreground sampling is O(1) only with it
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

`worker_init_fn` drops handles inherited across a `fork`. It is recommended
rather than required: the handle cache is PID-keyed and re-checks ownership on
every access, so a forked worker abandons the parent's handles on first use
rather than reading through or closing them.

With a clinical history (format 1.1), a **task** asks a question at a cutoff,
and the batch holds only what was known then --- images from the visit
available at the cutoff, the report version current at it, missing modalities
and censored targets as masks rather than zeros:

```python
from torch.utils.data import DataLoader
from medh5.task import TaskManifest
from medh5.torch import ClinicalTaskDataset, collate_clinical

task = TaskManifest.load("cohort/progression.task.json")
assert task.preflight().ok                  # every pin, clock and duplicate checked
train = ClinicalTaskDataset(task, partition="train")
batch = next(iter(DataLoader(train, batch_size=8, collate_fn=collate_clinical)))
batch["present"]["mr"], batch["events"]["mask"], batch["target"]["observed"]
```

See [clinical history](https://medh5.readthedocs.io/en/latest/guides/clinical/) and
[training on clinical tasks](https://medh5.readthedocs.io/en/latest/guides/clinical-training/).

## Command line

`pip install medh5` puts `medh5` on the path; so does the standalone binary,
which is the same code compiled ahead of time.

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

medh5 clinical select case.medh5 --cutoff-hours 24    # what was known then (1.1)
medh5 task preflight progression.task.json           # every row: in or out, and why
medh5 cache validate reports.medh5cache --task progression.task.json
```

## Interoperability

| Format | |
|---|---|
| **NIfTI** | affine and voxels bit-identical on round trip; RAS↔LPS is a sign flip, never a resample |
| **DICOM** | slices ordered by geometry, spacing measured between origins, modality LUT stored not applied, tags on an explicit allow-list; slices that disagree about orientation, spacing or rescale are refused rather than read off the first one |
| **DICOM SEG** | frames placed by geometry; segments matched by label, not number; import preserves overlap and `FRACTIONAL`, export writes `BINARY` |
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
import h5py, json, hdf5plugin       # pip install "medh5[h5py]"; hdf5plugin for Blosc2

with h5py.File("case_0001.medh5") as f:
    doc = json.loads(f["meta"][()])
    doc["identity"]["subject_id"]
    dict(f["grids"]["ct"].attrs)     # spacing, origin, direction
    f["images"]["CT"][10:20]
```

`medh5 recompress --profile portable` writes gzip, readable by any HDF5 build.

## Versioning

The **format** is 1.1: 1.0 plus the optional `clinical` profile, and nothing
else. A sample is written at the lowest version its content needs, so
imaging-only data is still 1.0, which 1.x reads; a reader opens a later minor
as a projection, never amending it. A minor version may add optional objects,
profiles, encodings and diagnostic codes; it may not change what an existing
one means (spec §16). The **package** follows semantic versioning from 1.0.0:
2.0 moved the implementation to the Rust engine and changed the Python API at
the HDF5 boundary ([what changed](https://medh5.readthedocs.io/en/latest/changelog/)).

0.x files are not readable by 1.0 and are not meant to be — `medh5 migrate`
converts them once. See [Migrate from 0.x](https://medh5.readthedocs.io/en/latest/guides/migrate-0x/).

## Contributing

See [CONTRIBUTING.md](https://github.com/XwK-P/medh5/blob/main/CONTRIBUTING.md)
for the development setup, the checks a change has to pass, and how the
specification and documentation are kept in step with the code. Release notes
are in [CHANGELOG.md](https://github.com/XwK-P/medh5/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/XwK-P/medh5/blob/main/LICENSE)
