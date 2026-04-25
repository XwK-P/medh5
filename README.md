# medh5

[![PyPI version](https://img.shields.io/pypi/v/medh5.svg)](https://pypi.org/project/medh5/)
[![Python versions](https://img.shields.io/pypi/pyversions/medh5.svg)](https://pypi.org/project/medh5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/XwK-P/medh5/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XwK-P/medh5/actions/workflows/ci.yml)
[![Release](https://github.com/XwK-P/medh5/actions/workflows/release.yml/badge.svg)](https://github.com/XwK-P/medh5/actions/workflows/release.yml)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](#)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-informational.svg)](medh5/py.typed)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**HDF5 + Blosc2 multi-array format for Medical Imaging ML workloads.**

> **Status:** Beta (0.6.0) — API may still change between minor versions.
> Backward compatibility is not guaranteed until 1.0.

Store multiple co-registered images (e.g. CT, MRI, PET) + segmentation
masks + bounding boxes + image-level label in a single `.medh5` file with
Blosc2 compression, chunk-size optimization for patch-based training, and
metadata as plain HDF5 attributes.

## Documentation

Full docs live under [`docs/`](docs/index.md):

- [Getting started](docs/getting-started.md)
- [File format](docs/file-format.md)
- [Python API](docs/python-api.md)
- [CLI reference](docs/cli.md)
- [PyTorch integration](docs/pytorch.md)
- [Converters (NIfTI / DICOM / nnU-Net v2)](docs/converters.md)
- [Datasets and statistics](docs/dataset-and-stats.md)
- [Review / QA workflow](docs/review.md)

## Installation

```bash
pip install medh5
```

With optional extras:

```bash
pip install medh5[torch]    # PyTorch dataset support
pip install medh5[nifti]    # NIfTI import/export (nibabel)
pip install medh5[dicom]    # DICOM import (pydicom)
pip install medh5[itk]      # Resampling via SimpleITK
```

Or install from source:

```bash
pip install -e ".[dev]"
```

## Quick start

### Write a sample

```python
import numpy as np
from medh5 import MEDH5File

ct = np.random.random((128, 256, 256)).astype(np.float32)
pet = np.random.random((128, 256, 256)).astype(np.float32)
seg = {
    "tumor": np.random.random(ct.shape) > 0.9,
    "liver": np.random.random(ct.shape) > 0.5,
}

bboxes = np.array([
    [[10, 30], [40, 80], [50, 90]],   # box 1: [z_min,z_max], [y_min,y_max], [x_min,x_max]
    [[60, 90], [20, 50], [100, 130]],  # box 2
])
bbox_scores = np.array([0.97, 0.82])
bbox_labels = ["tumor", "cyst"]

MEDH5File.write(
    "sample.medh5",
    images={"CT": ct, "PET": pet},
    seg=seg,
    bboxes=bboxes,
    bbox_scores=bbox_scores,
    bbox_labels=bbox_labels,
    label=1,
    label_name="malignant",
    spacing=[1.0, 0.5, 0.5],
    origin=[0.0, 0.0, 0.0],
    direction=[[1,0,0],[0,1,0],[0,0,1]],
    coord_system="RAS",
    patch_size=192,
    extra={"patient_id": "P001"},
    compression="balanced",  # or "fast", "max"
    checksum=True,
)
```

### Read a sample (eager)

```python
from medh5 import MEDH5File

sample = MEDH5File.read("sample.medh5")

print(sample.images.keys())             # dict_keys(['CT', 'PET'])
print(sample.images["CT"].shape)         # (128, 256, 256)
print(list(sample.seg.keys()))           # ['liver', 'tumor']
print(sample.bboxes.shape)               # (2, 3, 2)
print(sample.meta.label)                 # 1
print(sample.meta.image_names)           # ['CT', 'PET']
print(sample.meta.spatial.spacing)       # [1.0, 0.5, 0.5]
```

### Partial / patch read (lazy)

For large datasets where loading the whole volume is impractical:

```python
from medh5 import MEDH5File

with MEDH5File.open("sample.medh5") as f:
    patch = f["images/CT"][10:42, 50:114, 50:114]
    tumor_patch = f["seg/tumor"][10:42, 50:114, 50:114]
```

Or use the context-manager for typed access:

```python
from medh5 import MEDH5File

with MEDH5File("sample.medh5") as f:
    meta = f.meta
    patch = f.images["CT"][10:42, 50:114, 50:114]
    if f.seg is not None:
        seg_patch = f.seg["tumor"][10:42, 50:114, 50:114]
```

### Metadata-only read

Inspect metadata without touching the array data:

```python
from medh5 import MEDH5File

meta = MEDH5File.read_meta("sample.medh5")
print(meta.label)              # 1
print(meta.image_names)        # ['CT', 'PET']
print(meta.spatial.spacing)    # [1.0, 0.5, 0.5]
```

### In-place updates

Update metadata or add segmentation masks without rewriting image data:

```python
from medh5 import MEDH5File

# Simple convenience methods
MEDH5File.update_meta("sample.medh5", label=2, extra={"reviewed": True})
MEDH5File.add_seg("sample.medh5", "new_mask", mask_array)

# Unified update API — metadata, seg, and bbox in one call
MEDH5File.update(
    "sample.medh5",
    meta={"label": 3, "spacing": [1.0, 0.5, 0.5], "coord_system": "RAS"},
    seg_ops={"add": {"organ": organ_mask}, "remove": ["old_mask"]},
    bbox_ops={"bboxes": new_bboxes, "bbox_labels": ["tumor"]},
)
```

### Validate file structure

```python
from medh5 import MEDH5File

report = MEDH5File.validate("sample.medh5")
print(report.is_valid)        # True if no errors
print(report.errors)          # list of ValidationIssue(code=..., message=...)
print(report.warnings)        # e.g. missing checksum
print(report.ok(strict=True)) # False if any errors OR warnings
```

### Verify file integrity

`verify()` returns a tri-state `VerifyResult` so audit UIs can
distinguish "no checksum stored" from "verified good":

```python
from medh5 import MEDH5File, VerifyResult

match MEDH5File.verify("sample.medh5"):
    case VerifyResult.OK:       ...   # stored checksum matches
    case VerifyResult.MISSING:  ...   # no checksum was stored (opt-in)
    case VerifyResult.MISMATCH: ...   # data has diverged from digest
```

### Concurrent lazy reads (`open_shared`)

Multiple consumers in the same process (viewers, dashboards, napari
layers) can share a single underlying `h5py.File` via ref-counted
`open_shared`. The handle closes when the last caller exits its `with`
block:

```python
from medh5 import open_shared

with open_shared("sample.medh5") as f:
    patch = f["images/CT"][10:42, 20:84, 20:84]
```

Pair it with `on_reopened=` on any mutating call (`update`,
`update_meta`, `add_seg`, `set_review_status`) to rebind cached views
after a successful write.

### PyTorch integration

**Eager dataset** (full-volume read per sample):

```python
from medh5.torch import MEDH5TorchDataset

dataset = MEDH5TorchDataset(["s1.medh5", "s2.medh5", "s3.medh5"])
sample = dataset[0]
print(sample["images"]["CT"].shape)  # torch.Size([128, 256, 256])
print(sample["label"])               # 1
```

**Patch-based dataset** (lazy chunk-aligned reads):

```python
from medh5.sampling import PatchSampler
from medh5.transforms import Compose, Clip, Normalize, RandomFlip
from medh5.torch import MEDH5PatchDataset

sampler = PatchSampler(
    patch_size=(96, 96, 96),
    strategy="foreground",     # "uniform" | "foreground" | "balanced"
    foreground_seg="tumor",
    foreground_prob=0.7,
)

transform = Compose([
    Clip(min=-1000, max=1000),
    Normalize(mean={"CT": -200.0}, std={"CT": 350.0}),
    RandomFlip(axes=(1, 2), p=0.5),
])

dataset = MEDH5PatchDataset(
    paths=["s1.medh5", "s2.medh5"],
    sampler=sampler,
    transform=transform,
    samples_per_volume=4,
)
```

Both dataset classes use a per-worker LRU file-handle cache, so repeated
reads against the same file reuse one `h5py.File` handle instead of
re-opening from scratch every call.

**`DataLoader` with `num_workers > 0`**

h5py is not fork-safe, and open `h5py.File` handles cannot be pickled for
`multiprocessing_context="spawn"` (the default on macOS / Windows /
Python 3.14+). medh5 handles this in two ways: the handle cache
detects PID changes on access and transparently resets itself, and
`medh5.torch.worker_init_fn` clears the cache at worker startup for
belt-and-braces safety. Always pass `worker_init_fn=medh5.torch.worker_init_fn`
when you use `num_workers > 0`:

```python
from torch.utils.data import DataLoader
import medh5.torch as mt

dataset = mt.MEDH5PatchDataset(paths, sampler=sampler)

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    worker_init_fn=mt.worker_init_fn,   # <- required with num_workers > 0
    # multiprocessing_context="spawn",  # optional, default on macOS/Windows
)

for batch in loader:
    ...
```

Without `worker_init_fn`, a forked worker would observe stale h5py state
inherited from the parent process, which can deadlock or silently corrupt
reads. See [h5py's documentation](https://docs.h5py.org/en/stable/faq.html#multiprocessing)
for the underlying issue.

### NIfTI / DICOM conversion

Import NIfTI volumes into `.medh5`:

```python
from medh5.io import from_nifti, to_nifti

from_nifti(
    images={"CT": "ct.nii.gz", "PET": "pet.nii.gz"},
    seg={"tumor": "tumor.nii.gz"},
    out_path="sample.medh5",
    label=1,
    compression="balanced",
)
```

Resample multi-resolution modalities onto a shared grid (requires `medh5[itk]`):

```python
from_nifti(
    images={"CT": "ct_1mm.nii.gz", "PET": "pet_2mm.nii.gz"},
    seg={"tumor": "tumor_2mm.nii.gz"},
    out_path="sample.medh5",
    resample_to="CT",          # use CT grid as reference
    interpolator="linear",     # masks always use nearest-neighbor
)
```

Import a segmentation mask into an existing file:

```python
from medh5.io import import_seg_nifti

import_seg_nifti("sample.medh5", "edited_tumor.nii.gz", name="tumor", resample=True, replace=True)
```

Export back to NIfTI for editing in 3D Slicer / ITK-SNAP:

```python
to_nifti("sample.medh5", out_dir="export/")
# Writes export/image_CT.nii.gz, export/image_PET.nii.gz, export/seg_tumor.nii.gz
```

Import a DICOM series:

```python
from medh5.io import from_dicom

from_dicom(
    dicom_dir="path/to/series",
    out_path="sample.medh5",
    modality_name="CT",
    series_uid="1.2.3.4.5",    # optional: select specific series
    apply_modality_lut=True,   # apply RescaleSlope/Intercept (default)
    extra_tags=["PatientID", "StudyDate"],
)
```

### nnU-Net v2 dataset conversion

Convert a raw [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) dataset folder
(`imagesTr/`, `labelsTr/`, optional `imagesTs/`, `dataset.json`) into a
directory of per-case `.medh5` files and back. Each case becomes one
`.medh5` bundling every channel plus one boolean mask per foreground class
declared in `dataset.json`. The parsed `dataset.json` payload is stashed in
`extra["nnunetv2"]` so export reconstructs the exact source layout —
including channel order and integer label values. Requires `medh5[nifti]`.

```python
from medh5.io import from_nnunetv2, to_nnunetv2

# Raw nnU-Net v2 → directory of .medh5 files
from_nnunetv2(
    "Dataset042_BraTS/",
    "medh5_out/",
    include_test=True,     # also convert imagesTs/ (seg=None)
    compression="balanced",
    checksum=True,
)
# Writes medh5_out/imagesTr/{case}.medh5 (+ medh5_out/imagesTs/{case}.medh5)

# Directory of .medh5 files → raw nnU-Net v2 layout
to_nnunetv2("medh5_out/", "Dataset042_BraTS_roundtrip/")
```

The converters reject silent-data-loss conditions up front:

- **Import**: label volumes containing integer values that aren't declared
  in `dataset.json`'s `labels` map raise `MEDH5ValidationError` instead of
  silently dropping those voxels.
- **Export**: `.medh5` files whose seg-mask names or image channels do not
  match the nnU-Net metadata stored in `extra["nnunetv2"]` are rejected
  rather than silently omitted from the emitted dataset.
- Region-based labels (list-valued `labels`) are rejected with a clear
  error — convert to integer labels first.

### Dataset operations

Build a manifest, filter, and split:

```python
from medh5.dataset import Dataset, make_splits

ds = Dataset.from_directory("data/", recursive=True)
labeled = ds.filter(lambda r: r.label is not None)
ds.save("manifest.json")

splits = make_splits(
    ds,
    ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    stratify_by="label",
    group_by="extra.patient_id",
    seed=42,
)
splits["train"].save("train.json")
```

Compute dataset-level statistics:

```python
from medh5.stats import compute_stats

stats = compute_stats(ds, workers=4)
print(stats["CT"].mean, stats["CT"].std)
print(stats.label_counts)
```

### Review / QA workflow

Track annotation review status without a GUI:

```python
from medh5 import MEDH5File

MEDH5File.set_review_status(
    "sample.medh5",
    status="reviewed",       # pending | reviewed | flagged | rejected
    annotator="puyang",
    notes="ok",
)

review = MEDH5File.get_review_status("sample.medh5")
print(review.status, review.annotator, review.timestamp)
```

### CLI

```bash
# Single-file operations
medh5 info sample.medh5             # print metadata summary
medh5 info sample.medh5 --json      # machine-readable JSON output
medh5 validate sample.medh5         # check file structure
medh5 validate sample.medh5 --strict --json  # warnings become errors

# Batch operations
medh5 validate-all data/            # validate every .medh5 under a directory
medh5 validate-all data/ --fail-fast --workers 4
medh5 audit data/                   # verify SHA-256 checksums
medh5 recompress data/ --compression max  # rewrite with different compression

# Dataset management
medh5 index data/ -o manifest.json
medh5 split manifest.json --ratios 0.7,0.15,0.15 --stratify label -o splits/
medh5 stats data/ -o stats.json --json

# NIfTI / DICOM / nnU-Net v2 conversion
medh5 import nifti --image CT ct.nii.gz -o sample.medh5
medh5 import nifti --image CT ct.nii.gz --image PET pet.nii.gz \
      --resample-to CT --interpolator linear -o sample.medh5
medh5 import dicom /path/to/series -o sample.medh5
medh5 import dicom /path/to/series -o sample.medh5 \
      --series-uid 1.2.3.4.5 --no-modality-lut
medh5 import nnunetv2 Dataset042_BraTS/ -o medh5_out/
medh5 import nnunetv2 Dataset042_BraTS/ -o medh5_out/ \
      --no-test --compression max --checksum
medh5 export nifti sample.medh5 -o export/
medh5 export nnunetv2 medh5_out/ -o Dataset042_BraTS_roundtrip/

# Review workflow
medh5 review set sample.medh5 --status reviewed --annotator puyang
medh5 review get sample.medh5
medh5 review get sample.medh5 --json
medh5 review list data/ --status pending
medh5 review import-seg sample.medh5 --name tumor --from edited.nii.gz
medh5 review import-seg sample.medh5 --name tumor --from edited.nii.gz --resample --replace
```

### Inspect with standard HDF5 tools

Since `.medh5` is plain HDF5, you can use any HDF5 viewer:

```bash
h5ls -v sample.medh5
h5dump -A sample.medh5          # show all attributes
```

## On-disk layout

```
sample.medh5
├── images/            (group, required, >= 1 entry)
│   ├── CT             (dataset, N-D, Blosc2-compressed, chunked)
│   ├── PET            (dataset, N-D, Blosc2-compressed, chunked)
│   └── ...
├── seg/               (group, optional)
│   ├── tumor          (dataset, N-D bool, Blosc2-compressed, chunked)
│   ├── liver          (dataset, N-D bool, Blosc2-compressed, chunked)
│   └── ...
├── bboxes             (dataset, [n, ndims, 2], optional)
├── bbox_scores        (dataset, [n], optional)
├── bbox_labels        (dataset, [n] variable-length string, optional)
└── (root attrs)
    ├── schema_version: "1"
    ├── image_names: JSON list of modality names
    ├── label: int or str
    ├── label_name: str
    ├── has_seg: bool
    ├── seg_names: JSON list of mask names
    ├── has_bbox: bool
    ├── extra: JSON string
    └── checksum_sha256: str (optional)

images.attrs:
    ├── shape: int array
    ├── spacing: float array
    ├── origin: float array
    ├── direction: float array (flattened)
    ├── axis_labels: string array
    ├── coord_system: str
    └── patch_size: int array
```

## Compression presets

| Preset       | Compressor | Level | Use case                     |
|-------------|------------|-------|------------------------------|
| `"fast"`    | lz4        | 3     | Fast write, moderate ratio   |
| `"balanced"`| lz4hc      | 8     | Default, good ratio + speed  |
| `"max"`     | zstd       | 9     | Maximum compression ratio    |

## Chunk optimization

medh5 includes a chunk-size optimizer (ported from
[mlarray](https://github.com/MIC-DKFZ/mlarray)) that sizes HDF5 chunks
to fit L3 cache for efficient patch-based training reads:

```python
from medh5 import optimize_chunks

chunks = optimize_chunks(
    image_shape=(128, 256, 256),
    patch_size=192,
    bytes_per_element=4,   # float32
)
```

## Design decisions

**Why HDF5?** HDF5 provides native chunking, compression filters, attribute
metadata, and partial I/O -- all critical for efficient patch-based ML
training on large volumes.  It is widely supported (h5py, HDFView, MATLAB,
Julia) and inspectable without custom tooling.

**Why Blosc2?** Blosc2 is a high-performance meta-compressor optimized for
binary data.  It supports multi-threaded compression, multiple codecs
(lz4, zstd, etc.), and integrates with HDF5 via hdf5plugin.

**Why a single-sample file?** Each `.medh5` file represents one sample
(patient/scan).  This maps naturally to medical imaging workflows where
each scan is processed independently, and avoids the complexity of
multi-sample container formats.

**Multi-modality by default.** Medical imaging routinely involves multiple
co-registered modalities (CT + PET, multi-sequence MRI).  All modalities
share the same spatial grid, so they share spatial metadata and chunk layout.

**Trust model.** The `extra` field stores arbitrary JSON.  When reading
`.medh5` files from untrusted sources, be aware that very large or deeply
nested JSON could consume significant memory.

## Dependencies

- `h5py >= 3.10`
- `hdf5plugin >= 4.1`
- `numpy >= 1.24`
- `torch >= 2.0` (optional, for `medh5[torch]`)
- `nibabel >= 5` (optional, for `medh5[nifti]`)
- `pydicom >= 2.4` (optional, for `medh5[dicom]`)
- `SimpleITK >= 2.3` (optional, for `medh5[itk]`)

## License

MIT
