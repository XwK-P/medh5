# medh5

**HDF5 + Blosc2 multi-array format for ML workloads.**

Store multiple co-registered images (e.g. CT, MRI, PET) + segmentation
masks + bounding boxes + image-level label in a single `.medh5` file with
Blosc2 compression, chunk-size optimization for patch-based training, and
metadata as plain HDF5 attributes.

## Installation

```bash
pip install medh5
```

With PyTorch dataset support:

```bash
pip install medh5[torch]
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

MEDH5File.update_meta("sample.medh5", label=2, extra={"reviewed": True})
MEDH5File.add_seg("sample.medh5", "new_mask", mask_array)
```

### Verify file integrity

```python
from medh5 import MEDH5File

assert MEDH5File.verify("sample.medh5")  # True if checksum matches
```

### PyTorch integration

```python
from medh5.torch import MEDH5TorchDataset

dataset = MEDH5TorchDataset(["s1.medh5", "s2.medh5", "s3.medh5"])
sample = dataset[0]
print(sample["images"]["CT"].shape)  # torch.Size([128, 256, 256])
print(sample["label"])               # 1
```

### CLI

```bash
medh5 info sample.medh5       # print metadata summary
medh5 validate sample.medh5   # check file structure
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

- `h5py >= 3.8`
- `hdf5plugin >= 4.0`
- `numpy >= 1.24`
- `torch >= 2.0` (optional, for `medh5[torch]`)

## License

MIT
