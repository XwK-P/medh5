# mlh5

**HDF5 + Blosc2 multi-array format for ML workloads.**

Store image + segmentation + bounding boxes + image-level label in a
single `.mlh5` file with Blosc2 compression, chunk-size optimization for
patch-based training, and metadata as plain HDF5 attributes.

## Installation

```bash
pip install mlh5
```

Or install from source:

```bash
pip install -e ".[dev]"
```

## Quick start

### Write a sample

```python
import numpy as np
from mlh5 import MLH5File

image = np.random.random((128, 256, 256)).astype(np.float32)
seg   = np.random.randint(0, 4, size=image.shape, dtype=np.uint8)

bboxes = np.array([
    [[10, 30], [40, 80], [50, 90]],   # box 1: [z_min,z_max], [y_min,y_max], [x_min,x_max]
    [[60, 90], [20, 50], [100, 130]],  # box 2
])
bbox_scores = np.array([0.97, 0.82])
bbox_labels = ["tumor", "cyst"]

MLH5File.write(
    "sample.mlh5",
    image=image,
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
    extra={"modality": "CT", "patient_id": "P001"},
)
```

### Read a sample (eager)

```python
from mlh5 import MLH5File

sample = MLH5File.read("sample.mlh5")

print(sample.image.shape)       # (128, 256, 256)
print(sample.seg.shape)         # (128, 256, 256)
print(sample.bboxes.shape)      # (2, 3, 2)
print(sample.bbox_labels)       # ['tumor', 'cyst']
print(sample.meta.label)        # 1
print(sample.meta.spatial.spacing)  # [1.0, 0.5, 0.5]
print(sample.meta.extra)        # {'modality': 'CT', 'patient_id': 'P001'}
```

### Partial / patch read (lazy)

For large datasets where loading the whole volume is impractical:

```python
from mlh5 import MLH5File

with MLH5File.open("sample.mlh5") as f:
    patch = f["image"][10:42, 50:114, 50:114]   # reads only required chunks
    seg_patch = f["seg"][10:42, 50:114, 50:114]
```

### Metadata-only read

Inspect metadata without touching the array data:

```python
from mlh5 import MLH5File

meta = MLH5File.read_meta("sample.mlh5")
print(meta.label)              # 1
print(meta.spatial.spacing)    # [1.0, 0.5, 0.5]
print(meta.extra)              # {'modality': 'CT', ...}
```

### Inspect with standard HDF5 tools

Since `.mlh5` is plain HDF5, you can use any HDF5 viewer:

```bash
h5ls -v sample.mlh5
h5dump -A sample.mlh5          # show all attributes
```

## On-disk layout

```
sample.mlh5
├── image          (dataset, N-D float32, Blosc2-compressed, chunked)
├── seg            (dataset, N-D uint8, Blosc2-compressed, chunked, optional)
├── bboxes         (dataset, [n, ndims, 2], optional)
├── bbox_scores    (dataset, [n], optional)
├── bbox_labels    (dataset, [n] variable-length string, optional)
└── (root attrs)
    ├── schema_version: "1"
    ├── label: int or str
    ├── label_name: str
    ├── has_seg: bool
    ├── has_bbox: bool
    └── extra: JSON string

image.attrs:
    ├── spacing: float array
    ├── origin: float array
    ├── direction: float array (flattened)
    ├── axis_labels: string array
    ├── coord_system: str
    └── patch_size: int array
```

## Chunk optimization

mlh5 includes a chunk-size optimizer (ported from
[mlarray](https://github.com/MIC-DKFZ/mlarray)) that sizes HDF5 chunks
to fit L3 cache for efficient patch-based training reads:

```python
from mlh5 import optimize_chunks

chunks = optimize_chunks(
    image_shape=(128, 256, 256),
    patch_size=192,
    bytes_per_element=4,   # float32
)
print(chunks)  # e.g. (128, 256, 256) for small volumes
```

## Dependencies

- `h5py >= 3.8`
- `hdf5plugin >= 4.0`
- `numpy >= 1.24`

## License

MIT
