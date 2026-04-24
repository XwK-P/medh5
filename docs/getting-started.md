# Getting started

## Install

```bash
pip install medh5
```

Optional extras pull in the dependencies needed for specific features:

```bash
pip install "medh5[torch]"    # PyTorch datasets
pip install "medh5[nifti]"    # NIfTI import/export  (needs nibabel)
pip install "medh5[dicom]"    # DICOM import         (needs pydicom)
pip install "medh5[itk]"      # Resampling           (needs SimpleITK)
```

To install from source for development:

```bash
git clone https://github.com/XwK-P/medh5.git
cd medh5
pip install -e ".[dev,torch,nifti,dicom,itk]"
```

## Python requirement

Python >= 3.10. Supported matrix: 3.10, 3.11, 3.12.

## Write your first `.medh5`

```python
import numpy as np
from medh5 import MEDH5File

ct  = np.random.random((64, 128, 128)).astype(np.float32)
pet = np.random.random((64, 128, 128)).astype(np.float32)
tumor = np.random.random(ct.shape) > 0.95

MEDH5File.write(
    "sample.medh5",
    images={"CT": ct, "PET": pet},
    seg={"tumor": tumor},
    label=1,
    label_name="malignant",
    spacing=[1.0, 0.5, 0.5],
    origin=[0.0, 0.0, 0.0],
    direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    coord_system="RAS",
    extra={"patient_id": "P001"},
    compression="balanced",   # or "fast" / "max"
    checksum=True,
)
```

Writes are **atomic** — if the process is killed mid-write, any pre-existing
file at `sample.medh5` is preserved and no truncated file is left at the
destination.

## Read it back

```python
from medh5 import MEDH5File

sample = MEDH5File.read("sample.medh5")

print(sample.images.keys())         # dict_keys(['CT', 'PET'])
print(sample.images["CT"].shape)    # (64, 128, 128)
print(sample.meta.label)            # 1
print(sample.meta.spatial.spacing)  # [1.0, 0.5, 0.5]
```

For large volumes, read lazily instead:

```python
with MEDH5File("sample.medh5") as f:
    patch = f.images["CT"][10:42, 20:84, 20:84]
    if f.seg is not None:
        mask_patch = f.seg["tumor"][10:42, 20:84, 20:84]
```

## Validate and verify

```python
from medh5 import MEDH5File

report = MEDH5File.validate("sample.medh5")
assert report.ok()                  # no errors
assert MEDH5File.verify("sample.medh5")  # SHA-256 matches
```

See [Python API](python-api.md) for the full surface and
[File format](file-format.md) for the on-disk layout.

## Next steps

- Convert existing NIfTI / DICOM / nnU-Net v2 datasets → [Converters](converters.md)
- Train a PyTorch model on `.medh5` files → [PyTorch integration](pytorch.md)
- Build a dataset manifest and train/val/test splits → [Datasets and statistics](dataset-and-stats.md)
- Do everything from the shell → [CLI reference](cli.md)
