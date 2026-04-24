# Converters

medh5 ships round-trip converters for the formats most medical imaging ML
pipelines consume or produce: NIfTI, DICOM series, and raw nnU-Net v2
dataset folders.

All converter modules are lazy-imported from `medh5.io` — installing `medh5`
alone does not pull in nibabel, pydicom, or SimpleITK. Install the
appropriate extra:

```bash
pip install "medh5[nifti]"    # from_nifti / to_nifti / import_seg_nifti / nnU-Net v2
pip install "medh5[dicom]"    # from_dicom
pip install "medh5[itk]"      # SimpleITK resampling for from_nifti
```

## NIfTI

```python
from medh5.io import from_nifti, to_nifti, import_seg_nifti

from_nifti(
    images={"CT": "ct.nii.gz", "PET": "pet.nii.gz"},
    seg={"tumor": "tumor.nii.gz"},
    out_path="sample.medh5",
    label=1,
    compression="balanced",
    checksum=True,
)

# Export back for viewing in 3D Slicer / ITK-SNAP
to_nifti("sample.medh5", out_dir="export/")
# Writes export/image_CT.nii.gz, export/image_PET.nii.gz, export/seg_tumor.nii.gz
```

Spacing, origin, direction, and coordinate system are extracted from the
NIfTI affine automatically.

### Resampling onto a shared grid

If modalities live on different grids, `from_nifti` can resample them onto
a reference grid using SimpleITK (requires `medh5[itk]`). Masks always use
nearest-neighbor; images pick the interpolator you specify.

```python
from_nifti(
    images={"CT": "ct_1mm.nii.gz", "PET": "pet_2mm.nii.gz"},
    seg={"tumor": "tumor_2mm.nii.gz"},
    out_path="sample.medh5",
    resample_to="CT",               # use CT grid as reference
    interpolator="linear",          # "linear" | "nearest" | "bspline"
)
```

### Importing an edited mask

Round-trip edits from external tools back into a `.medh5` file without
rewriting images:

```python
import_seg_nifti(
    "sample.medh5",
    "edited_tumor.nii.gz",
    name="tumor",
    resample=True,      # reslice onto the file's grid if affines differ
    replace=True,       # overwrite an existing mask of that name
)
```

## DICOM

```python
from medh5.io import from_dicom

from_dicom(
    dicom_dir="path/to/series",
    out_path="sample.medh5",
    modality_name="CT",
    series_uid="1.2.3.4.5",     # optional — otherwise the largest series is chosen deterministically
    apply_modality_lut=True,     # apply RescaleSlope / RescaleIntercept (default)
    extra_tags=["PatientID", "StudyDate"],
)
```

Geometry is validated strictly — `ImageOrientationPatient`, `PixelSpacing`,
and slice spacing must be consistent across the selected series. Multi-frame
and non-grayscale DICOMs are rejected with clear errors.

Provenance (selected series UID, all available UIDs, instance count, LUT
application status) is recorded under `extra["dicom"]`.

## nnU-Net v2

Round-trip between a raw [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
dataset folder and a directory of per-case `.medh5` files. Each case
becomes one `.medh5` bundling every channel plus one boolean mask per
foreground class declared in `dataset.json`. The parsed `dataset.json`
payload is stashed in `extra["nnunetv2"]` so export reconstructs the exact
source layout.

```python
from medh5.io import from_nnunetv2, to_nnunetv2

# Raw nnU-Net v2 → directory of .medh5 files
from_nnunetv2(
    "Dataset042_BraTS/",
    "medh5_out/",
    include_test=True,          # also convert imagesTs/ (seg=None)
    compression="balanced",
    checksum=True,
)
# Writes medh5_out/imagesTr/{case}.medh5 (+ medh5_out/imagesTs/{case}.medh5)

# Directory of .medh5 files → raw nnU-Net v2 layout
to_nnunetv2(
    "medh5_out/",
    "Dataset042_BraTS_roundtrip/",
    dataset_name="Dataset042_BraTS",
    file_ending=".nii.gz",
)
```

Channel order, label integer values, and optional fields
(`overwrite_image_reader_writer`, `regions_class_order`, `name`) all
round-trip losslessly.

### Silent-data-loss guards

The converters reject silent-data-loss conditions rather than quietly
dropping voxels or masks:

- **Import** — `_split_label_volume` raises `MEDH5ValidationError` if the
  label volume contains integer values that are not declared in
  `dataset.json`'s `labels` map. Float label volumes whose values are all
  integer-valued (`0.0`, `1.0`, …) are accepted; genuinely non-integer
  voxels (e.g. `1.9`) are rejected.
- **Export** — seg-mask names or image channels that disagree with the
  nnU-Net metadata stored in `extra["nnunetv2"]` raise
  `MEDH5ValidationError` with a missing/extra report. Fix the metadata in
  `extra["nnunetv2"]["labels"]` or remove the extra mask/channel.
- **Region-based labels** — list-valued `labels` in `dataset.json`
  (overlapping region definitions) are rejected on import with a clear
  error. Convert to integer labels first.
