# Import from NIfTI and nnU-Net

You have volumes and masks as separate files. This turns them into one sample.

```bash
pip install "medh5[nifti]"
```

## Images and masks as separate files

```bash
medh5 convert from-nifti case.medh5 \
    --image CT=ct.nii.gz \
    --mask liver=liver.nii.gz --mask lesion=lesion.nii.gz
```

```python
from medh5.io.nifti import from_nifti

report = from_nifti({"CT": "ct.nii.gz"}, "case.medh5",
                    masks={"liver": "liver.nii.gz", "lesion": "lesion.nii.gz"})
report.ok
report.of_kind("encoding")     # which encoding was chosen, and why
```

Two conversions happen and both are recorded rather than assumed. NIfTI is
`(x, y, z)` and medh5 is `(z, y, x)`, so the array is transposed and the spacing
and direction permuted to match. NIfTI is RAS+ and medh5 defaults to LPS, so the
affine takes a sign flip — `diag(-1, -1, 1, 1)`, no resampling. Pass
`--coord-system RAS` to stay in RAS.

The round trip is exact: `from_nifti` → `to_nifti` reproduces the affine and the
voxels bit-for-bit.

## When the files disagree

**Two files with different affines are refused.** They are not the same grid, and
the importer will not resample one onto the other to make the import work.
Resample deliberately, in your own code, and say so.

**A file declaring no geometry is refused.** `sform_code == qform_code == 0` is
NIfTI saying it has voxel indices and no spatial mapping. nibabel still returns
an affine rebuilt from `pixdim`, and importing that mints a world grid nobody
measured:

```bash
medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz --assume-geometry
```

That takes the fallback on purpose, and the report records it as a guess.

[The reasoning for both](../explanation/refusals.md#nifti).

## nnU-Net v2 datasets

```bash
medh5 convert from-nnunet /Dataset001_Liver out/
```

Each case's channels and per-class masks become one sample. nnU-Net's class ids
are **kept**, so a model trained against the original dataset still means the
same thing, and region labels become label-set DAG parents — a region that is
the union of two components is a class those components name as a parent.

The parsed `dataset.json` is stashed in `extra["nnunetv2"]`, so an export later
reproduces the original dataset definition instead of inventing one.

Every channel and label volume must share one grid; a volume some other tool
resampled is refused rather than filed onto channel 0's grid.

## Check the result

```bash
medh5 info case.medh5
medh5 validate case.medh5 --level strict
```

## Related

- **[Converters](../reference/converters.md#nifti)** — options and Python entry points.
- **[What the converters refuse, and why](../explanation/refusals.md)** — the refusals in full.
- **[Export to other formats](export.md)** — going back the other way.
