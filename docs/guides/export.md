# Export to other formats

Getting data back out, and what each exporter will not do.

## NIfTI

```bash
medh5 convert to-nifti case.medh5 CT out.nii.gz
medh5 convert to-nifti case.medh5 CT liver.nii.gz --annotation organs --class liver
```

`--stored` writes the stored values rather than the physical ones; by default a
quantitative image is written after its rescale, so the numbers mean what the
units say.

The round trip from `from_nifti` is exact — affine and voxels bit-for-bit.

## DICOM SEG

```bash
pip install "medh5[dicomseg]"
medh5 convert to-dicom-seg case.medh5 organs out.dcm --source ct/*.dcm
```

`--source` is the original series the segmentation refers to; a SEG is only
meaningful against one. Overlapping segments and `FRACTIONAL` segmentations both
survive.

Writing goes through `highdicom` rather than assembling the IOD by hand, which
is how invalid SEGs get published.

## RTSTRUCT

```bash
medh5 convert to-rtstruct case.medh5 contours out.dcm --source ct/*.dcm
```

**This refuses a voxel annotation.** `to-rtstruct` on a mask is an error, not a
marching-squares fallback: the contours it would produce are not the contours
anyone drew, and an RTSTRUCT is a clinical document that asserts they are. Export
contours you imported, or drew, as contours.

## nnU-Net v2

```bash
medh5 convert to-nnunet /out case1.medh5 case2.medh5 --dataset-name Dataset001_Liver
```

Classes are matched by **id**, not by name — which is why import keeps nnU-Net's
own integers. A class the sample does not have is refused (`E402`) rather than
skipped; a skipped class produces a `dataset.json` listing it, label files of the
right shape, and every voxel zero.

## What has no exporter

**COCO**, deliberately. It has no world geometry, spacing or frame of reference,
so an export discards the geometry that makes a medical annotation reproducible.
[The full reasoning](../explanation/refusals.md#coco).

## Check before you ship

```bash
medh5 verify case.medh5        # the source still matches its digests
```

## Related

- **[Converters](../reference/converters.md)** — every command and option.
- **[What the converters refuse, and why](../explanation/refusals.md)** — including every export refusal.
