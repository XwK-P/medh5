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
medh5 convert to-dicom-seg case.medh5 organs out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
```

`--source` is the original series the segmentation refers to; a SEG is only
meaningful against one.

**Repeat the flag once per file.** `--source` takes exactly one value per
occurrence, so a glob like `--source ct/*.dcm` expands to several arguments and
the command exits with `unrecognized arguments` before it does anything. In a
shell, build the repetition:

```bash
args=(); for f in ct/*.dcm; do args+=(--source "$f"); done
medh5 convert to-dicom-seg case.medh5 organs out.dcm "${args[@]}"
```

Overlapping segments survive.

**Export is binary, and it thresholds.** `to-dicom-seg` writes
`SegmentationTypeValues.BINARY` from `annotation.dense()`, and for a `probmap`
`dense()` already applies the annotation's stored `threshold` (default `0.5`).
So a probability of 0.49 does not become 1 — it becomes **background**, and is
gone from the exported SEG:

```python
ann.threshold                    # 0.5 unless the file says otherwise
# probabilities  0.0  0.1  0.3  0.49  0.5  0.7  0.9  1.0
# exported       0    0    0    0     1    1    1    1
```

Fractional values survive the *import* direction, not this one. Check
`ann.threshold` before exporting, and set it deliberately if the default is not
the operating point you want — or keep the probabilities in the `.medh5` and
export something else.

Writing goes through `highdicom` rather than assembling the IOD by hand, which
is how invalid SEGs get published.

## RTSTRUCT

```bash
medh5 convert to-rtstruct case.medh5 contours out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
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
