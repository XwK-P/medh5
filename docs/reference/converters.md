# Converters

Every importer and exporter lives in `medh5.io` and behind `medh5 convert`.

```bash
pip install "medh5[nifti,dicom]"        # reading
pip install "medh5[dicomseg]"           # writing DICOM SEG (highdicom)
```

## Conversion reports

Every conversion returns a report, because the interesting part of an import is
not that it succeeded — it is what it had to work out.

```python
from medh5.io.nifti import from_nifti

report = from_nifti({"CT": "ct.nii.gz"}, "case.medh5", masks={"liver": "liver.nii.gz"})
report.ok
report.of_kind("encoding")       # what encoding was chosen and why
for note in report.notes:
    note.severity                # "decision" | "guess" | "warning" | "info"
    note.kind, note.message, note.detail
```

The distinction that matters:

| | |
|---|---|
| **decision** | determined from the data — "the overlap graph has one edge, so `layers`" |
| **guess** | assumed, because the source did not say — "timepoints ordered by file mtime" |
| **warning** | something is wrong or lost |

`--report FILE` writes it as JSON. A guess is not a failure; it is the thing
you go back and check.

## What each converter does

Every refusal below is explained in
[What the converters refuse, and why](../explanation/refusals.md); this page is
the commands and the contracts.

### NIfTI

```
$ medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz --mask liver=liver.nii.gz
$ medh5 convert to-nifti case.medh5 CT out.nii.gz
$ medh5 convert to-nifti case.medh5 CT liver.nii.gz --annotation organs --class liver
```

```python
from medh5.io.nifti import from_nifti, to_nifti, import_seg_nifti
```

**Axis order.** NIfTI is `(x, y, z)`; medh5 is `(z, y, x)`. The array is
transposed and the spacing and direction are permuted to match, so the affine
still describes the same physical volume.

**Coordinate system.** NIfTI is RAS+; medh5 defaults to LPS (as DICOM does). The
conversion is a sign flip on the affine, `diag(-1, -1, 1, 1)`, and nothing else
— no resampling, no reinterpretation. `--coord-system RAS` keeps RAS.

**Round trip.** `from_nifti` → `to_nifti` reproduces the affine and the voxels
bit-for-bit.

**`scl_slope` and `scl_inter` become the image's rescale**, the way the DICOM
modality LUT does. The voxels keep the dtype the file stores them in;
`read(physical=True)` applies the scale and `read()` returns the stored values
(§4.2). A mask file carrying a scale of its own is thresholded after it is
applied.

`--assume-geometry` imports a file that declares no spatial mapping, recording
the fallback as a guess. Without it, such a file is refused.

### DICOM

```
$ medh5 convert from-dicom /studies out/ --group-by subject
$ medh5 convert from-dicom /studies out/ --modality CT --series 1.2.840...
```

```python
from medh5.io.dicom import scan_dicom, read_series, from_dicom, select_series
```

**Slices are ordered by geometry** — position projected on the slice normal, not
`InstanceNumber`, which is a display order and is regularly wrong.

**Spacing is measured** as the gap between consecutive slice origins, not read
from `SliceThickness`.

**Modality LUT is stored, not applied.** `RescaleSlope` and `RescaleIntercept`
become the image's `rescale`, so `read(physical=True)` gives HU and `read()`
gives what the scanner stored.

**Tags are an explicit allow-list.** Imaging physics goes to `acquisition`;
everything else stays out (§11.4).

**Enhanced multi-frame objects are reported, not read.** The importer reads
classic single-frame series. An Enhanced CT/MR/PET object keeps its geometry
in per-frame functional groups it does not read, so such files are skipped
and named in the report as a warning — an import that found nothing says
why. Convert them to classic instances first.

**`MONOCHROME1` is stored as written.** Lower stored values are brighter in
such a series; the importer never rewrites values (§4.2), records the
`PhotometricInterpretation` in `acquisition`, and warns in the report so a
model trained on intensity can invert them.

**Images are named by modality and visit** — `CT_tp0` on grid `ct_tp0`,
`PT_tp1` on `pt_tp1`. A study holding several series of one modality numbers
them in `SeriesInstanceUID` order — `MR_1_tp0`, `MR_2_tp0` — and the report
records which series got which name. The timepoint's `series_uids` is keyed by
those image ids.

**`--group-by subject`** (the default) resolves patient identity across studies
and emits one multi-timepoint sample per patient, falling back to one sample per
study — with a warning and a record — when identity cannot be established.

### DICOM SEG

```
$ medh5 convert from-dicom-seg seg.dcm case.medh5 --id organs
$ medh5 convert to-dicom-seg case.medh5 organs out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
```

**Frames are placed by geometry**, from each frame's `PlanePositionSequence` —
not by frame index. **Segments match by label, not by number.**

Import preserves overlapping segments and `FRACTIONAL` values. **Export does
not**: `to_dicom_seg` casts to boolean and writes `BINARY`, so fractional data
is thresholded on the way out.

Writing needs `highdicom` (`pip install "medh5[dicomseg]"`).

### RTSTRUCT

```
$ medh5 convert from-rtstruct plan.dcm case.medh5 --id contours
$ medh5 convert from-rtstruct plan.dcm case.medh5 --rasterize
$ medh5 convert to-rtstruct case.medh5 contours out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
```

**Contours stay contours** (§8.6), in world coordinates. `--rasterize` is opt-in,
and the rule it used goes into the provenance graph.

Each imported polygon records the slice it lies on as `contour_plane`
`(0, k)`, in the grid's index space, so `by_plane()` on the result groups the
structure set the way the planner drew it.

### nnU-Net v2

```
$ medh5 convert from-nnunet /Dataset001_Liver out/
$ medh5 convert to-nnunet /out case1.medh5 case2.medh5 --dataset-name Dataset001_Liver
```

Each case's channels and per-class masks are bundled into one sample.

**nnU-Net's class ids are kept**, so a model trained against the original
dataset still means the same thing. **Region labels become §5.1 DAG parents**: a
region that is the union of two components is a class whose components name it
as a parent, which is exactly what the hierarchy is for.

The parsed `dataset.json` is stashed in `extra["nnunetv2"]`, so `to-nnunet`
reproduces the original dataset definition rather than inventing one.

### COCO

Not supported, deliberately —
[why](../explanation/refusals.md#coco).

## Related

- **[Import from DICOM](../guides/import-dicom.md)** — a directory of studies to one file per patient.
- **[Import from NIfTI and nnU-Net](../guides/import-nifti.md)** — volumes and masks as separate files.
- **[Export to other formats](../guides/export.md)** — getting data back out.
- **[Migrate from 0.x](../guides/migrate-0x.md)** — the one-way door.
- **[What the converters refuse, and why](../explanation/refusals.md)** — every refusal, with the reasoning.
