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

## NIfTI

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

**Coordinate system.** NIfTI is RAS+; medh5 defaults to LPS (as DICOM does).
The conversion is a sign flip on the affine, `diag(-1, -1, 1, 1)`, and nothing
else — no resampling, no reinterpretation. `--coord-system RAS` keeps RAS.

**Disagreeing grids are refused.** Two NIfTI files with different affines are
not the same grid, and `from_nifti` will not resample one onto the other to
make the import work. Resample them yourself, deliberately, and say so.

**A file that declares no geometry is refused.** `sform_code == qform_code == 0`
is NIfTI stating that the file carries no spatial mapping — it has voxel indices
and nothing else. nibabel still hands back an affine, rebuilt from `pixdim`, and
importing that mints a world grid nobody measured. Pass `--assume-geometry`
(Python: `assume_geometry=True`) to take that fallback deliberately; it is then
recorded as a **guess** in the report.

**An sform/qform disagreement is recorded, not resolved in silence.** When both
codes are set and the two matrices describe different geometry — the signature of
a file one tool updated and another did not — the sform is used, as is
conventional, and the report carries a guess naming the difference. A reader that
prefers the qform will place that volume somewhere else, and you want to know
which files those are before you train on them.

Round trip: `from_nifti` → `to_nifti` reproduces the affine and the voxels
bit-for-bit.

## DICOM

```
$ medh5 convert from-dicom /studies out/ --group-by subject
$ medh5 convert from-dicom /studies out/ --modality CT --series 1.2.840...
```

```python
from medh5.io.dicom import scan_dicom, read_series, from_dicom, select_series
```

**Slices are ordered by geometry.** Position projected on the slice normal —
not `InstanceNumber`, which is a display order and is regularly wrong. There is
a test whose fixture numbers instances backwards on purpose.

**Spacing is measured.** The gap between consecutive slice origins, not
`SliceThickness` (which is the slab, and is regularly different). If the gaps
disagree by more than a tolerance the stack is **refused**: it is not a regular
grid, and pretending otherwise silently distorts every distance measured from
it.

**Modality LUT is stored, not applied.** `RescaleSlope` and `RescaleIntercept`
become the image's `rescale`, so `read(physical=True)` gives HU and `read()`
gives what the scanner stored. Baking the LUT in loses the stored values.

**Tags are an explicit allow-list.** Imaging physics — kVp, exposure, kernel,
TR/TE — goes to `acquisition`. Everything else stays out, per §11.4. A wholesale
tag copy is how a "de-identified" export carries a patient name into a training
set.

**Grouping.** `--group-by subject` (the default) resolves patient identity
across studies and emits one multi-timepoint sample per patient. When identity
cannot be established — usually a de-identification pass that randomised
`PatientID` — it falls back to one sample per study, warns, names the affected
inputs, and records the fallback. It never infers identity from filenames,
dates or accession numbers.

## DICOM SEG

```
$ medh5 convert from-dicom-seg seg.dcm case.medh5 --id organs
$ medh5 convert to-dicom-seg case.medh5 organs out.dcm --source ct/*.dcm
```

**Frames are placed by geometry**, from each frame's
`PlanePositionSequence` — not by frame index. Overlapping segments and
`FRACTIONAL` segmentations both survive the round trip.

**Segments match by label, not by number.** DICOM numbers segments 1..N
positionally. If your label set uses ids 1 and 3, importing by number would
quietly turn "lesion" into whatever class holds id 2. `from_dicom_seg` matches
on `SegmentLabel` and records the mapping as a decision.

Writing needs `highdicom`, which is the reference implementation of the
Segmentation IOD. Building per-frame functional groups by hand is how invalid
SEGs get published.

## RTSTRUCT

```
$ medh5 convert from-rtstruct plan.dcm case.medh5 --id contours
$ medh5 convert from-rtstruct plan.dcm case.medh5 --rasterize
$ medh5 convert to-rtstruct case.medh5 contours out.dcm --source ct/*.dcm
```

**Contours stay contours** (§8.6), in world coordinates. Rasterising is opt-in,
and the rule it used — even-odd fill at voxel centres, holes excluded — goes
into the provenance graph, because somebody will need to know a year later.

Hole detection groups contours by plane in the grid's **index** space. In world
space the *z* of a contour varies within its own slice under a real oblique
orientation, so grouping there finds no holes at all.

**Export refuses a mask.** `to-rtstruct` on a voxel annotation is an error, not
a marching-squares fallback: the contours it would produce are not the contours
anyone drew.

## nnU-Net v2

```
$ medh5 convert from-nnunet /Dataset001_Liver out/
$ medh5 convert to-nnunet /out case1.medh5 case2.medh5 --dataset-name Dataset001_Liver
```

Each case's channels and per-class masks are bundled into one sample.

**nnU-Net's class ids are kept**, so a model trained against the original
dataset still means the same thing. **Region labels become §5.1 DAG parents**:
a region that is the union of two components is a class whose components name
it as a parent, which is exactly what the hierarchy is for.

The parsed `dataset.json` is stashed in `extra["nnunetv2"]`, so `to-nnunet`
reproduces the original dataset definition rather than inventing one.

## COCO

Not supported, deliberately.

COCO is a 2-D polygon/RLE format with no world geometry, no spacing and no
frame of reference. Importing one means inventing a grid; exporting one means
discarding the geometry that makes a medical annotation reproducible. Neither
direction can be done without a silent lie, and every other converter here is
built on not telling one.

A 2-D-native path can be added in a minor version if a concrete need appears —
§3.6 already supports 2-D grids.

## Migrating from 0.x

1.0 ships a *reader* for the 0.x layout, not an implementation of it. `medh5
migrate` is the one-way door.

```
$ medh5 migrate old/*.medh5 -o new/ --write-labels labels.json
# review labels.json, edit the keys and ids
$ medh5 migrate old/*.medh5 -o new/ --label-set labels.json \
      --group-by subject --subject-key extra.patient_id --report migration.json
```

Four things are not mechanical, and each is reported per file:

**Voxel encoding.** 0.x stored one boolean volume per mask name. 1.0 measures
the overlap graph and picks an encoding — which changes the size and nothing
else.

**Box corners.** 0.x boxes were slice-like integers `[min, max)`; 1.0 boxes sit
at voxel edges. Every corner shifts by −0.5, which is a real change in the
numbers, and it is reported as one. `[[2, 6]]` becomes `[[1.5, 5.5]]` and still
slices `2:6`.

**Label set.** 0.x had names, not classes. Mask names and box labels become
keys with minted ids, written to a sidecar so you can review and correct them
*before* converting the cohort — and ids are minted once for the whole cohort,
not per file, so `liver` is not id 1 in one sample and id 2 in the next.

**Grouping.** A 0.x file is study-scoped and carries no subject key, so the
default is one sample per file with a single `tp0`. `--group-by subject` merges
files sharing a key you name, ordering by date where there is one and by mtime
otherwise — and says which it used.

Instance correspondence is **never** inferred across merged files. Asserting
that lesion 2 at baseline is lesion 2 at follow-up would fabricate exactly the
tracking ground truth §7.4 exists to record.

A 0.x reader opening a 1.0 file fails on the missing `schema_version`, which is
the correct loud failure.
