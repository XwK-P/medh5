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

`--annotation` alone writes the annotation's labelmap, with every ignored voxel
as its ignore id (`65535` unless the header says otherwise). `--class` writes
one class as a 0/1 mask, which cannot say "ignored" — its `0` says "examined,
absent" — so it refuses an annotation with an ignore region: export the
labelmap instead. For the same reason it refuses a class the annotation never
examined (one outside `annotated_class_ids`): its empty mask would be a
verified negative to whoever imported it back. The output is a `.nii` or
`.nii.gz` file.

The round trip from `from_nifti` reproduces the affine and the voxels; a file
that had a scale comes back as its physical values unless `--stored` writes the
stored voxels and the scale, as the file had them.

## DICOM SEG

```bash
pip install "medh5[dicomseg]"
medh5 convert to-dicom-seg case.medh5 organs out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
```

`--source` is the original series the segmentation refers to; a SEG is only
meaningful against one. The annotation is one with classes — a `labelmap`,
`layers`, `bitmask`, `instances` or `probmap` — written one segment per
class; any other kind, a `mask` among them, is refused (`E401`).

**Repeat the flag once per file.** `--source` takes exactly one value per
occurrence, so a glob like `--source ct/*.dcm` expands to several arguments and
the command stops with `error: unexpected argument 'ct/2.dcm' found` (a usage
error, exit status 2) before it does anything. In a shell, build the
repetition:

```bash
args=(); for f in ct/*.dcm; do args+=(--source "$f"); done
medh5 convert to-dicom-seg case.medh5 organs out.dcm "${args[@]}"
```

Overlapping segments survive.

**Only the classes the annotation examined are written.** A segment's 0 says
"examined, absent", which a class outside `annotated_class_ids` never was, so
such a class is left out, and the report records it (`unexamined`);
`classes=` naming one is refused, and so is an annotation that examined none.

**A `probmap` goes out binary, unless you ask for fractional.** By default
(`--segmentation-type binary`) `to-dicom-seg` writes a `BINARY` SEG from
`annotation.dense()`, and for a `probmap` `dense()` applies the annotation's
stored `threshold` (default `0.5`). So a probability of 0.49 does not become
1 — it becomes **background**, and is gone from the exported SEG; the report
says where it cut (`threshold`):

<!-- illustrative -->
```python
ann.threshold                    # 0.5 unless the file says otherwise
# probabilities  0.0  0.1  0.3  0.49  0.5  0.7  0.9  1.0
# exported       0    0    0    0     1    1    1    1
```

`--segmentation-type fractional` writes the probabilities themselves, as a
`FRACTIONAL` SEG (`--fractional-type PROBABILITY`, the default, or
`OCCUPANCY`). It takes a `probmap` only; any other kind is refused, since it
has no fractional values to write. DICOM stores each value as an integer out
of 255, so every probability comes back within 1/510 of what it was. The
report's `fractional` note gives the largest error, and how many voxels the
rounding moved across the annotation's threshold. A SEG has no field for
`threshold` or `normalized`, so neither goes with it: read back, the map has
the default threshold of 0.5.

```bash
medh5 convert to-dicom-seg case.medh5 probs probs.dcm "${args[@]}" \
    --segmentation-type fractional --report probs.json
```

In Python it is `to_dicom_seg(..., segmentation_type="fractional")`, with
`fractional_type=` beside it.

**Each segment says what it is.** A SEG segment must carry a property type, a
property category and an algorithm type, and each comes from the sample:

| DICOM | From |
|---|---|
| property type | the class's first code whose system has a DICOM designator: `SNOMED-CT` (`SCT`), `RadLex` (`RADLEX`), `FMA`, `UBERON`, `LOINC` (`LN`), `ICD-10` (`I10`), `DCM` |
| property category | `properties["dicom_category"]` (an import writes it), else `category`: `organ`, `vessel`, `bone` → SCT 91723000 "Anatomical Structure"; `lesion` → SCT 49755003 "Morphologically Abnormal Structure"; `device` → SCT 260787004 "Physical object"; `region` → SCT 309825002 "Spatial and Relational Concept" |
| type modifier | `laterality`: `left` → SCT 7771000, `right` → SCT 24028007, `bilateral` → SCT 51440002 |
| algorithm type | the annotation's provenance: `predict` by a software agent with a version → `AUTOMATIC`, naming that software, family DCM 123110 "Artificial Intelligence"; `annotate` or `review` → `MANUAL` |

DICOM allows none of the three to be left out. So where the sample states one
the table cannot carry — no code, a system with no designator, no category, no
provenance — the export writes the generic value (SCT 91723000 "Anatomical
Structure", `MANUAL`) and records a guess (`property_type`,
`property_category`, `algorithm_type`) naming each class it did so for. It
never writes a more specific concept than the sample states, and never invents
a designator. Give the classes codes, categories and laterality before
exporting, and the guesses go away.

**An ignore region is refused.** A segment's 0 says the voxel was examined
and the structure is absent — binary or fractional, a SEG has no value for "not
looked at". The voxels an
annotation ignores (§7.7), in band or through its `ignore_mask`, would go out
as that 0, so `to-dicom-seg` refuses an annotation with any, and says how many.
`to-nifti --annotation` writes the labelmap, with each ignored voxel as the
annotation's ignore id.

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

**Markers go out as POINT ROIs.** `--points ANN` (`points=` in Python) writes a
`points` annotation beside the contours — the one an import of POINT ROIs
made, or landmarks of your own — one ROI per class, each row a `POINT`
contour of one point, with `RTROIInterpretedType` `MARKER` (contour ROIs are
`ORGAN`):

```bash
medh5 convert to-rtstruct case.medh5 contours out.dcm \
    --points contours_points --source ct/1.dcm --source ct/2.dcm ...
```

Every row needs a class (`E402` otherwise), and a class in both annotations is
refused: an ROI is drawn as areas or as points, not both.

Each ROI takes its class's display name and the label set's colour for it, and
text beyond ASCII is written as UTF-8 under `SpecificCharacterSet` `ISO_IR 192`.
The names are used only when every class exported has its own, and each fits
an `ROIName` (1 to 64 printable characters, no backslash); otherwise every ROI
is named for its class key, and the report says why (`roi_names`).
**Crossing contours mean their union** in medh5, while readers that combine an
ROI's contours by exclusive-or or even-odd fill cancel the overlap, so two
contours of one class that cross on a plane are a warning
(`crossing_contours`): merge them into one outline where such a reader matters.

## nnU-Net v2

```bash
medh5 convert to-nnunet /out case1.medh5 case2.medh5 --dataset-name Dataset001_Liver
```

Classes are matched by **id**, not by name — which is why import keeps nnU-Net's
own integers. The label table is every case's classes, and a case is written
against it only if it examined every one (`annotated_class_ids`): 0 in an
nnU-Net label volume is a verified negative, which a class nobody looked for is
not, so such a case is refused — `--class` exports a subset every case
examined. A class a case cannot name is refused (`E402`) rather than
skipped; a skipped class produces a `dataset.json` listing it, label files of
the right shape, and every voxel zero.

A case without the annotation is refused, since nnU-Net's training cases are
labeled; `--unlabeled test` writes it to `imagesTs` instead. Label values are
written `0..K`, as nnU-Net requires, and `dataset.json` keeps the class id of
any value that is not its own (`medh5_class_ids`), so `from-nnunet` restores it.
Every case is checked before any file is written, so a refused case leaves no
half-written dataset behind.

**Test cases follow a split.** `--split SET_ID` places each case by its
claim in that split set (§12.3): a case claiming `test` is written to
`imagesTs`, its labels to `labelsTs`, and one claiming `train` or `val` to
`imagesTr`, where nnU-Net cross-validates. A case the set does not place, or
holds out (`holdout`, `unassigned`), is refused: a mistyped set id would
otherwise train on every test case. `from-nnunet --include-test` writes these
claims (set `nnunetv2:<dataset>`), so a dataset comes back with its test
cases where they were:

```bash
medh5 convert from-nnunet /Dataset001_Liver cases/ --include-test
medh5 convert to-nnunet /out cases/*.medh5 --dataset-name Dataset001_Liver \
    --split nnunetv2:Dataset001_Liver
```

**An existing dataset directory must hold only what the export writes.** An
identical re-export replaces its own files and `dataset.json`, and files that
are not NIfTI are left alone. A NIfTI file in `imagesTr`, `labelsTr`,
`imagesTs` or `labelsTs` that the export would not write — a case or a
channel since dropped, a case moved between `imagesTr` and `imagesTs`, the
other file ending — refuses the export with nothing written: nnU-Net trains on
every case file it finds there, whatever `dataset.json` says. Export to a new
directory, remove those files first, or replace the dataset.

**`--replace` swaps a dataset whole.** The directory must be an nnU-Net
dataset — a directory, not a link, holding a `dataset.json` — or the export
is refused. The new dataset is built in a hidden sibling directory on the
same file system, with every file that is not the export's copied in
(`splits_final.json`, notes, anything outside the case folders); two renames
then swap it in, the old dataset aside and the new one into its place, and
if the second fails the first is undone, so the directory always holds one
whole dataset. The old tree is deleted last. The report's `replace` note
lists the NIfTI files removed and the files carried over.

nnU-Net normalises a channel by its name in `dataset.json`: `CT` selects its CT
scheme, and any other name z-scoring. A stashed `dataset.json` keeps its own
names; without one a channel is named for its image, and a CT image `CT`.

## What has no exporter

**COCO**, deliberately. It has no world geometry, spacing or frame of reference,
so an export discards the geometry that makes a medical annotation reproducible.
[The full reasoning](../explanation/refusals.md#coco).

Nothing writes these out to another format either; they stay in the `.medh5`:

- **Images as DICOM.** The DICOM importer reads a series, and no exporter
  writes one: a round trip through DICOM is an annotation's, as a SEG or an
  RTSTRUCT against the original series.
- **Boxes, keypoints and meshes**, which none of the formats above carries.
- **Transforms**.
- **Clinical records**, apart from their JSON (`medh5 clinical export`).

## Check before you ship

```bash
medh5 verify case.medh5        # the source still matches its digests
```

## Related

- **[Converters](../reference/converters.md)** — every command and option.
- **[What the converters refuse, and why](../explanation/refusals.md)** — including every export refusal.
