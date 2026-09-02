# What the converters refuse, and why

Every converter in medh5 would rather fail than guess. This page is the list of
what each one refuses, and the reasoning, because a refusal you do not
understand looks like a bug and a refusal you do understand is usually telling
you something true about your data.

The principle underneath all of it: **an import that silently invents geometry
produces a file that is wrong in a way nothing downstream can detect.** A
crashed import costs an afternoon. A successful import of a fabricated grid
costs a study.

## These refusals carry no diagnostic code

§15.2's [code table](../reference/diagnostic-codes.md) describes conditions found
**in a MEDH5 file**. A DICOM series that disagrees with itself, or two NIfTI
volumes on different grids, are not a MEDH5 file yet — so there is no code for
them, and borrowing one would be worse than having none.

`E202` is an image disagreeing with its grid; neither NIfTI volume is a grid.
`E101` is a reference to a grid that does not exist; nothing here is referenced.
Reporting either would send a reader looking for a defect in a file that was
never written. The converters raise with a message naming the offending input
instead.

## NIfTI

**Disagreeing grids are refused.** Two NIfTI files with different affines are
not the same grid, and `from_nifti` will not resample one onto the other to make
the import work. Resampling is a decision with a filter, an interpolation order
and a loss of information attached to it; make it deliberately, in your own
code, and say so.

**A file that declares no geometry is refused.** `sform_code == qform_code == 0`
is NIfTI stating that it carries no spatial mapping — voxel indices and nothing
else. nibabel still hands back an affine, rebuilt from `pixdim`, and importing
that mints a world grid nobody measured. `--assume-geometry` takes the fallback
deliberately, and it is then recorded as a **guess** in the report.

**An sform/qform disagreement is recorded, not resolved in silence.** When both
codes are set and the matrices describe different geometry — the signature of a
file one tool updated and another did not — the sform is used, as is
conventional, and the report carries a guess naming the difference. A reader
that prefers the qform will place that volume somewhere else, and you want to
know which files those are before you train on them.

## DICOM

**The stack must agree with itself.** A series is one volume with one geometry
and one modality LUT, so `ImageOrientationPatient`, `PixelSpacing` and
`RescaleSlope`/`RescaleIntercept` are read once — but from *every* slice, and
checked. A stack whose slices disagree is refused, naming the offending
`SOPInstanceUID`.

Taking the first slice's answer is not a gentler version of this; it is a
different, silent one. A per-slice rescale is ordinary in PET, and collapsing it
to slice 0 reports the wrong activity everywhere that LUT does not apply. One
rotated slice placed on the first slice's direction matrix puts those voxels
where the scanner never did. Neither shows up in the output.

**Irregular slice spacing is refused.** Spacing is measured as the gap between
consecutive slice origins, not read from `SliceThickness` — which is the slab,
and is regularly different. If the gaps disagree by more than a tolerance, the
stack is not a regular grid, and pretending otherwise silently distorts every
distance measured from it.

**A missing or malformed tag is refused**, rather than defaulted, or left to
surface as a `TypeError` from inside numpy. A series whose slices all omit
`PixelSpacing` used to fall through to 1 mm — which every slice then agreed on,
so the stack was written with an in-plane size the source never stated, and
nothing recorded the assumption.

Cardinality is checked as well as presence, for a reason worth stating: a
`PixelSpacing` of three values reads perfectly well as its first two, and a
stack where *every* slice carries the same wrong length passes an agreement
check that only compares slices to each other.

**Identity is never inferred from filenames, dates or accession numbers.** When
`--group-by subject` cannot establish identity — usually after a
de-identification pass randomised `PatientID` — it falls back to one sample per
study, warns, names the affected inputs and records the fallback. Guessing that
two studies belong to one patient is how a subject ends up in both the training
and the test set.

**Tags are an explicit allow-list.** Imaging physics — kVp, exposure, kernel,
TR/TE — goes to `acquisition`; everything else stays out (§11.4). A wholesale
tag copy is how a "de-identified" export carries a patient name into a training
set.

## DICOM SEG

**Segments match by label, not by number.** DICOM numbers segments 1..N
positionally. If your label set uses ids 1 and 3, importing by number would
quietly turn "lesion" into whatever class holds id 2. `from_dicom_seg` matches
on `SegmentLabel` and records the mapping as a decision.

**Writing requires `highdicom`** rather than assembling the IOD by hand.
Building per-frame functional groups yourself is how invalid SEGs get published.

## RTSTRUCT

**Export refuses a mask.** `to-rtstruct` on a voxel annotation is an error, not
a marching-squares fallback: the contours it would produce are not the contours
anyone drew, and an RTSTRUCT is a clinical document that asserts they are.

**Rasterisation is opt-in and recorded.** Contours stay contours (§8.6), in
world coordinates. When you do rasterise, the rule used — even-odd fill at voxel
centres, holes excluded — goes into the provenance graph, because somebody will
need to know a year later.

Hole detection groups contours by plane in the grid's **index** space. In world
space the *z* of a contour varies within its own slice under a real oblique
orientation, so grouping there finds no holes at all.

## nnU-Net v2

**Every channel and label volume must share one grid.** A case's channels are
registered to each other by construction — that is what makes them channels — so
a second channel at a different spacing or origin, or a label volume some other
tool resampled, is refused rather than filed onto channel 0's grid (§3.2).

This is the same `_same_grid` check `from_nifti` has always applied. Import used
to keep whichever grid it saw first, which wrote the later volumes' voxels
intact and their position silently wrong.

**Export refuses a class the sample does not have** (`E402`). `to-nnunet`
matches classes by **id**, not by name — which is precisely why import keeps
nnU-Net's own integers. Matching by name looks equivalent and is not: import
sanitises `dataset.json` names into label-set keys, so a dataset naming a class
`"Tumour Core"` or `"GTV"` resolved nothing. The miss fell into a bare
`except: continue`, and the export came out listing every class, with label
files of the right shape in which every voxel was 0.

## COCO

**Not supported, deliberately.**

COCO is a 2-D polygon/RLE format with no world geometry, no spacing and no frame
of reference. Importing one means inventing a grid; exporting one means
discarding the geometry that makes a medical annotation reproducible. Neither
direction can be done without a silent lie, and every other converter here is
built on not telling one.

A 2-D-native path can be added in a minor version if a concrete need appears —
§3.6 already supports 2-D grids.

## Related

- **[Converters](../reference/converters.md)** — the commands and their options.
- **[The data model](data-model.md)** — why geometry is stated once and never guessed.
- **[Diagnostic codes](../reference/diagnostic-codes.md)** — the codes that *do* describe a MEDH5 file.
