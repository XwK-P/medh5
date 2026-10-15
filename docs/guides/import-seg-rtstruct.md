# Import DICOM SEG and RTSTRUCT

<!-- sequential -->

A segmentation drawn in a viewer usually leaves as a DICOM SEG, and a radiotherapy plan's outlines
as an RTSTRUCT. Both refer to a CT series by its frame of reference, so they import onto a sample
made from that series. This page imports a CT series, a SEG of a liver and a lesion, and a
structure set in which two ROIs share the name `Lesion`, then shows what the importers refuse.
Its blocks run in order; they need `pip install "medh5[dicom]"`.

## Import the series first

```python
import medh5
from medh5.io.dicom import from_dicom

from_dicom("dicom/ct", "patient.medh5", group_by="study")

with medh5.open("patient.medh5") as sample:
    assert list(sample.grids) == ["ct_tp0"]
    frame = sample.grids["ct_tp0"].frame_uid      # the series' FrameOfReferenceUID
```

The grid keeps the series' frame of reference. That frame is what the SEG and the structure set are
matched to below.

## Import a SEG

```python
from medh5.io.dicom_seg import from_dicom_seg

report = from_dicom_seg("dicom/seg.dcm", "patient.medh5", ann_id="organs")
print(report.format(verbose=True))

with medh5.open("patient.medh5") as sample:
    organs = sample.annotations["organs"]
    assert [c.key for c in organs.annotated_classes] == ["liver", "lesion"]
    voxels = organs.dense().sum(axis=(1, 2, 3))     # per class
```

Three things happen that a careless import gets wrong:

- **Frames are placed by geometry**, each from its own `PlanePositionSequence`, not by frame
  number. A SEG may leave out its empty frames and store the rest in any order.
- **Segments match by label, not by number.** Segments that share a label are one class: their
  union for a `BINARY` SEG, their maximum for a `FRACTIONAL` one, recorded in the report.
- **Classes are minted from the segments**, each keeping the segment's coded property type as an
  ontology binding and its property category as `properties["dicom_category"]`.

The report lists every decision the importer made. Read it once for each new source: a decision
you disagree with is a reason to fix the source, not the file.

## Import a structure set

A structure set's contours stay contours (§8.6), in the grid's world coordinates.
`rasterize=True` also writes a mask from them, `<id>_mask`, and records the rule it used.

```python
from medh5.io.rtstruct import from_rtstruct

report = from_rtstruct("dicom/plan.dcm", "patient.medh5", ann_id="contours", rasterize=True)
merged = report.of_kind("merged_rois")              # which ROIs became one class
assert merged and merged[0].severity == "decision"

with medh5.open("patient.medh5") as sample:
    contours = sample.annotations["contours"]
    assert sorted(c.key for c in contours.annotated_classes) == ["lesion", "liver"]
    planes = sorted(contours.by_plane())            # the slices the planner drew on
    assert "contours_mask" in sample.annotations
```

The sample already has a label set (the SEG minted it), so each ROI takes the class its `ROIName`
names, by key or by display name: `Liver` is `liver`, and both ROIs named `Lesion` are `lesion`.
**ROIs whose names make one class are combined**, and the report's `merged_rois` decision names
their ROI numbers. Their contours combine as §8.6 defines, so outlines of two ROIs that cross
are their union.

## Keep same-named ROIs apart

When two ROIs share a name but are different things (a lesion and its boost volume, say), name
the class for each ROI yourself with `roi_classes={ROINumber: key}`. A key the sample's label set
lacks is refused (`E402`); a sample with no label set mints it:

```python
from_dicom("dicom/ct", "apart.medh5", group_by="study")
from_rtstruct("dicom/plan.dcm", "apart.medh5", ann_id="contours", roi_classes={2: "lesion_boost"})

with medh5.open("apart.medh5") as sample:
    keys = sorted(c.key for c in sample.annotations["contours"].annotated_classes)
    assert keys == ["lesion", "lesion_boost", "liver"]
```

From the shell, `--roi NUMBER=KEY`, repeatable:

```bash
medh5 convert from-rtstruct dicom/plan.dcm apart.medh5 --id contours --roi 2=lesion_boost
```

An import where one same-named ROI's contour lies inside another's on a slice is refused, naming
the ROIs and the slice: combined, the inner one would become a hole in what both drew as an area.
Keep such ROIs apart with `roi_classes=`.

## What is refused

The importers never move geometry to make it fit, nor guess where it goes. A SEG whose frame of
reference matches no grid is refused (`E101`), with a request to name the grid. Named, a grid
whose frame is known and is another is refused too (`E414`), before the sample is touched,
because equal geometry in two frames is no correspondence:

```python
from medh5.errors import MEDH5ValidationError

before = open("case.medh5", "rb").read()
refused = None
try:
    from_dicom_seg("dicom/seg.dcm", "case.medh5", ann_id="organs_seg", grid="ct")
except MEDH5ValidationError as refusal:
    refused = refusal.code
assert refused == "E414"
assert open("case.medh5", "rb").read() == before    # untouched
```

The same holds for a SEG frame that lies between the grid's slices or starts off its first voxel,
a frame turned or spaced against the grid (`E405`), a `LABELMAP` SEG, an RTSTRUCT contour between
two slices or outside the grid, and exclusive-or contours that cross. Each refusal names what it
found, and [What the converters refuse](../explanation/refusals.md) says why.

## Related

- [Converters › DICOM SEG](../reference/converters.md#dicom-seg) and
  [› RTSTRUCT](../reference/converters.md#rtstruct): every option, the export, and the full list of
  refusals.
- [Import from DICOM](import-dicom.md): a directory of studies, one sample per patient.
- [Export to other formats](export.md): writing SEG and RTSTRUCT back out.
