# Annotations

Thirteen kinds, one contract. This page is what each stores and how to read it;
spec §6–§9 is the normative version.

## The common header

Every annotation carries the same header, whatever its kind:

```python
ann.kind                   # how it is stored
ann.task                   # what it is for: segmentation, detection, classification
ann.grid_id                # the geometry it lives on
ann.timepoints             # which visits it describes
ann.class_ids              # classes it declares
ann.annotated_class_ids    # classes that were examined (§11.3)
ann.closure                # "explicit" | "complete"
ann.quality_key, ann.prov  # links into quality and provenance
```

### Coverage

`class_ids` is what is here. `annotated_class_ids` is what was looked for.

```python
ann.is_annotated("spleen")   # was this class examined at all?
ann.is_fully_covered         # nothing appears that was not examined
```

A class in `annotated_class_ids` with zero voxels was **examined and absent** —
a usable negative example. A class in neither was not examined, and its absence
carries no information. Training code that treats the second as a negative is
learning from the annotator's schedule.

An **ignore** region (class `65535`) marks voxels that must not contribute to a
loss in either direction:

```python
ann.has_ignore_region
ann.dense([65535])
```

## Voxel annotations

Five encodings, one API. The encoding is a storage decision; the reading code
does not change.

```python
ann.contains(class_id, (z, y, x))       # one voxel, one class
ann.dense(["liver", "lesion"])          # (C, *spatial) bool stack
ann.dense(["liver"], roi=(slice(0,64),)*3)
ann.labelmap()                          # (*spatial) class ids, priority order
ann.voxel_counts()                      # {class_id: count}
ann.class_bboxes()                      # {class_id: (S, 2) index-space bounds}
```

| Encoding | Stores | Good for | Overlap | Instances |
|---|---|---|---|---|
| `labelmap` | one integer volume | many disjoint classes | no | no |
| `layers` | a stack of integer volumes | a few overlapping classes | yes | no |
| `bitmask` | packed bit planes | many classes, heavy overlap | yes | no |
| `instances` | per-object masks with ids | counting, tracking, per-object metrics | yes | yes |
| `probmap` | float per class | soft labels, model output | yes | no |

### Choosing one

You do not have to. `add_segmentation(encoding="auto")` measures the class
overlap graph, colours it greedily, and costs each candidate:

```python
kind, stats = w.add_segmentation("organs", grid="ct", masks=masks)
kind             # "layers"
stats.edges      # the overlapping class pairs it found
```

`medh5 seg stats FILE ANN` shows the same numbers for an existing annotation,
including what each alternative encoding would cost.

### Transcoding

Any encoding converts to any other without loss of what both can express:

```
$ medh5 seg convert case.medh5 organs --to bitmask --dry-run
```

```python
w.transcode_annotation("organs", "bitmask")
```

Going to an encoding that cannot represent something is **refused, not silently
dropped**. Three cases:

| From → to | Why it is refused |
|---|---|
| anything dense → `instances` | A dense encoding records which voxels belong to a class, never which object. The conversion would merge every object of a class into one and mint an `instance_id` belonging to none of them (§7.4). |
| `labelmap`/`layers` carrying an in-band ignore region → `bitmask`/`probmap` | Those express ignore as a separate `mask` annotation (§7.7). Dropping it turns "nobody examined these voxels" into "verified absent". Write the ignore region as its own `mask` and reference it with `ignore_mask=` first. |
| `instances` → dense | Allowed, and one-way: identity is not recoverable afterwards. |

### Instances

Per-object masks, ids and boxes, from an encoding that carries identity:

```python
for obj in ann.instances():
    obj.instance_id, obj.class_id
    obj.mask                     # bool array
    obj.box                      # (S, 2) at voxel edges
```

Ask a `layers` annotation and it tells you what to do instead:

```
MEDH5ValidationError: annotation 'organs' of kind 'layers' does not carry
instance identity, and it cannot be recovered from a dense encoding:
transcoding to `instances` would merge every object of a class into one and
mint an id that belongs to none of them (§7.4). Re-derive the objects from
whatever source had them.
```

`instance_id` is **sample-scoped**, which is what makes it a longitudinal join:
object 7 at baseline and object 7 at follow-up are the same lesion. See
[Longitudinal](../guides/longitudinal.md).

## Geometric annotations

### Boxes

Axis-aligned, `float32`, at **voxel edges**:

```python
w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct",
            space="index", scores=[0.91], instance_ids=[7])
```

```python
b = s.annotations["lesions"]
b.boxes[0]              # [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]
b.as_slices()[0]        # (slice(2, 6), slice(4, 9), slice(5, 11))
b.class_ids, b.scores, b.instance_ids
```

`space="index"` or `space="world"`. The half-voxel offset between a box edge
and a slice bound is the convention written down in spec §8.1, and it is the
one thing to get right when writing a detector.

**2-D boxes on a slice.** A box with a degenerate axis (`lo == hi`) plus
`slice_index` is the common radiology annotation — a lesion drawn on one slice
of a 3-D study (§8.2):

```python
w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct",
            space="index", slice_index=[37, 41])
```

`slice_index` is a **per-box column**: one plane for each box, shape `(N,)`, and
each plane inside the grid. All three are enforced by the writer, by
`as_slices()` on read, and by `medh5 validate` — the same rule at all three, so
a file written before the check still fails the same way (`E405`). A short
`slice_index` used to drop the boxes past its end and a plane outside the grid
used to be clamped to the edge, both silently.

### Oriented boxes

```python
w.add_obb("nodules", centers, sizes, rotations, class_ids=["nodule"], grid="ct")
```

`rotations` are rotation matrices. Centre, size and rotation round-trip through
the corner representation exactly.

### Keypoints and points

```python
w.add_keypoints("landmarks", points, keypoint_classes, class_ids,
                grid="ct", visibility=vis, skeleton="spine-17")
w.add_points("fiducials", points, grid="ct", correspondence="paired")
```

Keypoints are structured (a skeleton, per-point classes, visibility); points
are a bare set, used for registration landmarks and TRE.

### Contours

```python
from medh5.annotations.geometric import Polygon

w.add_contours("rtstruct",
               [Polygon(vertices=xyz, class_id=1, plane=(-1, 0), role="outer"),
                Polygon(vertices=hole, class_id=1, plane=(-1, 0), role="hole")],
               grid="ct", space="world")
```

`plane` says which axis the contour lies in; `role` distinguishes an outer
boundary from a hole inside it.

Contours stay contours. Rasterising them is a separate, opt-in step whose rule
is recorded in provenance, because "even-odd fill at voxel centres" is a
decision somebody should be able to find later.

### Meshes

```python
w.add_mesh("surface", vertices, faces, space="world", normals=normals,
           frame_uid="pseudo:frame-a")
```

`space="world"` names a physical frame, so a `frame_uid` is required — from the
argument or from the grid. A mesh whose coordinates are in millimetres without
saying *whose* millimetres is not reproducible, and the writer refuses it
(`E412`).

## Classification

Sample-level, per-timepoint, per-instance, or per-slice:

```python
w.add_classification("response", {"progressive": 1.0},
                     scope="sample", timepoints=["tp1"])
w.add_classification("birads", {"birads_4": 1.0}, scope="instance",
                     scope_ids=[7], schemes=["BI-RADS"])
```

```python
c = s.annotations["response"]
c.labels          # {class_id: score}
c.scope           # "sample" | "timepoint" | "instance" | "slice"
c.scope_ids
```

A **change** label — one that describes a difference between two visits — is a
classification whose `timepoints` names both. See
[Longitudinal](../guides/classification.md#change-labels-span-an-interval).

## Label sets

```python
from medh5 import LabelSet, LabelClass

labels = LabelSet("abdomen-v3", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ",
               codes=[{"system": "SCT", "code": "10200004"}]),
    LabelClass(3, "lesion", "Lesion", parents=[1]),
])
w.label_set(labels)
```

`parents` makes a DAG: a lesion is inside the liver, so a model trained on
`liver` can be evaluated against `liver ∪ lesion` without a hand-written
mapping. `LabelSet.digest()` is canonical, so two files either agree on the
vocabulary or they demonstrably do not — which is how `medh5 dataset check`
catches a cohort where class 3 means two different things.

Bundled vocabularies:

```
$ medh5 labels registry list
```

```python
from medh5.labels.registry import load
load("brats-subregions")
```

Ids `0` (background) and `65535` (ignore) are reserved.

## Related

- **[Segmentation](../guides/segmentation.md)** — writing and reading masks.
- **[Detection and boxes](../guides/detection.md)** — boxes without an off-by-one.
- **[Classification and change labels](../guides/classification.md)** — choosing a scope.
- **[Partial labels and coverage](../guides/partial-labels.md)** — `annotated_classes` in full.
- **[Specification §6–§9](../spec/medh5-1.0.md)** — the normative model.
