# Python API

Everything in this page is reachable from the top-level `medh5` namespace.

## Opening

```python
import medh5

with medh5.open("case.medh5") as s:        # -> Sample
    ...

with medh5.open_collection("shard.medh5c") as c:   # -> Collection (Mapping[str, Sample])
    for key, sample in c.items():
        ...

from medh5.collection import open_any
with open_any(path, key=None) as opened:   # a Sample or a Collection, whichever it is
    ...
```

`medh5.open` is lazy: it parses `/meta` and opens no arrays. Use it as a
context manager, or call `.close()`.

## Sample

### Identity and structure

```python
s.identity            # Identity: sample_id, subject_id, sex, laterality, bodypart
s.cohort              # Cohort: dataset_id, site_id, scanner_id, group_id, protocol
s.document            # SampleDocument — the whole /meta document
s.profiles            # frozenset of conformance profiles
s.version             # "1.0"
s.kind                # "sample" or "collection"
s.content_id          # "sha256:..." or None
s.summary()           # JSON-safe description; what `medh5 info --json` prints
```

### Timepoints

```python
s.timepoints                       # Timeline — indexable by position or id
s.timepoints[0].label              # "baseline"
s.timepoints["tp1"].days_from_baseline
s.is_longitudinal                  # len(timepoints) > 1

view = s.at("tp1")                 # a timepoint-scoped view of the whole sample
view.images                        # only tp1's images
view.annotations                   # only tp1's annotations
```

### Geometry

```python
s.grids                    # {grid_id: Grid}
s.reference_grid           # the sample's principal grid

g = s.grids["ct_tp0"]
g.shape, g.spacing, g.origin, g.direction
g.affine                   # (n+1, n+1) index -> world
g.spatial_shape            # spatial axes only, for a 4-D grid
g.coord_system             # "LPS"
g.timepoint                # "tp0"
g.frame_uid                # frame of reference
g.physical_size            # extent in mm
```

### Images

```python
img = s.images["CT_tp0"]
img.shape, img.dtype, img.chunks, img.nbytes
img.modality               # "CT"
img.value_type             # "quantitative"
img.value_units            # "HU"
img.rescale                # (slope, intercept)
img.levels                 # multiscale pyramid levels

img.read()                                # whole array, stored values
img.read(physical=True)                   # rescale applied
img.read((slice(0, 16), slice(0, 64), slice(0, 64)))   # one block
img.dataset                               # the underlying h5py dataset
```

`physical=True` applies `slope` and `intercept`. When an image declares
neither, the stored values are already physical and the flag changes nothing.

### Annotations

Every annotation, whatever its kind:

```python
ann = s.annotations["organs_tp0"]
ann.kind                   # "layers" | "labelmap" | "bitmask" | "instances" |
                           # "probmap" | "mask" | "boxes" | "obb" |
                           # "keypoints" | "points" | "contours" | "mesh" |
                           # "classification"
ann.task                   # "segmentation" | "detection" | "classification" | ...
ann.grid_id, ann.grid
ann.timepoints             # which visits it describes
ann.class_ids              # classes it declares
ann.annotated_class_ids    # classes that were examined (§11.3)
ann.is_annotated("spleen") # was this class looked for?
ann.is_fully_covered
ann.classes                # LabelClass objects
ann.quality_key, ann.prov
```

Voxel annotations add:

```python
ann.contains(class_id, (z, y, x))
ann.dense(["liver", "lesion"])          # (C, *spatial) bool
ann.dense(["liver"], roi=(slice(0,16),)*3)
ann.labelmap()                           # (*spatial) of class ids
ann.voxel_counts()                       # {class_id: count}
ann.class_bboxes()                       # {class_id: (S, 2) or None}
ann.instances()                          # needs instance identity — see below
```

See [Annotations](annotations.md) for the per-kind API.

### Transforms

```python
s.transforms                       # {transform_id: Transform}
t = s.transform_between("tp0", "tp1")   # resolved through the frame graph
t.kind                             # "affine" | "displacement" | "bspline" | "composite"
t.from_frame, t.to_frame
t.is_invertible                    # the mapping is invertible
t.inverse()                        # the *stored* inverse, when the file has one
t.transform_points(points)         # world -> world, in mm

from medh5.transforms.apply import jacobian_determinant, target_registration_error
target_registration_error(t, fixed_points, moving_points)   # {"mean", "max", ...}
jacobian_determinant(field, grid)                           # for a displacement field
```

`transform_between` searches the frame graph, composing chains and using
inverses where a transform declares one. It returns `None` when no path
exists — it does not invent one.

### Tracking

```python
tracking = s.tracks("lesion")
tracking.timepoints                     # ("tp0", "tp1")
tracking.states(instance_id)            # {timepoint: present|resolved|unexamined}
tracking.state_at(instance_id, "tp1")
tracking.is_new(instance_id)
tracking.is_resolved(instance_id)
tracking.is_persistent(instance_id)
tracking.class_conflicts()              # objects whose class changed between visits
tracking.unexamined()                   # {timepoint: instance ids nobody looked for}
tracking.coverage                       # {timepoint: class ids examined there}

for instance_id, track in tracking.items():
    track.volumes                       # {timepoint: mm^3}
    track.relative_change("tp0", "tp1")
    track.at("tp1")                     # the Observation, or None
```

See [Longitudinal](../guides/longitudinal.md).

### Integrity

```python
s.verify()                    # VerifyResult
s.verify(partial=["images/CT_tp0"])
s.verify().ok
s.compute_content_id()        # recompute rather than read the stored one
```

## Writing

```python
with medh5.create("out.medh5", sample_id="c1", subject_id="s1",
                  codec="balanced") as w:
    ...
```

The writer builds a temporary file and `os.replace`s it into position on a
clean exit. An exception aborts and leaves nothing behind.

### Document

```python
w.identity(sex="F", bodypart="abdomen")
w.cohort(dataset_id="d", site_id="site-A", group_id="family-7")
w.add_timepoint("tp1", index=1, label="fu1", days_from_baseline=92,
                date="2026-05-04", study_uid="pseudo:...")
w.label_set(label_set)
w.extra("mytool", {"anything": "json-serialisable"})
w.acquisition("CT_tp0", kvp=120, exposure_mas=180)   # imaging physics only
w.deidentification(method="dicom-psi-profile", date_shift_days=-117)
w.split(set_id="cv5", partition="train", fold=1)     # replaces the same set_id
```

### Provenance and quality

```python
tool = w.software("nnU-Net", "2.4.2")
rad  = w.person("RAD-07")
org  = w.organization("Site A")

act = w.activity("predict", agent=tool, tool="nnUNetv2_predict",
                 params={"fold": "all"}, inputs=["images/CT_tp0"])

w.set_quality("organs_tp0", status="reviewed", reviewed_by=[rad.id])
```

Activity types: `import`, `annotate`, `review`, `predict`, `resample`,
`register`, `derive`, `deidentify`, `transcode`, `other`.

### Grids and images

```python
w.add_grid("ct_tp0", shape=(192, 256, 256), spacing=(1.5, 0.8, 0.8),
           origin=(-144.0, -102.4, -102.4), direction=np.eye(3),
           coord_system="LPS", timepoint="tp0",
           frame_uid="pseudo:frame-a", patch_hint=(96, 96, 96))

w.add_image("CT_tp0", array, grid="ct_tp0", modality="CT",
            value_type="quantitative", value_units="HU",
            rescale_slope=1.0, rescale_intercept=-1024.0, prov=act)

w.add_pyramid("WSI", [level0, level1, level2],
              grid_levels=["l0", "l1", "l2"], modality="SM")
```

`patch_hint` tells the chunk optimiser what shape you will read.

### Annotations

```python
kind, stats = w.add_segmentation(
    "organs_tp0", grid="ct_tp0",
    masks={"liver": liver, "lesion": lesion},   # or probabilities= or instances=
    encoding="auto",                            # or an explicit kind
    annotated_classes=["liver", "spleen", "lesion"],
    ignore=uncertain_mask,
    prov=act, quality={"status": "approved"},
)

w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct_tp0",
            space="index", scores=[0.91], instance_ids=[7])
w.add_obb("nodules", centers, sizes, rotations, class_ids=["nodule"], grid="ct")
w.add_keypoints("landmarks", points, keypoint_classes, class_ids, grid="ct")
w.add_points("fiducials", points, grid="ct", correspondence="paired")
w.add_contours("rtstruct", polygons, grid="ct", space="world")
w.add_mesh("surface", vertices, faces, space="world")
w.add_classification("response", {"progressive": 1.0}, scope="sample",
                     timepoints=["tp1"])
```

### Transforms

```python
w.add_transform("tp0_to_tp1", kind="affine",
                from_frame="pseudo:frame-a", to_frame="pseudo:frame-b",
                matrix=matrix4x4, invertible=True)

w.add_transform("warp", kind="displacement",
                from_frame="a", to_frame="b",
                field=field, field_grid="ct_tp0", vector_space="world")
```

### Derived data

```python
w.build_index()                          # sampling indices for every voxel annotation
w.build_index(["organs_tp0"], max_coords=8192)
w.transcode_annotation("organs_tp0", "bitmask")
w.remove_annotation("old_seg")           # takes its index with it
```

## Amending

```python
with medh5.amend("case.medh5") as w:
    w.set_quality("organs", status="approved")
```

Copy-on-write: a new file is built from the old and replaced atomically.
Objects this reader does not understand — including ones written by a future
minor version — are copied through untouched, so amending never silently drops
what it cannot read.

## Validation

```python
from medh5.validate import validate_file, validate_paths

report = validate_file("case.medh5", level="strict", profiles=["seg"])
report.ok
report.errors            # [Diagnostic]
report.warnings
report.diagnostics[0].code       # "E102"
report.diagnostics[0].location
print(report.format(verbose=True))
```

Levels: `structural` → `semantic` → `integrity` → `strict`. Codes are stable
API; see spec §15.2 and `medh5.CODES`.

## Exceptions

```
MEDH5Error
├── MEDH5FileError        (also OSError)
├── MEDH5VersionError
├── MEDH5SchemaError
├── MEDH5ValidationError  (also ValueError) — carries .code
└── MEDH5IntegrityError
```

`MEDH5ValidationError.code` is the §15.2 diagnostic code where one applies,
so a caller can branch on the defect rather than on the message text.

## Sub-packages

| | |
|---|---|
| `medh5.torch` | [Datasets and samplers](torch.md) |
| `medh5.monai` | [MetaTensor adapter](torch.md#monai) |
| `medh5.io` | [Converters](converters.md) |
| `medh5.dataset` | [Cohort manifests, splits, statistics](../guides/cohorts.md) |
| `medh5.curation` | [Provenance, agreement, tracking, de-identification](curation.md) |
| `medh5.conformance` | [The conformance suite](../spec/conformance.md) |
| `medh5.storage` | [Codecs, chunking, recompression](storage.md) |

## Related

- **[Write and read your first sample](../tutorials/first-sample.md)** — this API end to end.
- **[How-to guides](../guides/index.md)** — the same calls, arranged by task.
- **[Sample document schema](schema.md)** — the fields the writer writes.
- **[Specification](../spec/medh5-1.0.md)** — the normative model behind it.
