# Python API

The core of the package is reachable from the top-level `medh5` namespace;
where a name lives in a sub-package, the example imports it from there.

The package is a layer over the format engine, which is written in Rust
([Rust crate](rust.md)); the classes here are its Python face, typed for
`mypy --strict`, and the arrays they hand back are NumPy arrays.

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

`medh5.open` (also exported as `medh5.open_sample`) is lazy: it parses `/meta`
and opens no arrays. Use it as a context manager, or call `.close()`. It is
read-only; every edit goes through `medh5.amend`, which is copy-on-write.

## Sample

### Identity and structure

```python
s.identity            # Identity: sample_id, subject_id, sex, laterality, bodypart
s.cohort              # Cohort: dataset_id, site_id, scanner_id, group_id,
                      #         acquisition_protocol
s.label_set           # LabelSet, or None
s.document            # SampleDocument — the whole /meta document
s.profiles            # frozenset of conformance profiles
s.version             # "1.0"
s.kind                # "sample" or "collection"
s.content_id          # "sha256:..." or None
s.path                # where it was opened from
s.summary()           # JSON-safe description; what `medh5 info --json` prints
```

### Timepoints

```python
s.timepoints                       # Timeline — indexable by position or id
s.timepoints[0].label              # "baseline"
s.timepoints["tp1"].days_from_baseline
s.timepoints.ids                   # ("tp0", "tp1"), in acquisition order
s.timepoints.interval_days("tp0", "tp1")
s.is_longitudinal                  # len(timepoints) > 1

view = s.at("tp1")                 # a timepoint-scoped view of the whole sample
view.images                        # only tp1's images
view.annotations                   # only tp1's annotations
```

Time is stated once, on the grid, and inherited by everything on it. A
single-timepoint sample may leave `timepoint` off its grids (spec §3.7); the
reader resolves those to the one declared timepoint, so `s.at("tp0")`,
`Image.timepoint`, `Annotation.timepoints` and `s.tracks()` answer the same way
whether or not the writer wrote the attribute.

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
img.dataset                               # the stored dataset (a view, below)
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

<!-- illustrative -->
```python
ann.contains(class_id, (z, y, x))
ann.dense(["liver", "lesion"])          # (C, *spatial) bool
ann.dense(["liver"], roi=(slice(0,16),)*3)
ann.labelmap()                           # (*spatial) of class ids
ann.voxel_counts()                       # {class_id: count}
ann.class_bboxes()                       # {class_id: (S, 2) or None}
ann.instances()                          # needs instance identity — see below
```

Two reads span objects, so they are on the sample:

```python
s.ignore_region("organs_tp0")            # (*spatial) bool: the §7.7 ignore region,
                                         # in band or in its sibling mask
s.valid_region("CT_tp0", roi=(slice(0, 8),) * 3)   # the image's valid_mask (§4.4),
                                         # or all True where it declares none
```

These are what the loaders put in `item["ignore"]` and `item["valid"]`; see
[PyTorch and MONAI](torch.md#a-batch).

See [Annotation kinds](annotations.md) for the per-kind API.

### Transforms

```python
s.transforms                       # {transform_id: Transform}
t = s.transform_between("tp0", "tp1")   # resolved through the frame graph
s.resolve_frames(frame_a, frame_b) # the same, between two frame uids; memoised per handle
t.kind                             # "identity" | "affine" | "displacement" |
                                   # "bspline" | "composite"
t.from_frame, t.to_frame
t.is_invertible                    # the mapping is invertible
t.inverse()                        # the *stored* inverse, when the file has one
t.transform_points(points)         # world -> world, in mm

from medh5.transforms import jacobian_determinant, target_registration_error
target_registration_error(t, fixed_points, moving_points)   # {"mean", "max", ...}
jacobian_determinant(field, grid)                           # for a displacement field
```

`transform_between` accepts a timepoint id, a grid id or a frame uid at either
end, matched in that order. It searches the frame graph, composing chains and
traversing a link backwards only where its inverse can be *evaluated* — an
affine's or identity's analytic inverse, or a stored `inverse_id` — not merely where
`invertible=True` is declared. It returns `None` when no path exists — it does
not invent one — and raises `KeyError` for a key that is not a timepoint, a grid
or a frame of reference in the sample, so a mistyped `"TP1"` is not mistaken for
"no registration exists". [Registration between visits](../guides/registration.md)
covers what `None` means and when to store an inverse.

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
s.verify().unattested         # undigested datasets inside objects content_id covers
s.compute_content_id()        # recompute rather than read the stored one
```

`ok` is `False` when a digest mismatches, when the root does, or — in a file
that declares a `content_id` — when a dataset inside a grid, image, annotation
or transform carries no digest at all (`unattested`).

### Sampling index

```python
s.index                       # {ann_id: SamplingIndex} — every entry in the file
s.fresh_indices               # the ids whose source_digest still matches (§13.3)
```

A stale entry is ignored by the samplers and the statistics, never trusted; see
[Storage](storage.md#the-sampling-index).

### Stored objects

`s.root` is the sample's HDF5 root, and `img.dataset`, `ann.group`, `t.group`,
`s.index[...].group` and the writer's `w.handle` are objects under it. They are
the package's own views (`medh5.nodes`), not `h5py` objects:

```python
ds = s.root["images/CT"]          # Dataset: shape, dtype, chunks, filters, attrs
ds.shape, ds.dtype, ds.chunks
block = ds[0:4, :, :]             # one read, NumPy out; numpy.asarray(ds) reads all
ds.attrs["modality"]              # attributes, decoded as spec §2.5 fixes
"images" in s.root, s.root["annotations"].keys()
```

A view lives as long as the sample it came from: after `s.close()` using one
raises `MEDH5FileError`. For HDF5 features the views do not cover, open the
file with `h5py` (`pip install "medh5[h5py]"`); it is plain HDF5.

## Writing

<!-- illustrative -->
```python
with medh5.create("out.medh5", sample_id="c1", subject_id="s1",
                  codec="balanced") as w:
    ...
```

`create` and `amend` both return a `SampleWriter`. It builds a temporary file
and `os.replace`s it into position on a clean exit; an exception aborts and
leaves nothing behind. Without a `with` block, call `w.commit()` to finish or
`w.abort()` to discard.

### Document

<!-- illustrative -->
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

<!-- illustrative -->
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
`add_image` also takes `channel_names` for a channel axis, `window_center` /
`window_width` display presets, and `valid_mask` — the id of a `mask`
annotation delimiting the acquired field of view (§4.4):

<!-- illustrative -->
```python
w.add_mask("fov", fov, grid="ct_tp0")        # a bool volume, no classes
w.add_image("CT_tp0", array, grid="ct_tp0", modality="CT", valid_mask="fov")
```

### Annotations

```python
kind, stats = w.add_segmentation(
    "organs_tp0", grid="ct_tp0",
    masks={"liver": liver, "lesion": lesion},   # or probabilities= or instances=
    encoding="auto",                            # or an explicit kind
    # threshold=0.3,                            # with probabilities=: the contains() cut (§7.5)
    annotated_classes=["liver", "spleen", "lesion"],
    ignore=uncertain_mask,                      # voxels nobody examined (§7.7)
    prov=act, quality={"status": "approved"},
)

w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct_tp0",
            space="index", scores=[0.91], instance_ids=[7])
w.add_obb("nodules", centers, sizes, rotations, class_ids=["nodule"], grid="ct")
w.add_keypoints("landmarks", points, keypoint_classes, class_ids, grid="ct")
w.add_points("fiducials_tp0", points, grid="ct",
             correspondence="fiducials_tp1")   # the paired point set (§10.6)
w.add_contours("rtstruct", polygons, grid="ct", space="world")
w.add_mesh("surface", vertices, faces, space="world")
w.add_classification("response", {"progressive": 1.0}, scope="sample",
                     timepoints=["tp0", "tp1"])   # the interval, not one visit
```

`ignore=` is stored wherever the chosen encoding can hold it. `labelmap` and
`layers` carry it in band; under `bitmask`, `instances` or `probmap` — and under
every encoding when the region overlaps a class — the writer stores it as a
sibling `mask` annotation named `<ann_id>_ignore` and sets `ignore_mask` on the
header, so `encoding="auto"` never decides whether the region survives, and it
reads back equal to what you gave. `ignore_mask=` names a mask you wrote
yourself instead; passing both is refused.

Pass exactly one of `masks=`, `probabilities=` and `instances=`, with
`encoding="auto"` or the encoding the argument implies; anything else is
refused rather than half-honoured. `instances=[]` with `annotated_classes=`
records "examined, none found" (§7.4).

Every `commit()` runs the validator's structural and semantic error rules over
the finished file and refuses to write one it would reject, so a file this
writer produces passes `medh5 validate` at the default level. Grid `units` must
be one of `mm`, `um`, `m`, `px` (§3.2).

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

<!-- illustrative -->
```python
w.build_index()                          # sampling indices for every voxel annotation
w.build_index(["organs_tp0"], max_coords=8192)
w.transcode_annotation("organs_tp0", "bitmask")
w.remove_annotation("old_seg")           # takes its index with it
w.remap_frame_uids({"old-frame": "new-frame"})   # grids, transforms, world-space
                                                 # annotations — all at once
w.infer_profiles()                       # the profiles the content satisfies
```

## Amending

```python
with medh5.amend("case.medh5") as w:
    w.set_quality("organs", status="approved")
```

Copy-on-write: a new file is built from the old and replaced atomically.
Objects this reader does not understand — a `x_` group, an unknown attribute —
are copied through untouched, so amending never silently drops what it cannot
read. A file it cannot preserve is refused before anything is written: a later
minor version (read as a projection, `MEDH5VersionError`) or a profile this
package does not implement (`E007`). Anything holding the file open across an
`amend` keeps reading the old version.

## Clinical history (format 1.1)

```python
from medh5.clinical import HOUR

with medh5.open("cohort/P-03.medh5") as s:
    s.version                          # "1.1"; imaging-only samples stay "1.0"
    s.support                          # "full", or "projection" for a later minor
    c = s.clinical                     # -> Clinical, or None without the profile
    c.clock                            # Clock(id, reference, origin_description, unit)
    c.events                           # (Event, ...) in stored order
    c.event("lab0").available_us       # inclusive (lo, hi) bounds, or None
    c.documents                        # (DocumentInfo, ...): no text read yet
    c.text("report0_text_v1")          # one document's text, read now
    c.links                            # (Link, ...)
    chosen = c.select(24 * HOUR)       # -> Selection, strict prospective
    chosen.certified, chosen.event_ids
    chosen.admits("image", "CT_tp0")   # may an input read this payload?
    c.records()                        # -> ClinicalRecords: everything, as one bundle
```

| | |
|---|---|
| `Clock.relative(id, origin)` | The subject clock every time is measured on |
| `Event(event_id, record_id, kind, temporal_type, status, ...)` | One immutable version; times as `(lo, hi)` microseconds (an `int` is an exact instant) |
| `Document(document_id, text, ...)` | Source text, owned by one `document` event |
| `Link.between(source, relation, target, asserted_by=None, span=None)` | A typed relationship between sample-relative objects |
| `SelectionPolicy(...)` | Context window, kinds, plans, limits, ties (contract §3.4) |
| `select(events, links, cutoff_us, policy)` | Selection over records not read from a file |
| `augment(path, records, out=None)` | Add a history to a 1.0 or 1.1 sample; returns the report |
| `strip(path, out)` | The imaging projection, as a new 1.0 file |
| `imaging_events_from_timepoints(path)` | Day-precision imaging events from `days_from_baseline` |

The writer takes the records one at a time --- `w.set_clock(...)`,
`w.add_event(...)`, `w.add_document(...)`, `w.add_link(...)` --- or as a
bundle, `w.add_records(records)`; each accepts the dataclass, a dict, or
keywords. See [Clinical history beside the images](../guides/clinical.md).

## Tasks and caches

```python
from medh5.task import TaskManifest

task = TaskManifest.load("cohort/progression.task.json")
task.task_fingerprint                  # the definition; not the rows
report = task.preflight()              # -> Preflight: every row, eligible or why not
report.counts, report.findings
row = report.row("P-01@24h")           # -> RowView
row.events, row.slots["ct"].image_id, row.target.status
```

| Module | |
|---|---|
| `medh5.task` | `TaskManifest`, `SourceRef` (`pin`, `check`), `Slot`, `Target`, `preflight`, `Preflight`, `RowView`, `Finding` |
| `medh5.cache` | `CacheWriter`, `FeatureCache`, `validate_cache` (`CacheReport.stale` / `.corrupt`), `fitted_on`, `build_document_cache`, `HashingTextEncoder` |
| `medh5.torch` | `ClinicalTaskDataset`, `ConceptVocabulary`, `collate_clinical` ([PyTorch](torch.md#clinical-tasks-format-11)) |

The contract is [Task and cache contract 1](../spec/task-cache-1.md); the
walk-through is [Train on clinical tasks](../guides/clinical-training.md).

## Collections

```python
medh5.pack(["case.medh5", "case2.medh5"], "pair.medh5c")    # keys default to file stems
medh5.unpack("pair.medh5c", "restored/", keys=["case"])       # -> restored/case.medh5
```

Packing moves chunks as stored bytes, so every member keeps its `content_id`;
see [Storage](storage.md#collections).

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
API; see spec §15.2 and `medh5.CODES`. `validate_paths` takes several files and
returns one report per file, which is what `medh5 validate` prints.

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
| `medh5.clinical` | [The clinical profile](../guides/clinical.md) (format 1.1) |
| `medh5.task`, `medh5.cache` | [Tasks and feature caches](../guides/clinical-training.md) |
| `medh5.storage` | [Codecs, chunking, recompression](storage.md) |

## Related

- **[Write and read your first sample](../tutorials/first-sample.md)** — this API end to end.
- **[How-to guides](../guides/index.md)** — the same calls, arranged by task.
- **[Sample document schema](schema.md)** — the fields the writer writes.
- **[Specification](../spec/medh5-1.0.md)** — the normative model behind it, and
  [1.1](../spec/medh5-1.1.md) for the clinical profile.
