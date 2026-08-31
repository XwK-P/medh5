# Concepts

Ten minutes on the data model. The normative statement of all of this is the
[specification](spec/medh5-1.0.md); this page is the shape of it.

## A sample is a subject

```
case_0001.medh5
├── meta                 the sample document: identity, timepoints, labels, provenance
├── grids/               geometry — one entry per coordinate lattice
├── images/              arrays, each bound to a grid
├── annotations/         segmentations, boxes, contours, keypoints, labels
├── transforms/          registrations between frames
└── index/               derived sampling caches
```

One file holds one subject. If that subject was scanned three times, all three
visits are in the file, and the sample knows they belong to the same person.
This is the single decision most of the rest follows from:

- Splitting by file is subject-safe, because a file *is* a subject.
- A change annotation ("this lesion grew 40 %") has somewhere to live, because
  both timepoints it refers to are in scope.
- Registration between visits is an object in the file, not a convention
  between two filenames.

A curator who wants one file per study can still have it — set `sample_id` to a
study key — but then nothing above is true, and the format will not pretend
otherwise.

## Geometry lives in grids

A **grid** is a coordinate lattice: shape, spacing, origin, direction,
coordinate system. Images and annotations reference a grid; they never carry
their own geometry.

```python
with medh5.open(path) as s:
    grid = s.grids["ct_tp0"]
    grid.shape          # (192, 256, 256)
    grid.spacing        # (1.5, 0.8, 0.8)
    grid.affine         # 4x4 index -> world
    grid.coord_system   # "LPS"
```

The affine is `x = origin + direction · (spacing ⊙ i)`, and an integer index is
the **centre** of a voxel. Two arrays on the same grid are co-registered by
construction; two arrays on different grids are related by a transform or not
at all.

**Boxes are at voxel edges.** A box `[a, b]` in index space covers the numpy
slice `a+0.5 : b+0.5`, so the box `[1.5, 5.5]` is exactly `slice(2, 6)`. The
half-voxel is the whole reason the convention is written down: it is where
every off-by-one in medical detection comes from.

```python
boxes = s.annotations["lesions"]
boxes.boxes[0]        # [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]
boxes.as_slices()[0]  # (slice(2, 6), slice(4, 9), slice(5, 11))
```

## Timepoints belong to grids

A sample declares its timepoints; a grid names the one it belongs to; images
and annotations inherit it from their grid.

```python
s.timepoints["tp1"].days_from_baseline   # 92
s.at("tp1").images                       # only tp1's images
s.grids["ct_tp1"].timepoint              # "tp1"
```

Binding time to the *grid* rather than to each object means a follow-up CT and
its segmentation cannot disagree about which visit they describe.

## Classes come from a label set

A label set maps ids to keys, names and codes, and may declare a hierarchy.

```python
s.label_set["liver"].id        # 1
s.label_set[3].parents         # (1,)  — a lesion inside the liver
s.label_set.digest()           # canonical: two files agree or they do not
```

Ids `0` and `65535` are reserved: background and *ignore*. Everything else is
the curator's.

Annotations refer to classes by id, and every reader can go back to what the id
means — which is what makes `medh5 dataset check` able to say "class 3 means
two different things in this cohort".

## Coverage: what was looked for

Every annotation carries two sets:

| | |
|---|---|
| `class_ids` | classes this annotation **contains** |
| `annotated_class_ids` | classes that were **examined**, found or not |

```python
ann = s.annotations["organs"]
ann.class_ids               # (1, 2, 3)  — declared: liver, spleen, lesion
ann.annotated_class_ids     # (1, 2, 3)  — all three were examined
ann.voxel_counts()          # {1: 512, 2: 0, 3: 27}  — the spleen is absent
ann.is_annotated("vessel")  # False      — nobody looked for a vessel
ann.is_fully_covered        # True       — nothing appears that was not examined
```

Class 2 is in `class_ids` with zero voxels: examined, and absent. Class 4 is in
neither: not examined, so its absence says nothing at all.

A sample where the spleen was examined and not found is a **negative example**
of the spleen. A sample where nobody looked is not. Collapsing the two is how a
model learns that a whole site's scans have no spleens, and the format refuses
to let a writer do it silently.

An **ignore** region (`65535`) marks voxels that must not contribute to a loss
in either direction.

## One annotation API, five voxel encodings

A segmentation is stored as one of `labelmap`, `layers`, `bitmask`, `instances`
or `probmap` — chosen by measuring the overlap graph, not by asking the user —
and read through one contract:

```python
ann.contains(class_id, index)   # is this voxel in this class?
ann.dense(["liver", "lesion"])  # (C, *spatial) boolean stack
ann.labelmap()                  # (spatial) of class ids, in priority order
ann.voxel_counts()              # per-class foreground counts
```

`ann.instances()` — per-object masks, ids and boxes — needs an encoding that
carries instance identity. Ask a `layers` annotation for it and it says so, and
says what to do:

```
MEDH5ValidationError: annotation 'organs' of kind 'layers' does not carry
instance identity, and it cannot be recovered from a dense encoding:
transcoding to `instances` would merge every object of a class into one and
mint an id that belongs to none of them (§7.4). Re-derive the objects from
whatever source had them.
```

Overlapping classes (a lesion inside the liver) are expressible in every
encoding that supports them, and transcoding between any pair is lossless.
See [Annotations](annotations.md).

## Provenance and quality

Who made this, with what, and how good is it:

```python
s.document.provenance.activities   # import, annotate, review, predict, register
s.document.quality["organs"]       # status, reviewers, agreement
s.document.deidentification        # method, profile, date shift — or None
```

`None` there is meaningful: a file with no de-identification record must be
treated as potentially identifying. Absence is never evidence.

## Integrity

Every object carries a SHA-256 over its **decompressed** content, and the root
carries a Merkle `content_id` over those digests.

```python
s.verify().ok           # every object matches its digest
s.content_id            # "sha256:..." — the identity of the content
```

Recompressing a file changes every stored byte and no digest, because the
digest is over the content and not over its encoding. Editing a voxel changes
one object digest and the `content_id`.

## Profiles

A file declares which conformance profiles it satisfies, and a validator can
be asked to hold it to them.

```python
s.profiles   # {"core", "seg", "det", "curation", "longitudinal"}
```

`core`, `seg`, `det`, `cls`, `reg`, `curation`, `multiscale`, `training`,
`longitudinal`. A tool that needs boxes can require `det` and get a specific
diagnostic instead of a `KeyError` three layers down.
