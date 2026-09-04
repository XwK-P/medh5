# MEDH5 Format Specification — Version 1.0

**Status:** Final · **Schema version:** `1.0` · **Supersedes:** medh5 0.x (`schema_version = "1"`)
**Container:** HDF5 ≥ 1.10 · **Media type:** `application/x-medh5` · **Extensions:** `.medh5` (sample), `.medh5c` (collection)

---

## 1. Scope, conformance and terminology

### 1.1 Scope

MEDH5 1.0 defines a self-describing, single-container format for **one medical imaging sample and all of
its ground truth**. A sample is one subject's imaging record: one or more timepoints, each with one or
more images, together with every annotation, transform and curation record about them. The format is
designed for four AI task families:

| Task family | What MEDH5 1.0 stores |
|---|---|
| Classification | multi-class, multi-label and hierarchical labels at sample / timepoint / grid / region / slice scope, ordinal clinical scales, and change labels spanning timepoints |
| Detection | axis-aligned boxes, oriented boxes (OBB), keypoints, point sets — in 2D and 3D, in voxel or world coordinates |
| Segmentation | multi-label voxel annotations with overlap, four interchangeable dense/sparse encodings, contours and surface meshes |
| Registration | affine and dense deformation transforms, B-spline and composite transforms, paired anatomical landmarks — within a timepoint (multi-modal) and across timepoints (longitudinal) |

Because a sample may span timepoints, longitudinal work — change detection, growth and response
assessment, lesion tracking, follow-up registration — is expressed inside a single file rather than by
an out-of-band convention linking separate files.

A conforming file is **plain HDF5**. Every object is inspectable with `h5ls -r`, `h5dump -A` or HDFView
with no MEDH5 software installed. No object depends on a Python pickle, a proprietary codec, or an
external service for its structural interpretation.

### 1.2 Conformance keywords

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHOULD**, **SHOULD NOT**, **MAY** and
**OPTIONAL** are to be interpreted as described in RFC 2119.

A **writer** is software producing MEDH5 files. A **reader** is software consuming them. A **validator**
is software checking conformance (§15).

### 1.3 Conformance profiles

A file declares the profiles it claims via the root attribute `medh5_profiles`. A validator checks only
the declared profiles. Profiles compose; `core` is always required.

| Profile | Requires |
|---|---|
| `core` | §2 container, §3 geometry and timepoints, §4 images, §13 integrity |
| `seg` | `core` + §5 label set + at least one voxel annotation (§7) |
| `det` | `core` + §5 label set + at least one geometric annotation (§8) whose `task` is `detection`. Contours and meshes are geometry in service of segmentation (§6.3) and do not, alone, make a file a detection dataset. |
| `cls` | `core` + §5 label set + at least one classification annotation (§9) |
| `reg` | `core` + at least one transform (§10) with resolvable endpoints |
| `curation` | `core` + §11 provenance graph + `quality` on every annotation |
| `multiscale` | `core` + §4.3 pyramid layout on every image |
| `training` | `core` + §14.3 sampling index present and current |
| `longitudinal` | `core` + ≥ 2 declared timepoints (§3.7), `timepoint` on every grid, and stable instance ids for objects observed at more than one timepoint (§7.4) |

### 1.4 Terminology

| Term | Meaning |
|---|---|
| **Sample** | One subject observed at one **or more** timepoints, with one or more images per timepoint, plus all ground truth about them. The unit of a `.medh5` file. |
| **Timepoint** | One observation occasion for the subject — in DICOM terms usually one study. Declared in §3.7; every grid belongs to exactly one. |
| **Grid** | A named discrete sampling lattice with a full index→world geometry, belonging to one timepoint. A sample MAY have many. |
| **Image** | A dense array defined on exactly one grid, and therefore in exactly one timepoint. |
| **Label set** | The controlled vocabulary of classes used by the sample's annotations. |
| **Annotation** | One coherent unit of ground truth produced by one activity: a segmentation, a box set, a classification, a landmark set. |
| **Transform** | A spatial mapping between two frames of reference. |
| **Class id** | A `uint16` identifier of a semantic class, unique within a label set. |
| **Instance id** | A `uint32`/`uint64` identifier of one physical object; several instances MAY share a class id. |

---

## 2. Container model

### 2.1 File identity

Root attributes on the file (`/`):

| Attribute | Type | Req. | Value |
|---|---|---|---|
| `medh5_version` | `str` | **MUST** | `"1.0"`. Readers **MUST** refuse a major version they do not implement and **MAY** read a higher minor version, ignoring unknown objects. |
| `medh5_kind` | `str` | **MUST** | `"sample"` or `"collection"`. |
| `medh5_profiles` | `str[]` | **MUST** | Declared profiles (§1.3). |
| `content_id` | `str` | SHOULD | `"<algo>:<hex>"` Merkle root (§13.2). |
| `digest_algo` | `str` | SHOULD | `"sha256"` (default), `"sha512"` or `"blake2b"`. A validator **MUST** report any other value as E703. |
| `created` | `str` | SHOULD | RFC 3339 UTC timestamp. |
| `generator` | `str` | SHOULD | `"<name> <version>"` of the writing software. |

`medh5_version` is the *format* version and is independent of the `medh5` Python package version.

### 2.2 Sample root

All object paths in this specification are relative to a **sample root group**:

* in a `sample` file the sample root is `/`;
* in a `collection` file each sample root is `/samples/<sample_key>`, where `<sample_key>` matches
  `[A-Za-z0-9_.-]{1,255}` and is unique in the file.

A collection file **MUST** carry `medh5_kind = "collection"` at `/` and **MUST** repeat
`medh5_version` on `/`. Each sample root in a collection **MUST** be structurally identical to a
standalone sample and **MUST** carry its own `medh5_profiles` and `content_id` (E007, E010). This
makes `sample ⊂ collection` a strict containment: extracting a sample root into a new file is a pure
copy. Packing and unpacking **MUST NOT** re-encode bulk data — chunks move as stored bytes — so a
sample's `content_id` is unchanged by either operation and a shard is never a second encoding of the
samples it holds.

> **Rationale.** One sample per file is the primary mode: it keeps write locking trivial, makes
> content addressing and split membership per-sample, and lets datasets be assembled with `ln -s`.
> Because a sample is subject-scoped rather than study-scoped (§3.7), a split that assigns whole files
> cannot leak the same patient across partitions — the most common evaluation error in medical AI
> becomes structurally impossible rather than a discipline the tooling has to enforce.
> Collections exist only to amortise per-file overhead for very small samples (2D radiographs,
> patches, cell crops) where 100k files is an operational problem.

Grouping a subject's whole record into one file is permitted, not required: a curator MAY emit one
sample per timepoint when files would otherwise be unwieldy, and `identity.subject_id` still links
them. What the format fixes is that a sample **MUST NOT** span subjects.

### 2.3 Layout

```
<sample root>
├── meta                      dataset · scalar UTF-8 JSON string · REQUIRED   (§2.4)
├── grids/                    group · REQUIRED                                (§3.2)
│   └── <grid_id>/            group · geometry lives entirely in attributes
├── images/                   group · REQUIRED, ≥ 1 entry                     (§4)
│   ├── <image_id>            dataset  (single-scale)
│   └── <image_id>/           group    (multiscale: datasets "0", "1", …)     (§4.3)
├── annotations/              group · OPTIONAL                                (§6–§9)
│   └── <ann_id>/             group · one annotation, one encoding
├── transforms/               group · OPTIONAL                                (§10)
│   └── <transform_id>/       group
└── index/                    group · OPTIONAL, derived, regenerable          (§14.3)
    └── <ann_id>/             group · sampling and statistics caches
```

Identifiers (`<grid_id>`, `<image_id>`, `<ann_id>`, `<transform_id>`) **MUST** match
`[A-Za-z0-9_.-]{1,128}`, **MUST** be unique within their group, and **MUST NOT** be `meta`.
Identifiers are **case-sensitive** and are stable references: other objects point at them by name.

Objects not described by this specification **MAY** be present. Readers **MUST** ignore them; writers
performing an amend **MUST** preserve them (§14.4).

### 2.4 The sample document (`/meta`)

`meta` is a **scalar dataset of HDF5 variable-length UTF-8 string** holding a single JSON object: the
*sample document*. It **MUST** be valid UTF-8 JSON and **MUST** validate against
`schemas/medh5-sample-1.0.schema.json`. It **MUST NOT** be compressed: HDF5 filters do not apply to
variable-length data, which lives in the file's global heap, so a compression request on `meta` is
either silently ignored or an error depending on the library. A vocabulary large enough for the size
to matter uses `form = "ref"` (§5.1) instead of an inline copy.

The sample document carries everything that is a *document*: identity, cohort, label set, provenance,
quality, splits, acquisition and free-form extras. It **MUST NOT** duplicate any value that this
specification places in an HDF5 attribute.

> **Rationale — no mirroring.** 0.x split metadata between typed attributes and a JSON `extra` blob,
> and mirrored some values in both. Mirrors drift. The 1.0 rule is exact: **arrays and per-object
> facts live in HDF5 (attributes on the object they describe); documents live in `/meta`.** Every
> object is self-describing under `h5dump`, and no fact has two homes.

Top-level members of the sample document:

| Member | Req. | §  | Purpose |
|---|---|---|---|
| `identity` | **MUST** | §12.1 | sample and subject identifiers |
| `timepoints` | **MUST** | §3.7 | ordered declaration of the sample's observation occasions |
| `cohort` | SHOULD | §12.2 | dataset, site, group key for leakage-free splitting |
| `label_set` | seg/det/cls | §5 | inline label set or reference |
| `provenance` | `curation` | §11.1 | agents and activities |
| `splits` | MAY | §12.3 | split-membership claims |
| `acquisition` | SHOULD | §4.5 | per-image acquisition parameters |
| `deidentification` | SHOULD | §11.4 | method, profile, date shift |
| `extra` | MAY | — | free-form JSON, namespaced by writer |

### 2.5 Attribute encoding conventions

| Logical type | HDF5 encoding |
|---|---|
| string | variable-length UTF-8 (`h5py.string_dtype()`) |
| string list | 1-D array of variable-length UTF-8, **never** a JSON string |
| boolean | `np.bool_` scalar |
| integer / integer list | `int64` scalar / 1-D `int64` |
| float / float list | `float64` scalar / 1-D `float64` |
| matrix | 2-D array, **stored 2-D** (0.x flattened `direction`; 1.0 **MUST NOT**) |
| enum | lowercase `snake_case` string from the values listed in this spec |

Readers **MUST** accept both `bytes` and `str` for string attributes (h5py version drift) and
**SHOULD** normalise to `str`.

---

## 3. Coordinate systems, geometry and timepoints

Geometry is normative and complete in 1.0. There is exactly one way to express where a voxel is, and
— since a sample may span visits — exactly one way to express *when* it was acquired (§3.7).

### 3.1 Axes

An array's axes are described in **stored order** (axis 0 varies slowest; arrays are C-contiguous).
Every grid declares, per axis:

* `axis_names` — labels, e.g. `["z","y","x"]`, `["t","z","y","x"]`, `["c","y","x"]`;
* `axis_kinds` — one of `spatial`, `channel`, `time`, `other`.

Only `spatial` axes participate in geometry. There **MUST** be 2 or 3 `spatial` axes, at most one
`time` axis and at most one `channel` axis. `spatial` axes **MUST** be contiguous in stored order and
**MUST** be the trailing axes. Let *S* be the number of spatial axes and *σ(k)* the stored index of
the *k*-th spatial axis.

### 3.2 Grids

`grids/<grid_id>` is an **empty group** whose attributes fully define the lattice:

| Attribute | Type | Req. | Meaning |
|---|---|---|---|
| `shape` | `int64[N]` | **MUST** | full array shape in stored order |
| `axis_names` | `str[N]` | **MUST** | §3.1 |
| `axis_kinds` | `str[N]` | **MUST** | §3.1 |
| `spacing` | `float64[S]` | **MUST** | positive voxel size per spatial axis, in `units` |
| `origin` | `float64[S]` | **MUST** | world coordinate of voxel index `0` on every spatial axis |
| `direction` | `float64[S,S]` | **MUST** | column *k* = unit world direction of spatial axis *k*; **MUST** be orthonormal to 1e-4 |
| `coord_system` | `str` | **MUST** | `"LPS"`, `"RAS"`, `"RAI"`, … or `"custom"` |
| `units` | `str` | **MUST** | `"mm"` (default), `"um"`, `"m"`, `"px"` (no physical calibration) |
| `timepoint` | `str` | see §3.7 | id of the timepoint this grid was acquired in; **MUST** be present when the sample declares more than one |
| `frame_uid` | `str` | SHOULD | frame-of-reference identifier (§3.4) |
| `time_values` | `float64[T]` | if `time` axis | acquisition time per time index, in `time_units` |
| `time_units` | `str` | if `time` axis | `"s"` (default) or `"ms"` |
| `chunk_hint` | `int64[N]` | MAY | writer's preferred chunk shape (§14.1) |
| `patch_hint` | `int64[S]` | MAY | intended training patch size (§14.1) |

`grids` **MUST** contain at least one grid. The grid named `ref` (if present) is the sample's
reference grid; otherwise the reference grid is the one referenced by the first image in
lexicographic order. In a multi-timepoint sample the reference grid **SHOULD** belong to the
baseline timepoint.

Grids are empty groups holding only attributes, so they are essentially free. Two acquisitions with
identical lattice geometry in different timepoints are therefore **two grids**, not one shared grid —
which keeps `timepoint` and `frame_uid` single-valued per grid and removes any need for per-image
overrides.

### 3.3 The index→world affine (normative)

For a spatial index vector **i** ∈ ℝ^S, expressed in *continuous index coordinates* where the integer
value *i_k* denotes the **centre of voxel *i_k*** along spatial axis *k*:

```
x_world  =  origin  +  direction · (spacing ⊙ i)
```

Equivalently, with `A ∈ ℝ^(S+1)×(S+1)`:

```
A[:S,:S] = direction @ diag(spacing)      A[:S, S] = origin      A[S,:] = [0 … 0 1]
x_world_homogeneous = A @ [i, 1]
```

Consequences that readers and writers **MUST** honour:

1. `direction` columns are indexed by **stored spatial axis**, not by world axis. A `(z,y,x)` array and
   an `(x,y,z)` array of the same volume have transposed `direction` matrices, not different geometry.
2. The volume occupies the closed continuous region `[-0.5, shape_k - 0.5]` on each spatial axis.
   Voxel *i* spans `[i-0.5, i+0.5)`.
3. A `channel` or `time` axis has no geometric extent; it is never part of **i**.
4. Two grids are **physically comparable without a transform** if and only if they share `frame_uid`
   and `coord_system`. They need not share `shape`, `spacing` or `direction`.

### 3.4 Frames of reference

`frame_uid` identifies the physical space a grid lives in. It **SHOULD** be derived from the source
DICOM `FrameOfReferenceUID` (pseudonymised, §11.4), or be a UUID minted by the writer for
non-DICOM sources.

* Grids with equal `frame_uid` are aligned by their affines alone.
* Grids with different `frame_uid` require a transform in `transforms/` (§10) to relate them.
* Grids in **different timepoints MUST NOT** share a `frame_uid` unless the acquisitions genuinely
  share a physical frame (the subject was not repositioned, as in a single interrupted session).
  Follow-up imaging is a new frame; relating it to baseline is registration, and pretending otherwise
  by reusing a `frame_uid` silently asserts an alignment nobody computed. A reader that
  compares an annotation on grid *A* with an image on grid *B* under a different `frame_uid` **MUST**
  refuse rather than assume alignment when no transform in `transforms/` resolves *A*→*B*. That is
  a property of the query, not of the file, so no §15.2 code corresponds to it; the reference
  implementation's paired loader is where the refusal lives (Appendix C).

> **Rationale.** 0.x required every array in a file to have identical shape, which silently forced
> resampling of PET/CT and multi-sequence MR at ingest — a lossy, irreversible operation performed
> before anyone knew what resolution the model would need. 1.0 stores native geometry and makes
> resampling an explicit, provenance-tracked activity.

### 3.5 Units

World coordinates, `spacing`, `origin`, box extents in world space, mesh vertices and displacement
magnitudes are all expressed in the grid's `units`. `units = "px"` declares an uncalibrated image
(e.g. a screenshot or a de-calibrated 2D X-ray); geometric annotations on such a grid **MUST** use
`space = "index"`.

### 3.6 Dimensionality

| Data | `axis_kinds` | Notes |
|---|---|---|
| 2D radiograph | `["spatial","spatial"]` | S = 2, `direction` is 2×2 |
| 3D volume | `["spatial"]*3` | S = 3 |
| 4D DCE / cine / 4D-CT | `["time","spatial","spatial","spatial"]` | one geometry, `time_values` per frame |
| RGB / multi-echo | `["channel","spatial","spatial"]` | `channel_names` on the image (§4.1) |
| Multi-b-value DWI | `["channel"] + ["spatial"]*3` | b-values in `acquisition` (§4.5) |

Diffusion gradients, multiple echoes and multiple contrast phases **SHOULD** be separate images on a
shared grid when they are separately addressable in training, and a `channel` axis when they are
always consumed together.

### 3.7 Timepoints

A sample covers one subject at one or more **timepoints**. A timepoint is one observation occasion —
in DICOM terms usually one study. Timepoints are declared, in acquisition order, in
`/meta → timepoints`:

```json
"timepoints": [
  {"id": "tp0", "index": 0, "label": "baseline", "date": "2026-02-01",
   "days_from_baseline": 0, "study_uid": "pseudo:1.2.826.…100",
   "series_uids": {"CT": "pseudo:…1", "PET": "pseudo:…2"},
   "subject_age_years": 61.4},
  {"id": "tp1", "index": 1, "label": "follow_up_3mo", "date": "2026-05-04",
   "days_from_baseline": 92, "study_uid": "pseudo:1.2.826.…101",
   "series_uids": {"CT_fu": "pseudo:…3"}}
]
```

| Field | Type | Req. | Meaning |
|---|---|---|---|
| `id` | `str` | **MUST** | matches `[A-Za-z0-9_.-]{1,128}`, unique in the sample; referenced by grids and annotations |
| `index` | `int` | **MUST** | 0-based acquisition order; **MUST** be dense and strictly increasing with time |
| `label` | `str` | SHOULD | human-readable role: `baseline`, `follow_up_3mo`, `post_treatment`, `pre_contrast` |
| `date` | `str` | MAY | date, shifted per §11.4 |
| `days_from_baseline` | `number` | SHOULD | interval from `index = 0`; **survives date shifting** and is what models should consume |
| `study_uid` | `str` | SHOULD | pseudonymised source study identifier |
| `series_uids` | `object` | MAY | image id → pseudonymised source series identifier |
| `subject_age_years` | `number` | MAY | age at this timepoint |

Normative rules:

1. `timepoints` **MUST** contain at least one entry. A single-timepoint sample declares exactly one;
   nothing else in this specification changes for it.
2. When more than one timepoint is declared, every grid **MUST** carry a `timepoint` attribute naming
   a declared `id` (E106, E107).
3. **Timepoint is inherited, never repeated.** An image's timepoint is its grid's. An annotation's is
   the timepoint of its `grid`, unless it declares `timepoints` explicitly (§6.2). A transform's
   endpoints are the timepoints of the grids in its frames.
4. Dates **MAY** be absent or shifted; `index` and `days_from_baseline` **MUST** remain truthful,
   because interval is clinically load-bearing where absolute date is not.
5. A sample **MUST NOT** span subjects. Splitting by file is therefore always subject-safe (§2.2).

> **Rationale.** Longitudinal imaging is not a special case bolted onto a cross-sectional format: in
> oncology, neurology and screening, the *pair* is the unit of interest — the change is the finding.
> Modelling it as separate files linked by a naming convention pushes the join into every consumer,
> loses the correspondence between the same lesion at two visits, and makes the baseline→follow-up
> transform homeless. Making the subject the sample puts all three in the file that owns them, at the
> cost of larger files (§14.4) — the trade this specification chooses deliberately.

---

## 4. Images

### 4.1 Image datasets

`images/<image_id>` is either a dataset (single scale) or a group (multiscale, §4.3). Attributes on the
dataset — or on the group, inherited by every level:

| Attribute | Type | Req. | Meaning |
|---|---|---|---|
| `grid` | `str` | **MUST** | id in `grids/`; dataset `shape` **MUST** equal that grid's `shape` |
| `modality` | `str` | **MUST** | `"CT"`, `"MR"`, `"PT"`, `"US"`, `"CR"`, `"DX"`, `"MG"`, `"NM"`, `"OT"`, … (DICOM modality codes preferred) |
| `value_type` | `str` | **MUST** | §4.2 |
| `channel_names` | `str[C]` | if `channel` axis | e.g. `["R","G","B"]`, `["b0","b1000"]` |
| `rescale_slope` | `float64` | MAY | default `1.0` |
| `rescale_intercept` | `float64` | MAY | default `0.0` |
| `value_units` | `str` | SHOULD | `"HU"`, `"SUVbw"`, `"mm^2/s"`, `"a.u."`, … |
| `window_center` / `window_width` | `float64[]` | MAY | display presets |
| `valid_mask` | `str` | MAY | id of a `mask`-kind annotation delimiting the acquired FOV |
| `digest` | `str` | SHOULD | §13.1 |
| `prov` | `str` | SHOULD | activity id in the provenance graph (§11.1) |

### 4.2 Value semantics

`value_type` fixes what the stored numbers mean:

| `value_type` | Stored dtype (typical) | Meaning after rescale |
|---|---|---|
| `intensity` | any numeric | uncalibrated scanner intensity |
| `quantitative` | `int16`/`float32` | calibrated physical quantity named by `value_units` (HU, SUV, ADC, T1) |
| `rgb` | `uint8` | colour channels, `channel` axis required |
| `probability` | `float16`/`float32` | ∈ [0,1] |
| `displacement` | `float32` | vector field; see §10.3 |
| `mask` | `bool`/`uint8` | validity / FOV mask |

**Physical value** = `stored × rescale_slope + rescale_intercept`. Readers **MUST** apply this when
asked for physical values and **MUST NOT** apply it silently when returning raw arrays; the API
**MUST** make the choice explicit.

> Writers **SHOULD** store CT as `int16` HU with slope 1 / intercept 0 rather than `float32`. Measured:
> 24.0 MiB vs 48.0 MiB raw and 12.33 MiB vs 36.75 MiB compressed for a 192×256×256 volume — a 3.0×
> on-disk reduction with zero information loss (§14.2).

### 4.3 Multiscale pyramids (`multiscale` profile)

`images/<image_id>` as a group:

```
images/CT/
├── 0            dataset · level 0 · full resolution
├── 1            dataset · level 1
└── 2            dataset · level 2
```

Group attributes: `levels` (`int64`), `downsample_factors` (`float64[L,S]`, per level, per spatial
axis, relative to level 0), `downsample_method` (`"mean"`, `"nearest"`, `"gaussian"`, `"max"`), and
`grid_levels` (`str[L]`, one grid id per level).

Every level **MUST** have its own grid with a geometry consistent with level 0: for factor *f_k*,
`spacing'_k = spacing_k · f_k` and `origin' = origin + direction · (spacing ⊙ ((f − 1)/2))`
(the half-voxel shift that keeps the physical field of view identical). A validator **MUST** check
this to 1e-3 relative tolerance.

Voxel annotations MAY be pyramided identically, with `downsample_method = "nearest"` or `"max"`
(never `"mean"`) — see §7.

### 4.4 Validity masks

Real acquisitions have invalid regions: outside the reconstruction circle, zero-padded after
resampling, truncated FOV. `valid_mask` names a `mask`-kind annotation whose `true` voxels are
acquired data. Loss functions and intensity statistics **SHOULD** honour it. This is distinct from
the *annotation* coverage of §11.3, which is about what was labelled, not what was imaged.

### 4.5 Acquisition parameters

`/meta → acquisition` maps image id → free-form object. Keys **SHOULD** follow DICOM keyword names
where one exists:

```json
"acquisition": {
  "CT": {"KVP": 120, "XRayTubeCurrent": 210, "ConvolutionKernel": "B30f",
         "ContrastBolusAgent": "iodinated", "ContrastPhase": "portal_venous",
         "SliceThickness": 1.0, "Manufacturer": "SIEMENS", "ScannerModel": "SOMATOM Force"},
  "PET": {"Radiopharmaceutical": "FDG", "InjectedDose_MBq": 310, "DecayCorrection": "START"}
}
```

---

## 5. Label space

### 5.1 The label set

Annotations reference classes by `uint16` id. The mapping id → meaning is the **label set**, carried in
`/meta → label_set`, which is either **inline** or a **reference**:

```json
"label_set": {
  "id": "totalsegmentator-v2",          // stable identifier of the vocabulary
  "version": "2.1.0",
  "sha256": "9f2c…",                    // digest of the canonical serialization
  "uri": "https://…/totalsegmentator-v2.json",   // OPTIONAL, for `ref` form
  "form": "inline",                     // "inline" | "ref"
  "classes": [ … ],                     // REQUIRED when form == "inline"
  "relations": [ … ]                    // OPTIONAL
}
```

* `form = "inline"` — `classes` is present; the file is fully self-describing. **REQUIRED** for
  ≤ 4096 classes and **RECOMMENDED** always.
* `form = "ref"` — `classes` is absent; `uri` and `sha256` **MUST** both be present, `sha256` because
  it is the only thing that tells a reader *which* vocabulary it needs when the `uri` cannot be
  resolved. Permitted only for very large vocabularies; readers that cannot resolve `uri` **MUST**
  treat class names as unknown but **MUST** still read the annotation data. A collection **MAY**
  carry the resolved label set once at `/` and let sample roots use `form = "ref"` with
  `uri = "medh5:/label_set"`; the reference implementation neither writes nor resolves that form in
  1.x (Appendix C).

**Canonical serialization (normative).** `label_set.sha256` is the digest of the label set's *content*,
computed so that two implementations in two languages agree. The digested document is

```
{"id": …, "version": …, "classes": [ … ], "relations": [ … ], "skeletons": [ … ]}
```

with `classes` sorted by `id`, `relations` sorted by `(subject, predicate, object)`, `skeletons` sorted
by `id`, and `relations`/`skeletons` omitted when empty. It is serialized as JSON with **sorted keys,
no insignificant whitespace, and non-ASCII characters kept as UTF-8** (not `\u` escapes), then hashed.
`form`, `uri` and `sha256` are **excluded**: they describe how the vocabulary is *carried*, not what it
says, so an inline copy and a referenced copy of one vocabulary digest identically.

Two files are **vocabulary-compatible** iff their `label_set.id`, `version` and `sha256` match. A
dataset-level validator **MUST** report divergent vocabularies across a cohort.

### 5.2 Class entries

```json
{"id": 5, "key": "liver", "name": "Liver",
 "parents": [2], "category": "organ", "laterality": null,
 "color": [124, 62, 42, 255],
 "codes": [{"system": "SNOMED-CT", "code": "10200004", "name": "Liver"},
           {"system": "FMA", "code": "7197"}],
 "properties": {"is_lesion": false, "bodypart": "abdomen", "paired": false}}
```

| Field | Type | Req. | Notes |
|---|---|---|---|
| `id` | `int` 1…65534 | **MUST** | unique; stable across the vocabulary's life |
| `key` | `str` | **MUST** | unique machine name, `snake_case`, stable |
| `name` | `str` | **MUST** | display name |
| `parents` | `int[]` | MAY | zero or more parent ids — the hierarchy is a **DAG**, not a tree |
| `category` | `str` | SHOULD | `organ`, `lesion`, `vessel`, `bone`, `device`, `artifact`, `region`, … |
| `color` | `int[4]` | SHOULD | RGBA 0–255, for viewers |
| `codes` | `object[]` | SHOULD | ontology bindings; `system` ∈ {`SNOMED-CT`, `RadLex`, `FMA`, `UBERON`, `ICD-10`, `LOINC`, …} |
| `laterality` | `str∣null` | MAY | `left`, `right`, `bilateral`, `null` |
| `properties` | `object` | MAY | vocabulary-specific flags |

### 5.3 Reserved ids

| Id | Meaning |
|---|---|
| `0` | **background** — explicitly *not* any class. **MUST NOT** appear in `classes`. |
| `65535` | **ignore** — the voxel is outside what was annotated; it is neither foreground nor background. Loss functions **MUST** exclude it. **MUST NOT** appear in `classes`. |

Class ids are `uint16` throughout 1.0, capping a vocabulary at **65 534** classes. This is a
deliberate choice, not a limit inherited from the container: the width of a class id sets the dtype of
every `labelmap` and `layers` volume, where `uint8`/`uint16` is the difference between 1 and 2 bytes
per voxel per layer, and no clinical vocabulary surveyed approaches the cap. Vocabularies that do —
full-FMA anatomy, cell-type atlases — are served by the reserved `wide_labels` profile (§16) in a
future minor version. Note that **instance** ids are separately `uint32`/`uint64` (§7.4), so the cap
never limits how many *objects* a sample may contain.

`parents` forms a DAG (`liver_segment_iv` → `liver` → `abdominal_organ`; `left_kidney` → `kidney`,
and separately `part_of` → `urinary_system` via `relations`). Cycles are an error.

`relations` carries non-`is_a` edges:

```json
"relations": [{"subject": 12, "predicate": "part_of", "object": 40},
              {"subject": 12, "predicate": "adjacent_to", "object": 5}]
```

Every annotation declares how ancestors are to be interpreted via `closure`:

| `closure` | Meaning |
|---|---|
| `explicit` | Only the listed classes hold. `liver_segment_iv` present does **not** imply `liver`. |
| `implicit` | All `is_a` ancestors of a listed class also hold. |

Readers **MUST NOT** infer ancestors when `closure = "explicit"`. Writers **SHOULD** use `implicit`
for hierarchical classification and `explicit` for voxel annotations.

### 5.5 Skeletons

For `keypoints` annotations, the label set MAY declare skeletons:

```json
"skeletons": [{"id": "spine_c1_l5", "keypoints": [101, 102, …],
               "edges": [[101,102],[102,103], …]}]
```

---

## 6. Annotations — common model

### 6.1 Structure

Every annotation is a group `annotations/<ann_id>/` carrying data as datasets and a fixed attribute
header. The **encoding is an implementation detail behind a uniform read contract** (§7.6): callers ask
for classes and a region of interest, never for a storage layout.

### 6.2 Common attributes

| Attribute | Type | Req. | Meaning |
|---|---|---|---|
| `kind` | `str` | **MUST** | one of §6.3 |
| `task` | `str` | **MUST** | `segmentation`, `detection`, `classification`, `registration`, `other` |
| `grid` | `str` | voxel/index kinds | grid the annotation is defined on; fixes the annotation's timepoint (§3.7) |
| `timepoints` | `str[]` | see §3.7 | timepoint ids this annotation pertains to. Omit for the usual case (inherit from `grid`); **MUST** be present when the annotation spans timepoints, as a response assessment or a change label does (E409) |
| `space` | `str` | geometric kinds | `index` or `world` (§8.1) |
| `frame_uid` | `str` | if `space = world` | frame the coordinates live in |
| `class_ids` | `uint16[]` | **MUST** except `mask` | classes this annotation *can* express, in encoding order |
| `annotated_class_ids` | `uint16[]` | **MUST** | classes actually searched for (§11.3) — the coverage contract |
| `closure` | `str` | **MUST** | `explicit` or `implicit` (§5.4) |
| `ignore_id` | `uint16` | MAY | defaults to `65535` |
| `prov` | `str` | SHOULD | activity id (§11.1) |
| `quality` | `str` | SHOULD | key into `/meta → quality` (§11.2) |
| `derived_from` | `str[]` | MAY | annotation ids this was computed from (consensus, propagation) |
| `digest` | `str` | SHOULD | §13.1 |

`class_ids` ⊇ every class the data can encode. `annotated_class_ids` is what the annotator committed to
finding. `annotated_class_ids ⊆ class_ids` **MUST** hold. The set difference is the crux of partial
labelling: a class in `class_ids` but not in `annotated_class_ids` may be present in the patient and
absent from the file.

### 6.3 Annotation kind registry

| `kind` | Task | § | Data model |
|---|---|---|---|
| `labelmap` | seg | §7.1 | one dense integer volume, classes mutually exclusive |
| `layers` | seg | §7.2 | *L* dense integer volumes; mutually exclusive **within** a layer |
| `bitmask` | seg | §7.3 | *P* dense `uint64` bitplanes; one bit per class per voxel |
| `instances` | seg + det | §7.4 | per-object bbox + bbox-local bit-packed mask |
| `probmap` | seg | §7.5 | per-class `float16` probability volumes |
| `mask` | — | §4.4 | single boolean volume, no classes (FOV, ignore region) |
| `boxes` | det | §8.1 | axis-aligned boxes |
| `obb` | det | §8.2 | oriented boxes |
| `keypoints` | det | §8.3 | per-object keypoint sets |
| `points` | det/reg | §8.4 | unordered or named point sets, landmarks |
| `contours` | seg | §8.5 | planar polygons (DICOM RTSTRUCT interop) |
| `mesh` | seg | §8.6 | triangle surface mesh |
| `classification` | cls | §9 | labels at sample/timepoint/grid/roi/slice scope, incl. change across timepoints |

---

## 7. Voxel annotations

### 7.0 The overlap problem

The defining constraint: a voxel may belong to several classes at once (a lesion inside a liver
segment inside the liver; an aorta overlapping a stent; a tumour crossing two lobes), and there may be
hundreds of classes. 0.x stored one boolean volume per class. On a 160³ phantom with 200 classes
(24 mutually exclusive organs + 176 overlapping structures, 0.25 labels/voxel) that costs, measured:

| Encoding | Size (lz4-1) | 64³ read, all classes | 64³ read, one class | Objects |
|---|---|---|---|---|
| 0.x per-class `bool` | 3.57 MiB | 116.9 ms | 0.51 ms | 200 datasets |
| 1.0 `layers` (L = 5) | **0.55 MiB** | **6.7 ms** | **0.09 ms** | 1 dataset |
| 1.0 `bitmask` (P = 4) | 0.68 MiB | 10.1 ms | 2.79 ms | 1 dataset |
| 1.0 `instances` | 0.08 MiB | — | — | 4 datasets |

`layers` is 6.5× smaller, 17× faster for an all-class patch read and 5.7× faster for a single-class
read than the 0.x layout, on identical codec settings. §7.6 gives the selection rule; the reader API
is identical across encodings.

1.0 defines **four** normative encodings — `labelmap`, `layers`, `bitmask`, `instances` — plus
`probmap` for soft values. A run-length encoding was considered and **deferred**: compressed
chunked storage already leaves unwritten chunks unallocated, so RLE's disk saving over
`instances` is small, while it costs O(1) ROI access and a second sparse code path. The `rle`
kind name is reserved (§16) so a future minor version can add it without ambiguity; COCO and
DICOM-SEG run-length payloads are decoded by converters rather than stored (§7.6).

### 7.1 `labelmap`

One dense volume; each voxel has at most one class.

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `data` | grid `shape` | `uint8` or `uint16` | value = class id; `0` = background; `ignore_id` = ignore |

Use when classes are mutually exclusive (nnU-Net-style anatomy). `uint8` **MUST** be used when
`max(class_ids) ≤ 254` and no ignore region is present; otherwise `uint16`.

### 7.2 `layers` — **the default**

*L* dense labelmaps stacked on a leading `layer` axis. Classes within a layer are mutually exclusive;
classes in different layers may overlap freely.

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `data` | `(L, *grid.shape_spatial)` | `uint8`/`uint16` | value = class id, per layer |
| `layer_class_ids` | `(L, M)` | `uint16` | classes assigned to each layer, `0`-padded to width *M* |

Constraints:

* Every id in `class_ids` **MUST** appear in exactly one layer.
* `data` **MUST** be chunked as `(1, *spatial_chunk)` so one layer is readable without decompressing
  the others (§14.1).
* Writers **MUST** produce a minimal or near-minimal *L* by colouring the class overlap graph
  (§7.6).

`data[l]` at a voxel yields the single class from layer *l* holding that voxel, or `0`. The full set of
classes at a voxel is the union over layers.

### 7.3 `bitmask`

One bit per class per voxel, packed into `uint64` planes.

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `data` | `(P, *grid.shape_spatial)` | `uint64` | `P = ceil(len(class_ids)/64)` |
| `bit_class_ids` | `(len(class_ids),)` | `uint16` | position *p* ↔ plane `p//64`, bit `p%64` (LSB-first) |

`data` **MUST** be chunked `(1, *spatial_chunk)`. Bit ordering is LSB-first within each `uint64`, and
the value is interpreted in **native machine integer semantics**, not byte order — readers **MUST** use
integer shifts, never byte offsets.

Use when a voxel routinely carries many labels at once (deep hierarchies materialised as separate
classes, or dense multi-rater unions). Raw cost is `8·ceil(C/64)` bytes/voxel versus `2L` for
`uint16` layers, so `layers` wins on size whenever `L < 4·ceil(C/64)`; `bitmask` wins on the
"which classes are at this voxel" query, which is O(P) reads regardless of C.

### 7.4 `instances`

Per-object storage: a box plus a bbox-local bit-packed mask. The natural encoding for lesions,
nodules, cells, and any instance-segmentation task.

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `boxes` | `(N, S, 2)` | `float32` | `[lo, hi]` per spatial axis, in `space` coordinates (§8.1) |
| `class_ids` | `(N,)` | `uint16` | |
| `instance_ids` | `(N,)` | `uint32`/`uint64` | identifies one physical object; see the tracking rule below |
| `mask_offsets` | `(N+1,)` | `uint64` | offsets into `mask_data`; object *n* occupies `[o[n], o[n+1])` |
| `mask_shapes` | `(N, S)` | `int32` | dense shape of each object's cropped mask |
| `mask_data` | `(B,)` | `uint8` | concatenated `np.packbits` of each C-order cropped boolean mask |
| `scores` | `(N,)` | `float32` | OPTIONAL — for predictions / soft GT |

**Instance identity is sample-scoped and longitudinal.** An `instance_id` names one physical object
within the sample. The same object observed at several timepoints **MUST** reuse its id, in every
annotation that describes it; two distinct objects **MUST NOT** share one. Lesion tracking, growth
curves and per-lesion response therefore need no additional structure — joining on `instance_id`
across timepoints is the tracking. A validator warns (W909) when one id appears with two different
class ids, which is almost always a tracking error rather than a reclassification. Because identity
is sample-scoped, so is the check: the conflict that matters most is *between* two visits'
annotations, where each file is internally consistent and only the join is wrong.

An object present at one timepoint and absent at another is represented by its absence from that
timepoint's annotation; a *resolved* lesion is distinguishable from an *unexamined* one only through
`annotated_class_ids` (§11.3), which is why coverage is required.

`mask_data` **MAY** be absent, in which case the annotation is box-only and `kind` **SHOULD** be
`boxes` instead. Measured on the 200-structure phantom: 0.08 MiB versus 3.57 MiB for per-class dense —
a 45× reduction, because storage is proportional to object volume, not image volume.

### 7.5 `probmap`

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `data` | `(len(class_ids), *shape_spatial)` | `float16`/`float32` | values in [0,1] |
| `normalized` | attr `bool` | | `true` ⟹ channels sum to 1 across classes at each voxel |
| `threshold` | attr `float64` | | OPTIONAL, default `0.5`: the probability at or above which a voxel *contains* the class (§7.6); in [0, 1] |

For soft ground truth, inter-rater probability maps, distillation targets and predicted logits after
sigmoid/softmax. **MUST** be chunked `(1, *spatial_chunk)`. `threshold` is a spec-defined attribute
and so is covered by `content_id` (§13.2): two readers of one file answer `contains` identically.

### 7.6 Encoding equivalence and selection

All voxel encodings define the same predicate:

```
contains(annotation, class_id c, voxel v) -> bool
```

A reader **MUST** expose this predicate identically regardless of `kind`, and transcoding between
encodings **MUST** be lossless for every `c ∈ class_ids`, `v ∈ grid` — with the sole exception of
`probmap`, which is lossless only under its declared `threshold` (§7.5).

Writers **SHOULD** select the encoding automatically:

```
1. Compute per-class voxel counts and pairwise overlaps.
2. fill = Σ|class_c| / (|classes| · |grid|)            (density)
   depth = Σ|class_c| / |{v : any class at v}|          (mean labels per labelled voxel)
3. If every class is a compact localized object and fill < 1e-3   → instances
   If the overlap graph is edgeless (L == 1)                      → labelmap
   If greedy-coloured L < 4·ceil(C/64)                            → layers        [usual case]
   Else                                                            → bitmask
   If soft values are required                                    → probmap
```

`layers` colouring uses the standard greedy heuristic on the overlap graph ordered by descending
degree. On the 200-class phantom this produced `L = 5` for a mean degree of 3.4 — one dense volume per
5 rather than per class.

Formats that carry run-length payloads — COCO `segmentation`, DICOM-SEG fragments — are converted on
ingest and on export: a converter decodes runs into whichever encoding §7.6 selects, and re-encodes
runs on the way out. No `.medh5` file stores runs.

### 7.7 Ignore and unlabeled semantics

* In `labelmap` / `layers`, the value `ignore_id` marks ignore voxels.
* In `bitmask` and `probmap`, ignore regions **MUST** be expressed as a separate `mask`-kind
  annotation named by the `ignore_mask` attribute.
* `0` means **background — verified absent for `annotated_class_ids`**. It does not mean "unknown".
  A file with a partially annotated volume **MUST** either mark unlabelled regions as ignore or
  restrict `annotated_class_ids` accordingly. Getting this wrong is the single most common cause of
  silently mistrained segmentation models, so validators **MUST** warn (W904) when an annotation
  declares fewer `annotated_class_ids` than `class_ids` and carries no ignore region.

---

## 8. Geometric annotations

### 8.1 Coordinate conventions (normative)

`space` selects the coordinate system of every coordinate in a geometric annotation:

| `space` | Coordinates are | Requires |
|---|---|---|
| `index` | continuous index coordinates of `grid` (§3.3): integer = voxel centre, volume spans `[-0.5, n-0.5]` | `grid` |
| `world` | physical coordinates in `frame_uid`, in the grid's `units` | `frame_uid` |

**Boxes** are stored as `[lo, hi]` per spatial axis in that same continuous coordinate — measured at
voxel **edges**. Therefore:

```
numpy slice a:b   ⟺   lo = a − 0.5 ,  hi = b − 0.5      (extent hi − lo = b − a voxels)
box → slice       :    start = floor(lo + 0.5) , stop = floor(hi + 0.5)
```

The rounding is **half-up**: `floor(x + 0.5)`, not a language's default `round`. Half-to-even —
which is what Python's `round` and NumPy's `rint` do — breaks the extent identity above for a box on
integer edge coordinates, because `lo + 0.5` and `hi + 0.5` then round in opposite directions.

Boxes **MUST** be `float32` or `float64` and **MUST** satisfy `lo ≤ hi`. Writers **MUST NOT** store
integer boxes: rounding a box on resample or on world↔index conversion is lossy, and 0.x's
integer-only boxes could not represent a rotated or resampled box at all.

Every coordinate array carries `space`; readers **MUST** convert through the affine (§3.3) rather than
assuming a convention.

### 8.2 `boxes` — axis-aligned

| Dataset | Shape | dtype |
|---|---|---|
| `boxes` | `(N, S, 2)` | `float32` |
| `class_ids` | `(N,)` | `uint16` |
| `instance_ids` | `(N,)` | `uint32`/`uint64` — OPTIONAL |
| `scores` | `(N,)` | `float32` — OPTIONAL |
| `attributes` | `(N,)` | vlen UTF-8 JSON — OPTIONAL, per-box free-form |
| `slice_index` | `(N,)` | `int32` — OPTIONAL, for 2D boxes drawn on a slice of a 3D grid |

`slice_index` with `S = 3` and a degenerate axis (`lo == hi`) expresses "a 2D box on slice k",
the common radiology annotation.

### 8.3 `obb` — oriented boxes

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `centers` | `(N, S)` | `float32` | box centre |
| `sizes` | `(N, S)` | `float32` | **full** edge lengths along the box's local axes (not half-extents) |
| `rotations` | `(N, S, S)` | `float32` | column *k* = unit direction of the box's local axis *k*, in `space` |
| `class_ids`, `instance_ids`, `scores`, `attributes` | as §8.2 | | |

`rotations` **MUST** be a proper rotation (orthonormal, `det = +1`) to 1e-4. Quaternion and
Euler-angle forms are **not** stored; readers convert. The corner set is
`center + rotations @ (sizes/2 ⊙ s)` for `s ∈ {−1,+1}^S`.

> **Rationale.** Rotation matrices are dimension-generic (they cover 2D OBB and 3D OBB with one
> layout), have no ordering-convention ambiguity (unlike Euler angles) and no double-cover ambiguity
> (unlike quaternions). The cost — S² floats per box instead of 1 or 4 — is negligible next to image
> data.

### 8.4 `keypoints`

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `points` | `(N, K, S)` | `float32` | K keypoints for each of N objects |
| `visibility` | `(N, K)` | `uint8` | `0` = not labelled, `1` = labelled but occluded, `2` = visible |
| `keypoint_class_ids` | `(K,)` | `uint16` | class id per keypoint slot |
| `class_ids` | `(N,)` | `uint16` | object class |
| `skeleton` | attr `str` | | skeleton id from §5.5 |

### 8.5 `points` — landmarks and point sets

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `points` | `(N, S)` | `float32` | |
| `class_ids` | `(N,)` | `uint16` | OPTIONAL |
| `names` | `(N,)` | vlen UTF-8 | OPTIONAL — anatomical landmark names |
| `weights` | `(N,)` | `float32` | OPTIONAL — evaluation weights |
| `correspondence` | attr `str` | | OPTIONAL — id of the paired `points` annotation (§10.5) |

### 8.6 `contours`

Planar polygons, for DICOM RTSTRUCT round-trips and slice-wise manual annotation.

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `vertices` | `(V, S)` | `float32` | concatenated polygon vertices |
| `contour_offsets` | `(M+1,)` | `int64` | polygon *m* spans `[o[m], o[m+1])` |
| `contour_class_ids` | `(M,)` | `uint16` | |
| `contour_plane` | `(M, 2)` | `int32` | `(axis, index)` of the plane each polygon lies in; `axis = −1` for out-of-plane |
| `contour_role` | `(M,)` | `uint8` | `0` = outer boundary, `1` = hole |

Rasterisation to a voxel annotation is an explicit, provenance-tracked activity — never implicit.

### 8.7 `mesh`

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `vertices` | `(V, 3)` | `float32` | in `space` (usually `world`) |
| `faces` | `(F, 3)` | `int32` | triangles, counter-clockwise seen from outside |
| `normals` | `(V, 3)` | `float32` | OPTIONAL |
| `vertex_class_ids` | `(V,)` | `uint16` | OPTIONAL, for multi-structure meshes |
| `mesh_offsets` | `(M+1,)` | `int64` | OPTIONAL, several meshes in one annotation |
| `mesh_class_ids` | `(M,)` | `uint16` | OPTIONAL |

Meshes are **surfaces**, not fallbacks for voxel data: a `mesh` annotation does not satisfy the `seg`
profile on its own.

---

## 9. Classification annotations

`kind = "classification"`.

| Attribute | Type | Req. | Meaning |
|---|---|---|---|
| `scope` | `str` | **MUST** | `sample`, `timepoint`, `grid`, `roi`, `slice`, `instance` |
| `multilabel` | `bool` | **MUST** | `false` ⟹ exactly one positive class per scope unit |
| `closure` | `str` | **MUST** | §5.4 — `implicit` gives hierarchical labels for free |

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `class_ids` | `(K,)` | `uint16` | asserted classes — **the dataset**, distinct from the same-named §6.2 *attribute*, which declares the classes this annotation can express |
| `values` | `(K,)` | `float32` | `1.0` for a hard positive; ∈[0,1] for soft/consensus labels; `0.0` asserts **verified negative** |
| `scope_ids` | `(K,)` | `int64` | OPTIONAL — per assertion: slice index, roi id, instance id, or timepoint `index` when `scope = "timepoint"` |
| `scheme_values` | `(K,)` | vlen UTF-8 | OPTIONAL — ordinal value in a named scheme |
| `schemes` | `(K,)` | vlen UTF-8 | OPTIONAL — scheme id, e.g. `"BI-RADS"`, `"Lung-RADS"`, `"Gleason"`, `"mRS"` |

Semantics:

* A class in `annotated_class_ids` with **no** entry in the `class_ids` *dataset* is a **negative**:
  it was looked for and not found. A class not in `annotated_class_ids` is **unknown**. Because
  `annotated_class_ids ⊆ class_ids` must hold for the *attribute* (§6.2), a writer includes every
  searched-for class in the attribute even when no assertion carries it. This is the general rule for
  kinds whose `class_ids` attribute does not index storage — §8 and §9 — and it is why the attribute
  and the dataset are allowed to differ.
* `values = 0.0` records an explicit negative assertion (useful for multi-rater aggregation where
  "0 of 4 raters" differs from "not assessed").
* Ordinal scales are labels, not numbers: store `schemes = "BI-RADS"`, `scheme_values = "4b"`, plus a
  `class_ids` entry for the corresponding vocabulary class. Numeric comparison of `scheme_values` is a
  reader concern; the file records the assessment verbatim.
* Hierarchical classification uses `closure = "implicit"` and asserts only the most specific class;
  ancestors follow from the label-set DAG (§5.4).

Longitudinal classification uses two scopes:

* `scope = "timepoint"` — one assertion per visit, `scope_ids` carrying the timepoint `index`. This is
  how per-visit staging, per-visit severity grading or a per-visit finding is recorded.
* `scope = "sample"` with an explicit `timepoints` attribute (§6.2) — one assertion **about a set of
  timepoints**. This is how change is recorded: a response category (RECIST `partial_response`, RANO
  `progression`), a stability judgement, or a new-lesion flag. `timepoints` **MUST** list the
  timepoints compared, in order, so `["tp0","tp2"]` and `["tp1","tp2"]` are distinct assessments.

Change labels are ordinary classification annotations; the format adds no `change` kind. What makes
them well defined is that the compared timepoints are named rather than implied by file ordering.

Multi-rater labels are **separate annotations** with distinct `prov`, optionally accompanied by a
consensus annotation carrying `derived_from = [rater ids]` (§11.2). There is no special multi-rater
encoding — the annotation set *is* the model.

---

## 10. Registration

`transforms/<transform_id>` is a group. Transforms map **points**, not images.

Two uses dominate, and the format does not distinguish them — both are a mapping between frames:

* **Intra-timepoint (multi-modal)** — PET to CT, T1 to FLAIR, acquired in one session and often
  already aligned, in which case the grids share a `frame_uid` and no transform is needed at all.
* **Inter-timepoint (longitudinal)** — baseline to follow-up. Grids in different timepoints never
  share a frame (§3.4), so this transform is always required to compare them, and it is the object
  whose accuracy longitudinal landmark ground truth measures (§10.6).

A transform's timepoints are those of the grids in its frames; it needs no `timepoint` attribute of
its own.

### 10.1 Common attributes

| Attribute | Type | Req. | Meaning |
|---|---|---|---|
| `kind` | `str` | **MUST** | `identity`, `affine`, `displacement`, `bspline`, `composite` |
| `from_frame` | `str` | **MUST** | source `frame_uid` |
| `to_frame` | `str` | **MUST** | target `frame_uid` |
| `from_grid` / `to_grid` | `str` | SHOULD | representative grids in each frame |
| `units` | `str` | **MUST** | coordinate units, matching the frames' grids |
| `invertible` | `bool` | SHOULD | |
| `inverse_id` | `str` | MAY | id of the transform representing T⁻¹ |
| `prov` | `str` | SHOULD | activity that produced it (§11.1) |
| `metrics` | `str` | MAY | key into `/meta → quality` holding TRE, Dice-after-warp, folding fraction |
| `digest` | `str` | SHOULD | §13.1 |

### 10.2 Direction convention (normative)

**A transform T with `from_frame = F`, `to_frame = M` maps a point expressed in F to the
corresponding point in M:  `x_M = T(x_F)`.**

To resample an image defined in M onto a grid in F — the usual "warp the moving image onto the fixed
image" operation — evaluate T at each F-grid point and sample M at `T(x)`. This is the ITK / SimpleITK
`TransformPoint` convention and the inverse of the "forward warp" convention used by some optical-flow
literature. Writers **MUST** use this convention. There is no attribute to select the other one:
ambiguity here is the leading cause of silently mirrored registration results.

### 10.3 `affine`

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `matrix` | `(S+1, S+1)` | `float64` | homogeneous, world→world, last row `[0…0 1]` |

`x_M_h = matrix @ x_F_h`, both in world coordinates in `units`. Index-space affines are **not** stored;
compose with the grids' affines (§3.3) to obtain one.

### 10.4 `displacement`

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `field` | `(S, *grid.shape_spatial)` | `float32`/`float16` | displacement vector components |

| Attribute | Req. | Meaning |
|---|---|---|
| `field_grid` | **MUST** | grid the field is sampled on; **MUST** be in `from_frame` |
| `vector_space` | **MUST** | `world` (components along world axes, in `units`) or `index` (components along the field grid's index axes, in voxels) |
| `interpolation` | SHOULD | `linear` (default) or `cubic`, for evaluation off-grid |
| `extrapolation` | SHOULD | `zero` (default), `nearest` or `error` |

The transform is `T(x) = x + u(x)`, with `u` obtained by interpolating `field` at `x` expressed in the
field grid. Component order matches the field grid's spatial axes. Storing components on the leading
axis (`(S, Z, Y, X)`, chunked `(1, …)`) lets a reader fetch one component or one ROI without touching
the rest — the reason for that axis order rather than a trailing `(Z, Y, X, S)`.

`float16` is permitted and **RECOMMENDED** for displacement magnitudes below ~64 voxels; the loss of
precision (≈ 5e-4 relative) is far below registration accuracy and halves field size.

### 10.5 `bspline` and `composite`

`bspline`:

| Dataset / attr | Meaning |
|---|---|
| `control_points` `(S, *cp_shape)` `float64` | coefficients in `vector_space` |
| `cp_grid` attr | grid id describing control-point spacing/origin/direction |
| `order` attr `int` | spline order, default 3 |

`composite`:

| Attribute | Meaning |
|---|---|
| `components` `str[]` | ordered transform ids; applied **left to right**: `T(x) = T_n(…T_1(x))` |

Frames **MUST** chain: `components[i].to_frame == components[i+1].from_frame`, the first
`from_frame` equals the composite's, and the last `to_frame` equals the composite's. Validators check
this (E501).

### 10.6 Landmark correspondences

Registration ground truth and evaluation use paired `points` annotations (§8.5): two annotations on
different grids/frames, each referencing the other via `correspondence`, with equal `N` and matching
row order. Target registration error is then `‖T(p_i^F) − p_i^M‖`, weighted by `weights`.

A `reg`-profile file supplying landmark GT **MUST** carry both point sets and **SHOULD** carry the
transform whose accuracy they measure, with `metrics` populated.

For longitudinal registration the two point sets live in different timepoints, and corresponding
anatomical points **SHOULD** also carry equal `instance_ids` when they mark trackable objects, so
landmark correspondence and lesion tracking agree by construction.

---

## 11. Provenance, quality and curation

### 11.1 Provenance graph

`/meta → provenance` is a W3C PROV-lite graph reduced to two node types:

```json
"provenance": {
  "agents": [
    {"id": "r1", "type": "person",   "name": "pseudonym:RAD-07", "role": "annotator",
     "qualification": "board-certified radiologist, 9y thoracic"},
    {"id": "r2", "type": "person",   "name": "pseudonym:RAD-12", "role": "reviewer"},
    {"id": "s1", "type": "software", "name": "nnU-Net", "version": "2.5.1"},
    {"id": "s2", "type": "software", "name": "medh5",   "version": "1.0.0"},
    {"id": "o1", "type": "organization", "name": "Site B"}
  ],
  "activities": [
    {"id": "act_import", "type": "import", "agent": "s2",
     "started": "2026-02-03T09:11:02Z", "ended": "2026-02-03T09:11:40Z",
     "tool": "medh5 convert from-dicom",
     "inputs": ["dicom:1.2.826.0.1.3680043.…"], "outputs": ["images/CT", "grids/ct"],
     "params": {"modality_lut": "applied", "series_selection": "thinnest"}},
    {"id": "act_seg", "type": "annotate", "agent": "r1",
     "started": "2026-02-05T13:02:00Z", "ended": "2026-02-05T14:47:00Z",
     "tool": "3D Slicer 5.6.2", "outputs": ["annotations/organs"],
     "params": {"protocol": "TS-v2 abdominal protocol rev C"}},
    {"id": "act_review", "type": "review", "agent": "r2",
     "ended": "2026-02-06T08:30:00Z", "inputs": ["annotations/organs"],
     "outputs": ["annotations/organs"], "params": {"verdict": "approved_with_edits"}}
  ]
}
```

`type` ∈ {`import`, `annotate`, `review`, `predict`, `resample`, `register`, `derive`, `deidentify`,
`transcode`, `other`}. Objects link to activities through their `prov` attribute. Timestamps are
RFC 3339 UTC. A validator at level `semantic` **MUST** report dangling `prov` references (E601).

> Rationale: 0.x kept review state in a nested `extra["review"]` dict with an ad-hoc history list.
> That records *that* something was reviewed but not *what produced the data being reviewed*, and it
> could not describe a model-generated pre-annotation corrected by a human — the dominant real-world
> curation workflow. The two-node PROV graph handles both with one mechanism.

### 11.2 Quality

`/meta → quality` maps a key (named by an annotation's `quality` attribute) to:

```json
{"status": "approved",
 "confidence": 0.92,
 "reviewed_by": ["r2"],
 "agreement": [{"metric": "dice", "value": 0.913, "against": "annotations/organs_rater2",
                "per_class": {"5": 0.97, "12": 0.71}}],
 "issues": [{"code": "boundary_uncertain", "severity": "info",
             "class_ids": [12], "note": "portal vein margin blurred by motion"}],
 "edit_effort_s": 640}
```

`status` ∈ {`draft`, `submitted`, `reviewed`, `approved`, `rejected`, `deprecated`}. Status changes
are **activities** (§11.1), not fields with private history — the audit trail is the provenance graph.

### 11.3 Annotation coverage — partial labelling

The pair (`class_ids`, `annotated_class_ids`) is the coverage contract (§6.2). Combined with ignore
regions (§7.7) it expresses every real curation state:

| Situation | Encoding |
|---|---|
| Fully annotated for 3 classes | `annotated_class_ids = class_ids = [3 ids]`, no ignore region |
| Only the liver was annotated in an abdominal CT | `annotated_class_ids = [liver]`, `class_ids` may be larger |
| Annotated only in the thorax slab | `annotated_class_ids` full, plus ignore region outside the slab |
| Model pre-annotation, unreviewed | `quality.status = "draft"`, `prov` → a `predict` activity |
| Two raters disagree | two annotations + a `derived_from` consensus annotation |

Training code **MUST** consult `annotated_class_ids` before treating `0` as a negative. This is the
mechanism that makes partially-labelled datasets (the norm at scale) safely trainable.

### 11.4 De-identification

`/meta → deidentification`:

```json
{"method": "dicom-psi-profile", "profile": "DICOM PS3.15 E.1 basic + clean pixel",
 "date_shift_days": -117, "id_mapping": "external",
 "performed_by": "s2", "date": "2026-02-03T09:11:40Z", "burned_in_annotation_checked": true}
```

Normative rules:

* MEDH5 **MUST NOT** require any direct identifier. `identity.subject_id` and all UIDs **SHOULD** be
  pseudonyms.
* Writers **MUST NOT** copy DICOM tags wholesale into `acquisition` or `extra`; only the named
  parameters relevant to imaging physics.
* `date_shift_days` records that dates were consistently shifted, preserving intervals.
* A file whose `deidentification` is absent **MUST** be treated by tooling as potentially identifying.

---

## 12. Identity, cohorts and splits

### 12.1 `identity`

```json
"identity": {"sample_id": "BRATS-GLI-01234",
             "subject_id": "BRATS-GLI-01234",
             "sex": "F", "laterality": null, "bodypart": "brain"}
```

`sample_id` and `subject_id` are **REQUIRED**. `subject_id` prevents the most common evaluation error
in medical AI — the same patient appearing in train and test — and because a sample never spans
subjects (§3.7), assigning whole files to partitions is subject-safe without any further bookkeeping.

Per-occasion identifiers do **not** live here: `study_uid`, `series_uids`, dates and ages belong to
their timepoint entry (§3.7), because a sample may have several of each. A curator emitting one
sample per timepoint sets `sample_id` to a study-scoped key and `subject_id` to the patient key; a
curator bundling a whole record sets both to the patient key.

### 12.2 `cohort`

```json
"cohort": {"dataset_id": "abdomen-multiorgan-v3", "site_id": "site-B",
           "scanner_id": "SOMATOM-Force-042", "group_id": "BRATS-GLI-01234",
           "acquisition_protocol": "portal-venous 1mm"}
```

`group_id` defaults to `subject_id` and is the grouping key for leakage-free splits. It still matters
when samples are subject-scoped: multi-centre cohorts sometimes need a coarser group (a family, a
scanner, an enrolling site) than the subject. `site_id` and `scanner_id` support site-stratified
splitting and domain-shift analysis; when a subject was imaged on different scanners at different
visits, per-visit acquisition detail belongs in `/meta → acquisition` (§4.5), keyed by image id.

### 12.3 Splits

```json
"splits": [{"set_id": "cv5-2026-02", "partition": "train", "fold": 2,
            "assigned_by": "medh5 split", "assigned_at": "2026-02-10T…",
            "manifest_sha256": "3ab9…"}]
```

Each entry is a **membership claim**, not an authority. The dataset-level manifest is authoritative;
`manifest_sha256` lets a reader detect a file whose in-file claim predates the current split. A
validator **MUST** warn (W906) when two files claim the same `set_id` with different
`manifest_sha256`.

> **Rationale.** Splits are a property of a *cohort*, not of a sample, but training code overwhelmingly
> works file-by-file. Recording the claim in-file makes single-file debugging possible; hashing the
> manifest makes stale claims detectable instead of silently wrong.

---

## 13. Integrity

### 13.1 Object digests

Every dataset **SHOULD** carry a `digest` attribute `"<algo>:<hex>"` over its canonical byte stream:

```
H( object_path ‖ 0x00 ‖ dtype_str ‖ 0x00 ‖ shape_csv ‖ 0x00 ‖ raw C-order little-endian bytes )
```

`dtype_str` is the NumPy dtype string with explicit byte order normalised to little-endian
(`"<i2"`, `"<u8"`, `"|b1"`). Variable-length string datasets hash the UTF-8 payloads separated by
`0x00`. The digest covers **decompressed** content, so recompression does not invalidate it.

Datasets under `index/` (§14.3) are **excluded**: they carry no `digest`, and writers **MUST NOT**
stamp one. An index is derived, regenerable, and already bound to its source by `source_digest`
(§13.3). The exclusion is normative because it is not merely an omission a writer may make up its
own mind about — see §13.2.

### 13.2 Content id

The sample root **SHOULD** carry `content_id`, the Merkle root over the sorted digest list:

```
lines = sorted( f"{path}\t{digest}\n" for every dataset with a digest )
       + [ f"meta\t{H(meta_json_utf8)}\n" ]
       + [ f"@{obj}\t{H(canonical_attrs(obj))}\n" for every object with spec-defined attributes ]
content_id = "<algo>:" + hex( H( "".join(lines) ) )
```

`canonical_attrs` serialises the object's spec-defined attributes as sorted-key JSON with arrays as
nested lists and floats in `repr` shortest round-trip form. Each of the three groups of lines is
sorted independently and they are concatenated in the order shown. Paths are relative to the **sample
root**, so a sample extracted from a collection keeps its `content_id` (§2.2).

**`index/` is outside `content_id` entirely** — neither its datasets (they carry no digest, §13.1)
nor its attributes contribute a line. This is normative, not incidental: `content_id` is advertised
as a cache and dedup key *across implementations*, so two writers that disagree about whether a
derived cache is part of a sample's identity would compute different addresses for the same file and
the key would be worthless. It also means building, rebuilding or dropping an index does not change
the address of the sample it was built from, which is the property that makes the cache safe to
regenerate. A corrupted index is therefore **not** detectable through `content_id`; it is guarded by
`source_digest` (§13.3) against staleness, and is regenerable by definition.

At the root, the covered attributes are exactly `medh5_version`, `medh5_kind` and `medh5_profiles`.
`created` and `generator` are **excluded**, and `content_id` obviously cannot cover itself: two
byte-identical samples written an hour apart by different tools **MUST** share a `content_id`, or it
is not a content address and cannot serve as a cache or dedup key. An object's own `digest` attribute
is likewise excluded from its `canonical_attrs`, because the dataset lines already carry it.

Properties this buys, all absent from 0.x's single monolithic hash:

* **Incremental** — adding one annotation rehashes one object plus the root, not every voxel in the file.
* **Partial verification** — a reader that touched only `images/CT` can verify only `images/CT`.
* **Content addressing** — `content_id` is a cache key and a dedup key across a cohort.
* **Locality** — a mismatch names the object that changed.

### 13.3 Derived index invalidation

Every object under `index/` (§14.3) **MUST** carry `source_digest`, the digest of the annotation it
derives from. An annotation is a *group*, and only datasets carry a `digest`, so the quantity is
defined here: `source_digest` is

```
H( concat( sorted( f"{child_name}\t{child_digest}\n" for every dataset directly in the group ) ) )
```

prefixed `"<algo>:"`, where `child_name` is the dataset's name within the annotation group (`data`,
`layer_class_ids`, …) and `child_digest` is its §13.1 digest. Readers **MUST** ignore an index entry whose `source_digest` does not match the current
digest of its source, and **MUST NOT** treat that as a file error. Derived caches therefore cannot go
silently stale, and there is no invalidation protocol to get wrong.

---

## 14. Storage and performance

### 14.1 Chunking

Normative requirements:

* Image and voxel-annotation datasets **MUST** be chunked.
* `layers`, `bitmask`, `probmap` and `displacement` **MUST** use chunk shape `(1, *spatial_chunk)`, so
  one layer / plane / channel / component is readable without decompressing the others.
* Voxel annotations on a grid **SHOULD** use the same `spatial_chunk` as the images on that grid, so an
  image patch and its labels touch congruent chunk sets.

Recommended sizing (the 0.x L3-aware optimizer, generalised):

* target 0.5–4 MiB per chunk, defaulting to ≈ 80 % of one L3 slice;
* start from `2^ceil(log2(patch_hint))` per spatial axis, grow along the axis with the smallest
  chunk/patch ratio, and stop when the mean ratio exceeds 1.5;
* never exceed the array extent on any axis;
* `time` and `channel` axes get chunk 1 unless the whole axis is always read together.

### 14.2 Codec profiles

A file's datasets need not share a codec. Writers **SHOULD** expose these named profiles; the codec
actually used is discoverable from the HDF5 filter pipeline.

| Profile | Images | Labels / fields | Intended use |
|---|---|---|---|
| `training` | Blosc2 lz4 L1 + shuffle | Blosc2 lz4 L1 + shuffle | hot dataloader path; fastest decompression |
| `balanced` (default) | Blosc2 zstd L3 + shuffle | Blosc2 zstd L3 + bitshuffle | general use |
| `archive` | Blosc2 zstd L9 + bitshuffle | Blosc2 zstd L9 + bitshuffle | cold storage, distribution |
| `portable` | gzip L4 + shuffle | gzip L4 + shuffle | readers without `hdf5plugin` |

Measured on a 192×256×256 synthetic CT (12.6 M voxels, `int16` HU, 32×64×64 chunks):

| Profile | Write | Size | Ratio | 64³ patch read | Full-volume read |
|---|---|---|---|---|---|
| `training` (lz4 L1) | 0.03 s | 12.80 MiB | 1.9× | 0.08 ms | 0.01 s |
| `balanced`-ish (lz4hc L8) | 0.34 s | 12.33 MiB | 1.9× | 0.08 ms | 0.01 s |
| `archive` (zstd L9 + bitshuffle) | 2.39 s | 9.53 MiB | 2.5× | 0.08 ms | 0.03 s |
| `portable` (gzip L4) | 0.37 s | 9.72 MiB | 2.5× | 0.08 ms | 0.09 s |

`portable` reaches archive-class ratios but decompresses ~3× slower in bulk; `training` writes ~80×
faster than `archive` for a ~34 % size penalty. Patch reads are codec-insensitive at this chunk size
because a 64³ patch touches few chunks — which is exactly what §14.1 is for.

`portable` exists because Blosc2 requires `hdf5plugin` on the reader. A file written with `portable`
is readable by stock `h5py`, MATLAB, R `rhdf5` and `h5dump` with no plugins.

### 14.3 Sampling index (`training` profile)

`index/<ann_id>/` caches what a sampler would otherwise recompute per epoch:

| Object | Shape / type | Meaning |
|---|---|---|
| `class_ids` | `(C,)` `uint16` | classes covered by this index |
| `voxel_counts` | `(C,)` `int64` | foreground voxels per class — class-balanced sampling, loss weights |
| `class_bboxes` | `(C, S, 2)` `float32` | tight bounds per class — crop-to-foreground |
| `fg_coords/<class_id>` | `(n_c, S)` `int32` | uniform subsample of foreground voxel coordinates, `n_c ≤ max_coords` |
| `occupancy` | `(C, *coarse_shape)` `bool` | OPTIONAL low-res occupancy (default 1/8 per axis) for block-level rejection sampling |
| attr `source_digest` | `str` | §13.3 |
| attr `max_coords`, `seed` | | reproducibility of the subsample |

Measured effect on foreground patch sampling (160³ volume, one class, 33 533 foreground voxels):

| Path | Time | Resident |
|---|---|---|
| 0.x: load full mask + `np.argwhere` | 9.2 ms | 0.8 MiB (grows with volume × class count) |
| 1.0: read 4096-coordinate index | **0.52 ms** | **48 KiB** |

That is 18× faster per sample and, more importantly, **O(1) in volume size**. The 0.x `PatchSampler`
cached `argwhere` output per (file, class) in process memory; at 512³ with 200 classes that cache is
tens of GiB and cannot exist.

### 14.4 Concurrency and the write model

* **One writer.** HDF5 has no multi-writer story. Writers **MUST** hold exclusive access.
* **Create** is atomic: write to a sibling temporary file, `fsync`, `os.replace`, `fsync` the
  directory. (0.x already does this; 1.0 keeps it normative.)
* **File size scales with the record.** A subject-scoped sample holds every timepoint, so files are
  larger than a study-scoped equivalent and copy-on-write amend costs proportionally more. Writers
  curating long series **SHOULD** either accept the rewrite cost, use attribute-only in-place edits
  where applicable, or emit one sample per timepoint (§2.2). Appending a new timepoint to an existing
  sample is an amend, not an append.
* **Amend** — adding an annotation, correcting metadata — **MUST** default to copy-on-write: build a
  new file from the old and atomically replace. HDF5 does not reclaim space on `del`, so repeated
  in-place add/remove monotonically bloats a file and fragments its chunk index. In-place amendment
  is permitted only for attribute-only edits and **MUST** be opt-in.
* **Amend preserves unknown objects.** A 1.0 writer amending a file containing objects from a future
  minor version **MUST** copy them through untouched.
* **Readers** open `mode="r"`. Concurrent readers across processes are safe. A single `h5py.File`
  **MUST NOT** be shared across threads without external locking, nor inherited across `fork` — a
  handle cache **MUST** be keyed by PID and dropped in the child.
* **SWMR** (`libver="latest"`, `swmr_mode=True`) **MAY** be used to read a file while it is being
  appended; readers **MUST** re-verify `content_id` before trusting a SWMR snapshot.
* **Network filesystems**: HDF5 file locking is unreliable on NFS/Lustre/GPFS. Tooling **SHOULD**
  document `HDF5_USE_FILE_LOCKING=FALSE` and **MUST NOT** set it implicitly.

### 14.5 Reader guidance (normative for the reference implementation)

* Slice in **one** call: `d[(k, *roi)]`, never `d[k][roi]`. The latter materialises the whole
  sub-array first — measured 40× slower for a 64³ ROI out of a 160³ `uint64` bitplane (32.8 ms vs
  0.8 ms). The reference API **MUST NOT** expose an interface that makes the slow form natural.
* Read a multi-class region **by plane, not by class**. In `layers` and `bitmask` one stored plane
  serves many classes, so answering a *C*-class query with *C* reads re-decompresses the same chunks
  — up to 64× for a bitplane. Group the requested classes by the plane that carries them and read
  each plane once; measured 4.0 ms versus 11.0 ms for an eight-class 64³ patch, and the gap widens
  with class count.
* Prefer whole-chunk-aligned ROIs when the caller does not care about exact placement.
* `read_direct_chunk` **MAY** be used to hand compressed chunks to a GPU decompressor; the format
  imposes nothing that prevents it.

---

## 15. Validation

### 15.1 Levels

| Level | Checks |
|---|---|
| `structural` | layout, required attributes, dtypes, shapes, identifier syntax, JSON schema of `/meta` |
| `semantic` | cross-references resolve; geometry consistency; class ids ⊆ label set; encoding invariants; transform frame chaining; profile requirements |
| `integrity` | `digest` per object, `content_id`, index `source_digest` currency |
| `strict` | all of the above with warnings promoted to errors |

### 15.2 Error codes

Codes are stable API: a code's meaning never changes and a retired code is never reused. Validators
emit `(code, severity, location, message)` where `location` is an HDF5 path or a JSON pointer into
`/meta`. The table below is complete for 1.0; a minor version may add codes but **MUST NOT** redefine
one.

| Range | Domain | Examples |
|---|---|---|
| `E0xx` | container | `E001` missing `medh5_version`; `E002` unsupported major version; `E003` bad identifier; `E004` `/meta` absent or not valid JSON; `E005` `/meta` fails schema; `E006` missing or unknown `medh5_kind`; `E007` missing `medh5_profiles` or unknown profile; `E008` a group required by §2.3 is absent; `E009` a declared profile's requirements are not met; `E010` a sample root in a `collection` lacks its own `content_id` |
| `E1xx` | geometry | `E101` referenced grid does not exist; `E102` `direction` not orthonormal; `E103` spatial axes not trailing/contiguous; `E104` `spacing ≤ 0`; `E105` multiscale geometry inconsistent; `E106` grid without `timepoint` in a multi-timepoint sample; `E107` grid `timepoint` not declared; `E108` `timepoints` empty, or `index` not dense and increasing; `E109` required grid attribute missing or of the wrong rank; `E110` `axis_kinds` invalid for the declared dimensionality; `E111` `grids` contains no grid |
| `E2xx` | images | `E201` `images` empty; `E202` image shape ≠ grid shape; `E203` unknown `value_type`; `E204` `channel_names` length ≠ channel extent; `E205` required image attribute missing |
| `E3xx` | label set | `E301` missing label set for a declared profile; `E302` duplicate class id or key; `E303` reserved id used; `E304` hierarchy cycle; `E305` `ref` label set lacking `uri`/`sha256`, or unresolvable; `E306` class entry missing a required field, out of id range, or naming an unknown parent |
| `E4xx` | annotations | `E401` unknown `kind`; `E402` class id not in label set; `E403` `annotated_class_ids ⊄ class_ids`; `E404` encoding invariant violated (e.g. a class in two layers); `E405` shape mismatch with grid; `E406` box `lo > hi`; `E407` `rotations` not a proper rotation; `E408` offsets not monotonic; `E409` `timepoints` references an undeclared timepoint; `E410` a dataset required by the `kind` is absent; `E411` dataset dtype not permitted for the `kind`; `E412` required annotation attribute missing; `E413` reference to a skeleton, correspondence, ignore mask or source annotation that does not exist; `E414` `space` invalid for the annotation's grid or frame |
| `E5xx` | transforms | `E501` composite frame chain broken; `E502` unknown transform kind; `E503` field grid not in `from_frame`; `E504` affine last row ≠ `[0…0 1]`; `E505` `inverse_id` not mutually consistent |
| `E6xx` | curation | `E601` dangling `prov` reference; `E602` unknown `quality` key; `E603` unknown agent or activity type; `E604` non-RFC3339 timestamp; `E605` activity names an undeclared agent |
| `E7xx` | integrity | `E701` object digest mismatch; `E702` `content_id` mismatch; `E703` malformed digest string |
| `W9xx` | warnings | `W901` no digests; `W902` uncompressed or unchunked bulk dataset; `W903` no `deidentification`; `W904` partial coverage without an ignore region; `W905` stale `index/` entry; `W906` conflicting split claims; `W907` `float32` storage where `int16 + rescale` is lossless; `W908` `layers` count far from the greedy-colouring optimum; `W909` one `instance_id` carrying two class ids; `W910` grids in different timepoints sharing a `frame_uid`; `W911` multi-timepoint sample with no transform relating any two timepoints; `W912` a class used by an annotation carries no ontology binding |

---

## 16. Versioning and extension

* `medh5_version` is `MAJOR.MINOR`. A **minor** bump may add objects, attributes, `kind` values,
  activity types and error codes. It **MUST NOT** change the meaning of existing ones, remove a
  requirement, or alter a coordinate or direction convention.
* Readers **MUST** reject an unknown MAJOR, **MUST** accept a higher MINOR, and **MUST** ignore
  objects, attributes and enum values they do not recognise.
* Third-party extensions live under `/meta → extra.<reverse-dns-namespace>` and under HDF5 groups
  named `x_<namespace>_<name>`. Neither is touched by validators, and both survive amend.
* Registering a new annotation `kind` or transform `kind` requires a MINOR bump and an entry in
  §6.3 / §10.1.

**Reserved names.** The following were specified during 1.0 design, deliberately deferred, and are
**reserved**: a 1.0 writer **MUST NOT** emit them, and a future minor version **MUST** use these names
for the meanings given rather than reassigning them.

| Reserved | Kind of | Deferred because | Intended meaning |
|---|---|---|---|
| `rle` | annotation `kind` | Chunked compression already leaves unwritten chunks unallocated, so the disk saving over `instances` is small, while it costs O(1) ROI access and a second sparse code path (§7.0). Converters decode run-length payloads instead (§7.6). | per-class run-length encoding over C-order voxel indices |
| `wide_labels` | profile | `uint16` class ids cap a vocabulary at 65 534, which covers every clinical use case surveyed; `uint8`/`uint16` labelmap dtype economics dominate the common path (§5.3). | `uint32` class ids for vocabularies beyond 65 534 classes |

---

## Appendix A — Worked example

A two-timepoint oncology sample: baseline CT + PET, three-month follow-up CT, organ and lesion
segmentation at both visits, a response assessment across them, and the registration relating them.

```
$ h5ls -r case_0001.medh5
/                        Group
/annotations             Group
/annotations/lesions_tp0 Group      # kind=instances, grid=ct_tp0
/annotations/lesions_tp0/boxes         Dataset {7, 3, 2}       float32
/annotations/lesions_tp0/class_ids     Dataset {7}             uint16
/annotations/lesions_tp0/instance_ids  Dataset {7}             uint32   # 1…7
/annotations/lesions_tp0/mask_data     Dataset {41213}         uint8
/annotations/lesions_tp0/mask_offsets  Dataset {8}             uint64
/annotations/lesions_tp0/mask_shapes   Dataset {7, 3}          int32
/annotations/lesions_tp1 Group      # kind=instances, grid=ct_tp1
/annotations/lesions_tp1/instance_ids  Dataset {6}             uint32   # 1,2,4,5,7,8 — 3 and 6 resolved, 8 is new
/annotations/organs_tp0  Group      # kind=layers, grid=ct_tp0
/annotations/organs_tp0/data           Dataset {5, 160, 160, 160} uint16
/annotations/organs_tp0/layer_class_ids Dataset {5, 48}        uint16
/annotations/organs_tp1  Group      # kind=layers, grid=ct_tp1
/annotations/organs_tp1/data           Dataset {2, 152, 160, 160} uint16
/annotations/response    Group      # kind=classification, scope=sample, timepoints=["tp0","tp1"]
/annotations/response/class_ids        Dataset {1}             uint16   # partial_response
/annotations/response/values           Dataset {1}             float32
/annotations/staging_tp0 Group      # kind=classification, scope=timepoint, scope_ids=[0]
/grids                   Group
/grids/ct_tp0            Group      # timepoint=tp0, frame_uid=…100
/grids/ct_tp1            Group      # timepoint=tp1, frame_uid=…101  — a new frame
/grids/pet_tp0           Group      # timepoint=tp0, frame_uid=…100  — same frame as ct_tp0
/images                  Group
/images/CT_tp0           Dataset {160, 160, 160} int16   # grid=ct_tp0, HU
/images/CT_tp1           Dataset {152, 160, 160} int16   # grid=ct_tp1, shorter z coverage — nothing resampled
/images/PET_tp0          Dataset {80, 80, 80}    float32 # grid=pet_tp0, SUVbw
/index                   Group
/index/organs_tp0        Group      # voxel_counts, class_bboxes, fg_coords/*, source_digest
/meta                    Dataset {SCALAR}        string  # identity, timepoints, label_set, provenance, quality
/transforms              Group
/transforms/tp0_to_tp1   Group      # kind=affine, from_frame=…100 → to_frame=…101
```

Reading it: `PET_tp0` needs no transform to reach `CT_tp0` — same `frame_uid`. `CT_tp1` does, because
follow-up is a new frame. Lesion 3 appears at baseline and not at follow-up, and because
`annotated_class_ids` covers lesions at both visits, that absence means *resolved*, not *unexamined*.

## Appendix B — 0.x → 1.0 mapping

| 0.x | 1.0 |
|---|---|
| `images/<name>` (all same shape) | `images/<name>` + one `grids/ref` shared by all |
| `images.attrs.spacing/origin/direction/coord_system/axis_labels` | `grids/ref` attributes; `direction` stored 2-D |
| `images.attrs.shape` | `grids/ref.shape` |
| `images.attrs.patch_size` | `grids/ref.patch_hint` |
| `seg/<name>` boolean volumes | one `annotations/<id>` of kind `layers` (or `instances` / `bitmask` per §7.6); mask names become label-set `key`s |
| `bboxes` `(n, ndim, 2)` int | `annotations/<id>` kind `boxes`, `float32`, `space="index"`, `lo = min − 0.5`, `hi = max − 0.5` |
| `bbox_scores`, `bbox_labels` | `scores`, `class_ids` (+ new label-set entries) |
| `label`, `label_name` | `annotations/<id>` kind `classification`, `scope="sample"` |
| (no notion of time) | a single declared timepoint `tp0` with `index = 0`; every grid gets `timepoint = "tp0"` |
| `extra.review` | `/meta → provenance.activities` (type `review`) + `/meta → quality` |
| `extra.nnunetv2` | `/meta → extra.nnunetv2` (preserved verbatim) + a generated label set |
| `checksum_sha256` (whole file) | per-object `digest` + root `content_id` |
| `has_seg`, `has_bbox`, `seg_names`, `image_names` | removed — derived by enumerating groups; a flag that can disagree with the data is a bug generator |

`medh5 migrate` performs this mapping and emits a report of every non-mechanical decision
(chosen encoding, minted class ids, half-voxel box conversion, timepoint grouping). Class ids are
minted once for a whole cohort and written to a reviewable sidecar, so `liver` cannot be id 1 in one
sample and id 2 in the next.

**Grouping.** Converters that read study-scoped sources — 0.x files, a DICOM tree, an nnU-Net
dataset — **SHOULD** default to grouping by subject, producing one multi-timepoint sample per
patient, because that is the unit this specification defines (§3.7). When patient identity cannot be
established across studies — commonly because de-identification replaced it — a converter **MUST**
fall back to one sample per study and **MUST** report the fallback prominently rather than silently
emitting study-scoped samples that look subject-scoped. Migration from 0.x defaults to study
grouping, since a 0.x file carries no reliable subject key of its own.

## Appendix C — Implementation status

### C.1 Reference implementation

Sections §2–§15 are **implemented** in the `medh5` package and exercised by a conformance corpus
(§15) of 115 files: valid samples covering every encoding, annotation kind, transform kind,
dimensionality, profile and container kind, plus one deliberately-invalid file per diagnostic code.
Running the corpus against a validator is how a third-party implementation demonstrates conformance:

```
$ medh5 conformance run ./corpus
115/115 cases pass
```

**Every code in §15.2 has a corpus case.** The implementation gates on `ruff`,
`mypy --strict` and ≥ 90 % test coverage, and a test asserts that the §15.2 table and the
implementation's code registry are identical, so the two cannot drift.

The §14 performance claims are reproducible rather than asserted: `medh5 bench` re-measures them on
any machine. On a 192×256×256 synthetic CT with eight classes, a multi-class 64³ label read costs
4.0 ms, foreground centre sampling 0.90 ms (O(1) in volume size, via §14.3), a metadata-only read
0.21 ms, and `open()` → first patch 2.4 ms.

Fifteen clauses have been corrected — ten during implementation and five in the 1.x package releases
that followed — each because writing the code showed the text was not implementable, not unambiguous,
or not what the implementation could honestly promise, as written:

| Clause | Correction |
|---|---|
| §2.4 | `/meta` **MUST NOT** be compressed. HDF5 filters do not apply to variable-length data, so the previous "SHOULD compress when > 64 KiB" was not expressible. |
| §5.1 | The canonical serialization behind `label_set.sha256` is now defined (sorted keys, no whitespace, UTF-8, carriage fields excluded), so two implementations agree on the digest. |
| §13.2 | `content_id` covers exactly `medh5_version`, `medh5_kind` and `medh5_profiles` at the root. `created` and `generator` are excluded, or identical samples written at different times would not share an address. |
| §13.3 | "the `digest` of the annotation" is now defined for a multi-dataset group, since only datasets carry digests. |
| §1.3 | `det` requires a §8 annotation whose `task` is `detection`, not any §8 kind: §6.3 assigns contours and meshes to segmentation, so the two clauses contradicted each other. |
| §9 | The `class_ids` *dataset* (asserted classes) and the same-named §6.2 *attribute* (classes the annotation can express) are now explicitly distinguished, since `annotated_class_ids ⊆ class_ids` would otherwise make "looked for and not found" inexpressible. |
| §2.2 | §2.2 requires every sample root in a collection to carry its own `content_id`, but §15.2 had no code to report a missing one. `E010` was added. |
| §7.4 | W909 is **sample-scoped**, matching the scope of `instance_id` itself. Checking it per annotation would miss the case it exists for: one lesion classified differently at two visits, each annotation internally consistent. |
| §8.1 | The box↔slice rounding is `floor(x + 0.5)`, stated explicitly. "round" was read as a language's default, and both Python's `round` and NumPy's `rint` round half to **even** — under which `lo + 0.5` and `hi + 0.5` round in opposite directions, so a one-voxel box on integer edges became empty or two voxels wide depending on its parity. The extent identity in the same clause was the thing being violated. |
| §2.1 | `digest_algo` names `sha256`, `sha512` and `blake2b` — the algorithms every Python ships — rather than `blake3` and `xxh3-128`, which the reference implementation never carried; a validator reported the two the text permitted as **E703** malformed, so the text and the validator could not both be right. Any other value is E703, normatively. |
| §7.5 | `threshold` is now a spec-defined `probmap` attribute, default `0.5`. §7.6 already said transcoding was "lossless only under a declared threshold", and the reference reader honoured a `threshold` attribute, but nothing defined it — so it sat outside `content_id` and two implementations could disagree about `contains` for one file. |
| §3.4 | The sentence requiring a *validator* to report an annotation compared with an image on another frame with no relating transform described a query, not a property of a file; no code existed for it and none could. It is now reader guidance, and the refusal lives in the reference loader. |
| §5.1 | Collections carrying one label set at `/` with `uri = "medh5:/label_set"` is MAY, not SHOULD: the reference implementation neither writes nor resolves it, and §5.1 already requires the inline form at the sizes where it would matter. |
| §7.1, §3.2, §15.2 | Two rules the writer enforced and the validator did not are validated: a `labelmap` stored `uint16` where §7.1 requires `uint8` is E411, and a `time` axis without `time_values` is E109. E603's summary reads "unknown agent or activity type", as this table has always said. |
| §13.1, §13.2 | `index/` is **excluded** from object digests and from `content_id`, normatively rather than by convention. The reference implementation had always skipped it — a derived cache should not change the address of the sample it derives from — but the text did not say so, so a conforming implementation that stamped index digests would compute a different `content_id` for the same bytes, and `content_id` is only useful as a cross-implementation key if every implementation agrees on what it covers. |

### C.2 Prototype checks

The specification is also accompanied by an executable prototype,
[`docs/examples/reference_writer.py`](../examples/reference_writer.py), which
writes a sample exercising `core + seg + det + cls + reg + curation + training + longitudinal` by
following this document literally, then checks it. The sample is one subject at two timepoints:
baseline CT + PET sharing a frame, a follow-up CT on its own grid with shorter z coverage and its own
frame, organ and lesion annotations at both visits, a RECIST response label across them, and the
registration relating them. It confirms:

| Check | Result |
|---|---|
| `/meta` against `schemas/medh5-sample-1.0.schema.json` | passes |
| Cross-reference checks (grids, label set, `prov`, `quality` — E101/E402/E403/E601/E602) | clean |
| `direction` orthonormality and 2-D storage (E102) | clean |
| `layers` invariant: every class in exactly one layer (E404) | clean |
| Box `lo ≤ hi` (E406) | clean |
| Per-object digests and `content_id` (§13) | all match |
| `index/` `source_digest` currency (§13.3, W905) | current |
| index→world→index round-trip (§3.3) | exact (0.0 error) |
| box ↔ slice round-trip (§8.1): `[11.5, 39.5] ↔ slice(12, 40)`, extent 28 | exact |
| `instances` decode: box extents match stored mask shapes (§7.4) | exact |
| Lossless transcoding `layers ↔ bitmask` for every class (§7.6) | exact |
| Two grids sharing a `frame_uid` with different shape and spacing (§3.4) | supported |
| Two timepoints declared, every grid bound to one (§3.7, E106/E107/E108) | clean |
| Grids in different timepoints on different frames (§3.4, W910) | clean |
| Per-timepoint grids with different extents — 160 vs 152 slices, no resampling | supported |
| Instance ids tracked across timepoints (§7.4): persisted `[1]`, resolved `[3, 6]`, new `[8]` | exact |
| Change label `scope = "sample"` with `timepoints = ["tp0","tp1"]` (§9, E409) | clean |
| One `instance_id` never carrying two class ids (W909) | clean |
| A transform relating two timepoints exists (W911) | clean |

The prototype predates the reference implementation and is kept because it is short enough to read
end to end: it demonstrates the format with no library between the reader and the bytes. The
implementation and its conformance corpus (§C.1) are what a reader should measure against.
