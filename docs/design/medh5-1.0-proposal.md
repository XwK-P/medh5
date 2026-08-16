# medh5 1.0 — Design Proposal

**Status:** Draft for review · **Author:** design pass over medh5 0.6.0 · **Date:** 2026-08-15
**Companion documents:** [Specification](../spec/medh5-1.0.md) · [Implementation plan](medh5-1.0-implementation-plan.md)

---

## 1. Summary

medh5 0.x is a competent single-sample HDF5 container: multi-modality images, boolean masks, integer
boxes, one scalar label, spacing/origin/direction, atomic writes, checksums. It is well engineered
inside its model. The model itself is the problem — it encodes **one task** (binary-mask segmentation
with a classification label bolted on) at **one scale** (tens of classes, one grid, one annotator,
one moment in time).

Version 1.0 replaces that model with five load-bearing ideas:

1. **The sample is the subject, not the study.** One file holds one patient at one *or more*
   timepoints, each with one or more images. Change detection, response assessment, lesion tracking
   and follow-up registration become in-file relationships instead of out-of-band conventions
   linking separate files — and because a sample never spans subjects, assigning whole files to
   splits is patient-leakage-safe by construction.
2. **Grids are first-class.** Images declare which lattice they live on. Several lattices per sample,
   related by frames of reference and explicit transforms. This unblocks native-resolution PET/CT,
   multi-sequence MR, multiscale pyramids, per-timepoint acquisition geometry and registration in one
   move.
3. **Annotations are a uniform abstraction over interchangeable encodings.** A segmentation is
   `contains(class, voxel)`. How it is stored — labelmap, layers, bitmask, instances, probmap —
   is chosen by measurement and hidden behind one read API. This is what makes hundreds of
   overlapping classes cheap without exposing users to bit-twiddling.
4. **Ground truth carries its own epistemics.** Every annotation states which classes were *looked
   for*, who produced it, from what, and how good it is. Absence of a label stops meaning "absent"
   and starts meaning what it actually meant.
5. **Metadata has exactly one home per fact.** Arrays and per-object facts in HDF5 attributes;
   documents in one schema-validated JSON blob. No mirrors, no `extra` dumping ground for things the
   format should model.

Nothing here requires backward compatibility, and none is retained. A migration tool maps 0.x files
mechanically (Spec Appendix B).

---

## 2. What breaks in 0.6.0, with evidence

### 2.1 Label-space scaling: one dataset per class

`seg/` holds one boolean volume per mask. On a 160³ phantom with 200 structures — 24 mutually
exclusive organs plus 176 overlapping substructures/lesions, 0.25 labels per voxel — measured at
identical codec settings (Blosc2 lz4 L1 + shuffle, 32×64×64 chunks):

| Encoding | On disk | 64³ patch, **all** classes | 64³ patch, **one** class | HDF5 objects |
|---|---|---|---|---|
| **0.x per-class `bool`** | 3.57 MiB | 116.9 ms | 0.51 ms | 200 |
| `layers`, L = 5 | **0.55 MiB** | **6.7 ms** | **0.09 ms** | 1 |
| `bitmask`, P = 4 | 0.68 MiB | 10.1 ms | 2.79 ms | 1 |
| `instances` | **0.08 MiB** | — | — | 4 |

The all-class read is the one that matters: a multi-label training batch needs every class in the
patch, and 0.x pays 200 chunk lookups, 200 decompressions and 200 h5py round-trips to get it. At
117 ms per patch, a single dataloader worker caps out near 8 patches/s on *labels alone* — before
touching images or augmentation.

Disk cost compounds at the dataset level: HDF5 spends ≈ 360 bytes of metadata per dataset (measured:
1 000 trivial datasets ⇒ 359.9 KiB of pure bookkeeping). A 50 000-sample cohort with 200 classes
carries ~3.7 GiB of HDF5 group bookkeeping that describes nothing.

And it cannot express the actual requirement anyway: 0.x masks are `bool`, so a 200-class annotation
*is* 200 volumes. There is no representation in the format where overlapping structures share storage.

### 2.2 One shape for everything

`_validate_write_inputs` rejects any image whose shape differs from the first. PET at 4 mm and CT at
0.7 mm therefore must be resampled at ingest — a lossy, irreversible decision made before anyone knows
what the model needs, with no record that it happened. Multiscale pyramids, native-resolution
multi-sequence MR and any registration workflow are structurally impossible.

### 2.3 Geometry is optional and under-specified

`spacing`, `origin`, `direction`, `coord_system` are all `Optional`, `coord_system` is documented as
"informational", and `direction` is stored flattened. `SpatialMeta.as_affine()` returns `None` when
the direction is near-identity, so callers get "no affine" for a perfectly valid axis-aligned volume
and have to reconstruct one. There is no statement anywhere of whether an integer index is a voxel
centre or a corner, nor of which axis order `direction`'s columns follow. Every consumer guesses; the
guesses differ.

### 2.4 Ground truth is a single scalar plus boxes

- Classification: one `label: int | str`. No multi-label, no hierarchy, no per-region scope, no
  ordinal scales, no negative assertions, no per-rater labels.
- Detection: integer `(n, ndim, 2)` boxes only. No OBB, no keypoints, no per-box attributes, no world
  coordinates, and integer corners that cannot survive a resample or a rotation.
- Registration: absent entirely.
- Meshes, contours, landmarks: absent.

### 2.5 There is no notion of time

0.x has one `label`, one set of images, and no way to say *when*. A subject imaged three times is
three unrelated files, joined — if at all — by a filename convention and a `subject_id` buried in
`extra`. Everything longitudinal is then homeless:

- **The pair is the finding.** In oncology, neurology and screening the clinically meaningful object
  is the *change* between visits. There is nowhere to record "partial response between baseline and
  month 3" that binds to the two studies it compares.
- **Lesion correspondence is lost.** "This nodule is that nodule, three months later" cannot be
  written down, so growth curves and per-lesion response have to be recomputed, unreliably, by
  matching coordinates after the fact.
- **The baseline→follow-up transform has no owner.** It relates two files, so it lives in neither.
- **Splits leak.** A per-study file means the same patient's baseline and follow-up can land in
  train and test respectively. Preventing that is left to a `group_by` argument the caller must
  remember to pass, on a field the format does not require.

### 2.6 Provenance is a nested dict in a JSON blob

`extra["review"]` holds `{status, annotator, timestamp, notes, history[]}`. That records *that* a
review happened. It cannot record what produced the data being reviewed — so the dominant real
workflow, "model pre-annotates, human corrects, second human approves", has no representation. There
is no link from an annotation to the activity that made it, no software versions, no tool parameters.

Worse, `extra` is simultaneously the provenance store, the nnU-Net interop store, the checksum
subsystem store and the user scratchpad — with a hand-rolled warning system (`_warn_malformed_extra`)
policing the parts that turned out to be load-bearing. That is a schema, discovered late, without a
schema's guarantees.

### 2.7 Monolithic checksum

`compute_checksum` hashes every image, every mask and every box array into one digest. Adding one
annotation to a 2 GB file rehashes 2 GB. Verifying that `images/CT` is intact requires reading
everything. A mismatch says "something changed" and nothing more. And `update()` recomputes it on
every mutation, so curation workflows pay full-file hashing per edit.

### 2.8 The sampler cannot scale

`PatchSampler._foreground_start` loads the entire mask and calls `np.argwhere`, caching the result
per (file, mask) in a process-level dict. Measured on 160³ with one class: 9.2 ms and 0.8 MiB
resident per (file, class). At 512³ with 200 classes that cache is tens of gigabytes per worker. The
precomputed index costs 0.52 ms and 48 KiB, and is O(1) in volume size.

### 2.9 Denormalised flags

`has_seg`, `has_bbox`, `seg_names`, `image_names` and `shape` duplicate facts derivable by
enumerating groups — and `_validate_open_file` contains five separate checks for the case where they
disagree. Every one of those checks is a bug that the schema invited.

---

## 3. Design principles

| # | Principle | Consequence in the spec |
|---|---|---|
| 0 | **The file's unit is the subject** | A sample holds one patient's record across timepoints, so longitudinal relationships and split safety are structural, not conventional. §2.2, §3.7 |
| 1 | **One fact, one home** | Arrays and per-object facts in HDF5 attributes on the object they describe; documents in `/meta`. No mirroring, no derived flags. §2.4, §2.5 |
| 2 | **Storage layout is not API** | `contains(class, voxel)` is the contract; encodings are swappable and auto-selected. §7.7 |
| 3 | **Geometry is mandatory and unambiguous** | One affine formula, one voxel-centre convention, one box-corner convention, one transform direction. §3.3, §8.1, §10.2 |
| 4 | **Absence must be distinguishable from ignorance** | `annotated_class_ids`, ignore ids, explicit negative assertions. §6.2, §7.8, §11.3 |
| 5 | **Everything is attributable** | PROV agents/activities; every object links to the activity that made it. §11.1 |
| 6 | **Self-describing without our software** | `h5ls -r` + `h5dump -A` tells you everything structural; `portable` codec profile needs no plugins. §2.5, §14.2 |
| 7 | **Derived data is cheap and self-invalidating** | `index/` entries carry `source_digest`; stale entries are ignored, never trusted. §13.3 |
| 8 | **Measure, then choose** | Encoding selection, chunk sizing and codec profiles are decided from measured properties of the data, not defaults. §7.7, §14.1, §14.2 |

---

## 4. Key decisions and the alternatives considered

### 4.0 Sample scope: one subject, one or more timepoints

**Alternatives weighed**

| Option | Verdict |
|---|---|
| One sample = one study (0.x, implicitly) | Rejected — §2.5. Every longitudinal relationship becomes an out-of-band join, and split safety depends on a field the format does not require. |
| One sample = one study, plus a dataset-level "subject graph" sidecar | Rejected. Moves the join out of filenames and into a second file that can go stale, disagree with the samples, or be lost in transit. The correspondence between a lesion at two visits still has no home. |
| One sample = one subject's entire record | Rejected as a *requirement*. Twenty-year screening series and dense cine studies would produce unmanageable files, and copy-on-write amend would become punitive. |
| **One sample = one subject at one or more timepoints, curator's choice** | **Chosen.** The format fixes only what must be fixed — a sample never spans subjects — and leaves the grouping granularity to whoever knows the cohort. |

**How time enters the schema.** Timepoints are declared once in `/meta → timepoints` with an id, a
dense `index`, and `days_from_baseline` (which survives date shifting, unlike `date`). The binding to
data is a single attribute: **`timepoint` on the grid**. Everything else inherits — an image's
timepoint is its grid's, an annotation's is its grid's, a transform's endpoints are the timepoints of
the grids in its frames.

Putting it on the grid rather than the image was the real decision. A grid is where geometry meets
acquisition, and grids are empty attribute-only groups, so two visits with identical lattice geometry
are simply two grids. That keeps `timepoint` and `frame_uid` single-valued per grid and removes any
need for a per-image override — the alternative, `timepoint` on images with grids shared across time,
requires an override mechanism the moment anything is resampled to a common lattice.

**What falls out for free.** Instance ids were already specified as identifying one physical object;
scoping them to the sample makes *lesion tracking* nothing more than joining on `instance_id` across
timepoints — no `track` object, no correspondence table. Change labels are ordinary classification
annotations that name the timepoints they compare. And follow-up registration is the transform that
was already in the schema, now with both endpoints in the same file.

### 4.1 Multi-label voxel storage → four encodings behind one API

**Alternatives weighed**

| Option | Verdict |
|---|---|
| Keep one boolean volume per class | Rejected — §2.1. Fails the primary requirement outright. |
| One-hot `(C, Z, Y, X)` `bool` | Rejected. Same cost as per-class, no addressing benefit, worse chunking. |
| Bitmask only | Rejected as the sole encoding. Optimal only when overlap depth is high; measured 2.79 ms vs 0.09 ms against `layers` for a single-class patch read, because a `uint64` plane decompresses 32 B/voxel to answer a 1-bit question. |
| Sparse (RLE/COO) | Rejected as the sole encoding — large organs are *dense*, RLE inflates them and destroys O(1) ROI access — and then **deferred entirely**. Chunked compression already leaves unwritten chunks unallocated, so the disk saving over `instances` is small, while a run-length path costs a second sparse implementation to write, test and transcode. The `rle` name is reserved (spec §16); converters decode COCO and DICOM-SEG runs on ingest instead. |
| Layers only (Slicer-style) | Rejected as the sole encoding. Degenerates to one volume per class when the overlap graph is a clique — exactly the deep-hierarchy case. |
| **Four encodings + a selection rule + lossless transcoding** | **Chosen.** Each regime has a representation within 1.3× of its own optimum, and callers never see the difference. |

The cost of "four encodings" is borne once, in the reference implementation's resolver, and is
bounded: each encoding is 50–150 lines of well-tested `contains`/`dense`/`instances` logic. The cost
of picking one encoding is borne forever, by every user whose data is in the wrong regime.

**Why `layers` is the default.** For C classes and greedy-coloured overlap depth L, raw cost is
`2L` B/voxel (uint16 layers) versus `8·⌈C/64⌉` B/voxel (bitmask): layers wins whenever
`L < 4⌈C/64⌉` — for C ≤ 64 that is L < 4, for C = 200 it is L < 16. Real anatomy is sparse in the
overlap graph (measured mean degree 3.4 over 200 structures ⇒ L = 5), so `layers` is the common case
by a wide margin, and it is simultaneously the fastest for both single-class and all-class reads.

### 4.2 Metadata: HDF5 attributes vs one JSON document

**Chosen: both, with a partition rule and no overlap.**

- Attributes are typed, cheap, visible in `h5dump -A`, and attach to the object they describe. They
  are right for grid geometry, image semantics and annotation headers.
- JSON is right for the label set (a nested DAG), the provenance graph, quality records and free-form
  extras — none of which fit HDF5's attribute model without inventing compound-type gymnastics that
  no `h5dump` user will ever read.

The failure mode of hybrids is drift. The rule that prevents it: **no fact appears in both**. A
validator can check the partition mechanically because the spec enumerates which side each fact is on.

Rejected: **all-attributes** (label sets and PROV graphs become unreadable attribute soup);
**all-JSON** (a bare `h5ls` shows nothing, and geometry becomes invisible to non-medh5 tools);
**a compound-dtype label table** (better for `h5dump`, but the hierarchy DAG, ontology codes and
per-class properties are ragged — the natural expression is JSON).

### 4.3 Boxes: continuous edge coordinates in the grid's index space

The half-voxel question has to be answered exactly once, in the spec, or it is answered a hundred
times, inconsistently, in user code. Chosen: **one coordinate space** (continuous index coordinates
where integer = voxel centre, matching the affine), with **box corners measured at voxel edges**, so
`numpy slice a:b ⟺ [a−0.5, b−0.5]` and box extent is exactly `b − a` voxels. This is the
ITK/VTK convention.

Rejected: a second "edge space" coordinate system (two spaces is a footgun); integer corners
(0.x — cannot represent a resampled or rotated box); normalised `[0,1]` coordinates (loses meaning
when the grid changes, and invites silent aspect-ratio bugs).

`space = "world"` is also allowed, because a detection target defined in millimetres survives
resampling and a voxel-space one does not.

### 4.4 OBB as rotation matrices

Rejected quaternions (double cover: `q` and `−q` are the same rotation, so digests and equality
comparisons differ for identical geometry; and they do not generalise to 2D) and Euler angles
(twelve conventions, no way to know which). Rotation matrices are unambiguous, dimension-generic and
directly composable with the grid affine. `S²` floats per box is noise next to image data.

### 4.5 Registration: ITK point-transform direction, mandated

`T` with `from_frame = F`, `to_frame = M` satisfies `x_M = T(x_F)`; warping the moving image onto the
fixed grid samples M at `T(x)`. Deliberately **no attribute to select the opposite convention**: a
configurable convention means every consumer must handle both, which means half of them handle
neither correctly. Displacement components go on the leading axis (`(S, Z, Y, X)`, chunked `(1, …)`)
so one component or one ROI is readable in isolation.

### 4.6 Integrity: per-object digests + Merkle root

Rejected keeping the monolithic hash (§2.7) and rejected dropping checksums (clinical data
provenance requires them). Per-object digests make verification incremental, partial and local, and
the root doubles as a content-address for caching and dedup. Digests are computed over
**decompressed** canonical bytes, so recompressing a file for a different codec profile does not
invalidate them — which is what makes the `training`/`archive` profile switch a routine operation.

### 4.7 Derived caches: `source_digest`, not invalidation

`index/` could have been a sidecar file, a timestamp comparison or an explicit `--rebuild-index`
step. All three go stale silently. Embedding the source object's digest makes staleness *detectable
by construction*: a reader compares two strings and falls back to computing from source. There is no
protocol to get wrong and no correctness risk from a stale cache.

### 4.8 One sample per file, with an escape hatch

Kept: single-sample files are the unit of locking, content addressing, split membership and
`ln -s`-based cohort assembly — now subject-scoped (§4.0), which is what makes assigning whole files
to partitions leakage-safe. Added: `collection` files whose sample roots are *structurally
identical* to standalone samples, so packing and unpacking are pure copies. This addresses the small-
sample regime (2D radiographs, patches, cell crops) without letting a second layout leak into the
core schema — every rule in the spec is written against "the sample root", which is `/` or
`/samples/<key>`.

### 4.9 Why not adopt an existing format instead

| Format | Why not | What we take from it |
|---|---|---|
| **DICOM** (+ SEG, RTSTRUCT, SR, REG) | The interoperability standard, and the right archival target. But: per-slice objects, no chunked random access, no compression suited to ML, and reading a segmentation requires a full DICOM toolchain. Nobody trains from raw DICOM. | Frame-of-reference model, modality codes, acquisition keywords, de-identification profiles, the RTSTRUCT contour model |
| **NIfTI** | One array plus a 4×4 affine. No multi-label, no provenance, no boxes, no label names. Datasets become directory conventions with filename semantics. | The affine-is-the-truth discipline |
| **OME-NGFF / OME-Zarr** | Excellent multiscale + chunking model, and cloud-native. But Zarr's many-small-objects layout is a poor fit for per-sample local training corpora, and NGFF's annotation model is thin for clinical GT (no provenance, no coverage semantics, no registration). | Multiscale layout, the axes/`axis_kinds` model, transformation metadata |
| **nnU-Net raw layout** | A directory convention, not a format: no geometry validation, single labelmap, no overlap, no provenance. | `dataset.json` channel/label conventions for interop |
| **MONAI / torchio datasets** | Loader-side abstractions over other formats; no storage semantics of their own. | Sample-dict shape for the loader API |
| **HDF5 as raw substrate** | What we are doing — the value is the schema, not the container. | — |

The gap MEDH5 1.0 fills: **one local, chunk-random-access file per sample that carries every task's
ground truth plus the epistemics needed to train on it honestly.** No existing format does that.

---

## 5. Measured results

All benchmarks: macOS, Python 3.12, h5py 3.16, hdf5plugin 6.0, local SSD. Sources in
`docs/design/benchmarks/`. Numbers are medians over 10–50 repetitions.

### 5.1 Multi-label encodings — 160³, 200 classes, 0.25 labels/voxel, L = 5, P = 4

| Codec | Encoding | Size | 1-class 64³ read | all-class 64³ read |
|---|---|---|---|---|
| lz4 L1 + shuffle | per-class bool (0.x) | 3.57 MiB | 0.51 ms | 116.89 ms |
| lz4 L1 + shuffle | **layers** | **0.55 MiB** | **0.09 ms** | **6.69 ms** |
| lz4 L1 + shuffle | bitmask | 0.68 MiB | 2.79 ms | 10.14 ms |
| zstd L5 + bitshuffle | per-class bool (0.x) | 3.01 MiB | 0.33 ms | 139.27 ms |
| zstd L5 + bitshuffle | **layers** | **0.20 MiB** | 0.18 ms | 13.48 ms |
| zstd L5 + bitshuffle | bitmask | **0.15 MiB** | 10.31 ms | 37.06 ms |

`instances` on the same data: 0.08 MiB — 45× smaller than per-class dense, because cost tracks object
volume, not image volume.

### 5.2 Codec profiles — 192×256×256 `int16` CT, 32×64×64 chunks

| Profile | Write | Size | Ratio | 64³ read | Full read |
|---|---|---|---|---|---|
| `training` lz4 L1 | 0.03 s | 12.80 MiB | 1.9× | 0.08 ms | 0.01 s |
| lz4hc L8 (0.x default) | 0.34 s | 12.33 MiB | 1.9× | 0.08 ms | 0.01 s |
| `archive` zstd L9 + bitshuffle | 2.39 s | 9.53 MiB | 2.5× | 0.08 ms | 0.03 s |
| `portable` gzip L4 | 0.37 s | 9.72 MiB | 2.5× | 0.08 ms | 0.09 s |

Storing the same volume as `float32` instead of `int16 + rescale`: 36.75 MiB vs 12.33 MiB at lz4hc L8
— a **3.0× penalty for zero information gain**, which is why §4.2 of the spec makes `int16` HU the
recommendation and W907 flags the alternative.

### 5.3 Foreground sampling — 160³, one class, 33 533 foreground voxels

| Path | Time | Resident memory |
|---|---|---|
| 0.x: full mask + `np.argwhere` (cached per file×class) | 9.2 ms | 0.8 MiB, O(volume) |
| 1.0: 4096-coordinate index read | **0.52 ms** | **48 KiB**, O(1) |

### 5.4 Access-pattern trap

`d[k][roi]` versus `d[(k, *roi)]` on a 160³ `uint64` bitplane: **32.8 ms vs 0.8 ms** — the first
materialises the entire plane before slicing. The reference API must not make the slow form
expressible (§14.5).

---

## 6. What this costs

Honest accounting of the downsides.

| Cost | Assessment |
|---|---|
| **Spec size** — 1.0 is ~10× the surface of 0.x | Mitigated by conformance profiles: a segmentation-only user implements `core` + `seg` and ignores the rest. The `core` profile is smaller than 0.x's implicit schema *because* the derived flags are gone. |
| **Five voxel encodings to implement and test** | ~500 lines total plus a transcoding matrix. Property-based tests assert `contains()` equality across all ordered pairs, so the matrix is one test, not twenty. Deferring `rle` removed the one encoding whose ROI access needed bespoke indexing. |
| **`/meta` JSON adds a parse to every open** | Measured: sub-millisecond for typical documents; a 500-class inline label set is ~120 KB of JSON, ~15 KB compressed. Metadata-only reads (`Dataset` indexing) parse only `/meta`, which is *faster* than 0.x's attribute-by-attribute reconstruction. |
| **Multiple grids complicate every consumer** | Real complexity, but it is the domain's complexity: PET/CT genuinely have different lattices. Consumers that only want one grid read `grids["ref"]` and are exactly as complex as before. |
| **Copy-on-write amend rewrites whole files** | A 2 GB file takes ~2 s to rewrite on SSD. The alternative — HDF5 in-place `del` — leaks space monotonically and fragments the chunk index. Attribute-only edits stay in place. |
| **Subject-scoped samples make files larger** | The real cost of §4.0: a file now holds every timepoint, so it is N× a study-scoped file and amend costs scale with it. Adding a follow-up visit is a rewrite, not an append. Mitigated by the curator's freedom to emit one sample per timepoint when a series is long, and by the fact that annotation edits — the frequent operation — touch small objects while images dominate the bytes. |
| **Longitudinal correctness is now the format's problem** | Stable instance ids and honest coverage across visits are things a writer must get right, and a validator can only warn (W909, W911). The format makes the correct thing expressible and the incorrect thing detectable; it cannot make it automatic. |
| **Total rewrite of a working library** | The 0.x test suite (18 files, ~2 900 lines) does not transfer directly, but the *scenarios* do. Phase 0 of the implementation plan converts them into spec conformance tests before any 1.0 code is written. |

---

## 7. Non-goals

- **Not a PACS or an archival format.** DICOM is the archive. MEDH5 is the training-time
  representation, with lossless round-trips to DICOM/NIfTI where the source permits.
- **Not a whole-slide-imaging format.** The `multiscale` + `channel` model can hold WSI tiles, but
  gigapixel pathology is better served by DICOM-WSI or OME-Zarr; MEDH5 will not add tile-server
  semantics.
- **Not a distributed/cloud object store layout.** Single files on a POSIX filesystem. Cloud users
  shard with `collection` files or an object-store cache in front.
- **Not a labelling tool or a model zoo.** The format describes ground truth; producing it is
  someone else's job.
- **Not multi-writer.** HDF5 cannot do it; pretending otherwise would be a correctness lie.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Encoding auto-selection picks badly on unusual data | Medium | Selection is measured, overridable (`encoding=` explicit), and reversible — transcoding is lossless. `medh5 seg convert` re-encodes in place. W908 warns when `layers` is far from the colouring optimum. |
| The spec is large enough that implementations diverge | Medium | Conformance profiles + a published conformance suite of golden files with expected validator output. A file that passes the suite interoperates. |
| Users put the wrong thing in `annotated_class_ids` | High | It is **required**, so the writer must make a decision; converters set it explicitly from source semantics; W904 warns on the dangerous combination; documentation leads with the failure mode. |
| Blosc2 dependency limits who can read files | Low | `portable` profile; `medh5 recompress --profile portable`; the spec requires no codec. |
| `float16` displacement fields lose accuracy | Low | Permitted, not required; ≈ 5e-4 relative error, two orders below registration accuracy; `float32` remains available. |
| Migration from 0.x loses information | Medium | Mapping is mechanical (Appendix B) except for encoding choice and half-voxel box conversion, both of which `medh5 migrate` reports per file. 0.x `extra` is preserved verbatim under `/meta → extra`. |
| Long series produce unmanageably large samples | Medium | The grouping granularity is the curator's; `medh5 convert` takes an explicit `--group-by study\|subject`, and the migration and converter docs lead with the trade-off rather than burying it. |
| Instance ids reused inconsistently across timepoints, silently corrupting tracking | Medium | `instance_id` semantics are normative and sample-scoped; W909 fires when one id carries two class ids; converters that cannot establish correspondence **must** mint fresh ids rather than guess, and say so in the migration report. |
| Scope creep during implementation | High | Phased plan with a hard `core` + `seg` milestone before `det`/`reg` work starts; profiles make partial implementations legitimate rather than broken. |

---

## 9. Decisions taken

The eight questions raised for review are resolved. Each is recorded with what was decided, the
reasoning, and where the decision now lives normatively.

| # | Question | Decision | Rationale | Lands in |
|---|---|---|---|---|
| 1 | Class id width | **Keep `uint16`** for 1.0 | Class-id width sets the dtype of every `labelmap` and `layers` volume — 1 vs 2 bytes per voxel per layer on the hottest path. No surveyed clinical vocabulary approaches 65 534. Instance ids stay `uint32`/`uint64`, so the cap never limits object count. | Spec §5.3; `wide_labels` reserved in §16 |
| 2 | Collections in 1.0 or 1.1 | **Ship in 1.0** | Deferring would leave the small-sample regime — 2D radiographs, patches, cell crops — without an answer for a whole release, and the design is already load-bearing: every rule is written against "the sample root", so collections cost implementation, not schema. | Spec §2.2; Phase 5 |
| 3 | `rle` normative or interop-only | **Defer; reserve the name** | Chunked compression already leaves unwritten chunks unallocated, so RLE's saving over `instances` is small, while it costs O(1) ROI access and a second sparse code path to write, test and transcode. COCO and DICOM-SEG runs are decoded by converters; no `.medh5` stores runs. | Removed from spec §7; reserved in §16 |
| 4 | Inline label-set threshold | **4096 classes is enough** | A 500-class set is ~15 KB compressed per file; 4096 covers every vocabulary that should be inline, and beyond it the `ref` form is the right answer anyway. | Spec §5.1 (unchanged) |
| 5 | Ontology bindings | **Stay recommended** (`SHOULD`) | Requiring `codes` would bar legitimate ad-hoc and research datasets from the `seg` profile without improving the data — vocabulary discipline is a cohort-level policy, enforceable by `medh5 dataset check`, not a file-level gate. | Spec §5.2 (unchanged) |
| 6 | Default codec profile | **`balanced`** | Writes are one-time, reads are not; `balanced` costs little on write and is a defensible default for files whose lifetime is unknown. Users who know they are writing for immediate training pass `training`; `recompress` moves files between profiles without invalidating digests. | Spec §14.2 (unchanged) |
| 7 | Converter grouping default | **Subject, falling back to study with a loud warning** | Subject grouping is the point of the redesign (§4.0). But de-identification frequently destroys cross-study patient identity, and guessing it from dates, filenames or accession numbers would fabricate the very linkage the format is meant to make trustworthy. Fall back, warn, and record it in the report. | Spec Appendix B; plan §5, §6 |
| 8 | Is `timepoints` required always | **Required, minimum one entry** | One code path. A grid can always resolve its visit, readers never branch on presence, and the cross-sectional cost is a single two-field object. The alternative reintroduces an optional-ness check in every consumer. | Spec §3.7 rule 1 (unchanged) |

Four of the eight — 4, 5, 6 and 8 — confirm what the draft already specified; those sections are
unchanged. Decisions 1 and 2 removed hedging. Decisions 3 and 7 changed the specification: `rle` is
gone from §7 and reserved in §16, and converter grouping became a normative `SHOULD`/`MUST` pair in
Appendix B.

### What remains open

Nothing blocking. Two things are deliberately left to be answered by implementation rather than by
review:

- **Whether `layers` auto-selection needs a better heuristic than greedy colouring.** Greedy is
  within a small factor of optimal on the anatomy tested, but W908 exists precisely so that real
  cohorts can tell us if it is not. Phase 2 collects the distribution.
- **Whether the `training` codec profile should become the default once real dataloader numbers
  exist.** Decision 6 picks `balanced` on reasoning; Phase 6 produces the measurement that would
  justify revisiting it.
