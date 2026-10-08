# Design rationale

Why format 1.0 is shaped the way it is: what 0.x could not express, the
alternatives that were weighed for each load-bearing decision, the costs that
were accepted, and what the format deliberately does not try to be.

This page is explanation, not specification. Where it and
[the specification](../spec/medh5-1.0.md) disagree, the specification is right.

## What 0.x could not express

medh5 0.x was a competent single-sample container — images, boolean masks,
integer boxes, one scalar label, atomic writes, a checksum — and well engineered
inside its model. The model was the problem: one task (binary-mask
segmentation, with a label bolted on) at one scale (tens of classes, one grid,
one annotator, one moment in time).

| Limit | What it cost |
|---|---|
| **One boolean dataset per class** | A 200-class annotation *was* 200 volumes. An all-class 64³ patch read cost 117 ms — 200 chunk lookups and decompressions — which caps one dataloader worker near 8 patches/s on labels alone. |
| **One shape for every image** | PET at 4 mm and CT at 0.7 mm had to be resampled at ingest: a lossy, irreversible decision taken before anyone knew what the model needed, with no record that it happened. |
| **Optional, under-specified geometry** | Nothing said whether an integer index is a voxel centre or a corner, or which axis order `direction` follows. Every consumer guessed, and the guesses differed. |
| **Ground truth as a scalar plus boxes** | No multi-label or hierarchical classification, no oriented boxes, keypoints, contours or meshes, integer boxes that could not survive a resample, and no registration at all. |
| **No notion of time** | A subject scanned three times was three unrelated files. A response label had nothing to bind to, lesion correspondence was lost, the baseline→follow-up transform had no owner, and a split by file could put one patient's baseline in train and follow-up in test. |
| **Provenance as a JSON blob** | `extra["review"]` recorded *that* a review happened, not what produced the data being reviewed — so "model pre-annotates, human corrects, second human approves" had no representation. |
| **A monolithic checksum** | Adding one annotation rehashed the whole file, and a mismatch said "something changed" and nothing more. |
| **A sampler that loaded whole masks** | `np.argwhere` over the full mask, cached per file and class: 9.2 ms and 0.8 MiB per draw at 160³, and tens of gigabytes per worker at 512³ with 200 classes. |
| **Denormalised flags** | `has_seg`, `seg_names`, `shape` and friends duplicated facts derivable from the groups, and the reader carried five checks for the ways they could disagree. |

## Principles

| # | Principle | Where it lands |
|---|---|---|
| 0 | **The file's unit is the subject.** Longitudinal relationships and split safety are structural, not conventional. | §2.2, §3.7 |
| 1 | **One fact, one home.** Per-object facts in HDF5 attributes on the object they describe; documents in `/meta`. No mirrors, no derived flags. | §2.4, §2.5 |
| 2 | **Storage layout is not API.** `contains(class, voxel)` is the contract; encodings are swappable and chosen by measurement. | §7.6 |
| 3 | **Geometry is mandatory and unambiguous.** One affine formula, one voxel-centre convention, one box-edge convention, one transform direction. | §3.3, §8.1, §10.2 |
| 4 | **Absence must be distinguishable from ignorance.** Coverage, ignore regions and explicit negatives. | §6.2, §7.7, §11.3 |
| 5 | **Everything is attributable.** Every object links to the activity that produced it. | §11.1 |
| 6 | **Self-describing without this software.** `h5ls -r` and `h5dump -A` show the structure; the `portable` profile needs no plugin. | §1.1, §2.5, §14.2 |
| 7 | **Derived data is cheap and self-invalidating.** An index carries the digest of its source and is ignored when stale. | §13.3, §14.3 |
| 8 | **Measure, then choose.** Encoding, chunk shape and codec are decided from measured properties of the data. | §7.6, §14.1, §14.2 |

## The decisions, and what was weighed

### A sample is one subject at one or more timepoints

| Option | Verdict |
|---|---|
| One sample = one study (0.x, implicitly) | Rejected. Every longitudinal relationship becomes an out-of-band join, and split safety depends on a field the format does not require. |
| One study per sample, plus a cohort-level "subject graph" sidecar | Rejected. The join moves out of filenames into a second file that can go stale, disagree with the samples, or be lost in transit — and a lesion at two visits still has nowhere to record that it is one lesion. |
| One sample = a subject's *entire* record, required | Rejected. Twenty-year screening series and dense cine studies would make unmanageable files, and copy-on-write amend would become punitive. |
| **One subject at one or more timepoints, the curator's choice** | **Chosen.** The format fixes only what must be fixed — a sample never spans subjects — and leaves the grouping granularity to whoever knows the cohort. |

Time enters through a single attribute: **`timepoint` on the grid**. An image's
timepoint is its grid's, and so is an annotation's. Putting it on the grid
rather than on each image was the real decision: grids are empty,
attribute-only groups, so two visits with identical lattice geometry are simply
two grids, and `timepoint` and `frame_uid` stay single-valued per grid with no
per-image override mechanism.

Several things then fall out for free. Scoping `instance_id` to the sample makes
lesion tracking a join on `instance_id` across timepoints — no track object, no
correspondence table. A change label is an ordinary classification naming the
timepoints it compares. Follow-up registration is the transform the format
already had, now with both endpoints in one file.

### Voxel annotations: several encodings behind one API

| Option | Verdict |
|---|---|
| One boolean volume per class | Rejected: the 0.x cost above, and no representation in which overlapping structures share storage. |
| One-hot `(C, Z, Y, X)` boolean | Rejected. The same cost as per-class, no addressing benefit, worse chunking. |
| Bitmask only | Rejected as the only encoding. Optimal only under heavy overlap; a single-class read decompresses 8 bytes per voxel to answer a one-bit question — 2.79 ms against 0.09 ms for `layers`. |
| Run-length (RLE/COO) only | Rejected, then deferred entirely. Large organs are dense, RLE inflates them and loses O(1) region access, and chunked compression already leaves empty chunks unallocated. The `rle` name is reserved (§16). |
| Layers only | Rejected as the only encoding. Degenerates to one volume per class when the overlap graph is a clique. |
| **Several encodings, a selection rule, lossless transcoding** | **Chosen.** `labelmap`, `layers`, `bitmask` and `instances`, plus `probmap` for soft values. Each regime gets a representation close to its own optimum, and reading code never sees the difference. |

The cost of several encodings is paid once, in the reader; the cost of a single
encoding would be paid forever, by everyone whose data is in the wrong regime.

**Why `layers` is the usual choice.** For *C* classes packed into *L* layers by
greedy colouring of the overlap graph, raw cost is `2L` bytes per voxel for
`uint16` layers against `8·⌈C/64⌉` for bitmask planes, so `layers` wins whenever
`L < 4·⌈C/64⌉`. Real anatomy is sparse in the overlap graph — on the 200-class
phantom, mean degree 3.4 gives `L = 5` — so `layers` is the common case by a wide
margin, and the fastest for both single-class and all-class reads (§7.0).

### Metadata: attributes *and* one JSON document, never both for one fact

Attributes are typed, cheap, visible in `h5dump -A` and attached to what they
describe — right for grid geometry, image semantics and annotation headers. JSON
is right for the label set (a DAG), the provenance graph, quality records and
free-form extras, none of which fit HDF5's attribute model without compound-type
gymnastics nobody would read.

The failure mode of a hybrid is drift, and the rule that prevents it is that
**no fact appears in both**. Rejected: *all attributes* (label sets and
provenance become attribute soup); *all JSON* (`h5ls` shows nothing, and
geometry becomes invisible to every tool but this one); *a compound-dtype label
table* (the hierarchy, ontology codes and per-class properties are ragged).

### Boxes at voxel edges, in continuous index coordinates

The half-voxel question has to be answered once, in the specification, or it is
answered a hundred times, inconsistently, in user code. There is **one**
coordinate space — continuous index coordinates where an integer is a voxel
centre, matching the affine — and box corners sit at voxel **edges**, so the
numpy slice `a:b` is the box `[a − 0.5, b − 0.5]` and a box's extent is exactly
its voxel count (§8.1). This is the ITK/VTK convention.

Rejected: a second "edge space" (two coordinate systems is a trap); integer
corners (the 0.x form, which cannot represent a resampled or rotated box);
coordinates normalised to `[0, 1]` (meaningless once the grid changes, and an
invitation to aspect-ratio bugs). `space = "world"` is also allowed, because a
target defined in millimetres survives resampling and a voxel-space one does not.

### Oriented boxes as rotation matrices

Quaternions were rejected for their double cover — `q` and `−q` are one
rotation, so digests and equality tests differ for identical geometry — and
because they do not generalise to 2-D. Euler angles were rejected because there
are twelve conventions and no way to know which one a file used. Rotation
matrices are unambiguous, dimension-generic and compose directly with the grid
affine; `S²` floats per box is noise next to image data (§8.3).

### One transform direction, mandated

A transform with `from_frame = F` and `to_frame = M` satisfies `x_M = T(x_F)` —
the ITK `TransformPoint` convention (§10.2). There is deliberately **no
attribute to select the opposite convention**: a configurable convention means
every consumer has to handle both, and in practice half of them then handle
neither. Displacement components go on the leading axis, chunked one component
per chunk, so one component or one region reads in isolation.

### Integrity: per-object digests and a Merkle root

Keeping the monolithic hash was rejected, and so was dropping checksums —
clinical data needs them. Per-object digests make verification incremental,
partial and local, and the root doubles as a content address for caching and
deduplication. Digests are computed over **decompressed** content, so moving a
file between codec profiles changes every stored byte and no digest (§13).

### Derived caches carry the digest of their source

The sampling index could have been a sidecar file, a timestamp comparison or an
explicit rebuild step — all three go stale silently. Embedding the source
annotation's digest makes staleness detectable by construction: a reader
compares two strings and falls back to computing from the source. There is no
invalidation protocol to get wrong (§13.3).

### One sample per file, with collections as an escape hatch

Single-sample files are the unit of locking, content addressing, split
membership and `ln -s`-based cohort assembly — and, being subject-scoped, the
unit a leakage-free split assigns. Collections (`.medh5c`) exist for the
small-sample regime — 2-D radiographs, patches, cell crops — where a hundred
thousand files is an operational problem. Their sample roots are structurally
identical to standalone samples, so packing and unpacking are pure copies and
every rule in the specification is written once, against "the sample root"
(§2.2).

### Why not an existing format

| Format | Why not, on its own | What medh5 takes from it |
|---|---|---|
| **DICOM** (SEG, RTSTRUCT, SR, REG) | The interoperability standard and the right archive. But per-slice objects, no chunked random access, no ML-friendly compression, and a full DICOM toolchain to read a segmentation. Nobody trains from raw DICOM. | The frame-of-reference model, modality codes, acquisition keywords, de-identification profiles, the RTSTRUCT contour model |
| **NIfTI** | One array and a 4×4 affine: no multi-label, no provenance, no boxes, no label names. A dataset becomes a directory convention with meaning in its filenames. | The affine-is-the-truth discipline |
| **OME-NGFF / OME-Zarr** | An excellent multiscale and chunking model, and cloud-native. But many small objects suit local per-sample training corpora poorly, and the annotation model is thin for clinical ground truth: no coverage semantics, provenance or registration. | The multiscale layout and the axes / axis-kind model |
| **nnU-Net raw layout** | A directory convention, not a format: no geometry validation, a single labelmap, no overlap, no provenance. | `dataset.json` channel and label conventions, for interoperability |
| **MONAI / TorchIO datasets** | Loader abstractions over other formats, with no storage semantics of their own. | The shape of a sample dictionary |

The gap 1.0 fills: **one local, chunk-random-access file per subject that
carries every task's ground truth, plus what is needed to train on it
honestly** — coverage, provenance, geometry and integrity.

## Choices that were close

| Question | Decision | Why |
|---|---|---|
| Class-id width | **`uint16`** | The class-id width sets the dtype of every `labelmap` and `layers` volume — 1 or 2 bytes per voxel per layer on the hottest read path — and no clinical vocabulary surveyed approaches 65 534 classes. Instance ids are separately `uint32`/`uint64`, so the cap never limits object count. `wide_labels` is reserved for the day it does (§5.3, §16). |
| Collections in 1.0 or later | **In 1.0** | Every rule is already written against "the sample root", so collections cost implementation rather than schema; deferring them would have left the small-sample regime without an answer. |
| Run-length encoding | **Deferred, name reserved** | Its saving over `instances` is small once empty chunks are unallocated, and it costs O(1) region access and a second sparse code path. Converters decode COCO and DICOM SEG runs on the way in (§7.0, §16). |
| Inline label-set limit | **Inline required up to 4096 classes** | A 500-class set is roughly 120 KB of JSON in `/meta` — affordable for a file that is otherwise images; beyond 4096 classes the `ref` form is the right answer anyway (§5.1). |
| Ontology codes | **Recommended, not required** | Requiring them would bar legitimate research vocabularies from `seg` without improving the data. Vocabulary discipline is a cohort policy; `W912` reports what is missing (§5.2). |
| Default codec | **`balanced`** | Writes happen once and reads do not; `balanced` costs little to write and is defensible for a file of unknown lifetime. `training` is one `recompress` away, and moving between profiles invalidates no digest (§14.2). |
| Converter grouping | **By subject, falling back to study — loudly** | Subject grouping is the point of the model. But de-identification often destroys cross-study identity, and guessing it from dates, filenames or accession numbers would fabricate exactly the linkage the format exists to make trustworthy (Appendix B). |
| Is `timepoints` always required? | **Yes, at least one** | One code path: every grid can resolve its visit and no reader branches on presence. A cross-sectional sample pays one two-field object (§3.7). |

## The costs accepted

| Cost | Why it was worth paying |
|---|---|
| **A larger specification** than 0.x | Conformance profiles let a segmentation-only implementation implement `core` and `seg` and ignore the rest — and `core` is smaller than 0.x's implicit schema, because the derived flags are gone. |
| **Several voxel encodings to implement and test** | Bounded: each is a small `contains` / `dense` / `instances` implementation, and the transcoding matrix is covered by property-based tests over every encoding rather than by twenty hand-written cases. |
| **A JSON parse on every open** | A metadata-only read measures 0.19 ms — faster than 0.x's attribute-by-attribute reconstruction ([the numbers](../guides/performance.md#the-numbers)). |
| **Multiple grids complicate every consumer** | Real complexity, but it is the domain's: PET and CT genuinely have different lattices. A consumer that wants one grid reads the reference grid and is no more complex than before. |
| **Copy-on-write amend rewrites the whole file** | The alternative, in-place deletion, leaks space monotonically and fragments the chunk index, because HDF5 does not reclaim it (§14.4). |
| **Subject-scoped files are larger** | A file holds every visit, so amend costs scale with the record and a new visit is a rewrite, not an append. Mitigated by the curator's freedom to emit one sample per timepoint for a long series, and by the fact that annotation edits — the frequent operation — touch small objects. |
| **Longitudinal correctness becomes the format's problem** | Stable instance ids and honest coverage across visits are the writer's to get right. A validator sees only their symptoms — one `instance_id` carrying two classes (`W909`), partial coverage with no ignore region (`W904`) — so the format makes the correct thing expressible and some of the incorrect things detectable. It cannot make it automatic. |

## One engine, three frontends (2.0)

Through 1.x the format was implemented in Python, over `h5py`. That served
Python users and nobody else: a Rust or C++ pipeline, a viewer, or a cluster
node without a Python environment had the specification and the conformance
corpus, but no implementation to call. 2.0 moved the implementation, not the
format.

| Decision | Why |
|---|---|
| **One engine, in Rust** | The canonical implementation is a library every frontend calls, not a Python package other languages re-implement. Rust gives a C-compatible native library with no runtime to ship, and the memory safety a parser of untrusted files should have. |
| **Three frontends over it** | The Rust crate, the Python package (through PyO3) and the native `medh5` binary run the same code, so there is one behaviour to specify, test and keep conformant --- the corpus runs through both the Python and the native command line. |
| **Python keeps what is Python's** | NumPy at the API, PyTorch and MONAI datasets, and the converters, whose job is to call nibabel, pydicom and highdicom. Moving those into Rust would re-implement libraries the ecosystem already maintains. |
| **HDF5 compiled in, statically** | A wheel, a crate and a binary that need nothing installed, and one HDF5 and one Blosc2 everywhere --- so a chunk one frontend writes is the chunk the others read. |
| **Reads below the filter pipeline** | HDF5 decompresses every chunk a window touches, whole, under its global lock; a Blosc2 chunk is a grid of separately compressed blocks. The engine reads the stored chunk and decompresses only the blocks the window covers, outside the lock --- a random 64³ label patch about three times faster, and threads that decompress at once. The bytes are HDF5's or the read goes through HDF5. |
| **The format stayed 1.0** | Nothing in the file needed to change: a new major format would have bought nothing and cost every 1.x reader. The package is 2.0 because its Python API changed at the HDF5 boundary --- it no longer hands out `h5py` objects --- and semantic versioning says so. |

Re-implementing the format was also a test of the specification: three
clauses turned out to name a Python function where they meant bytes, and are
now defined in HDF5 and JSON terms, and one promised a reproducibility only
NumPy's generator could give (Appendix C.1). A specification only one
implementation can satisfy is a description of that implementation.

## Non-goals

- **Not a PACS or an archive format.** DICOM is the archive. medh5 is the
  training-time representation, with lossless round trips to DICOM and NIfTI
  where the source permits.
- **Not a whole-slide imaging format.** The multiscale and channel model can
  hold WSI tiles, but gigapixel pathology is better served by DICOM-WSI or
  OME-Zarr, and medh5 will not add tile-server semantics.
- **Not a cloud object-store layout.** Single files on a POSIX filesystem. Cloud
  users shard with collections, or put a cache in front of an object store.
- **Not a labelling tool or a model zoo.** The format describes ground truth;
  producing it is somebody else's job.
- **Not multi-writer.** HDF5 cannot do it, and pretending otherwise would be a
  correctness lie (§14.4).

## Where the numbers come from

The measurements behind these decisions are in the specification next to the
clause each justifies — encodings in §7.0, codec profiles in §14.2, the sampling
index in §14.3, access patterns in §14.5 — and the scripts that produced them,
with their recorded results, are on [Runnable examples](../examples/index.md).
The package's own performance targets, and `medh5 bench` to reproduce them, are
in [Tune performance](../guides/performance.md).

## Related

- **[The data model](data-model.md)** — the model these decisions produced.
- **[What the converters refuse, and why](refusals.md)** — principle 3 applied to import and export.
- **[Specification](../spec/medh5-1.0.md)** — the normative statement.
- **[Changelog](../changelog.md)** — what changed in each release since 1.0.
