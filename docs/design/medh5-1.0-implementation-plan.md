# medh5 1.0 — Implementation Plan

**Companion documents:** [Specification](../spec/medh5-1.0.md) · [Design proposal](medh5-1.0-proposal.md)

**Status: phases 0–4 are complete.** The core container, the label space, all five voxel encodings,
every geometric annotation, classification (including change labels), registration, the validator and
the conformance corpus are implemented and gated on `ruff`, `mypy --strict` and ≥ 90 % coverage.
**Every diagnostic code in §15.2 now has a conformance case.** Phase 5 (curation, longitudinal joins
and collections) is next.

---

## 1. Package layout

Sub-packages map 1:1 onto specification sections, so a spec change has one obvious home.

Sub-packages marked ✅ are implemented.

```
medh5/
├── __init__.py         ✅ public surface: open, create, amend, Sample, exceptions
├── __about__.py        ✅ version, kept importable without the package
├── _hdf5.py            ✅ attribute codecs, identifier rules, atomic create, CoW amend
├── errors.py           ✅ exception hierarchy + the E/W code table (§15.2)
├── document.py         ✅ the `/meta` sample document and its JSON Schema
├── image.py            ✅ lazy image access, rescale, pyramid levels
├── schemas/            ✅ the packaged JSON Schema (a test asserts it matches `schemas/`)
│
├── geometry/           ✅ §3
│   ├── grid.py         ✅ Grid: shape, axes, spacing, origin, direction, affine, frame, timepoint
│   ├── affine.py       ✅ index↔world, box↔slice, half-voxel rules, orthonormality checks
│   └── multiscale.py   ✅ §4.3 pyramid geometry derivation and validation
│
├── labels/             ✅ §5
│   ├── labelset.py     ✅ LabelSet, LabelClass, DAG closure, canonical digest, inline/ref
│   ├── registry.py     ✅ bundled vocabularies, loadable by name
│   └── vocabularies/   ✅ binary-foreground, brats-subregions, amos22-organs (JSON)
│
├── annotations/           §6–§9
│   ├── base.py         ✅ Annotation ABC: the contains/dense/labelmap/instances contract
│   ├── voxel/          ✅
│   │   ├── payload.py  ✅ mask helpers shared by the voxel encoders
│   │   ├── labelmap.py ✅ layers.py ✅ bitmask.py ✅ instances.py ✅ probmap.py ✅ mask.py ✅
│   │   ├── select.py   ✅ §7.6 overlap graph, greedy colouring, cost model, auto-selection
│   │   └── transcode.py ✅ lossless conversion between any pair of encodings
│   ├── payload.py      ✅ the arrays-in/arrays-out intermediate every encoder produces
│   ├── geometric.py    ✅ boxes, obb, keypoints, points, contours, mesh (§8)
│   └── classification.py ✅ labels, ordinal schemes, change labels (§9)
│
├── transforms/         ✅ §10
│   ├── base.py         ✅ Transform ABC, the direction convention, header, registry
│   ├── affine.py       ✅ identity and affine, with analytic inverse and Jacobian
│   ├── displacement.py ✅ dense fields, component-major, Jacobian and folding
│   ├── bspline.py      ✅ free-form deformation, basis written out rather than imported
│   ├── composite.py    ✅ ordered chains with frame-chain checking
│   ├── resolve.py      ✅ frame-graph search, inverses, multi-hop chains
│   └── apply.py        ✅ interpolation, Jacobian determinant, TRE
│
├── curation/           ✅ §11–§12
│   ├── provenance.py   ✅ agents/activities graph, reference resolution
│   ├── quality.py      ✅ quality records, agreement metrics
│   ├── identity.py     ✅ identity, cohort, split claims, de-identification
│   └── timeline.py     ✅ §3.7 timepoint declaration, grid binding, inheritance
│
├── integrity/          ✅ §13
│   ├── digest.py       ✅ canonical byte stream, streaming per-object digests, content_id
│   └── verify.py       ✅ partial + full verification, index currency
│
├── storage/            ✅ §14
│   ├── chunking.py     ✅ L3-aware optimizer, decomposed into spatial + composition
│   ├── codecs.py       ✅ four profiles, filter-pipeline introspection
│   └── index.py        ✅ sampling index build/read, bounded-memory subsampling, occupancy
│
├── sample.py           ✅ Sample / SampleWriter / open / create / amend — the assembly point
├── validate/           ✅ §15 rule engine, levels, profiles, report model
├── conformance/        ✅ the golden corpus and its expected-output manifest
├── cli/                ✅ info, tree, validate, verify, timeline, track, labels, seg, index
├── collection.py          §2.2 collection files, pack/unpack
├── dataset/               manifests, splits, streaming statistics
├── io/                    nifti, dicom, dicom_seg, rtstruct, nnunetv2, coco, learn2reg
├── torch/                 datasets, samplers, collate
├── monai.py               MetaTensor adapter (affine-preserving)
└── legacy/             ✅ the whole 0.x implementation, moved here intact
```

**Two departures from the plan as written, both deliberate.**

*The 0.x tree moved rather than being deleted.* `medh5/legacy/` holds the 0.x implementation verbatim,
importable and still covered by its own tests, because `medh5 migrate` (phase 7) has to read 0.x files
and because a rewrite that breaks the shipped package on day one is a rewrite nobody can bisect.
Phase 8's "delete 0.x" becomes `rm -rf medh5/legacy` plus one `pyproject.toml` edit. It is excluded
from `mypy --strict`'s `type-arg` check, scoped to that package with a comment saying why.

*Three modules the plan did not name.* `document.py` (the `/meta` document is a distinct concern from
the objects it describes), `image.py` (lazy reads, rescale and pyramid levels are more than
`sample.py` should carry), and `conformance/` (the corpus is a shipped artifact, not a test fixture:
third-party implementations run it).

**Reused from 0.x with modification:** `chunks.py` (generalised to N-D and non-spatial axes),
atomic-write machinery in `core.py`, `_shared.py` handle sharing, `dataset/split.py` grouping logic,
`stats.py` Welford aggregation, the CLI dispatch pattern.

**Deleted:** `meta.py` (replaced by `geometry/` + `/meta` document), `review.py` (replaced by
`curation/provenance.py`), the `MEDH5File` god-class, `_warn_malformed_extra` (replaced by JSON
Schema validation), all `has_*`/`*_names` denormalised flags.

---

## 2. Public API

### 2.1 Reading

```python
import medh5

with medh5.open("case_0001.medh5") as s:            # -> Sample
    s.identity.subject_id                            # "BRATS-GLI-01234"
    s.profiles                                       # {"core", "seg", "det", "curation", "longitudinal"}

    # --- timepoints: the sample is a subject, not a study ---------------
    len(s.timepoints)                                # 2
    s.timepoints["tp1"].days_from_baseline           # 92
    s.timepoints[0].label                            # "baseline"  (index or id)
    s.at("tp1").images                               # a timepoint-scoped view of the whole sample
    s.at("tp1").annotations["organs_tp1"]            # same API, filtered
    s.images.by_timepoint("tp0")                     # ("CT_tp0", "PET_tp0")

    # --- geometry -------------------------------------------------------
    g = s.grids["ct"]
    g.affine                    # (4,4) index -> world, voxel-centre convention
    g.world_to_index(pts)       # ndarray (N,3) -> (N,3)
    g.frame_uid

    # --- images ---------------------------------------------------------
    ct = s.images["CT"]                              # lazy
    hu = ct.read(roi=np.s_[64:128, 96:160, 96:160], physical=True)   # applies rescale
    ct.value_units                                   # "HU"
    ct.level(2).read()                               # multiscale, if present

    # --- one API over every voxel encoding ------------------------------
    seg = s.annotations["organs"]                    # kind is an implementation detail
    seg.kind                                         # "layers"
    seg.classes                                      # (LabelClass, ...) resolved names
    seg.dense(["liver", "spleen"], roi=roi)          # (2, z, y, x) bool
    seg.labelmap(roi=roi, priority=["lesion", ...])  # (z, y, x) uint16, ties broken explicitly
    seg.contains("liver", (91, 120, 133))            # bool
    seg.instances()                                  # iterate objects where meaningful
    seg.annotated_classes                            # coverage contract, NOT seg.classes
    seg.is_annotated("kidney_left")                  # False -> absence means "unknown"

    # --- other tasks ----------------------------------------------------
    s.annotations.by_task("detection")               # -> tuple[Annotation, ...]
    boxes = s.annotations["lesions"]
    boxes.as_world()                                 # (N, 3, 2) mm
    boxes.as_slices()                                # [(slice, slice, slice), ...]

    s.annotations["staging"].labels                  # {"T3": 1.0, "N1": 1.0}
    s.annotations["staging"].scheme("BI-RADS")       # "4b"

    T = s.transforms["ct_to_mr"]
    T.transform_points(pts_world)                    # fixed -> moving, ITK convention
    T.jacobian_determinant(roi=roi)                  # for displacement fields

    # --- curation -------------------------------------------------------
    seg.provenance.agent.name                        # "pseudonym:RAD-07"
    seg.quality.status                               # "approved"
    seg.quality.agreement[0].value                   # 0.913

    # --- longitudinal ---------------------------------------------------
    s.track(9)                                       # class -> {instance_id: {tp: Instance}}
    s.track(9)[1].timepoints                         # ("tp0", "tp1")  -- this lesion persisted
    s.track(9)[3].resolved_after                     # "tp0"           -- absent at tp1, and covered
    s.changes()                                      # annotations whose `timepoints` span >1 visit
    s.transform_between("tp0", "tp1")                # resolves via frames, not by name

    # --- sampling -------------------------------------------------------
    idx = s.index["organs"]
    idx.voxel_counts["liver"]                        # 1_284_302
    centers = idx.sample_foreground("liver", n=8, rng=rng)   # (8, 3) int32, O(1) in volume
```

### 2.2 Writing

A builder, not a 25-parameter function. Every `add_*` validates immediately; `commit()` is atomic.

```python
with medh5.create("case_0001.medh5", codec="balanced") as w:
    w.identity(sample_id="…", subject_id="…", sex="F", bodypart="abdomen")
    w.cohort(dataset_id="abdomen-v3", site_id="site-B")

    w.add_timepoint("tp0", label="baseline", date="2026-02-01", days_from_baseline=0,
                    study_uid="pseudo:…100", series_uids={"CT_tp0": "pseudo:…1"})
    w.add_timepoint("tp1", label="follow_up_3mo", date="2026-05-04", days_from_baseline=92,
                    study_uid="pseudo:…101")

    act = w.activity("import", agent=w.software("medh5", medh5.__version__),
                     tool="medh5 convert from-dicom", params={...})

    w.add_grid("ct_tp0", shape=(160,)*3, spacing=(1.5, 0.8, 0.8), origin=(-190.2, -170.0, -170.0),
               direction=np.eye(3), coord_system="LPS", timepoint="tp0",
               frame_uid="pseudo:1.2.…100", patch_hint=(96, 96, 96))
    w.add_image("CT_tp0", ct_int16, grid="ct_tp0", modality="CT",
                value_type="quantitative", value_units="HU",
                rescale_slope=1.0, rescale_intercept=0.0, prov=act)
    # follow-up: its own grid, its own extent, its own frame -- nothing is resampled
    w.add_grid("ct_tp1", shape=(152, 160, 160), spacing=(1.5, 0.8, 0.8), origin=(-184.2, -170.0, -170.0),
               direction=np.eye(3), coord_system="LPS", timepoint="tp1",
               frame_uid="pseudo:1.2.…101")

    w.label_set(medh5.labels.registry.load("totalsegmentator-v2"))

    seg_act = w.activity("annotate", agent=w.person("pseudonym:RAD-07", role="annotator"),
                         tool="3D Slicer 5.6.2")
    w.add_segmentation(
        "organs",
        masks={"liver": liver, "spleen": spleen, ...},   # or labelmap=…, or instances=…
        encoding="auto",                                  # §7.7 measures and chooses
        annotated_classes="all_given",                    # or an explicit list
        closure="explicit", prov=seg_act,
        quality={"status": "approved", "reviewed_by": ["r2"]},
    )
    w.add_boxes("lesions_tp0", boxes=boxes_f32, class_ids=[...], instance_ids=[1, 3, 6],
                space="index", grid="ct_tp0", scores=scores, prov=seg_act)
    w.add_boxes("lesions_tp1", boxes=fu_boxes, class_ids=[...], instance_ids=[1, 8],
                space="index", grid="ct_tp1", prov=fu_act)   # id 1 persists, 3 and 6 resolved, 8 is new
    w.add_classification("staging_tp0", labels={"T3": 1.0, "N1": 1.0},
                         scope="timepoint", scope_ids=["tp0"], multilabel=True, closure="implicit",
                         annotated_classes=STAGING_CLASSES, prov=seg_act)
    w.add_classification("response", labels={"partial_response": 1.0},
                         scope="sample", timepoints=["tp0", "tp1"], multilabel=False,
                         closure="explicit", annotated_classes=RECIST_CLASSES, prov=review_act)
    w.add_transform("tp0_to_tp1", kind="affine", matrix=M,
                    from_frame="pseudo:1.2.…100", to_frame="pseudo:1.2.…101", prov=reg_act)

    w.build_index(["organs_tp0", "organs_tp1"], max_coords=4096)
    w.commit()                     # validate -> digests -> content_id -> fsync -> rename
```

Amending is copy-on-write by default and preserves unknown objects:

```python
with medh5.amend("case_0001.medh5") as w:              # CoW; attrs_only=True stays in place
    w.add_segmentation("organs_rater2", masks=..., prov=w.activity("annotate", agent=r2))
    w.set_quality("organs", status="reviewed", reviewed_by=["r2"])
```

### 2.3 Loader integration

```python
from medh5.torch import PatchDataset, VolumeDataset, PatchSampler, collate

sampler = PatchSampler(
    patch_size=(96, 96, 96),
    strategy="balanced", foreground_prob=0.6,
    foreground_classes=["pancreas", "tumor"],     # uses index/, never argwhere
    class_weights="inverse_frequency",            # from index voxel_counts
)
ds = PatchDataset(paths, sampler=sampler,
                  images=["CT"], annotations={"organs": ["liver", "pancreas", "tumor"]},
                  label_format="onehot",          # "onehot" | "labelmap" | "instances"
                  physical=True)                  # rescale to HU
loader = DataLoader(ds, batch_size=2, num_workers=8,
                    worker_init_fn=medh5.torch.worker_init_fn, collate_fn=collate)
```

Longitudinal loading uses a paired sampler — the change-detection and registration equivalent of a
patch sampler:

```python
from medh5.torch import PairedPatchDataset, TimepointPairSampler

pairs = TimepointPairSampler(mode="consecutive")      # or "baseline_vs_all", "all_pairs"
ds = PairedPatchDataset(paths, sampler=sampler, pair_sampler=pairs,
                        images=["CT"], annotations={"organs": [...]},
                        align="transform",            # warp the follow-up patch through tp0→tp1
                        label="response")             # the change annotation spanning the pair
```

`align="transform"` resolves the transform between the two frames and samples corresponding patches;
`align="none"` returns unregistered pairs for models that learn the alignment. A sample with one
timepoint yields no pairs and is skipped, with a count reported rather than a silent drop.

MONAI: `medh5.monai.to_metatensor(sample, "CT")` returns a `MetaTensor` with the correct affine, so
`Spacingd`, `Orientationd` and `SaveImaged` work unmodified.
nnU-Net v2: `medh5 convert to-nnunet` and a `medh5`-backed `nnUNetDataset` shim.

---

## 3. Phases

Each phase ends with green CI, updated docs, and a tagged pre-release. Estimates assume one
experienced developer; the phases are largely independent after Phase 2.

| Phase | Scope | Exit criteria | Est. |
|---|---|---|---|
| **0 · Conformance harness** ✅ | Golden-file corpus, validator report model, spec-cross-referenced test IDs; port the 0.x test *scenarios* to spec assertions | **Done.** 75-case corpus with per-code expectations; `medh5 conformance run` green; a test asserts the §15.2 table and the code registry are identical | 1 w |
| **1 · Core container** ✅ | `_hdf5`, `geometry/` incl. §3.7 timepoints and grid binding, `images`, `/meta` + JSON Schema, atomic create, CoW amend, `integrity/` | **Done.** `core` profile validates; geometry round-trips; timepoint inheritance resolves; multiscale pyramids write and validate; four spec clauses corrected (Appendix C) | 2.5 w |
| **2 · Label sets + voxel annotations** ✅ | `labels/`, the five voxel encodings, `select.py`, `transcode.py`, coverage semantics, `index/` | **Done.** `seg` profile complete; the transcoding matrix passes `contains()` equality for every ordered pair; encoding auto-selection measured; sampling index is O(1) in volume | 2.5 w |
| **3 · Geometric + classification annotations** ✅ | boxes, obb, keypoints, points, contours, mesh, classification | **Done.** `det` and `cls` profiles complete; box↔slice round-trips over 200 randomised boxes; OBB centre/size/rotation recovered from its corners; two more spec clauses corrected (§1.3 `det`, §9 `class_ids`) | 2 w |
| **4 · Registration** ✅ | affine, displacement, bspline, composite, landmarks, `apply.py`, inter-timepoint transform resolution | **Done.** `reg` profile complete; TRE computed from paired landmark sets; `transform_between` resolves through the frame graph, using inverses where they exist and refusing to invent them where they do not; every §15.2 code now has a corpus case | 2 w |
| **5 · Curation + longitudinal + collections** | provenance graph, quality, identity/cohort/splits, instance tracking joins, change annotations, `.medh5c` pack/unpack (**in 1.0**, not deferred) | `curation` and `longitudinal` profiles complete; tracking join round-trips; W909/W910/W911 fire on crafted inputs; pack/unpack byte-identical on sample subtrees | 2 w |
| **6 · Loaders + performance** | torch datasets/samplers/collate, paired/longitudinal sampling, MONAI adapter, codec profiles, chunking policy, `recompress` | Throughput target met (§4.3); paired sampling correct against a hand-checked fixture; no per-worker memory growth over a 10-epoch soak | 2.5 w |
| **7 · Converters** | NIfTI, DICOM, DICOM-SEG, RTSTRUCT, nnU-Net v2, COCO, `migrate` from 0.x, subject-grouping across studies | Round-trip fidelity tests per converter; `--group-by subject` produces correct multi-timepoint samples from a multi-study DICOM tree; migration report on a real 0.x cohort | 3 w |
| **8 · Release** | Docs, tutorials, conformance suite publication, PyPI 1.0.0 | Spec + suite published; napari-medh5 updated against 1.0 | 1 w |

Critical path: 0 → 1 → 2 → 6. Phases 3, 4, 5 and 7 parallelise after Phase 2. Total ≈ 17.5 weeks:
the longitudinal model costs about half a week in the core and half in curation — buying the
paired-sampling and tracking work that would otherwise land downstream in every consumer — and
deferring `rle` returns half a week in Phase 2.

---

## 4. Testing

### 4.1 Layers

| Layer | Content |
|---|---|
| **Unit** | Per module. Geometry: affine round-trips, orthonormality, half-voxel box conversion at boundaries. Encodings: `contains()` against a brute-force reference. |
| **Property-based** (Hypothesis) | Randomised grids (2D/3D/4D, anisotropic, oblique, negative determinant) and randomised class-overlap graphs. Invariants: (a) `write → read` is identity; (b) `transcode(A→B→A)` is identity for every ordered pair of the five voxel encodings; (c) `box → slices → box` round-trips within a half voxel; (c2) timepoint inheritance agrees with explicit annotation for every object; (c3) an instance id joined across timepoints yields the same object set as a brute-force scan; (d) `T ∘ T⁻¹ ≈ identity` for invertible transforms; (e) `content_id` is stable under recompression and unstable under any content change. |
| **Conformance corpus** | ~40 golden files, each targeting spec clauses, with expected validator output as JSON. Third-party implementations run the same corpus. Includes deliberately-invalid files, one per error code. |
| **Interop round-trips** | NIfTI, DICOM-SEG, RTSTRUCT, nnU-Net v2, COCO. Assert geometry, class identity and voxel equality; assert the failure is *loud* where a source cannot express a MEDH5 feature. |
| **Benchmarks in CI** | §5 of the proposal, run on every PR with a regression threshold (fail at > 20 % slower or > 10 % larger). Committed as `docs/design/benchmarks/`. |
| **Longitudinal fixtures** | Hand-built samples covering: 1 timepoint; 2 timepoints with a persisting, a resolved and a new lesion; a timepoint annotated for fewer classes than another; a frame reused across timepoints (W910); a multi-timepoint sample with no relating transform (W911). Expected validator output committed alongside. |
| **Soak** | 10-epoch dataloader run over ≥ 1 000 files × 8 workers; assert flat RSS, no fd leaks, no handle-cache growth across `fork`. |

### 4.2 Coverage and gates

Keep the 0.x gates: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest --cov-fail-under=90`.
Add: JSON Schema validation of every golden file's `/meta`; a spec-coverage report asserting every
normative **MUST** in the spec maps to at least one test id (`test_spec.py::test_S7_2_layers_unique_class`).

### 4.3 Performance targets

| Metric | Target | Baseline (0.x) |
|---|---|---|
| 64³ patch, 200-class multi-label labels only | ≤ 10 ms | 117 ms |
| Foreground centre sampling | ≤ 1 ms, O(1) memory | 9.2 ms, O(volume) |
| Metadata-only read (`/meta` parse) | ≤ 2 ms | ~1.5 ms (attribute reconstruction) |
| Full `open()` → first patch | ≤ 15 ms | ~120 ms with labels |
| Sustained 96³ multi-modal patch throughput, 8 workers | ≥ 400 patches/s | ~60 patches/s |
| Storage vs 0.x, 200-class cohort | ≤ 0.25× | 1.0× |

---

## 5. CLI

```
medh5 info FILE [--json]                       # summary: grids, images, annotations, coverage, quality
medh5 tree FILE                                # annotated h5ls with spec roles
medh5 validate PATH... [--level structural|semantic|integrity|strict] [--profile P] [--json]
medh5 verify PATH... [--partial OBJ]           # digests / content_id
medh5 fix PATH... [--rebuild-index] [--rewrite-digests]

medh5 timeline FILE [--json]                    # timepoints, per-visit images/annotations, intervals
medh5 track FILE [--class KEY] [--json]        # instance ids joined across timepoints: persisted/resolved/new

medh5 labels show FILE | check PATH... | registry list

medh5 seg convert FILE ANN --to layers|bitmask|instances|labelmap [--dry-run]
medh5 seg stats FILE ANN                       # per-class counts, overlap graph, chosen-encoding cost model

medh5 index build PATH... [--max-coords N] [--occupancy 8]
medh5 recompress PATH... --profile training|balanced|archive|portable

medh5 dataset index ROOT -o manifest.json      # metadata-only scan
medh5 dataset split manifest.json --k-folds 5 --group-by cohort.group_id --stratify-by …
medh5 dataset stats manifest.json --workers 8
medh5 dataset check manifest.json              # vocabulary drift, split claim conflicts, coverage report

medh5 convert from-nifti | from-dicom | from-dicom-seg | from-rtstruct | from-nnunet | from-coco
                 [--group-by subject|study]    # default: subject; falls back to study + warning
medh5 convert to-nifti   | to-dicom-seg | to-nnunet   | to-coco
medh5 migrate PATH... -o OUTDIR [--report report.json]     # 0.x -> 1.0
                 [--group-by subject|study] [--subject-key extra.patient_id]   # default: study

medh5 pack ROOT -o shard.medh5c | medh5 unpack shard.medh5c -o ROOT
medh5 scrub PATH... --profile basic            # de-identification sweep + attestation
medh5 bench PATH                               # reproduce the benchmark table on your hardware
```

Exit codes stay Unix-conventional (0 ok, 1 handled error, 2 usage). `--json` on every inspection
command.

**Grouping behaviour (settled).** `medh5 convert` defaults to `--group-by subject`: it resolves patient
identity across studies and emits one multi-timepoint sample per patient. When identity cannot be
established — the usual cause being a de-identification pass that stripped or randomised
`PatientID` — it falls back to one sample per study, prints a warning naming the affected inputs, and
records the fallback in the conversion report. It never guesses identity from dates, filenames or
accession numbers. `medh5 migrate` inverts the default to `--group-by study`, because a 0.x file has
no reliable subject key of its own; `--subject-key extra.patient_id` opts into subject grouping when
the cohort put one in `extra`.

---

## 6. Migration from 0.x

`medh5 migrate` applies Appendix B of the spec. Four decisions are not mechanical, and the tool
reports each one per file:

1. **Voxel encoding.** 0.x boolean masks are measured and re-encoded per §7.7. The report gives the
   overlap-graph statistics, the chosen encoding and the size delta.
2. **Box corners.** 0.x integer boxes were slice-like `[min, max)`; they convert to
   `lo = min − 0.5`, `hi = max − 0.5`. Reported as a half-voxel shift so it can be audited.
3. **Label set.** 0.x mask names and `bbox_labels` strings become class `key`s with minted ids. If an
   `extra.nnunetv2.labels` mapping exists, those ids are reused. The generated label set is written
   to a sidecar so it can be reviewed, edited and reapplied cohort-wide before conversion.
4. **Sample grouping.** With `--group-by study` (the default for migration) each 0.x file becomes a
   one-timepoint sample declaring `tp0`, and nothing about time is invented. With
   `--group-by subject` the tool reads a subject key — `--subject-key extra.patient_id` — merges a
   patient's files into one multi-timepoint sample, and orders timepoints by file mtime unless a date
   field is supplied. Ordering inferred from mtime is reported per sample, because it is a guess.
   Instance correspondence across merged timepoints is **never** inferred: every 0.x file's objects
   get fresh instance ids, and the report says so, since silently asserting that lesion 2 at baseline
   is lesion 2 at follow-up would fabricate ground truth.

`annotated_class_ids` is set to the migrated mask names — the only defensible inference — and the
report flags every file so the curator can widen or narrow it. `extra` is copied verbatim to
`/meta → extra`, with `extra.review` *additionally* projected into the provenance graph.

Migration is one-way. 0.x readers cannot open 1.0 files, by design: `medh5_version` is a new
attribute and 0.x's `read_meta` raises `MEDH5SchemaError` on the missing `schema_version`, which is
the correct loud failure.

---

## 7. Documentation deliverables

| Document | Audience |
|---|---|
| `docs/spec/medh5-1.0.md` | Implementers of readers/writers in any language |
| `docs/design/*` (this set) | Reviewers and maintainers |
| `schemas/medh5-sample-1.0.schema.json` | Machine validation of `/meta` |
| `docs/getting-started.md` | New users — 20 lines to a written file |
| `docs/guides/segmentation.md`, `detection.md`, `classification.md`, `registration.md` | Task-oriented recipes, one per profile |
| `docs/guides/partial-labels.md` | The coverage contract — the concept most likely to be misused |
| `docs/guides/longitudinal.md` | Timepoints, instance tracking, change labels, paired sampling, and when to split a subject across samples |
| `docs/guides/performance.md` | Chunking, codec profiles, index, dataloader tuning |
| `docs/conformance/` | Golden corpus + expected validator output |
| `docs/migration-0x-to-1.0.md` | Existing users |

---

## 8. Release checklist

- [ ] Conformance corpus passes on Linux and macOS, Python 3.10–3.13
- [ ] Every normative **MUST** has a mapped test id
- [ ] Benchmarks meet §4.3 targets and are recorded in the release notes
- [ ] `mypy --strict`, `ruff`, coverage ≥ 90 % green
- [ ] JSON Schema published and versioned alongside the spec
- [ ] `medh5 migrate` validated on a real 0.x cohort with a reviewed report, both grouping modes
- [ ] napari-medh5 updated and tested against 1.0 (multi-grid + layers rendering)
- [ ] `docs/` rebuilt; 0.x docs archived under `docs/0.x/`
- [ ] `pyproject.toml` → `1.0.0`, `Development Status :: 5 - Production/Stable`
- [ ] Announcement covering: what breaks, what it buys, how to migrate
