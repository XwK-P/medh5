# medh5 2.2.0

A minor release: format 1.2 (one new annotation kind), a native NIfTI
converter, enhanced multi-frame DICOM, free-threaded CPython, new platforms,
resampling as a recorded derivation, and the de-identification and
registration work 2.1.0 left for later. Read [plans/README.md](../README.md)
first (its rules hold here too, with `w22/` branches and `W22-` titles), then
[decisions.md](decisions.md), then your workstream.

- **Baseline.** v2.1.0, `main` after the 2.1.0 release pull request merged.
  This plan was written before 2.1.0 was implemented: each workstream lists the
  2.1.0 pieces it builds on, and checks they are on `main` before starting. If a
  2.1.0 item was deferred, the 2.1.0 release pull request names it; add it here
  as a workstream (W22-00) with its 2.1.0 text, before anything that depends on
  it.
- **Format.** 1.2 adds the `polylines` kind and nothing else (E1); a sample is
  written at 1.2 only when it holds one (E2). 1.0 and 1.1 files are written
  exactly as 2.1.0 writes them.
- **API.** Additive, under 2.1.0's rules. The deprecations 2.1.0 announced stay
  deprecated: their removal is 3.0's (the ledger in plans/README.md).
- **Platforms.** Nothing is dropped. Python 3.10 stays supported (E5), reversing
  the 3.11 floor 2.1.0's D36 had planned for this release.

## Themes

1. **Open curves.** Vessel centrelines, catheter and applicator paths, and
   RTSTRUCT open contours become a first-class kind, readable by 1.1 readers as
   a projection.
2. **One converter, three frontends.** NIfTI import and export move into the
   engine, byte for byte what 2.1.0's Python converter writes; the standalone
   binary converts NIfTI without Python.
3. **More DICOM.** Enhanced and legacy-converted multi-frame CT, MR and PET.
4. **Free-threaded Python, more platforms.** `gil_used = false` after an audit;
   musllinux; macOS x86_64 cross-built before GitHub retires Intel runners.
5. **De-identification that finishes the job.** Collections, and rewriting
   clinical text with spans kept valid.
6. **Registration and clinical import at scale.** ITK's HDF5 transform files,
   B-spline transforms where they can be represented exactly, and importing a
   whole MEDS or FHIR bulk export in one pass.
7. **Resampling, on purpose and on the record.** `medh5 resample` puts an image
   and its annotations onto another grid, through the sample's own transforms,
   and records it as a `resample` activity. Converters and loaders still refuse
   to resample, and now say which command does.

## Workstreams

| Id | Workstream | Size | Pull requests | Builds on (2.1.0) |
|---|---|---|---|---|
| [W22-01](w22-01-format-1.2.md) | Format 1.2: `polylines` | L | 3 | W21-02 (version accessors, `next_minor`), W21-13 PR 3 (RTSTRUCT points) |
| [W22-02](w22-02-native-nifti.md) | A native NIfTI converter | L | 6 | W21-13 PRs 1, 5, 6 (helpers, float64 qform, `--plane-as-volume`, `import-seg-nifti`) |
| [W22-03](w22-03-enhanced-dicom.md) | Enhanced multi-frame DICOM import | L | 3 | W21-13 PRs 1, 5 (`_frames.py`, per-slice rescale) |
| [W22-04](w22-04-free-threading.md) | Free-threading | L | 4 | W21-01 PR 3 (cp314t wheels), W21-15 PR 1 (GIL) |
| [W22-05](w22-05-platforms.md) | Platforms: musllinux, macOS x86_64, `abi3t` | L | 4 | W21-00 PR 7 (`build.yml`) |
| [W22-06](w22-06-fuzzing.md) | Fuzzing the C libraries | M | 1 | W21-01 PR 1 |
| [W22-07](w22-07-deidentification.md) | De-identification: collections, clinical text | L | 2 | W21-10 |
| [W22-08](w22-08-registration.md) | Registration I/O: ITK HDF5 files, B-splines | M | 2 | W21-07 PR 4 |
| [W22-09](w22-09-clinical-cohorts.md) | Clinical import across a cohort | M | 2 | W21-12 |
| [W22-10](w22-10-resample.md) | Resampling as a recorded derivation | L | 2 | `transform_between` and its visit refusal (D28), the inverse of a chain (W21-07), one replace routine (W21-05 PR 1) |

## Order of work

- **First:** W22-02 PR 1 (the parity corpus must be recorded from 2.1.0 before
  any NIfTI code changes).
- **Then, in parallel**, with these edges:

| Before | After | Why |
|---|---|---|
| W22-02 PR 1 | W22-02 PRs 2–6 | parity is measured against 2.1.0's recorded output |
| W22-01 PR 1 | W22-01 PRs 2–3 | the content-aware version |
| W22-01 PR 2 | W22-01 PR 3 | the kind exists before RTSTRUCT writes it |
| W22-04 PRs 1–3 | W22-04 PR 4 | the audit before `gil_used = false` |
| W22-02 PR 4 | W22-10 PR 2 | the NIfTI refusals, `import_seg_nifti`'s among them, move into the engine; PR 2 rewords them where they then live |

### Lanes

| Lane | Sequence |
|---|---|
| A, platform | W22-05 → W22-06 |
| B, format | W22-01 → W22-08 |
| C, converters | W22-02 → W22-03 |
| D, Python | W22-04 |
| E, de-identification and clinical | W22-07 → W22-09 |
| F, derivations | W22-10 |

## Corpus and code-table targets

| | 2.1.0 (planned) | 2.2.0 |
|---|---|---|
| Codes | 100 | 100 (E408's summary names polyline offsets) |
| Format versions written | 1.0, 1.1 | 1.0, 1.1, 1.2 |
| Format cases | 172 | 175: `shapes-polylines` (valid, 1.2), `E408-polyline-offsets`, `E404-polyline-one-vertex` |
| The higher-minor case | "1.2" | "1.3" (derived, no edit) |

## Out of scope

| Item | Why |
|---|---|
| Clinical extension columns in a format minor | needs the projection study of E9 first; candidate for 1.3 |
| A `bspline` transform kind | only if W22-08's study shows a sampled field cannot be exact (E9) |
| `medh5.cache/2` | a recorded limit (2.1.0 D19); no demand yet |
| Removing anything 2.1.0 deprecated; `Error` variants; wrapping `hdf5::Group` | 3.0 |
| Resampling inside a converter or a loader | by design: geometry is never invented, and a resample is a decision with a filter and a loss that must be recorded. W22-10 adds the deliberate, recorded derivation instead (E11). |
| `rle` | the kind name is reserved; a minor could add it, but only for a measured case `instances` serves badly (E9) |
| Image-free subjects | needs format 2.0: no 1.x minor can relax E111 or E201 (E9) |
| SWMR | reading is already allowed (1.0 §14.4; the engine creates files with HDF5 1.10's superblock, `h5/file.rs`), and an `open(swmr=True)` is a small addition when someone needs it. Writing is out by design: SWMR cannot create objects, and an amendment is copy-on-write with digests over finished content. |
| Storing clinical data as FHIR or OMOP | by design: the profile is not a FHIR or OMOP serialization (1.1 §10), because neither keeps the two time pairs and immutable versions the profile is built on |
| Exporting clinical data to MEDS, FHIR or OMOP | not planned; candidates for 2.3, a MEDS export first (it round-trips with the 2.1.0 importer). OMOP needs a vocabulary mapping the user supplies, since medh5 never guesses a standard code. |
| COCO, cloud object stores | out of scope by design |

## Release checklist (the coordinator)

As 2.1.0's, plus:

1. `docs/spec/medh5-1.2.md` is complete, linked from the README and the spec
   index, and 1.0 §6.3's registry lists `polylines` as 1.2's.
2. `medh5.FORMAT_VERSION` (the newest written) and every page that names the
   newest version say 1.2; `TestStatedCounts` and 1.0 Appendix C.1 recounted.
3. The native NIfTI parity test passes on every CI platform against the 2.1.0
   corpus; nibabel is out of the runtime dependencies.
4. `release.yml`'s platform list includes musllinux and the free-threaded
   wheels; `docs/install.md` lists them; the packaging test that release tuples,
   build callers and `TARGETS` agree passes.
5. `requires-python` is still `>= 3.10` and the wheel tag `cp310-abi3` (E5).
6. `medh5 resample` is in the command-line reference, and its guide is linked
   from the guides index (W22-10).
