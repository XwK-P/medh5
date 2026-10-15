# medh5 2.2.0

A minor release: format 1.2 (one new annotation kind), a native NIfTI
converter, enhanced multi-frame DICOM, free-threaded CPython, new platforms,
resampling as a recorded derivation, and the de-identification and
registration work 2.1.0 left for later. Read [plans/README.md](../README.md)
first (its rules hold here too, with `w22/` branches and `W22-` titles), then
[decisions.md](decisions.md), then your workstream.

- **Baseline.** v2.1.0, released 2026-10-15 from `main` at 9c5e0ec. This
  plan was written before 2.1.0 was implemented and revised after it: every
  "State at 2.1.0" claim was checked against `main` at f836f46 (v2.1.0 plus
  the plans), and the ones that did not hold were corrected. Nothing planned
  for 2.1.0 was deferred, so there is no W22-00. Each workstream still checks
  its claims before starting, since `main` moves under the lanes.
- **Format.** 1.2 adds the `polylines` kind and nothing else (E1); a sample is
  written at 1.2 only when it holds one (E2). 1.0 and 1.1 files are written
  exactly as 2.1.0 writes them.
- **API.** Additive, under 2.1.0's rules. The `h5`, `json`, `numeric` and
  `pyval` modules stayed public in 2.1.0 (W21-06 could not hide them without a
  semver break), so `cargo semver-checks` holds their `pub` items too: a
  workstream adds beside them and changes none. The deprecations 2.1.0
  announced stay
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

## What 2.1.0 taught, and how this plan answers it

Measured on 2.1.0's pull requests (#34–#65), its 35 merge-queue runs and its
lane threads:

| In 2.1.0 | So in 2.2.0 |
|---|---|
| A merge-queue run took 18–28 minutes alone and 43–74 minutes when three to five overlapped; one timed out at 60 minutes with every check green. The Intel Mac build, the slowest job, ran only in the queue, so an Intel-only failure (#51) cost a whole re-queue. | The queue's timeout is 120 minutes. W22-05's macOS x86_64 cross-build lands first, as lane A's enabler, and runs on pull requests again. The maintainer may lower the queue's build concurrency (the release checklist). |
| Most queue drops were conflicts in four files that nearly every pull request touched: the API snapshot, `_core.pyi`, `cli.md` and `app.rs`. | plans/README.md says how to bring `main` into each; the shared-files table below names them. |
| Lanes sat idle for hours with a dropped pull request (#55, 07:32 to 15:47) or a merged edge, because nothing woke them. | plans/README.md, "Waiting and waking": check-ins, subscriptions to the pull requests a lane waits on, an hourly sweep by the coordinator. |
| One lane carried four pull requests in series, and the documentation waited on its last; lanes B, D and F were done by early afternoon. | The lanes below are balanced against their edges: NIfTI has a lane to itself, and enhanced DICOM fills lane F's wait for it. |
| Codex found 31 issues, 29 fixed, but failed silently on six pull requests and never ran on six more. Drafts were reviewed only when marked ready, hours later; a finding on a queued pull request could not be pushed. | plans/README.md, "Review before the queue": open ready, check the review completed, fix, then arm auto-merge; a finding while queued is a follow-up at once. |
| Each session had one branch; two lanes had to ask for a second. | Each lane's brief names every branch below, and the lane may push them all. |
| Lanes told the maintainer pull requests "need your approval"; none ever had one. | The queue needs `ci-ok` only, and plans/README.md says so. |
| D36's "2.2.0 will require Python 3.11" reached the release pull request before it was decided. | A changelog promise names a settled decision; the release checklist lists them (E5 now keeps 3.10). |
| The plan's "State at 2.1.0 (planned)" claims were written before 2.1.0; some were wrong (W22-02's `content_id` parity, nibabel's other callers, W22-04's counts). | Re-checked against v2.1.0 and corrected in each file. |

## Workstreams

"Items" are the numbered "PR n" rows of each file; "Pull requests" is how they
land (plans/README.md: one pull request per workstream, except as the
release names).

| Id | Workstream | Size | Items | Pull requests | Builds on (2.1.0) |
|---|---|---|---|---|---|
| [W22-01](w22-01-format-1.2.md) | Format 1.2: `polylines` | L | 3 | 1 | W21-02 (version accessors, `next_minor`), W21-13 PR 3 (RTSTRUCT points) |
| [W22-02](w22-02-native-nifti.md) | A native NIfTI converter | L | 6 | 3: item 1; items 2–3; items 4–6 | W21-13 PRs 1, 5, 6 (helpers, float64 qform, `--plane-as-volume`, `import-seg-nifti`) |
| [W22-03](w22-03-enhanced-dicom.md) | Enhanced multi-frame DICOM import | L | 3 | 1 | W21-13 PRs 1, 5 (`_frames.py`, per-slice rescale) |
| [W22-04](w22-04-free-threading.md) | Free-threading | L | 4 | 1 (`gil_used = false` its last commit) | W21-01 PR 3 (cp314t wheels), W21-15 PR 1 (GIL) |
| [W22-05](w22-05-platforms.md) | Platforms: musllinux, macOS x86_64, `abi3t` | L | 4 | 2: item 2 (the enabler); items 1, 3, 4 | W21-00 PR 7 (`build.yml`) |
| [W22-06](w22-06-fuzzing.md) | Fuzzing the C libraries | M | 2 | 1 | W21-01 PR 1 |
| [W22-07](w22-07-deidentification.md) | De-identification: collections, clinical text | L | 2 | 1 | W21-10 |
| [W22-08](w22-08-registration.md) | Registration I/O: ITK HDF5 files, B-splines | M | 2 | 1 | W21-07 PR 4 |
| [W22-09](w22-09-clinical-cohorts.md) | Clinical import across a cohort | M | 2 | 1 | W21-12 |
| [W22-10](w22-10-resample.md) | Resampling as a recorded derivation | L | 2 | 2: item 1; item 2 | `transform_between` and its visit refusal (D28), the inverse of a chain (W21-07), one replace routine (W21-05 PR 1) |

So 2.2.0 lands as 14 pull requests and the release pull request. W22-02 is
split because it is large (a codec, an importer and an exporter, each
reviewed on its own) and because W22-10 waits on its last third. W22-10 is
split because its second item waits on W22-02.

## Order of work

- **First, each alone:** W22-05 PR 2 (the macOS x86_64 cross-build: it puts the
  slowest job back on pull requests and off the Intel runner, so every later
  queue run is shorter); W22-02 PR 1 (the parity corpus must be recorded from
  2.1.0 before any NIfTI code changes). Neither waits on the other.
- **Then, in parallel**, with these edges. An edge "A → B" means A merges
  before B can; B's lane codes ahead meanwhile, and subscribes to A's pull
  request so its merge wakes it (plans/README.md, "Waiting and waking").

| Before | After | Why |
|---|---|---|
| W22-02 PR 1 | W22-02 PRs 2–6 | parity is measured against 2.1.0's recorded output |
| W22-02 PRs 2–3 | W22-02 PRs 4–6 | export and the facade build on the engine's reader |
| W22-02 PRs 4–6 | W22-10 PR 2 | the NIfTI refusals, `_same_grid`'s and `import_seg_nifti`'s among them, live in the engine from W22-02 PR 3 on; W22-10 PR 2 rewords them there, and lists the new wording in W22-02's parity test as an allowed difference |

Within a workstream, items land in the order its file gives; W22-04's
`gil_used = false` is the last commit of its pull request, after the audit
and the stress job pass (E4).

### Lanes

| Lane | Sequence | Branches |
|---|---|---|
| A, platform | W22-05 PR 2 (enabler) → W22-05 → W22-06 | `w22/platforms-x86`, `w22/platforms`, `w22/fuzzing` |
| B, format | W22-01 → W22-08 | `w22/format-1.2`, `w22/registration` |
| C, NIfTI | W22-02 PR 1 → W22-02 PRs 2–3 → W22-02 PRs 4–6 | `w22/native-nifti`, `w22/native-nifti-2`, `w22/native-nifti-3` |
| D, Python | W22-04 | `w22/free-threading` |
| E, de-identification and clinical | W22-07 → W22-09 | `w22/deidentification`, `w22/clinical-cohorts` |
| F, derivations and DICOM | W22-10 PR 1 → W22-03 → W22-10 PR 2 | `w22/resample`, `w22/enhanced-dicom`, `w22/resample-2` |

Lane C is the critical path: W22-02 is the largest workstream, and lane F's
last pull request waits on it. Lane F does W22-03 while it waits; W22-03 and
W22-10 PR 2 both change `medh5/io/dicom.py`, so one lane holds both. With
fewer than six sessions, merge lanes in order: A with D, B with E, and C and F
stay apart.

### Shared files (2.2.0)

| File | Owner | Others touch it for |
|---|---|---|
| `crates/medh5/src/version.rs`, `annotations/header.rs` | W22-01 | — |
| `crates/medh5/src/sample/writer*.rs` | W22-01 (`add_polylines`) | W22-10 (writing resampled objects through `amend`) |
| `crates/medh5/src/conformance/**`, stated corpus counts | W22-01 | — (no other workstream adds a case) |
| `medh5/io/nifti.py`, `crates/medh5/src/convert/nifti/` | W22-02 | W22-10 PR 2 (refusal wording, after W22-02) |
| `medh5/io/itk.py`, `medh5/io/nnunetv2.py` | W22-08 (`itk.py`) | W22-02 PRs 4–6 (their NIfTI calls move to the engine) |
| `medh5/io/dicom.py`, `medh5/io/_frames.py` | W22-03 | W22-10 PR 2 (refusal wording; same lane) |
| `.github/workflows/{ci,build,release}.yml`, `.github/scripts/build-dist.sh`, `tests/project/test_packaging.py` | W22-05 | W22-02 PR 2 (the `gz*` symbol check), W22-04 (the free-threaded job), W22-06 (`fuzz.yml` only) |
| `crates/medh5-python/src/**` pyclasses | W22-04 | each binding's owner adds new pyclasses `frozen` from the start (E4) |
| `tests/project/python_api.txt`, `medh5/_core.pyi`, `docs/reference/cli.md`, `crates/medh5-cli/src/app.rs` | no owner | every workstream that adds an API or a command; see plans/README.md, "Shared files" |

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

## Before the lanes start (the maintainer)

1. The merge queue keeps its 120-minute status-check timeout. Consider a build
   concurrency of 2 or 3: in 2.1.0, runs that overlapped took two to three
   times as long as runs alone, and a run ahead that changes restarts every
   run behind it. (Inferred from the timings, not measured as a setting.)
2. Each lane's brief names its branches (Lanes above), so no lane has to ask.
3. The coordinator schedules its hourly sweep (plans/README.md, "Waiting and
   waking").

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
7. Every changelog sentence that promises something about a later release is
   listed in the release pull request, and the maintainer confirms each
   (plans/README.md, "Changelog fragments").
8. The release is its own pull request (version, pins, fragments, regenerated
   files), as 2.1.0's #65 was. Tag `v2.2.0` on the last merge that should be
   in the tagged tree, the release's or a plans pull request after it.
9. After publishing: Read the Docs has built `latest` from the tag's commit
   and activated the `v2.2.0` version (its automation rule activates new
   tags), and `stable` shows 2.2.0; then mark this plan released in
   [plans/README.md](../README.md).
