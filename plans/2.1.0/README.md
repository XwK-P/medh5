# medh5 2.1.0

A minor release: features, errata and deprecations, with no new format
version. Read [plans/README.md](../README.md) first (the rules every
workstream follows), then [decisions.md](decisions.md), then your workstream.

- **Baseline.** v2.0.2, `main` after the 2.0.2 release pull request
  (XwK-P/medh5#30) was squash-merged. "State at 2.0.2" in every file refers to
  that tree.
- **Format.** Files are written at 1.0 or 1.1 only: what 2.0.2 writes, 2.1.0
  writes with the same content and `content_id`, unless an item fixes it (bytes
  that carry a time, such as `created`, differ as they always have). The
  specification changes by errata only, and the code table gains a second
  revision (W915–W919, D1, D2).
- **API.** Additive. Everything 2.1.0 changes in a way a caller could notice is
  a fix of a silently wrong answer (listed under Behaviour changes) or a
  deprecation for 3.0.

## Themes

1. **Integrity that cannot be talked past.** An amendment no longer re-attests
   a file someone edited (D3); a third party's uppercase digest is malformed,
   not tampered (D4); the validator warns where `verify` fails (D6); lost
   updates between two writers are refused (D29).
2. **Answers that were silently wrong.** Tracking calls a lesion resolved where
   it was never imaged (D12); `transform_between` answers by grid names (D28);
   a procedure's own assertion did not attest its mask (D17); two same-named
   RTSTRUCT ROIs became a hole (D10); ROI Jacobians depended on where the ROI was
   cut; MONAI could not batch medh5 items (D30).
3. **Clinical data in and out.** MEDS and FHIR importers; tasks built from a
   cohort and its split; repinning; real clocks and day bounds; caches that
   survive their sources moving; de-identification of clinical samples (D15).
4. **Training at scale.** DDP sharding, sliding-window stitching, a MONAI
   reader, per-class ignore masks, `--workers` for the per-file commands, fewer
   GIL holders.
5. **Converters that say what they did.** Coverage, codec and export flags on
   the command line; FRACTIONAL SEG; codes, categories and laterality in SEG;
   per-slice PET rescale; RTSTRUCT points; nnU-Net test cases, `--replace` and a
   labelmap input that does not need one volume per class; ITK transforms.
6. **One implementation.** Python facades that re-computed engine answers now
   delegate; every file is written by one replace routine (D23); task manifests
   are written by the engine (D24).
7. **A process that catches 2.0.2's surprises early.** One pull request per
   workstream with full CI, changelog fragments, API gates, a deprecation
   mechanism, per-platform builds.

## What 2.0.2 taught, and how this plan answers it

| In 2.0.2 | So in 2.1.0 |
|---|---|
| 21 workstreams merged into one branch; full CI and the review bot first ran on the release pull request, and found two real bugs and three jobs running for the first time. | Each workstream lands on `main` through its own pull request, with full CI, through the merge queue (W21-00 PR 1, PR 7). |
| The changelog was drafted at the end, reconciling about twenty claims against the merged code. | Each pull request writes its own fragment while the change is fresh (W21-00 PR 4). |
| A patch release added fields to three public Rust structs and grew two public arrays, caught by hand. | `cargo semver-checks` and a Python API snapshot gate every pull request (W21-00 PR 8); the API rules in plans/README.md. |
| Failures appeared only on CI's other platforms and floors: Windows hard links, NumPy 2.2's `bool_`, highdicom 0.23, 28 tests without an optional-dependency guard. | Per-platform builds start each platform's tests sooner (W21-00 PR 7); the local gate section says how to reproduce floors, missing extras and Windows paths. |
| A flaky test turned out to be HDF5 descriptors inherited by child processes; on Windows the same bug makes every amend fail while a child runs. | Fixed on Windows, with the race closed, before feature work (W21-00 PR 6). |
| There was no way to deprecate anything. | A mechanism first (W21-00 PR 5), and a 3.0 ledger. |
| Several audit items were already done, mis-scoped or wrong (the lattice Jacobian, the `transform_between` rule, the `digest_algo` premise; refactors called "no behaviour change" that change output). | Every item here was re-verified at 2.0.2, and every workstream re-checks its "State" claims before coding. |
| Research for this plan found bugs the audit missed: amend re-attests a stale root, non-canonical hex reported as tampering, the self-asserted link, a hostile manifest's disclosure, the FIFO hang, lost updates, two `check_held` holes, the MONAI collate failure, same-named ROI holes. | Each is a fix in this release, and the patch-class ones can go to a 2.0.3. |

## Workstreams

| Id | Workstream | Size | Items | Backport |
|---|---|---|---|---|
| [W21-00](w21-00-infrastructure.md) | Infrastructure, phase 0 | M | 8 | PR 2, PR 6 |
| [W21-01](w21-01-platform.md) | Platforms, fuzzing and the supply chain | L | 8 | PR 8 (docs) |
| [W21-02](w21-02-integrity.md) | Integrity, versions and the code table | L | 4 | — |
| [W21-03](w21-03-conformance.md) | Conformance suite | L | 3 | — |
| [W21-04](w21-04-writer.md) | The writer: amendment safety and completeness | L | 4 | — |
| [W21-05](w21-05-files.md) | Files: one replace routine, identities, groups | M | 4 | — |
| [W21-06](w21-06-python.md) | Python API and layering | L | 6 | — |
| [W21-07](w21-07-transforms.md) | Transforms and registration I/O | L | 4 | — |
| [W21-08](w21-08-cohorts.md) | Cohorts: manifests, statistics, splits, checks | L | 4 | — |
| [W21-09](w21-09-curation.md) | Curation: tracking and agreement | M | 2 | — |
| [W21-10](w21-10-deidentification.md) | De-identification | L | 3 | — |
| [W21-11](w21-11-task-cache.md) | Task and cache companion | L | 5 | PR 1 |
| [W21-12](w21-12-clinical-import.md) | MEDS and FHIR importers | L | 3 | — |
| [W21-13](w21-13-converters.md) | Converters | L | 9 | — |
| [W21-14](w21-14-training.md) | Training: MONAI, DDP, stitching, class masks | M | 4 | — |
| [W21-15](w21-15-concurrency.md) | Concurrency: GIL and `--workers` | M | 2 | — |
| [W21-16](w21-16-documentation.md) | Documentation | M | 5 | — |

Sizes: S is a day or two of focused work, M about a week, L more. The release
itself is the coordinator's, at the end of this file.

## Order of work

**Phase 0.** W21-00 PRs 1–7, in order, by one session. No feature pull request
merges before W21-00 PR 4 (fragments) and PR 5 (deprecations) have; sessions
may start coding before that. PR 8 (the API gates) compares against the newest
published release, v2.0.2, on PyPI and crates.io since 2026-10-14.

**Then, in parallel.** Every other workstream, respecting the edges below,
each as one pull request (four enablers land ahead of theirs; see Lanes),
merged through the merge queue once W21-00 PR 7 has given CI its
`merge_group` trigger and the maintainer has turned the queue on.

### Dependencies

An edge "A → B" means A must be merged before B can merge. An item named here
("W21-05 PR 1") is merged when the pull request holding it is: its
workstream's one pull request, or the enabler it lands in. Within a
workstream, items land in the order its file gives.

| Before | After | Why |
|---|---|---|
| W21-00 PR 4, PR 5 | every feature pull request | fragments and the deprecation helper |
| W21-00 PR 6 | W21-03 PR 1, W21-05 PR 3, W21-15 PR 2 | `h5/file.rs` open and create; the handle helper; children started under the HDF5 lock |
| W21-00 PR 7 | W21-00 PR 8, W21-01 PR 3 | job names |
| W21-00 PR 8 | W21-06 PR 6 | hiding modules needs the semver gate to judge it (D35) |
| W21-02 PR 1 | every new code (W21-02 PRs 2–4, W21-13 PR 2) | the code-table revision (D1) |
| the first revision-2 code (W21-02 PR 2, or W21-13 PR 2) | W21-03 PR 2 | scoring bounded by a revision needs a later revision to bound |
| W21-04 PR 1 | W21-05 PR 1, W21-10 PR 3 | `AtomicFile` snapshot preserved by the refactor; the clinical hold fixed before a clinical derivation |
| W21-04 PR 2 | W21-13 PR 8 | both add writer entry points in `writer_annotations.rs` |
| W21-05 PR 1 | each owner moving its write sites (W21-07 PR 4, W21-11 PR 3, W21-13 PR 9, W21-14 PR 4) | `_core.write_atomic` and `stage_file` |
| W21-08 PR 1 | W21-08 PR 3, W21-11 PR 4 | split keys and cohort entries resolve through the manifest path rule (D21) |
| W21-11 PR 2 | W21-12 | one rounding of hours and days |
| W21-11 PR 3 | W21-11 PR 4 | `SourceAccess` |
| W21-11 PR 4 | W21-12, W21-16 PR 4 | `Clock.utc`, `day_bounds`, the task builder the tutorial uses |
| W21-15 PR 2 | W21-16 PR 5 | the concurrency page describes `--workers` |
| W21-09 PR 2, W21-13 PR 3 | W21-16 PR 2 | two of its guides describe `map_classes` and same-named ROIs |

A page describes what `main` does when its pull request lands. When a later
pull request changes what a page says (W21-10 PR 3 makes `scrub` cover the
clinical profile that W21-12 PR 3's guide says it does not), the later pull
request updates the page; that is not an edge.

### Lanes

With six sessions, these sequences keep each session's files apart and meet
every edge above; with fewer, merge lanes in order (A with F, B with E, C with
D).

| Lane | Sequence |
|---|---|
| A, infrastructure | W21-00 → W21-01 → W21-16 |
| B, format core | W21-02 PR 1 → W21-04 PR 1 → W21-05 PR 1 (the three enablers) → W21-02 → W21-04 → W21-03 → W21-05 |
| C, clinical | W21-11 → W21-12 → W21-10 |
| D, converters and registration | W21-13 → W21-07 |
| E, Python and cohorts | W21-08 PR 1 (an enabler) → W21-08 → W21-06 → W21-09 |
| F, training and concurrency | W21-14 → W21-15 |

Each name in a sequence is one pull request: a workstream's whole file, or,
for an enabler, its first item alone. Four first items land as enablers,
ahead of the rest of their workstream, because other lanes wait on them:
W21-02 PR 1 (lane D's W21-13 PR 2), W21-04 PR 1 (lane C's W21-10 PR 3),
W21-05 PR 1 (lane C's W21-11 PR 3, lane D's W21-07 PR 4 and W21-13 PR 9,
lane F's W21-14 PR 4) and W21-08 PR 1 (lane C's W21-11 PR 4). The rest of each
of those workstreams (items 2 on) is its second pull request. So 2.1.0 lands
as W21-00's eight pull requests, 20 others and the release pull request.

A workstream's pull request merges when every edge into any of its items has
merged; its session codes ahead from the latest `main` and rebases as the
edges land. W21-16 is one pull request, lane A's: its items 1 and 3 can be
written first, and it merges after W21-09, W21-11, W21-13 and W21-15, whose
features its other pages describe.

## Corpus and code-table targets

What the shipped conformance suite and the code table should hold when every
workstream has landed. `TestStatedCounts` and the code-table tests hold the
truth; the release pull request recounts every stated number.

| | 2.0.2 | 2.1.0 | From |
|---|---|---|---|
| Codes | 95 | 100 (W915–W919, revision 2) | W21-02, W21-13 |
| Format cases | 163 | 172 | below |
| valid / invalid | 51 / 112 | 58 / 114 | |
| `clinical` cases | 36 | 37 | W21-10 PR 2 |
| Task fixtures | 14 | 15 (`valid-self-asserted-mask`) | W21-11 PR 1 |
| Cache fixtures | 0 | 9 (T401–T407, a valid event cache, a valid patient cache) | W21-03 PR 3 |

New format cases: `W915-undigested-attested-dataset`,
`integrity-index-sha512`, `E703-digest-not-canonical` (W21-02 PRs 2–3);
`W916-implicit-negative-ancestor`, `W917-plane-and-volume-share-a-frame`,
`W918-implicit-voxel-annotation` (W21-02 PR 4); `W919-contour-role-depth`,
`E404-contour-role-value` (W21-13 PR 2);
`clinical-shifted-clock-withheld-shift` (W21-10 PR 2). No existing case
changes.

## Out of scope

| Item | Where it went |
|---|---|
| RTSTRUCT open curves as `polylines` (format 1.2) | 2.2.0 |
| Enhanced multi-frame DICOM import | 2.2.0 |
| A native NIfTI converter in the engine | 2.2.0 (2.1.0 makes the qform float64, its prerequisite) |
| Free-threading (`gil_used = false`), musllinux, `abi3t`, the macOS x86_64 cross-build | 2.2.0 |
| Dropping Python 3.10 | not planned: kept in 2.2.0 (D36 revised, 2.2.0 E5) |
| Scrubbing collections; rewriting clinical text | 2.2.0 |
| ITK `.h5` transform files, B-spline transforms | 2.2.0 |
| A batch MEDS import across a cohort | 2.2.0 |
| Fuzzing the C libraries | 2.2.0 (2.1.0 fuzzes the Rust readers) |
| Annotation pyramids | reserved (D11), no release planned |
| `medh5.cache/2` with a digest of what a row read | recorded as a known limit (D19), not planned |
| `Error` gains variants; wrapping `hdf5::Group` | 3.0 |
| Resampling in converters or loaders | by design; 2.2.0 adds a recorded `medh5 resample` derivation instead (2.2.0 E11, W22-10) |
| `rle`, image-free subjects, SWMR, FHIR/OMOP | 2.2.0's Out of scope says, item by item, what each would need |
| COCO, cloud object stores | out of scope by design |

## Release checklist (the coordinator)

1. Every workstream is merged, or its remaining items are moved, with their
   text, into the 2.2.0 plan and named in the release pull request.
2. The guard tests' allow-lists are empty (write sites, W21-05; vocabularies,
   W21-06), and `tests/project/undocumented_api.txt` only shrank (W21-16).
3. `python .github/scripts/changelog.py check` passes; write
   `changes/lead.md`, the release's bold summary paragraph.
4. Set `[workspace.package] version = "2.1.0"` and the three `=2.1.0` pins in
   `[workspace.dependencies]`; `cargo update -w`; regenerate
   `THIRD_PARTY_NOTICES`, `tests/project/python_api.txt` and
   `docs/reference/api-index.md`.
5. `python .github/scripts/changelog.py build 2.1.0 --date <day>`.
6. Recount every stated corpus and code number (`TestStatedCounts`, 1.0
   Appendix C.1's sentence). (The check that `docs/install.md` says 2.1.x is
   the last line to support Python 3.10 was withdrawn with D36's announcement.)
7. Read the assembled changelog against the merged commits, as 2.0.2's
   coordinator did: each claim matches its code. Check the release notes length
   (`release_notes.py`, 125,000 characters).
8. Open the release pull request, drive it green, merge it, tag `v2.1.0` on the
   merge commit. `release.yml` publishes.
9. Afterwards: mark this plan released in [plans/README.md](../README.md),
   cut `release/2.1` only when a 2.1.1 is needed, and start 2.2.0.
