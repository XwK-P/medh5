# PR #25 audit (`ca0634b`): verification and fix plan

Audited head `ca0634b9bfba82edcb3793db966efa2e4b28dc16`. Verified on 2026-10-09 against a fresh
release build of that head (an abi3 wheel and the native `medh5` CLI, macOS arm64), using
reproducers written independently of the audit's probes.

## 1. Verdict

**All 26 findings are real; none is a false positive.** 25 were reproduced by running them. F24
was confirmed from the code; the docstring of the function at fault describes the faulty
ordering. For F10 I measured the unbounded allocation; the abort itself needs Linux
`RLIMIT_AS`, which macOS does not enforce. Several findings are broader than the report states
(§3). I agree with **hold merge and release**.

The 9 P1 / 17 P2 split stands, with three caveats:

- **F01 is adversarial-only.** It needs a file crafted *before* pinning: h5py aliases that sort
  before `images/`, digests restamped at the alias paths, and the root resealed. The public
  writer never creates links, and an alias added *after* pinning is detected. It stays P1
  because it opens a hole in the pin contract, which B03 already treated as blocking, and the fix
  is small.
- **F03 is defense in depth.** It needs a cache whose entries were mis-joined by user code. It
  stays pre-merge because the check is cheap and the clinical stack promises no leakage.
- **F17 and F19 are unlikely in practice.** F17 needs a `|` inside a code system or code. F19
  needs one subject imaged in mm at one visit and in m at the next. Both are cheap; fix them in
  the same pass.

## 2. Verification results

| ID | P | Verified | Observed at `ca0634b` |
|---|---|---|---|
| F01 | P1 | run | CT[0] 1028→2345 after one post-pin hard relink; `content_id` unchanged and recomputes; `verify`, shallow and deep pin `check` clean; validator reports only W903 (no de-id record) |
| F02 | P1 | run | Text attached to an inherited text-less document event, and a second image attached to inherited `ct0`: 0 errors; at cutoff 24 h the event available at 1 h owns the later text |
| F03 | P1 | run | Entry `event_id=rep_v1`, `document_id=rep_text_v2`: `validate_cache` ok, no findings; `ClinicalTaskDataset` serves it at cutoff 24 h, availability age 20 h |
| F04 | P1 | run | Stash names `c1` only; class 2's 8 voxels written as 0; `dataset.json` omits the class |
| F05 | P1 | run | `sample_id="../../../victim"` overwrote a file outside the export root; an **absolute** `sample_id` writes anywhere |
| F06 | P1 | run | mm transform on m grids: no errors, `aligned="transform"`, values 18–19 from the far edge; matched mm/mm and m/m controls give 5–6 |
| F07 | P1 | run | Full read and native `validate --level integrity` abort with SIGABRT at `blosc2_filter.c:534`; window read unaffected; valid control fine |
| F08 | P1 | run | Python and native `clinical strip`, portable and balanced: output verifies at 1.0, yet the synthetic marker is in its raw bytes |
| F09 | P1 | run | Metre grid gives `0.003` where 3 mm is expected; RAS x/y signs not flipped; frame `9.9.9` relabeled as the source image's frame |
| F10 | P2 | measured | A 16,976-byte file drives integrity validation to about 687 MB peak RSS (baseline 34 MB); uncapped it completes with E701 |
| F11 | P2 | run | Pack then unpack: `verify` fails (`zzz_alias` mismatched, root not ok). No-op amend changes `content_id` and splits the alias |
| F12 | P2 | run | Recompress reports `ok`, `verified` and `content_id_preserved`; the enum mapping is gone |
| F13 | P2 | run | `subtrees_identical(root, root)` on a self-cycle: SIGSEGV. Tree control fine; `validate` of the cyclic file fine |
| F14 | P2 | run | `classes:[3.0]` loads as `()` and `patch:[4.0,8,8]` as `(8,8)`, with no diagnostic |
| F15 | P2 | run | 7-day window at day +1: a completed event at day −100 is excluded, an identical *planned* one admitted |
| F16 | P2 | run | Empty row has shape `(0,2)`, populated row `(1,2,3)`; collation works in one order and raises in the other |
| F17 | P2 | run | Both concepts become `observation\|alpha\|beta\|gamma` |
| F18 | P2 | run | `as_world(metre grid)` returns mm numbers; `world_corners` on the same grid scales correctly |
| F19 | P2 | run | 8 mm³ vs 8e-9 m³: relative change −0.999999999 for an unchanged lesion |
| F20 | P2 | run | Zero-weight class drawn 14 of 30 times without a fresh index, 0 of 30 with one |
| F21 | P2 | run | Identified OBBs at two visits: Python raises `TypeError`; native `track` exits 1 |
| F22 | P2 | run | 1 file in imagesTr, 0 in labelsTr, `numTraining` 1 |
| F23 | P2 | run | `classes=[3]` gives labels `{background:0, c3:3}`; nnU-Net's verifier asserts consecutive labels |
| F24 | P2 | code | `_withdraw_volume_statements` (`nifti.py:183`) runs before `nib.save` (`:186`) |
| F25 | P2 | run | A non-degenerate 4×3 affine (`decompose_affine`, `index_to_world`) and a 3-D box with a 2-D shape (`box_to_slices`) raise `PanicException`, a `BaseException` |
| F26 | P2 | run | Migrated `/meta.extra` has only a `legacy` key, with `nnunetv2` nested inside it |

Of the non-blocking observations I spot-checked only the version-bump runbook, and it is
confirmed: CONTRIBUTING says the version lives in one place, but `Cargo.toml:28-30` also pins
`=2.0.0`. I did not re-verify the others.

## 3. Wider than the report states

- **F01:** `SourceRef::pin` (`companion/source.rs:45`) checks nothing. It copies the stored
  `content_id`, so a crafted file pins without complaint.
- **F05:**
  - An absolute `sample_id` defeats the export root entirely, because pathlib discards the base
    when joined with an absolute path.
  - Duplicate sample IDs silently overwrite an earlier case while `numTraining` counts both. On
    macOS and Windows, IDs that differ only in case collide too.
- **F06:** `add_transform` defaults `units` to `"mm"` without looking at the grids
  (`writer_annotations.rs:823`; also `transforms/model.rs:80,144`). Any non-mm sample that omits
  `units` therefore writes a mislabeled transform. The paired loader ignores `units` altogether,
  so whether it lands correctly depends on the numbers, not on what is declared.
- **F08:**
  - `SampleWriter.drop_clinical()` is public in Python, so any amend that drops clinical records
    (and maybe re-adds some) leaks the same way.
  - `write_clinical` uses the same copy-then-unlink pattern. That is bloat rather than a leak,
    because records are append-only.
- **F14:**
  - `dataset/manifest.rs:186` has the same silent-drop parser.
  - The clinical parser already *refuses* non-integers explicitly (`clinical/model.rs:138`,
    E811). That is the house precedent to follow.
- **F15:** read literally, the spec's window `[c − w, c]` would exclude every future plan under
  `order_by = effective`. The code tests only the lower edge, which is the sensible reading; the
  spec should say so.
- **F20:** even with a fresh index, if every class present has zero weight, `pick_class` returns
  `None` and the code falls through to the *unweighted* scan.
- **F21:** §10.6 says landmark `points` SHOULD carry `instance_ids` for trackable objects. A
  third-party file that does so aborts tracking the same way. The public `add_points` cannot write
  them.
- **F23:** the problem is not limited to `classes=` subsets. Any exported set of class IDs with a
  gap, such as {1, 3}, fails nnU-Net's consecutive-label check.

## 4. Fix plan

Land the fixes as focused commits on the PR branch, or on a follow-up branch merged before tagging,
in the order below. Each commit carries its regression tests, named for the clause they hold
(`test_S13_2_…` in Python, `s13_2_…` in Rust). Record them as the round-4 fixes in the CHANGELOG.

Sizes: **S** is under half a day, **M** is 1–2 days, **L** is 3 or more days.

### WP-A Native robustness: F07, F13, F25, F10 (land first; small and isolated)

**F07 (S).** `crates/medh5-sys/vendor/hdf5-blosc2/blosc2_filter.c:534`

- Replace `assert(outbuf_size >= size)` with a checked branch: `PUSH_ERR(...)` and
  `goto b2nd_decomp_out`.
- Compute `size = typesize × Π shape` with overflow checks. When `ndim >= 0`, also require
  `outbuf_size == typesize × Π chunkshape`.
- Turn the compression-side asserts (lines 226–263) into error returns too.
- Record the local patch beside the vendored sources and offer it upstream. Do not "fix" this by
  defining `NDEBUG`.
- Tests:
  - Plant the bad `cd_values[3]` in a subprocess that never imports `hdf5plugin`. Otherwise
    `set_local` rewrites the client data and the defect disappears.
  - Read in a second subprocess. Assert an ordinary `MEDH5Error` and no signal, for both the full
    read and the window read.
  - Native `medh5 validate --level integrity` exits 1 with a diagnostic.
  - If the corpus's damage generator can patch bytes, add this as a damage case so both CLIs run
    it on every platform.

**F13 (S).** `integrity/verify.rs:271`, `compare_groups`

- Carry two maps of `ObjectId`s, `a→b` and `b→a`, plus a depth bound (`ops::MAX_DEPTH`).
- When a pair is already mapped consistently, stop. When the mapping is inconsistent, report
  "alias topology differs": an alias and a duplicate must still compare unequal.
- Compare soft links by their stored target rather than following them, as recompress does.
- Tests: equal and unequal cyclic graphs, alias vs duplicate, soft and dangling cycles, and a
  Python subprocess at the default stack size.

**F25 (S).** `geometry/affine.rs:51` (`decompose_affine`), `index_to_world` / `world_to_index`,
and `:153` (`box_to_slices`)

- Add one `require_square_affine(&a) -> Result<usize>`: rows equal columns, and at least 2. This
  also stops the `nrows() - 1` underflow on an empty array.
- Add a check that the points' width matches the affine.
- `box_to_slices` requires `shape.len() == S`.
- Sweep every `pub fn` in `geometry/` that indexes by a dimension the caller supplies.
- Tests: the three counterexamples and empty arrays raise `Exception`, and the tests assert it is
  not `PanicException`. Valid controls still pass. Add Rust unit tests.

**F10 (M).** `h5/data.rs:324` (`read_strings`), `h5/attrs.rs:360` (`read_fixed_strings`), and the
string branch of `dataset_digest`

- Budget the *decoded* result, `n × (size_of::<String>() + width)`, through `ensure_allocatable`,
  and `try_reserve_exact` the `Vec<String>`.
- Digest string datasets in slabs of `STREAM_BYTES`, hashing and dropping each slab's elements, so
  integrity validation never holds every string at once.
- Attributes cannot be read partially: apply the decoded budget to them and refuse beyond it.
- Tests:
  - A Rust test for the slab budget, which is an engine internal.
  - A Linux-only subprocess test with `RLIMIT_AS` set to baseline + 512 MiB, expecting E701 and no
    signal. CI has Linux runners, so this is not a test that only ever skips.

### WP-B Graph-aware copy: F11, F12 (before WP-C, which reuses the copier)

**F11 (M–L).** `collection.rs:216` (`copy_root`, used by pack, unpack and extract) and
`sample/writer.rs:331` (`inherit`, used by amend) call `H5Ocopy` once per top-level member. HDF5
shares no object map across those calls, so cross-member hard links become duplicates.

- Lift recompress's walker (`storage/recompress.rs:265`, `copy_group`) into
  `h5::ops::copy_graph`. It already keeps a shared `Copied` map, keeps soft links as links,
  re-links hard links to objects already copied, and bounds depth.
- Give it a raw mode that `H5Ocopy`s leaf datasets only, which keeps pack's promise that chunks
  move as raw bytes.
- Use it for `copy_root` and for everything `inherit` copies, with **one** map for the whole
  sample.
- Walk in the same byte order, so first paths, which `content_id` lines are keyed by, do not
  change.
- Verify the staged output before replacing anything: pack and unpack must keep each member's
  `content_id`. Never restamp to hide a change.
- Tests: a cross-root alias, a cycle and a soft link, each through pack, unpack and amend. Assert
  object identity (equal h5py `.id`) and equal `content_id`.

**F12 (S–M).** `storage/recompress.rs:342` (`copy_dataset`) rebuilds a dataset from a primitive
`DType` once it is above the size threshold.

- When the file datatype is committed (`H5Tcommitted`) or an enum (or anything `data::kind`
  reduces to a primitive), copy the dataset as stored. Report it in the result as kept, not
  re-encoded.
- A committed type shared by several datasets goes through the WP-B map, or
  `H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG`.
- Tests:
  - Cover named and anonymous committed types with attributes, and enums, both above and below the
    threshold.
  - Assert the enum mapping, committed status and type attributes survive.
  - Assert digests and `content_id` are preserved.

### WP-C Privacy: F08 (P1)

- **`clinical strip`** (`clinical/augment.rs:248`): drop the `fs::copy` → `amend` → unlink
  sequence. Build the projection with a writer that reads `path` and **never copies
  `clinical/`** into the new file. An inherit option can exclude it from `copy_unknown`; profiles
  are re-derived as they are now.
- **`drop_clinical()` on an ordinary amend** (`sample/writer.rs:320`, public in Python): set a flag.
  At commit, if the flag is set, compact the staged file before the atomic replace by copying its
  live graph into a fresh temporary file with WP-B's copier.
  - An alternative is to make clinical inheritance lazy, copying it at commit only if unchanged.
    It is more invasive and not needed for this fix.
- Keep current behaviour: the source is never touched, an existing target is refused, and a
  failure leaves no output.
- Tests:
  - A unique synthetic marker is absent from the output's raw bytes, for the Python and native
    commands and for portable and balanced storage.
  - Strip followed by a read-only `scrub.scan`.
  - Amend, then `drop_clinical()`, with and without re-adding records afterwards.
  - The source hash is unchanged, and a failure leaves no `out`.

### WP-D Integrity binding: F01 (P1)

- **Rule.** Every attested dataset path (from `attested_datasets()`, soft links resolved) must
  itself be a key of `collect_digests(root, ["index"])`, that is, the path its object's line
  names.
  - Anything else is unbound: an alias that sorts earlier, a soft link, or an object first reached
    elsewhere.
  - This subsumes today's identity test in `unattested()` (`integrity/verify.rs:55`).
  - The `content_id` formula does not change, so no conforming file's address changes.
- **Wire it through:**
  - `VerifyResult`: an unbound path makes the result not ok.
  - `SourceRef::check`: report T302.
  - The validator (`validate/rules/integrity.rs`): a new **E704**.
  - `SourceRef::pin`: refuse an unbound file, so a crafted file cannot be pinned at all.
- **Spec 1.0 §13.2:** a spec-defined path is covered only when it is the path its object's line
  names. Record this in Appendix C, and add E704 to `codes.json`.
- **Corpus:**
  - Add an invalid case (an aliased attested path) and a valid one (a later extension alias,
    `zzz_alias → images/CT`).
  - Update the stated counts that `TestStatedCounts` checks.
  - Look for an existing *valid* case with an earlier alias inside an attested group. It would
    become invalid, which is intended.
- **Tests:**
  - Hard and soft relinks between two covered objects, for both `images/CT` and
    `clinical/events/value_num`.
  - `verify`, `pin`, preflight and the validator all refuse.
  - A later alias stays valid as a positive control.
  - No test reseals after pinning.
- **Rejected alternative:** adding alias lines to `content_id`. It changes existing addresses and
  breaks the rule that `content_id` must not depend on which version wrote the file.

### WP-E Clinical: F02 and F03 (P1); F15, F16, F17, F14

**F02 (S).** `sample/writer_clinical.rs:132`, `add_link`

- In the `Inherited` branch of `clinical_records()` (`:32`), record the inherited event IDs in a
  set of their own. `clinical_ids.0` cannot serve, because it also grows with new events.
- In `add_link`, refuse a `describes` link from an inherited event to a `document` or `image` with
  **E809**: "an event version is immutable; adding or replacing an owned payload needs a new event
  version that supersedes it (1.1 §7.3)".
- Non-structural links, such as groundings and links with `asserted_by_event_id`, stay allowed.
- `add_records` and `clinical augment` both go through `add_link`, so they are covered.
- Spec 1.1 §7.3/§10: note that the writer enforces this at the amendment boundary, because a
  single file cannot show it.
- Tests:
  - Attaching a document and attaching an image to an inherited version are both refused.
  - A new version available at 48 h that supersedes the old one: its report is excluded at cutoff
    24 h and admitted at 72 h.

**F03 (M).** `companion/cache.rs:565` (`validate_cache`) and `medh5/torch/clinical.py:717`
(`_document_features`)

- **Validation:** for an event-level entry, use the source already opened for the pin check.
  `event_id` must be an event version of that source. When `document_id` is present, it must be
  the document that version owns through a structural `describes` link. Anything else is a new
  **T407**.
- **Loader:** before using a cached feature, check the entry's `document_id` against the owned
  document that the row selected. Expose the entry's metadata next to `event_feature` for this.
  The check then also covers a cache that was lazily reopened or unpickled.
- **Spec:** task-cache-1 §7.2 and the code list in §7.4. Add T407 wherever T401–T406 are listed.
- **Tests:**
  - A wrong-owner event and a foreign-source event are refused, both by `validate_cache` and when
    `ClinicalTaskDataset` is constructed.
  - A correct-owner control passes.
  - Lazy reopen and a pickle round trip.
  - The existing N11 manifest check is unchanged.

**F15 (S).** `clinical/select.rs:543`

- Apply the window's lower edge to plans before admitting them, honouring `context_boundary` and
  `uncertainty`.
- Leave the upper edge off for plans: a plan may lie after the cutoff by definition.
- Factor the window test into one closure used by both branches.
- Spec 1.1 §9 step 3: say that the window bounds order times from below for every admitted event,
  plans included, and that plans may lie after `c`. Record it in Appendix A. Add a selection case
  if the companion corpus carries selection cases.
- Tests: availability and effective ordering; a stale past plan is `outside_context`; a future plan
  is admitted; boundary equality, closed and open; contained and overlaps.

**F16 (S).** `medh5/torch/clinical.py:764` (`_feature_dim`) and `:984` (`_pad_sequences`)

- Return the full declared output shape, and build an empty row as `np.zeros((0, *shape))`. An
  encoder `dim` given as an int becomes `(dim,)`.
- In `_pad_sequences`, take the trailing shape from the first non-empty row and refuse rows that
  disagree with it.
- Tests: both row orders, all-empty and all-populated batches, rank-1 and rank-2 outputs, and the
  masks.

**F17 (S).** `medh5/torch/clinical.py:97` (`concept_of`) and `medh5/task.py:770` (`concepts`)

- Use one shared `concept_token(kind, system, code)` that escapes `\` and `|` in `code_system` and
  `code`.
- Tokens without those characters stay byte-identical, so vocabularies already fitted remain valid.
- Tests: delimiter and backslash cases, Unicode, coded vs uncoded, and stability across saving and
  reloading a vocabulary.

**F14 (S).** `companion/task.rs:81` (`Slot::from_json`) and `dataset/manifest.rs:186` (`list_int`)

- Never `filter_map` numbers. Refuse a non-integer JSON number explicitly with T101, naming the
  field and the value, as the clinical parser does with E811.
- Refuse a non-positive `patch` as well.
- Tests: integers, `3.0`, out-of-range, fractional and negative values, and mixed arrays. Each must
  normalise identically or be refused before preflight.

### WP-F Converters: F05, F04 and F09 (P1); F22, F23, F24, F26

**nnU-Net export: F05, F04, F22 and F23 (M–L).** One refactor of `medh5/io/nnunetv2.py:521`
(`_plan`), which settles everything before the first file is written.

- **F05:**
  - Every `sample_id` must pass `_core.validate_sample_key` (`[A-Za-z0-9_.-]{1,255}`) and must not
    be `.` or `..`.
  - Refuse duplicate IDs, compared case-insensitively.
  - As defense in depth, resolve each planned output path and require it to lie under the export
    root.
  - Apply the same check to `dataset_name`.
  - A refusal writes nothing.
- **F22:**
  - A case without the annotation is refused by default, and the error names the cases.
  - Optionally add `unlabeled="test"` to write such cases to `imagesTs` instead.
  - `numTraining` counts only the labeled cases written.
- **F04:**
  - When reusing a stash (`:615`), every class in the dataset-wide union must be named by the
    stashed labels.
  - A scalar stash is extended with the missing classes, and the decision is logged. A region stash
    (`regions_class_order`) is refused.
  - As a backstop, `_not_given_back` (`:783`) checks every exported class, not only those in the
    vocabulary.
- **F23:**
  - If the final scalar label values are not `0..K`, map the class IDs to `1..K` in ascending
    order.
  - Write the volumes through the map, put the ignore label at `K+1`, and report the mapping.
  - Also store the mapping in `dataset.json` (for example as `medh5_class_ids`) so
    `from_nnunetv2` can restore the original IDs.
  - A stash that is still valid is left untouched.
- **Tests:**
  - A real import, then an amend adding a class, then an export.
  - A mix of annotated and unannotated cases, sparse ID sets and region mode.
  - Traversal, absolute paths, both separators, duplicates and case-only duplicates. After every
    refusal, all pre-existing files are byte-identical.
  - Port nnU-Net's two checks (consecutive labels including the ignore label; a label file for
    every training case) into a test helper and run it on every export.

**F09 (M).** `medh5/io/rtstruct.py:393`, `to_rtstruct`

- **Coordinates:** convert to DICOM patient coordinates. Scale by `medh5.io.nifti.MM_PER_UNIT` and
  flip x and y for RAS, the inverse of the SEG code's `_from_patient` (`dicom_seg.py`).
  - Refuse `px` and any convention other than LPS or RAS.
  - Refuse world-space contours with no grid, since their units are unknown.
- **Frames:**
  - Every source image must share one `FrameOfReferenceUID`; check all of them, not only the
    first.
  - A grid that declares `frame_uid` must agree with that frame. Mirror `dicom_seg.py:966`, and
    use `frames_agree` with a `frame_salt=` parameter for pseudonymised frames.
  - A grid without a frame logs a guess, as the SEG path does.
- **Tests:** an mm/LPS/same-frame control; metre; RAS; an unrelated frame; mixed-frame sources;
  index and world space. Each inspects the emitted `ContourData`.

**F24 (S).** `medh5/io/nifti.py:183-189`

- Reorder the steps:
  1. Validate the sidecar without changing anything.
  2. `nib.save` to the temporary file.
  3. Withdraw the old per-volume fields and the `.bval`.
  4. `os.replace` the image into place.
  5. Write the new statements.
- This keeps the docstring's invariant (a new image is never paired with old timing), and a
  failure to serialise changes nothing.
- Split `_withdraw_volume_statements` into a step that plans and a step that applies.
- Tests: inject an `OSError` in `nib.save` (the old image, `.bval` and sidecar are unchanged);
  inject one in `os.replace`; the success path is unchanged.

**F26 (S–M).** `convert/legacy.rs:632`

- Write `extra.nnunetv2` to `/meta.extra.nnunetv2` verbatim, as Appendix B specifies.
- Keep the remaining 0.x keys under `extra.legacy`, keyed per study or timepoint, so a grouped
  migration merges them instead of replacing them.
- If grouped studies carry different `nnunetv2` metadata, keep the first and report it, or refuse
  (decision 10 in §5).
- Spec Appendix B: state where the other 0.x `extra` keys go.
- Tests:
  - A single file has its metadata at the specified path.
  - Migration followed by `to_nnunetv2` planning finds it.
  - Two studies with different metadata are both preserved, and a conflict is reported.

### WP-G Geometry and curation: F06 (P1); F18, F19, F20, F21

**F06 (M).** The writer, the validator and the loader change together.

- **Writer** (`writer_annotations.rs:802`): take the default `units` from the endpoint grids
  (`from_grid` and `to_grid`, or else the grids in each frame). Refuse an explicit `units` that
  disagrees with them, as a new **E506**.
- **Validator** (`validate/rules/transforms.rs:15`): `units` must be present (it is a MUST) and
  equal to the units of the grids in both frames, or E506. Composite legs are already checked
  (`:253`).
- **Loader** (`medh5/torch/datasets.py:788`) and any Rust consumer of a resolved transform,
  including inverses and chains: refuse a units mismatch (E506) before reading a patch. Do not
  convert, because the spec defines no cross-unit transform.
- **Compatibility:** a 1.x file with metre grids and transforms left at the default `"mm"` becomes
  invalid. Say so under Behaviour changes, and consider a `repair` rule that restamps `units` from
  the grids on request.
- **Tests:**
  - The two matched controls.
  - An explicit mismatch.
  - Omitted `units` on metre grids, where the writer now writes `"m"`.
  - Validator E506; the loader refuses.
  - Resolution of inverse and composite transforms.

**F18 (S).** `annotations/read_geometric.rs:335`

- Take the fast path that returns stored values only for the annotation's own grid.
- For any other grid, compute the bounds of `world_corners(grid)`, which already converts units
  and refuses incompatible frames and conventions with E414.
- Tests: mm to m, an unrelated frame, RAS vs LPS, and an own-grid control.

**F19 (S–M).** `curation/tracking.rs:305` (`measure`) and `:167` (`relative_change`)

- Express calibrated volumes in mm³, or carry the unit on each observation and convert in
  `relative_change`.
- Return `None` across px vs calibrated grids and across 2-D vs 3-D.
- The CLI output follows the same rule.
- This changes behaviour only for non-mm grids.
- Tests: the same lesion in mm, m and µm gives a change of 0; real growth across a unit change;
  px vs mm and area vs volume give `None`; the CLI.

**F20 (S–M).** `sampling.rs:323` (`foreground_center`) and `:389` (`scan_center`)

- **Scan fallback:** count each candidate class, one dense mask at a time. Then `pick_class` with
  the configured weights, then sample a voxel within the chosen class.
- **Fresh index:**
  - Use the index only if it covers every candidate class.
  - When `pick_class` returns `None`, return `None`: an empty weighted pool must not turn into an
    unweighted one.
- **Behaviour change:** seeded draws on the no-index path will differ; note it.
- **Tests:**
  - Absent, fresh and stale indexes give the same class frequencies under fixed seeds, and exactly
    0 for zero-weight classes.
  - All-zero weights give `None`.
  - The frequency and inverse-frequency policies.

**F21 (M).** `curation/tracking.rs:301` (`carries_instance_ids`) vs `annotations/read.rs:1075`
(`instances()`)

- Make the two agree.
- Implement `obb` instances:
  - Use the enclosing axis-aligned box for joins.
  - Volume is Π sizes × voxel volume in index space; rotation does not change it.
  - In world space, volume is Π sizes.
- Implement `points` that carry `instance_ids` as presence-only observations (§10.6).
- Skip any other kind and report it on `Tracking` and in the CLI; never abort.
- Optional follow-up: `add_points(instance_ids=...)`, so the writer can follow §10.6.
- Tests: OBB-only and mixed mask/OBB samples over several visits; stable IDs; oriented volume; the
  CLI exits 0.

### WP-H Spec, codes, docs and release notes

- **Spec 1.0:** §13.2 (F01), §10.1 (F06), Appendix B (F26), and Appendix C entries.
- **Spec 1.1:** §7.3/§10 (F02), §9 step 3 (F15), and Appendix A entries.
- **task-cache-1:** §7.2 and §7.4 (F03).
- **`crates/medh5/data/codes.json`:** add E704, E506 and T407. The docs hook renders the tables,
  and the code-table test checks the spec against the table.
- **Conformance corpus:**
  - Add cases for F01, F06 and F15.
  - Add F07 too, if damage cases can patch bytes.
  - Update the stated counts.
- **`medh5/_core.pyi`:** add any new fields or methods; stubtest checks it in CI.
- **CHANGELOG, Behaviour changes:**
  - F01: aliased files are refused.
  - F06: unit mismatches are refused, and the writer's default changes.
  - F14: non-integer numbers are refused.
  - F19: volume units change.
  - F20: seeded draws change.
  - F23: nnU-Net labels are relabeled.
  - F26: the `extra` namespace changes.
- **Non-blocking observations:**
  - Fix the version-bump runbook: document updating the `=2.0.0` pins in `Cargo.toml:28-30`, or add
    a bump helper.
  - Decide the target-status policy for planned and cancelled outcomes, in the spec.
  - H01–H03 can follow the release.

## 5. Decisions to make before implementing

| # | Question | Recommendation |
|---|---|---|
| 1 | F01: refuse aliased attested paths, or version `content_id` to commit the bindings? | Refuse; no address changes |
| 2 | F22: refuse unlabeled cases, or route them to `imagesTs`? | Refuse by default; `unlabeled="test"` as an opt-in |
| 3 | F23: remap to consecutive labels, or refuse gaps? | Remap, report the mapping, and record it for the round trip |
| 4 | F04: extend a stale stash, or refuse it? | Extend scalar stashes; refuse region-based ones |
| 5 | F14: accept exactly integral floats, or refuse them? | Refuse, as the clinical parser does |
| 6 | F15: should plans get a window with a lower bound only? | Yes, and clarify the spec |
| 7 | F06: refuse unit mismatches, or define a conversion? | Refuse, and derive the writer's default from the grids |
| 8 | F19: always report mm³, or keep native units plus a unit field? | mm³ in `relative_change`, with the unit carried on observations |
| 9 | F26: always key `extra.legacy` per timepoint, or only when studies are grouped? | Always, so there is one shape |
| 10 | F26: grouped studies with different `nnunetv2` metadata: keep the first and report, or refuse? | Keep the first and report |

## 6. Gate before merge

1. Every P1 counterexample becomes a regression test that fails at `ca0634b` and passes after its
   fix. Every P2 is either fixed or documented as a limit of the supported scope.
2. Re-run the audit bundle's reproducers (`audit-probes/`) against a fresh build, keeping native
   failures inside their subprocess wrappers.
3. Run the full pre-commit gate from CLAUDE.md, then CI on the replacement head: native and wheel
   conformance, Windows damage tests, MSRV, minimum dependencies and the worker tests.
