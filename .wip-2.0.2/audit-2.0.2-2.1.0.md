# medh5 audit: what to fix in 2.0.2 and what to add in 2.1.0

**Audited:** `main` at `1959938` (release 2.0.1 plus one Unreleased release-notes fix), 2026-10-11.
**Method:** 27 parallel reviews, one per area (every crate, the Python package, converters, torch/MONAI, the docs, the specs, CI/release, tests, performance, security, frontend parity, roadmap). Then 3 adversarial verifiers re-ran the 18 most consequential claims against a local release build. **Review only:** nothing in the repository was changed. The build lived in a scratch virtualenv, and every test file was written to scratch.
**Revision 2 (same day): cross-checked against an independent audit.** A third-party audit of the same code (v2.0.1 `9cc6574`; its runtime code is identical to `1959938`) reported 22 findings, F01–F22. Nine verifiers re-ran every one of them on the local build. **All 22 reproduced.** About half reach further than the audit stated (F01, F05, F07–F11, F20, F21). Two are narrower: for F02 only *crossing* XOR contours are wrong, and F03 affects hard links only. We rate F02 and F07 lower than the audit does. The cross-check also exposed problems in this audit: two wrong fixes (strided Jacobian ROIs, the clinical-slot channel count), one severity rated too low (cross-file instance ids), six "checked and fine" notes that were incomplete, and one mis-described performance item (the window read). Section 7 has the finding-by-finding table and the differences between the two audits. New items are tagged (F01)…(F22) in sections 2–4.
**Companion file:** `medh5-audit-appendix-area-reports.md` holds every area report verbatim (about 390 findings, each with location, evidence and fix), the three verification reports, and, as of revision 2, the nine cross-check reports.

---

## 1. Baseline: the project is in good shape

| Check | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets -D warnings` (rustc **1.97.0**) | clean |
| `cargo test --workspace` | all pass |
| `pytest tests/` (no torch/MONAI installed) | **2568 passed, 36 skipped**, 4m13s. 34 skips are torch/MONAI; 2 are POSIX-permission tests that cannot run as root (`tests/format/test_writer.py:530`, `tests/tools/test_dataset.py:175`). Revision 1 said "all skips are torch/MONAI", which was wrong by those 2. |
| `pytest tests/` with torch 2.14.1 and MONAI 1.6.1 (revision 2) | **2723 passed, 2 skipped** (the same two root-only tests), 3m35s. Nothing deselected: the 7 DataLoader-worker tests the independent audit had to deselect (its sandbox refused AF_UNIX sockets) pass here. |
| Line coverage | 83% without torch/MONAI; **94.8%** with them (7628 statements, 394 missed; the independent audit measured 94.82%, 395 missed). CI enforces 90%, and the floor holds. |
| Conformance corpus through Python | 157/157 |
| Diagnostic code table | all 95 codes emitted somewhere, all exercised by the corpus, severities taken from the table |
| Re-verification | **18 of 18** claims reproduced (16 fully, 2 partially; several with corrected line numbers or severity) |
| Independent audit cross-check (revision 2) | **22 of 22** findings reproduced by 9 verifiers. Two are narrower in scope, about half are broader, and the audit's line numbers match this checkout. See section 7. |

The reviewers' "checked and fine" sections are long, covering the window read path, canonical JSON, the digest stream, the box↔slice convention, coverage semantics, two-time selection, atomic writes, the handle cache, the release workflow and more. The problems are concentrated where the tests don't reach:

1. hostile-file handling: a crafted file can crash any process that opens it, or make HDF5 load a plugin;
2. de-identification key matching, and identifiers stored where the sweep never looks (aliases, link names, nested types);
3. silent data loss on uncommon but documented inputs, and on output paths that collide with inputs;
4. validator robustness on malformed files, including an integrity verdict of "ok" for a file none of whose digests parses;
5. semantic queries that skip a rule the spec states: implicit closure, spatial coverage in tracking, class and instance identity across files;
6. converter geometry and contour semantics (2-D NIfTI planes, DICOM through-plane drift, RTSTRUCT contour types and per-ROI frames);
7. per-class re-reads on the label path;
8. the newest surfaces (task/cache, PairedPatchDataset, the 1.1 spec text).

Severity counts across all reports, before de-duplication: about 20 high, 115 medium and 230 low. Several findings were reported independently by two reviewers; this document merges them. Revision 2 adds the independent audit's 22 findings. 7 of them were already here in some form, though narrower or lower-rated (F02, F09, F10, F13, F19, F20, F22). 1 was only in the appendix, at low severity (F17). 14 are new.

---

## 2. Top issues: fix these first

✔ = independently re-verified against the build. "Reproduced" = the area reviewer reproduced it at runtime. Line numbers are as corrected by verifiers.

### Security: before anything else
| # | Issue | Where | Severity | Release |
|---|---|---|---|---|
| S1 | **A crafted file crashes any process that opens it.** `open_read` always runs the self-containment check through HDF5's recursive `H5Lvisit`, which has no depth limit, and each callback resolves the full path from the root. Results by nesting depth: <br>• 30,000 nested groups: `medh5.open()` dies with **SIGSEGV** ✔ (reproduced twice, independently) <br>• 2,000: open takes 1.9 s <br>• 10,000: 62 s and 326 MB <br>Every door (CLI, `validate`, collections, task/cache, DataLoader workers) opens through it, so "validate an untrusted file" crashes instead of reporting. | `crates/medh5/src/h5/file.rs:24-30`; `h5/ops.rs:533-553, ~330-370` | **high** (DoS) | 2.0.2 |
| S2 | **A file naming an unknown filter id makes HDF5 load and run shared libraries from its default plugin search path.** <br>• That path is baked in at build time: the build directory, then `/usr/local/hdf5/lib/plugin`. On Windows, by HDF5's CMake config, it includes `%ALLUSERSPROFILE%\hdf5\lib\plugin`, which standard users can create by default. That is a DLL-planting route to code execution as whoever opens a crafted `.medh5`. <br>• Blosc2 and Zstd are registered statically, so medh5 never needs dynamic plugins. <br>• Reproduced on Linux: a planted library's constructor ran on read. The baked-in path is visible with `strings` on the built extension ✔. | HDF5 initialisation (`h5/mod.rs`, `crates/medh5-sys/src/lib.rs`); no `H5PL*` call anywhere | **medium-high** | 2.0.2 |
| S3 | **Zstd decompression bomb.** The filter allocates the frame's declared content size, or doubles its buffer without bound. A 32 KB chunk drove 1 GB RSS; a 4 MB chunk can demand about 128 GB. The vendored Blosc2 filter's non-B2ND branch likewise trusts the header's `nbytes` and ignores `blosc2_cbuffer_sizes`'s return (likely). | `crates/medh5-sys/src/zstd_filter.rs:41-117`; `crates/medh5-sys/vendor/hdf5-blosc2/blosc2_filter.c:614-633` | medium | 2.0.2 |

### Correctness, privacy, performance, process
| # | Issue | Where | Severity | Release |
|---|---|---|---|---|
| 1 | **Scrub, amend, recompress and fix on a symlinked path replace the link with an independent copy and leave the real file untouched.** For `scrub --apply` this leaves the identifying original in place while the user believes it was scrubbed. ✔ | `crates/medh5/src/h5/file.rs` (`AtomicFile`: temp derived from the unresolved target, `fs::rename(tmp, target)` ~232-247) | medium-high (privacy) | 2.0.2 |
| 2 | **Scrub misses identifiers, and some misses are attested as done:** <br>• dead underscore keys (`nhs_number`, `birth_date`, `date_of_birth`, `*_uid`) ✔ <br>• top-level `extra`/`acquisition` keys (`PatientID`, `AccessionNumber`) skip key rules ✔ <br>• UIDs inside strings ✔ <br>• whole dates under unlisted keys never shifted <br>• `date_shift_days=0` is recorded as a shift and silences every later date scan ✔ | `crates/medh5/src/curation/scrub.rs` (tables 62-63/149-152/166-168 vs `normalise` 350-352; 895-904; 765-812; 136-153; 880, 1894-1915) | high (aggregate) | 2.0.2 |
| 3 | **`add_segmentation(encoding="mask")` ORs every class into one mask and writes `class_ids=[]`, `annotated_class_ids=[]`**, and the file validates clean (strict, 0 warnings). ✔ | `sample/writer_annotations.rs:361-363`; `annotations/select.rs` (`prefer` returned unchecked) | high | 2.0.2 |
| 4 | **DICOM SEG import keeps only the last of several segments that share a label** (one "Lesion" segment per lesion is common). Half the voxels are lost with 0 warnings. ✔ | `medh5/io/dicom_seg.py:690, 775-794` | high | 2.0.2 |
| 5 | **0.x migration:** every review record the 0.x API wrote (`annotator`/`timestamp`) fails migration with E005, and `pending`/`flagged` become `reviewed`. ✔ The sample-level `label`/`label_name` classification is dropped unless it matches a mask or box name, contrary to Appendix B. ✔ | `convert/legacy.rs:850-885, 252-296, 796-826` | high | 2.0.2 |
| 6 | **Validator blind spots on malformed files:** <br>• a `/meta` group or dangling link, or an `images` dataset, validates clean at the default level ✔ <br>• (F09) a dangling `/images` or `/images/<id>` passes all four levels, **including strict** (via F01's early return), and so do string/compound images, which `read()` then refuses; `fix --rewrite-digests` restamps such a file and it passes strict with zero codes ✔ <br>• one corrupt chunk aborts the integrity pass and hides every other E701/E702 and the dataset name ✔ <br>• §6.2 MUST attributes (`task`, `closure`, `class_ids`) unchecked <br>• stored `ignore_id` not range-checked (`0` makes all background "ignored") <br>• `bit_class_ids`/`layer_class_ids` unchecked, so false *verified negatives* | `validate/rules/container.rs:68-76,155-158,172-174`; `rules/images.rs:22,28,44-56`; `rules/mod.rs:158-175` (`children()` drops unresolvable members); `integrity/verify.rs:355-367`; `rules/annotations.rs:93-127, 249, 483-495`; `annotations/header.rs`, `layout.rs` | medium-high | 2.0.2 |
| 7 | **Label reads do per-class work:** <br>• `dense()` on a `labelmap` re-reads the window once per class (60 classes: 86 ms vs 8 ms) <br>• every training item re-reads the window for an ignore region the file doesn't have (+55–90% label cost) <br>• `bbox`/`occupancy`/`analyse`/index build iterate per voxel with `IxDyn` and per class (10–20× slower than the read; 8-class index build 2.9–5.3 s) | `annotations/read.rs:672-676, 731-741, 360-364, 885-887`; `sample/reader.rs:555-558`; `annotations/select.rs:108-123,146-173`; `storage/index.rs:94-145` | high (perf) | 2.0.2 |
| 8 | **Cohort stats with a NaN voxel** (PET, resampled data): min/max depend on file order and can drop a finite extreme. The stats JSON writes `null` and reloads as `mean 0, std 0`, so normalisation silently becomes (0, 1). ✔ | `dataset/stats.rs:45-126`; `json.rs:410` | medium | 2.0.2 |
| 9 | **Task/cache:** <br>• a cache stored away from the task's directory validates as entirely stale (two reviewers) <br>• rows in `error` add spurious T406 <br>• `CacheWriter` commits manifests its own reader rejects (T401) <br>• duplicate `entry_id`s hide a wrong checksum <br>• `build_document_cache` encodes a changed source under the old pin | `companion/cache.rs:272-365, 427-442, 633-648, 770-791`; `companion/view.rs:1238-1241`; `medh5/cache.py:384-407, 477-500` | medium | 2.0.2 |
| 10 | **Torch loaders:** <br>• `PairedPatchDataset(label=)` serves one pair's change label as the target of every pair <br>• `annotation=` fails on every pair not starting at that visit (the longitudinal guide's own example) <br>• (F20) `boxes` annotations crash with `AttributeError` in *every* label format and in MONAI `to_dict` ✔ <br>• (F13) a mixed present/empty optional clinical slot cannot be batched, even with fixed patches when the slot is multichannel ✔ <br>• (F12, F21) see the revision 2 table below | `medh5/torch/datasets.py:674-678, 693-694, 739-741, 304-306, 324, 349`; `medh5/monai.py:347`; `medh5/torch/clinical.py:547-578` | medium | 2.0.2 |
| 11 | **Writer state:** a refused commit (e.g. E201) aborts the writer, and a second `commit()` returns `None` as if it had succeeded, with no file (two reviewers). | `sample/writer.rs:1318-1327, 372-377`; `crates/medh5-python/src/writer.rs:1254-1257` | medium | 2.0.2 |
| 12 | **Spec text does not pin `content_id` bytes:** <br>• the float rule's literal condition leaves out zero (mitigated by the `-0.0` example and the json.dumps reference) <br>• spelling of the root/object paths, the meta-line bytes and the line sort order <br>• `blake2b` output size <br>• the class form hashed by `label_set.sha256` <br>• 1.0 vs 1.1 disagree on whether a later minor may extend covered attributes | `docs/spec/medh5-1.0.md:512-518, 1290-1343, 86, 494-502`; `medh5-1.1.md:68-70` | medium-high (interop) | 2.0.2 (errata) |
| 13 | **The conformance scorer ignores severity**, so a validator that reports every E-code as a warning (accepts every invalid file) scores 157/157. ✔ | `conformance/suite.rs:205-212`; `conformance/mod.rs:166-190` | medium | 2.0.2 |
| 14 | **Free-threaded builds claim to be GIL-free without an audit.** PyO3 0.29.3 emits `Py_MOD_GIL_NOT_USED` when `gil_used` is absent, so a from-source 3.14t build runs `_core` without the GIL. Mutable pyclasses raise borrow errors, and the Python handle cache is check-then-act. ✔ (PyO3 source) | `crates/medh5-python/src/lib.rs:44` (`#[pymodule]`) | medium | 2.0.2: `gil_used = true` |
| 15 | **Two read-count regression tests cannot fail.** They count `h5py.Dataset.__getitem__` calls, but the package no longer uses h5py, so the 1.x "190 reads per draw" regression could return unseen. ✔ | `tests/tools/test_sampling.py:326-349`; `tests/format/test_readers.py:533-564` | medium (test) | 2.0.2 |
| 16 | **Release builds use moving inputs:** `stable` Rust plus `clippy -D warnings`, re-run on the tag; newest maturin/cmake; `curl \| sh` rustup in the container. There is no RustSec/`cargo deny` scanning of the ~125 statically linked crates, and no CI caching (HDF5 is compiled from source in about 11 jobs per push). | `.github/workflows/ci.yml:38,68,303`; `release.yml:237`; `.github/scripts/build-dist.sh:18-30`; `dependabot.yml` | medium | 2.0.2 / 2.1.0 |

### Added in revision 2: the independent audit's findings, reproduced here
Ranked by our severity. Findings that were already in the tables above (F09, F10, F13, F19, F20) are merged there or into section 3. The audit's own priority is in brackets.

| ID | Issue | Where | Severity | Release |
|---|---|---|---|---|
| F01 [P1] | **Integrity and strict validation report ok for a file in which no digest parses.** An early return drops the E703 (malformed digest), E702 (`content_id`) and W905 (stale index) diagnostics it has already computed. Examples: a third-party writer that spells the algorithm differently (`md5:`, `SHA256:`), or a damaged attribute on a one-image sample. `medh5 validate --level strict` exits 0 on a file whose pixels changed, while `verify` fails it. The corpus misses it: its E703 case also carries a valid digest. | `crates/medh5/src/validate/rules/integrity.rs:60-62`; `integrity/verify.rs:345-353`; `conformance/build.rs:1735-1742` | **high** | 2.0.2 |
| F14 [P2] | **`closure = "implicit"` is stored but never applied by any reader.** A classification asserting only `left_kidney`, with `kidney` examined, answers `state(kidney) = negative`: a false negative for exactly the pattern §9 recommends. Voxel `dense([kidney])` and `contains` omit the descendants. `LabelSet.close()` implements the closure but no reader calls it. | `annotations/read_geometric.rs:745-751`; `annotations/read.rs:589-688`; `labels/mod.rs:614`; spec 1.0 §5.4, §9 | medium-high | 2.0.2 (+ §9 erratum) |
| F15 [P2] | **Tracking reports a lesion `resolved` where the follow-up's ignore region covers it.** This holds even when the ignore region covers the whole volume. Coverage is class-level only: it never consults the ignore region or `valid_mask`. The same root cause produces false `new` lesions when the baseline ignores the region. | `curation/tracking.rs:255-262, 269-275, 477-499`; spec §7.4 vs §7.7/§11.3 | medium-high | 2.0.2 (+ §7.4 erratum) |
| F17 [P2] | **Cross-file object agreement matches by `instance_id` whenever any id is shared.** Two raters who each number from 1 get F1 = 1 and `mean_iou` = 0 for disjoint lesions; the threshold is ignored. This is the normal case, not an edge case. | `curation/agreement.rs:494-501, 539-566`; `one_sample` at `:361-363` | medium-high (our appendix had low/2.1.0) | 2.0.2 |
| F02 [P1] | **RTSTRUCT import ignores `ContourGeometricType`:** <br>• crossing `CLOSEDPLANAR_XOR` contours are unioned (63 voxels instead of 54); nested ones are already right <br>• `OPEN_PLANAR` lines are filled as areas <br>• `OPEN_NONPLANAR` applicators are flattened onto one plane, and the class is marked examined with 0 voxels, a fabricated negative <br>• POINT ROIs vanish without a note <br>• export writes `CLOSED_PLANAR` for everything | `medh5/io/rtstruct.py:65-74, 189-203, 293-306, 396-404, 618` | medium-high (we rate it P2: ordinary closed contours are fine) | 2.0.2 skip-and-report / refuse; open curves need a §8.6 change (2.1.0) |
| F03 [P2] | **A hard-link alias of `/meta` (e.g. `x_acme/meta`) keeps the MRN after `scrub --pseudonymise-ids`.** Apply returns ok, and a fresh strict scan is clean. Amend skips `/meta` by path, not by object, so the alias carries the old document across. A soft-link alias is safe. | `sample/writer.rs:460-462, 1345-1346`; `curation/scrub.rs:1712, 1805-1843` | medium | 2.0.2 |
| F04 [P2] | **Strict scrub never sees five kinds of identifier carrier:** later hard-link names, soft-link names, attributes and names of committed datatypes, nested compound string fields, and compound fields holding string arrays. All scan clean before and after apply. | `curation/scrub.rs:1231-1245, 1345-1356, 1385-1391`; `h5/ops.rs:204-242` | medium | 2.0.2 |
| F11 [P2] | **`--report` or `--out` naming an input or an output overwrites it, and the command exits 0.** This applies to every converter and to native `migrate`. `from-dicom-seg`/`from-rtstruct --report case.medh5` destroys the only copy of the sample just amended. `migrate -o .` replaces the 0.x original. `clinical export S -o S` overwrites the sample. | `medh5/cli.py:129-131, 184-193`; `crates/medh5-cli/src/convert.rs:80-90`; `crates/medh5-cli/src/clinical.rs:~237` | medium | 2.0.2 |
| F08 [P2] | **Re-exporting to nnU-Net into the same directory leaves the old files:** <br>• removed cases <br>• dropped channels <br>• a case moved to `imagesTs` stays in `imagesTr`, which is train/test leakage <br>• both file endings <br>`dataset.json` disagrees with the inventory. nnU-Net lists `imagesTr`, and medh5's own `from_nnunetv2` brings the removed case back. | `medh5/io/nnunetv2.py:572-666, 117-129` | medium | 2.0.2 refuse unplanned files; replace mode 2.1.0 |
| F05 [P2] | **A 2-D NIfTI plane loses its z offset** (z=50 exports as z=0). Two planes at different z, or an image and a mask from different slices, are accepted onto one grid. | `medh5/io/nifti.py:848-854, 101-102`; test `tests/integrations/test_nifti.py:594` locks the loss in | medium | 2.0.2 refuse |
| F06 [P2] | **RTSTRUCT applies the first file-level frame to every ROI.** An ROI's own `ReferencedFrameOfReferenceUID` is never read. With frames [A, B] the file is accepted silently; with [B, A] it is refused whole. An ROI naming another frame than the file's is accepted. | `medh5/io/rtstruct.py:54-57, 83-88, 133-137, 230-241` | medium | 2.0.2 |
| F16 [P2] | **Cross-file agreement and cohort stats treat label-set-local class ids as global.** "liver" in A is scored against "spleen" in B (Dice 1), and `compute_stats`/`dataset stats` merge them into one class that feeds `class_weights`. `dataset check` does flag the mismatch (C101/C102), but only if someone runs it. | `curation/agreement.rs:265-342, 490`; `dataset/stats.rs:193-219, 407-419` | medium | 2.0.2 refuse; key mapping 2.1.0 |
| F18 [P2] | **`box_iou` of a 2-D box against a 3-D box returns about 10**, and `compare_instances` reports it as a match. A box with zero spatial axes scores 1.0. | `curation/agreement.rs:66-84, 426-433`; `geometry/grid.rs:424-426`; `crates/medh5-python/src/curation_tools.rs:270-283` | medium | 2.0.2 |
| F12 [P2] | **A clinical task slot takes the ignore regions of every voxel annotation on the grid**, including annotations that supply and examine none of the requested classes. An unrelated organ annotation with a whole-grid ignore region removed all 8 lesion positives from the loss mask, while the label, `annotated` and preflight all looked correct. The non-clinical loaders and MONAI key ignore regions per annotation and are not affected. | `medh5/torch/clinical.py:590-607`; `crates/medh5/src/companion/view.rs:662-664` | medium | 2.0.2 (filter contributors); per-class ignore 2.1.0 |
| F07 [P2] | **DICOM stacking bounds each gap at 1% of the median, never the accumulated position error.** A 401-slice series with its middle slice 4.5 mm (1.8 voxels) out is accepted as uniform. Conversely, positions rounded to 2 decimals at 0.625 mm are refused. | `medh5/io/dicom.py:52, 366-388` | low-medium (we rate it P3) | 2.0.2 |
| F22 [P3] | **String datasets break the generic `Dataset` interface.** `read()` and `np.asarray(ds)` raise "holds strings, not numbers", though the docs and the stub promise both. `ds[...]` and `ds[:,]` raise; on a 2-D string dataset `ds[0]` returns one element, not the row. | `crates/medh5-python/src/nodes.rs:262-305` | low-medium | 2.0.2 (was 2.1.0) |
| F21 [P3] | **A warm handle-cache lease keeps serving a replaced file's old, pinned bytes; a cold one raises T302.** In one DataLoader epoch, worker 0 served the old snapshot and worker 1 refused the same row. task-cache-1 §6 promises refusal once the source no longer carries the pin. No wrong data is served, but runs fail nondeterministically after an amend. | `medh5/torch/handles.py:65-75, 190-207`; `docs/spec/task-cache-1.md:307-309` | low | 2.0.2 (a stat identity check per lease) |

---

## 3. 2.0.2: recommended scope (patch: bug fixes, docs, performance; no API or format change)

Checklists, grouped by workstream; the most important items in each come first. Engine paths such as `h5/file.rs` are relative to `crates/medh5/src/`. Paths starting `medh5/`, `crates/`, `docs/`, `tests/` or `.github/` are relative to the repository root.

### W0. Untrusted-file security. Ship first.
- [ ] Replace `H5Lvisit` in the self-containment check with medh5's own iterative, depth-capped walk: `MAX_DEPTH = 64` already exists at `h5/ops.rs:177`. Use `H5Literate` per group and resolve each link relative to its parent, not by full path from the root. Add a deep-chain regression test, and extend the damaged-file CI harness (`.github/scripts/damaged.py`) with structural hostile inputs. (S1) ✔
- [ ] Turn off dynamic filter plugins at HDF5 initialisation (`H5PLset_loading_state(0)`). Re-enable them only on explicit opt-in (`HDF5_PLUGIN_PATH` set, or `MEDH5_ALLOW_PLUGINS=1`). At minimum, `H5PLreplace` the default search path so no build-prefix or `%ALLUSERSPROFILE%` entry ships in wheels or binaries. (S2) ✔
- [ ] Cap filter output at the chunk size HDF5 passes:
  - in the Zstd filter's reverse path, refuse a declared or streamed size larger than `*buf_size`;
  - do the same in the vendored Blosc2 filter's non-B2ND branch, and check `blosc2_cbuffer_sizes`'s return. (S3)
- [ ] Remove `blosc2_destroy()` from the Blosc2 filter's failure path (`crates/medh5-sys/vendor/hdf5-blosc2/blosc2_filter.c:658`; another `medh5:` patch). It tears down global Blosc2 state, contexts and mutexes, while window reads decompress outside the HDF5 lock with the GIL released. This is a possible use-after-free (unverified).
- [ ] `FeatureCache.get` reads the whole entry before comparing its shape and dtype with the cache header (`companion/cache.rs:470-505`). A hostile cache can make every `get()` allocate up to physical memory. Check first.
- [ ] Also in this workstream, though listed elsewhere:
  - the symlink replace (top issue 1);
  - unchecked `data::nbytes` multiplication on file-stated shapes (W3);
  - the zstd cap that the CLI review flagged independently.

### W1. De-identification (privacy). Ship with W0.
- [ ] Resolve symlinks before an atomic replace, or refuse them: temp file beside the *real* file, rename over the real path (`h5/file.rs` `AtomicFile`). Document it in §14.4 and the amend/scrub docs. ✔
- [ ] Store the scrub key tables already normalised, and add a test that `entry == normalise(entry)` for every entry (`curation/scrub.rs:62-63, 149-152, 166-168`). ✔
- [ ] Route top-level `extra`/`acquisition` keys through `entry()`/`key()`, as `identity()` does (`scrub.rs:895-904`). `deidentify.md:52` already promises this. ✔
- [ ] Refuse `date_shift_days == 0` in engine, CLI and Python, and treat a recorded 0 as "not shifted" (`scrub.rs:880`). ✔
- [ ] Shift any value that is *wholly* a date regardless of key, e.g. `RadiopharmaceuticalStartDateTime`, which today survives a strict apply with `ok=True` (`scrub.rs:136-153, 789-792`). Recognise UIDs embedded in text (`text()` 765-812, reuse `has_uid_token`). ✔
- [ ] (F03) Aliases of `/meta`: amend copies `/meta` by path, so a hard-link alias keeps the original document and its MRN after `--pseudonymise-ids` ✔. In the writer, record every hard path to the source `/meta` object and re-link each one to the new `/meta` after commit (`sample/writer.rs:460-462, 1345-1346`). In scrub, report any non-canonical object that is `/meta`, or that decodes to a document with an `identity`, as an actionable identity finding. Refusing aliased metadata is acceptable but leaves plain `amend` splitting the alias. Our "sweep for replaced values" item below catches the copy only during the same apply run, never in a fresh scan.
- [ ] (F04) Name and type coverage. Scan every link name, including later hard links and soft links, with a walk that expands each group once so cycles terminate; `ops::visit` deduplicates objects, which is right for digests but wrong for names. Scan the attributes and names of committed datatypes. Make string-type detection recursive through compound, array and vlen types, with a depth cap that reports `unreadable` instead of passing (`scrub.rs:1231-1245, 1345-1356, 1385-1391`) ✔. `deidentify.md:41-44` promises "every object name, every HDF5 attribute".
- [ ] Report timepoint dates added *after* a recorded shift (`scrub.rs:1001-1003`). Today they are invisible to every later scan.
- [ ] Report non-string datasets outside the imaging groups as "not examined", and list them in `not_checked` (`scrub.rs:1344-1360, 431-434`).
- [ ] After `--pseudonymise-ids`, sweep every string for the replaced values (`scrub.rs:1847-1882`).
- [ ] Route strict staff/organisation pseudonyms through `pseudonym()` (salted, recorded in `uid_map`). Scan the `deidentification` record and agents' `organization` (`scrub.rs:1112-1122, 1737-1740, 924-925, 1124`).
- [ ] Refuse an explicit `date_shift_days` that differs from the shift already recorded (`scrub.rs:1677-1686`).
- [ ] Python `scrub_paths`: continue per file like the CLI, so `uid_map`s of files already rewritten are never lost. Take explicit keywords, so `apply=True` (a typo for `apply_changes`) cannot silently become a scan (`medh5/curation/scrub.py:316-332`). It has no test today.
- [ ] `remap_frame_uids` should set `compact`, so the commit drops old UID bytes from freed space (`sample/writer.rs:789-821`). This also lets scrub drop its second full-file `repack` and that step's failure window. ✔ (Scrub's own output is clean today because of that repack.)
- [ ] Legacy migration must not write the input's absolute path into `identity.subject_id`/`sample_id` (`convert/grouping.rs:226-229`; `legacy.rs:439-447, 503`). Local paths often contain MRNs or names, and `content_id` then depends on the machine. Use the stem or a digest, with the same rule in `migrate()` and `migrate_paths`.
- [ ] `docs/guides/deidentify.md:82-94`: the suggested publication gate passes clinical samples and open identity findings, because they are non-actionable. Block on them, or say so.

### W2. Silent data loss and wrong results
**Annotations and writer**
- [ ] `add_segmentation(encoding=…)`: accept only `auto` plus transcodable kinds, and call `check_target` (`writer_annotations.rs:361-363`). Make `select_encoding` refuse an unknown `prefer`. ✔
- [ ] `transcode_payload`/`check_roundtrip` (public as `medh5.annotations.voxel.transcode_payload`): apply the writer's refusals (dense→`instances`, in-band ignore), and compare ignore regions (`annotations/encode.rs:990-1026`). ✔
- [ ] Range-check a stored `ignore_id` (1..65535, not a class id) at open and in the validator (`annotations/header.rs`; `validate/rules/annotations.rs:249,520`). Two reviewers found this.
- [ ] Check `bit_class_ids` (unique, equal as a set to `class_ids`, shape `(C,)`) and `layer_class_ids` (rows equal planes) at open and in the validator (`annotations/layout.rs`; `read.rs:~1046`; `rules/annotations.rs:483-495`).
- [ ] `payload_to_masks` panics (Python `PanicException`) on inconsistent hand-built payloads. Run the shape table first (`encode.rs:~799, 814, 823`).
- [ ] (F14) Apply `closure = "implicit"` in the semantic readers, as §5.4 requires ("all `is_a` ancestors of a listed class also hold") ✔:
  - `Classification.state`: a class is positive when it or any descendant is asserted > 0 in the same scope unit (`read_geometric.rs:745-751`). Keep `value()`, `labels()` and `asserted_class_ids` literal.
  - `dense_class`/`dense`/`contains`: OR the descendants' planes (`read.rs:589-688`; `LabelSet::descendants`, `labels/mod.rs:602`), inside `dense_class`, so every encoding inherits it.

  `explicit` stays as is (readers MUST NOT infer there). A missing `closure` silently defaults to `explicit` (the E412 item in W3), so both fixes are needed.
- [ ] Refuse non-finite box coordinates (use `!(lo <= hi)` plus `is_finite`) in the encoders, `bad_boxes` and `box_to_slices` (`encode_geometric.rs:~175`; `rules/geometric.rs:179`; `geometry/affine.rs:196`). Do the same for affine matrix entries (`transforms/encode.rs:22-45`; `rules/transforms.rs:156-166`) and pyramid factors (`multiscale.rs:65`). ✔ for boxes. (F18) Also make `box_iou_f64` and the Python `agreement_box_iou` fallible: refuse unequal or zero row counts, `!(lo <= hi)` and non-finite bounds, and never clamp (`agreement.rs:66-84`; `curation_tools.rs:270-283`).
- [ ] `dense([0])` answers differently per encoding. Refuse class 0 as 65535 is refused (`read.rs:388`).
- [ ] Smaller items:
  - `instances_from_masks` emits objects `encode_instances` refuses;
  - `encode_mesh` offsets are unchecked until commit;
  - `boxes` without `instance_ids` get 0-based ids, and NaN scores are not filtered;
  - `has_ignore_region` disagrees with `ignore_region`.

**Converters (Python)**
- [ ] SEG import: OR (binary) or max (fractional) segments that map to one class, or refuse. Record a decision (`dicom_seg.py:690`). ✔
- [ ] `to_dicom_seg` and `to_nifti(class_key=)`: export `annotated_class_ids` by default, and refuse an unexamined class (`dicom_seg.py:924,935`; `nifti.py:1446`). Today unexamined classes become empty segments, i.e. verified negatives on re-import.
- [ ] (F11) Output-path collisions. Before any handler runs (`medh5/cli.py:129-131`):
  - gather each command's input and output files;
  - refuse a `--report` or `--out` that is the same file as any of them, by resolved path and by `samefile`, and a report inside an input root or the nnU-Net dataset directory;
  - in `_finish` (`:184-193`), refuse a report that aliases `report.outputs`, and write it through a temp file and `os.replace`;
  - for native `migrate`, check the report path first and refuse a `--report`/`--write-labels` path that is an input (`crates/medh5-cli/src/convert.rs:80-90`), and refuse a target that is one of its sources (`legacy.rs:501-503`);
  - guard `clinical export S -o S` too (`crates/medh5-cli/src/clinical.rs:~237`).

  Today every converter exits 0 after replacing the named file ✔.
- [ ] (F08) nnU-Net export into an existing dataset: before the first `mkdir`, list the NIfTI files already under `imagesTr`/`labelsTr`/`imagesTs`, and refuse, writing nothing, when any is not in the new plan (`medh5/io/nnunetv2.py:572-666`). An identical re-export stays allowed and unrelated files are untouched. Today removed cases, dropped channels, re-partitioned cases and old file endings survive under a new `dataset.json` ✔. A `--replace` staging mode is 2.1.0. `docs/guides/export.md:111-112` promises "no half-written dataset behind".
- [ ] RTSTRUCT import: report or refuse contours that are off-slice, out-of-grid, non-axial, `OPEN_PLANAR` or POINT (`rtstruct.py:65-74, 294-299, 383-386`). (F02) In detail ✔:
  - read `ContourGeometricType` per contour, and treat a missing one as `CLOSED_PLANAR` with a recorded guess;
  - skip ROIs made only of POINT, `OPEN_PLANAR` or `OPEN_NONPLANAR` contours, with one warning per ROI and type, so no class is marked examined with 0 voxels; refuse an ROI that mixes open and closed contours; report the `<3`-point drop at `:73`;
  - for `CLOSEDPLANAR_XOR`, refuse a mix with other types, as the standard does. Accept contours that are nested or disjoint, which import correctly today, and refuse crossing ones by naming the ROI and plane. Use a segment-intersection test, since vertex-in-polygon misses a "plus" shape;
  - in `to_rtstruct`, warn when two contours of one class cross on a plane: medh5 means their union, while XOR and even-odd readers cancel the overlap.

  Representing open curves faithfully needs an open/closed flag in §8.6 (2.1.0).
- [ ] (F06) RTSTRUCT frames: read each ROI's `ReferencedFrameOfReferenceUID` (`rtstruct.py:54-57`). Use the one frame the contoured ROIs share, and fall back to the file-level list only for ROIs that state none. Refuse with E414, before opening the sample, when ROIs span frames, listing which ROI is in which frame. Warn when an ROI's frame is missing from the file-level list, match grids on the ROI frame (`_match_grid`, `:432`), and keep the salted comparison (`_require_same_frame`). Today the first file-level frame is applied to every ROI (`:83-88, 133-137, 230-241`), so the outcome depends on list order ✔.
- [ ] (F05) 2-D NIfTI: in `_reduce_plane` (`nifti.py:848-854`), refuse a plane whose discarded world coordinate is nonzero (> 1e-3 mm after unit scaling), and tell the user to save the image as `(x, y, 1)` to import it as a one-slice volume, which already keeps z. Today z=50 becomes z=0, and two planes, or an image and a mask from different slices, share one grid ✔. Change `tests/integrations/test_nifti.py:594`, which asserts the lossy round trip, to use z=0. Reword the "tilted in 3-D" refusal of sagittal planes to "not parallel to the x-y world plane".
- [ ] (F07) DICOM slice stacking (`dicom.py:366-388`):
  - estimate the spacing as `(last − first)/(n − 1)` or by a least-squares fit, not by the median of gaps;
  - bound the through-plane residual of every slice against `origin + k·spacing·normal` with `tol_z = max(0.1·spacing, 0.01 mm)`, and keep the half-pixel in-plane bound;
  - keep a per-gap test only to name a missing slice (a gap above 1.5× the spacing);
  - report `max_through_plane_error_mm` in the `slice_spacing` decision.

  This refuses accumulated drift, as in the 401-slice case 4.5 mm out, and accepts 2-decimal rounding, which today's 1% per-gap rule refuses ✔. **Refuse; never resample:** the audit's "or explicitly resample" contradicts "geometry is never invented". highdicom 0.28.2 uses the same per-gap rule. Align with the SEG reader's absolute per-plane test and document one policy.
- [ ] `from_dicom`: skip non-image SOP classes and multi-frame objects, and record per-series refusals as warnings, so one RTDOSE or localizer series doesn't abort the whole tree (`dicom.py:185-213, 757-760`). Wrap pixel-decoder errors.
- [ ] `read_dicom_seg`: don't invent a 1 mm through-plane step for a one-plane SEG (`dicom_seg.py:587-588`); geometry is never invented.
- [ ] 5-D NIfTI `(x,y,z,1,n)`: never take timing from the squeezed axis (`nifti.py:536-538, 829-830, 899-924`).
- [ ] `to_rtstruct`: write `SpecificCharacterSet`, label-set colours and display names (`rtstruct.py:568-631`). Non-ASCII PatientID aborts `from_dicom` (`sanitize_stem`).
- [ ] `import_seg_nifti`: replay `read_nifti` notes, accept `assume_geometry`, fix the provenance tool name. `read_nifti` on Analyze/MGH should give a clean refusal.

**Legacy migration**
- [ ] Read 0.x review records as 0.x wrote them (`annotator`, `timestamp`). Map `pending`/`flagged` explicitly and report the mapping as a guess. Never write an activity with no agent (`legacy.rs:850-885`). ✔
- [ ] Migrate the sample-level `label`/`label_name` classification per Appendix B (`legacy.rs:252-296, 796-826`). ✔
- [ ] `parse_date` panics on non-ASCII, reachable from `group_by_subject` (`convert/grouping.rs:108-121`).
- [ ] Study mode with `--subject-key` gives every study the same `sample_id` (`legacy.rs:503`). `output_name`'s `starts_with("study")` misfires on real ids like `Study-7` (`grouping.rs:325-333`).
- [ ] Read 0.x ≤0.5 root `shape`; keep a non-object `extra`; don't leave decision notes for groups that failed.

**Geometry and transforms**
- [ ] `Transform::chain` / `ChainTransform` must check that frames connect (`transforms/model.rs:295-338`). Today a wrong-order chain silently maps points.
- [ ] Pyramid origin check: measure against voxel size, not |origin| (`geometry/multiscale.rs:179-187`). It misses a missing half-voxel shift whenever |origin| is more than about 500× the spacing (whole-body CT, µm stages).
- [ ] `jacobian_determinant(roi)` / `folding_fraction(roi)`: resolve slices like `read_field`. Strided ROIs are silently wrong (`model.rs:771-778, 1108-1131`). (F19) **Revision 1's fix was wrong.** "Multiply the spacing by the step" corrects world-space vectors but leaves index-space vectors exactly as wrong as today, because det(I + S·M) = det(I + M·S). The verifier showed this algebraically and reproduced it numerically. For 2.0.2:
  - refuse a ROI with more than S axes, and pad a shorter one with full slices;
  - resolve each slice with `Slice::resolve(n)`;
  - refuse step ≠ 1 with the wording of `annotations/read.rs:527`;
  - give a clear error for an axis with fewer than 2 voxels or an empty ROI;
  - pass the resolved (start, stop) pairs to `cropped_grid`.

  Today `np.s_[:-1]` raises E109 for shape (-1, 8, 8) and `::2` reports 100% folding on a valid field ✔. If strided ROIs are wanted (2.1.0), convert index vectors with the *original* grid and differentiate with spacing × step.
- [ ] Python `box_corners`/`apply_affine_to_box`: check shape `(S, 2)`. A `(2, 3)` box is silently misread (`crates/medh5-python/src/geometry.rs`). `slices_to_box` should refuse open slices.
- [ ] Honour `chunk_hint` for stacked encodings (`storage/chunking.rs:172-178`). `recompress --profile portable` should rewrite small plugin-filtered datasets. `describe_filters` should name zstd (32015).

**Dataset, splits, stats, collections**
- [ ] Stats: count or exclude non-finite values (or refuse, naming the file), make min/max order-independent, and refuse `null` moments on reload (`dataset/stats.rs`). ✔ Also stop W907 recommending int16 for an image containing NaN (`sample/image.rs:360-369`).
- [ ] Manifest paths depend on the CWD, and `root` is recorded but unused (`dataset/manifest.rs:433, 521, 549`). Resolve against `root` or the manifest's directory.
- [ ] Scan silently skips symlinked directories and `*.MEDH5`, and picks up `._*` (`manifest.rs:490-500`).
- [ ] The `splits` audit ignores `fold`, so a subject in two k-fold folds is not reported as a leak (`curation/splits.rs:340-364`).
- [ ] `extract(shard, key, out=shard)` replaces the whole collection with one member (`collection.rs:444-461`). Refuse when it is the same file. (F10) The same happens through a relative or symlinked alias, and in `unpack` when a member's output name is the collection itself (`out/a.medh5` unpacked into `out/`), from Python and from `medh5 unpack` ✔. Compute every `unpack` destination after the case-collision loop (`:431`) and check each with `same_file` *before the first write*.
- [ ] (F16) Cohort stats bin by numeric `class_id` alone, so two label sets where id 1 means "liver" and "spleen" merge into one class that feeds `class_weights` (`dataset/stats.rs:193-219, 407-419`) ✔. Record each class's key in `stats_for` (the manifest already holds `class_keys`). Refuse, or report in `failures` and make `class_weights` refuse, when one id carries two keys, and add `key` to `ClassStats` and its JSON (additive).
- [ ] C201 counts claims, not samples (`check.rs:353-358`). `write_claims` should refuse a manifest the split wasn't made from. Sampling accepts a misspelled annotation name for uniform draws (`sampling.rs:289-292`).

**Curation semantics: tracking and agreement**
- [ ] (F15) Tracking: an absent object is `resolved` only when some annotation at that visit examined its class *and* either has no ignore region or has one that provably misses the object's last observed location. "Provably misses" means the location is mapped through the shared frame or `transform_between`, then through `box_to_slices`. Otherwise the object is `unexamined`, an existing state, so no API is added. The simplest safe patch is to treat any ignore region on the covering annotation as `unexamined` (`curation/tracking.rs:255-262, 477-499`). Today even a whole-volume ignore yields `resolved` ✔. Apply the same rule to `is_new` (`:269-275`), which reports false new lesions when the baseline ignores the region. Update the `medh5 track` legend and `longitudinal.md:124-125`. Field-of-view (`valid_mask`) checks and an `indeterminate` state with reasons are 2.1.0.
- [ ] (F16) Agreement across files: before scoring, check that each compared id has the same class *key* on both sides, the rule C102 uses (`check.rs:270-276`); equal label-set digests short-circuit the check. Resolve string keys on each side and refuse when the ids differ. Make a mismatch an error, not a `skipped` entry (`agreement.rs:265-342`, `:490`). Keep numeric comparison when neither side has a label set. `medh5 agree` compares within one sample and cannot reach this; Python and Rust can ✔.
- [ ] (F17) Object agreement: use `instance_id` matching only when `one_sample(a, b)` (`agreement.rs:494-501`), as our appendix proposed. Do not widen it to equal `sample_id`, because raters' copies of one case share it and still allocate ids independently. Today two raters numbering from 1 get F1 = 1 for disjoint lesions, and ids {1,2} vs {1,3} on identical boxes give F1 = 0.5 ✔. Report `threshold: null`, or apply it, for id-matched pairs (`:539-566`).
- [ ] (F18) Dimension checks: `compare_instances` raises E414 when the two sides have different spatial dimensions, and `Grid::comparable_with` requires equal S (`geometry/grid.rs:424-426`, also exposed at `crates/medh5-python/src/geometry.rs:339`). The `box_iou` refusal is under "Annotations and writer" above.

**Clinical (1.1)**
- [ ] Asserting events: the code requires the asserter to be *admitted* (kinds, window and limit filters), but 1.1 §7.3 says *eligible* (`clinical/select.rs:627-650`). Under the default policy an unknown-time asserter never attests, so `roi=eligible_instances` falls back to the centre. Decide which is intended, and align the spec, `design-rationale.md:243` and the code.
- [ ] `select` on a projection (later minor) reports `certified` (`crates/medh5-python/src/clinical.rs:138-142`; `crates/medh5-cli/src/clinical.rs:187-200`).
- [ ] E816 checks only neighbouring versions. A duplicate `supersedes` row is reported as branch+merge. One cycle suppresses the chain check for unrelated records. E805 blames the wrong columns when the first column is the outlier (`check.rs:597-643, 676-679`; `columns.rs:340-370`).

**Torch / MONAI**
- [ ] `PairedPatchDataset(label=)`: attach the label only when `ann.timepoints == (first, second)`, and refuse non-classification annotations (`datasets.py:674-678`).
- [ ] `PairedPatchDataset(annotation=)`: use the annotation only at its own visit and auto-select elsewhere. Fix `longitudinal.md:163-176` (`datasets.py:693-694, 739-741`).
- [ ] Refuse non-voxel annotations by name (they crash today), including in `medh5.monai.to_dict`, where the `isinstance` check comes after `labelmap()` (`datasets.py:304-306, 324, 349`; `monai.py:347`). (F20) Every label format crashes, not only `instances`: `onehot` fails on `.dense` and `labelmap` on `.labelmap` ✔. Add the named refusal right after the annotation lookup (`datasets.py:288`) and before `labelmap()` in `to_dict`, and document the voxel-only contract. In 2.1.0, `instances` could accept boxes: carry world boxes into the read grid's index space, refuse what cannot be carried, then apply the existing edge-convention patch intersection.
- [ ] Clinical optional slots (F13) ✔:
  - Take the placeholder's channel extent from the task, not the row. **Revision 1 said "from the modality", but a modality string such as `MR` does not fix a channel count**, and "the first observed row" depends on read order.
  - In 2.0.2, derive the extent once in `ClinicalTaskDataset.__init__` from the preflight fills of *all* rows of the task, reading grid metadata only; refuse inconsistent extents, naming the slot and images; use 1 when no row fills the slot. Only shapes cross rows, never values (`torch/clinical.py:547-557, 577`).
  - In 2.1.0, add an optional declared `channels` on `Slot`, which changes the task-cache normal form (a contract-version note) and is checked by preflight.
  - For `patch=None`, the collate error should say "give the slot a `patch` or use batch_size=1", not "sample fixed-size patches".
- [ ] (F12) Clinical slot ignore: in `_slot`, skip the ignore region of any annotation that neither labels nor examined a requested class (`torch/clinical.py:590-607`) ✔. With several requested classes the union over contributors is still conservative, and can discard independent supervision; document that in `docs/reference/torch.md:298`. Per-class ignore `(B, K, *patch)` is 2.1.0. Optionally filter `label_annotations` by class in preflight (`companion/view.rs:662-664`), so the `RowView` agrees.
- [ ] (F21) Handle-cache leases ✔:
  - record `(st_dev, st_ino, st_size, st_mtime_ns)` when `_acquire` opens a key, stat before and after `_open`, and reopen on a difference (`handles.py:65-75`);
  - in `lease`, when the cached handle matches the pin, stat the path again; if its identity changed, take the existing stale path, so T302 is raised, or a repack with the same `content_id` is served. One `stat` per lease (`handles.py:190-207`);
  - task-cache-1 §6 and `docs/reference/torch.md`: a lease in progress finishes on the bytes it opened; every new lease re-checks the source;
  - the non-clinical datasets lease without a pin, so after an amend one epoch can mix versions of a file. Document it; the same stat check would let them reopen.
- [ ] Smaller items:
  - `HandleCache.clear()` in the owning process leaks every handle (`handles.py:388-394`);
  - `_compat` turns real torch import errors into "not installed";
  - `timepoint=` is ignored when `images=` is given;
  - MONAI meta lacks `filename_or_obj`;
  - `collate_clinical` raises a bare `ValueError`.

**Python API**
- [ ] Writer states Open/Committed/Aborted: `commit()` after abort or failure raises, and checks that run before anything is changed (E201, references) don't kill the writer (`sample/writer.rs:1318-1327`).
- [ ] `CacheWriter.__exit__` commits unconditionally, so `commit()`/`abort()` inside `with` raises (`medh5/cache.py:240-249`).
- [ ] `recompress_paths`/`fix_paths` stop at the first failure and lose the results of files already rewritten (`storage/recompress.rs:411-413`; `integrity/repair.rs:229-231`). Return per-path outcomes, as the CLI does.
- [ ] `diagnose`/`fix` ignore `malformed` digests: `verify` fails while `fix` says "nothing to fix" (`integrity/repair.rs:19-46, 93-109`). Two reviewers found this. It is a sibling of F01, not the same bug: `Diagnosis` has no `malformed` field. Fix both.
- [ ] (F22) String datasets through the generic `Dataset`:
  - `read()` and `np.asarray(ds)` raise "holds strings, not numbers", although the `medh5/nodes.py` docstring, `docs/reference/python-api.md:248` and the stub (`read() -> NDArray`) promise them;
  - `ds[...]` and `ds[:,]` raise;
  - on a 2-D string dataset `ds[0]` returns one element, not the row;
  - a scalar string dataset ignores its key ✔.

  Fill `read_strings` into an object array of the stored shape (0-d for a scalar), return it from `read()` (which fixes `__array__`), and index that array in `__getitem__` (`crates/medh5-python/src/nodes.rs:262-305`). Since this restores documented behaviour, it moves from 2.1.0 (section 4.2) to 2.0.2.
- [ ] `fix --rebuild-index` rebuilds with default `max_coords`/occupancy/seed instead of the values stored on the index entry (`repair.rs:182`).
- [ ] `Image.read(physical=True)` returns the stored dtype when there is no rescale; the docstring says float32 (`sample/image.rs:295-306`).
- [ ] Amend switches `digest_algo` to sha256, and with `digests=False` leaves mixed algorithms that still validate (`sample/writer.rs:1355-1358`).
- [ ] `storage.dataset_kwargs` changed meaning from 1.x (h5py keywords → `{"chunks","codec"}`) without a CHANGELOG note, and it ignores a custom `CodecProfile` (`medh5/storage.py:156-174`).
- [ ] Small items:
  - `repr()` of a closed `Sample`/`Image` raises;
  - `Packed[-1]` returns `""`;
  - NumPy integers are rejected as clinical times;
  - `SourceRef()` defaults give an invalid `""` or colliding stem ids;
  - `Tracking[...]` raises a non-`KeyError`;
  - `add_mask` coerces a float mask (0.2 → True) silently;
  - `bench` accepts `repeats=0`;
  - `rules_for("bogus")` returns every rule;
  - `PatchSampler` revalidates only in `__init__`;
  - `scan(on_error="warn")` never warns;
  - `sample_stride=0` is accepted;
  - `SampleDocument.loads` decodes invalid UTF-8 lossily.

### W3. Validator and integrity robustness
- [ ] (F01) **Delete the early return at `validate/rules/integrity.rs:60-62`.** When no digest in the file parses, it discards the E703, E702 and W905 already computed, so `validate --level integrity|strict` passes a file that `verify` fails ✔. Keep W901 gated as at `:63`. Add a corpus case `E703-every-digest-malformed`; the existing E703 case also carries a valid digest, so it cannot catch this. Then update the stated counts that `TestStatedCounts` checks. Add a test that validation and `verify` agree on every digest case.
- [ ] Check node *kinds* (E004 for a non-dataset `meta`; E008 for non-group `grids`/`images`/`annotations`) (`rules/container.rs:68-75,155-158`; `rules/images.rs:22`). ✔ (F09) Also:
  - report a dangling image member as E001 (`images.rs:28` skips it today);
  - report a non-numeric image or pyramid level as E001; §4.2 says "any numeric", but no E2xx code covers dtype;
  - make `Context::children()` report members it cannot resolve instead of dropping them, so dangling annotations and transforms stop being invisible to every rule (`rules/mod.rs:158-175`).
- [ ] Per-dataset error handling in the integrity pass: report the unreadable dataset by path and keep checking (`integrity/verify.rs:355-367`). ✔
- [ ] Per-object error handling in `check_images`/`annotations`/`transforms`/`probmaps`, as `rules/mod.rs:3-7` promises. Use `is_group` where grid and transform ids are resolved. Report non-group members in `check_geometry`.
- [ ] Emit E412 for missing §6.2 MUST attributes `task`, `closure`, `class_ids` (`rules/annotations.rs:93-127`).
- [ ] Read `/meta` and `clinical/meta` bytes strictly as UTF-8 (E004/E801), and report a non-string `meta` as E004, not E001 (`rules/container.rs:173`; `rules/clinical.rs:254`).
- [ ] Skip cross-reference checks when `/meta` didn't parse (avoids E409/E413 cascades). W911 is suppressed by any external-frame transform. Use `version::parse` and stop after E002.
- [ ] Missing file or directory: stat first. Today the raw HDF5 error stack, including a timestamp and a pointer, makes `validate --json` output nondeterministic (`h5/file.rs:25-28`).
- [ ] `close_everything` ignores a failed final `H5Fclose`, so a commit could rename a file whose close failed (`h5/file.rs:219-233`). `data::nbytes` multiplies file-derived shapes without overflow checks (`h5/data.rs:784-794`). `recreate_fixed_strings` can lose the original on a mid-swap failure.
- [ ] `check_clinical_records` clones every clinical table; use `take()` (`rules/clinical.rs:368`).

### W4. Task-and-cache companion
- [ ] Resolve cache entry URIs against the task's base when a task is given, and state the base in task-cache-1 §7.1 (`companion/cache.rs:648`; `medh5/cache.py:384-407`; CLI `clinical.rs:435-447`).
- [ ] Don't add T406 for rows whose preflight is `error`; report one T404 (`view.rs:1238-1241`; `cache.rs:770-791`).
- [ ] Schema-check entries in `CacheWriter::add`/`commit` (`cache.rs:272-365`). Refuse duplicate `entry_id`s, event keys and row ids at open (`cache.rs:427-442`), and add the rule to §7.1.
- [ ] `build_document_cache`: check `content_id` against the pin before encoding (`medh5/cache.py:477-500`).
- [ ] Smaller items:
  - a missing cache file is reported as "corrupt" T401;
  - `file://` URIs are prefix-stripped only (`localhost`, percent-escapes, Windows drives);
  - `TaskManifest::save`/`task reconcile` write non-atomically over the input;
  - the binding reads a non-integer `cutoff_us` as absent.
- [ ] Document now that the default `min_follow_up_us = horizon_us` makes negative labels practically unreachable. Change the default in 2.1 (`task.rs:166-169`; `view.rs:861, 892-900`).

### W5. CLI
- [ ] `dataset stats` exits 0 with part of the cohort unreadable, and prints failures on stdout (`crates/medh5-cli/src/dataset.rs:251-255`).
- [ ] `index build`/`seg convert` rewrite the file, with a new inode, even when nothing changes. `seg convert` transcodes twice; short-circuit `from == to`. `index build` with nothing to index exits 1 silently (`seg.rs:163-185, 226-250`).
- [ ] Use one failure shape (`ok`, `error`) for batch `--json` output across `fix`/`index build`/`labels`/`splits`, and send unreadables to stderr (`inspect.rs:528`; `seg.rs:237`; `labels.rs:96`; `curation.rs:350`).
- [ ] `verify` text output omits unverifiable datasets. `fix` exits 0 with an index still stale, and drops notes (`inspect.rs:433-458, 489-532`).
- [ ] Flags silently ignored without their partner:
  - `fix --reason/--by` without `--rewrite-digests`;
  - `split --fold/--assigned-by` without `--write-claims`;
  - `scrub --date-shift-days/--salt/--by` without `--apply`.
- [ ] Other CLI fixes:
  - `agree --threshold` accepts NaN, negatives and values above 1;
  - help goes to stdout for bare `medh5` but to stderr for groups;
  - Python-API wording appears in CLI messages;
  - `migrate --report DIR` is checked only after converting;
  - the NativeHost probe hides Python's stderr and accepts a 1.x `medh5`;
  - `PyStream` is unbuffered.

### W6. Performance (no API change)
- [ ] Label path:
  - `dense()` on labelmap: read once and fill every class plane in one pass;
  - `labelmap()` on layers/bitmask: read each plane once;
  - cache `encodes_ignore`, and skip the in-band ignore read when the annotation has none;
  - return the ignore region from the block already decoded (`annotations/read.rs`; `sample/reader.rs:555-558`). Measured gains are 10× for labelmap onehot and 55–90% for every training item.
- [ ] Index, bbox, analyse: work on `Ix3` or flat offsets instead of `IxDyn` `indexed_iter`. Fuse count, bbox, coords and occupancy into one pass. Read each plane once (`storage/index.rs:94-145`; `select.rs:108-173`). The writer runs `analyse` twice for `layers`: pass `colouring` through (one line, `writer_annotations.rs:361-373`). `analyse` copies every mask and is quadratic in classes (11 s at 200 classes). `compare_voxel` decodes per class (`agreement.rs:362-364`).
- [ ] `dataset stats` reads plane by plane, decompressing each chunk once per plane (0.60 s vs 0.03 s): step by the chunk extent (`dataset/stats.rs:443-453`). Digest slabs are not chunk-aligned (each chunk decoded 2–4×) (`integrity/digest.rs:189-208`).
- [ ] `Split.partition_of`/`fold_of` rebuild the whole split per call, so the documented loop takes about 34 s at 3k subjects. Cache a map, and point docs at `split.paths()` (`medh5/dataset/split.py:85-93`).
- [ ] Re-reads:
  - `BoxesAnnotation.boxes` and the other geometric columns re-read the dataset on every access; use the handle's cached columns (`medh5/annotations/geometric.py:74-124`);
  - `polygons()` re-reads all vertices per polygon (`read_geometric.rs:533-560`);
  - `instance(id)` decodes every mask (`read.rs:1302`);
  - cubic displacement fields re-prefilter on every evaluation (`transforms/apply.rs:354-390`);
  - torch `instances` format decodes every object mask per patch.
- [ ] Release the GIL in the remaining long native calls:
  - annotation encoders, `transcode`, `annotation_to_masks`;
  - `add_mask`, `tracks(measure=True)`, `IndexHandle` reads;
  - storage `build_index`/`occupancy`;
  - transform sampling and Jacobians;
  - integrity helpers;
  - dataset `find`/`manifest_stale`;
  - `CacheWriterHandle.add`/`commit`, `ClinicalHandle.records`.
- [ ] Chunk optimiser:
  - clamp the starting chunk to the 4 MiB ceiling (patch 96 → 8 MiB chunks, 160 → 32 MiB);
  - read L3 per sharing CPU (`storage/chunking.rs:28-31, 104-105`);
  - or correct the docs, which promise 512 KiB–4 MiB and per-core L3.
- [ ] Quadratic and repeated work:
  - `anatomy_units` (2.5 s at 40k groups);
  - check/pack `Vec` lookups;
  - `CacheWriter.add_row` builds an `Event` per selected event;
  - `TaskManifest` fingerprints re-parse per access;
  - `len(FeatureCache)` builds every entry;
  - the event sort clones ids per comparison;
  - `attrs::has` lists all names (use `H5Aexists`);
  - the window path re-opens the file id and queries HDF5's file size for every chunk, under the lock (`h5/window.rs:108`). Revision 1 called this a "re-stat". It is HDF5's cached end-of-file, not an OS `stat`, so it costs lock time but cannot notice a replaced file (see F21);
  - redundant copies on image reads.
- [ ] Release profile: add `strip = "symbols"` (about 5 MB of symbols in the 30 MB extension).

### W7. Documentation corrections
- [ ] Guides: the "check it" step `--level strict` fails on every freshly converted file (W903 and W912 promoted) ✔. Recommend `--level integrity`, or explain (`import-nifti.md:106`, `import-dicom.md:127`, `segmentation.md:92`, `migrate-0x.md:92`, `detection.md:75`, `README.md:188`).
- [ ] Revision 2 doc changes that ship with the fixes above:
  - `deidentify.md:41-44` promises "every object name, every HDF5 attribute"; make it true (F04) or qualify it;
  - `import-dicom.md`: state the slice-position tolerance as a physical bound (F07);
  - `import-nifti.md:25-35`: a 2-D plane must lie at z = 0, otherwise save it as `(x, y, 1)` (F05);
  - `export.md`: say what an existing nnU-Net destination does (F08);
  - the converters reference: which RTSTRUCT contour types import, and that crossing contours mean union (F02);
  - `longitudinal.md:124-125` and the `medh5 track` legend: "examined at this visit, outside any ignore region, and not found" (F15);
  - `docs/reference/annotations.md` and the `Classification.state` docstring: what `closure` means to `state()`, `dense()` and `contains()` (F14);
  - `docs/reference/curation.md:116`: id matching applies within one sample only (F17);
  - the `Transform.jacobian_determinant` docstring: what `roi` accepts (F19).
- [ ] `transform_between(tp, tp)` is described wrongly: it returns `None` at the first shared-frame pair, depending on grid order (`longitudinal.md:221-223`; `registration.md:255-257`; code `sample/reader.rs:756-768`).
- [ ] `segmentation.md:76-87`: `instances` → dense is refused without `--drop-identity`. `partial-labels.md` contradicts its own example. `performance.md:35-52`: `used_index` is a list after collate, and "L3 per core" is wrong.
- [ ] Clinical guides:
  - state the setup command (`python docs/examples/clinical_longitudinal.py cohort`);
  - have `clinical.md` amend a copy, since it damages the cohort the next guide uses;
  - show a `SelectionPolicy` with `context_us`/`plans`;
  - document the `TextEncoder` contract (`encode`, `dim`, `header`);
  - correct `Link.between` (keyword-only, `target_annotation_id`).
- [ ] `design-rationale.md:213` promises a "C-compatible native library", but there is no C ABI. `design-rationale.md:258` and `export.md:118-122` promise "lossless round trips to DICOM", but there is no DICOM image exporter. List what has no exporter: images to DICOM, boxes, keypoints, meshes, transforms, clinical.
- [ ] Reference:
  - `reference/index.md:20` miscounts the annotation kinds;
  - commit is described as `os.replace` (it is fsync, rename, then fsync of the directory);
  - prefix abbreviation of long flags is undocumented;
  - `rust.md` and the crate README omit public modules and put annotations at §5–§9 (should be §6–§9);
  - `curation.md:114`: `matched` holds indices, not ids;
  - the "views raise after close" claim is only partly true;
  - `MEDH5Error` is "base of every error" but HDF5 I/O errors are plain `OSError`.
- [ ] Site:
  - pin `pymdown-extensions` and `Markdown`, which supply the slugs the `--strict` anchor check relies on;
  - drop the unused `pymdownx.snippets`/`inlinehilite`;
  - exclude `examples/*.medh5` from the docs build;
  - set `site_url` from `READTHEDOCS_CANONICAL_URL`.
- [ ] Front door:
  - link the 1.1 spec and task-cache-1 beside 1.0;
  - reconcile CHANGELOG 2.0.0 perf numbers with `performance.md`;
  - CLAUDE.md's pre-commit chain lacks the native corpus run and `third_party_notices.py --check`;
  - the tutorial's `labelmap()` warning is unmentioned;
  - the hero snippet's names don't match the page's sample;
  - `medh5/__init__.py:20-21` still says "1.0 ships a reader".
- [ ] Code docs:
  - `write_claims` docstring describes a fixed bug as current;
  - `written_version` omits the `drop_clinical` exception;
  - `check_roundtrip` says A→B→A;
  - `select.rs` cites task-cache §4 for the policy (it is §3.4);
  - medh5-sys says "three things" and lists five;
  - amend docs assume POSIX rename semantics (on Windows, close readers first).
- [ ] Bundled `brats-subregions` vocabulary: id 4 is a synthetic "tumor" parent, but in BraTS ≤2021 label 4 is enhancing tumour. Document the 2021→2023 relabelling.

### W8. Spec errata (2.0.2: wording that matches what the reference implementation already does)
- [ ] 1.0 §5.1: "when `x` is zero or `1e-4 ≤ |x| < 1e16`". Add `0.0`/`-0.0` examples.
- [ ] 1.0 §13: the root attribute line's `obj` is empty. `object_path` is relative to the sample root with no leading `/`. The meta line hashes `/meta`'s stored bytes. `/meta` carries no `digest`. Lines sort by UTF-8 bytes. `blake2b` means BLAKE2b-512. Define the class form `label_set.sha256` hashes (omitted null or empty members, order of `parents`/`codes`).
- [ ] Pick one rule for later-minor covered attributes: 1.0 §13.2 says "never covered", 1.1 §2.2 says "may cover".
- [ ] Revision 2 errata (all of them match the fixes in W2/W3):
  - §9: "a class in `annotated_class_ids` with no entry is a negative" conflicts with "ancestors follow from the label-set DAG" under `implicit`. Add "unless, under `closure = "implicit"`, a descendant is asserted positive (§5.4)". In §5.4, state that readers MUST answer state and membership queries for ancestors under `implicit`, while stored assertions stay literal (F14).
  - §7.4: "resolved is distinguishable from unexamined only through `annotated_class_ids`" contradicts §7.7 and §11.3, where an ignore region is the official way to say "not examined here". Make it "through `annotated_class_ids` and the annotation's ignore region" (F15).
  - §4.2: images are numeric ("any numeric"); say MUST, so a string or compound image is an error (F09).
  - §3.6: say which two world axes a 2-D grid's `coord_system` refers to; `RAS_TO_LPS_2D` assumes the first two (F05).
- [ ] 1.0 §2.1 still has the pre-erratum "MAY read a higher minor"; point it to §16 and 1.1 §2.2.
- [ ] Other errata:
  - `annotated_class_ids` is MUST except `mask`, matching the validator;
  - §7.1 "no ignore region" means no voxel holds `ignore_id`;
  - §2.5 integer widths;
  - §10 "never share a frame" vs §3.4;
  - E702's summary should include the uncovered-digest case;
  - require finite box coordinates and affine entries (E406/E504).
- [ ] 1.1 / task-cache-1:
  - tie groups are connected components of the overlap relation;
  - §3.5 patch placement around the centre (start = c − ⌊p/2⌋, shifted inside the extent, padded only when an axis is shorter);
  - §5 negative-label bound test;
  - §7.1 entry URI base and uniqueness;
  - §9 needs a real code table, not just the placeholder;
  - the 1.1 §8 "any context ≥ 72 h" claim holds only for `closed`.
- [ ] `conformance.md`: say whether severity is scored, and fix the scorer so E-codes must be errors (W8 ties to item 13). Say that 1.1 reclassifies some 1.0-declared files (E009/E803 vs a pure 1.0 validator's E007). Cite §2.2, not §2.1, for collections.

### W9. Tests
- [ ] Replace the two h5py read-count tests with engine-level counts (a Rust test, or a debug counter in `_core`).
- [ ] Windows: add a read-only-amend test. The engine sets the temporary file read-only and *then* opens it for write to fsync, and `MoveFileEx` refuses a read-only target (`h5/file.rs:232-247`, `fsync_path`, `set_mode`); the Python test is skipped on win32. Fsync before `set_mode`, and clear the target's read-only bit before the rename.
- [ ] Test the `_live.py` mutation methods: 60% covered. `pop`/`popitem`/`clear`/`update`/`setdefault`/`|=`, and the list `insert`/`remove`/`sort`/`+=` etc., must each write through or refuse a stale anchor.
- [ ] `test_docs_python.py` passes any `SystemExit`, including a failing gate. Test names `test_S16_amend_keeps_a_later_minor` and `…the_scrub_and_fix_doors_are_the_same_gate` no longer match what they check. Half of the L3-probe test patches Python functions the Rust probe never calls. The fork test for `HandleCache.get` passes even if inherited handles were closed.
- [ ] Add a regression test for every confirmed bug above. Most have none today: zero date shift, symlink replace, same-label SEG, `encoding="mask"`, 0.x review dict, stats NaN, commit-after-failure, cache outside the task dir, etc. The cross-check verifiers named a test for each F-finding (appendix sections 32–40).
- [ ] Test relationships, not just successful calls. This is the independent audit's main process point, and it would have caught most of F01–F22:
  - integrity validation agrees with `verify`;
  - physical placement survives conversion (NIfTI z, DICOM positions);
  - semantic queries respect hierarchy and vocabulary identity;
  - adding an unrelated annotation cannot erase requested supervision;
  - resampling a linear field cannot change its analytic Jacobian;
  - a second export has exactly the second plan's inventory;
  - no command's output path may alias its input.

  Promote the corrupt-container cases (dangling `/images`, wrong-kind nodes, all-malformed digests) into the conformance corpus, so both command lines are held to them.
- [ ] Speed: two tests take 34 s of the 4m13s run (`test_bench.py:102`, `test_packaging.py:217`). Wall-clock bounds in Rust unit tests run on unoptimised Windows and macOS runners.
- [ ] Read engine output with an older HDF5 (1.10–1.14) in CI. The spec promises "HDF5 ≥ 1.10", but the engine links HDF5 2.2.0 with `libver_bounds(V110, latest)`, and the test suite's h5py 3.16 bundles HDF5 2.0.0. Alternatively, cap the upper bound.

### W10. CI and release hygiene
- [ ] Pin the Rust toolchain (`rust-toolchain.toml`, bumped on purpose), maturin and cmake in `build-dist.sh`, and optionally the manylinux image digests.
- [ ] Advisory gates (both audits agree): a scheduled and release-time `cargo audit`/`cargo deny` over `Cargo.lock`, and `pip-audit` over the extras. Also keep a native inventory (HDF5 2.2.0, zlib 1.3.2, C-Blosc2 3.3.5, LZ4 1.10.0, Zstd 1.5.7) with a reasoned disposition per advisory, because RustSec cannot see vendored C. The first entry is CVE-2026-85091:
  - it is a heap overflow in zlib's `gz_vacate`, and zlib 1.3.1.2–1.3.2 is affected, so the bundled 1.3.2 is in range;
  - it is not reachable: `nm` on the built `_core` shows no `gz*` symbol, only `compress2`/`uncompress`/`deflate`/`inflate`;
  - scanners keyed on the version will still flag the wheels, so record a VEX "not affected" statement and bump `libz-sys` once a fixed zlib ships.

  `dependabot.yml` covers GitHub Actions only.
- [ ] Add `Swatinem/rust-cache` to non-release jobs, `timeout-minutes` to every job, `fail-fast: false` to the Python matrix, and short artifact retention for PRs.
- [ ] Add Python 3.15 to the test matrix (`allow-prereleases` until available) and to the classifiers once green.
- [ ] A backport (e.g. 2.0.x after 2.1.0) would become GitHub "Latest" and downgrade the Homebrew tap. Compare the tag with the highest release, and pass `--latest=false` / skip the tap when it is not the newest (`release.py:322-326`; `release.yml:323-336`).
- [ ] Smaller fixes:
  - the concurrency group's comment promises every `main` run completes, but pending runs are cancelled;
  - a yanked crate version counts as unpublished, so the publish fails confusingly;
  - CI never runs the pre-commit hooks;
  - `pre-commit-hooks` is pinned to 2024.
- [ ] Write `#[pymodule(gil_used = true)]` explicitly (see top issue 14).
- [ ] Test the extras' declared floors (`torch>=2.0`, `monai>=1.3`, `nibabel>=5`, `pydicom>=3.0`, `highdicom>=0.23`), or raise them to what CI tests.

---

## 4. 2.1.0: recommended scope (minor)

### 4.1 Features, ranked by value ÷ effort
1. **Clinical-records importers** (S/M each): MEDS (optional `pyarrow` extra) and a minimal FHIR Observation/DiagnosticReport reader, following the mapping 1.1 §10 already defines (unknown availability stays null), wired into `medh5 clinical augment --from-meds`. Today the only way in is building `Event`s by hand.
2. **Task from a cohort** (M):
   - `TaskManifest.from_cohort(manifest, set_id)` takes partitions from split claims (today they are re-typed by hand);
   - `medh5 task new`, `medh5 cache ls/info`;
   - `TaskManifest.repin()` / `medh5 task repin`;
   - `Clock.utc` / `Clock.shifted_utc` and `day_bounds()` helpers, so nobody hand-rolls inclusive bounds;
   - with a task, resolve cache entries through the task's pins by `content_id`, so relocated files validate; `cache validate --task-base`.
3. **Scrub for clinical samples** (M/L; the code calls the refusal temporary, "until this tool covers it"):
   - shift the clinical clock with `--date-shift-days` (`utc` → `shifted_utc` plus the record);
   - sweep event text and documents with the deferred reader;
   - an "attest external de-identification" amendment.

   Also: scrub for `.medh5c` collections, a scrub-to-copy `--out`, and optional per-subject date offsets.
4. **Registration I/O** (M): ITK `.tfm` affine (no new dependency) and displacement NIfTI import/export, plus `medh5 convert from-itk-transform`, with frames given explicitly and never guessed.
5. **Training at scale** (M):
   - `FileGroupedSampler(num_replicas, rank)` for DDP, plus `file_groups()` for `ClinicalTaskDataset`;
   - an overlap-blending `GridPatchDataset` aggregator ("stitch back" is documented but no helper exists);
   - `medh5.monai.MedH5Reader` (an `ImageReader`) and `LoadMedH5d`;
   - `--workers` for validate, verify, index build, fix and recompress (hashing and Blosc2 decode run outside the HDF5 lock).
6. **Converters** (S–L):
   - CLI flags for coverage and codec: `--annotated-classes`, `--codec`, `--class`, `--series-description`, `--label`, `--mask-id`, `--coord-system`, `--file-ending`. Today the CLI cannot record a class that was examined but found absent;
   - FRACTIONAL SEG export for probmaps;
   - ontology codes into SEG `SegmentedPropertyType`;
   - per-slice PET rescale as an explicit, recorded decision;
   - enhanced multi-frame CT/MR/PET import;
   - nnU-Net `imagesTs`;
   - a `labelmap=` input to `add_segmentation`. nnU-Net import builds one boolean volume per class: about 13.6 GB for 104 classes at 512²×500;
   - (revision 2) RTSTRUCT open curves and points, which need an open/closed flag or a polyline/point mapping in §8.6 first (F02); a `--replace` nnU-Net export that builds in a staging directory and swaps it in (F08); an opt-in `--plane-as-volume` for 2-D NIfTI planes off z = 0, recorded as a decision (F05).
7. **Writer completeness** (S/M):
   - `remove_image`, `remove_grid`, `set_image_attrs` (each sets `compact`);
   - lost-update detection on amend (record source identity, refuse if it changed);
   - float16 displacement fields (§10.4 RECOMMENDED; `FLOAT16_SAFE_VOXELS` exists but is unused);
   - annotation pyramids;
   - `add_mask` returns its `Group`.
8. **Splits**: an incremental re-split (`make_splits(previous=…)`). Adding one subject to 100 moves 15 existing ones, 4 from test to train. A per-stratum stratification algorithm: small strata are under-filled today. Both change outputs, so version the algorithm in the split JSON.
9. **Native NIfTI converter in the engine** (L), so the Homebrew/standalone binary handles the most common conversion without Python.
10. **Docs:**
    - a third tutorial (clinical → task → preflight → batch);
    - guides for collections, provenance/quality/agreement, label sets and vocabularies, importing SEG/RTSTRUCT into a sample, and running the corpus against a third-party implementation;
    - a reference entry for every public Python name, or a test that each `__all__` name is documented or marked internal;
    - publish all four JSON schemas (only the sample schema is on the site);
    - a severity column for cohort checks (move severity into `CHECK_CODES`);
    - a concurrency/threading section.
11. **Curation semantics (revision 2):**
    - agreement across vocabularies: map B's class ids into A's by key, and an explicit `match_by="instance_id"` or id map for callers who know two files share an allocation (F16, F17);
    - tracking: an `indeterminate` state, or a reason field ("ignored", "outside `valid_mask`", "no registration"), and field-of-view checks (F15);
    - Jacobians on strided ROIs, defined as differentiation on the sampled lattice (F19).
12. **Conformance:**
    - cache fixtures for T401–T407, and a `conformance score --companion` door;
    - a `format_version` per case in `expected.json`, so a 1.0-only implementation has a defined pass criterion;
    - byte-reproducible builds (disable group/root time tracking, pin `created`);
    - derive the companion test's code list from `CODES`.

### 4.2 API changes and deprecations
- `ClassStats.prevalence` → `None`/`null` for classes nobody examined. Today it is 0.0, which contradicts the module's own §11.3 invariant.
- Default `min_follow_up_us` (task-cache 1.x clarification) and the asserter rule (eligible vs admitted), once decided.
- `transform_between(tp, tp)`: search all frame pairs, return `None` if any pair shares a frame, refuse an ambiguity (E501).
- `is_invertible` for chains and composites (true today where no inverse can be evaluated), or evaluate the reversed chain.
- Default `sample_id` and pack key: strip only the suffix. Today `1.2.840.….medh5` becomes `1` and every UID-named file collides. Deprecate first.
- `storage.dataset_kwargs`: rename to `dataset_layout`, deprecate the old name, and honour or refuse custom profile objects.
- Map HDF5 I/O errors to `MEDH5FileError` (already an `OSError` subclass), and keep `errno`/`FileNotFoundError` for ENOENT.
- `Image.read` accepts `Ellipsis`. `Instance` gets array-aware `__eq__` and `__hash__ = None`. (String `Dataset` reads moved to 2.0.2 in revision 2; see F22.)
- New diagnostics, which change the code table:
  - a warning when an assertion under `implicit` closure contradicts a positive descendant (F14);
  - a warning when a 2-D and a 3-D grid share one frame of reference (F18). Not an error, because a 2-D key image tagged with the CT's FrameOfReferenceUID is realistic converter output today;
  - optionally, a warning when a voxel annotation declares `implicit` (§5.4 SHOULD).
- `PatchSampler.draws(grid=…)` / `iter_patches` yield lazily.
- Rust crate: `#[doc(hidden)]`/`#[deprecated]` on `pub use hdf5`, `h5`, `pyval`, `json`, `numeric`, `bench`. Any hdf5-metno bump is otherwise a semver break for crates.io users. Remove in 3.0.
- Python 3.10 is EOL this month: announce deprecation in 2.1, and plan `abi3-py311` for 2.2/3.0.

### 4.3 Refactors and dead code (no behaviour change)
- **Facades that re-implement engine logic and can drift:**
  - validate `Report` (strict promotion, counts, `format`);
  - `conformance.summarize`;
  - `ClassStats.prevalence`/`to_json`;
  - `Entry.from_json`;
  - `RecompressResult`/`Diagnosis`/`Repair` helpers;
  - `task.STATUSES`/`LABELS`/`ROI`/`CENSORING`;
  - `validate.LEVELS`;
  - `document.schema`;
  - `_sample_file` in `medh5/cli.py`;
  - `file://` handling in `task.py`.

  Use `_core` instead.
- **Logic in the wrong layer:**
  - `labels check` grouping lives only in the CLI;
  - `task reconcile` is duplicated in the CLI and the binding;
  - manifest `save` writes different bytes from Python and from the engine;
  - clinical `hours()` rounds half-even in Python and half-away in Rust (1758 inputs differ by 1 µs).
- **Engine `pub fn`s with no caller** (about 22, listed in the parity report), plus `codes::code`, `scrub_paths`, the `Chains` re-export, `Deidentification.extra`, `data::filters`' unused buffer, and an unreachable arm in `validate_file`.
- **Converter helper duplication:** `_from_patient`/`_to_patient`, `_sidecar`/`_sidecar_path`, the SEG notes.

### 4.4 Platform and ecosystem
- Free-threading: audit `_core`, add a `3.14t` CI job with `PYTHON_GIL=0`, and ship cp314t wheels (abi3 cannot cover them). Track PEP 803 `abi3t`.
- New wheels:
  - musllinux_1_2 x86_64/aarch64;
  - Windows arm64 (the CFLAGS are already in `.cargo/config.toml`);
  - plan for after `macos-15-intel` (GitHub's last Intel image).
- Release provenance and dependencies:
  - build-provenance attestations for CLI archives;
  - crates.io trusted publishing;
  - `cargo deny` advisories and licences;
  - a weekly grouped Cargo.lock refresh that regenerates `THIRD_PARTY_NOTICES`.
- zstd filter: cap the decompressed size, as was already done for Blosc2 (`crates/medh5-sys/src/zstd_filter.rs:52-81, 115-124`).
- Windows: identity memo via `GetFileInformationByHandle`, so every open does not re-walk the file. On amend, preserve group ownership, not just mode.
- `cargo install medh5-cli` doesn't get the workspace's LTO profile; document it or carry the settings.

### 4.5 Security hardening (beyond the 2.0.2 fixes)
- Task-manifest source URIs may be absolute or contain `..`, so a shared manifest chooses which local paths medh5 opens. The opened bytes are still gated by the `content_id` pin, but existence leaks into findings and a FIFO blocks the run. Add an option (or make it the default) to confine relative URIs under the manifest directory and flag absolute ones (`companion/source.rs:68-76`).
- A `cargo-fuzz` target over `open_read` + `validate_file`, and over the clinical column readers. CI already damages files per platform; fuzzing would find the next S1/S3 before a user does.
- Document a threat model for third-party files: what `open` and `validate` guarantee on hostile input, and that plugins are off.

---

## 5. Should not be added (out of scope by the stated design)
- COCO import/export: it has no geometry (`refusals.md`).
- Resampling in converters, loaders or task slots: geometry is never invented.
- `rle` encoding or a `wide_labels` profile without demonstrated need; that would be a 1.2 format change if ever.
- Image-free (EHR-, text- or waveform-only) subjects: a format 2.0 question per 1.1's scope table.
- Cloud object-store/ROS3 layouts, SWMR or multi-writer append, WSI tile serving, labelling tools, encoders inside the contract: stated non-goals.
- Automatic 1.0→1.1 upgrade on amend, and clinical "edits": revisions are new events linked by `supersedes`.
- A FHIR or OMOP *serialization* of the profile: 1.1 §10 defines import semantics only.

---

## 6. How the work was split
- **Engine:** HDF5 layer and primitives; sample I/O and versioning; annotations and labels; geometry, transforms and storage; validator and code table; clinical profile; task and cache; curation, scrub and integrity; dataset, sampling and collections; legacy migration and conformance.
- **Frontends:** CLI and medh5-sys; Python core I/O; Python clinical, task and cache; Python tools; remaining bindings; torch/MONAI; converters.
- **Docs and specs:** front-door docs; how-to guides; reference and site; normative specs.
- **Cross-cutting:** performance; security; frontend parity; tests; CI and release; roadmap.
- **Verification:** 3 adversarial re-runs of the 18 highest-impact claims.
- **Cross-check (revision 2):** 9 verifiers re-ran the independent audit's 22 findings, two to four each, grouped by area: integrity and validation; RTSTRUCT; NIfTI and DICOM geometry; scrub; export and extract paths; labels and tracking; agreement and stats; transforms and nodes; torch.
  - Each verifier read the cited code and spec, rebuilt the trigger with the public writer (h5py only to plant defects), ran it through Python and the CLI, and named a regression test.
  - The coordinator re-read the code behind F01, F11, F14 and F15 and the spec text behind F14, and checked the audit's release, dependency and test-count claims.
  - The test suite was re-run with torch and MONAI installed.

The appendix file has each report verbatim, including its "checked and fine" coverage notes.

---

## 7. Cross-check with the independent audit (revision 2)

### 7.1 The other audit, and how it was checked
- **What it covers:** v2.0.1 at `9cc6574`, which has the same runtime code as this audit's `1959938`. It reports 22 findings (2 P1, 17 P2, 3 P3, using its own engineering priorities, not CVSS). It also reports:
  - its executed checks: 2,712 tests passed, 94.82% coverage, both command lines 157/157, the release workflow's 34 jobs, and source and hash comparison of the published artifacts;
  - independent numerical differentials: 7,900 HDF5 read comparisons with zero mismatches, interpolation, affine round trips, analytic Jacobians and patch tiling;
  - a dependency and native-code advisory assessment, a small preflight benchmark, and a corrective-release order.
- **How it was checked:** every finding was reproduced here from scratch, and the line numbers match this checkout. Section 6 has the method and appendix sections 31–40 the reports.

### 7.2 Finding by finding
"Here" is where the finding now sits in this document.

| ID | Finding | Theirs | Verdict | Ours → release | Here | Note |
|---|---|---|---|---|---|---|
| F01 | Integrity/strict validation drop malformed-digest failures | P1 | **Confirmed, broader:** any file where no digest parses; W905 and dangling-image E702 are lost too | high → 2.0.2 | §2 table, W3 | Delete `integrity.rs:60-62`; add a corpus case |
| F02 | RTSTRUCT ignores contour types | P1 | **Confirmed, scope corrected:** nested XOR is already right, only crossing XOR is wrong; OPEN_NONPLANAR fabricates a negative; POINT vanishes | medium-high (P2) → 2.0.2 skip/refuse; open curves 2.1.0 | §2, W2 | Ordinary closed contours are fine |
| F03 | `/meta` alias keeps the MRN after pseudonymisation | P2 | **Confirmed** for hard links; soft links are safe | medium → 2.0.2 | §2, W1 | |
| F04 | Scrub misses link names and nested strings | P2 | **Confirmed**, all five carriers plus committed-datatype names | medium → 2.0.2 | §2, W1 | Our "compound flagged unreadable" note holds only for flat compounds |
| F05 | 2-D NIfTI plane loses z | P2 | **Confirmed, broader:** a mask from another slice is accepted too | medium → 2.0.2 refuse | §2, W2 | An existing test locks the loss in |
| F06 | RTSTRUCT applies the file frame to every ROI | P2 | **Confirmed;** the outcome depends on list order | medium → 2.0.2 | §2, W2 | Their cited lines are slightly off |
| F07 | DICOM per-gap rule allows accumulated drift | P2 | **Confirmed,** plus the converse: rounded positions are refused | low-medium (P3) → 2.0.2 | §2, W2 | Refuse, never resample |
| F08 | nnU-Net re-export leaves stale files | P2 | **Confirmed, broader:** channels, labels, partitions (leakage), endings; medh5's own re-import revives removed cases | medium → 2.0.2; replace mode 2.1.0 | §2, W2 | |
| F09 | Validation accepts unreadable layouts, non-numeric images | P2 | **Confirmed, broader:** dangling images pass *strict*; `fix` restamps a string image | medium → 2.0.2 | top issue 6, W3 | Extends ours |
| F10 | Extract overwrites its own collection | P2 | **Confirmed;** also `unpack`, the CLI and aliased paths | low-medium → 2.0.2 | W2 | We had extract only |
| F11 | `--report` overwrites inputs or outputs | P2 | **Confirmed, much broader:** every converter, `--out`, native `migrate`, `clinical export` | medium → 2.0.2 | §2, W2 | |
| F12 | An unrelated ignore region erases slot supervision | P2 | **Confirmed;** clinical slots only, non-clinical loaders are fine | medium → 2.0.2; per-class 2.1.0 | §2, W2 | |
| F13 | Multichannel missing-slot placeholder | P2 | **Confirmed** | medium → 2.0.2 | top issue 10, W2 | Corrects our fix: the modality does not fix the channel count |
| F14 | Readers ignore implicit closure | P2 | **Confirmed;** §9 needs an erratum | medium-high → 2.0.2 | §2, W2, W8 | |
| F15 | Tracking resolves inside ignored space | P2 | **Confirmed, broader:** whole-volume ignore; false `new` lesions | medium-high → 2.0.2 | §2, W2, W8 | Our tracking note was incomplete |
| F16 | Class ids compared across vocabularies | P2 | **Confirmed** (Python, `dataset stats`); `medh5 agree` cannot reach it | medium → 2.0.2 refuse; mapping 2.1.0 | §2, W2 | |
| F17 | Cross-file instance-id collisions | P2 | **Confirmed;** this is the normal case | medium-high → 2.0.2 | §2, W2 | We had it in the appendix as low/2.1.0, too low |
| F18 | 2-D vs 3-D boxes give IoU > 1 | P2 | **Confirmed;** also a zero-axis box scores 1.0 | medium → 2.0.2; W code 2.1.0 | §2, W2 | |
| F19 | Strided Jacobian ROIs | P2 | **Confirmed** | medium → 2.0.2 refuse step ≠ 1 | W2 | Our revision 1 fix was wrong for index vectors |
| F20 | Box annotations crash the instance loader | P3 | **Confirmed, broader:** every label format, and MONAI | low-medium → 2.0.2 refuse; 2.1.0 support | top issue 10, W2 | |
| F21 | Lease refusal depends on cache warmth | P3 | **Confirmed;** also across DataLoader workers | low → 2.0.2 | §2, W2 | Our lease note was incomplete |
| F22 | String `Dataset` reads | P3 | **Confirmed** | low-medium → 2.0.2 | §2, W2 | Moved from 2.1.0 |

### 7.3 Where the two audits differ
**Found only by the independent audit (14):** F01, F03, F04, F05, F06, F07, F08, F11, F12, F14, F15, F16, F18, F21.
- They share a pattern: each operation succeeds, but a rule the spec or the user relies on is quietly not applied.
- The independent audit's differential method tests relationships, e.g. "validation agrees with `verify`" or "a second export has exactly the second plan's inventory". That finds what area-by-area review misses, and W9 now asks for such tests.

**Found only by this audit:**
- S1, S2 and S3:
  - S1 is a SIGSEGV on deep group nesting;
  - S2 is plugin loading from a baked-in search path, which allows code execution where that directory is writable, as on Windows `%ALLUSERSPROFILE%`;
  - S3 is a Zstd decompression bomb;
  - their malformed-input smoke tests mutated chunk frames under a 2 GiB address-space cap and 10-second limits, a setup that never builds deep group structure, never names an unknown filter, and turns a bomb into an ordinary error;
  - so their "no P0 or demonstrated code-execution exploit" holds for their search, not for the code.
- The scrub key-table misses and `date_shift_days=0`.
- `encoding="mask"`, SEG same-label merges, 0.x migration and stats NaN.
- The per-class label re-reads and the other performance items.
- The `content_id` spec ambiguities and the conformance scorer ignoring severity.
- `gil_used` and the read-count tests that cannot fail.
- CI's moving inputs, the backport "Latest" problem, and about 40 documentation corrections.

**Disagreements with the independent audit:**
- **F02** is P2, not P1. Ordinary and nested contours import correctly. The harm is in crossing XOR contours and in non-area types, and a skip-and-report patch contains it.
- **F07** is P3. A systematic same-sign bias under 1% across hundreds of slices is unusual, because scanners write absolute positions. Its fix must refuse, never "explicitly resample": converters never invent geometry.
- **F03:** the regression list implies soft aliases are affected. They are not.
- **Release order:** they say "fix F01 and F02 first". We would ship W0 (S1–S3) and F01 first, then privacy (W1 with F03/F04), then the silent wrong-result items. See 7.5.

**Corrections to this audit from the cross-check:**
- **F19:** "multiply the spacing by the step" is wrong for index-space vectors. Refuse step ≠ 1 in 2.0.2.
- **F13:** "channel count from the modality" is not enough. Derive it task-wide from the fills, or declare it on the slot.
- **Severity and release:** F17 was low/2.1.0 and is now medium-high/2.0.2; F22 was 2.1.0 and is now 2.0.2; the RTSTRUCT item was medium and is now medium-high.
- **Six "checked and fine" notes were incomplete** (appendix section and line; revision 2 only appends to the appendix, so these line numbers still hold):
  - tracking coverage (§8, line 1038), which missed the spatial coverage of §7.7 and §11.3;
  - `_reduce_plane` (§17, line 2280), which checks tilt but not translation;
  - DICOM "irregular gaps refused" (§17, line 2287), which covers per-gap irregularity only;
  - scrub's compound handling (§8, line 1023), which covers flat compounds only;
  - `HandleCache.lease` T302 (§13, line 1714), which covers cold handles only;
  - agreement coverage (§8, lines 1034-1038), which is right but runs on ids whose meaning is never compared.
- **Verifier V2's correction** that strict/integrity reports are not empty for wrong-kind nodes tested only `images` as a dataset. A dangling `/images` passes strict.
- **The window-read perf item** called `H5Fget_filesize` a "re-stat". It is HDF5's cached end-of-file, so it cannot see a replaced file.
- **The baseline** said "all 36 skips are torch/MONAI"; 2 are root-only permission tests.

### 7.4 The independent audit's other claims, checked here
- **Test counts and coverage: agree.** With torch 2.14.1 and MONAI 1.6.1: 2723 passed, 2 skipped (root-only), 94.8% coverage. Their figures are 2,712 passed, 2 skipped, 7 deselected for AF_UNIX, and 94.82%. The 7 worker tests run here and pass.
- **Native inventory: agree**, verified from sources and `Cargo.lock`: HDF5 2.2.0, zlib 1.3.2, C-Blosc2 3.3.5, LZ4 1.10.0, Zstd 1.5.7.
- **CVE-2026-85091 (zlib `gz_vacate`) not reachable: agree.** The bundled zlib is in the affected range, but `nm` on the built extension shows no `gz*` symbol. A VEX statement and a later `libz-sys` bump are now in W10.
- **No advisory gate, and Dependabot covers Actions only: agree** (W10).
- **Release engineering strengths** (exact-tag CI, SHA-pinned actions, OIDC, channel verification): consistent with our CI review. Our moving-input items (unpinned toolchain, `curl | sh` rustup) still stand.
- **Numerical differentials:** not re-run here. They agree with our reviewers' "checked and fine" notes on the window read path, interpolation bases and Jacobian composition. F19's failure is ROI handling, not the derivative.
- **HDF5 2.2.0** added optional signature checking for dynamically loaded plugins. That is a fallback for S2; turning plugin loading off stays the fix, because medh5 registers its filters statically.

### 7.5 Merged priority order for 2.0.2
1. **Can crash or run code:** S1, S2, S3 (W0).
2. **Acceptance gates that say "ok" wrongly:** F01; then F09 and the rest of W3; the conformance scorer (top issue 13).
3. **Privacy:**
   - the symlink replace;
   - scrub key misses and `date_shift_days=0`;
   - F03 and F04.
4. **Silent wrong results in data people train or report on:**
   - `encoding="mask"`, SEG same-label merges, 0.x migration;
   - F14 closure, F15 tracking, F17 and F16 agreement, F12 slot ignore;
   - F05 NIfTI z, F02 and F06 RTSTRUCT;
   - stats NaN, F18.
5. **Data loss through paths:** F11, F08, F10, the writer state (top issue 11).
6. **Everything else in W2–W10**, including F07, F13, F19–F22 and performance.
