# 2.2.0 decisions

The specification, format and platform decisions 2.2.0 implements, settled by
the coordinator. The conventions are those of
[2.1.0's decisions](../2.1.0/decisions.md): where, what, why, the text to write,
and who implements it. These assume 2.1.0 shipped as planned; a 2.2.0
workstream first checks that the 2.1.0 pieces it builds on are on `main`
(each workstream file says which).

## E1 Format 1.2 adds one kind: `polylines`

- **Where.** A new delta document `docs/spec/medh5-1.2.md`, as 1.1 is to 1.0;
  1.0 §6.3's registry gains a row; 1.0 §8.6 an informative RTSTRUCT mapping.
  W22-01.
- **Decision.** 1.2 adds the `polylines` annotation kind (open curves:
  centrelines, catheter and applicator paths, RTSTRUCT OPEN_PLANAR and
  OPEN_NONPLANAR contours) and nothing else. A sample is written at 1.2 only when
  it holds a polylines annotation; otherwise at 1.0 or 1.1 exactly as before.
- **Why.** A minor may add kinds, profiles and objects (1.0 §16), and a new kind
  is the one addition a 1.1 reader already handles: its projection drops the
  unknown kind as W913 and validates the rest. Open curves cannot be stored
  today: `contours` are closed and planar (a third role would change what an
  existing kind means, which §16 forbids, and an older reader fails on it), and
  `points` are unordered. Clinical extensions (FHIR `interpretation` and
  `referenceRange`, MEDS code modifiers) are *not* in 1.2: a new column inside an
  existing clinical table is not obviously something a 1.1 projection drops
  safely, and that needs its own study (E9).
- **Text to write** (the core of 1.2 §3, "the `polylines` kind"):

  > **`polylines`: open curves.** Ordered vertices, each joined to the next; the
  > last vertex does not join the first, and a polyline encloses nothing.
  >
  > | Dataset | Shape, dtype | |
  > |---|---|---|
  > | `vertices` | `(V, S)` `float32` | in the annotation's `space` |
  > | `polyline_offsets` | `(M+1,)` `int64` | polyline *m* spans `[o[m], o[m+1])`; non-decreasing, `o[0] = 0`, `o[M] = V` (E408) |
  > | `polyline_class_ids` | `(M,)` `uint16` | each polyline's class (E402) |
  > | `polyline_plane` | `(M, 2)` `int32`, OPTIONAL | the `(axis, index)` of the grid plane a planar polyline lies in; `axis = −1` for none |
  > | `instance_ids` | `(M,)`, OPTIONAL | as for other kinds (§7.4) |
  >
  > Each polyline has at least two vertices (E404). A polyline is never
  > rasterised, and a polylines annotation does not satisfy `seg` or `det`. Its
  > `task` is `other` unless the writer states another.

  E404, E405 and E408 report polylines' defects as they report the other
  kinds', and E408's summary names polyline offsets beside contour and mesh
  offsets, under 1.0 §15.2's rule for a kind a minor adds (written by 2.1.0's
  D1): what a code applies to widens, what it means does not. No new code.
- **Versioning text** (1.2 §2): what 1.1 §2 says of 1.1, for 1.2: written at the
  lowest version its content needs; a 1.1 reader projects a 1.2 file (W913 for
  the polylines annotation, the version, and an unrecomputable `content_id`), and
  never amends it.
- **Consequence.** `version::format_versions()` gains "1.2";
  `annotations::kinds("1.2")` gains `polylines`; `version::next_minor` makes the
  W913 fixture and companion fixture T306 "1.3" without edits (2.1.0 D20).
  Corpus: `shapes-polylines` (valid, 1.2), `E408-polyline-offsets`,
  `E404-polyline-one-vertex`; every stated count changes.

## E2 The version a sample is written at follows its content

- **Where.** 1.1 §2.1's rule, restated for kinds in 1.2 §2. W22-01.
- **Decision.** `required_for` (`version.rs`) and `written_version`
  (`sample/writer.rs`) look at profiles *and* kinds: a content-aware sibling,
  `version::required_for_content(profiles, kinds) -> &'static str`, which the
  writer uses. Removing the last polylines annotation in an amendment lowers the
  version, as removing the clinical profile does (1.1 Appendix A's §2.1 row).
- **Why.** Writing every 2.2.0 sample at 1.2 would stop 2.0 and 2.1 readers from
  amending files that hold nothing new.

## E3 Native NIfTI is byte-for-byte the Python converter

- **Where.** No specification text. W22-02.
- **Decision.** The engine gains a NIfTI-1/NIfTI-2 reader and writer (a
  hand-written header codec; both byte orders; `.nii`, `.nii.gz`, `.hdr`/`.img`)
  behind all three frontends. Its contract is parity with 2.1.0's Python
  converter: on a golden corpus recorded from 2.1.0, every object the import
  writes has the same digest, its `/meta` is the same JSON except the version
  the `software` agent records (`medh5.__version__`, which `/meta`'s line in
  `content_id` covers, 1.0 §13.2), and its report JSON is the same except the
  paths it names; exports agree by header fields and voxel bytes (not by gzip
  stream). Those are the only differences the parity test allows, plus any it
  lists by name (W22-10's reworded refusals). nibabel becomes a test
  dependency, and nothing in the runtime imports it: the other modules that
  reach it through `medh5/io/nifti.py` (ITK displacement fields in
  `medh5/io/itk.py`, nnU-Net export in `medh5/io/nnunetv2.py`) move to the
  engine's reader and writer with it. The `nifti` extra stays (empty) for
  compatibility.
- **Why.** The standalone binary and Homebrew users cannot convert the most common
  format without Python; three frontends should run one implementation. The
  `nifti` crate (0.18) reads NIfTI-1 only and pulls in several dependencies;
  medh5 accepts NIfTI-2 and `.hdr`/`.img` through nibabel today.
- **gzip (security).** Inflate through flate2's pure-Rust `rust_backend`
  (miniz_oxide), memory buffers only, never zlib's `gz*` interface, so the VEX
  statement for CVE-2026-85091 (no `gz*` symbol linked) still holds; cap the
  decompressed size at the byte count the header implies. A CI check asserts no
  `gz*` symbol is linked.

## E4 Free-threading: `gil_used = false` only after the audit

- **Where.** `crates/medh5-python/src/lib.rs`; no specification text. W22-04.
- **Decision.** The module declares `gil_used = false` only when every item of
  the audit checklist (W22-04) is done and a free-threaded stress job passes.
  Mutable pyclasses become `frozen` with an inner `Mutex`, and concurrent use is
  refused deterministically (`try_lock`, a `MEDH5Error` "… is used by one thread
  at a time"), never a deadlock or `RuntimeError: Already borrowed`.
- **Why.** 2.1.0 ships cp314t wheels that re-enable the GIL at import (safe).
  Turning it off without the audit risks data races in the engine's caches, the
  torch handle cache and the leases.

## E5 Python 3.10 stays supported

- **Where.** `pyproject.toml` and `Cargo.toml` (pyo3 `abi3-py310`) stay as they
  are. No workstream.
- **Decision.** 2.2.0 keeps `requires-python >= 3.10`, the `cp310-abi3` wheel,
  3.10 in the test matrix and `test-min-deps` on 3.10. This reverses 2.1.0's D36,
  which planned the 3.11 floor for 2.2.0; the maintainer decided on 2026-10-15,
  before `v2.1.0` was tagged, and the announcement was withdrawn from the 2.1.0
  release pull request (#65).
- **When the floor does rise.** When medh5 itself can no longer be built or
  tested on 3.10: PyO3, maturin or the runtime dependency (NumPy, from 1.24)
  stops supporting it, or the CI runners stop providing it. NumPy 2.3 requiring
  3.11 does not force it, since 1.24–2.2 still install. An optional extra that
  drops 3.10 does not force it either: CI installs that extra on 3.11 and later
  only, where its tests still run, so nothing is skipped without a job that
  runs it. The floor then rises in a minor, announced one minor ahead
  (Deprecated in the changelog, `docs/install.md`, `README.md`), the mechanism
  D36 set up.
- **Why.** Dropping 3.10 buys medh5 little: one `tomllib` workaround in
  `test_packaging.py` and one CI row. No 2.2.0 workstream needs 3.11 (`abi3t`
  needs 3.15, E6). Meanwhile Ubuntu 22.04's system Python is 3.10, in standard
  support until April 2027, and it is common on hospital and lab servers.

## E6 `abi3t` when its dependencies exist

- **Decision.** Build a single `abi3t` wheel for free-threaded CPython (PyO3's
  `abi3t-py315`) only once CPython 3.15 is final and NumPy and h5py ship cp315t
  wheels on every platform medh5 targets; until then keep the version-specific
  cp314t (and cp315t) wheels. W22-05 checks the conditions when it starts and
  says which it did.

## E7 A redacted clinical document is a new document

- **Where.** 1.1 §6 already says changed bytes make a new document and a new
  version; 1.1 §10's de-identification bullet (2.1.0 D15). W22-07.
- **Decision.** `scrub --apply --redact-text` rewrites a document's text as a new
  document, owned by a new version of its event (as D15 re-mints versions), with
  every span in the sample that pointed into the old text remapped through the
  edit (a span that overlapped a redacted range is refused, never moved); the
  attestation records the redaction tool and the digests of the bytes written.
  Pattern hits that the redaction does not remove still block.

## E8 Collections are scrubbed member by member, into a new collection

- **Decision.** `scrub --apply` on a `.medh5c` writes a new collection (`--out`
  required; the source is read-only), applying the sample rules to each member
  with one salt, and per-subject shifts by each member's subject; the collection's
  own members list and digests are rebuilt. W22-07.

## E9 What 1.2 does not include, and why

| Not in 1.2 | Why, and what it would need |
|---|---|
| Clinical extension columns (FHIR `interpretation`, `referenceRange`; MEDS code modifiers) | a study first: can a 1.1 reader's projection drop an unknown *column* of a known table without misreading the rest (1.1 §4 closes the column set, E804)? If not, they need a new table, which is a new object a projection can drop. Candidate for 1.3. |
| Annotation pyramids | reserved by 2.1.0 (D11); nobody has defined `nearest` under even factors or `max` over ids |
| A `bspline` transform kind | ITK B-spline transforms are imported, when exact, as sampled displacement fields (W22-08); a kind of its own needs a demonstrated case the field cannot represent |
| `rle` | the kind name is reserved (1.0 §16), so a minor could add it as a new kind a 1.1 reader projects, as `polylines` is added here; but 1.0 §7 measured little saving over `instances` for the cost of O(1) region reads and a second sparse path. Needs a measured case `instances` serves badly. |
| `wide_labels` | out of scope by design (2.1.0 README) |
| Image-free subjects | not possible in a 1.x minor: E111 and E201 are core errors, and a 1.1 reader projecting a newer file still applies them, so an image-free file would read as invalid (§16). It needs format 2.0's subject-first core (design rationale, "A minor version, and an optional profile"). |

## E10 HDF5's own close-on-exec

- **Decision.** If hdf5-metno-src ships an HDF5 whose sec2 driver opens with
  `O_CLOEXEC` (and `Wopen` with `_O_NOINHERIT`), adopt it (a `-sys` bump, with
  `SECURITY.md`'s row and advisories re-read) and keep the engine's mark as a
  no-op for safety. If not, nothing changes. W22-05.

## E11 Resampling is a derivation, never a conversion

- **Where.** No specification text: 1.0 already defines everything a resampled
  object needs. §10.2 defines how to resample through a transform, §4.4's
  `valid_mask` covers voxels that "resampling" leaves without data, §11.1's
  activity type `resample` and `prov` attribute record it, and §6.2's
  `derived_from` links a derived annotation to its source. W22-10.
- **Decision.** A deliberate command, `medh5 resample` (with
  `medh5.transforms.resample(...)` in Python, `transforms::resample` in the
  engine), adds resampled objects to a sample. Each object is either moved
  onto an existing grid of the sample, through `transform_between` when the
  frames differ, or onto a new grid in the source's frame with a requested
  spacing. The source
  objects stay untouched, and every object it writes has a `prov` reference to
  one `resample` activity naming the inputs, the target grid, the transform
  path and the interpolation. Converters and loaders keep refusing to resample;
  their refusals name the command instead.
- **Rules.**
  - Grids are never invented. Two grids are mapped by their affines only when
    §3.3 rule 4 makes them comparable (a shared frame with another
    `coord_system` or S, W917, is refused); otherwise through
    `transform_between`, and without a path the command refuses. It never
    assumes alignment. A new grid keeps the source's voxel edges, not its first
    centre.
  - Images are interpolated linearly by default (`nearest` and `cubic` on
    request; cubic is the engine's SciPy-exact sampler) on stored values, so
    the rescale attributes carry over. Linear and cubic write `float32` unless
    a `dtype` is requested (rounded half to even, clipped). Only spatial axes
    are resampled, and only level 0 of a multiscale image.
  - Voxels whose source point falls outside the source extent take a fill value
    (0 unless given). The image's one `valid_mask` (§4.4) is the in-extent
    region intersected with the source's own `valid_mask`, resampled.
  - Every output gets a new id, the source's plus a suffix (`_on_<grid>` by
    default); an id already in the file is refused before anything is written.
  - Segmentations use `nearest`, in their own encoding; `instances` object by
    object, keeping each `instance_id` (an object that vanishes is refused);
    probmaps `linear`, keeping `normalized` only where no voxel was filled. All
    keep the source's `class_ids` and `annotated_class_ids` and name it in
    `derived_from`.
  - World-space geometric annotations stay as they are in the same frame. Under
    a transform, points, keypoints and meshes are mapped exactly. Boxes and
    contours are refused under any transform that does not keep them
    axis-aligned or planar, never approximated. Index-space ones are
    re-expressed through world coordinates, and refused under the same rule.
- **Why.** The format exists so that resampling is not done silently at ingest
  (1.0 §3's rationale: "makes resampling an explicit, provenance-tracked
  activity"). An opt-in derivation that records itself keeps that promise.
  Resampling inside a converter or a loader would not, because the choice of
  filter and grid disappears into a training run. MONAI already resamples medh5
  items correctly from their affine, so loaders gain nothing a pre-resampled
  sample does not give them.
- **No new code, no format change, no corpus change.** Refusals are
  `MEDH5ValidationError`s with messages, as the converters' are. The output is
  an ordinary 1.0 (or 1.1) sample, written at the version its content needs.
