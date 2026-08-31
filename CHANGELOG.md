# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] — 2026-08-30

A correctness release from a full review of the library. The **format version is unchanged**: 1.1.0
reads and writes exactly the files 1.0 does, and `__format_version__` stays `"1.0"`. The package
takes a **minor** bump rather than a patch because several fixes change what existing code produces
or accepts — see **Behaviour changes** below before upgrading a pipeline.

Two clauses of the specification were corrected (Appendix C now lists ten), and neither changes what
a conforming file looks like: one pins a rounding rule that was under-specified, the other writes
down an exclusion the reference implementation already applied.

### Fixed — converters that assumed instead of checking

A second review pass covered the three subsystems the first one never reached: the DICOM family,
the cohort tooling, and the §14 storage claims. The converter findings share one shape — read an
attribute from element zero, assume the rest of the stack agrees, never check — and each produces a
file that looks entirely well-formed.

- **nnU-Net channels and label volumes were filed onto the first channel's grid, whatever their own
  geometry.** `from_nnunetv2` kept `geometry or geo`, so a second channel at a different spacing or
  origin — or a label volume resampled by some other tool — was written onto channel 0's grid with
  its voxels intact and its position silently wrong. Both now go through the same `_same_grid`
  refusal `from_nifti` has always used (§3.2, E101/E202).

- **nnU-Net export wrote an all-background label volume for most real datasets.** `_labelmap_for`
  matched classes by their `dataset.json` name, but the import sanitises that name into the label-set
  key, so any dataset naming a class `"Tumour Core"` or `"GTV"` resolved nothing — and the failure
  fell into a bare `except: continue`. `dataset.json` was written listing every class, the label
  files were the right shape, and every voxel was 0. Classes are matched by id now (the import keeps
  nnU-Net's own integers precisely so they can be), and a class genuinely absent is refused (E402)
  rather than dropped. The round-trip test missed it because its fixture's labels — `edema`,
  `enhancing` — are already valid keys, so sanitising was a no-op.

- **A DICOM series took its modality LUT, orientation and pixel spacing from slice 0 alone.** A
  per-slice `RescaleSlope` — ordinary in PET — was collapsed to the first slice's, reporting a value
  1740 HU out on every slice it did not apply to; a single rotated slice was placed on the first
  slice's direction matrix. All three are now checked across the stack and refused, naming the
  offending SOPInstanceUID. These refusals carry **no diagnostic code**: §15.2's table describes
  conditions in a MEDH5 file, and a DICOM series is not one yet — no code in it means "these slices
  disagree", so borrowing one would have reported a modality-LUT problem as malformed `channel_names`.
  A slice missing one of these tags outright is refused the same way, rather than surfacing the raw
  `AttributeError` that `scan_dicom` (which does not require the tags) makes reachable — as is a tag
  of the wrong *length*, which extracts perfectly well and so passed the agreement check when every
  slice carried the same wrong length. A five-value `ImageOrientationPatient` then reached
  `np.cross`; a three-value `PixelSpacing` was silently read as its first two elements, giving the
  grid an in-plane size nobody wrote down.

- **A split could put one subject in two partitions.** `group_id` is declared per file and defaults
  to the subject, so two visits curated at different times can disagree about it — the subject then
  becomes two groups, dealt independently, and the same anatomy lands in train and val. `Split.leaks`
  cannot see this: each group really was assigned once, which is all it can know from the assignments.
  `make_splits` now refuses a grouping finer than the subject it is meant to contain (new cohort code
  `C204`), which is where the entries are still in hand. The existing coverage passed because its
  fixture gives every file of a subject the same `group_id`.

### Verified — no change needed

- **§14's storage claims hold as written.** Measured rather than assumed: every image and voxel
  annotation is chunked, stacked encodings use `(1, *spatial)` so one plane reads without the others,
  chunks land inside the 0.5–4 MiB target, and `portable` uses only native `shuffle`+`gzip` while the
  other profiles use Blosc2 as documented. Recompression across `training`/`archive`/`portable` left
  `content_id` and every voxel unchanged with `verify()` passing, and a forked child read correctly
  with the parent uncorrupted.

### Added

- **A corpus smoke test over the whole public read surface.** The conformance corpus checked that
  each case reports its expected diagnostic codes but never called `summary()`, `verify()` or the
  grid/image/annotation/transform accessors on those files. Two contracts now hold across all 103
  cases: a valid case survives the full read surface, and *any* case — including the deliberately
  malformed ones — fails only through `MEDH5Error`, never an `AttributeError` or `KeyError` a caller
  cannot catch by the documented type.

### Fixed — silent loss of ground truth

- **A box on integer edge coordinates lost or gained voxels according to its parity.**
  `box_to_slices` rounded with `np.rint`, which rounds half to **even**, so `lo + 0.5` and
  `hi + 0.5` rounded in opposite directions: a one-voxel box at `[1.0, 2.0]` became an *empty*
  slice and one at `[2.0, 3.0]` became two voxels wide. Boxes built from `slices_to_box` are
  half-integer and never reach the tie, which is why the round-trip tests could not see it; boxes
  from a world→index conversion, an even-factor resample or a pyramid level change are
  integer-valued and reach it constantly. Five call sites depended on it, including
  `curation/tracking.py`, where a lesion's volume is `prod(stop − start) × voxel_volume` — so the
  same lesion measured 0 mm³ or 8× across visits, in the longitudinal measurement the format exists
  for. §8.1 now states `floor(x + 0.5)` explicitly.

- **`labelmap()` deleted overlapping voxels without saying so, on three exits.** `to_nifti(annotation=...)`,
  `VolumeDataset(label_format="labelmap")` and `medh5.monai.to_dict` all flatten on the way out, and
  each silently dropped the overlap region: with a lesion inside a liver, the liver came back 224
  voxels instead of 256. One integer volume cannot hold overlapping classes — which is why `layers`,
  `bitmask` and `probmap` exist (§7.0) — so `labelmap()` now warns, naming the count, unless the
  caller passed an explicit `priority`.

- **`annotated_classes="all"` was byte-identical to `"all_given"`.** `_resolve_annotated` intersected
  the label set back down to `class_ids`, so it could never claim a class that had no mask. Every
  class the annotator searched for and did not find was recorded as *never looked for* instead of
  *verified absent* — the one distinction the coverage contract exists to keep (§11.3) — and no
  validator fired, because W904 only warns when `annotated_class_ids` is a strict subset and this
  made the two equal. `"all"` now also works with `probabilities=`, which needed
  zero-valued planes injected for the absent classes the same way absent masks get an
  empty mask; without them the write failed `E403`, since `annotated_class_ids` named
  more than `class_ids` held.

- **Transcoding destroyed an in-band ignore region.** `labelmap`/`layers` carry ignore in the data;
  `bitmask` and `probmap` express it as a separate `mask` annotation (§7.7), which a
  payload-returning function cannot create. The region was simply dropped, turning "nobody examined
  these voxels" into "verified absent for every annotated class" — what §7.7 names as the most
  common cause of a silently mistrained segmentation model. Transcoding to an encoding that cannot
  hold it is now refused, with the two ways forward in the message. A non-default
  `ignore_id` is carried to the target encoder along with the mask — passing only the
  mask left the header naming a value the data did not contain, which put the region
  right back to reading as background. `LayersAnnotation` gained the `ignore_mask()`
  that `labelmap` already had: it could report `has_ignore_region` while offering no way
  to read the region, so a caller written as `getattr(a, "ignore_mask", None)` — the
  refusal above among them — concluded there was none.

- **Transcoding *to* `instances` merged every object of a class into one.** A dense encoding records
  which voxels belong to a class and never which object, so the conversion collapsed two lesions
  into a single object carrying a freshly minted `instance_id` that belonged to neither — in the
  field §7.4 makes the entire longitudinal join. Refused now, for the same reason
  `instances_from_masks` already refuses to *split* components.

- **The MONAI bridge gave annotations another image's geometry.** `to_dict` took the affine from the
  first requested image rather than from the grid the annotation was bound to. On a sample holding
  CT and PET on different grids — the case this format exists for — the label tensor arrived with
  PET's shape and CT's affine: 70 mm off in z, at half the true spacing, against a docstring
  promising the opposite. It also fabricated an identity affine when no image was requested. The
  affine now comes from the annotation's own grid, and `to_dict` has a test; it had none, and its
  body had never executed even in the dedicated MONAI CI job.

- **`class_ids` and the payload disagreed after transcoding to `probmap`.** `probmap` carries plane
  order only in the §6.2 `class_ids` attribute and its encoder always emits ascending, while
  `transcode_annotation` kept the source header verbatim. A file whose encoding order was not
  ascending — which §6.2 permits, and §7.3 makes explicit for `bitmask` via `bit_class_ids` — read
  back with each class's mask under a different class's name, per-class voxel counts unchanged and
  nothing for the validator to see. Not reachable through this writer, which normalises to
  ascending; reachable through a conforming third-party file.

- **A class examined and found empty vanished on the `instances` decode.** `payload_to_masks` keyed
  off the objects present rather than the declared `class_ids`, and `check_roundtrip` decodes
  through that same path — so the module's own losslessness check could not see the loss.

### Fixed — geometry, registration and sampling

- **An ambiguous frame path resolved silently.** Two registrations between the same pair of
  frames — which §10.1 permits, and which a rigid plus a deformable pair makes ordinary —
  left `transform_between` picking by dict iteration order, i.e. lexicographic transform id.
  Two affines disagreeing by 109 mm resolved to whichever was named first, with the
  validator silent. §10.2 exists because "ambiguity here is the leading cause of silently
  mirrored registration results", so resolution now refuses, names both candidates, and
  points at `sample.transforms` to select one. A single route resolves exactly as before.
  Routes that *converge* before the destination count as two: marking a frame seen on the
  first path to reach it discarded the second, so `A→B→D→T` and `A→C→D→T` arrived as one
  and the tie went undetected.

- **`check_pyramid` never compared a level's `direction` to level 0.** Spacing, origin,
  `coord_system`, `units` and `frame_uid` were checked; orientation was not — and because
  the *expected* origin is derived from the base's direction, a level with permuted axes
  passed both remaining checks. That is precisely the failure §4.3 exists to prevent: a
  model trained at level 2 has its predictions mapped back to level 0 and they land
  transposed. Level extent is now bounded too — one voxel either side of `n / f`, which
  admits both rounding conventions and still rejects a level that is not a resampling of
  its parent.

- **A transform's `units` were never checked against the chain it sits in.** §10.1 makes
  `units` a MUST — "coordinate units, matching the frames' grids" — but only frames were
  compared, so a composite chaining an `mm` leg to a `um` leg validated clean and applied a
  1000× error to half the transform. `check_chain` compares units, and a resolved
  `ChainTransform` refuses to assemble steps that disagree.

- **`strategy="uniform"` was not uniform.** Centres were drawn over every voxel and then
  clamped inward, so every centre in the leading half-patch collapsed onto window start 0 —
  measured at 3.6× the uniform share for the first window and 2.8× for the last, on a
  24-voxel axis with an 8-voxel patch, and worse as the patch grows. Border-heavy training
  data from a strategy named for the opposite. The centre is now drawn from the range that
  maps one-to-one onto valid window starts; clamping stays on the foreground path, where the
  centre is a voxel the caller specifically wants included.

- **`as_slices()` ignored `slice_index`**, so §8.2's canonical "2D box on slice k" — the
  common radiology annotation — selected **no voxels at all**: a degenerate axis converts to
  a zero-thickness slice. The named slice now gets one voxel of thickness. A box with real
  extent on every axis is left alone rather than reinterpreted.

- **The classification accessors contradicted each other on multi-assertion scopes.** §9
  makes several assertions per class ordinary — `scope_ids` is "per assertion", and
  `scope = "timepoint"` means one per visit — but `value()` returned the first matching row
  while `labels` kept the last, so `state()` answered "negative" for a class `positives`
  listed as positive, on one file. `value()` and `state()` take `scope_id=` to select, and
  every collapsing accessor refuses rather than picking when a class carries more than one
  assertion. Single-assertion files are unaffected — including in `summary()`, which keeps
  its flat shape there and reports per scope unit only where a class is asserted more than
  once, so `Sample.summary()` and `medh5 info` keep working on the very files this supports.

### Fixed — de-identification and access control

- **A scrubbed file still contained the DICOM UID it had pseudonymised.** `scrub --apply` runs
  inside the copy-on-write amend, which copies each object and *then* rewrites the attribute; HDF5
  never reclaims what it supersedes, so the released file carried the original
  `FrameOfReferenceUID` in freed space — recoverable with `strings` while every API read returned
  the pseudonym and the file attested `id_mapping: "external"`. A UID links back to the originating
  study in the source PACS. `--apply` now compacts the file before returning; digests and
  `content_id` are unaffected, since this rewrites storage and not content.

- **`id_mapping: "external"` was attested whenever a salt was given**, even when no UID matched and
  the mapping was empty. It is §11.4's strongest claim; it is now recorded only when something was
  actually mapped.

- **Every copy-on-write command widened file permissions.** `amend`, `scrub --apply`, `fix` and
  `recompress` replace the file, and the replacement was created under the process umask — so a
  `0o600` sample came back `0o644` and world-readable, in the commands most likely to be pointed at
  sensitive data on a shared filesystem. The mode of the file being replaced is now carried across.

### Fixed — tools that must not lie or crash

- **`medh5 validate` crashed on roughly a third of corrupted files.** Bytes damaged past the header
  raise out of h5py's traversal or its decompressor; the command exited with a traceback, printed
  nothing on stdout, and `--json` emitted no JSON at all. Failing to read an object is a finding
  about the file, not a crash of the tool: it is reported as `E001` now, per rule, so one unreadable
  object does not hide everything else. 120 randomly corrupted files, up to ten byte flips each, now
  all produce a valid report.

- **`--level strict` promoted warnings in the verdict but not in the counts**, so a report read
  `FAILED … (0 errors, 2 warnings)` and a CI job gating on `errors == 0` passed a file the same
  payload called not-ok. §15.1 defines strict as the other levels with warnings promoted, so at
  strict there are no warnings; each diagnostic keeps its measured `severity`, and the JSON gains a
  `promoted` count.

- **`amend` dropped unknown attributes on the sample root.** Grids, images, annotations and unknown
  groups already kept theirs; the root was the one level that did not, so a 1.0 tool amending a 1.1
  file silently discarded what 1.1 had added — against §16, which permits a minor version to add
  attributes and requires readers to ignore ones they do not recognise.

- **`from_nifti` invented geometry and reported no doubt.** A NIfTI with `sform_code == qform_code
  == 0` states that it carries *no* spatial mapping; nibabel still returns an affine rebuilt from
  `pixdim`, and importing it minted a world grid nobody measured, with the report saying "0
  guesses". It is refused now unless `assume_geometry=True` (CLI: `--assume-geometry`), which
  records it as a guess. An sform and qform that *disagree* — the signature of a file one tool
  updated and another did not — is likewise recorded rather than resolved in silence.

- **The exported payload encoders accepted reserved and out-of-range class ids** and wrapped them
  into the label dtype: `0` became background, `-1` became 255, `65535` became the ignore value,
  `70000` became 4464. The public writer already refused these; the encoders under it, which a
  third-party converter calls directly, now raise `E303` too.

- **`instance_id` was hard-cast to `uint32`**, so an id minted from a 64-bit key wrapped — `2³² + 7`
  became `7`, taking another object's identity. §7.4 permits `uint64`; the width now follows the
  data.

- **`encode_obb` raised a bare `ValueError`** on an empty collection where `boxes`, `mesh` and
  `instances` all raise a coded `E405`. An empty detection annotation is the verified negative the
  coverage contract records, not a degenerate input.

- **`medh5 conformance` with no subcommand** printed its usage to stdout; the other six group
  commands write to stderr.

- **`to_metatensor` emitted a spurious MONAI warning on every call**, passing the affine both
  positionally and inside `meta`. The affine was correct; the warning said otherwise.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction, and each can change what existing code
produces or accepts:

- **Converter refusals no longer carry a format diagnostic code.** `from_nifti`'s grid-disagreement
  refusals raised `E202` (shape) and `E101` (spacing/origin/direction), and the new DICOM per-slice
  checks initially borrowed `E102`/`E104`/`E204`. §15.2's table describes conditions found *in a
  MEDH5 file*, and a NIfTI volume or DICOM series is not one yet: none of those codes means "these
  inputs disagree", so a caller branching on the code was told an untrue story — a modality-LUT
  problem read as malformed `channel_names`, a grid disagreement as a dangling grid reference. These
  refusals are now uncoded. **Code branching on `exc.code` for a converter refusal must switch to the
  exception type**; the messages are unchanged and still name what disagreed.

- **A DICOM series that declares no `PixelSpacing` at all is now refused.** One slice omitting it was
  already caught, because it disagreed with the others; *every* slice omitting it meant they all
  agreed on a 1 mm default, and the stack was written with an in-plane size the source never stated.
  Files reaching this path carry `ImagePositionPatient`, so they are cross-sectional images for which
  `PixelSpacing` is mandatory — its absence is a broken series, not one to guess about. A series that
  previously converted with assumed 1 mm spacing will now refuse; that spacing was never the source's.

- Boxes on integer edge coordinates now yield the extent they describe. ROIs derived from
  `box_to_slices` — crops, instance decoding, `as_slices`, tracked lesion volumes — change where
  they were previously off.
- `annotated_classes="all"` now records the whole label set, so files written with it gain
  `class_ids` and `annotated_class_ids` entries (and the zero masks behind them).
- `from_nifti` refuses a NIfTI declaring no spatial mapping; pass `--assume-geometry` to keep the
  old behaviour, now reported as a guess.
- Transcoding refuses two conversions it used to perform silently: to an encoding that cannot hold
  an in-band ignore region, and from a dense encoding to `instances`.
- `labelmap()` warns when it flattens real overlap and no `priority` was given.
- The payload encoders raise `E303` for class ids the public writer already rejected.
- `validate --level strict` reports promoted warnings in `errors` rather than in `warnings`.
- `transform_between` raises rather than choosing when two equally short routes exist; select
  one by id from `sample.transforms`.
- `strategy="uniform"` places windows uniformly, so **the same seed now draws different
  patches**. The change removes a border bias; it does not make previous runs invalid, but it
  does mean a run is not bit-reproducible across this upgrade.
- The classification accessors (`value`, `state`, `labels`, `positives`) raise on a file that
  asserts one class more than once; pass `scope_id=` or read `assertions()`.
- `check_pyramid` — and therefore `E105` — now fires on a level whose `direction` or extent
  disagrees with level 0.

### Changed

- **`medh5 tree` gained `--json`.** It was the only inspection command without a machine-readable
  form, and naming each object with the spec clause that gives it its role is exactly what a cohort
  audit wants.
- **Python 3.13 is tested.** `requires-python` has always admitted it; the matrix and the classifiers
  stopped at 3.12.
- **The specification's executable prototype runs in CI.** Appendix C.2 publishes a table of its
  results, and nothing referenced it — unlike the conformance corpus beside it. It also writes its
  output to the working directory now, rather than next to its own source, so running it does not
  dirty a checkout.
- `medh5.monai.available()` is measured rather than excluded from coverage by a blanket pragma —
  the same shape as the untested-because-skipped problem the MONAI CI job was added to prevent.

### Removed

- `medh5.storage.index.index_attrs()`, a stub returning `{}` that nothing called. It was the
  unfinished half of putting `index/` attributes into `content_id`; §13.2 now states the exclusion
  instead.

### Specification

- **§8.1** pins the box↔slice rounding to `floor(x + 0.5)`. "round" was read as a language default,
  and both Python's `round` and NumPy's `rint` round half to even — under which the extent identity
  in the same clause does not hold.
- **§13.1, §13.2** state normatively that `index/` is excluded from object digests and from
  `content_id`. The reference implementation always skipped it, on the grounds that a derived cache
  should not change the address of the sample it derives from; the text did not say so, so a
  conforming implementation that stamped index digests would compute a different `content_id` for
  the same bytes — and `content_id` is only useful as a cross-implementation key if every
  implementation agrees on what it covers.

### Internal

- Test suite 924 → 969, coverage 93% → 93.5%. Two tests that could not fail were repaired:
  `json.dumps(..., default=str)` coerces anything, so two "is JSON-safe" assertions were vacuous.
  `test_the_format_version_is_not_the_package_version` asserted the package version starts with
  `"1.0"`, tying it to the format version in exactly the way its own docstring forbids; it only
  looked right while the package sat on 1.0.x.

## [1.0.1] — 2026-08-18

Fixes against MEDH5 format 1.0. The **format version is unchanged**: 1.0.1 reads and writes exactly
the files 1.0.0 does, and `__format_version__` stays `"1.0"`.

### Fixed

- **A multi-echo NIfTI is no longer imported as a time series** ([#9]). NIfTI-1 puts time in
  `dim[4]`, but §3.6 gives that axis a `time` row and a `channel` row, and a multi-echo series
  states neither an intent code nor a temporal unit — so it fell through to the guess and arrived
  with per-frame timings the converter had invented. `from_nifti` now reads the BIDS JSON sidecar,
  the same convention `.bval` already covers for DWI.

  Evidence has to be **per volume**: a scalar `EchoTime` is in every MRI sidecar ever written, so
  only a list whose length matches the frame count settles the axis. `EchoTime`, `EchoNumber`,
  `InversionTime` and `FlipAngle` name a channel axis; `VolumeTiming` names a time axis *and*
  supplies measured frame times, which beats a ramp rebuilt from `pixdim[4]` for sparse-sampled
  acquisitions. Echo times are recorded in `acquisition` under the DICOM keyword §4.5 asks for.
  A sidecar stating both kinds at once is refused, and `fourth_axis=` still overrides everything.

- **`medh5.__version__` can no longer drift from the published wheel.** The release workflow
  verified the tag against `pyproject.toml` alone, so a half-completed bump could publish a wheel
  that stamped the wrong version into every file's `generator` and every dataset manifest. Both the
  workflow and the test suite now check the two agree.

### Changed

- **The MONAI tests run in CI** ([#8]). They are guarded by `pytest.importorskip` and MONAI was in
  neither the `dev` extra nor the test matrix, so `to_metatensor(level=...)` shipped covered by a
  test that had never executed. A dedicated job installs the `monai` extra and asserts the import
  before running pytest — without that assertion a failed install leaves the tests skipping and the
  job green.

[#8]: https://github.com/XwK-P/medh5/issues/8
[#9]: https://github.com/XwK-P/medh5/issues/9

## [1.0.0] — MEDH5 format 1.0

A clean-slate reimplementation of the format. **Not backward compatible with 0.x**, by design: a
1.0 reader refuses a 0.x file rather than guessing, and a 0.x reader raises on the missing
`schema_version`. See [the specification](docs/spec/medh5-1.0.md) and
[the implementation plan](docs/design/medh5-1.0-implementation-plan.md).

### The model

A **sample is one subject at one or more timepoints**, each with one or more images. Longitudinal
work — change detection, response assessment, lesion tracking, follow-up registration — lives in one
file, and assigning whole files to train/val/test cannot leak a patient across partitions.

### Added

- `medh5.open` / `create` / `amend` and the `Sample` / `SampleWriter` API.
- **Geometry as a first-class object.** Grids carry the full index→world affine, a frame of
  reference and a timepoint; images and annotations inherit rather than repeat it. Multiscale
  pyramids validate the half-voxel origin shift that silently misplaces predictions when omitted.
- **Label sets** as DAGs with ontology bindings, explicit/implicit closure, canonical digests, and
  three bundled vocabularies (`binary-foreground`, `brats-subregions`, `amos22-organs`).
- **Five voxel encodings** — `labelmap`, `layers`, `bitmask`, `instances`, `probmap` — behind one
  read contract (`contains`, `dense`, `labelmap`, `instances`). The encoding is chosen by measuring
  the class overlap graph, and any pair transcodes losslessly, so it is a storage decision rather
  than a data-model decision.
- **The coverage contract.** `annotated_class_ids` records what the annotator committed to finding,
  so `0` reads as "verified absent" only where that is true. Partially-labelled cohorts become
  safely trainable instead of quietly mistrained.
- **Content addressing.** Per-object SHA-256 digests over decompressed content plus a Merkle
  `content_id`, so verification is incremental, partial and local, and recompression does not
  invalidate anything.
- **A sampling index** that answers foreground patch sampling in O(1) in volume size — 0.52 ms and
  48 KiB against 9.2 ms and O(volume) for the 0.x `argwhere` path.
- **A validator** with four levels and a stable diagnostic-code table, and a **103-case conformance
  corpus** with per-code expectations that a third-party implementation can run. Every code in the
  table has a case.
- **Every geometric annotation** (§8): axis-aligned boxes that convert to numpy slices without
  rounding, oriented boxes stored as rotation *matrices* (dimension-generic, no ordering convention,
  no double cover), keypoints with per-slot classes and visibility, landmark point sets with
  correspondence, planar contours for RTSTRUCT round-trips, and triangle surface meshes. Coordinates
  live in a declared `space` — a grid's continuous index coordinates or a named frame — and readers
  convert through the affine instead of assuming.
- **Classification** (§9), including the three-state semantics that make partial labels safe
  (positive / verified-negative / unknown), ordinal schemes stored verbatim rather than coerced to
  numbers, and **change labels**: an ordinary classification whose `timepoints` names the visits
  compared, so `["tp0","tp2"]` and `["tp1","tp2"]` are distinct assessments.
- **Registration** (§10): identity, affine, dense displacement fields, B-spline free-form
  deformations and composites. One direction convention, `x_M = T(x_F)`, with no attribute to
  reverse it — ambiguity there is the leading cause of silently mirrored results. Displacement
  fields store components on the leading axis so one component or one ROI reads without the rest,
  and report their Jacobian determinant and folding fraction. `transform_between` resolves through
  the **frame graph** rather than by name, uses inverses where they exist, and refuses to invent
  one for a dense field — approximating it would report an accuracy nobody measured. Target
  registration error is computed from paired landmark sets (§10.6).
- **Curation** (§11, §12): a two-node PROV graph of agents and activities that describes the
  dominant real workflow — a model pre-annotation corrected by a human — where a review-status field
  cannot; quality records whose status is current state, with history living in the graph; and
  `agreement` computed from the annotations themselves (per-class Dice/IoU, object-level F1), so a
  number in a file is reproducible from that file. Classes one side never examined are reported as
  *not scored* rather than scored zero, because a class nobody looked at is not a disagreement.
- **Longitudinal tracking joins** (§7.4): `Sample.tracks()` groups objects on `instance_id` across
  visits and reports per-visit volumes and growth. Absence resolves to **`resolved`** only where the
  class is in that visit's `annotated_class_ids`, and to **`unexamined`** otherwise — a growth curve
  that reads "not assessed" as volume zero reports a complete response that never happened. W909 is
  sample-scoped, so it catches the conflict that matters: one lesion classed differently at two
  visits, each annotation internally consistent and only the join wrong.
- **A cross-file split audit** (§12.3). A per-file validator cannot see either failure that matters:
  two files claiming one `set_id` against different manifests (W906), or one subject appearing in
  two partitions — the leakage that inflates every reported metric in medical AI. `medh5 splits`
  reports both, and keeps them separate because the remedies differ.
- **Collections** (§2.2): `.medh5c` shards for cohorts where one file per sample is an operational
  problem. `pack` and `unpack` move stored chunks rather than re-encoding them, so a round trip is
  byte-identical and every `content_id` survives it — a shard is a container for samples, never a
  second encoding of them. A packed sample root *is* a sample root: every reader, validator and
  loader works on it unchanged.
- **Loaders** (plan §2.3). `medh5.sampling` chooses patch windows and visit pairs and depends on no
  deep-learning framework, because where to read is a geometry question. Foreground sampling reads
  the cached coordinate subsample (§14.3) instead of scanning a mask — **0.90 ms and O(1) in volume
  size**, against 9.2 ms and O(volume) for the 0.x `argwhere` path — and a file without an index
  still works but says so on every patch it returns, since a silent 20× slowdown in a dataloader is
  indistinguishable from a slow disk. `medh5.torch` adds `VolumeDataset`, `PatchDataset`,
  `GridPatchDataset`, `PairedPatchDataset`, a collate that keeps ragged detection targets ragged,
  and a **PID-keyed handle cache** (§14.4) — an HDF5 handle inherited across `fork` returns corrupt
  reads that look like data errors, so the cache abandons its contents the moment it notices a new
  process.
- **Paired longitudinal sampling.** `align="transform"` maps the patch centre through the transform
  relating two visits, so both patches cover the same anatomy; `align="none"` returns unregistered
  pairs. A cross-sectional file contributes no pairs and is *counted*, not silently dropped.
- **A MONAI adapter**: `to_metatensor` hands MONAI the correct affine, labelled with its world
  convention rather than converted to RAS behind the caller's back. An ROI shifts the origin, so a
  cropped tensor still lands where its anatomy is.
- **`recompress`** (§14.2). Because digests cover *decompressed* content (§13.1), re-encoding
  changes every stored byte and no `content_id`: a cache keyed on it stays valid, and moving a
  cohort from `training` to `archive` is a storage decision rather than a data migration.
- **Faster multi-class label reads.** `dense()` on `layers` and `bitmask` now groups by **plane
  rather than by class**: a 200-class annotation packed into four layers costs four reads, not two
  hundred. A 64³ multi-class patch read is **4.0 ms** against the 117 ms the 0.x layout needed.
- **`medh5 bench`**, which reproduces every performance target in the plan on the reader's own
  hardware. All are met: sustained patch throughput measures 600–850 patches/s against a target of
  400.
- **Converters** for NIfTI, DICOM, DICOM SEG, RTSTRUCT and nnU-Net v2, plus `migrate` from 0.x.
  Each returns a report of what it *decided* and what it *guessed*, because the interesting part of
  an import is never that it succeeded:
  - **NIfTI**: RAS↔LPS is a sign flip on the affine, never on the voxels, and which convention was
    written is recorded rather than assumed. Volumes that disagree on a grid are refused instead of
    silently resampled.
  - **DICOM**: slices are ordered by their projection on the slice normal, not by `InstanceNumber`
    (a display hint that routinely disagrees with geometry); z spacing is *measured* between slice
    origins rather than taken from `SliceThickness`, which is the slab and not the increment; an
    irregular stack is refused, because it is not a grid. Rescale is stored, not applied (§4.2).
    Only a named list of acquisition tags is copied — §11.4 forbids bulk-copying DICOM into a file
    that claims to be de-identified.
  - **DICOM SEG**: frames are placed by geometry, so a SEG that stores them out of order still
    reads; overlapping segments and `FRACTIONAL` both survive, where flattening to a labelmap would
    drop the tumour inside the organ. Segments match an existing label set by `SegmentLabel`, since
    DICOM segment numbers are positional and carry no identity.
  - **RTSTRUCT**: contours are stored as contours (§8.6) and round-trip exactly. Rasterisation is
    opt-in and its rule is written into the provenance graph, because "does a boundary voxel count"
    is a decision that belongs in the record. A contour enclosed by another on the same slice is a
    hole, not a second region.
  - **nnU-Net v2**: the dataset's own integer ids are kept, so a model's predictions map back with
    no translation table; region labels become classes whose components are their children in the
    §5.1 DAG; `dataset.json` is stashed verbatim, so an export reproduces the dataset rather than
    reconstructing it.
  - **`migrate`**: applies Appendix B and reports each non-mechanical step — the encoding chosen,
    the ids minted (cohort-wide, into a reviewable sidecar), the **half-voxel box shift** from 0.x's
    `[min, max)` integers to voxel edges, and whether a timepoint order was read from dates or
    guessed from mtimes. Instance correspondence across merged files is never inferred.
  - **Grouping**: identity comes from a declared key and never from a filename, a date or an
    accession number. Where it cannot be established the converter falls back to one sample per
    study, names the inputs, and records the fallback.
- **A CLI**: `info`, `tree`, `validate`, `verify`, `timeline`, `track`, `labels`, `seg stats`,
  `seg convert`, `index build`, `pack`, `unpack`, `ls`, `prov`, `agree`, `splits`, `recompress`,
  `bench`, `convert`, `migrate`, `conformance`.

- **Cohort tools** (`medh5.dataset`, `medh5 dataset`). A manifest is a metadata-only scan cached as
  JSON, and it is the *authority* for splits (§12.3): its digest covers membership and grouping —
  which samples, grouped how — and deliberately not content, because writing a claim into a file
  changes the file, and a content-covering digest would make every claim stale the moment it was
  written. Splitting groups before it splits, never by file, and is deterministic given
  `(digest, seed, parameters)`. Groups are dealt by largest deficit against the target ratios
  rather than sliced by index — slicing six groups at 70/15/15 gives train all of them — and where
  an indivisible set of groups genuinely cannot meet the ratios, the split *says which partition got
  nothing* instead of leaving an empty test set to be discovered after the results are written up.
  Strata are interleaved with a rotating lead and dealt against one global tally, so stratifying
  balances the cohort instead of starving val and test. Statistics stream with an exact
  Chan-Golub-LeVeque merge weighted by voxel count, read class counts from the sampling index when
  it is current, and never count an unexamined class as a zero. `dataset check` asks the questions
  no single file can answer — one label set or several, a class id meaning two things, a claim from
  another manifest, a subject in two partitions, a class examined in a tenth of the cohort — under
  its own `C1xx` codes, because a file is not non-conforming because the cohort around it is.
- **`medh5 fix`**, which separates rebuilding a derived cache from restamping a claim. Rebuilding
  an index recomputes a cache from what it caches. Rewriting digests is *not* repair: a mismatch is
  evidence the bytes changed, and recomputing it destroys the evidence. So `--rewrite-digests`
  requires a reason, records an activity naming what it did not verify, and says so on stdout.
- **`medh5 scrub`** (§11.4): a de-identification sweep over the container that attests to exactly
  what it did. UIDs are pseudonymised rather than deleted, since a frame UID is how two files agree
  they share a frame of reference, and the pseudonym is stable so a cohort scrubbed file by file
  still joins. Only a *salted* run records `id_mapping: external` — an unsalted hash is recoverable
  by anyone holding the original UIDs. Dates shift rather than vanish so intervals survive, and
  running it twice does not shift them twice. It reads metadata and not pixels, so it writes
  `burned_in_annotation_checked: false` and lists what it did not check, rather than writing
  "de-identified".
- **A publishable conformance suite.** `medh5 conformance publish` writes a standalone directory —
  103 cases, `expected.json`, the §15.2 code table as data, the JSON Schema, `SHA256SUMS` and a
  README — so an implementer needs nothing installed to be measured. `medh5 conformance score`
  scores any validator, in any language, from `[{file, errors, warnings}, ...]`. `medh5 validate
  --json` emits a superset of that shape, so the reference implementation is scored through exactly
  the same door as everybody else, and a test asserts it.
- **Thirteen documentation pages** written against the 1.0 API, every snippet executed against a
  real sample rather than proofread.

### Changed

- **The 0.x implementation is gone.** 1.0 ships a *reader* for the old layout (~200 lines,
  documenting the format in full) so `medh5 migrate` still works, but not an implementation of it:
  shipping the old package inside the new one would let a curator keep writing the format they are
  migrating away from. The `medh5-0x` console script is removed.
- Eight specification clauses were corrected because implementing them showed the text was not
  implementable, or contradicted itself: `/meta` cannot be compressed, the label-set canonical
  serialization is now defined, `content_id` excludes `created`/`generator`, "the digest of an
  annotation" is defined for a multi-dataset group, the `det` profile requires a *detection-task*
  annotation rather than any §8 kind, §9's `class_ids` dataset and attribute are explicitly
  distinguished, `E010` was added because §2.2 stated a MUST with no code to report it, and W909 is
  stated to be sample-scoped. See Appendix C of the specification.
- `add_segmentation` now keeps every class named in `annotated_classes` expressible, encoding an
  empty one rather than dropping it. Without that, "searched for and not found" collapsed into
  "never looked for" — the distinction §11.3 exists to preserve.

- `writer.split()` now **replaces** a claim for the same `set_id` instead of appending one. Two
  claims for one set is precisely the W906 conflict §12.3 defines, so appending on a re-split
  manufactured the defect the validator exists to catch.
- `set_quality(issues=[...])` accepts constructed `Issue` and `Agreement` records as readily as
  JSON, instead of raising `TypeError: 'Issue' object is not subscriptable`.

### Fixed

- `medh5 dataset check` reported `C201` on the split it had just written, because the manifest
  digest covered `content_id` and `--write-claims` rewrites every file. The digest now covers
  membership and grouping; content drift is a separate question answered by `C401` and `--deep`.
- Stratified splitting put every group in `train` on small cohorts: each stratum was dealt against
  its own tally, and every small stratum rounds that way. Fixed by interleaving strata against one
  global tally, with the lead stratum rotating per round so the small partitions do not fill from
  the same stratum every time.

### Notes

- `content_id` is a Merkle digest over *stored object digests*, so editing a dataset without
  restamping it breaks that object's digest and leaves the root matching. `verify` therefore checks
  every object rather than only the root, and there is a test that says so.
- Reading a case at a level deeper than the conformance manifest declares is **not** safe: 71 of
  the invalid cases are built by editing a valid file, so an integrity pass adds a `content_id`
  mismatch the case never claimed. The published README says so; it originally said the opposite,
  and running it corrected that.

**COCO was dropped from 1.0.** It has no world geometry, no spacing and no frame of reference, so
importing one means inventing a grid and exporting one means discarding the geometry that makes a
medical annotation reproducible. Every other converter here is built on not telling that kind of
silent lie. A 2-D-native path can be added in a minor version — §3.6 already supports 2-D grids.

## [0.6.0]

Hardening pass driven by the napari-medh5 plugin integration report:
clearer single-open diagnostics, process-shared read handles that
obsolete downstream registry workarounds, tri-state checksum
verification so audit UIs can distinguish "no checksum" from "verified
good", and a handful of small ergonomic additions that several
downstream consumers had been re-implementing locally.

### Added

- **`medh5.open_shared(path)`**: ref-counted read-only context manager.
  Multiple callers in the same process (and across threads) share a
  single underlying `h5py.File`; the handle closes only after the last
  caller releases it. Keyed by `Path.resolve()` so symlinks share a
  handle. Replaces hand-rolled handle registries in lazy-read consumers
  (napari plugins, dashboards, viewers).
- **`medh5.VerifyResult`**: `StrEnum` with `OK`, `MISSING`, `MISMATCH`.
  `MEDH5File.verify()` now returns this enum so callers can distinguish
  "no checksum was ever stored" from "checksum verified successfully"
  — the two cases previously both returned `True`, making trustworthy
  audit UIs impossible to build.
- **`medh5.validate_bboxes(bboxes, sample_shape)`**: public clamping
  helper. Returns `(clamped, issues)` where issues is a list of
  `(index, axis, reason)` tuples describing every `"min<0"`,
  `"max>shape"`, or `"min>max"` adjustment applied. Shape mismatches
  raise `MEDH5ValidationError`.
- **`SpatialMeta.as_affine(ndim)`**: compose
  `direction · diag(spacing) + origin` into an `(ndim+1, ndim+1)`
  homogeneous matrix, or return `None` when the rotation is effectively
  identity so consumers can fall back to simpler `scale`+`translate`.
  Obsoletes ~30 lines of hand-rolled affine composition that every
  viewer-style consumer was writing.
- **`on_reopened` callback on `MEDH5File.update` / `update_meta` /
  `add_seg` / `set_review_status`**: fired with `path` only after the
  HDF5 write handle has closed successfully. Lets lazy-read consumers
  re-acquire handles or rebind cached views without reinventing an
  event system.
- **`ValidationIssue.location`**: optional `str | None` field
  (e.g. `"images/CT"`, `"seg/tumor"`, `"bboxes"`,
  `"extra.nnunetv2.labels"`). `_validate_open_file` populates it at
  every error/warning site so downstream UIs can highlight the offending
  dataset without re-parsing `message`. Non-breaking — `to_dict()`
  omits the key when it is `None`.
- **Subsystem `schema_version` stamping**: `set_review_status` stamps
  `extra["review"]["schema_version"] = 1`; the nnU-Net v2 converter
  stamps `extra["nnunetv2"]["schema_version"] = 1`. `read_meta` emits
  a `UserWarning` when a subsystem's stamp is newer than this library
  understands, so consumers can fail loudly on schema drift instead of
  silently mis-rendering.
- **Malformed-`extra` warnings**: `read_meta` validates the shape of
  well-known subsystems (`review.status` must be str;
  `nnunetv2.labels` must be `dict[str, int]`; subsystems must be
  dicts) and emits `UserWarning` on mismatches. The raw payload is
  preserved so consumers can still introspect.
- **Initial "pending" in review history**: `set_review_status` now
  always records the prior state (treating absent as `"pending"`), so
  the audit trail captures the sample's pre-review life from the very
  first call.
- **Clearer single-open diagnostics**: `MEDH5File.update` and
  `set_review_status` detect HDF5's "file is already open" /
  "unable to lock file" errors and raise
  `MEDH5FileError("'{path}' is already open in this process; close
  other MEDH5File handles before …")` with the original as `__cause__`,
  instead of passing the raw h5py message through. Docstrings document
  the exclusive-access requirement and point at `open_shared` for the
  cooperative read side.
- **`"Choosing the right read API"` section** in `docs/python-api.md`:
  table comparing `read()` / `read_meta()` / `MEDH5File(path)` context
  manager so consumers pick the right path the first time.
- **Tests**: expanded to 255 passing (92% coverage) — new
  `tests/test_shared.py`, `tests/test_validate.py`,
  `tests/test_bbox_validation.py`, `tests/test_meta.py`; the existing
  update/review/integrity/io/cli suites now exercise `VerifyResult`,
  `on_reopened`, initial-pending-in-history, and `location` field
  propagation.

### Changed

- **BREAKING**: `MEDH5File.verify(path)` now returns `VerifyResult`
  instead of `bool`. Callers that did `if MEDH5File.verify(p): ...` must
  switch to `if MEDH5File.verify(p) is VerifyResult.OK: ...` (or the
  looser `is not VerifyResult.MISMATCH` for the previous semantics).
  `verify_checksum(f)` in `medh5.integrity` returns the same enum.
  Per the project's pre-1.0 policy in CLAUDE.md, backward compatibility
  is not guaranteed.
- **`set_review_status` returns `ReviewStatus`** instead of `None`, so
  UIs can refresh without re-reading the file. Non-breaking for callers
  that ignored the return value.
- `MEDH5File.__init__` now closes the underlying h5py handle if any
  post-open assignment ever raised (belt-and-braces — impossible in
  practice today, but removes the last bare-open-without-with in the
  module).
- `medh5` CLI `audit` and `recompress --checksum` routes migrated to
  `VerifyResult`: audit still passes on `OK` or `MISSING` (checksums
  remain opt-in); `recompress --checksum` now requires `OK` after
  post-write verification (was "not False", which let `MISSING` slip
  through).

### Fixed

- `set_review_status` history no longer skips the sample's entire
  pre-review life — the first call now records the implicit initial
  `"pending"` state before overwriting it with the user's chosen
  status.
- `MEDH5File.update` no longer leaks the HDF5 write handle on
  post-open exceptions from `__init__` field assignments (defensive;
  no known trigger pre-fix).

## [0.5.0]

First PyPI release. Bundles the 0.4.0 work (never released) with a
dedicated release-hardening pass covering data-safety, PyTorch
multiprocessing, spatial-metadata validation, statistics numerics, CLI
exit codes, packaging, and adds the nnU-Net v2 dataset converter and a
post-review refactor round that split the CLI into a package and
consolidated duplicated helpers.

### Added

- **Atomic writes**: `MEDH5File.write()` now writes to a sibling temp file,
  `fsync`s, and `os.replace`s into place. An interrupted write (Ctrl-C,
  OOM, crash) can no longer leave a truncated `.medh5` file at the
  destination path. Any pre-existing file at the target path is preserved
  on failure.
- **Checksum verification before in-place updates**: `MEDH5File.update()`
  (and by extension `update_meta`, `add_seg`, `set_review_status`) now
  verifies any stored SHA-256 *before* mutating, so an externally
  corrupted file cannot silently have a fresh checksum baked in over top
  of the corruption. New `force=True` escape hatch for intentional
  repairs.
- **Fork/spawn-safe PyTorch handle cache**: `medh5.torch._HandleCache` is
  now PID-scoped — a forked worker observes the PID mismatch and resets
  to a cold cache instead of inheriting parent h5py state. Works
  transparently with `multiprocessing_context="spawn"` (default on macOS
  / Windows / Python 3.14+).
- **`medh5.torch.worker_init_fn`**: the supported `DataLoader(
  worker_init_fn=…)` helper for `num_workers > 0`. Documented in
  README.
- **`PatchSampler(include_bboxes=True)`**: opt-in bbox return from
  `PatchSampler.sample()`. Bboxes are translated into patch-local
  coordinates and filtered to the ones intersecting the patch;
  `bbox_scores` / `bbox_labels` are filtered consistently.
- **`RandomFlip` geometry sync**: flipping now negates the corresponding
  column of `meta.spatial.direction` (via `dataclasses.replace`, so the
  file's cached `SampleMeta` is not mutated) and mirrors any bboxes in
  the sample dict, keeping physical-space metadata consistent with the
  flipped voxel data.
- **`MEDH5File.is_valid(path)`**: thin convenience wrapper returning a
  plain `bool` for the common "is this file OK?" check (swallows
  `MEDH5ValidationError`).
- **Dimension checks in `SampleMeta.validate()`**: `direction` must be
  `ndim × ndim` and `axis_labels` length must equal `ndim`. A malformed
  `direction` attribute on read now raises `MEDH5SchemaError` instead of
  emitting a warning.
- **Numerically-stable parallel stats**: `compute_stats` now accumulates
  per-file `(n, mean, M2)` via Welford and merges with Chan's parallel
  algorithm. Large uint16 CT volumes no longer suffer catastrophic
  cancellation on variance.
- **CLI exit codes**: `medh5 <no args>` and unknown subcommands return
  exit code 2; runtime errors (`MEDH5Error`, `ValueError`, `ImportError`)
  return 1; success returns 0. Replaced the `if cmd == …` ladder with a
  typed dispatch table (`_TOP_HANDLERS`, `_SUB_DISPATCH`).
- **macOS CI job**: `test-macos` on `macos-latest` + Python 3.12
  exercises the `spawn` multiprocessing path that the Linux matrix does
  not cover.
- **Release-build CI job**: runs `python -m build`, `twine check dist/*`,
  inspects the wheel for `medh5/py.typed` + `LICENSE`, and uploads the
  dist/ artifact.
- **PyPI packaging metadata**: authors, project URLs (Homepage,
  Repository, Issues, Changelog), classifiers (Development Status ::
  4 - Beta, Topic :: Scientific/Engineering :: Medical Science Apps.,
  Typing :: Typed), `package-data = {medh5 = ["py.typed"]}`,
  `license = {file = "LICENSE"}`. `LICENSE` file (MIT, Puyang Wang,
  2026) added to the repo root and bundled in both wheel and sdist.
- **Tightened lower bounds**: `h5py >= 3.10`, `hdf5plugin >= 4.1`,
  `numpy >= 1.24`. No upper bounds.
- **nnU-Net v2 dataset converters** (`medh5.io.nnunetv2`): `from_nnunetv2()`
  converts a raw nnU-Net v2 dataset folder (`imagesTr/`, `labelsTr/`,
  optional `imagesTs/`, `dataset.json`) into a directory of per-case
  `.medh5` files, bundling every channel and splitting the integer label
  volume into one boolean mask per foreground class declared in
  `dataset.json`. `to_nnunetv2()` is the reverse: it emits a raw nnU-Net
  v2 layout from a directory of `.medh5` files. The parsed `dataset.json`
  payload is stashed in each file's `extra["nnunetv2"]` so export is
  lossless — channel order, label integer values, and optional fields
  (`overwrite_image_reader_writer`, `regions_class_order`, `name`) all
  round-trip. Region-based (list-valued) labels are rejected with a clear
  error. Requires the `nifti` extra. Lazy-imported from `medh5.io`.
- **CLI nnU-Net v2 subcommands**: `medh5 import nnunetv2 <src> -o <dst>`
  and `medh5 export nnunetv2 <src> -o <dst>` with `--no-test`,
  `--compression`, `--checksum`, `--dataset-name`, and `--file-ending`
  flags.
- **`MEDH5File.is_valid(strict=...)`**: `is_valid()` now forwards a
  `strict` kwarg to `ValidationReport.ok()`, so callers that want the
  one-call "did this file pass cleanly, warnings included?" check can
  get it without building a report object themselves.
- **Deterministic `stats.compute_stats` sampling**: per-file percentile
  sample seeds now derive from a stable BLAKE2b digest of the file path
  instead of Python's hash-randomized `hash()`, so percentile estimates
  are reproducible across runs and across Python invocations.
- **Tests**: expanded to 217 passing (91% coverage), including
  `test_dataloader_workers[spawn]`, `test_patch_dataloader_spawn`,
  `test_interrupted_write_*`, `test_update_verifies_checksum`,
  `test_include_bboxes_*`, `test_randomflip_direction_sync`,
  `test_compute_stats_parallel_matches_serial`, `test_is_valid_*`,
  CLI exit-code tests, end-to-end `medh5 import dicom` CLI coverage,
  `TestFromNnunetv2`/`TestToNnunetv2` happy-path and silent-data-loss
  guards, and a `medh5 import/export nnunetv2` CLI round-trip test.

### Changed

- `MEDH5File.read()` returns `sample.seg = None` when the `seg/` group
  exists but is empty, and `read_meta()` reports `has_seg = False` in
  the same case — previously both could be inconsistent with file
  state.
- Bounding-box datasets are only Blosc2-compressed when `n > 64`;
  tiny bbox arrays are written raw to avoid per-chunk filter overhead.
- **`MEDH5File.validate()` no longer takes `strict`**: strictness is
  applied on the returned `ValidationReport` via `report.ok(strict=...)`,
  keeping the report layer policy-free. The one-call `is_valid()`
  shortcut accepts `strict` as described above.
- **CLI split into `medh5.cli` package**: the 819-line flat
  `medh5/cli.py` is now a package grouped by command —
  `cli/inspect.py` (`info`/`validate`/`validate-all`/`audit`/`recompress`),
  `cli/dataset.py` (`index`/`split`/`stats`), `cli/convert.py`
  (`import`/`export` subgroups), `cli/review.py` (`review set`/`get`/
  `list`/`import-seg`), and `cli/_common.py` for shared helpers. Each
  submodule exposes `register(sub)` and `dispatch(cmd, args) -> int | None`;
  `cli/__init__.py::main()` composes them. Public surface
  (`medh5.cli:main`, `python -m medh5.cli`) is unchanged.
- **`.medh5` suffix helper consolidated**: the duplicate
  `_validate_suffix` / `_SUFFIX` pair in `core.py` and `review.py` was
  hoisted into `medh5.meta` and re-used from both modules.

### Fixed

- `MEDH5File.write()` no longer leaves partial output when interrupted
  mid-write (see "Atomic writes" above).
- `MEDH5File.update()` no longer silently re-hashes corrupted data
  (see "Checksum verification" above).
- `MEDH5PatchDataset` + `DataLoader(num_workers > 0)` no longer
  deadlocks under `fork` or crashes pickling under `spawn`.
- `RandomFlip` no longer silently desynchronizes `meta.spatial.direction`
  from the flipped voxel grid; downstream NIfTI export and
  physical-space metrics now see consistent geometry.
- `compute_stats(workers > 1)` no longer suffers precision loss on
  large integer volumes.
- `medh5 <no args>` now returns exit code 2 instead of 0, unbreaking
  shell automation like `medh5 validate … || exit 1`.
- nnU-Net v2 import no longer silently drops voxels whose integer label
  is not declared in `dataset.json` — `_split_label_volume` raises
  `MEDH5ValidationError` listing the offending values, and rejects
  float label volumes that contain genuinely non-integer voxels while
  still accepting integer-valued floats (`0.0`, `1.0`, …).
- nnU-Net v2 export no longer silently drops seg masks whose names are
  not declared in the nnU-Net label map when merging back to an integer
  label volume; it raises `MEDH5ValidationError` and asks the caller to
  update `extra["nnunetv2"]["labels"]` or remove the extra mask.
- nnU-Net v2 export no longer silently omits per-file image channels
  that disagree with the dataset-wide channel set resolved from the
  first file's metadata; channel mismatches raise
  `MEDH5ValidationError` with a clear missing/extra report.

## [0.4.0]

Bundled into 0.5.0 — never released on PyPI. Entries below describe
work landed under the 0.4 development branch.

### Added

- **Structured validation** (`ValidationReport`, `ValidationIssue`): `MEDH5File.validate()`
  returns a report with typed error/warning codes instead of plain strings.
  Supports `strict` mode where warnings are treated as failures. `ValidationReport`
  is exported from `medh5`.
- **Unified update API** (`MEDH5File.update()`): single entry point for in-place
  metadata, segmentation (add/replace/remove), and bounding-box mutations.
  Automatically resyncs `image_names`, `shape`, `has_seg`, `seg_names`, `has_bbox`
  from file state and recomputes checksums when present.
- **DICOM series selection**: `from_dicom()` now accepts `series_uid` to select
  a specific series when multiple exist. Without it, the largest series is chosen
  deterministically. Available series UIDs are recorded in `extra["dicom"]`.
- **DICOM geometry validation**: strict checks for consistent
  `ImageOrientationPatient`, `PixelSpacing`, and uniform slice spacing across
  the selected series. Multi-frame and non-grayscale DICOM are rejected with
  clear errors.
- **DICOM modality LUT**: `apply_modality_lut` parameter (default `True`) applies
  RescaleSlope/RescaleIntercept before writing via `pydicom.pixels`.
  Disable with `apply_modality_lut=False` or `--no-modality-lut` on the CLI.
- **SimpleITK resampling** for NIfTI imports: `from_nifti(resample_to=...)` resamples
  all images and masks onto a shared reference grid. Supports `"linear"`,
  `"nearest"`, and `"bspline"` interpolators. Masks always use nearest-neighbor.
- **`import_seg_nifti()`** (`medh5.io`): import a NIfTI segmentation mask into
  an existing `.medh5` file with optional resampling and replace semantics.
- **Expanded checksum coverage**: SHA-256 now covers segmentation masks, bounding
  boxes, and critical metadata attributes — not just image datasets. Review status
  updates also recompute the checksum when one is stored.
- **JSON output on CLI**: `--json` flag on `info`, `validate`, `stats`, and
  `review get` commands for machine-readable output.
- **CLI flags**: `--strict` on `validate`, `--fail-fast` on `validate-all`,
  `--resample-to`/`--interpolator` on `import nifti`, `--series-uid`/`--no-modality-lut`
  on `import dicom`, `--resample`/`--replace` on `review import-seg`.
- **Dataset record fields**: `DatasetRecord` now includes `shape`, `spacing`,
  `coord_system`, `patch_size`, and `review_status`.
- **Metadata validation**: `SampleMeta.validate()` now checks `patch_size`
  length and element types.
- **`meta.py` attribute lists**: `_ROOT_META_ATTRS` and `_IMAGE_META_ATTRS`
  tuples canonically define which HDF5 attributes belong to the schema.
  `write_meta()` clears stale attributes before writing.

### Changed

- `MEDH5File.update_meta()` now delegates to `MEDH5File.update()` internally.
- `MEDH5File.add_seg()` now delegates to `MEDH5File.update()` internally.
- `_validate_file()` in `cli.py` replaced by `MEDH5File.validate()`.
- DICOM `_read_series()` returns provenance metadata (selected UID, available
  UIDs, instance count, LUT application status).
- `from_dicom()` now raises on missing `ImageOrientationPatient`,
  `ImagePositionPatient`, or `PixelSpacing` instead of falling back to defaults.
- CLI `main()` wraps all command handlers in a top-level
  `except (ImportError, MEDH5Error, ValueError)` for consistent error reporting.
- **Tests**: expanded from 135 to 167 tests (90% coverage).

### Fixed

- `ValidationPayload` type alias was defined after `if __name__ == "__main____"`
  in `cli.py`, making it unreachable during normal imports. Moved to module top.
- `_build_info_payload()` opened the file twice (once via `MEDH5File` context
  manager, once via `get_review_status()`). Now extracts review status from
  `meta.extra` inline.
- `_validate_open_file()` loaded the entire `bboxes` dataset into memory just
  to check its shape. Now reads only HDF5 dataset metadata.
- Duplicate attribute-name tuples in `integrity.py` (`_HASHED_ROOT_ATTRS`,
  `_HASHED_IMAGE_ATTRS`) now reuse the canonical tuples from `meta.py`.

## [0.3.0]

### Added

- **NIfTI converter** (`medh5.io.nifti`): `from_nifti()` and `to_nifti()` for
  round-trip conversion between NIfTI and `.medh5`. Automatically extracts
  spacing, origin, direction, and coordinate system from the NIfTI affine.
  Requires optional `nibabel` dependency (`pip install medh5[nifti]`).
- **DICOM converter** (`medh5.io.dicom`): `from_dicom()` ingests a DICOM
  series directory into `.medh5`, extracting spatial metadata from standard
  tags and storing selected DICOM attributes under `extra["dicom"]`. Requires
  optional `pydicom` dependency (`pip install medh5[dicom]`).
- **Dataset manifest** (`medh5.dataset`): `Dataset.from_directory()` scans a
  directory tree for `.medh5` files and builds a lightweight manifest (no
  array reads). Supports `filter()`, `save()`/`load()` (JSON), and staleness
  detection via file mtime/size.
- **Dataset splitting** (`medh5.dataset.make_splits`): reproducible
  train/val/test splitting with stratification (`stratify_by`), patient-level
  grouping (`group_by` with dotted-path support into `extra`), and k-fold
  cross-validation.
- **Dataset statistics** (`medh5.stats.compute_stats`): streaming
  per-modality mean, std, min, max, and percentiles (p01/p99) using Welford
  merge across files. Supports foreground-restricted stats via a named
  segmentation mask, label distribution counts, shape histograms, and
  segmentation coverage fractions. Multi-process via `ProcessPoolExecutor`.
- **Patch sampler** (`medh5.sampling.PatchSampler`): lazy, chunk-aligned
  patch extraction with three strategies: `uniform`, `foreground` (biased
  toward a named seg mask), and `balanced` (alternating). Caches foreground
  voxel coordinates per file for efficiency.
- **Pure-numpy transforms** (`medh5.transforms`): `Compose`, `Clip`,
  `Normalize`, `ZScore`, and `RandomFlip`. No torch or PIL dependency.
- **Patch-based PyTorch dataset** (`medh5.torch.MEDH5PatchDataset`): uses
  `PatchSampler` for lazy patch reads instead of full-volume eager loads.
  Configurable `samples_per_volume` for virtual dataset length.
- **Per-worker file handle cache** (`medh5.torch._HandleCache`): LRU cache
  (default 32 handles) shared by both `MEDH5TorchDataset` and
  `MEDH5PatchDataset`. Each DataLoader worker gets its own cache (forked
  process). Eliminates redundant `h5py.File()` opens across epochs.
- **Review/QA workflow**: `MEDH5File.set_review_status()` and
  `MEDH5File.get_review_status()` for tracking annotation review state
  (`pending`/`reviewed`/`flagged`/`rejected`), annotator, timestamp, and
  notes. Prior states are appended to a `history` list. Stored under
  `extra["review"]` (no schema change). `ReviewStatus` dataclass exported
  from `medh5`.
- **Batch CLI commands**:
  - `medh5 validate-all <dir>` — parallel validation of all `.medh5` files.
  - `medh5 audit <dir>` — parallel SHA-256 checksum verification.
  - `medh5 recompress <dir|file> --compression <preset>` — rewrite files with
    a different compression preset. Supports `--out-dir` or atomic in-place
    rewrite via tempfile + rename. Optional `--checksum` flag.
- **Dataset CLI commands**:
  - `medh5 index <dir> -o manifest.json` — build a manifest.
  - `medh5 split <manifest> --ratios 0.7,0.15,0.15 -o splits/` — split with
    optional `--stratify`, `--group`, `--k-folds`, `--seed`.
  - `medh5 stats <dir|manifest> -o stats.json` — compute dataset statistics.
- **Import/export CLI commands**:
  - `medh5 import nifti --image <name> <path> -o out.medh5`
  - `medh5 import dicom <dir> -o out.medh5`
  - `medh5 export nifti <file> -o <dir>`
- **Review CLI commands**:
  - `medh5 review set <file> --status <status> --annotator <name>`
  - `medh5 review get <file>`
  - `medh5 review list <dir> --status <status>`
  - `medh5 review import-seg <file> --name <mask> --from <nifti>`
- **Optional dependency extras** in `pyproject.toml`: `nifti`, `dicom`, `itk`.
- **Tests**: expanded from 62 to 135 tests (91% coverage).

## [0.2.0]

### Breaking Changes

- **Multi-modality images**: The `image` parameter in `MEDH5File.write()` is
  replaced by `images: dict[str, np.ndarray]`.  Each key is a modality name
  (e.g. `"CT"`, `"MRI_T1"`, `"PET"`).  All arrays must share the same shape.
- **On-disk layout**: Image data is stored under an `images/` HDF5 group
  instead of a top-level `image` dataset.
- **`MEDH5Sample.image`** is replaced by `MEDH5Sample.images` (a dict).
- **Schema version** remains `"1"` for the current multi-image layout.
- `SampleMeta` gains `image_names: list[str]`.

### Added

- **Compression presets**: `compression="fast"`, `"balanced"`, or `"max"` as
  a shorthand for `cname`/`clevel` pairs.
- **Context-manager protocol**: `MEDH5File` is now instantiable and supports
  `with MEDH5File("file.medh5") as f:` for typed lazy access via `f.images`,
  `f.seg`, `f.meta`.
- **Custom exceptions**: `MEDH5Error`, `MEDH5ValidationError`,
  `MEDH5FileError`, `MEDH5SchemaError`.
- **Write-time validation**: seg shape vs image shape, bbox count vs
  scores/labels, bboxes shape, clevel range, empty images dict.
- **Schema version checking**: reading a file with a future schema version
  raises `MEDH5SchemaError`.
- **`MEDH5File.update_meta()`**: update label, label_name, or extra metadata
  without rewriting arrays.
- **`MEDH5File.add_seg()`**: add a segmentation mask to an existing file.
- **`MEDH5File.verify()`**: verify SHA-256 checksum of image data.
- **`checksum=True`** parameter on `write()` to store a SHA-256 digest.
- **CLI**: `medh5 info <file>` and `medh5 validate <file>` commands.
- **PyTorch integration**: `MEDH5TorchDataset` in `medh5.torch` (optional
  dependency via `pip install medh5[torch]`).
- **`__repr__`** for `MEDH5Sample` and `SampleMeta`.
- **`py.typed`** marker for downstream type checking.
- **Chunk optimizer**: named `_CHUNK_OVERSHOOT_LIMIT` constant, optional
  L3 cache auto-detection.
- **CI**: GitHub Actions workflow (lint, typecheck, test on Python 3.10-3.12).
- **Tooling**: ruff linting/formatting, pre-commit hooks.
- **Tests**: expanded from 12 to 62 tests with pytest-cov.

### Fixed

- Removed unused `from copy import deepcopy` import in `chunks.py`.
- Malformed `direction` attribute now emits a warning instead of crashing.

## [0.1.0]

Initial release with single-image `.medh5` format, HDF5 + Blosc2 compression,
segmentation masks, bounding boxes, labels, spatial metadata, and chunk
optimization.
