## [2.0.2] — DATE

**A patch release, with no API or format change.** A valid sample is written as
2.0.1 wrote it, with the same `content_id`. The converters no longer overwrite
the files they read, and what gave a silently wrong answer — a merged class, a
fabricated negative, an identifier a scrub left behind, a validation that passed
a file `verify` failed — is now refused or reported, as Behaviour changes lists.
A hostile file can no longer crash the process that opens it, make it load a
library or exhaust its memory, and the torch loaders no longer serve wrong
targets.

### Security

- **A file can no longer crash the process that opens it by nesting groups.**
  Opening a file checked that it was self-contained with HDF5's recursive
  `H5Lvisit`, which has no depth limit: 30,000 nested groups crashed the process
  (SIGSEGV), and 10,000 took a minute. The check now walks the file itself, a
  group at a time, and a file nested more than 64 groups deep is refused at open
  (`E001`).
- **HDF5's filter plugins load only when asked for.** A dataset names its
  filters by number, and HDF5 loaded one it had not registered from the shared
  libraries in its built-in search path — on Windows
  `%ALLUSERSPROFILE%\hdf5\lib\plugin`, which any local user can create — as
  whoever opened the file. Blosc2 and Zstandard are compiled in, so plugins are
  now off, and a dataset whose filter is not built in fails to read with HDF5's
  "filter plugins disabled". `HDF5_PLUGIN_PATH` loads them as before, and
  `MEDH5_ALLOW_PLUGINS` (set, and not `0`) from HDF5's standard directory only,
  never the directory HDF5 was built in; both are read once per process.
- **A chunk can no longer decompress to far more than its dataset's chunk.** The
  Zstandard and Blosc2 filters allocated whatever size a chunk stated: 32 KiB of
  run-length blocks asked for a gigabyte, and a Blosc2 header can state 2 GB in
  a few bytes. Both now cap their output at twice the buffer HDF5 hands them
  plus 64 KiB, refusing a stated size over it before anything is allocated, and
  the Blosc2 filter checks that a chunk's header parses
  (`vendor/hdf5-blosc2/PATCHES.md`, patch 4).
- **A damaged Blosc2 chunk no longer tears down Blosc2's global state.** The
  filter's failure path called `blosc2_destroy()`, which frees the library's
  global context and destroys the mutexes a window read on another thread may be
  using, and a frame of no chunks made it free an uninitialised pointer. Both
  are fixed (patches 5 and 6), with tests that call the filter as HDF5 does.
- **A rewrite through a symbolic link rewrites the file the link names.**
  `amend`, `scrub --apply`, `recompress` and `fix` renamed their result over the
  link, which became an independent copy, and left the linked file as it was —
  after `scrub --apply`, the identifying original. The replacement is now
  written beside the real file and renamed over it, following a chain of links,
  and the link is kept (§14.4); a link in a sticky, world-writable directory
  that another user owns is refused, as Linux's `fs.protected_symlinks` refuses
  it.
- **`scrub` reads every name, and every copy of the document.** A second hard
  link's name, a soft link and its target, a committed datatype's name and
  attributes, and `/meta` under a second name or copied elsewhere all scanned
  clean. Each name and attribute is now scanned, a second name for `/meta` or a
  copy of the document is an identity finding that `--apply` removes, and what
  the scan cannot decode — strings nested in compound, array or variable-length
  types, and byte buffers or undecoded types outside the imaging groups — is
  reported `unreadable`, with every non-string dataset outside those groups
  named in `not_checked`. A file that scanned clean may not now.
- **An amendment keeps no copy of the identifiers it replaced.** It copied
  `/meta` by path, so a hard-link alias of it such as `x_acme/meta` kept the old
  document — the identifiers `scrub --pseudonymise-ids` had replaced — and
  `remap_frame_uids` left every replaced UID in the file's free space, where
  HDF5 keeps an overwritten value. The commit now re-links each alias to the new
  document and, after an alias or a remap, rewrites the file from its root.
- **`--pseudonymise-ids` replaces an old id wherever a string holds it.** A
  record number in a label, a note or an attribute outlived its pseudonym; every
  string in the file is now swept for each replaced id of four characters or
  more (a shorter one, such as `1` or `CT`, would be found in every number and
  modality).
- **Identifiers the scrub's tables could never match are found.** The key tables
  were stored unnormalised, so `nhs_number`, `birth_date`, `date_of_birth` and
  every `*_uid` key never matched, top-level `extra` and `acquisition` keys such
  as `PatientID` escaped the key rules, and a value that is wholly a date under
  an unlisted key, a UID inside text, the de-identification record and agents'
  `organization` were never examined. Each is now found, and shifted,
  pseudonymised or removed as its kind is.

### Fixed

- **A rewrite that fails leaves the file as it was.** A commit whose final close
  failed renamed the file into place anyway; it now fails the commit. A
  fixed-length string dataset widened to fit a longer value, as `scrub --apply`
  may widen one, is swapped in so that a failure at any step leaves the original
  and no temporary behind.
- **A dataset whose size does not fit a machine word is an error.** Its byte
  count, the product of a shape the file states, wrapped round, or panicked in a
  debug build.
- **A read-only file can be amended on Windows, and stays read-only.** The
  temporary file was made read-only before it was opened to be flushed, and
  Windows does not rename over a read-only file; the order is fixed, and a test
  amends a read-only file on every platform.
- **`W907` no longer advises `int16` for a float image holding NaN or
  infinity.** Only the finite values were tested, so an image with NaN outside
  its field of view was advised into a type that cannot hold it.
- **`repr()` of a closed `Sample` or `Image` says that it is closed.** It
  raised, and debuggers, loggers and test reports all ask for it.
- **Under `closure = "implicit"` the readers answer for a class's ancestors
  (§5.4, §9).** The closure was stored and never applied, so a classification
  asserting only `left_kidney`, with `kidney` examined, answered `kidney`
  negative — a false negative for exactly the pattern §9 recommends. `dense()`,
  `contains()`, voxel counts, bounding boxes, the index and
  `Classification.state()` now treat a class as present wherever a descendant
  is; what is stored (`labelmap()`, `value()`, `labels`) stays literal, and
  `explicit` is unchanged ([Annotations](docs/reference/annotations.md)).
- **`payload_to_masks` reports an inconsistent payload as an error.** It
  panicked with a `PanicException`, which `except Exception` does not catch; it
  raises `E405` or `E404`.
- **`has_ignore_region` agrees with `ignore_region`.** It answered `True` where
  the sibling ignore mask is empty, which ignores nothing.
- **`instances_from_masks` skips an empty mask.** It made an object of one,
  which `encode_instances` then refused; the class keeps its place in the
  numbering.
- **Integrity and strict validation fail a file in which no digest parses.** An
  early return dropped the `E703`, `E702` and `W905` already found, so
  `medh5 validate --level strict` exited 0 on a file whose pixels had changed
  while `verify` failed it. The corpus gains `E703-every-digest-malformed`, and
  a test holds validation and `verify` to one answer on every digest case.
- **The validator sees a broken container at every level.** A `meta`, `grids`,
  `images` or grid of the wrong kind (`E004`, `E008`, `E109`), a member that
  leads nowhere (`E001`, and `E201` when no image resolves), and an image or
  pyramid level that is not numeric (`E001`) passed all four levels, strict
  included. Each is now reported, and five such files join the conformance
  corpus, which has 163 cases (it had 157).
- **One unreadable chunk no longer hides the rest of the integrity report.** It
  ended the pass, hiding every `E701` and `E702` after it and the name of the
  dataset; it is now `E001` at its own path and the pass goes on, and every rule
  reports an object it cannot read and checks the next.
- **The validator checks what it passed over.** A missing `task`, `closure` or
  `class_ids` (§6.2) is `E412` — a missing `closure` was read as `explicit` —
  and `/meta` and `clinical/meta` must be UTF-8 (`E004`, `E801`), where a lossy
  read put U+FFFD in place of a bad byte and the document parsed.
- **An atlas registration no longer silences `W911`, and an unparsed `/meta` no
  longer floods the report.** A transform with one end in a frame no grid is in
  counted as relating two timepoints, so one registration to an atlas hid `W911`
  for visits nothing relates; and with no document to resolve against, every
  reference to a timepoint or label was reported (`E409`, `E413`) beside
  `/meta`'s own finding.
- **`diagnose` and `fix` see a digest that does not parse.** `verify` failed a
  file holding one while `fix` found nothing to fix; `Diagnosis.malformed` now
  lists such digests, and `fix --rewrite-digests` restamps them. `fix_paths`
  attempts every file and raises one error naming each that failed, where it
  stopped at the first, and `fix --rebuild-index` keeps each entry's
  `max_coords`, `seed` and occupancy planes, where it rebuilt with the defaults.
- **`Tracking[key]` raises `KeyError` for any key no track has.** A key that is
  no instance id, such as `-1` or `"7"`, raised `OverflowError` or `TypeError`,
  which `get` and `in` already answered as absent.
- **A timepoint date added after a recorded shift is reported.** A file that
  records a date shift had its dates passed over by every later scan, so a date
  added since was never seen; a date whose `days_from_baseline` implies another
  baseline than the shifted dates do, by more than a day and a half, is now a
  finding.
- **A pseudonym under a UID key is no longer a finding.** Each was reported
  again, so a scrubbed file holding a UID could not scan clean.
- **Every name a scrub pseudonymises is recorded in `uid_map`.** Staff and
  organisation names a strict scrub replaced, and names replaced inside text,
  were left out of the map, the only key to the pseudonyms a file now holds.
- **Cohort statistics survive a NaN voxel.** One NaN or infinite value (PET,
  resampled data) made the moments NaN, and the minimum and maximum depended on
  the order of the files and could drop a finite extreme. Non-finite values are
  left out of the moments and counted (`Moments.nonfinite`), and the result no
  longer depends on file order.
- **`extract` and `unpack` never write a member over its own collection.**
  `extract(shard, key, out=shard)` replaced the whole collection with that one
  member and exited 0, through the same path or a relative or linked alias of
  it, and `unpack` did the same to a shard named for one of its members. Every
  destination is now checked against the collection before the first is written,
  in Python and in `medh5 unpack`.
- **`C201` counts samples, not claims.** A sample with two stale claims counted
  as two.
- **The cohort scan follows symlinked directories, matches `.MEDH5` in any case
  and skips AppleDouble `._*` files.** It passed over the first two and took the
  third for samples; each linked directory is walked once.
- **A DICOM SEG with several segments of one label imports all of them.** Each
  segment replaced the one before, so every lesion but the last was lost without
  a warning. Segments of one class are combined — their union, or their maximum
  in a `FRACTIONAL` SEG — and two segments with one label no longer fail `E302`
  when classes are minted.
- **No converter writes its report or output over a file it reads or writes.**
  `from-dicom-seg --report case.medh5` replaced the sample it had just amended
  with JSON and exited 0, and an output naming an input replaced the input.
  Every converter now refuses such a path before it runs, comparing paths
  resolved and as files, so `sub/../x`, a symbolic link and a hard link are each
  caught, and writes its report atomically.
- **A non-ASCII name no longer aborts an import.** `from_dicom` kept the Unicode
  letters of a `PatientID` in the sample id, and `from_nifti` those of a file
  name, which failed `E003`; one shared rule folds them to ASCII (`Müller Hans`
  becomes `Muller_Hans`), and `from_dicom`'s subject id keeps the original.
- **`to_rtstruct` declares the character set of its text, and colours each ROI
  as the label set does.** Text beyond ASCII went out as Latin-1 under the ASCII
  default, and every ROI was red; such text is now UTF-8 under `ISO_IR 192`, and
  each ROI takes its class's colour.
- **A slice that will not decode is refused by name.** A pixel decoder's failure
  was a raw traceback naming no file; it is a refusal naming the series and the
  file, and a series in a transfer syntax no installed decoder reads is refused
  before any of it is decoded.
- **`import_seg_nifti` reports what it guessed, and Analyze and MGH files are
  refused by name.** It dropped its reader's notes, did not take
  `assume_geometry`, and recorded a command that does not exist as its
  provenance tool; nibabel reads Analyze, MGH and MINC through the NIfTI call,
  and the first NIfTI field read raised a bare `ValueError`.
- **An unrelated annotation's ignore region no longer erases a clinical slot's
  supervision.** A task slot took the ignore region of every voxel annotation on
  its grid, so an organ annotation ignoring the whole grid removed every lesion
  positive from the loss mask. Only annotations that label or examined one of
  the slot's classes contribute to `ignore[slot]` now; across several classes it
  is still one conservative union.
- **`PairedPatchDataset` labels and samples each pair at its own visits.**
  `label=` served one pair's change label as the target of every pair, and
  `annotation=` failed on every pair that did not start at its visit, as the
  longitudinal guide's own example did. The label is attached only to the pair
  it spans, and the annotation is used only at its own visit.
- **A missing multichannel optional slot batches beside a present one.** The
  placeholder's channel count came from its own row, so rows with and without a
  multichannel image could not be collated, even with fixed patches; it now
  takes the channel count of the slot's images across the task, read once from
  grid metadata.
- **Warm and cold handle caches agree after a pinned source is replaced.** A
  warm lease kept serving the replaced file's old bytes while a cold one refused
  it (`T302`), so in one epoch one worker served the old version and another
  refused the row. Each new pinned lease re-checks its file with one `stat`: a
  replaced file is refused (`T302`), or reread when it holds the same version
  (task-cache-1 §6).
- **`HandleCache.clear()` and `worker_init_fn` close the handles their own
  process opened.** They abandoned every handle, as a forked child must, so in
  the process that had opened them each stayed open until exit.
- **A broken PyTorch install is reported with its cause, and MONAI metadata
  names the source file.** Every import error read as PyTorch not being
  installed. `filename_or_obj` is now set, so MONAI's `SaveImage` names its
  output after the sample.
- **A release page breaks no line of its notes mid-sentence.** GitHub prints
  each newline in a release's notes as a line break, and the notes are this
  file's sections, wrapped at 80 columns, so every release page broke each
  paragraph at every line. The notes now put each paragraph on one line and
  render as the section does; `python .github/scripts/release_notes.py vX.Y.Z`
  writes a published release's notes again, for `gh release edit`.

### Behaviour changes

- **A missing file or a directory is reported in fixed words.** The report was
  HDF5's error stack, which carries the time and an address, so
  `validate --json` differed from run to run; it is `E001`, ending "no such file
  or directory" or "it is a directory, not a file".
- **`add_segmentation(encoding="mask")` is refused (`E404`).** It merged every
  class into one volume written with empty `class_ids` and
  `annotated_class_ids`, and the file validated clean. `encoding` takes `"auto"`
  or one of `TRANSCODABLE`, and `select_encoding` refuses any other preference.
- **A float mask holding a value other than 0 and 1 is refused (`E411`).**
  `add_mask`, `add_segmentation(masks=…, ignore=…)` and `InstanceInput` read 0.2
  as set; threshold the mask first, or pass a boolean array.
- **A commit refused before it changes anything leaves the writer open.** A
  refused commit — a sample with no image (`E201`), or a reference that does not
  resolve — aborted the writer, and a second `commit()` returned `None` as if
  the file had been written. The writer can now be mended and committed again,
  and committing an aborted one raises; in a `with` block a refused commit still
  aborts.
- **An amendment keeps the file's `digest_algo`.** A `sha512` or `blake2b` file
  was restamped in `sha256`, with a new `content_id`, or with `digests=False`
  left holding two algorithms; `fix --rewrite-digests` restamps in the file's
  algorithm too, where it failed such a file with `E701`.
- **`drop_clinical()` keeps the version of a sample without the clinical
  profile.** Only an amendment that removes the profile lowers the version (1.1
  §2.1); one with nothing to remove lowered it too.
- **`dense()` and `contains()` refuse class 0 (`E404`), as they refuse the
  ignore id.** Background is no class: a `labelmap` answered with every voxel no
  class holds, and every other encoding with none.
- **A coordinate or matrix entry that is not finite is an error.** The encoders
  refuse a box or oriented-box coordinate that is not finite (`E406`), which
  passed, and mesh offsets that do not delimit their data (`E408`), which were
  found only at commit; the validator reports a non-finite box bound (`E406`) or
  affine entry (`E504`) in a file.
- **An annotation whose ignore id or class-id tables contradict its classes is
  refused when opened.** A stored `ignore_id` outside 1..65535 or equal to a
  class id, a `bit_class_ids` that is not one bit per class, and a
  `layer_class_ids` without one row per layer read as false verified negatives —
  an `ignore_id` of 0 marked all background ignored. Each is refused at open and
  reported by the validator (`E404`, `E405`).
- **`transcode_payload` refuses what the writer refuses.** It converted a dense
  payload to `instances` and dropped an in-band ignore region, both of which a
  write refuses, and `check_roundtrip` now compares the ignore regions as well
  as the class masks.
- **A geometric annotation's coordinate columns are read once, and read-only.**
  The same array comes back on every access, so writing into one raises; copy it
  to edit it. A score that is not finite reads as `None` on the objects an
  annotation yields.
- **The conformance scorer compares severity.** A validator that reported every
  code as a warning, rejecting nothing, passed every case; an expected error
  reported as a warning now fails it ([Conformance](docs/spec/conformance.md)).
- **`rules_for` refuses a level it does not know.** It returned every rule for a
  misspelt level; Python raises `ValueError`, and the engine returns no rule.
- **`fix --rewrite-digests` refuses a file the validator rejects.** It restamped
  a structurally invalid file, a dangling `/images` say, which then passed
  `strict` with no code; the writer's commit gate now refuses it.
- **A file of another major is not read past its version.** Validation reports
  `E002` and stops, where this major's rules went on to judge a file that is not
  theirs, and a `medh5_version` that is not a string is `E001`.
- **Tracking no longer resolves an object an ignore region may hide.** An
  annotation with any ignored voxel no longer counts its classes as examined, so
  an object absent there is `unexamined`, not `resolved`, and a baseline that
  ignored the place no longer makes a lesion `new` (§7.4, §7.7, §11.3); the
  `medh5 track` legend says so. Where a review ignored only some slabs this says
  `unexamined` more often than it must, and never wrongly.
- **A class id with two meanings is refused, not merged.** Cross-file agreement
  scored `liver` in one file against `spleen` in another (a Dice of 1), and
  cohort statistics counted the two as one class in `class_weights`. Both now
  compare class keys and refuse a clash, naming the files (`C102` in agreement,
  as `dataset check` reports it); `ClassStats` carries each class's `key`,
  `medh5 dataset stats` prints it, and where neither side has a label set ids
  are compared as numbers.
- **Object agreement matches instance ids only within one sample.** Two raters
  who each number their lesions from 1 scored F1 = 1 for disjoint lesions;
  across files objects are matched by IoU, and a match by id reports its
  `threshold` as `null` in JSON.
- **Box IoU refuses boxes it cannot measure.** A 2-D box against a 3-D one
  scored about 10 and matched, and a box of no spatial axes scored 1; unequal or
  zero axis counts are a `ValueError`, a bound that is not finite or a `lo`
  above its `hi` is `E406`, and nothing is clamped. `compare_instances` raises
  `E414` across dimensions, and `Grid.comparable_with` is false for grids with
  different numbers of spatial axes.
- **A subject in two folds of one split is a leak.** K-fold claims are `train`
  in every fold but one, so a subject in folds 0 and 3 read `train` twice and
  passed `medh5 splits`, `Split.leaks()` and `dataset check` (`C202`); each now
  reports it, naming the place with its fold, as `train (fold 0)`.
- **A date shift of 0, or one other than the recorded shift, is refused.** A
  shift of 0 was recorded as a shift, after which every later scan passed over
  the file's dates; a recorded 0 now reads as no shift.
- **`scrub` refuses options that only `--apply` acts on.** `--date-shift-days`,
  `--salt` and `--by` without `--apply` did nothing, and so did `scrub_paths`'s
  apply options without `apply_changes=True`. `scrub_paths` takes explicit
  keywords, so a misspelt `apply=True` is a `TypeError` where it ran a scan, and
  it processes every path, then raises the first refusal with `.reports` set
  (`None` for a refused path).
- **The cohort tools say when they did not see the whole cohort.**
  `medh5 dataset stats` printed the files it could not read among the numbers
  and exited 0; it names them on stderr and exits 1, with `--json` still one
  document, and `scan(on_error="warn")` now issues the warning it promised.
- **Saved statistics with a `null` moment are refused on reading.** A NaN moment
  was written as `null` and read back as a mean and standard deviation of 0, so
  normalisation silently became (0, 1).
- **`write_claims` refuses a manifest the split does not place.** Claims stamped
  with a split's digest were written for a manifest the split was not made from;
  a filtered manifest is still accepted.
- **Sampling refuses an annotation it cannot draw from.** A misspelt name drew
  `uniform` windows without a word, and `balanced` failed only when its coin
  chose foreground; an unknown name is refused before any draw, and so is a
  non-voxel annotation where a draw may need foreground. `PatchSampler` checks a
  setting changed after construction as its constructor does, and a
  `sample_stride` below 1 is refused.
- **Every DICOM slice is held to where a regular stack puts it.** Each gap was
  held to 1 % of the spacing, which passed 4.5 mm of accumulated drift and
  refused positions rounded to two decimals; a slice must now lie within a tenth
  of the spacing, and at least 0.01 mm, of its place
  ([Import DICOM](docs/guides/import-dicom.md)). `SPACING_TOLERANCE` is that
  fraction, 0.1, and `read_dicom_seg` holds each plane to the same bound.
- **One unreadable series no longer aborts `from_dicom`.** A localizer or an RT
  Dose series refused the whole tree. A series that cannot be imported is left
  out with a `series` warning while the rest import, and the command exits 1; RT
  Dose is skipped with a note, multi-frame objects with a warning, and only a
  tree with nothing readable is refused.
- **A one-plane DICOM SEG is not given a 1 mm step.** Its
  `SpacingBetweenSlices`, else its `SliceThickness` (recorded as a decision), is
  the step, and a SEG stating neither is refused.
- **RTSTRUCT contours are read as their `ContourGeometricType` says.** Open
  curves were filled as areas, a flattened applicator marked its class examined
  with no voxels — a fabricated negative — and POINT ROIs vanished. POINT and
  open ROIs are left out with a warning, and crossing `CLOSEDPLANAR_XOR`
  contours, an ROI mixing types, and contours between or across slices or
  outside the grid are refused.
- **RTSTRUCT import uses each ROI's own frame of reference.** It applied the
  first frame the file listed to every ROI, so the result depended on the order
  of that list; ROIs in several frames are refused (`E414`) before the sample is
  opened.
- **`to_rtstruct` names each ROI for its class's display name, and warns where
  contours of one class cross.** It used the class key; a display name is used
  where every class has a distinct one that fits an `ROIName`, and the import
  resolves it. Crossing contours are a warning because medh5 means their union,
  and readers that combine an ROI's contours by exclusive-or cancel the overlap.
- **An export never turns an unexamined class into a verified negative.**
  `to_dicom_seg` wrote a segment for every class, and `to_nifti(class_key=)` a
  mask for any, so a class nobody examined came back examined and absent.
  `to_dicom_seg` writes only the classes the annotation examined, and both
  refuse a named class it did not examine.
- **A 2-D NIfTI plane off z = 0 is refused.** It was moved to z = 0, so two
  planes, or an image and a mask from different slices, could share one grid;
  save the plane as `(x, y, 1)` to import it as a one-slice volume.
- **A 5-D `(x, y, z, 1, n)` NIfTI is read as channels, not time.** Its timing
  came from the squeezed fourth axis, making a measured time series of it; an
  axis on `dim[5]` with nothing saying otherwise is a channel axis, recorded as
  a guess (pass `fourth_axis=` to say otherwise).
- **`to_nnunetv2` refuses a destination holding NIfTI files it would not
  write.** A re-export into one directory left removed cases, dropped channels,
  old file endings and a case moved to `imagesTs` still in `imagesTr`, leaking
  test into training, and `from_nnunetv2` brought the removed case back. An
  identical re-export is still allowed, and other files are left alone.
- **A converter's report is refused inside a tree it reads or writes.** A
  `--report` inside the `from-dicom` or `from-nnunet` input tree, or inside the
  nnU-Net dataset `to-nnunet` writes, is refused before the command runs.
- **The torch loaders refuse, by name, what they misread.** A non-voxel
  annotation failed with `AttributeError` in every label format and in
  `medh5.monai.to_dict`, and `label=` naming an annotation that is not a
  classification, or `images=` not at `timepoint=`, were read silently. Each is
  now refused, naming the annotation or image.
- **A clinical slot whose images disagree in channel count is refused at
  construction (`T102`).** Where some rows leave the slot empty, no placeholder
  can match them all. `collate_clinical` raises `MEDH5ValidationError` (still a
  `ValueError`) for a shape mismatch, and says to give the slot a `patch` or use
  `batch_size=1`.

### Performance

- **Window reads and attribute checks ask HDF5 less.** A windowed read queried
  the file's size once per chunk, under HDF5's lock, and now once per read;
  checking that an attribute exists listed every attribute name, and now asks
  HDF5 directly.
- **Writing `layers` measures the overlap graph once, and image reads copy
  less.** The writer analysed a `layers` annotation twice; `read(physical=True)`
  converts and rescales in one pass, and a read in the stored dtype no longer
  copies. Values are identical.
- **Label reads read each volume, layer or bit plane once.** `dense()` re-read a
  `labelmap` window for every class, and the index, `class_bboxes` and `analyse`
  walked each class over the whole volume: a 60-class `dense()` is about ten
  times faster, and `build_index`, `class_bboxes` and `analyse` 15 to 200 times.
  Repeated access to boxes, `polygons()` and `instance(id)` no longer re-read or
  decode every object, and every output is identical, `content_id` included.
- **Digests decode each chunk once.** Their slabs did not follow chunk
  boundaries, so each chunk was decoded two to four times; the digests are
  identical.
- **Agreement and split audits scale with the data.** `compare_voxel` decoded
  the annotation once per class: at 40 classes over a million voxels, `layers`
  takes 0.38 s instead of 1.15 s and `bitmask` 0.38 s instead of 5.12 s. Finding
  anatomy units was quadratic, and 40,000 of them take 0.28 s instead of 2.43 s.
- **A scrubbed file is rewritten once.** `--apply` compacts the file at its
  commit instead of repacking it whole a second time.
- **Cohort statistics read chunk-sized slabs, and split lookups are cached.**
  Statistics read plane by plane, decompressing each chunk once per plane, and
  now give bit-identical results from whole chunks. `Split.partition_of` and
  `fold_of` rebuilt the split on every call and are cached, the cohort guide
  lists a partition with `split.paths()`, and pack and `dataset check` use hash
  lookups.
- **A DICOM series is checked before any of it is decoded.** `read_series`
  checks every header first and decodes one slice at a time into the volume,
  where it held the decoded series about three times over before the first
  check.
- **The torch `instances` format no longer decodes every object mask for each
  patch.** It reads the object columns instead, with the same output.

### Documentation

- **The import guides check an import with `--level integrity`.**
  `--level strict` promotes `W903` and `W912`, which every fresh import carries,
  so the check the DICOM and NIfTI guides gave failed on every converted file.
  The DICOM guide states the slice tolerance and the NIfTI guide that a 2-D
  plane lies at z = 0, and the converters reference says which RTSTRUCT contour
  types import.
- **The export guide says what has no exporter.** It promised lossless round
  trips to DICOM, which has no image exporter; it now lists what none writes —
  images to DICOM, boxes, keypoints, meshes, transforms and clinical data — and
  what an existing nnU-Net destination does.
- **The de-identification guide states its coverage exactly.** Its suggested
  publication gate passed open identity findings, clinical samples and anything
  not examined, as not actionable; it blocks on them now
  ([De-identify](docs/guides/deidentify.md)).
- **The conformance page says that severity is scored.** It also says that 1.1
  reclassifies some files a pure 1.0 validator reports as `E007` (`E009`,
  `E803`), and cites §2.2 for collections; the code table's summaries of `E004`,
  `E008`, `E109`, `E406`, `E504` and `E702` say what each now covers.
- **The torch reference states the loaders' contracts.** It now covers voxel
  annotations only; the scope of a pair's `label=` and `annotation=`;
  `ignore[slot]` as a conservative union; a missing slot's placeholder and
  `patch=None`; that every new lease re-checks its source while the unpinned
  datasets can mix versions of a file across an amend; and `clear()`,
  `timepoint=` and `filename_or_obj` ([PyTorch](docs/reference/torch.md)). The
  longitudinal guide's `PairedPatchDataset` example runs as written.
- **The annotations reference says what `closure` means to each reader.**
  `brats-subregions` uses the BraTS 2023 numbering, and its description now says
  so: in BraTS 2021 and earlier label 4 is the enhancing tumour, which the
  vocabulary numbers 3, and its own id 4 is a synthetic parent.
- **The writer's docstrings say what it does.** `Image.read(physical=True)`
  returns `float32` only for an image that declares a rescale and reads any
  other as stored, an amendment on Windows needs every reader of the file closed
  before it commits, and `written_version` names its `drop_clinical` exception.
- **The `medh5-sys` documentation no longer announces "three things" before a
  list of five.** It says what the crate links, and that the engine turns HDF5's
  dynamically loaded plugins off unless asked.

<!-- END OF CHANGELOG TEXT -->

## Notes for the coordinator (not CHANGELOG text)

### Discrepancies

- scrub, "strict name pseudonyms are salted and recorded in `uid_map`": they
  were already salted (`pseudonymise(name, salt)`, `b193fbe~1`); only the
  recording in `uid_map` is new. Corrected, and moved from Security to Fixed.

- annot, "a NaN score reads as None": the reader drops any score that is not
  finite, infinities too (`annotations/read.rs`). The entry says "not finite".

- valid's drafts leave out four merged changes, added from `82d1631`: `W911` no
  longer silenced by one atlas registration; no `E409`/`E413` cascade after an
  unparsed `/meta`; `clinical/meta` read strictly as UTF-8 (`E801`); `E201` when
  no image member resolves.

- conv2's drafts leave out the `import_seg_nifti` provenance fix (it recorded
  `medh5 convert import-seg-nifti`, a command that does not exist); folded into
  its entry.

- sec: the commit message says the filters cap output "at the chunk HDF5
  expects"; the code caps at twice the buffer HDF5 hands the filter plus 64 KiB,
  as the report says. The entry follows the code.

- conv1 lists "ROINames are now display names" as a behaviour change and "keeps
  the label set's names and colours" as a fix: split as Fixed (character set,
  colours) and Behaviour changes (display names, crossing warning).

- dataset: the audit's figures (stats 0.60 s to 0.03 s; 34 s for the split loop
  at 3,000 subjects) were not re-measured by the workstream, so the entry gives
  none.

Every other drafted claim matched its merged commit: `17bd3e9` (writer),
`b157a90` (conv1), `69e81e0` (sec), `f6e04ca` (curation), `b193fbe` (scrub),
`6cf90c9` (annot), `82d1631` (valid), `6f68e5c` (dataset), `280f014` and
`c3b9f0e` (conv2), `a7351b6` (torch). The writer commit is the 17th from
`a7351b6`, one past `-16`.

### Deferred

- sec, S1 (PARTIAL): `.github/scripts/damaged.py` was not extended with
  structural hostile inputs (deep nesting, an unknown filter id, a Zstandard
  bomb); a safety classifier stopped the agent. Rust tests cover all three; the
  user decides.

- dataset, item 4 (SKIPPED): manifest paths still resolve against the working
  directory, and `root` is recorded but unused. Fixing it needs a
  manifest-format decision (a `paths_relative_to` marker); deferred to 2.1.0.

Reported DONE, but narrower than the item, or handed off:

- writer, item 9: `Image.read(physical=True)` keeps returning the stored dtype
  for an image without a rescale; documented, not changed (as the prompt
  preferred).

- curation, F15: the simple rule (any ignored voxel makes the class unexamined);
  the precise "provably misses" rule is not done.

- conv2, F11: Python converters only; native `migrate` and `clinical export -o`
  belong to legacy and clin (pending).

- valid handoffs, not fixed: a dangling image raises `KeyError` and an image of
  strings `TypeError` at read (should be `MEDH5Error`); an undigested dataset
  under a `content_id` fails `verify` with no validator code.

- writer: an unknown `digest_algo` (`E703`) still falls back to sha256;
  `check_held` opens `self.path`, not the amend source (unreachable today).

- conv1: two RTSTRUCT ROIs with one name still fail `E302` when classes are
  minted.

- torch: per-class ignore masks are 2.1.0; `ignore[slot]` stays one union across
  a slot's classes.

### Open questions

- "No API change" does not hold for the Rust crate. Public signatures changed:
  `curation::agreement::box_iou` and `box_iou_f64` return `Result<f64>` (were
  `f64`); `annotations::select_encoding` returns `Result<String>` (was
  `String`); `curation::scrub::IDENTIFYING_KEYS` is `[&str; 76]` (was 78) and
  `DATE_KEYS` `[&str; 14]` (was 16); and `Moments.nonfinite`, `ClassStats.key`
  and `Diagnosis.malformed` are new public fields of structs a caller can build
  by literal. Under Cargo's SemVer rules each can break a dependent on
  `medh5 = "2.0"` that uses it, and `cargo update` takes 2.0.2 unasked. Either
  keep the old signatures (say, `box_iou` beside a new `try_box_iou`) or qualify
  the lead ("no format change, and no Python API change") and add, under
  Behaviour changes: "**Three Rust functions return a `Result`.**
  `curation::agreement::box_iou` and `box_iou_f64` refuse boxes they cannot
  measure, and `annotations::select_encoding` a preference it does not know,
  where each answered; `Moments`, `ClassStats` and `Diagnosis` gain a public
  field each."

- Security entries carry their own behaviour changes (plugins opt-in, the
  64-group refusal, the sticky-directory link refusal, the scrub's new findings)
  and are not repeated under Behaviour changes. Add one-line cross-references
  there?

- The symbolic-link fix is sec's Fixed item; I put it under Security, because
  `scrub --apply` through a link left the identifying original. Confirm.

- Finding ids (F01 to F22) and "top issue" numbers are left out: they cite the
  audit in `.wip-2.0.2/`, which is deleted before release. Map: F01 and F09 are
  the two validation entries; F03 and F04 the scrub and amendment entries; F05
  the 2-D NIfTI plane; F08 nnU-Net; F10 extract/unpack; F11 converter paths;
  F12, F13, F20 and F21 torch; F14 closure; F15 to F18 curation.

- Policy for the sections: an item a report lists under both Fixed and Behaviour
  changes is under Behaviour changes only; a data-loss fix made by a refusal
  (converter paths, extract/unpack) stays in Fixed. Most of curation's F15 to
  F18 therefore sit under Behaviour changes.

- The Windows read-only fix has never run on Windows; CI's `test-windows` job
  will be its first run. Keep the entry only if that job passes.

- The annot timings compare a release build (before) with a dev opt-level-1
  build (after), so the entry gives ratios, as the report does. Re-measure on
  one build if you want 2.0.0-style figures.

- The `.wip-2.0.2/README.md` records pytest, cargo test, the corpus and fmt,
  ruff and mypy at `a7351b6`, but no `cargo clippy` or `mkdocs build --strict`
  on the merged tree. The draft's links are `docs/` pages that exist, with no
  anchors.

- writer's question for spec: is an index's `source_digest` always sha256
  (§13.3), or the file's `digest_algo`? An amendment now keeps the algorithm;
  the index digest does not follow it.

### Layout, for the later reports

Fixed, Behaviour changes and Performance run in one order of areas: files and
HDF5; writing and images; annotations and labels; validation, integrity and
conformance; curation (tracking, agreement, splits); de-identification; cohorts,
collections and sampling; converters; torch and MONAI; and, in Fixed only,
release tooling (the entry from `[Unreleased]`). Security runs hostile files,
then rewrites, then de-identification. Suggested slots: pyapi after writing and
images; geom after annotations; spec after validation (2.0.0 put specification
fixes early); legacy and cli after converters; comp with torch; clin after
cohorts; ci beside the release-page entry; docs under Documentation.
