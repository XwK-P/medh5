# 2.1.0 decisions

The specification and API decisions 2.1.0 implements, settled by the
coordinator. Each says where it applies, what was decided and why, the text to
write (where there is text), and which workstream implements it. "Text to
write" is the normative text: put it in the clause named, as given, and add the
erratum row. If implementing it shows the text wrong, stop and ask.

Spec references: "1.0 §n" is `docs/spec/medh5-1.0.md`, "1.1 §n" is
`docs/spec/medh5-1.1.md`, "TC §n" is `docs/spec/task-cache-1.md`. Erratum rows
go in 1.0 Appendix C.1, 1.1 Appendix A, or TC Appendix A (created by D19).

**No new format version.** Every decision below is an erratum to a published
version, a new diagnostic code allocated by the rule of D1, or API. 2.1.0
writes 1.0 and 1.1 files only: what 2.0.2 writes, 2.1.0 writes with the same
content and `content_id`, unless an item fixes it (D20).

## Format and specification

### D1 Code-table revisions

- **Where.** 1.0 §15.1, §15.2, §16, Appendix C.1; 1.1 §11.2;
  `crates/medh5/data/codes.json`. Implemented by W21-02 PR 1, with the scoring
  side in W21-03. The pull request that allocates the first revision-2 code
  (W21-02 PR 2's W915, or W21-13 PR 2's W919 if it merges first) makes the
  table revision 2: it changes the last sentence below from "revision 1" to
  "revision 2", marks its row (r2), and sets the table's revision in
  `codes.json`. Every later revision-2 code only marks its row.
- **Decision.** Errata may allocate codes, in numbered revisions of the code
  table. A code never changes a file's `medh5_version`.
- **Why.** §15.2 and §16 say only a minor version adds codes, yet 2.0.0 already
  added E506 and E704 by erratum, and every condition W915–W919 report exists
  in 1.0 files: holding them for a 1.2 would report them only on files that
  declare 1.2. Allocating ad hoc (as E506 and E704 were) silently moves a
  third-party validator's conformance score and a `--level strict` CI verdict.
  Numbered revisions let both be pinned.
- **Text to write**, replacing the last sentence of 1.0 §15.2's opening
  paragraph ("The table below is complete for 1.0; a minor version may add codes
  but **MUST NOT** redefine one."):

  > Codes are allocated in two ways. A minor version allocates the codes for
  > what it adds (§16; 1.1 §11.2). An erratum to a published version allocates
  > a code for a rule that version already states: an error only for content
  > its text already makes invalid; a warning for a recommendation (SHOULD,
  > SHOULD NOT), or for a condition the text names without requiring anything
  > of it. No allocation narrows, redefines or renumbers a code or changes its
  > severity, and a retired code is never reused. A minor version that adds a
  > kind reports its datasets' defects under the codes that report the same
  > defect for the kinds before it, and names the kind in those codes'
  > summaries: what a code applies to widens, what it means does not. Every
  > allocation, by a minor version or an erratum, is a *revision* of the
  > table, recorded in Appendix C.1: revision 1 is every code allocated before
  > revisions were numbered, and each later allocation is the next number.
  > `codes.json` gives each code the version whose rules it reports (`since`)
  > and the revision that allocated it (`revision`). A validator states the
  > revision it implements, and the conformance suite scores it against that
  > revision. The table below is complete for 1.0 at revision 2; the codes
  > revision 2 allocated are marked (r2).

  Until a revision-2 code exists, W21-02 PR 1 writes the last sentence as "The
  table below is complete for 1.0 at revision 1." and marks nothing.

  Add to the first bullet of 1.0 §16, after "A **minor** bump may add objects,
  attributes, `kind` values, activity types and error codes.":

  > A code for a rule a published version already states may instead be
  > allocated by an erratum (§15.2); it changes no file's version.

  Add to 1.0 §15.1, after the levels table (the `strict` row is a table cell,
  so the sentence is a paragraph below the table):

  > A file that passes `strict` under one revision of the code table (§15.2)
  > may fail under a later one; a validator offering `strict` lets its caller
  > fix the revision.

  1.1 §11.2, after "…which they extend without redefining any code.": "They
  belong to revision 1 of the table (1.0 §15.2)."
- **Data.** `codes.json` is already an object, `{"$comment": …, "codes":
  [...]}`, and every reader takes its `"codes"` member (`codes.rs`'s `Table`,
  `medh5/errors.py`, `hooks/mkdocs_hooks.py`, `tests/format/test_conformance.py`),
  so a top-level `"revision"` member is free: `1` from W21-02 PR 1, `2` from the
  first revision-2 code. Each entry gains `"since": "1.0" | "1.1"` and
  `"revision": 1 | 2`. `codes::Code` gains no field (it is exhaustive): the
  engine reads the new members through functions. The suite's own `codes.json`
  is rebuilt entry by entry (`conformance/suite.rs`, which writes `code`,
  `severity`, `domain`, `summary`): it must carry `since`, `revision` and the
  table's revision too.
- **Consequence.** The conformance suite's `expected.json` records each case's
  format version and the suite's code revision; scoring can be bounded by both
  (W21-03).

### D2 New codes, revision 2

| Code | Severity | Domain | Level | Since | Decision | Summary for the table |
|---|---|---|---|---|---|---|
| W915 | warning | integrity | integrity | 1.0 | D6 | a dataset inside an object `content_id` covers carries no `digest` |
| W916 | warning | annotations | semantic | 1.0 | D7 | under `implicit` closure, a class asserted absent where an `is_a` descendant is asserted present |
| W917 | warning | geometry | semantic | 1.0 | D8 | two grids name one `frame_uid` but differ in dimensionality or `coord_system` |
| W918 | warning | annotations | semantic | 1.0 | D9 | a voxel annotation declares `closure = "implicit"` |
| W919 | warning | annotations | semantic | 1.0 | D10 | a contour's `contour_role` disagrees with its depth among its class's polygons in its plane |

- Each needs a conformance case (a test fails until it has one) and a row in
  the §15.2 table marked (r2). The corpus grows from 163 to 172 cases (see
  [README](README.md#corpus-and-code-table-targets)); the code table from 95 to
  100 codes.
- Free ranges for any further code: E012+, E112+, E206+, E307+, E415+, E507+,
  E606+, E705+, E820+, W920+. Ask the coordinator before using one.
- Users pinning `strict` read this in the changelog, under Behaviour changes:
  "`--level strict` fails a file that draws W915–W919; pass `--codes-revision 1`
  to keep 2.0's verdicts."

### D3 An amendment carries attestation forward, never past a break

- **Where.** 1.0 §14.4 (a new bullet) and §2.1 (the `digest_algo` row of the
  root-attribute table). W21-04 PR 1.
- **Decision.** Amending refuses a source whose declared `content_id` does not
  recompute from what is stored (E702), and a source holding a digest, or
  declaring a `digest_algo`, the writer cannot read (E703), before writing
  anything. `medh5 fix --rewrite-digests --reason …` stays the one way to
  re-attest such a file, and records it.
- **Why.** At 2.0.2 a file edited outside medh5 (E702) validates clean after any
  no-op amendment (a scrub, a split claim, an added annotation), because the
  commit recomputes `content_id` from whatever is stored. An unknown
  `digest_algo` (E703) is silently replaced by `sha256`. Both turn an integrity
  failure into a valid file with no record, which is exactly what `content_id`
  exists to prevent.
- **Text to write**, a new bullet in 1.0 §14.4:

  > **An amendment carries attestation forward, never past a break.** A writer
  > **MUST NOT** amend a file whose declared `content_id` does not recompute from
  > what is stored (E702), or that holds a digest, or declares a `digest_algo`,
  > it cannot read (E703): restamping would attest bytes nobody verified.
  > Re-attesting such a file is a repair, made deliberately and recorded as an
  > activity (§11.1).

  Replacing the last sentence of the `digest_algo` row's Value cell in 1.0
  §2.1's table ("An amendment keeps the algorithm the file declares."), on the
  row's one line, since a table cell cannot break:

  > An amendment keeps the algorithm the file declares, and refuses a file that
  > declares one outside this list (E703): it could neither keep that algorithm
  > nor check the digests made with it. Only an explicit re-attestation,
  > recorded as an activity (§11.1), restamps such a file, in `sha256`.

- **Consequence.** No file changes validity. Behaviour change (a fix): commands
  that amend (`fix` without `--rewrite-digests`, `scrub --apply`, split claims,
  `clinical augment`/`strip`, `seg`, `medh5.amend`) refuse such a source and
  name the repair. Two erratum rows.

### D4 A digest's hex is canonical

- **Where.** 1.0 §13.1, after "`H` is the algorithm `<algo>` names (§2.1)."
  W21-02 PR 3. (D6, in PR 2, edits a later sentence of the same section; the
  two do not touch.)
- **Decision.** A digest is canonical when it is `<algo>:` followed by lowercase
  hex of the algorithm's length; anything else is malformed (E703), and `fix
  --rewrite-digests` restamps it. `parse_digest` stays lenient (it is public and
  its test pins the leniency); a new `is_canonical_digest` decides.
- **Why.** `parse_digest` imitates Python's `int(hex, 16)`, so an uppercase
  digest of unchanged bytes is reported as E701 "does not match its content"
  plus E702: a third-party writer's spelling is reported as tampering, and the
  diagnosis sends people looking for corruption that does not exist.
- **Text to write**, in 1.0 §13.1, as the sentence after "`H` is the
  algorithm `<algo>` names (§2.1).":

  > `<hex>` is the digest in lowercase hexadecimal, two characters per byte (64
  > for `sha256`, 128 for `sha512` and `blake2b`); any other spelling is
  > malformed (E703).

- **Consequence.** No corpus case changes (no corpus digest is non-canonical);
  one new invalid case. A file with uppercase digests moves from E701/E702 to
  E703/E702.

### D5 The algorithm of an index's `source_digest`

- **Where.** 1.0 §13.3. W21-02 PR 3.
- **Decision.** The prefix is self-describing, as for a dataset digest: a
  reader recomputes under the algorithm the entry names. The reference writer
  keeps writing `sha256`.
- **Why.** §13.3 says "prefixed `"<algo>:"`" and never says what the hash is; the
  engine always uses sha256. A third party that reads `<algo>` as the file's
  algorithm writes a sha512 index into a sha512 file, and the engine declares it
  stale (W905): its files fail `strict`. Reading the prefix keeps every current
  index current, and 2.0.x readers see what they see now.
- **Text to write**, replacing in 1.0 §13.3 the two sentences from "prefixed
  `"<algo>:"`, where `child_name` is…" through "…and **MUST NOT** treat that as
  a file error." (the next sentence, "Derived caches therefore cannot go
  silently stale…", stays):

  > prefixed `"<algo>:"`, where `<algo>` is an algorithm of §2.1 and `H` the
  > algorithm it names: as for a dataset's digest (§13.1), the prefix says
  > which, and it need not be the root's `digest_algo`. `child_name` is the
  > dataset's name within the annotation group (`data`, `layer_class_ids`, …)
  > and `child_digest` its `digest` as stored, or, for a dataset that carries
  > none, its §13.1 digest under `<algo>`; a dataset with no canonical byte
  > stream (§13.1) has no line. A reader computes the current digest under the
  > algorithm the entry names. Readers **MUST** ignore an index entry whose
  > `source_digest` is malformed, names an algorithm outside §2.1, or does not
  > match the current digest of its source, and **MUST NOT** treat that as a
  > file error (W905).

### D6 An undigested dataset under a `content_id`: W915

- **Where.** 1.0 §13.1, §15.2. W21-02 PR 2.
- **Decision.** The validator reports W915, at the integrity level, for each
  dataset under `grids/`, `images/`, `annotations/` or `transforms/` that
  carries no `digest` in a file declaring a `content_id`. Not under `clinical/`
  (E818 covers it), and not when W901 is reported (no digests at all). `verify`
  is unchanged.
- **Why.** `medh5 validate --level strict` passes a file that `medh5 verify`
  fails (`unattested`), so a CI gating on the validator misses an edit nobody
  can detect. A SHOULD gets a warning; making it an error would invalidate
  content the text allows. Changing `verify` instead would bring back the L-26
  bug (1.4.2).
- **Text to write**, in 1.0 §13.1, after "The digest covers
  **decompressed** content, so recompression does not invalidate it." (the
  sentence that states the SHOULD ends in a colon and a formula, so the new
  sentence cannot follow it directly):

  > A dataset without a `digest`, inside an object a declared `content_id`
  > covers, is reported W915: the root does not attest its bytes, so an edit to
  > them changes no line of §13.2.

- **Invariant the tests hold.** `verify` fails exactly when the validator, at
  the integrity level, reports E701, E702, E703, E704, E818 or W915, or W913
  for a projection's root.

### D7 A contradiction under implicit closure: W916

- **Where.** 1.0 §5.4, §9, §15.2. W21-02 PR 4.
- **Decision.** W916 when, in one scope unit of a classification annotation
  with `closure = "implicit"`, a class is asserted with a value ≤ 0 and an
  `is_a` descendant is asserted above 0. Not for a class examined but not
  asserted (§9's recommended pattern), not for soft values that are not
  monotone, not under `explicit`, and not when E304 or E306 is reported.
- **Why.** Since 2.0.2 `state(kidney)` answers positive under implicit closure
  when `left_kidney` is asserted 0.9, while `value(kidney)` is still the stored
  0.0. A model trained on one gets the opposite of a reader that uses the
  other, and nothing says the data disagrees with itself.
- **Text to write**, at the end of 1.0 §5.4's paragraph on `implicit`:

  > An assertion that a class is absent (a value of 0 or less) in a scope unit
  > where an `is_a` descendant is asserted present contradicts the inference
  > readers make, and is reported W916.

### D8 A plane and a volume naming one frame: W917

- **Where.** 1.0 §3.3, §15.2. W21-02 PR 4.
- **Decision.** W917 once per pair of grids, in any timepoints, that share a
  `frame_uid` but differ in S or `coord_system`. Converters keep the UID they
  were given; the warning advises.
- **Why.** A converter that tags a 2-D key image with the CT's
  FrameOfReferenceUID produces exactly this; a reader trusting `frame_uid`
  alone overlays the plane with no slice position, and since 2.0.2 the
  comparison tools refuse it (E414) with no earlier warning. An error would
  reject realistic converter output.
- **Text to write**, after 1.0 §3.3 rule 4:

  > Two grids that name one `frame_uid` but that this rule still keeps apart (a
  > different S or `coord_system`) are reported W917; a converter that was given
  > the identifier keeps it.

### D9 A voxel annotation declaring implicit closure: W918

- **Where.** 1.0 §5.4, §15.2. W21-02 PR 4.
- **Decision.** W918 for each `labelmap`, `layers`, `bitmask`, `instances` or
  `probmap` annotation declaring `closure = "implicit"`.
- **Why.** §5.4 says writers SHOULD use `explicit` for voxel annotations. Since
  2.0.2 voxel reads under `implicit` include every descendant's voxels in each
  ancestor's, so a file declaring it reads differently from 2.0.1, silently.
- **Text to write**, in 1.0 §5.4 after "Writers SHOULD use … `explicit` for
  voxel annotations":

  > (a voxel annotation declaring `implicit` is reported W918).

### D10 Combining a class's polygons; W919; an unknown role is E404

- **Where.** 1.0 §8.6, §15.2. W21-13 PR 2 (it owns the shared depth function
  and the RTSTRUCT importer that motivated it).
- **Decision.** The region of a class in a plane is defined by its polygons'
  nesting depth, as the reference rasteriser already computes it. A writer
  SHOULD write `contour_role` as the depth's parity; a reader computes the
  region from depth and never fails on a role. A role that disagrees with depth
  is W919 (warning); a role other than 0 or 1 is E404 (the existing "encoding
  invariant violated").
- **Why.** §8.6 gives `contour_role` "0 = outer boundary, 1 = hole" and no rule
  for combining polygons, so two RTSTRUCT ROIs with one name, one inside the
  other, became a hole at 2.0.2 (105 voxels where 121 were drawn): roles came
  from nesting per ROI, the rasteriser nested per class. The text was
  ambiguous, so a disagreeing file stays valid with a warning; only a role
  value the text never defined is an error. The engine reader raises
  `Error::Index` on a role of 2 today, against §16's rule that readers ignore
  values they do not know.
- **Text to write**, a new paragraph at the end of 1.0 §8.6:

  > **Combining polygons.** A class's region in a plane is fixed by its
  > polygons in that plane (`contour_plane`) alone. Polygon *p* encloses *q*
  > when every vertex of *q* lies inside *p*; a polygon's *depth* is the number
  > of the class's other polygons in that plane that enclose it. Polygons of
  > one depth are united, and the region is the points inside an odd number of
  > those layers: a hole subtracts from the region around it, an island inside
  > a hole adds back, and outlines that overlap without nesting are their
  > union. `contour_role` **SHOULD** be `1` exactly for a polygon of odd depth;
  > readers compute the region from depth, not from the role, and a role that
  > disagrees is reported W919. A `contour_role` other than `0` or `1` is
  > E404. A polygon with `axis = −1` takes part in no combination. A writer
  > **SHOULD NOT** store crossing polygons of one class in one plane, since
  > readers that combine by exclusive-or cancel their overlap.

- **Consequence.** Files that 2.0.x wrote from same-named nested ROIs draw W919
  (the import that made them is refused from 2.1.0). Cases:
  `W919-contour-role-depth` (valid) and `E404-contour-role-value` (invalid).

### D11 Annotation pyramids are reserved, not defined

- **Where.** 1.0 §4.3, §16 (the reserved-names table). W21-02 PR 4.
- **Decision.** Withdraw §4.3's "Voxel annotations MAY be pyramided
  identically…" and reserve the names for a later minor.
- **Why.** The sentence defines nothing: which voxel `nearest` takes under an
  even factor, what `max` means over class ids, how coverage and the ignore
  region downsample, whether levels are digested or indexed. Nothing writes,
  reads or validates one. The sampling index's occupancy planes already give a
  coarse per-class map.
- **Text to write**, replacing the §4.3 sentence:

  > Voxel annotations are not pyramided in 1.x: an annotation's datasets hold
  > one level, on the annotation's grid.

  A row in the §16 reserved-names table: reserved `levels`,
  `downsample_factors`, `downsample_method`, `grid_levels` on an annotation;
  attributes; deferred because the rule for `nearest` under even factors, the
  meaning of `max` over class ids, and how coverage and the ignore region
  downsample were never defined, and nothing implemented them; intended
  meaning: a voxel annotation's pyramid, laid out as §4.3 lays out an image's.
  A writer **MUST NOT** emit those attributes on an annotation; the reference
  writer refuses them.

### D12 When an absence is a finding

- **Where.** 1.0 §7.4. W21-09 PR 1.
- **Decision.** Tracking keeps its three states (`present`, `resolved`,
  `unexamined`) and gives every unseen (object, visit) a reason: `located` or
  `class_examined` (state `resolved`); `outside_fov`, `ignored`, `unrelated`,
  `no_identity` or `not_examined` (state `unexamined`). An object is located
  only through the shared frame or a path of affine transforms; a nonlinear
  path does not locate it.
- **Why.** At 2.0.2 a lesion is `resolved` at a follow-up whose grid does not
  contain it (verified with the follow-up grid 100 mm away): tracking never
  located the object. A fourth state would break every three-way match and
  duplicate `unexamined`; making everything unlocated `unexamined` would remove
  every negative from unregistered longitudinal samples, which are common.
- **Text to write**, replacing in 1.0 §7.4 the paragraph that begins "An
  object present at one timepoint and absent at another…" (it ends "…which is
  why coverage is required."):

  > An object present at one timepoint and absent at another is represented by
  > its absence from that timepoint's annotation; a *resolved* lesion is
  > distinguishable from an *unexamined* one only through
  > `annotated_class_ids`, the annotation's ignore region (§7.7, §11.3) and
  > where the object lies — which is why coverage is required. An absence is a
  > finding only where the object was looked for. An object's *located extent*
  > in an annotation is the axis-aligned box, rounded out to whole voxels of
  > the annotation's grid, that encloses the corners of every box observed for
  > it at other timepoints, each mapped into that grid through the shared frame
  > or a path of affine transforms (§10). An absence is *resolved* when an
  > annotation at that timepoint that carries `instance_ids` lists the class in
  > `annotated_class_ids` and either the located extent lies within the grid
  > and within the acquired voxels of its images (§4.4) and holds no ignored
  > voxel, or no extent can be located and the annotation ignores no voxel.
  > Otherwise it is *unexamined*: an absence inside, or possibly inside, an
  > ignore region, or where the file places the object outside what was
  > imaged, was not looked for.

  The sentence's old clause "an absence inside, or possibly inside, an ignore
  region is unexamined" survives as the last sentence's first case.

### D13 An agreement record names an annotation of its own sample

- **Where.** 1.0 §11.2; the `against` description in
  `crates/medh5/data/medh5-sample-1.0.schema.json`. W21-09 PR 2.
- **Decision.** A comparison across samples has `against = None`. It is
  recorded only when the caller names an annotation of the recording sample
  that holds what was compared against (the other rater's annotation copied in,
  say); the writer checks that the path names an annotation of that sample and
  refuses otherwise.
- **Why.** At 2.0.2 a cross-sample result records `against =
  "annotations/<B's id>"`, which resolves inside A: a self-reference, verified.
  §11.2 shows `against` only in its JSON example and never defines it; the
  schema's description ("HDF5 path of the annotation compared against") does
  not say whose.
- **Text to write**, in 1.0 §11.2, a sentence after the JSON example (before
  "`status` ∈ …"):

  > An `agreement` entry's `against` is the path, from the sample root, of an
  > annotation of the same sample; an agreement with another sample's
  > annotation **MUST NOT** be recorded under a path of this one.

  The schema's description of `against` becomes "Path, from the sample root,
  of an annotation of the same sample, compared against." (a description only:
  what validates is unchanged).

### D14 A date shift may be withheld

- **Where.** 1.0 §11.4 (its third rule); 1.1 §3. W21-10 PR 2.
- **Decision.** A scrub can withhold the amount of its shift
  (`--withhold-shift`), and take a per-subject shift from a range and the salt.
  2.1 warns when a shift is recorded without an explicit choice; 3.0 withholds
  by default.
- **Why.** The scrub writes the exact amount into every file, and the §11.4
  example records −117: one known date restores all of them. DICOM PS3.15
  records only *that* dates were modified. The validator already accepts a
  `shifted_utc` clock whose amount is not recorded, so no format change is
  needed.
- **Text to write**, replacing 1.0 §11.4's rule on `date_shift_days`:

  > * `date_shift_days` records that dates were consistently shifted, preserving
  >   intervals, and by how much; a recorded `0` means no shift. Because the
  >   amount restores every date, a file shared beyond whoever holds the key
  >   **SHOULD NOT** record it: it omits `date_shift_days`, and the
  >   de-identification's provenance activity states that dates were shifted. A
  >   shift **MAY** differ between subjects and **MUST NOT** differ between the
  >   files of one subject.

  In 1.1 §3, "after the de-identification shift `/meta` declares" becomes
  "after the de-identification shift, whose amount `/meta` may withhold". New
  valid case `clinical-shifted-clock-withheld-shift`.

### D15 De-identifying a clinical sample is a derivation

- **Where.** 1.1 §10 (a new bullet), 1.1 Appendix A. W21-10 PR 3.
- **Decision.** `scrub --apply` on a clinical sample writes new versions of a new
  sample: the clock becomes `shifted_utc` under a new `clock.id` (re-minted from
  the salt, so a subject's fragments still share a clock), every time moves by
  the one shift, ids are replaced consistently, and text it does not rewrite is
  attested by whoever de-identified it, with the digests of the bytes attested.
- **Why.** A clinical scrub cannot amend the same versions: `set_clock` refuses
  another clock (E802) and 1.1 §7.3 holds a re-added version to what it was.
  Shifting in place would violate §7.3; refusing (2.0.2) leaves users only
  `clinical strip`, which loses the history.
- **Text to write**, a bullet in 1.1 §10:

  > - **De-identification** is a separately requested derivation, as downgrade
  >   is: the result is a different sample with its own `content_id`. Every
  >   event time moves by the one shift applied to `/meta`'s dates, effective
  >   and available bounds alike, so a selection at a cutoff moved by that shift
  >   admits the same versions; the clock becomes `shifted_utc` under a new
  >   `clock.id`, and event, record and document ids are replaced consistently.
  >   The versions it writes are new versions of a new sample, not amendments of
  >   the source's (§7.3). Text it does not rewrite is attested as
  >   de-identified by whoever did, with the digests of the bytes attested.

### D16 An attested payload's attributes are held too

- **Where.** 1.1 §7.3, 1.1 Appendix A (a clarification). W21-04 PR 1.
- **Decision.** An amendment's hold on attested payloads compares their
  datasets' digests and the attribute lines §13.2 gives them, so an edit to an
  attested image's `rescale_slope` is E809 like an edit to its pixels.
- **Why.** `check_held` compares dataset digests only, which is what §7.3 says;
  an attribute edit to attested evidence passes silently.
- **Text to write**, in 1.1 §7.3, where attested payloads are "compared by its
  datasets' digests": "compared by its datasets' digests and the attribute
  lines 1.0 §13.2 gives it". It binds writers only and changes nothing a reader
  does.

### D17 An asserter that is its own endpoint

- **Where.** 1.1 §7.3, §9.1 step 5, Appendix A; TC §3.5. W21-11 PR 1 (backport).
- **Decision.** An event endpoint that *is* the link's asserting version needs
  only the eligibility its assertion does; the link's other event endpoints,
  and its document endpoints, must still be admitted.
- **Why.** Both shipped examples phrase the link `seg0 describes lesions_tp0`,
  asserted by `seg0` itself. Selection still requires the asserter to be
  *admitted* when it is also the endpoint, so under `order_by = effective` a
  procedure of unknown time never attests its mask, and the crop falls back to
  `center_fallback`; the 2.0.2 changelog calls fixed what the worked example
  still does.
- **Text to write**, replacing in 1.1 §7.3's first paragraph the sentence "The
  link's event and document endpoints are still admitted ones (§9.1 step 5)."
  (the paragraph continues "Adding or replacing an owned payload…"):

  > The link's other event endpoints, and its document endpoints, are still
  > admitted ones (§9.1 step 5); an endpoint that is the asserting version
  > itself needs only the eligibility its assertion does. A procedure of unknown
  > time that asserts it `describes` the mask it drew thereby attests the mask,
  > though no selection ordered by effective time admits the procedure; a
  > document is a payload, and a row reads one only through its admitted event.

  In §9.1 step 5: "…of links asserted by an eligible version (steps 1–2) whose
  event and document endpoints are themselves admitted, an event endpoint that
  is the asserting version needing only its eligibility."

  1.1 Appendix A row: "§7.3, §9.1 | An event endpoint that is the link's own
  asserting version need only be eligible; the link's other event endpoints
  and its document endpoints must still be admitted. | Found after 2.0.2 made
  an asserter need only be eligible: a version asserting a link from itself,
  the worked example's `seg0 describes lesions_tp0`, still had to be admitted
  for its assertion to count, so under `order_by = effective` a procedure of
  unknown time never attested its masks."

  TC §3.5: "the **eligible** (1.1 §7.3, §9.1 step 5) `instances` and `boxes`
  annotations, a mask asserted by its own eligible procedure included, on the
  image's grid…"
- **Consequence.** No fingerprint changes, so a patient-level cache built on
  such a crop still validates; the changelog says "rebuild caches whose slots
  read self-asserted masks" (the structural gap is D19's known limit).

### D18 MEDS column names

- **Where.** 1.1 §10, Appendix A. W21-12 PR 1.
- **Decision.** 1.1 §10 names MEDS's own columns, `time`, `numeric_value` and
  `text_value` (MEDS 0.3), and says onto which of this profile's columns each
  maps.
- **Why.** §10 lists "`subject_id`, the clock time, `code`, `value_num` and
  `value_text`": this profile's column names, not MEDS's. An importer written
  from the text looks for columns no MEDS dataset has.
- **Text to write**, replacing in 1.1 §10 "**MEDS** rows map from
  `subject_id`, the clock time, `code`, `value_num` and `value_text`;":

  > **MEDS** rows map from `subject_id`, `time`, `code`, `numeric_value` and
  > `text_value` (MEDS 0.3) onto the event's subject, effective time, `code`,
  > `value_num` and `value_text`;

  1.1 Appendix A row: "§10 | MEDS's columns are named as MEDS names them
  (`time`, `numeric_value`, `text_value`) and mapped onto this profile's. |
  Found planning the MEDS importer: the text gave this profile's column names
  for MEDS's, so an importer written from it looked for columns no MEDS dataset
  has."

### D19 task-cache-1 errata

- **Where.** TC, a new "Appendix A. Errata" (created by W21-11 PR 1, which
  carries D17's TC row), and the clauses below.
- **§2, opening is the runner's choice** (W21-11 PR 1 and PR 3). Text to write,
  after §2's table:

  > **Opening is the runner's choice.** A manifest's author chooses its URIs;
  > whoever runs it chooses what may be opened. An implementation **SHOULD** let
  > the runner confine sources to one directory, by default the base relative
  > URIs resolve against, and, confined, **MUST** refuse a URI naming a path
  > outside it once its `.` and `..` segments are resolved lexically (an
  > absolute path, a `file:` URI, a relative path that climbs out) without
  > opening or examining that path (**T301**, naming the URI and nothing about
  > the path), and open the path so resolved. It **MUST NOT** open a source that
  > is not a regular file: a pipe or a device is no sample, and opening one can
  > block. A source that fails any of steps 1–4 is read no further: it adds no
  > identity, clock, event, document or link to its subject (§3.3), and no
  > finding but its own T301 or T302 speaks of its content.

  Why: at 2.0.2, after a T302 preflight keeps reading the file (T303 prints its
  subject id, T305 its event ids, `--json` its whole event table); `reconcile`
  checks no pin; and a named pipe given as a source blocks `task preflight`,
  `task reconcile` and `cache validate` inside `H5Fopen`. Confinement by
  default would break every manifest with `../data/…`, so 2.1 makes it opt-in
  with a warning, and 3.0 the default.
- **§5, `min_follow_up_us`** (W21-11 PR 5). The default cannot change under
  `/1`: it is in every normal form, so in every task fingerprint and every
  patient-cache pin. Text to write, appended to the row:

  > A target with `negative` values **SHOULD** state it. The default is the one
  > `medh5.task/1` has always filled in: it is part of every normal form
  > (§3.1), so of every task fingerprint (§3.2), and is kept for that reason; a
  > later version of this contract may require the member whenever `negative`
  > is non-empty.

- **§7.1, entries through the task's pins** (W21-11 PR 4). Text to write,
  replacing the sentences on how an entry's relative `uri` resolves:

  > An entry copies its task's source reference, so a relative `uri` is
  > relative to the **task manifest's directory**. A validator given the task
  > finds each entry source through the task's own references: the reference
  > pinning the same `content_id` and `sample_key` is opened where it says, so a
  > cache whose sources moved (its task's URIs updated, the pins unchanged)
  > still validates, as §8 compares sources by `content_id`. An entry source the
  > task does not pin is resolved by its own `uri`, against the base the
  > validator is given: by default the task manifest's directory with a task,
  > the cache's own directory without.

- **Known limit**, an Appendix A paragraph, not a change (W21-11 PR 5): a
  patient-level entry pins the task, the row fingerprint and the sources, not
  what the row read (the slot image, the crop centre, the admitted payloads),
  so a selection fix such as D17 changes inputs while every pin holds. A
  `medh5.cache/2` could add a digest over the row's admitted payloads and slot
  fills; until then the changelog carries rebuild notes.
- **§10** lists the commands and fixtures that exist after 2.1.0 (W21-03 and
  W21-11 update it).

### D20 No new format version in 2.1.0

- **Decision.** 2.1.0 writes 1.0 and 1.1 only. The "1.2" literals that stand for
  "a later minor" (the `W913-higher-minor-projection` case, companion fixture
  T306, `version.rs`, tests) are derived from the newest supported version
  (`version::next_minor`), so 2.2.0's real 1.2 moves them to 1.3 without edits
  (W21-02 PR 1).
- **Why.** A 1.2 can only add what a 1.1 reader already skips: a new kind,
  profile or object, which the projection drops as W913. 1.0 §16 has readers
  ignore unknown attributes, so a 1.2 *attribute* that changed how an existing
  object reads would be silently misread by 2.0 and 2.1. Nothing in 2.1.0
  needs new content; 2.2.0 brings 1.2 for one new kind.

## Companion and reference contracts

### D21 The cohort manifest: `medh5.dataset/1`

- **Where.** A reference section (`docs/reference/` cohort manifest), not the
  sample specification: 1.0 §12.3 calls the dataset manifest authoritative and
  gives no format. W21-08 PR 1.
- **Decision.** Every save writes `"schema": "medh5.dataset/1"`; paths are
  relative to the manifest's directory when they lie beneath it, absolute
  otherwise; a manifest without `schema` is legacy and read against the
  working directory, with a `FutureWarning` only when the two readings differ.
  An unknown schema is refused. `sha256` is unchanged.
- **Why.** At 2.0.2 a manifest written from one directory and loaded from
  another lists every file as stale, `check` reports C402 falsely, and the
  statistics open nothing: a cohort copied to a cluster mount points at
  nothing. task-cache-1 already resolves a source `uri` against its manifest's
  directory.
- **Text to write** (reference page):

  > In a `medh5.dataset/1` manifest, `root` and each entry's `path` are
  > absolute or relative to the directory holding the manifest; a reader
  > resolves a relative path against that directory, never against its working
  > directory. A writer writes a path relative only when it lies beneath that
  > directory, absolute otherwise.

  (No `..`: a lexical `..` and the system's disagree across a symbolic link.)

### D22 The split algorithm: `medh5.split/2`

- **Where.** `docs/guides/cohorts.md` and the split JSON (a reference contract,
  not the sample spec; §12.3 claims are unchanged). W21-08 PR 3.
- **Decision.** A second, versioned algorithm, opt-in in 2.1 and the default in
  3.0, plus `previous=` to continue a split without moving any group. A split
  JSON without `algorithm` reads as `medh5.split/1`.
  - **Totals per partition**: largest remainder of N·r.
  - **Counts per stratum**: controlled rounding of n_s·r_p: each cell is the
    floor or the ceiling, rows sum to n_s and columns to the totals (always
    solvable as an integer flow); strata in key order, partitions in
    `PARTITIONS` order.
  - **Within a stratum**: groups fill partitions in hash-rank order. k-fold uses
    equal shares.
  - **`previous=`**: a group already placed keeps its partition or fold; new
    groups fill each stratum's deficit against /2's targets for the whole
    cohort, ties broken by rank; dropped groups are listed. `seed`, `ratios`,
    `group_by`, `stratify_by` and `k_folds` must equal the previous split's
    ("another split; use a new set_id"). A previous /1 split is accepted; its
    new groups are dealt by /2.
- **Why.** /1 interleaves strata and deals by global deficit, so a group's
  partition depends on every other group: adding one group to 100 moves a
  median of 34 (9 test→train), to 1000 a median of 300; rare strata of 3, 5 or
  7 at 70/15/15 get no test group. A model retrained on a re-split has trained
  on old test subjects. A prototype of /2 moved a median of 0–2 groups, and
  rare strata of 3, 5, 7 came out 2/0/1, 3/1/1, 5/1/1.
- **Deprecation.** Only where the answer will differ. When `algorithm` is
  omitted, `make_splits` deals the cohort both ways (no file is read twice) and
  raises `FutureWarning` only when /2 would place some group in another
  partition or fold: "medh5 3.0 splits with medh5.split/2 by default, which
  places N of these groups differently: pass algorithm="medh5.split/1" to keep
  this split". A split both algorithms agree on never warns. The native
  `dataset split` prints the same sentence to standard error, as `warning:
  …`. `previous=` deals new groups by /2 whatever was omitted, and refuses an
  explicit `algorithm="medh5.split/1"` (/1 has no continuation).

## API and behaviour

### D23 One replace routine; links are followed and kept

- **Decision.** Every file medh5 writes goes through the engine's replace
  routine (W21-05): Python's `_files.write_text`, `TaskManifest.save` and the
  converters' outputs follow a symbolic link and replace its target, keeping
  the link, as `AtomicFile` and `write_atomic` (1.0 §14.4) do; they keep the
  target's mode, fsync, and write `\n` on every platform.
- **Why.** At 2.0.2 one kind of write follows three link rules: the engine
  follows and keeps the link, Python's `write_text` replaces the link (pinned by
  `test_a_report_replaces_a_link_rather_than_writing_through_it` in
  `tests/tools/test_cli.py`), and the native CLI's `std::fs::write` writes
  through it in place. `--report` alone has two rules. Python also creates its
  temporaries 0o666 under the umask, drops a 0o600 mode, and writes CRLF on
  Windows. §14.4 and the 2.0.2 reference already describe the engine's rule.
- **Consequence.** Behaviour change: the 2.0.2 test pinning the replaced link is
  rewritten to pin the kept link.

### D24 Task manifests are written and read by the engine

- **Decision.** One writer and one reader: Python's `TaskManifest.save` and
  `load` go through the engine, so every frontend writes the same bytes
  (ASCII-escaped JSON, as the CLI, the cohort manifest and Python's own
  `json.dumps` default already do) and refuses `NaN` (T101). W21-11 PR 3.
- **Why.** Python wrote raw UTF-8 and the engine `\u` escapes (621 against 629
  bytes for one manifest), so a checked-in manifest changes bytes whenever the
  other frontend saves it; Python accepted `NaN`, the engine refuses it.

### D25 `labels check` groups as C101 does

- **Decision.** `labels check` groups files by the label set's digest, the rule
  `dataset check`'s C101/C103 already use: an empty label set is a declared
  vocabulary. The engine gains the rule; the CLI and a new
  `medh5.labels.check(paths)` use it. W21-08 PR 4.
- **Why.** Two empty label sets with different ids are one vocabulary to
  `labels check` (both `<none>`) and two to C101, and the 16-hex key groups by a
  prefix of the digest.
- **Consequence.** Behaviour change: `labels check` can change its verdict for
  files with empty label sets.

### D26 Unsalted name pseudonyms are refused

- **Decision.** Reverse 2.0.2's "strict without a salt allowed (unsalted,
  recorded)" for names: `apply` refuses before writing, worded as the
  `pseudonymise_ids` refusal, and names go through the salted pseudonym so
  `uid_map` lists them. Unsalted UID pseudonyms become a review finding,
  `unsalted_pseudonym`, in 2.1, and need a salt in 3.0. W21-10 PR 1.
- **Why.** Verified: strict mode without `--salt` wrote "John Smith" as
  `pseudo:ef61a579…`, which is `sha256("John Smith")[:32]`: hashing candidate
  names reverses it. A pseudonym anyone can reverse is not one.
- **Consequence.** Security entry; a behaviour change for strict scrubs without
  a salt.

### D27 `Image.read(physical=True)` is unchanged

- **Decision.** Keep 2.0.2's contract: exact physical values in the narrowest
  type that holds them exactly (float32 only where a rescale applies); pass
  `dtype` for one type across a cohort. Documentation and a `STALE_CLAIMS`
  entry ("physical=True returns float32") only. W21-06 PR 2.
- **Why.** Both answers are exact; always returning float32 would double the
  memory of every physical read of the CT layout §4.2 recommends (int16 HU,
  slope 1), and the remedy is one keyword the docstring already names. The
  torch and MONAI paths pass `dtype` already.

### D28 `transform_between` refuses a visit of several frames

- **Decision.** For timepoint keys, `transform_between` answers only when each
  visit's framed grids share one frame; when either visit holds several, it
  raises an uncoded `MEDH5ValidationError` naming each visit's frames and grids
  and pointing to `by="grid"` or `resolve_frames`. A new keyword `by` chooses
  how keys are read (`"timepoint"`, `"grid"`, `"frame"`). With one frame a side,
  answers are unchanged. W21-07 PR 1.
- **Why.** At 2.0.2 the answer for a visit holding an MR in its own frame next
  to a CT depends on the grids' names (verified: `apet0`/`apet1` gave `None`,
  `pet0`/`pet1` gave a transform), and `None` reads as "aligned". The audit's
  rule ("None if any pair shares a frame") would declare two visits aligned on
  the strength of one shared series. No code: 1.0 §3.4 calls this "a property of
  the query, not of the file".

### D29 Lost updates are detected, not locked out

- **Decision.** An amendment and a rewrite in place snapshot the source when
  they read it (on Unix the file identity read from HDF5's own descriptor; on
  Windows size and last-write time; on both the declared `content_id`), and
  refuse to commit if it changed: "'x.medh5' changed after this amendment read
  it; nothing was written: amend it again". No opt-out. W21-04 PR 1.
- **Why.** `amend` closes its source once it has inherited the content, so two
  amendments, or an amendment and a `repack`, both rename over the file and the
  first is lost without an error. A lock held for the session is unreliable on
  NFS, Lustre and GPFS (§14.4) and Windows cannot replace an open file;
  `content_id` alone misses `index/`, uncovered attributes and extensions.
- **Text** (optional, informative, 1.0 §14.4): "a writer that cannot hold
  exclusive access SHOULD refuse to replace a source that changed after it was
  read."

### D30 MONAI metadata spells `None` as `"none"`

- **Decision.** In the MONAI adapter's metadata only, an absent `frame_uid`,
  `value_units`, `modality`, `timepoint` or `subject_id` is `"none"`
  (`TraceKeys.NONE`, MONAI's own spelling), and `from_metatensor` maps it back.
  W21-14 PR 1.
- **Why.** MONAI's `list_data_collate` raises on a `None` in metadata, which
  every sample without a `frame_uid` carries (every NIfTI and nnU-Net import):
  MONAI cannot batch medh5 items at 2.0.2. Behaviour change for code testing
  `meta["medh5"]["frame_uid"] is None`.

### D31 File errors gain `errno` and the standard subclasses

- **Decision.** Additive in 2.1, completed in 3.0. `MEDH5FileError(message,
  errno=None, filename=None)`; new `MEDH5FileNotFoundError(MEDH5FileError,
  FileNotFoundError)`, `MEDH5PermissionError(…, PermissionError)`,
  `MEDH5IsADirectoryError(…, IsADirectoryError)`; HDF5 read failures raise a
  new `MEDH5IOError(OSError)`, not yet a `MEDH5Error`, so catch order is
  unchanged. `str()` is the message, so the E001 words and the CLI do not
  change. W21-06 PR 1.
- **Why.** One missing file gives three different errors today; `except
  FileNotFoundError` misses medh5's, and `errno`/`filename` are `None`. Mapping
  everything to `MEDH5FileError` now would change which branch catches it for
  code that catches `MEDH5Error` before `OSError` (`medh5/cli.py` does).

### D32 One rounding of clock hours and days

- **Decision.** Half to even everywhere, written once in the engine
  (`clinical::{round_micros, hours, days}`); `--cutoff-hours` and augment's visit
  days use it. Python's `hours`/`days` keep their arithmetic (already half to
  even) and a test holds them equal to the engine over the ties. W21-11 PR 2.
- **Why.** The CLI and augment rounded half away from zero, so a cutoff given in
  hours differed by 1 µs between a script and the command line at exact ties,
  and admission is a closed bound. Python's rule wins because Python's cutoffs
  are stored in task manifests and fingerprinted.

### D33 Default source ids

- **Decision.** The engine derives the default (`SourceRef::default_id(uri,
  sample_key)`): a `file:` URI resolved, any other split on `/` only (a
  backslash is an ordinary character), the id characters checked against the
  schema. When a manifest is assembled, only a *derived* id that collides is
  widened, first by the directories above it, then with `@2`, `@3`. An explicit
  id that collides is still T204. W21-11 PR 2.
- **Why.** Python's `PurePath` split gives different ids (so different
  fingerprints) on POSIX and Windows for one script; `a/site-a/P-01.medh5` and
  `b/site-a/P-01.medh5` collide. Only manifests that fail T204 today change.

### D34 `ConceptVocabulary.digest` stays as it is

- **Decision.** Keep the digest (it is not §5.1's canonical JSON: it is
  ASCII-escaped) and say so in its docstring. W21-14.
- **Why.** It is part of what a feature cache records about its encoder;
  changing it would make every cache built with a non-ASCII concept stale for no
  gain in correctness.

## Process and platform

### D35 API gates without waivers

- **Decision.** Both API gates (Rust `cargo semver-checks`, Python snapshot
  comparison) gate, with no waiver list: a 2.1 change that a gate calls a break
  is redesigned as an addition. A repository variable `API_RELEASE_TYPE`
  raises the release type for 3.0 work only. W21-00 PR 8.
- **Hiding modules.** W21-06 PR 6 puts `#[doc(hidden)]` on `pub mod h5`,
  `json`, `pyval` and `numeric` only if the semver gate accepts it; if
  semver-checks reports it as a break, hiding moves to 3.0 and 2.1.0 only
  documents those modules as plumbing outside semver. Hiding does not make an
  hdf5-metno bump safe either way: about 119 public items outside `h5` take or
  return `hdf5::` types, so hdf5-metno 0.15, ndarray 0.17 and IndexMap 2 are
  declared public dependencies (a bump of any is a major medh5 release).

### D36 Python 3.10

- **Decision.** 2.1.0 announces that 2.1.x is the last line supporting Python
  3.10 (CHANGELOG under Deprecated, `docs/install.md`, `README.md`), with no
  runtime warning; 2.2.0 drops it (`abi3-py311`). W21-01.
- **Why.** 3.10 reaches end of life in October 2026; dropping an interpreter in
  a minor without notice breaks installs, and an import-time warning breaks
  downstream suites run with `-W error`.
- **Revised (2026-10-15, before the tag).** The maintainer kept Python 3.10 for
  2.2.0 (2.2.0 E5). The announcement was withdrawn from the release pull
  request's changelog, `README.md` and `docs/install.md`; the mechanism stays,
  for when a dependency forces the floor up.

### D37 Wheels in 2.1.0

- **Decision.** Add cp314t wheels (PyO3 builds a version-specific extension for
  a free-threaded interpreter; `gil_used = true` stays, so CPython re-enables
  the GIL at import with a warning: safe, and medh5 installs without a
  compiler). Windows arm64 is not planned: the release stays on the five
  targets it ships. musllinux, `abi3t`, the free-threading audit and the x86_64 macOS
  cross-build are 2.2.0. W21-01.

### D38 Deprecations

The mechanism, the message prefix (`medh5 3.0 `), the two categories
(`DeprecationWarning` for a removal or rename, `FutureWarning` for a result
that changes) and the Rust attribute rules are in
[plans/README.md](../README.md#deprecations). W21-00 PR 5 builds it; every
deprecation in 2.1.0 uses it and adds a line to the 3.0 ledger.
