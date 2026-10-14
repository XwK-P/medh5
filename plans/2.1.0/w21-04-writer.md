# W21-04 The writer: amendment safety and completeness

| | |
|---|---|
| Branch | `w21/writer` |
| Size | L overall: PR 1 M–L, PR 2 M, PR 3 S, PR 4 M |
| Lands after | W21-00 PR 4, PR 5 |
| Blocks | W21-05 PR 1 (after PR 1 here), W21-10 PR 3 (after PR 1), W21-13 PR 8 (after PR 2) |
| Owns | `crates/medh5/src/sample/{writer.rs,writer_*.rs}`, `crates/medh5/src/integrity/repair.rs` (the re-attestation path), `crates/medh5-python/src/writer.rs`, `medh5/sample.py`, `medh5/collection.py` (`pack`'s keys); touches `h5/file.rs` (`AtomicFile`) for the snapshot |
| Decisions | D3, D16, D29; the deprecation rules |
| Backport | none |

## Session prompt

> Implement workstream W21-04 (the writer: amendment safety and completeness)
> of medh5 2.1.0 in this repository. First read `plans/README.md`,
> `plans/2.1.0/README.md` and `plans/2.1.0/decisions.md` (D3, D16, D29), then
> `plans/2.1.0/w21-04-writer.md` in full. Work from the latest `main` on
> `w21/writer`; land the four pull requests in order. PR 1 changes what every
> amending command does (fix, scrub, split claims, clinical augment, seg,
> `medh5.amend`): expect internal tests that amend a damaged file on purpose, and
> move them to the repair path rather than weakening the refusal. Re-check each
> "State at 2.0.2", write tests first and see them fail, write the spec text the
> decisions give with their erratum rows, run the local gate from `CLAUDE.md`,
> add fragments, open `W21-04: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 04.1 A stale root is not re-attested; 04.2 an unknown algorithm or unreadable digest is not amended; 04.3 the clinical hold holds; 04.4 lost updates are refused | M–L |
| 2 | 04.5 Remove an image or a grid, edit an image's attributes; 04.6 `add_mask` returns its group (Python) | M |
| 3 | 04.7 The default `sample_id` and pack key: deprecate the first-dot rule | S |
| 4 | 04.8 float16 displacement fields | M |

## Items

### 04.1 An amendment does not re-attest a stale root (PR 1, D3)

- **State at 2.0.2.** Reproduced: an h5py edit of `grids/ct_tp0`'s `spacing`
  gives E702 at `/`; a no-op `medh5.amend` then validates clean. With a
  malformed dataset digest, `digests=False` likewise drops the E702.
  `commit_inner` (`sample/writer.rs`) recomputes `content_id` from whatever is
  stored; only restamped dataset digests are checked against the source
  (`refuse_restamping`), never the root. Every amending command inherits this:
  `fix` (`integrity/repair.rs`), clinical augment and strip
  (`clinical/augment.rs`), scrub (`curation/scrub.rs`), split claims
  (`dataset/split.rs`, whose pre-check checks only dataset digests), the CLI's
  `seg` (`medh5-cli/src/seg.rs`), Python `amend` (`crates/medh5-python/src/writer.rs`).
- **Change.** `amend_into`, after `require_amendable` and before writing
  anything, recomputes the source's declared `content_id` from its stored
  digests and covered attributes (attributes and `/meta` only, never bulk data)
  and refuses E702 when it differs, naming `medh5 fix --rewrite-digests --reason`.
  Order: 04.2's E703 checks first, then E702. A file that declares no
  `content_id` is not refused. `refuse_failed_digests` also checks the root. A
  crate-private `amend_reattesting` entry is the only bypass, used by
  `repair::fix` with `--rewrite-digests`. `dataset/split.rs`'s pre-check calls
  the same check. D3's §14.4 bullet and a C.1 row.
- **Tests.** `test_S14_4_an_amendment_does_not_reattest_a_stale_root`: refused
  E702, and the source byte-identical, for `medh5.amend`, `scrub --apply` and
  split claims; `test_S14_4_fix_reattests_a_stale_root_deliberately` (the file
  validates clean and the activity records the repair); Rust
  `s14_4_a_stale_root_is_refused_before_anything_is_written`.
- **Docs.** `docs/guides/validate.md` (the repair), `docs/explanation/refusals.md`.
- **Changelog.** Behaviour changes, area `writer`: "**An amendment refuses a file
  whose `content_id` no longer matches it.** … previously the next scrub, split
  claim or annotation re-attested the edit; repair it with `medh5 fix
  --rewrite-digests --reason`."

### 04.2 An unknown algorithm, or a digest it cannot read, is not amended (PR 1, D3)

- **State at 2.0.2.** 1.0 §2.1 allows `sha256` (default), `sha512`, `blake2b`
  and makes anything else E703. With a root declaring `blake3`:
  - validate: E703 at `/`, root unchecked (correct);
  - `verify`, `diagnose`, `fix`, `medh5 fix --rewrite-digests`: raise `[E703]
    unsupported digest algorithm 'blake3'` (a 1.x choice);
  - a no-op amend keeps `digest_algo` only if it is known (`sample/writer.rs`),
    otherwise writes `sha256`: a blake3 root over sha256 digests comes out
    declaring sha256 and validates clean, with no record;
  - amend with every digest `blake3:…` and `digests=True` is refused with a
    misleading E701 ("use `medh5 fix --rewrite-digests`", which then raises);
    with `digests=False` it writes a file declaring sha256 that still holds
    blake3 digests.
- **Change.** `amend_into` refuses before writing: a `digest_algo` outside §2.1
  as E703 at `/`; any stored digest that does not parse as E703 at that dataset
  (and, once W21-02 PR 3 has landed, any that is not canonical: whichever of the
  two lands second wires `is_canonical_digest` in here). `stamp_in` and
  `refuse_restamping` raise E703, not E701, for a digest they cannot read. `fix
  --rewrite-digests --reason` uses the bypass, diagnoses without verifying the
  root, restamps in `sha256` (`DEFAULT_ALGO`), and records `"digest_algo":
  {"from": "blake3", "to": "sha256"}` in the activity's params beside `reason`,
  `restamped` and `verified_content: false`; `Repair.notes` says so. `verify`,
  `diagnose` and a plain `fix` keep raising E703, with a message naming the
  repair. D3's §2.1 sentence and a C.1 row. No signature changes.
- **Tests.** Rust `s2_1_an_unknown_digest_algorithm_is_not_amended` (E703; the
  source's bytes unchanged; both `digests` values);
  `test_S2_1_an_amendment_refuses_an_algorithm_it_cannot_compute`;
  `test_S2_1_fix_restamps_an_unknown_algorithm_and_records_it`;
  `test_S13_1_an_unreadable_digest_refuses_an_amendment_as_E703`; CLI
  `test_S2_1_fix_names_the_repair_for_an_unknown_algorithm`.
- **Docs.** `docs/guides/validate.md` (repair), `docs/reference/cli.md#medh5-fix`.
- **Changelog.** Behaviour changes, area `writer` (one entry with 04.1 is fine).

### 04.3 The clinical hold holds (PR 1, D16)

- **State at 2.0.2.** Three holes in `check_held` (`sample/writer_clinical.rs`),
  each a silently wrong result:
  1. It reads the *output* path (`self.path`). When `amend_into(src, out)`
     writes elsewhere, which the clinical augmenter does
     (`clinical/augment.rs`), it reads no file or an earlier output, and passes.
  2. It compares attested payloads by dataset digests only, as 1.1 §7.3 says,
     so an edit to an attested image's `rescale_slope` passes.
  3. `amend_into(p, p, None, &["clinical"])` never fills `inherited_events`
     (`writer.rs`), so `check_held` returns on its first test: a version
     re-added under its old id with a new payload or clock is not held, against
     1.1 §7.3 ("one added again in the same amendment is held to what it was").
     Found by reading the code; reproduce it in the test first.
- **Change.** `check_held` reads `self.source.as_deref().unwrap_or(&self.path)`;
  it also compares the §13.2 attribute lines of attested objects
  (`canonical_attrs`, `integrity/digest.rs`) and reports E809; `amend_into`
  fills `inherited_events` from the source even when the clinical group is not
  carried over, so every re-added version is held. D16's text in 1.1 §7.3 and an
  Appendix A row.
- **Tests.** `test_S7_3_an_attested_image_s_attributes_are_immutable`;
  `test_S7_3_an_amendment_to_another_path_is_held_to_its_source`; Rust
  `s7_3_amend_into_without_clinical_holds_a_readded_version`.
- **Changelog.** Fixed, area `clinical`.

### 04.4 Lost updates are refused (PR 1, D29)

- **State at 2.0.2.** `amend_into` drops its source handle once it has inherited
  the content (`writer.rs`), and `AtomicFile::commit` renames over the target
  without checking anything (`h5/file.rs`). 1.0 §14.4: writers MUST hold
  exclusive access, and an amend MUST default to copy-on-write. Two amendments,
  or an amendment and a `repack`, both succeed, and the first one's work
  disappears. The engine already reads a file identity from HDF5's own
  descriptor on Unix (`h5/ops.rs`, `opened_identity`).
- **Change.** A crate-private `Snapshot` of the source, taken in `amend_into` and
  in `atomic_rewrite` (`h5/file.rs`): on Unix the identity from HDF5's
  descriptor; on Windows `(file_size, last_write_time)` (creation time excluded:
  Windows carries it over when a file is replaced by rename; W21-05 PR 3 widens
  the Windows identity later); on both, the declared `content_id`. It is
  checked twice: in `check_commit`, before the source is read again, and in
  `AtomicFile::commit` just before `replace()` (`AtomicFile::expect_unchanged(&mut
  self, source: PathBuf, seen: Snapshot)`, crate-internal). A change is refused
  with `Error::File` (Python `MEDH5FileError`): "'x.medh5' changed after this
  amendment read it; nothing was written: amend it again". No opt-out. The
  optional informative sentence of D29 in §14.4. This is detection, not
  exclusion: the last check and the rename are two calls, and no portable
  rename compares before it replaces, so two commits that both pass the last
  check inside that window can still both rename. Say so in
  `design-rationale.md`; closing the window needs the exclusive access §14.4
  already asks of writers.
- **Tests.** `test_S14_4_a_commit_refuses_a_source_another_writer_replaced`, and
  `..._modified_in_place`; `test_S14_4_an_amendment_that_read_before_another_committed_is_refused`
  (the two run one after the other, not raced);
  Rust `s14_4_a_source_replaced_before_the_rename_is_not_overwritten` (the
  `BEFORE_RENAME` hook in `h5/file.rs`), `s14_4_a_repack_refuses_a_source_that_changed`.
- **Docs.** `docs/reference/python-api.md` (amend), `docs/reference/curation.md`,
  `docs/explanation/refusals.md`, `docs/explanation/design-rationale.md` (why
  detection rather than a lock).
- **Changelog.** Behaviour changes, area `files`: "**An amendment refuses to
  replace a file that changed after it read it.**"

### 04.5 Remove an image or a grid; edit an image's attributes (PR 2)

- **State at 2.0.2.** Only annotations and transforms can be removed
  (`sample/writer_annotations.rs`); each remover unlinks the group, sets
  `compact = true` (the commit rebuilds the file) and refuses an unknown id with
  `Error::invalid`. No `remove_image`, `remove_grid` or attribute editor: fixing
  one image's `modality` means rewriting its pixels.
- **Change.**
  - Rust: `remove_image(&mut self, image_id: &str) -> Result<()>`,
    `remove_grid(&mut self, grid_id: &str) -> Result<()>`,
    `set_image_attrs(&mut self, image_id: &str, edit: ImageEdit) -> Result<()>`,
    with `#[non_exhaustive] #[derive(Debug, Clone, Default)] pub struct
    ImageEdit` whose pub fields are `Option<Option<T>>` (outer: touch it; inner
    `None`: remove it) for `modality`, `value_type`, `value_units`,
    `channel_names`, `rescale_slope`, `rescale_intercept`, `window_center`,
    `window_width`, `valid_mask`, `prov`.
  - Each sets `compact` and refuses an unknown id with `Error::invalid`.
    Everything else is checked at commit, so the calls may come in any order:
    the commit's semantic validation reports a grid still named (E101 or E503),
    the last image (E201) and the last grid (E111).
  - `remove_image` drops the image's `/meta` `acquisition` entry and its mention
    in `series_uids` at the time of the call (carrying them to an image re-added
    under the same id would be a guess).
  - `set_image_attrs` refuses `grid`, `digest`, the pyramid attributes and names
    the specification does not define, and checks values as `add_image` does:
    E203, E204, a finite rescale, E205 when a required attribute would be
    removed.
  - Python, in the binding and `_core.pyi`: `remove_image(image_id: str) ->
    None`, `remove_grid(grid_id: str) -> None`, `set_image_attrs(image_id: str,
    **attrs: Any) -> None` (the stub lists the accepted names as keyword-only
    parameters).
- **Tests.** `test_S14_4_remove_image_leaves_no_bytes_behind`;
  `test_S14_4_removals_may_come_in_any_order`;
  `test_S3_2_remove_grid_is_refused_at_commit_while_named`;
  `test_S4_1_set_image_attrs_edits_and_refuses_grid` (E205 when `modality` is
  removed).
- **Docs.** `docs/reference/python-api.md` (the writer table),
  `docs/reference/curation.md`.
- **Changelog.** Added, area `writer`.

### 04.6 `add_mask` returns its group (PR 2)

- **State at 2.0.2.** `add_mask` returns `()` (`writer_annotations.rs`; stub in
  `_core.pyi`), while every other `add_*` returns its `Group`.
- **Change.** Python returns the `Group` now (the binding looks it up after the
  write). Rust keeps `-> Result<()>` until 3.0 (ledger line present).
- **Tests.** `test_S6_3_add_mask_returns_its_group`.
- **Changelog.** Added, area `python-api`.

### 04.7 The default `sample_id` and pack key (PR 3)

- **State at 2.0.2.** `SampleWriter::new` takes the file name up to its *first*
  dot as `sample_id` and the same value for `subject_id` (`writer.rs`);
  `default_key` does the same for pack keys (`collection.rs`). Docs and help say
  "stem" (`medh5/collection.py`, `docs/reference/cli.md`,
  `docs/reference/python-api.md`, `crates/medh5-cli/src/app.rs`). `migrate` and
  the conformance builder use the real stem. So `1.2.840.113619.2.55.medh5` gets
  the sample id `1`; `case.v2.medh5` and `case.v3.medh5` both become `case`
  (`pack` refuses with E003; a manifest holds two samples under one id);
  `.hidden.medh5` fails "[E005] identity requires both sample_id and subject_id",
  naming neither the file nor the cause. Switching now would silently rename
  samples, and split `P001.ct`/`P001.mr` into two subjects, the same patient on
  both sides of a subject-level split.
- **Change.**
  - `pub(crate) fn default_ids(path: &Path) -> Result<DefaultIds>` in
    `writer.rs`, giving the 2.x id and the 3.0 id.
  - When no `sample_id` is given and the two differ: `FutureWarning` from the
    binding (`SampleWriter.__new__`, `create`, through `deprecation.rs`):
    "medh5 3.0 names this sample 'case.v2' (the file stem), not 'case': pass
    sample_id= (and subject_id=) to keep 'case'". `pack` warns the same way per
    key; the CLI prints a note.
  - An id that comes out empty is refused with a message naming the path (a
    fix).
  - Docs and help: "the file name up to its first dot".
  - 3.0 (ledger): the stem; `subject_id` defaults to it only when the stem has no
    dot, otherwise refused ("pass subject_id: a dot may separate a subject from a
    series"). The Rust API is unchanged in 2.1.
- **Tests.** `test_S12_1_a_dotted_name_warns_that_its_default_id_will_change`;
  `test_S12_1_an_undotted_name_or_explicit_id_does_not_warn`;
  `test_S12_1_a_name_that_gives_no_id_is_refused_by_name`;
  `test_S2_2_pack_warns_where_a_default_key_will_change`.
- **Changelog.** Deprecated, area `writer`; Fixed for the empty-id refusal.

### 04.8 float16 displacement fields (PR 4)

- **State at 2.0.2.** `encode_transform` passes `DType::F32` to
  `encode_displacement` (`writer_annotations.rs`), though the encoder takes a
  dtype (`transforms/encode.rs`). `FLOAT16_SAFE_VOXELS = 64.0`
  (`transforms/model.rs`) is exported and unused. 1.0 §10.4: float16 is
  permitted and RECOMMENDED for displacements below ~64 voxels. Reading,
  validating, amending and recompressing a float16 field already work (verified
  with a planted field: W903 only at strict).
- **Change.** Rust `add_transform_with(&mut self, transform_id, kind, from_frame,
  to_frame, spec: TransformSpec, field_dtype: FieldDtype) -> Result<hdf5::Group>`
  with `#[non_exhaustive] pub enum FieldDtype { Float32, Float16, Auto }`;
  `add_transform` delegates with `Float32`; `TransformSpec` unchanged. Python
  `add_transform(..., field_dtype="float32")` (binding and stub). `"auto"` stores
  float16 when the largest Euclidean displacement in voxels of the field grid is
  below `FLOAT16_SAFE_VOXELS` (`index` vectors are in voxels; `world` vectors are
  divided by the grid's smallest spacing). An explicit `"float16"` is refused
  (uncoded `Error::invalid`) when a finite value would overflow to infinity
  (beyond 65504). Both are refused for any kind but `displacement`. The default
  stays float32, permanently.
- **Tests.** `test_S10_4_a_float16_field_is_stored_and_read_as_written`;
  `test_S10_4_auto_stores_float16_below_the_bound_and_float32_above` (index and
  world vectors); `test_S10_4_a_field_float16_cannot_hold_is_refused`;
  `test_S10_4_field_dtype_is_refused_for_other_kinds`.
- **Docs.** `docs/guides/registration.md` (field storage),
  `docs/reference/python-api.md`.
- **Changelog.** Added, area `geometry`.
