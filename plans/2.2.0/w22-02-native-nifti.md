# W22-02 A native NIfTI converter

| | |
|---|---|
| Branch | `w22/native-nifti` |
| Size | L (two to three weeks) |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | the consolidated helpers and the float64 qform (W21-13 PR 1), `--plane-as-volume` (PR 5), `import-seg-nifti` and the converter flags (PR 6), the converter write sites through `stage_file` (PR 9) |
| Owns | `crates/medh5/src/convert/nifti/` (new), the NIfTI bindings, `medh5/io/nifti.py` (becomes a facade), the native routes in `crates/medh5-cli/src/convert.rs`, `crates/medh5/tests/nifti_parity.rs` (new) |
| Decisions | E3 |

## Session prompt

> Implement workstream W22-02 (a native NIfTI converter) of medh5 2.2.0 in this
> repository. First read `plans/README.md`, `plans/2.2.0/README.md` and
> `plans/2.2.0/decisions.md` (E3), then `plans/2.2.0/w22-02-native-nifti.md` in
> full. PR 1 records the parity corpus from the released 2.1.0 converter and must
> land before any NIfTI code changes: install `medh5==2.1.0` from PyPI into a
> separate virtual environment to record it. The contract is byte-for-byte
> parity: same report JSON, same `content_id`, same refusals, on every platform.
> Untrusted gzip is inflated in pure Rust, from memory, with a size cap, and no
> zlib `gz*` symbol may be linked. Write tests first, run the local gate from
> `CLAUDE.md`, add fragments, open `W22-02: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 02.1 A parity corpus recorded from 2.1.0 | S–M |
| 2 | 02.2 The header codec and reading, with safe gzip | M |
| 3 | 02.3 Import (`from_nifti`) in the engine | M–L |
| 4 | 02.4 Export (`to_nifti`) and `import_seg_nifti` in the engine | M |
| 5 | 02.5 Python's converter becomes a facade; nibabel a test dependency | S–M |
| 6 | 02.6 The standalone binary converts NIfTI itself | S |

## Items

### 02.1 A parity corpus recorded from 2.1.0 (PR 1)

- **State at 2.1.0 (planned).** `medh5/io/nifti.py` (about 1,700 lines) runs over
  nibabel: `read_nifti` (refusals, geometry notes, units, rescale kept unapplied,
  squeeze, RAS→LPS, transpose, 2-D planes, fourth-axis evidence, time axis,
  b-value sidecars), `from_nifti` (same-grid checks, label-set minting, identity
  rules, writes, notes), `to_nifti`/`for_export` (sform_code 2, qform_code 0, xyzt
  in mm, magic `n+1`, gzip with mtime 0), `import_seg_nifti`. The native CLI hands
  every converter to Python.
- **Change.** `tests/data/nifti-parity/`: small generated inputs (both byte
  orders; `scl_slope`/`scl_inter`; sform-only, qform-only, both; 2-D, 3-D, 4-D;
  b-value sidecars; NIfTI-2; `.hdr`/`.img`; malformed headers; every documented
  refusal), and for each the 2.1.0 import's report JSON, `content_id` and refusal
  text, and the 2.1.0 export's header fields and voxel bytes, recorded by a script
  run under `medh5==2.1.0` (commit the script, the inputs and the records; keep
  inputs small).
- **Tests.** A test that the records exist for every input and were made by
  2.1.0 (`generator` in each).

### 02.2 The header codec and reading, with safe gzip (PR 2)

- **Change.** `crates/medh5/src/convert/nifti/{header,read}.rs`: a hand-written
  codec for NIfTI-1 (348 bytes) and NIfTI-2 (540 bytes), both byte orders,
  `.nii`, `.nii.gz`, `.hdr`/`.img`. gzip through flate2's default
  `rust_backend` (miniz_oxide), memory buffers only, decompressed size capped at
  the byte count the header implies. New crates (flate2, miniz_oxide, adler2,
  crc32fast): regenerate `THIRD_PARTY_NOTICES`; `SECURITY.md` gains no C row but
  states the gz constraint. A CI step asserts no `gz*` symbol is linked (`nm` on
  the extension and the binary).
- **Tests.** `gzip_bomb_is_refused`, `truncated_gzip_is_refused`,
  `nifti2_and_hdr_img_read`, header round trips; a fuzz target for the header
  codec (beside 2.1.0's targets).

### 02.3 Import in the engine (PR 3)

- **Change.** `medh5::convert::nifti::{read_nifti(path: &Path, options:
  &ReadOptions) -> Result<NiftiRead>, from_nifti(images: &[(String, PathBuf)],
  out: &Path, options: &ImportOptions, report: &mut ConversionReport) ->
  Result<()>}`, every options struct `#[non_exhaustive]` with `Default`, notes
  through `convert/report.rs`, each rule ported with its 2.1.0 wording.
- **Tests.** `crates/medh5/tests/nifti_parity.rs`: every input's report JSON and
  `content_id` equal 2.1.0's, byte for byte, on every CI platform (the float64
  qform makes that possible).

### 02.4 Export and `import_seg_nifti` in the engine (PR 4)

- **Change.** `to_nifti(sample: &Path, image: &str, out: &Path, options:
  &ExportOptions) -> Result<PathBuf>` (through `StagedFile`), and
  `import_seg_nifti`; exports compared by header fields and voxel bytes, not by
  the gzip stream.
- **Tests.** The parity test's export half.

### 02.5 Python's converter becomes a facade (PR 5)

- **Change.** `read_nifti`, `from_nifti`, `to_nifti`, `import_seg_nifti` call
  `_core`, with unchanged signatures; `tests/integrations/test_nifti.py` passes
  unchanged. nibabel moves to the test dependencies (the `nifti` extra stays, for
  compatibility, and the install docs say it is no longer needed).
- **Changelog.** Behaviour changes (none expected beyond speed; any difference
  the parity test allowed is listed), Performance, area `converters`.

### 02.6 The standalone binary converts NIfTI itself (PR 6)

- **Change.** `crates/medh5-cli/src/convert.rs` runs `from-nifti`, `to-nifti` and
  `import-seg-nifti` natively, with the path checks `medh5/cli.py` makes ported.
- **Tests.** The native binary's NIfTI commands without Python on `PATH` (a test
  that sets `MEDH5_PYTHON` to a non-existent interpreter).
- **Docs.** `docs/reference/rust.md`, `docs/reference/cli.md` (which commands are
  native), `docs/reference/converters.md`, `docs/install.md`.
- **Changelog.** Added, area `cli`.
