# W22-01 Format 1.2: `polylines`

| | |
|---|---|
| Branch | `w22/format-1.2` |
| Size | L overall: PR 1 M, PR 2 M–L, PR 3 M |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `version::format_versions()`, `annotations::kinds(version)`, `version::next_minor` (W21-02 PR 1); RTSTRUCT points and `--open-curves omit` (W21-13 PR 3); the code-table revision rule (D1) |
| Owns | `docs/spec/medh5-1.2.md` (new), `crates/medh5/src/version.rs`, the `polylines` encoder, reader and validator rules, `medh5/annotations/geometric.py` (`Polyline`), the RTSTRUCT open-curve paths |
| Decisions | E1, E2, E9 |

## Session prompt

> Implement workstream W22-01 (format 1.2: the `polylines` kind) of medh5 2.2.0
> in this repository. First read `plans/README.md`, `plans/2.2.0/README.md` and
> `plans/2.2.0/decisions.md` (E1, E2, E9), then
> `plans/2.2.0/w22-01-format-1.2.md` in full. Check that 2.1.0's version
> accessors and `next_minor` are on `main` (W21-02 PR 1). Work from the latest
> `main` on `w22/format-1.2`; land the three pull requests in order. This is a
> format version: 1.2 may add only what a 1.1 reader already skips (a new kind),
> a sample is written at 1.2 only when its content needs it, and a 1.1 reader
> must read a 1.2 file as its projection with W913.
> Write tests first (including a 2.1.0 reader reading a 2.2.0 file, through the
> 1.1 projection tests), write the specification as E1 gives it, run the local
> gate from `CLAUDE.md` (both conformance runs), add fragments, open `W22-01: …`
> pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 01.1 1.2 exists, and the version follows the content | M |
| 2 | 01.2 The `polylines` kind | M–L |
| 3 | 01.3 RTSTRUCT open curves import and export as polylines | M |

## Items

### 01.1 1.2 exists, and the version follows the content (PR 1, E2)

- **State at 2.1.0 (planned).** `version::format_versions()` is `["1.0", "1.1"]`;
  `required_for` (`version.rs`) and `written_version` (`sample/writer.rs`) look at
  profiles only; "a later minor" is `next_minor` of the newest supported version,
  so the W913 case and companion fixture T306 write "1.2".
- **Change.** `format_versions()` gains "1.2"; `annotations::kinds("1.2")` is 1.1's
  kinds plus `polylines` (the deprecated arrays keep 1.1's content).
  `version::required_for_content(profiles: &[&str], kinds: &[&str]) -> &'static
  str` (a new function; `required_for` keeps its signature and its answer for
  profiles); the writer uses it, so a sample is written at 1.2 only when it holds
  a polylines annotation, and an amendment removing the last one lowers the
  version (1.1 §2.1's rule for the clinical profile, applied to the kind). The
  projection (`sample/projection.rs`) already drops unknown kinds as W913: a test
  holds it for polylines. `docs/spec/medh5-1.2.md` with §1 (scope: one kind), §2
  (versioning, E1's text) and an Appendix A for its own changes; the README and
  the spec index link it; 1.0 §6.3's registry row. The W913 case and T306 now write
  "1.3" (derived; recount nothing else).
- **Tests.** `test_S2_1_a_sample_without_polylines_is_written_at_1_0_or_1_1`;
  `test_S2_1_a_polylines_annotation_raises_the_version_to_1_2`;
  `test_S2_1_removing_the_last_polylines_lowers_the_version`;
  `test_S2_2_the_later_minor_is_1_3`.
- **Changelog.** Added, area `specification`.

### 01.2 The `polylines` kind (PR 2, E1)

- **Change.**
  - The encoding of E1 (`vertices`, `polyline_offsets`, `polyline_class_ids`,
    optional `polyline_plane` and `instance_ids`), in the writer, the reader and
    the validator: E404 for a polyline of fewer than two vertices, E405 for
    shapes, E408 for offsets, E402 for classes; no new code (E408's summary
    names polyline offsets, under 1.0 §15.2's rule for a kind a minor adds,
    E1).
  - Rust: `#[non_exhaustive] pub struct Polyline` with `Polyline::new(vertices,
    class_id)` and builders `plane(axis, index)`, `instance_id(id)`;
    `encode_polylines`; `SampleWriter::add_polylines(&mut self, ann_id: &str,
    polylines: &[Polyline], options: GeometricOptions) -> Result<hdf5::Group>`
    (use the existing geometric options type if it is `#[non_exhaustive]`, else a
    new `PolylineOptions`).
  - Python: `medh5.annotations.geometric.Polyline(vertices, class_id, plane=(-1,
    0))`; `add_polylines(ann_id, polylines, *, grid=None, space="index",
    frame_uid=None, instance_ids=None, annotated_classes=None,
    closure="explicit", timepoints=None, prov=None, quality=None,
    derived_from=None, task="other", codec=None) -> Group`; a
    `PolylinesAnnotation` reader (`polylines()`, `to_world()`, `to_index()`);
    `_core.pyi`.
  - A polyline is never rasterised; a polylines annotation satisfies neither
    `seg` nor `det`.
  - Corpus: `shapes-polylines` (valid, declares 1.2), `E408-polyline-offsets`,
    `E404-polyline-one-vertex`; each case states `format_version` "1.2".
- **Tests.** `test_v1_2_S3_polylines_round_trip` (index and world, with planes and
  instance ids); `test_v1_2_S3_a_polyline_of_one_vertex_is_E404`;
  `test_v1_2_S3_offsets_that_do_not_delimit_are_E408`;
  `test_v1_2_S3_polylines_satisfy_no_profile`;
  `test_v1_2_S3_a_1_1_reader_projects_polylines` (the 1.1 projection reports W913
  for the annotation and validates the rest).
- **Docs.** `docs/spec/medh5-1.2.md` §3 (E1's text), `docs/reference/annotations.md`,
  `docs/reference/python-api.md`.
- **Changelog.** Added, area `annotations`.

### 01.3 RTSTRUCT open curves import and export as polylines (PR 3)

- **State at 2.1.0 (planned).** `from_rtstruct(..., open_curves="omit")` skips
  OPEN_PLANAR and OPEN_NONPLANAR ROIs with a warning naming the option; POINT ROIs
  become `points`.
- **Change.** `open_curves="polylines"` (CLI `--open-curves polylines`) imports
  them as a polylines annotation (id `f"{ann_id}_curves"`, or `curves_id=`),
  OPEN_PLANAR contours with their `polyline_plane`, OPEN_NONPLANAR with `axis =
  −1`; opt-in because `from_rtstruct` amends an existing sample and a polylines
  annotation raises its version to 1.2 (the import's report says so). `annotated_classes`
  is split as for points. `to_rtstruct(..., curves: str | None = None)` (CLI
  `--curves ANN`) exports a polylines annotation as OPEN_PLANAR (with a plane) or
  OPEN_NONPLANAR. §8.6 gains the informative mapping: POINT → `points`, OPEN_* →
  `polylines`.
- **Tests.** `test_v1_2_S3_open_planar_imports_as_polylines`;
  `test_v1_2_S3_open_nonplanar_has_no_plane`;
  `test_v1_2_S3_polylines_raise_the_version_to_1_2`;
  `test_v1_2_S3_curves_round_trip_through_to_rtstruct`.
- **Docs.** `docs/reference/converters.md`, `docs/guides/export.md`.
- **Changelog.** Added, area `converters`.
