# W21-13 Converters

| | |
|---|---|
| Branch | `w21/converters` |
| Size | L overall; each pull request S–M, PR 8 M–L |
| Lands after | W21-00 PR 4, PR 5; PR 2 after W21-02 PR 1; PR 8 after W21-04 PR 2; PR 9 after W21-05 PR 1 |
| Blocks | nothing |
| Owns | `medh5/io/**` (except `itk.py`, `meds.py`, `fhir.py`), `crates/medh5-python/src/io.rs` (the bindings the converters share); touches `crates/medh5-python/src/convert.rs` (W21-06's) for PR 8's mask copy, `crates/medh5/src/convert/**`, the converter commands in `crates/medh5-cli/src/{app.rs,convert.rs}` and `medh5/cli.py`, `docs/reference/converters.md`, `docs/guides/{import-*.md,export.md}`; touches `writer_annotations.rs` (PR 8) and `validate/rules/geometric.rs` (PR 2) |
| Decisions | D10 |
| Backport | none |

## Session prompt

> Implement workstream W21-13 (converters) of medh5 2.1.0 in this repository.
> First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D10), then `plans/2.1.0/w21-13-converters.md` in
> full. Work from the latest `main` on `w21/converters`; land the nine items in
> order as one pull request, which merges once every item's "Lands after" has.
> Converters refuse rather than resample, guess a grid, invent a code or a
> designator, or fabricate a transform; every choice they make is a recorded
> note, and every behaviour change gets a refusal that names the option
> restoring it. CI's `test-min-deps` job runs the converters at the dependency
> floors (highdicom 0.23, pydicom 3.0.1, nibabel's floor): reproduce there
> before pushing a DICOM change. Write tests first, run the local gate from
> `CLAUDE.md`, add a fragment per item, open the `W21-13: …` pull request, and
> do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 13.1 One home for each converter helper; 13.2 the NIfTI qform in float64 | S |
| 2 | 13.3 Same-named RTSTRUCT ROIs; how a class's polygons combine (D10) | S–M |
| 3 | 13.4 RTSTRUCT POINT ROIs become `points` | S–M |
| 4 | 13.5 SEG: FRACTIONAL export; codes, category, laterality and algorithm | M |
| 5 | 13.6 Per-slice PET rescale; 13.7 `--plane-as-volume` | S |
| 6 | 13.8 The converter command line: every documented option, and `import-seg-nifti` | S–M |
| 7 | 13.9 nnU-Net: `imagesTs`, split claims, `--replace` | M |
| 8 | 13.10 A `labelmap=` input to `add_segmentation` | M–L |
| 9 | 13.11 Converter outputs through the engine's replace routine | S |

## Items

### 13.1 One home for each converter helper (PR 1)

- **State at 2.0.2.** `_from_patient` (`medh5/io/dicom_seg.py`) and `_to_patient`
  (`medh5/io/rtstruct.py`) compute the same grid↔patient mapping; `_sidecar` and
  `_sidecar_path` (`io/nifti.py`) resolve the same sidecar and each re-spell
  `NIFTI_SUFFIXES`; the SEG `segments` and `fractional` notes are written in two
  places (`dicom_seg.py`), and so is SEG volume allocation; `MM_PER_UNIT` lives in
  `nifti.py` and is imported by `dicom_seg.py`, `rtstruct.py` and
  `medh5/monai.py`, while the engine exports `geometry::grid::mm_per_unit` and
  `curation/tracking.rs` keeps a private copy.
- **Change.** `medh5/io/_patient.py` takes the patient↔grid mapping and the
  frame-of-reference helpers; `medh5/io/_frames.py` the functional-group
  placement helpers (`dicom_seg.py` onwards; 13.x and 2.2.0's enhanced import
  reuse them); one `_seg_notes()`; `_sidecar` defined through `_sidecar_path`;
  `MM_PER_UNIT` from the engine as `medh5.geometry.MM_PER_UNIT` (bound in
  `_core`), `medh5.io.nifti.MM_PER_UNIT` kept as an alias; `tracking.rs` calls
  `geometry::grid::mm_per_unit`. No behaviour change.
- **Tests.** The existing converter suites, unchanged;
  `test_mm_per_unit_is_the_engines`.
- **Changelog.** `<!-- no entry: internal -->`.

### 13.2 The NIfTI qform in float64 (PR 1)

- **State at 2.0.2.** nibabel builds the qform matrix in `np.longdouble`, 80-bit
  on x86-64: forcing float64 changed direction and spacing in the last bit and
  the `content_id` (`b89d7a…` against `aaf3b8…`). A qform-only NIfTI import's
  `content_id` therefore depends on the platform, and 2.2.0's native reader
  cannot claim parity.
- **Change.** `read_nifti` builds the qform in float64 itself (from the quatern
  parameters, as the NIfTI-1 standard defines), never through nibabel's
  long-double path.
- **Tests.** `test_S3_a_qform_import_is_the_same_on_every_platform` (a fixed
  expected `content_id` for a qform-only fixture, held on every CI platform).
- **Changelog.** Behaviour changes, area `converters`: qform-only imports on
  x86-64 can change in the last bit, and are now the same everywhere.

### 13.3 Same-named RTSTRUCT ROIs; how a class's polygons combine (PR 2, D10)

- **State at 2.0.2.** Minting classes, each key is `_key(ROIName)`
  (`rtstruct.py`): two ROIs named "GTV" fail "[E302] label set 'rtstruct':
  duplicate class keys ['gtv']". With an existing label set both resolve to one
  class (by key, then name, then `_named`) and merge without a note; roles come
  from nesting within each ROI (`_assign_roles`) while `_rasterize` nests per
  class, so a nested ROI of the same name becomes a hole: an 11×11 "Bladder"
  holding a 4×4 "Bladder" rasterises to 105 voxels instead of 121 (verified).
  §8.6 defines no combining rule; the validator never reads `contour_role`; the
  engine reader raises `Error::Index` on a role ≥ 2
  (`annotations/read_geometric.rs`). (SEG import already merges segments that
  share a label.)
- **Change.**
  - ROIs whose names make one key become one class, minting or not, with a
    decision note `merged_rois` naming the ROI numbers. Roles are recomputed over
    the merged class, plane by plane. If any polygon's depth in the merged class
    differs from its depth within its own ROI (one ROI's contour encloses, or is
    enclosed by, another's on the same plane), the import is refused, naming the
    ROIs and the plane. Crossing contours from different ROIs sit at one depth and
    are their union.
  - `from_rtstruct(..., roi_classes: Mapping[int, str] | None = None)`
    (ROINumber → key; CLI `--roi NUMBER=KEY`, repeatable) imports ROIs apart; the
    key must exist in an existing label set (E402), otherwise it is minted.
  - One engine depth function, shared by the validator and Python (bound in
    `_core`), replacing `_depths`.
  - D10: the §8.6 paragraph; the engine reader computes regions from depth and
    never fails on a role; a validator rule in the contours branch of
    `validate/rules/geometric.rs` reports W919 (role disagrees with depth parity)
    and E404 (a role other than 0 or 1). `codes.json`: W919 (revision 2, since
    1.0); the §15.2 row marked (r2); a C.1 row.
  - Corpus: `W919-contour-role-depth` (valid, semantic) and
    `E404-contour-role-value` (invalid).
  - If this merges before W21-02 PR 2, W919 is the first revision-2 code, and
    this pull request makes the table revision 2 (D1's Where): it rewrites D1's
    last sentence from "revision 1" to "revision 2", sets `codes.json`'s
    top-level revision, and turns on W21-02's pinned-revision test.
- **The Bladder kit** (`tests/kits.py`, `NestedSameNameRoiKit`): one CT plane
  of 16×16 voxels at 1 mm and an RTSTRUCT on it with two ROIs both named
  "Bladder": ROI 1 a square contour enclosing 11×11 voxel centres, ROI 2 a
  square enclosing 4×4 voxel centres strictly inside it, on the same plane.
  Drawn, the class covers 121 voxels; 2.0.2 imports it as 105 (121 − 16).
- **Tests.** `test_S8_6_same_named_rois_are_one_class` (both paths, with the
  note); `test_S8_6_nested_same_named_rois_are_refused` (the Bladder kit, both
  paths); `test_S8_6_roi_classes_import_rois_apart`;
  `test_S8_6_role_disagreeing_with_depth_is_W919`;
  `test_S8_6_unknown_role_is_E404`; `test_S8_6_a_reader_takes_the_depth_not_the_role`.
- **Docs.** `docs/reference/converters.md` (RTSTRUCT), `docs/reference/annotations.md`,
  `docs/explanation/refusals.md`.
- **Changelog.** Behaviour changes, area `converters` (the silent merge now
  writes a note or refuses); Fixed, area `annotations` (the reader).

### 13.4 RTSTRUCT POINT ROIs become `points` (PR 3)

- **State at 2.0.2.** `_areas` (`rtstruct.py`) skips POINT and OPEN_* ROIs with a
  `contour_type` warning and marks no class examined for them. `to_rtstruct`
  writes every contour `CLOSED_PLANAR` and every ROI "ORGAN". §8.5 `points` holds
  "landmarks and point sets"; §8.6 planar polygons only. Open curves need a new
  kind (`polylines`, format 1.2, 2.2.0).
- **Change.**
  - Each POINT contour becomes one row of the existing `points` kind (a POINT
    contour holding more than one triple is refused as malformed); rows use
    `space="world"` on the contours' frame; `class_ids` take the ROI's class;
    `task="other"` (markers are not detection findings and must not satisfy
    `det`); annotation id `points_id`, default `f"{ann_id}_points"`. A structure
    set holding only points writes no contours annotation. An explicit
    `annotated_classes` list is split: a class drawn as points is examined in the
    points annotation, every other class it names in the contours annotation;
    `"all"` applies to the contours annotation only.
  - `from_rtstruct(..., points_id: str | None = None, open_curves: str = "omit")`;
    CLI `--points-id ID`, `--open-curves omit` (only `omit` in 2.1; `polylines`
    arrives in 2.2.0). The OPEN_* warning names the option.
  - `to_rtstruct(..., points: str | None = None)`; CLI `--points ANN`: POINT items
    with `RTROIInterpretedType` "MARKER"; a class in both annotations is refused.
- **Tests.** `test_S8_5_rtstruct_point_rois_import_as_points` (rows, frame, task,
  no warning); `test_S8_5_point_only_structure_set_writes_no_contours`;
  `test_S8_5_points_round_trip_through_to_rtstruct` (POINT items, "MARKER",
  coordinates equal to float32 precision);
  `test_S8_5_annotated_classes_split_across_points_and_contours`.
- **Docs.** `docs/reference/converters.md`, `docs/guides/export.md`,
  `docs/explanation/refusals.md`.
- **Changelog.** Added, area `converters`.

### 13.5 SEG: FRACTIONAL export; codes, category, laterality and algorithm (PR 4)

- **State at 2.0.2.** `to_dicom_seg` (`medh5/io/dicom_seg.py`) writes `BINARY`
  from `dense()`, thresholding a probmap (documented in `docs/guides/export.md`).
  Every segment gets the category *and* the type `SCT 91723000 "Anatomical
  Structure"`, and `algorithm_type` MANUAL: wrong for a lesion, a device or a
  model's output. Import (`_segments`) keeps only the type code (no category, no
  modifier) and stores the designator "SCT" as `system` where §5.2 spells it
  "SNOMED-CT"; `_mint` merges segments sharing a label even when their modifiers
  differ, so Left and Right Kidney become one class.
- **Change.**
  - **The rule for a concept DICOM requires and the sample does not give.** A
    SEG segment must carry a property category, a property type and an
    algorithm type. Where the sample states one, the exporter carries it (the
    tables below); where it states none, the exporter writes the generic value
    2.0.2 already writes (SCT 91723000 "Anatomical Structure" for a concept,
    MANUAL for the algorithm type) and records a `guess` in the conversion
    report naming each class it applied to. It never writes a specific concept
    the sample does not state, and never invents a designator: this is the one
    place a converter writes a value it was not given, and only the generic
    one, because DICOM gives no way to leave it out.
  - **FRACTIONAL.** `to_dicom_seg(..., segmentation_type="binary",
    fractional_type="PROBABILITY")`; `"fractional"` only for `probmap` (any other
    kind refused); reads `probabilities()` as float32; highdicom scales to
    `max_fractional_value=255`. The report states the quantisation bound (≤
    1/510), how many voxels change side of the stored threshold after
    quantisation, and that `threshold` and `normalized` are not carried. An ignore
    region is refused and only examined classes are written, as for binary. A
    binary export of a probmap gains a decision note "thresholded at t".
  - **Systems and designators**, one table in both directions, in
    `medh5/io/_codes.py`: SNOMED-CT↔SCT, RadLex↔RADLEX, FMA↔FMA, UBERON↔UBERON,
    LOINC↔LN, ICD-10↔I10, DCM↔DCM. A system the table does not know falls back
    to the generic concept with a guess note; a designator is never invented.
    Comparisons normalise through the table; import keeps the designator the file
    states.
  - **Property type**: the class's first code; a class without one gets the
    generic concept, with a guess note naming those classes.
  - **Property category**: `properties["dicom_category"]` when present (import
    writes it when minting); otherwise from `category`: organ, vessel, bone → SCT
    91723000 "Anatomical Structure"; lesion → SCT 49755003 "Morphologically
    Abnormal Structure"; device → SCT 260787004 "Physical object"; region → SCT
    309825002 "Spatial and Relational Concept"; anything else → SCT 91723000 with
    a guess note.
  - **Laterality** ↔ `SegmentedPropertyTypeModifierCodeSequence` (left SCT
    7771000, right 24028007, bilateral 51440002); highdicom has no parameter for
    it, so the exporter sets (0062,0011) on the description dataset. Import reads
    it into `laterality`; `_mint` refuses one label carrying different modifiers.
  - **Algorithm type**, from provenance: `predict` → AUTOMATIC, with an
    `AlgorithmIdentificationSequence` naming the agent's software and version,
    family DCM 123110 "Artificial Intelligence"; `annotate` or `review` → MANUAL;
    anything else → MANUAL with a guess note.
  - CLI `to-dicom-seg --segmentation-type binary|fractional --fractional-type
    PROBABILITY|OCCUPANCY`.
- **Tests** (`tests/integrations/test_dicom_seg.py`, under its `highdicom`
  fixture): `test_fractional_export_round_trips_within_1_510`;
  `test_fractional_export_refuses_a_labelmap`;
  `test_S5_2_codes_category_laterality_round_trip`;
  `test_S5_2_lesion_category_is_morphologically_abnormal`;
  `test_S5_2_unknown_system_is_a_noted_guess`;
  `test_S5_2_mint_refuses_one_label_with_two_modifiers`;
  `test_S11_1_a_predicted_annotation_exports_automatic`;
  `test_S5_2_an_unstated_concept_is_the_generic_one_with_a_guess`.
- **Docs.** `docs/guides/export.md` (rewrite "Export is binary, and it
  thresholds"), `docs/reference/converters.md`, `docs/explanation/refusals.md`.
- **Changelog.** Added; Behaviour changes, area `converters` (categories of
  lesion, device and region classes; AUTOMATIC for predictions; the modifier
  refusal; import records category and laterality).

### 13.6 Per-slice PET rescale (PR 5)

- **State at 2.0.2.** `_plan_series` requires one rescale for the series
  (`medh5/io/dicom.py`, through `_agreed`), whose own docstring calls a per-slice
  PET rescale "ordinary": such a series is refused. PET is written with
  `value_type` "intensity" and no units.
- **Change.** `from_dicom(..., per_slice_rescale="refuse")` and
  `read_series(..., per_slice_rescale="refuse")`; CLI `from-dicom
  --per-slice-rescale refuse|apply`; the refusal names the option. `"apply"`
  computes `stored × slope_i + intercept_i` in float64 slice by slice and stores
  float32 with no rescale attributes, records a decision note (the distinct
  slopes and their slices), and, only when slices disagreed, the activity gets
  `{"per_slice_rescale": "applied"}`. A slice stating no rescale among slices
  that do is still refused. SUV is never computed; `value_type` and units do not
  change. A uniform series is byte-identical whatever the option.
- **Tests.** `test_S4_2_per_slice_rescale_is_refused_naming_the_option`;
  `test_S4_2_per_slice_rescale_applied_stores_physical_values`;
  `test_S4_2_uniform_rescale_is_unchanged_by_apply` (equal `content_id`);
  `test_S4_2_slice_without_rescale_is_refused`.
- **Docs.** `docs/guides/import-dicom.md`, `docs/reference/converters.md`,
  `docs/reference/cli.md`.
- **Changelog.** Added, area `converters`.

### 13.7 `--plane-as-volume` (PR 5)

- **State at 2.0.2.** 1.0 §3.6 already says a slice that lies elsewhere, or not
  parallel to z = 0, "is a one-slice 3-D grid, which keeps where it lies".
  `_reduce_plane` (`io/nifti.py`) refuses a plane more than `PLANE_TOL_MM = 1e-3`
  off z = 0 and a plane not parallel to x-y; both refusals say "save it as (x,
  y, 1)". Making it the default would change a grid's dimensionality on a header
  offset (two slices of one series on different kinds of grid).
- **Change.** `read_nifti(..., plane_as_volume: bool = False)`,
  `from_nifti(..., plane_as_volume=False)`, `import_seg_nifti(...,
  plane_as_volume=False)`; CLI `from-nifti --plane-as-volume` (and on
  `import-seg-nifti`, PR 6). With it, every 2-D input (images and masks; planes at
  z = 0 too, so a cohort comes out uniform) is reshaped to `(x, y, 1)` before the
  transpose and keeps its full 4×4 affine; the third axis is the affine's third
  column, which must be non-zero and orthogonal to the first two within
  `ORTHONORMAL_TOL`, otherwise refused (no normal or thickness is invented). The
  result equals importing the plane re-saved as `(x, y, 1)`. A decision note
  `plane_as_volume` records the plane's z and normal; the activity's params gain
  `"plane_as_volume": true` only when a plane was reshaped. Without the flag,
  nothing changes but the refusals name the flag.
- **Tests** (`tests/integrations/test_nifti.py`):
  `test_S3_6_plane_as_volume_keeps_its_z` (a plane at z = 12.5 mm gives a `(1, y,
  x)` grid with origin z = 12.5, digests equal to the re-saved form);
  `test_S3_6_plane_as_volume_refuses_a_degenerate_third_column`;
  `test_S3_6_off_plane_refusal_names_the_flag`;
  `test_S3_6_plane_as_volume_changes_no_volume_import`;
  `test_S3_6_plane_mask_matches_a_one_slice_grid`.
- **Docs.** `docs/guides/import-nifti.md`, `docs/reference/converters.md`,
  `docs/reference/cli.md`.
- **Changelog.** Added, area `converters`.

### 13.8 The converter command line (PR 6)

- **State at 2.0.2.** `crates/medh5-cli/src/app.rs` `convert()` and
  `report_args`: the Python converters accept more than the command line exposes;
  `to-nifti` has no `report_args`; `migrate` passes a hard-coded `"balanced"` codec
  (`crates/medh5-cli/src/convert.rs`); `import_seg_nifti` has no command. With
  only a liver mask, `from_nifti(annotated_classes=["liver","tumor"])` raises a
  bare `KeyError` ("label set 'converted' has no class 'tumor'"): an importer that
  mints classes cannot record "examined, absent" (SEG and RTSTRUCT too).
  `to_rtstruct` writes a `StructureSetLabel` longer than 16 characters, invalid
  DICOM.
- **Change.** One typed flag per parameter (a generic `--param k=v` could not be
  documented or checked):

  | Flag | Commands | Python |
  |---|---|---|
  | `--annotated-classes all_given\|all\|K1,K2` | from-nifti, from-dicom-seg, from-rtstruct, import-seg-nifti | `annotated_classes=` |
  | `--codec PROFILE` | from-nifti, from-dicom, from-nnunet, migrate | `codec=` |
  | `--class K` (repeatable) | to-dicom-seg | `classes=` |
  | `--series-description TEXT` (LO, ≤ 64; longer refused) | to-dicom-seg | `series_description=` |
  | `--label TEXT` (SH, ≤ 16; longer refused) | to-rtstruct | `label=` |
  | `--mask-id ID` (requires `--rasterize`) | from-rtstruct | `mask_id=` |
  | `--coord-system LPS\|RAS` | from-nnunet | `coord_system=` |
  | `--file-ending .nii.gz\|.nii` | to-nnunet | `file_ending=` |
  | `--value-units NAME=UNIT` | from-nifti | `value_units=` |
  | `--label-set FILE` | from-nifti | new `label_set=` |
  | `--report FILE`, `--json` | to-nifti | `report=` |

  - New command `import-seg-nifti SAMPLE --mask NAME=PATH … [--id ANN] [--grid G]
    [--assume-aligned] [--assume-geometry] [--plane-as-volume]`.
  - A named class the input does not supply is "examined, absent": where the
    import mints classes, it is minted after the supplied ids, with a decision
    note; otherwise it must already be in the sample's label set (E402). This
    replaces the `KeyError`.
  - `to_rtstruct` refuses a label longer than 16 characters (a behaviour change).
  - `from_nifti(..., label_set: LabelSet | str | os.PathLike[str] | None = None)`
    (a path is loaded with `medh5.labels.load_file`); `medh5/cli.py`'s `_files()`
    learns `--label-set` and the mask paths (the path checks 2.0.2 added).
  - `migrate` reads `--codec`.
- **Tests.** `tests/tools/test_cli.py::test_convert_flags_reach_the_converter`
  (parametrised over the table, asserting each flag's effect);
  `test_S11_3_annotated_classes_mint_an_examined_absent_class` (NIfTI, SEG,
  RTSTRUCT); `test_S11_3_annotated_class_missing_from_label_set_is_E402`;
  `test_label_set_fixes_class_ids_across_imports`; the documented flags are
  checked by `test_docs_examples.py`.
- **Docs.** `docs/reference/cli.md`, `docs/reference/converters.md`.
- **Changelog.** Added; Behaviour changes (the label length); Fixed (the
  `KeyError`), area `converters`.

### 13.9 nnU-Net: `imagesTs`, split claims, `--replace` (PR 7)

- **State at 2.0.2.** `cases()` lists `imagesTr` only (`medh5/io/nnunetv2.py`),
  so a round trip through `to_nnunetv2(unlabeled="test")` drops the test cases
  without a note. `to_nnunetv2` writes unlabeled cases to `imagesTs` under
  `unlabeled="test"` and refuses stray case files (`_require_no_strays`).
- **Change.**
  - `from_nnunetv2(..., include_test=False)`, CLI `--include-test`: `imagesTs`
    cases imported, labels from `labelsTs` where present; a case in both
    partitions refused; each imported case gets a split claim `{set_id:
    "nnunetv2:<dataset>", partition: "train"|"test", assigned_by: "nnU-Net v2
    dataset layout", manifest_sha256: sha256(dataset.json)}`. Without the flag
    nothing changes, but a warning counts the `imagesTs` cases left out.
  - `to_nnunetv2(..., split: str | None = None)` (CLI `--split SET_ID`) sends a
    case that claims `test` in that set to `imagesTs`/`labelsTs`;
    `_require_no_strays` also covers `labelsTs`.
  - `to_nnunetv2(..., replace=False)`, CLI `--replace`: `out` must be a directory,
    not a link, holding a `dataset.json`, or the export is refused. The whole
    dataset is built in a hidden sibling `.<name>.medh5-staging-<random>` on the
    same filesystem (non-NIfTI files the export does not own copied in); two
    renames swap it in (the old dataset to `.<name>.medh5-old-…`, then staging to
    `out`; if the second fails, the first is rolled back); the old tree is then
    deleted. The report lists files removed and carried over.
- **Tests.** `test_from_nnunet_include_test_round_trips_unlabeled_cases`;
  `test_from_nnunet_include_test_writes_split_claims`;
  `test_S12_3_split_routes_test_cases_to_imagesTs`;
  `test_replace_swaps_a_dataset_and_carries_other_files`;
  `test_replace_refuses_a_directory_without_dataset_json` (and its symlink
  variant); `test_replace_failure_leaves_the_old_dataset` (a fault injected at
  the second rename).
- **Docs.** `docs/guides/export.md`, `docs/reference/converters.md`,
  `docs/reference/cli.md`.
- **Changelog.** Added, area `converters`.

### 13.10 A `labelmap=` input to `add_segmentation` (PR 8, after W21-04 PR 2)

- **State at 2.0.2.** nnU-Net import builds one bool mask per class, unioning
  regions (`_masks_from`), passes them with `annotated_classes="all"`, and the
  binding copies each mask again (`typed().into_owned()`,
  `crates/medh5-python/src/convert.rs`): measured, 84 MB of masks add 92 MB of
  peak at 40 classes × 128³; at 104 classes × 512²×500 that is 13.6 GB of masks
  and about 27 GB at peak. `SegmentationSource` is an exhaustive enum and
  `SegmentationOptions` has only pub fields (no variant, no field).
- **Change.**
  - Python: `add_segmentation(ann_id, *, grid, labelmap: NDArray[np.integer] |
    None = None, label_values: Mapping[int, ClassKey | Sequence[ClassKey]] | None
    = None, ignore_value: int | None = None, …)`; exactly one of masks,
    probabilities, instances or labelmap. A value maps to one class or, for an
    nnU-Net region, to several; 0 is background; a nonzero value mapping to
    nothing is refused.
  - Rust: `#[non_exhaustive] pub struct LabelmapSource<'a>` borrowing an integer
    `ArrayViewD`, built by `LabelmapSource::new(view, values)?.ignore_value(v)`;
    `SampleWriter::add_segmentation_labelmap(&mut self, ann_id: &str, grid: &str,
    source: LabelmapSource<'_>, options: SegmentationOptions) -> Result<(String,
    Option<OverlapStats>)>`; `annotations::select::analyse_labelmap`, analysing and
    encoding chunk by chunk. The binding borrows NumPy's buffer.
  - Payload, attributes and `OverlapStats` equal the masks route byte for byte.
    nnU-Net import switches to it: peak becomes the labelmap plus the output and
    the chunk buffers (about 0.5–1.5 GB at the size above).
- **Tests.** `test_S7_labelmap_input_writes_the_masks_bytes` (random labelmaps
  with regions and ignore, over every encoding `auto` picks); Rust
  `labelmap_source_matches_masks_source`, `labelmap_encoding_holds_one_chunk_of_planes`
  (a slab-budget test).
- **Docs.** `docs/reference/annotations.md` (writer inputs),
  `docs/reference/converters.md`, `docs/guides/performance.md`.
- **Changelog.** Added and Performance, area `converters`.

### 13.11 Converter outputs through the engine's replace routine (PR 9, after W21-05 PR 1)

- **State at 2.0.2.** `write_nifti` (`temporary_beside`, `nib.save`,
  `os.replace`), `_replace_text` (sidecars, `.bval`), nnU-Net's `dataset.json`
  (`Path.write_text`, in place), `dicom_seg.py` and `rtstruct.py` (`save_as`, in
  place): a crash during `save_as` leaves a truncated SEG where the previous one
  was.
- **Change.** `with _core.stage_file(target) as f:` around `nib.save(image,
  str(f.path))` (and `_withdraw_volume_statements`), and around each `save_as(str(
  f.path), …)`; `_core.write_atomic` for `_replace_text` and `dataset.json` (the
  unlinks that order the multi-file sidecar protocol stay). Remove these from
  W21-05's allow-list, with `medh5/io/nifti.py`'s import of
  `temporary_beside`; if W21-11 PR 3 has already moved `TaskManifest.save`,
  delete `temporary_beside` from `medh5/_files.py` and its allow-list entry
  here. Skip any site an earlier pull request here already moved.
- **Tests.** For each writer, an interruption (`nib.save` or `save_as` raising)
  leaves the previous file byte-identical and no temporary behind; W21-05's
  parametrised link/mode test covers the new sites.
- **Changelog.** Fixed, area `converters`.
