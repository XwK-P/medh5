# W21-09 Curation: tracking and agreement

| | |
|---|---|
| Branch | `w21/curation` |
| Size | M overall; each pull request M |
| Lands after | W21-00 PR 4, PR 5 |
| Blocks | nothing |
| Owns | `crates/medh5/src/curation/{tracking.rs,agreement.rs}`, `crates/medh5-python/src/{tracking.rs,curation_tools.rs}` (agreement), `medh5/curation/{tracking.py,agreement.py}`, `crates/medh5-cli/src/inspect.rs` (`track`), the `agree` command |
| Decisions | D12, D13 |
| Backport | none |

## Session prompt

> Implement workstream W21-09 (curation: tracking and agreement) of medh5 2.1.0
> in this repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D12, D13), then `plans/2.1.0/w21-09-curation.md`
> in full. Work from the latest `main` on `w21/curation`; land the two items in
> order as one pull request. Coverage is the invariant: `class_ids` is what an
> annotation can express, `annotated_class_ids` what was looked for; never
> collapse the two. Deprecating a Rust method means migrating every workspace
> use in the same pull request, or `clippy -D warnings` fails. Write tests
> first, write D12's and D13's text with their Appendix C.1 rows, run the local
> gate from `CLAUDE.md`, add a fragment per item, open the `W21-09: …` pull
> request, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 09.1 Tracking says why an absence is or is not a finding | M |
| 2 | 09.2 Agreement across vocabularies, an explicit object match, and an honest `against` | M |

## Items

### 09.1 Tracking says why an absence is or is not a finding (PR 1, D12)

- **State at 2.0.2.** `Tracking::state_at` (`curation/tracking.rs`) returns
  `resolved` when an annotation at the visit carries instance ids, lists the
  class in `annotated_class_ids` and ignores nothing (2.0.2's conservative rule:
  any ignore region means `unexamined`). `is_new` and `is_resolved` read those
  states. Nothing locates the object. Verified with an identity registration and
  the follow-up grid 100 mm away: tp1 comes out `resolved` and `is_resolved` is
  True although the lesion maps to index −94.5…−90.5. 1.1 §7.4 already has
  `outside_fov` for assessments and says the helper's states are not redefined.
  `transform_between`'s `None` means both "same frame" and "no path", so it is the
  wrong tool here.
- **Change.** Three states stay; every unseen (object, visit) gets one reason.
  `located` and `class_examined` (state `resolved`); `outside_fov`, `ignored`,
  `unrelated`, `no_identity`, `not_examined` (state `unexamined`); when
  annotations disagree, the first in that order wins. A path with a nonlinear
  component does not locate the object (corners do not bound a warped box), and
  every observation must map or the object is not located. `class_examined` and
  `unrelated` keep 1.0's class-level rule where no geometry exists, and tell the
  user a registration would decide the case.
  - Rust (`curation/tracking.rs`):

    ```rust
    #[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Absence { Located, ClassExamined, OutsideFov, Ignored, Unrelated, NoIdentity, NotExamined }
    impl Absence { pub fn as_str(self) -> &'static str; pub fn is_finding(self) -> bool }
    pub const REASONS: &[&str] = &[/* each as_str(), in precedence order */];
    #[non_exhaustive] #[derive(Debug, Clone, PartialEq)]
    pub struct LocatedTracking { pub tracking: Tracking, pub absences: BTreeMap<(u64, String), Absence> }
    pub fn locate_tracks(sample: &Sample, class_key: Option<&ClassKey>, do_measure: bool) -> Result<LocatedTracking>;
    impl LocatedTracking {
        pub fn state_at(&self, instance_id: u64, timepoint: &str) -> Option<&'static str>;
        pub fn reason_at(&self, instance_id: u64, timepoint: &str) -> Option<Absence>;
        pub fn is_new(&self, instance_id: u64, located: bool) -> bool;
        pub fn is_resolved(&self, instance_id: u64, located: bool) -> bool;
    }
    ```

  - Mapping uses `resolve_frames` (`sample/reader.rs`); it reads only the
    extent's window of `ignore_region` and of each `valid_mask`.
  - Deprecate `Tracking::{state_at, states, is_new, is_resolved}` with
    `#[deprecated(since = "2.1.0", note = "class-level only: an object outside the
    visit's field of view reads as resolved; use locate_tracks; removed in 3.0")]`,
    and move the CLI and the bindings in the same pull request.
  - Python (`Sample.tracks` → `locate_tracks`; `crates/medh5-python/src/tracking.rs`):
    `Tracking.reason_at(instance_id, timepoint) -> str | None`,
    `Tracking.reasons(instance_id) -> dict[str, str | None]`, `is_new(instance_id,
    *, located=False)`, `is_resolved(instance_id, *, located=False)`;
    `_rebuild(..., reasons=None)` so old pickles load; `coverage` unchanged and
    documented as class-level; `_core.pyi`.
  - JSON gains `tracks[].reasons`; `medh5 track` gains a reason column, and its
    legend (`crates/medh5-cli/src/inspect.rs`) states the rule.
  - D12's §7.4 text and a C.1 row.
- **Tests.** `test_S7_4_an_object_outside_the_follow_up_grid_is_unexamined` (the
  100 mm case: `outside_fov`, `is_resolved` False);
  `test_S7_4_an_object_outside_the_valid_mask_is_unexamined`;
  `test_S7_4_an_ignore_region_away_from_the_object_leaves_a_finding` (`located`)
  and `…_touching_it_is_ignored`;
  `test_S7_4_unregistered_frames_fall_back_to_class_coverage` (`class_examined`,
  or `unrelated` with an ignore region); `test_S7_4_a_semantic_mask_alone_is_no_identity`;
  `test_S7_4_a_displacement_field_does_not_locate`;
  `test_S7_4_the_located_extent_rounds_out_at_voxel_edges` (a box edge at 4.5:
  boxes sit at voxel edges); `test_S7_4_is_new_needs_the_baseline_to_have_imaged_the_place`;
  Rust `s7_4_reason_precedence_prefers_a_finding`. No conformance case (it is a
  query).
- **Docs.** `docs/guides/longitudinal.md` (a table of states and reasons),
  `docs/reference/curation.md`, `docs/reference/cli.md` (`medh5 track`),
  `docs/explanation/design-rationale.md` (why there is no fourth state).
- **Changelog.** Behaviour changes, area `curation`: "**An object outside a
  follow-up's field of view is no longer `resolved` there.**"; Added (reasons);
  Deprecated (the class-level Rust methods).

### 09.2 Agreement across vocabularies, an explicit object match, and an honest `against` (PR 2, D13)

- **State at 2.0.2.** Classes are compared by id, and `check_meanings`
  (`curation/agreement.rs`) refuses (C102) an id whose key differs: two
  vocabularies cannot be compared at all (liver=1 against liver=6 gives C102).
  Objects match by `instance_id` only within one sample, otherwise by greedy
  IoU. Cross-sample results set `against: "annotations/<B's id>"`, which
  `to_record()` writes into A: a self-reference (verified). The schema defines
  `against` as the "HDF5 path of the annotation compared against". An instance
  comparison lists "liver" twice in `skipped`.
- **Change.**
  - `map_classes` = `"id"` (default, today), `"key"`, or an explicit injective
    map. `"key"` needs inline label sets on both sides (a label set by reference
    is refused); codes are cross-checked where both give one (one key with two
    codes is C102); a key B lacks goes in `skipped` with its reason, never scored
    0; an explicit map must be injective (two A classes onto one B class count it
    twice).
  - `match_by` = `"auto"`, `"iou"`, `"instance_id"`; `instance_id` refuses two
    samples (ids are scoped to a sample, §7.4). No id map across samples: the
    format defines no identity across samples.
  - A cross-sample result has `against = None`; `to_record()` refuses it unless
    `against=` is given. `skipped` is deduplicated.
  - Cohort stats across vocabularies keep 2.0.2's refusal.
  - Rust:

    ```rust
    #[non_exhaustive] pub enum ClassMap { Id, Key, Explicit(IndexMap<i64, i64>) }
    #[non_exhaustive] pub enum MatchBy { Auto, Iou, InstanceId }
    #[non_exhaustive] #[derive(Debug, Clone, Default)]
    pub struct CompareOptions { pub metric: Option<String>, pub threshold: Option<f64>,
        pub classes: Option<Vec<ClassKey>>, pub map_classes: ClassMap, pub match_by: MatchBy }
    #[non_exhaustive] pub struct Compared { pub comparison: Comparison, pub pairs: IndexMap<String, (i64, i64)> }
    pub fn compare_with(a: &Annotation, b: &Annotation, options: &CompareOptions) -> Result<Compared>;
    impl VoxelAgreement { pub fn to_record_against(&self, against: &str) -> Result<Agreement> }  // InstanceAgreement too
    ```

    (`ClassMap` and `MatchBy` need `Default`: `Id`, `Auto`.)
  - Python (`medh5/curation/agreement.py`): `compare(a, b, *, metric=None,
    threshold=None, classes=None, map_classes: Literal["id", "key"] |
    Mapping[int | str, int | str] = "id", match_by: Literal["auto", "iou",
    "instance_id"] = "auto")`; `compare_voxel` takes `map_classes`;
    `compare_instances` takes both; results gain `b_class_ids: Mapping[str, int]`;
    `to_record(self, *, against: str | None = None)`.
  - CLI `medh5 agree`: `--map-classes id|key`, `--map-class A=B` (repeatable),
    `--match-by auto|iou|instance-id`.
  - D13's §11.2 text and a C.1 row.
- **Tests.** `test_S11_2_key_mapping_scores_liver_against_liver_across_vocabularies`;
  `test_S11_2_a_key_missing_from_b_is_skipped_not_zero`;
  `test_S11_2_one_key_with_two_codes_is_C102`;
  `test_S11_2_an_explicit_map_must_be_injective`;
  `test_S11_2_a_label_set_by_reference_refuses_key_mapping`;
  `test_S7_4_match_by_instance_id_refuses_two_samples`;
  `test_S7_4_match_by_iou_ignores_shared_ids`;
  `test_S11_2_a_cross_sample_record_needs_against`;
  `test_S11_2_skipped_names_each_class_once`.
- **Docs.** `docs/reference/curation.md`, `docs/guides/longitudinal.md`,
  `docs/reference/cli.md`.
- **Changelog.** Added (mapping, matching), Fixed (`against`, `skipped`), area
  `curation`.
