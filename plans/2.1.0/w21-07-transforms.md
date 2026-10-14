# W21-07 Transforms and registration I/O

| | |
|---|---|
| Branch | `w21/transforms` |
| Size | L overall: PR 1 M, PR 2 S, PR 3 S–M, PR 4 M–L |
| Lands after | W21-00 PR 4, PR 5; PR 4 after W21-05 PR 1 (`_core.write_atomic`) |
| Blocks | nothing |
| Owns | `crates/medh5/src/transforms/**`, `crates/medh5/src/sample/reader.rs` (`transform_between`, `resolve_frames`), `medh5/transforms.py`, `medh5/sample.py` (`transform_between`), `medh5/io/itk.py` (new), the `test-itk` CI job |
| Decisions | D28 |
| Backport | none |

## Session prompt

> Implement workstream W21-07 (transforms and registration I/O) of medh5 2.1.0
> in this repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D28), then `plans/2.1.0/w21-07-transforms.md` in
> full. Work from the latest `main` on `w21/transforms`; land the four pull
> requests in order. Geometry is never invented: where a frame, a vector
> convention or an order cannot be known from the input, the API takes it as a
> required argument or refuses. PR 4 adds a CI job (SimpleITK): it changes
> `tests/project/test_packaging.py` too. Write tests first, run the local gate
> from `CLAUDE.md`, add fragments, open `W21-07: …` pull requests, and do not
> merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 07.1 `transform_between` refuses a visit of several frames | M |
| 2 | 07.2 The inverse of a chain or composite; 07.3 a composite reuses its components' cubic coefficients | S |
| 3 | 07.4 A region's Jacobian is the field's own | S–M |
| 4 | 07.5 ITK affine and displacement-field import and export | M–L |

## Items

### 07.1 `transform_between` refuses a visit of several frames (PR 1, D28)

- **State at 2.0.2.** `Sample::transform_between` (`sample/reader.rs`) expands
  each key through `frames_for`, trying it as a timepoint, then a grid, then a
  frame; it loops over frame pairs in grid-id order, returns `None` at the first
  pair sharing a frame, and otherwise the first pair that resolves. On one
  sample, timepoints with grids `apet0`/`apet1` gave `None`, grids named
  `pet0`/`pet1` gave `AffineTransform('A_to_C')`: only the names differed.
  `medh5/torch/datasets.py` avoids the timepoint form. 1.0 §3.4: a reader MUST
  refuse rather than assume alignment when no transform resolves. 2.0.2's
  `test_transform_between_over_visits_is_described_as_it_behaves`
  (`tests/project/test_docs_examples.py`) pins the grid-id order.
- **Change.** A timepoint key has an answer only when that visit's framed grids
  share one frame. When either visit holds several, raise an uncoded
  `MEDH5ValidationError` naming each visit's frames and grids and pointing to
  `by="grid"` or `resolve_frames`. With one frame a side, answers stay: `None`
  for a shared frame, the transform when a path exists, `None` when none, E501
  for a tie. Rust `transform_between_as(&self, source: &str, target: &str, by:
  Option<KeyKind>) -> Result<Option<Transform>>` with `#[non_exhaustive] pub enum
  KeyKind { Timepoint, Grid, Frame }`; `transform_between` delegates. Python
  `Sample.transform_between(source, target, *, by=None)` (`medh5/sample.py`,
  `_core.pyi`).
- **Tests.** `test_S10_a_visit_of_several_frames_has_no_one_transform` (both grid
  namings); `test_S10_one_frame_each_answers_as_before`;
  `test_S10_by_grid_reads_a_grid_named_like_a_timepoint`;
  `test_S3_4_a_visit_without_framed_grids_is_unrelated`; rewrite
  `test_transform_between_over_visits_is_described_as_it_behaves` to pin the
  refusal.
- **Docs.** `docs/guides/longitudinal.md`, the "When there is no path" section of
  `docs/guides/registration.md`, `docs/reference/python-api.md`.
- **Changelog.** Behaviour changes, area `geometry`: "**`transform_between` over
  two visits refuses when a visit holds several frames.** It answered from
  whichever grids sorted first…"

### 07.2 The inverse of a chain or composite (PR 2)

- **State at 2.0.2.** `Transform::is_invertible` (`transforms/model.rs`) reports
  a *claim* (declared, analytic, or true for a chain whose every step is
  invertible); `can_invert` reports whether the inverse can be *evaluated*. A
  `ChainTransform([a01, a12])` and a composite of two affines are
  `is_invertible` True, `can_invert` False, and `inverse()` raises.
  `docs/guides/registration.md` documents the distinction.
- **Change.** `inverse()` of a `Chain`, or of a stored composite with no usable
  `inverse_id`, is the `Chain` of its steps' inverses in reverse order, when
  every step can be inverted; `can_invert` reports true for these. `is_invertible`
  stays the claim, and its docstring says so. The frame graph does not change:
  `edges` (`transforms/resolve.rs`) keeps 2.0.2's rule through a private
  `has_reverse_edge` (new reverse edges would create E501 ties, e.g. F2→F0 between
  `a12⁻¹·a01⁻¹` and `d21·a01⁻¹`).
- **Tests.** `test_S10_5_the_inverse_of_a_chain_is_its_steps_inverted_in_reverse`
  (points round-trip to 1e-9); `test_S10_5_a_composite_of_affines_inverts_without_an_inverse_id`;
  `test_S10_5_a_step_that_cannot_be_inverted_still_refuses`;
  `test_S10_1_resolution_is_unchanged_by_invertible_composites`.
- **Docs.** `docs/guides/registration.md`, `docs/reference/python-api.md`.
- **Changelog.** Added, area `geometry` (a refusal becomes an answer).

### 07.3 A composite reuses its components' cubic coefficients (PR 2)

- **State at 2.0.2.** A `Transform` keeps cubic coefficients in `cubic:
  Arc<CubicCache>`, shared by clones, within a 256 MiB process-wide budget
  (`CUBIC_CACHE_BYTES`, `transforms/apply.rs`). But `components()`
  (`model.rs`) calls `Transform::open` on every evaluation, so a cubic field
  inside a composite is prefiltered over the whole field on every
  `transform_points` call: a paired loader evaluating a composite per item pays a
  full prefilter per item.
- **Change.** `open_transforms` creates a private `Arc<KeptCaches>` (a
  `Mutex<HashMap<String, Arc<CubicCache>>>` keyed by transform id) and gives it to
  every transform it opens; `components()` and `inverse()` open through it
  (`pub(crate) fn open_kept`). The public `Transform::open` keeps its signature
  and a fresh cache. The private field is semver-safe (`Transform` already has
  private fields). The budget is unchanged.
- **Tests.** Rust `s10_5_a_composite_shares_its_components_coefficients`
  (`Arc::ptr_eq`, not a global counter, which would be flaky in parallel tests);
  Python `test_S10_5_a_composite_prefilters_a_cubic_component_once`
  (`_core._read_stats()` read counts flat over the second evaluation).
- **Docs.** One line in `docs/guides/performance.md`.
- **Changelog.** Performance, area `geometry`.

### 07.4 A region's Jacobian is the field's own (PR 3)

- **State at 2.0.2.** `Transform::jacobian_determinant(roi)` (`model.rs`) reads
  the ROI window and differentiates it as if it were the whole field; `gradient`
  (`transforms/apply.rs`) is one-sided at the *window's* edges. Refused: a step ≠
  1 (`roi_bounds`), an empty axis, an axis under 2 voxels; `folding_fraction(roi)`
  inherits all three. Verified: tiles [0:4], [4:8], [8:12] report 16.7 % folding
  where the whole field has 8.3 %, and ROI [0:6] gives det 1.0 at x=5 where the
  field gives 0.2. (The audit's idea, differentiating on the sampled lattice,
  measures a different quantity and misses a fold narrower than the step,
  verified: a step of 2 missed the planted fold. Do not do that.)
- **Change.** `jacobian_determinant(roi) == jacobian_determinant()[roi]` for every
  ROI, strides and 1-voxel ROI axes included (the field still needs at least 2
  voxels on each axis). `model.rs`: `roi_selection(grid, roi, owner) ->
  Result<Vec<(usize, usize, usize)>>` (start, stop, step, via `Slice::resolve`);
  per axis read `[max(start−1, 0), min(last+2, n))`, slabbed along axis 0 when the
  window exceeds the read budget; differentiate with `apply::jacobian_determinant`
  on the window's grid; select `(start−w0)..(stop−w0)` by the step. Differences
  are one-sided only at the field's own edges. `folding_fraction(roi)` is the
  fraction over the selected voxels. Signatures unchanged; rewrite the
  docstrings (`medh5/transforms.py`, `model.rs`). No spec text: 1.0 defines no
  Jacobian.
- **The fold kit** (`tests/kits.py`, `FoldKit`): a `displacement` field on a
  12×2×2 grid at 1 mm, identity direction, zero except u_x = −3 mm at x = 6.
  Under central differences the voxels at x = 5 fold (det −0.5), so the whole
  field's `folding_fraction()` is 1/12; at 2.0.2 the ROIs [0:6] and [6:12] see
  no fold at all (each differentiates its own edge one-sidedly).
- **Tests.** `test_S10_the_jacobian_of_a_roi_is_the_fields_own` (a property test
  over random contiguous and strided ROIs); `test_S10_tiled_rois_fold_as_the_whole_field`
  (the fold kit, tiled [0:4]/[4:8]/[8:12] and [0:6]/[6:12]: the tiles'
  determinants, concatenated, equal the whole field's, and their fold counts
  sum to its one fold); `test_S10_a_strided_roi_keeps_a_one_voxel_fold`;
  `test_S10_a_one_voxel_roi_axis_is_accepted`;
  `test_S10_a_one_voxel_field_axis_is_still_refused`; update the existing ROI
  tests in `tests/format/test_transforms.py`; Rust
  `s10_the_roi_window_reads_one_voxel_beyond`.
- **Docs.** `docs/guides/registration.md`, `docs/reference/python-api.md`.
- **Changelog.** Behaviour changes, area `geometry`: edge values of contiguous
  ROIs change (they depended on where the ROI was cut); strided ROIs are accepted.

### 07.5 ITK affine and displacement-field import and export (PR 4)

- **State at 2.0.2.** `add_transform` writes `affine` (§10.3), `displacement`
  (§10.4, with `field_grid` in `from_frame`) and `composite`, whose components
  apply left to right (§10.5). §10.2 is ITK's convention, `x_M = T(x_F)`. Nothing
  reads or writes ITK files (no entry in `medh5/io/__init__.py` or `medh5/cli.py`);
  `docs/guides/registration.md` writes matrices by hand. An intermediate-frame
  chain (a displacement F→mid plus an affine mid→M) resolves as a
  `ChainTransform` and validates clean apart from W903, but does not earn `reg`
  (1.0 §1.3); storing a `composite` between the two grid frames does. ITK applies
  the last listed transform first; §10.5 applies the first listed first. Whether
  a given ITK version writes displacement vectors as LPS or RAS is unverified.
- **Change.** `medh5/io/itk.py`, loaded lazily, no new dependency (fields need
  nibabel, the `nifti` extra):

  ```python
  @dataclass(frozen=True)
  class ItkTransform:  # LPS mm, x_moving = T(x_fixed)
      name: str
      matrix: npt.NDArray[np.float64]
      components: tuple[ItkTransform, ...] = ()

  def read_itk_transform(path: str | os.PathLike[str]) -> ItkTransform: ...
  def write_itk_transform(path: str | os.PathLike[str], transform: ItkTransform) -> None: ...  # .tfm or .mat by suffix
  def from_itk_transform(path, sample, *, transform_id: str, fixed: str, moving: str,
                         warp=None, vectors: Literal["lps", "ras"] | None = None,
                         inverse_warp=None, report: ConversionReport | None = None) -> ConversionReport: ...
  def to_itk_transform(sample, out, *, transform_id: str, warp_out=None,
                       vectors: Literal["lps", "ras"] | None = None) -> ConversionReport: ...
  ```

  - Parse `.tfm`/`.txt` with the standard library and ANTs' MATLAB v4 `.mat` with
    `struct` (a variable named after the type, plus `fixed`). Types: Affine,
    MatrixOffsetTransformBase, Translation, Identity, Euler3D/2D (honouring the
    ZYX flag), VersorRigid3D, Similarity3D/2D, Scale, and a Composite of these.
    Refused: `px` and custom coordinate systems, mismatched dimensions, ITK
    `.h5` and B-spline files (2.2.0).
  - Frames: the caller names `fixed` and `moving` grids, in ITK's words; the
    importer stores `from_frame` = the fixed grid's frame.
  - Composites are flattened in reverse (ITK order to §10.5 order).
  - Affine conversion: `[[A, t+c−Ac], [0, 1]]`, then `F·M·F` with `F = diag(−1, −1,
    1, 1)` for an RAS grid, and the units scale from `MM_PER_UNIT`.
  - A field: `vectors` is **required** (the convention depends on the writing
    tool and its ITK version). `read_nifti` turns `(X,Y,Z,1,3)` into `(3,Z,Y,X)`;
    components are flipped and scaled; the field reuses `fixed` as its grid when
    the geometry matches, otherwise a new grid `{id}_field` in the fixed grid's
    frame; stored with `vector_space="world"`.
  - An ANTs affine plus warp is stored as the warp F→`pseudo:<id>.mid`, the
    affine mid→M, and a `composite` F→M (which earns `reg` and is the unique
    shortest route).
  - Provenance recorded as `import_seg_nifti` does. Export writes
    `AffineTransform_double_3_3` and fields as DISPVECT (intent 1006), through
    `_core.write_atomic` / `stage_file`.
  - CLI: `medh5 convert from-itk-transform TRANSFORM SAMPLE --id ID --fixed GRID
    --moving GRID [--warp FILE --vectors lps|ras] [--inverse-warp FILE] [--report
    FILE] [--json]`; `medh5 convert to-itk-transform SAMPLE OUT --id ID
    [--warp-out FILE --vectors lps|ras]`.
  - CI: a `test-itk` job installing SimpleITK and running `-k "Itk or itk"`
    (gating; add it to `ci-ok` and `test_packaging`'s job lists).
- **Tests.** `tests/integrations/test_itk.py` (main job), golden files in
  `tests/data/itk/`: `test_S10_2_an_itk_affine_maps_fixed_to_moving`,
  `test_S10_5_a_composite_is_flattened_last_first`,
  `test_S10_3_the_centre_folds_into_the_translation`,
  `test_S3_1_an_ras_grid_flips_two_axes`,
  `test_S10_4_vectors_are_required_for_a_field`, `test_an_ants_pair_earns_reg`,
  `test_round_trip_tfm_and_mat`. `TestItkCrossCheck` in the `test-itk` job:
  agrees with SimpleITK's `TransformPoint` within 1e-6 mm, and pins the vector
  convention SimpleITK writes.
- **Docs.** `docs/guides/registration.md`, `docs/reference/converters.md`,
  `docs/reference/cli.md`.
- **Changelog.** Added, area `converters`.
