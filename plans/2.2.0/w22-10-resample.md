# W22-10 Resampling as a recorded derivation

| | |
|---|---|
| Branch | `w22/resample` |
| Size | L overall: PR 1 M–L, PR 2 M |
| Lands after | v2.1.0 released; PR 2 after W22-02 PR 4 |
| Builds on (2.1.0) | `transform_between` and its visit refusal (D28), the inverse of a chain (W21-07), the SciPy-exact cubic sampler (`transforms/apply.rs`), every write through one replace routine (D23), the agreement and torch refusals that name a grid mismatch |
| Owns | `crates/medh5/src/transforms/resample.rs` (new), its bindings, `medh5.transforms.resample`, the `medh5 resample` command, `docs/guides/resample.md` (new); touches the refusal messages in `medh5/io/`, `medh5/torch/datasets.py`, `medh5/_monai_readers.py` and `crates/medh5/src/curation/agreement.rs` |
| Decisions | E11 |

## Session prompt

> Implement workstream W22-10 (resampling as a recorded derivation) of medh5
> 2.2.0 in this repository. First read `plans/README.md`,
> `plans/2.2.0/README.md` and `plans/2.2.0/decisions.md` (E11), then
> `plans/2.2.0/w22-10-resample.md` in full, then 1.0 §3, §4.4, §10.2 and §11.1.
> Work from the latest `main` on `w22/resample`; land the two pull requests in
> order. Geometry is never invented: without a transform path the command
> refuses, and a box or contour that a transform would bend is refused, never
> approximated. Converters and loaders keep refusing to resample; only their
> messages change, to name the command. Nothing here is specification text, a
> new code or a format change; if the work seems to need one, stop and ask.
> Write tests first, run the local gate from `CLAUDE.md`, add fragments, open
> `W22-10: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 10.1 Images onto a grid, recorded; 10.2 the command and the Python function | M–L |
| 2 | 10.3 Annotations follow their image; 10.4 refusals name the command; 10.5 the guide | M |

## Items

### 10.1 Images onto a grid, recorded (PR 1, E11)

- **State at 2.1.0.** Nothing resamples. The converters refuse disagreeing grids
  (`medh5/io/nifti.py`, "resample before converting"), the torch loader refuses
  mixed grids (`medh5/torch/datasets.py`, `timepoint=` or "resample the volumes
  onto a common grid first"), and agreement refuses annotations on different
  grids (`curation/agreement.rs`). The engine samples displacement fields
  linearly and cubically (`transforms/apply.rs`: `linear_sample`,
  `cubic_sample`), and `Sample::transform_between` resolves a path between
  grids. The provenance vocabulary already holds `resample`
  (`curation/provenance.rs`).
- **Change.** `transforms::resample(source, plan) -> Result<Resampled>` in the
  engine:
  - **The target** is either an existing grid of the sample (`onto = grid id`),
    or a new grid in the source grid's frame and timepoint with a requested
    `spacing`. A new grid keeps the source's direction and covers its extent:
    the origin is the source's first voxel centre, and each axis gets
    `ceil(extent / spacing)` voxels.
  - **The mapping** goes from each target voxel centre to world, through
    `transform_between(target grid, source grid)` when the frames differ
    (§10.2's convention: evaluate T at the fixed grid's points and sample the
    moving image there), then to source index. There is no path: refused, with
    the two frames named.
  - **Interpolation** is `linear` (the default), `nearest` or `cubic`, using
    the existing samplers generalised from fields to image channels. Only
    spatial axes are resampled, channel by channel, and only level 0 of a
    multiscale image (the result is not multiscale). Values are stored values,
    so `rescale_slope`, `rescale_intercept`, `value_units` and `value_type`
    carry over. `nearest` keeps the dtype. `linear` and `cubic` write `float32`
    unless `dtype=` is given, rounding half to even and clipping to the dtype's
    range.
  - **Outside the source:** a target voxel whose source point lies outside the
    source extent (§10.4's half-voxel margin) takes `fill` (0 unless given).
    When any voxel does, a `mask` annotation `<image>_valid` on the target grid
    marks the voxels that did not, and the image's `valid_mask` names it (§4.4).
  - **Recorded:** one `resample` activity in `/meta → provenance` with `inputs`
    (the source objects), `outputs`, and `params` (`onto` or `spacing`, the
    transform ids on the path, `interpolation`, `fill`, `dtype`). Every object
    written carries `prov` naming it (§11.1).
  - Written through `amend` (copy-on-write, D23), at the lowest version its
    content needs. The source objects are untouched.
- **Tests.**
  - `test_S10_2_resample_through_an_affine_matches_scipy`: `map_coordinates`
    with orders 0, 1 and 3 on the composed index mapping.
  - `test_S10_2_a_resample_without_a_transform_path_is_refused`.
  - `test_S4_4_voxels_outside_the_source_are_filled_and_masked`.
  - `test_S11_1_a_resample_is_one_activity_with_every_output`.
  - `test_resampling_onto_the_source_grid_is_the_identity`: same bytes, for
    `nearest`, and to 1e-6 for `linear` and `cubic`.
  - `test_integer_output_rounds_half_to_even_and_clips`.
  - A Rust test that a large resample is computed in slabs, bounded by the
    target's chunk shape rather than its volume.
- **Changelog.** Added, area `geometry`.

### 10.2 The command and the Python function (PR 1)

- **Change.**
  - `medh5.transforms.resample(path, image_ids, *, onto=None, spacing=None,
    interpolation="linear", fill=0, dtype=None, out=None, annotations=True) ->
    ResampleReport`. Exactly one of `onto` and `spacing` is given. `out=None`
    amends `path` in place; otherwise the result is a new file, and an existing
    `out` is refused, as the converters refuse one.
  - `medh5 resample PATH --image ID [--image ID …] (--onto GRID | --spacing
    SX,SY[,SZ]) [--interpolation linear|nearest|cubic] [--fill V] [--dtype T]
    [-o OUT] [--no-annotations] [--json]`, in the native CLI with no Python
    needed.
  - `ResampleReport` lists the objects written, the activity id, and the count
    of filled voxels.
- **Tests.** The command and the function agree, by `content_id`. A usage
  error exits `EXIT_USAGE` and a refusal `EXIT_ERROR`, with the message. Every
  flag is documented (`test_docs_examples.py`).
- **API.** Additive: a new function, a result type, and its stub in
  `_core.pyi`, `python_api.txt` and `api-index.md`.

### 10.3 Annotations follow their image (PR 2, E11)

- **Change.** With `annotations=True` (the default), every annotation on a
  resampled image's grid is carried to the target:
  - **`mask`, in every encoding:** `nearest`, re-encoded by §7.6's selection
    rule, with `class_ids` and `annotated_class_ids` unchanged (coverage is
    never collapsed) and `derived_from = [source id]`. An `ignore_mask` or
    `valid_mask` reference is resampled with it.
  - **`probmap`:** `linear`, clipped to [0, 1].
  - **World-space geometric annotations** in the same frame need nothing: they
    are not rewritten. Under a transform, points, keypoints and meshes are
    mapped exactly. A box is carried only when the transform keeps axes
    axis-aligned (a translation with a positive diagonal linear part); a
    contour only when it keeps the contour's plane a grid plane. Anything else
    is refused, naming the annotation, or skipped with
    `--no-annotations` and listed in the report as not carried.
  - **Index-space geometric annotations** are re-expressed through world
    coordinates under the same rules. Boxes sit at voxel edges, indices at
    voxel centres (§8.1).
- **Tests.**
  - `test_S7_a_resampled_mask_keeps_its_coverage`.
  - `test_S8_1_a_box_survives_a_spacing_change_at_its_edges`: the half-voxel
    shift.
  - `test_S8_a_rotated_box_is_refused_not_approximated`.
  - `test_S6_2_a_resampled_annotation_names_its_source`.
  - Agreement between the source mask and the resampled one, after resampling
    back, as a sanity check rather than an equality.
- **Changelog.** Added, area `geometry`.

### 10.4 Refusals name the command (PR 2)

- **Change.** The messages that tell a user to "resample first" now give the
  command to run. These are the NIfTI and DICOM converters' grid refusals,
  `import_seg_nifti`, the nnU-Net converter, the torch loader's mixed-grid
  refusal, the MONAI reader's channel refusal, and agreement's different-grids
  refusal. Behaviour does not change: every one still refuses.
- **Tests.** Each refusal's test asserts the command appears in its message.
  `STALE_CLAIMS` gains nothing, because no page said medh5 resamples.
- **Changelog.** A sentence in 10.1's Added entry: the refusals now name the
  command.

### 10.5 The guide (PR 2)

- **Change.** A new `docs/guides/resample.md` covers four things:
  - when to resample, and when not to (the model's input grid, not the
    archive's);
  - onto a grid compared with onto a spacing;
  - what the activity records;
  - why the loaders do not resample.

  `docs/explanation/refusals.md`'s NIfTI section links to it. MONAI users get
  a paragraph on when MONAI's `Spacingd` is enough, which is when nothing needs
  to be recorded.
- **Changelog.** Documentation (the guide).

## Acceptance

- `medh5 resample` and `medh5.transforms.resample` write the same file, which
  validates at `integrity` with no new code, and whose `medh5 prov` output
  shows one `resample` activity.
- No converter, loader or sampler resamples, and each refusal names the
  command.
