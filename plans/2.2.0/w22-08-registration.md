# W22-08 Registration I/O: ITK HDF5 files, B-splines

| | |
|---|---|
| Branch | `w22/registration` |
| Size | M overall: PR 1 S–M, PR 2 M (a study, then code or a refusal) |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `medh5/io/itk.py`, `from_itk_transform`, `to_itk_transform`, the `test-itk` job (W21-07 PR 4) |
| Owns | `medh5/io/itk.py`, the ITK commands |
| Decisions | E9 (the `bspline` kind is not added) |

## Session prompt

> Implement workstream W22-08 (registration I/O: ITK HDF5 files, B-splines) of
> medh5 2.2.0 in this repository. First read `plans/README.md`,
> `plans/2.2.0/README.md` and `plans/2.2.0/decisions.md` (E9), then
> `plans/2.2.0/w22-08-registration.md` in full. Check that 2.1.0's ITK I/O is on
> `main`. Work from the latest `main` on `w22/registration`; land the two pull
> requests in order. PR 2 begins with a measurement: import a B-spline only if the
> stored field reproduces ITK's transform exactly (within 1e-6 mm, everywhere in
> the transform's domain), and otherwise keep the refusal and write down why.
> Geometry is never invented. Write tests first, run the local gate from
> `CLAUDE.md` and the `test-itk` job's tests, add fragments, open `W22-08: …` pull
> requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 08.1 ITK's HDF5 transform files | S–M |
| 2 | 08.2 B-spline transforms: exact, or refused with the reason | M |

## Items

### 08.1 ITK's HDF5 transform files (PR 1)

- **State at 2.1.0 (planned).** `read_itk_transform` parses `.tfm`/`.txt` and
  ANTs' MATLAB v4 `.mat`; ITK `.h5` transform files are refused. h5py is a test
  dependency only.
- **Change.** Read and write ITK's HDF5 transform layout (`/TransformGroup/<n>/`
  with `TransformType`, `TransformParameters`, `TransformFixedParameters`) for the
  types 2.1.0 supports, through the engine's HDF5 (a small reader in
  `crates/medh5/src/convert/` bound to Python, since medh5 cannot depend on h5py),
  with the same refusals; a composite in an `.h5` file is flattened in reverse as
  for `.tfm`.
- **Tests.** `test_an_itk_h5_affine_matches_its_tfm`; the cross-check against
  SimpleITK in the `test-itk` job for `.h5` files.
- **Changelog.** Added, area `converters`.

### 08.2 B-spline transforms: exact, or refused with the reason (PR 2)

- **State at 2.1.0 (planned).** B-spline files are refused. medh5's displacement
  fields interpolate linearly or by cubic B-spline after prefiltering their
  samples (1.0 §10.4).
- **Study first.** Store an ITK `BSplineTransform` as a displacement field on its
  control-point grid, with the field's samples set to the spline evaluated at the
  control nodes, and compare medh5's cubic evaluation with SimpleITK's
  `TransformPoint` over the transform's whole domain, boundary included (ITK's
  border coefficients and medh5's prefilter boundary condition may differ there).
- **Then.** If the difference is within 1e-6 mm everywhere: import B-splines that
  way (`from_itk_transform` accepts them; the decision note states the
  representation), and export them back. If not: keep the refusal, make its
  message say why ("a B-spline transform cannot be stored exactly as a sampled
  field: …"), and add a paragraph to `docs/explanation/design-rationale.md`
  stating what a `bspline` transform kind would need (a candidate for a later
  minor, E9). Either way, put the measurements in the pull request.
- **Tests.** Either `test_a_bspline_imports_exactly` (the cross-check) or
  `test_a_bspline_is_refused_with_its_reason`.
- **Changelog.** Added, or Documentation, area `converters`.
