# W22-09 Clinical import across a cohort

| | |
|---|---|
| Branch | `w22/clinical-cohorts` |
| Size | M overall; each pull request M |
| Lands after | W22-05 PR 1 |
| Builds on (2.1.0) | `read_meds`, `read_fhir`, `augment_from_meds`, `augment_from_fhir`, `clinical augment --from-meds/--from-fhir`, the `meds` extra and job (W21-12) |
| Owns | `medh5/io/{meds.py,fhir.py}`, the `clinical augment-cohort` command |

## Session prompt

> Implement workstream W22-09 (clinical import across a cohort) of medh5 2.2.0 in
> this repository. First read `plans/README.md` and `plans/2.2.0/README.md`, then
> `plans/2.2.0/w22-09-clinical-cohorts.md` in full. Check that 2.1.0's MEDS and
> FHIR importers (W21-12) are on `main`. Work from the latest `main` on
> `w22/clinical-cohorts`; land the two pull requests in order. Each subject's
> result must be byte-identical to importing that subject alone with 2.1.0's
> per-sample importer: the batch path changes the cost, never the answer. Memory
> stays bounded by one subject's rows. Write tests first, run the local gate from
> `CLAUDE.md` and the `meds` job's tests, add fragments, open `W22-09: …` pull
> requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 09.1 A MEDS dataset into a cohort, in one pass | M |
| 2 | 09.2 A FHIR bulk export into a cohort | M |

## Items

### 09.1 A MEDS dataset into a cohort, in one pass (PR 1)

- **State at 2.1.0 (planned).** `read_meds` scans the dataset once per sample, so
  a cohort of N subjects costs N scans.
- **Change.** `augment_cohort_from_meds(dataset, mapping: Mapping[int, str |
  os.PathLike[str]], *, out_dir=None, **options) -> CohortImport` (`mapping` from
  MEDS `subject_id` to the sample it augments): one pass over the dataset's
  shards (pyarrow's dataset API, filtering to the mapped subjects, grouped by
  subject), each subject's rows mapped exactly as `read_meds` maps them and
  written through `augment`; unmapped subjects counted in an `info` note; a
  subject that refuses is reported and the others still written (as `scrub_each`
  does). CLI `medh5 clinical augment-cohort --from-meds DIR --mapping MAP.csv
  [--out-dir DIR] [--json]` (the same options as `clinical augment`).
- **Tests.** `test_a_cohort_import_equals_each_subjects_import`;
  `test_a_cohort_import_reads_the_dataset_once`;
  `test_a_refused_subject_leaves_the_others`;
  `test_memory_is_bounded_by_one_subject` (a large synthetic dataset, peak RSS
  checked on POSIX).
- **Changelog.** Added and Performance, area `clinical`.

### 09.2 A FHIR bulk export into a cohort (PR 2)

- **Change.** `augment_cohort_from_fhir(export_dir, mapping: Mapping[str, str |
  os.PathLike[str]], *, out_dir=None, **options)`: FHIR Bulk Data NDJSON (one file
  per resource type, many patients), streamed and grouped by `subject` reference,
  each subject mapped exactly as `read_fhir` maps a Bundle. CLI `--from-fhir-bulk
  DIR` on `clinical augment-cohort`.
- **Tests.** `test_a_bulk_export_import_equals_each_subjects_bundle_import`;
  `test_a_resource_for_an_unmapped_subject_is_counted`.
- **Docs.** `docs/guides/clinical-import.md` (cohorts).
- **Changelog.** Added, area `clinical`.
