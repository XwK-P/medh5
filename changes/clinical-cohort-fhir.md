<!-- area: clinical -->
### Added

- **`augment_cohort_from_fhir` imports a FHIR Bulk Data export into a
  cohort's samples.** The export's NDJSON files are streamed and grouped by
  subject reference, one subject in memory at a time (where each line is, is
  indexed in a temporary file), and each subject is mapped exactly as
  `read_fhir` maps a Bundle of its Observations and DiagnosticReports.
  Resources of unmapped subjects, and resource types the profile does not
  read, are counted. On the command line, `medh5 clinical augment-cohort
  --from-fhir-bulk DIR --mapping MAP.csv`.
