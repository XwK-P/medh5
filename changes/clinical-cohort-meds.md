<!-- area: clinical -->
### Added

- **`augment_cohort_from_meds` imports a whole MEDS cohort into its
  samples.** A mapping from `subject_id` to sample says where each subject's
  rows go, in place or into `out_dir`; each subject is mapped exactly as
  `augment_from_meds` maps it alone. A subject that refuses is listed with
  its reason and the others are written; subjects the mapping does not name
  are counted. On the command line, `medh5 clinical augment-cohort
  --from-meds DIR --mapping MAP.csv`, with `clinical augment`'s options.

### Performance

- **A MEDS cohort is read once, not once per sample.** The dataset is opened
  once and read a Parquet row group at a time: its subject column, then the
  mapped subjects' rows. A subject is imported as soon as its last row is
  read, so a dataset in MEDS order holds one row group and one subject in
  memory at a time.
