<!-- area: cohorts -->
### Added

- **`ClassStats.examined_prevalence` is `None` for a class no sample
  examined.** Rust's `ClassStats::examined_prevalence` returns
  `Option<f64>`.

### Deprecated

- **`ClassStats.prevalence` answers 0.0 for a class no sample examined until
  3.0.** It raises a `FutureWarning` there, and only there: medh5 3.0 returns
  `None`. Rust's `ClassStats::prevalence` is deprecated for
  `examined_prevalence`.

### Behaviour changes

- **The prevalence of a class no sample examined is `null` in the statistics
  JSON.** A saved 0.0 recorded a measurement that was never made: a class drawn
  where nobody looked for it read as "looked for everywhere, found nowhere".
  `medh5 dataset stats` prints `-` for it.
