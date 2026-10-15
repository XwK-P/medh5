<!-- area: cohorts -->
### Added

- **Each cohort check code has one severity, in `CHECK_CODE_SEVERITY`.**
  Python's `medh5.dataset.CHECK_CODE_SEVERITY` and Rust's `CHECK_SEVERITY`
  and `code_severity` give every `C` code's severity; `C401` is the one a
  flag changes, an error under `dataset check --deep`.

### Documentation

- **The cohort check codes page shows each code's severity.** It is rendered
  from the engine's table, with `C401`'s promotion under `--deep` beneath it.
