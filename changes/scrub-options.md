<!-- area: deidentification -->
### Added

- **`ScrubOptions` holds every scrub option, checked once for every door.**
  `ScrubOptions::check` refuses an apply-only option without `--apply`, a
  pseudonym or a range without a salt, and a shift of 0, in one message the
  command line and the Python API share; `apply_with` and `scrub_each` take
  it. `--pseudonymise-ids` is no longer tied to `--apply` by the grammar, so
  its refusal says why.

### Deprecated

- **`scrub_paths` and `STRICT_RULES` (Rust).** `scrub_each` returns every
  path's result, and `strict_rules()` lists the rules a later minor adds;
  both old names are removed in 3.0.

### Fixed

- **A refused file no longer loses the reports of the files scrubbed before
  it in Rust.** `scrub_paths` stopped at the first refusal, and the `uid_map`
  of each file already rewritten — the only key to its pseudonyms — went with
  it; `scrub_each` keeps them.
- **`Deidentification::shift_days` reads a recorded 0 as no shift,** as the
  scrub always had.
