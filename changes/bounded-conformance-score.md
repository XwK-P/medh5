<!-- area: validation -->
### Added

- **Each conformance case states its format version, and a score can be
  bounded by version and by code revision.** `expected.json` gives each case
  a `format_version`, the latest version among the codes it expects, which is
  not always what its file declares. It also gives the suite's
  `codes_revision`. `medh5 conformance score --format-version 1.0` scores a
  1.0-only implementation on the cases addressed to 1.0.
  `--codes-revision 1` scores one written against the first revision of the
  code table without the codes later revisions allocated. Either way the
  cases left out are counted as skipped. The same bounds are available as
  `medh5.conformance.score(..., format_version=, codes_revision=)` and Rust's
  `score_with` with `ScoreOptions`. Unbounded, a third party's score fell
  with every case added by an erratum it was never told about.
