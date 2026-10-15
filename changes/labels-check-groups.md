<!-- area: cohorts -->
### Added

- **`medh5.labels.check(paths)` reports which files share a vocabulary.** It
  returns what `medh5 labels check --json` prints; Rust has
  `check_vocabularies`, `vocabulary_of` and `vocabulary_key`.

### Behaviour changes

- **`medh5 labels check` groups files by their label set's digest, as
  `dataset check` does.** An empty label set is a vocabulary of its own, told
  apart by its id, where it printed `<none>` alongside the files that declare
  no label set; two such files with different ids now fail the check, as they
  raise `C101`.
