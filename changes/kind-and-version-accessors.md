<!-- area: annotations -->
### Deprecated

- **`annotations::header::{ANNOTATION_KINDS, GEOMETRIC_KINDS}` and
  `version::FORMAT_VERSIONS` are deprecated in the Rust crate.** Their lengths
  are part of their types, so a kind or a version added in a later minor
  would break every caller; use `annotations::kinds(version)`,
  `annotations::geometric_kinds(version)` and `version::format_versions()`,
  which answer with slices. Removed in 3.0; Python's `ANNOTATION_KINDS` and
  `GEOMETRIC_KINDS` are unchanged.
