<!-- area: geometry -->
### Behaviour changes

- **`transform_between` over two visits refuses when a visit holds several
  frames.** It answered from whichever grids sorted first, so renaming a grid
  turned "no registration" into a CT's registration for a PET pair. It now
  raises `MEDH5ValidationError` naming each visit's grids and frames; with one
  frame a visit, answers are unchanged. The new keyword `by="timepoint"`,
  `"grid"` or `"frame"` (Rust `transform_between_as` and `KeyKind`) says how
  the keys are read, so a grid named like a visit can be asked about.
