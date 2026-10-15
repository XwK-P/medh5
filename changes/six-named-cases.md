<!-- area: release -->
### Deprecated

- **`Deidentification.extra` is deprecated, in the Rust crate and in
  Python.** The schema's `deidentification` object allows no member it does
  not define, so anything `extra` holds makes the document invalid. Reading
  it warns; 3.0 removes it.

### Fixed

- **A dataset's filter pipeline reports every client value of a filter.**
  `h5::data::filters` read at most 32, so a filter with more lost the rest,
  and it recorded an entry for a filter it could not read; it now reads them
  all, and a filter it cannot read is an error.
- **An unknown diagnostic code is quoted the same from Rust and Python.**
  `codes::code` quoted it as Rust does (`"X999"`) where `medh5.errors.code`
  quoted it as Python does (`'X999'`); `errors.code` now asks the engine.
