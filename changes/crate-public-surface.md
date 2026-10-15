<!-- area: release -->
### Added

- **`medh5::AttrValue` is re-exported at the crate root.** The documented
  `extra` and `attrs` fields of grids, transform and annotation headers and
  payloads hold it, and `Image::attr` returns it; it was reachable only as
  `medh5::h5::attrs::AttrValue`, which stays.

### Documentation

- **The Rust crate states its public surface.** hdf5-metno 0.15, ndarray 0.17
  and IndexMap 2 are public dependencies, so a bump of any of them is a major
  medh5 release. `h5`, `json`, `numeric` and `pyval` are plumbing outside
  semver from 2.1; `rng` and `error` join the module table, which the crate's
  README and the reference page now keep equal.
