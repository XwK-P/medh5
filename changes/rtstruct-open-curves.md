<!-- area: converters -->
### Added

- **RTSTRUCT open curves import and export as polylines.**
  `from_rtstruct(open_curves="polylines")` (`--open-curves polylines`)
  imports `OPEN_PLANAR` and `OPEN_NONPLANAR` ROIs as a polylines annotation,
  `<id>_curves` unless `curves_id=` (`--curves-id`) names another, each
  `OPEN_PLANAR` curve with the grid plane it lies in; the report records that
  the sample is now written at format 1.2. `to_rtstruct(curves=...)`
  (`--curves ANN`) writes one back as `OPEN_PLANAR` or `OPEN_NONPLANAR`
  contours. `omit` stays the default, and its warning now names the option.
