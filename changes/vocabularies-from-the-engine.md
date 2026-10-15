<!-- area: python-api -->
### Added

- **`medh5.geometry.COORD_SYSTEMS`, the coordinate systems the converters
  map.** It is `("LPS", "RAS")`, the engine's tuple, which the NIfTI, DICOM
  SEG, RTSTRUCT and MONAI layers now share instead of each spelling it.
