<!-- area: converters -->
### Added

- **ITK and ANTs transforms import and export.** `medh5.io.itk` reads and
  writes ITK `.tfm`/`.txt` and ANTs `.mat` affines (`read_itk_transform`,
  `write_itk_transform`), and `from_itk_transform` / `to_itk_transform`
  store one between two grids' frames, re-expressed from ITK's LPS
  millimetres in each grid's coordinate system and units, with an ANTs
  displacement field (`warp`, and an optional `inverse_warp`) as a
  `composite` through an intermediate frame. `vectors="lps"` or `"ras"` is
  required with a field, whose file does not say. The command line has
  `medh5 convert from-itk-transform` and `to-itk-transform`. B-spline
  transforms and `.h5` transform files are refused.
