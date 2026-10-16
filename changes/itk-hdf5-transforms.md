<!-- area: converters -->
### Added

- **ITK's HDF5 transform files are read and written.** `read_itk_transform`,
  `from_itk_transform` and `medh5 convert from-itk-transform` take an `.h5`
  (`.hdf5`, `.hdf`) file as ITK writes one, a composite flattened in reverse
  as for `.tfm`, and `to_itk_transform` and `write_itk_transform` write one;
  through the engine's HDF5, so the converters still need no `h5py`. They
  were refused before. Parameters under the misspelling earlier ITK
  releases wrote (`TranformParameters`) are read, as ITK reads them.
