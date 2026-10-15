<!-- area: converters -->
### Behaviour changes

- **A NIfTI, SEG or RTSTRUCT export keeps a symbolic link at its path, and
  the file's mode.** `to_nifti` and `write_nifti` replaced a link at their
  path with a new file and wrote a `0o600` file back as `0o644`; they now
  replace the file the link names and keep its mode, as every file medh5
  writes does (§14.4). An nnU-Net export writes through a link at a case
  file inside the dataset, and refuses one naming a file outside it at any
  channel's file — only the first channel's and the labels' were checked.

### Fixed

- **An interrupted DICOM SEG or RTSTRUCT export leaves the previous file
  whole.** `save_as` wrote over the previous file in place, so a crash or a
  full disk during it left a truncated SEG or structure set. The file is now
  written to a temporary beside it and renamed over it once whole, through
  the engine's replace (§14.4); a failure leaves the previous file
  byte-identical and no temporary behind.
- **An interrupted nnU-Net export leaves the previous `dataset.json` whole.**
  It was written in place; it now goes through the engine's replace, and is
  the last file an export writes, so an export that stops before it keeps
  the old description. Its NIfTI files were already written whole; they now
  go through the same replace.
