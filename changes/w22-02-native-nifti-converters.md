<!-- area: converters -->
### Added

- **The engine writes NIfTI and adds NIfTI masks to a sample.**
  `medh5::convert::nifti` gains `to_nifti`, `write_nifti`, `for_export`,
  `write_field` and `import_seg_nifti`, 2.1.0's converter ported rule for
  rule and held to the same parity corpus as the reader: every one of its 348
  cases gives 2.1.0's objects, `/meta`, report and refusal.

### Behaviour changes

- **NIfTI conversion no longer needs nibabel.** `medh5.io.nifti`, the ITK
  displacement fields of `medh5.io.itk` and the nnU-Net exporter read and
  write NIfTI through the engine, with their signatures unchanged. nibabel is
  neither installed nor imported; the `nifti` extra is kept, empty, so
  `pip install "medh5[nifti]"` still works, and `require_nibabel` says medh5
  no longer needs it rather than naming that extra.
- **A header NIfTI cannot hold is a `ValueError`.** It was nibabel's
  `HeaderDataError`, a `ValueError` subclass, with the same message.
- **A `.nii.bz2` or `.nii.zst` file is refused.** nibabel read them through
  Python's codecs; the engine inflates gzip only, and says to decompress the
  file first.
- **`convert to-nifti` keeps its overlap warning in the report.** Flattening
  overlapping classes into one labelmap was a Python `UserWarning` on stderr
  from the command line; it is now a warning in the report, as the other
  notes are. `to_nifti` in Python still warns, as it did.
