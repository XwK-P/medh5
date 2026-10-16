<!-- area: converters -->
### Added

- **The engine reads and imports NIfTI itself.** `medh5::convert::nifti`
  holds `read_nifti` and `from_nifti`, 2.1.0's converter ported rule for
  rule: NIfTI-1 and NIfTI-2 in either byte order, `.nii`, `.nii.gz` and
  `.hdr`/`.img`, with the same geometry, notes, objects and refusals, which
  a parity corpus recorded under 2.1.0 holds it to. gzip is inflated in Rust,
  never past the bytes the header implies, so a small file cannot make the
  reader decompress or allocate more; zlib's gzip file interface stays
  unlinked. The Python converter and the command line move onto it next.
