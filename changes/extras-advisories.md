<!-- area: release -->
### Documentation

- **An advisory against what an extra installs no longer holds a release.**
  medh5 neither ships nor pins PyTorch, MONAI, pydicom or the other extras'
  projects, so the release reports their advisories in its run's summary,
  for `SECURITY.md` to disposition; one against a compiled-in crate or NumPy
  still stops the release.
