<!-- area: python-api -->
### Documentation

- **A physical read is float32 only where a rescale applies.** The reference,
  the PyTorch page and the cohort and DICOM guides now say so, and that
  `dtype` gives one type across a cohort; `physical=True` itself is unchanged.
