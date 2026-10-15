<!-- area: torch -->
### Added

- **`ClinicalTaskDataset(class_ignore=True)` gives each class its own ignore
  mask.** `class_ignore[slot]`, `(K, *patch)`, holds per class the padding and
  the ignore regions of the annotations that label or examined that class
  only (1.0 §7.7), so one rater's uncertain band no longer masks another
  rater's class from a per-class loss. `ignore[slot]` is unchanged, and is the
  union of the channels.
