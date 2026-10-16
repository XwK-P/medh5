<!-- area: converters -->
### Added

- **Enhanced multi-frame DICOM imports.** Enhanced and legacy-converted CT,
  MR and PET objects, skipped with a warning until now, are split into stacks
  — by `StackID` and the dimensions of `DimensionIndexSequence` that tell
  stacks apart — and each stack imports as the classic series it would have
  been: the same grid and the same values, placed by the frames' functional
  groups and decoded one frame at a time. Per-frame rescales follow
  `per_slice_rescale`, acquisition attributes come from the shared and
  per-frame groups, and a concatenation is joined into one object. A frame
  with no position, an irregular stack, a stack whose
  `InStackPositionNumber` its positions contradict or that skips a number,
  and a concatenation missing a part are refused with the reason. `Series`
  gains `frames` and `stack`.

### Fixed

- **A CT's units follow its `RescaleType`.** Every CT was written as
  quantitative HU, though DICOM lets a derived or multi-energy CT rescale to
  another quantity. `value_units` is now the stated `RescaleType` (`HU` when
  none is stated), `US` gives `intensity` with no units, and a series whose
  slices state different types is refused.
