<!-- area: converters -->
### Added

- **Enhanced multi-frame DICOM imports.** Enhanced and legacy-converted CT,
  MR and PET objects, skipped with a warning until now, are split into stacks
  — by `StackID` and, where positions repeat, the dimensions of
  `DimensionIndexSequence` that tell stacks apart — and each stack imports
  as the classic series it would have been: the same grid and the same
  values, placed by the frames' functional groups and decoded one frame at a
  time. Per-frame rescales follow
  `per_slice_rescale`, acquisition attributes come from the shared and
  per-frame groups, and a concatenation is joined into one object. A frame
  with no position, an irregular stack, a stack whose
  `InStackPositionNumber` its positions contradict or that skips a number,
  a concatenation missing a part, a part with no `ConcatenationUID`, and an
  instance whose `NumberOfFrames` its per-frame groups contradict are
  refused with the reason. `Series` gains `frames` and `stack`.

### Fixed

- **A CT's units follow its `RescaleType`.** Every CT was written as
  quantitative HU, though DICOM lets a derived or multi-energy CT rescale to
  another quantity. `value_units` is now the stated `RescaleType`. An omitted
  type is `HU` in a classic CT and in an original, non-localizer,
  single-energy enhanced frame, and unknown elsewhere. `US` or an unknown type
  gives `intensity` with no units, and a series whose slices state different
  types is refused.
- **Acquisition attributes are recorded only where the slices agree.** A
  series recorded its first slice's `XRayTubeCurrent`, `ExposureTime` or
  `EchoTime` for the whole volume, even where the slices stated different
  ones. An attribute the slices disagree on is now left out, with a warning
  in the report.
