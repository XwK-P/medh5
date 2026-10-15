<!-- area: converters -->
### Added

- **`from_dicom(per_slice_rescale="apply")` imports a PET series with a
  rescale per slice.** An image stores one modality LUT, so a series whose
  slices state different `RescaleSlope`/`RescaleIntercept` is refused, and
  the refusal now names the option (`--per-slice-rescale apply`). With it,
  each slice is stored as `stored × slope + intercept`, computed in float64,
  as float32 with no rescale; a `per_slice_rescale` decision lists the slopes
  and their slices, and the import activity says `"per_slice_rescale":
  "applied"`. SUV is never computed, a slice stating no rescale among slices
  that do is still refused, and a series with one rescale is written the
  same either way. `read_series` takes the same option.
