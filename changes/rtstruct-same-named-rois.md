<!-- area: converters -->
### Added

- **`from_rtstruct(roi_classes=…)` puts an ROI in a class of your choosing.**
  `{ROINumber: key}`, `--roi NUMBER=KEY` on the command line, imports ROIs
  that share a name apart; the key is minted, or must be in the sample's
  label set (`E402`).

### Behaviour changes

- **Same-named RTSTRUCT ROIs are one class, with a note, or refused.** ROIs
  whose names make one key are combined into one class, minting or not, and
  the report records a `merged_rois` decision naming them; minting used to
  fail with `E302`, and an existing label set merged them silently. Roles
  now come from nesting over the whole class: an import where one ROI's
  contour lies inside another's on a slice is refused, naming the ROIs and
  the slice, where 2.0 made the inner one a hole (an 11×11 "Bladder" holding
  a 4×4 one, 121 voxels drawn, imported as 105).
