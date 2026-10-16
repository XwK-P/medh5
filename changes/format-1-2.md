<!-- area: specification -->
### Added

- **Format 1.2 adds one annotation kind, `polylines`: open curves.** Vessel
  centrelines, catheter and applicator paths, needle tracks and RTSTRUCT's
  open contours had no kind to go into: `contours` are closed polygons, and
  `points` are unordered. A sample is written at 1.2 only when it holds a
  polylines annotation, so imaging alone stays 1.0 and the clinical profile
  1.1; removing the last one lowers the version again. A 1.1 reader reads a
  1.2 file as its projection, the curves left out as `W913`, and its digests
  and `content_id` verify. No diagnostic code is added; `E408`'s summary names
  polyline offsets. The corpus has 175 cases: `shapes-polylines`,
  `E408-polyline-offsets` and `E404-polyline-one-vertex` are new, and the
  higher-minor case declares 1.3. The specification is
  `docs/spec/medh5-1.2.md`.
