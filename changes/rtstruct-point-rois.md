<!-- area: converters -->
### Added

- **RTSTRUCT POINT ROIs import as a `points` annotation, and export again.**
  `from_rtstruct` writes each POINT contour as one row of `points_id`
  (`<ann_id>_points` by default; `--points-id`), in world coordinates of
  the contours' frame, of the ROI's class, with `task="other"`; it was left
  out with a warning. A structure set of points alone writes no contours.
  `to_rtstruct(points=…)` (`--points`) writes them back as POINT items
  interpreted as `MARKER`. Open curves are still left out, now with a
  warning naming `open_curves="omit"` (`--open-curves`), the only value
  until `polylines` arrive in 2.2.0.
