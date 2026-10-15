<!-- area: validation -->
### Behaviour changes

- **`W919` reports a contour role that disagrees with its depth, and `E404`
  one that is not 0 or 1.** §8.6 now says how a class's polygons in a plane
  combine: by nesting depth, with `contour_role` the depth's parity. Files
  2.0 imported from same-named nested RTSTRUCT ROIs draw `W919`; under
  `--level strict` they fail; pass `--codes-revision 1` to keep 2.0's
  verdicts.
