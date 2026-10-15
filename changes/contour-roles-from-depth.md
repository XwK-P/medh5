<!-- area: annotations -->
### Fixed

- **Reading contours no longer fails on a `contour_role` value.** The reader
  raised an index error on a role other than 0 or 1, against §16's rule that
  readers ignore values they do not know. A polygon's role is now its
  nesting depth's parity among its class's polygons in its plane, from
  which §8.6 now defines the region; a polygon out of any plane keeps the
  role stored for it.
