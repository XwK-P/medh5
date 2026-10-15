<!-- area: specification -->
### Fixed

- **Voxel annotations are not pyramided in 1.x.** §4.3 said a voxel
  annotation "MAY be pyramided identically, with `nearest` or `max`" and
  defined nothing more: not which voxel `nearest` keeps under an even factor,
  what `max` means over class ids, or how coverage and the ignore region
  downsample. Nothing wrote, read or validated one. §4.3 now says an
  annotation's datasets hold one level, and §16 reserves `levels`,
  `downsample_factors`, `downsample_method` and `grid_levels` on an
  annotation for a later minor. The writer refuses them (E412), so a
  transcode no longer carries one a third-party header holds.
