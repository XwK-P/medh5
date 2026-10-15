<!-- area: geometry -->
### Added

- **A displacement field can be stored in `float16`.** `add_transform` takes
  `field_dtype=`: `"float32"` (the default, as before), `"float16"`, which
  refuses a component `float16` cannot hold, or `"auto"`, which stores
  `float16` when every displacement is finite and below the 64 voxels §10.3
  recommends it for, and `float32` otherwise. The field takes half the space.
