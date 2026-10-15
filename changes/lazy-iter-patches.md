<!-- area: python-api -->
### Added

- **`PatchSampler.draws` and `iter_patches` take `grid=`, as `draw` does.** A
  run of draws can now be pinned to one grid, such as one visit's, whether or
  not an annotation lives there.

### Performance

- **`iter_patches` draws one patch per `next()`.** It drew all `n` patches
  before yielding the first, so `iter_patches(n=10_000)` held every patch at
  once. It yields exactly what `draws` returns for the same seed, and checks
  the annotation and the grid when it is called.
