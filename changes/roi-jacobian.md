<!-- area: geometry -->
### Behaviour changes

- **A region's Jacobian is the field's own.**
  `jacobian_determinant(roi)` differenced the region as if it were the whole
  field, one-sidedly at its own edges, so its edge values depended on where
  it was cut: two halves of a field could each report no folding where the
  field folds. It is now `jacobian_determinant()[roi]` exactly — the
  differences reach one voxel past the region — and `folding_fraction(roi)`
  is the fraction over those voxels. A slice with a step and an axis of one
  voxel, refused before, are accepted; the field itself still needs two
  voxels on every axis. A large region is read in slabs.
