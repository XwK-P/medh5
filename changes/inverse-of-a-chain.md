<!-- area: geometry -->
### Added

- **The inverse of a chain or a composite is its steps' inverses, in
  reverse.** `inverse()` of a `ChainTransform`, or of a stored composite
  without a usable `inverse_id`, returns that chain when every step can be
  inverted, where it returned `None`, and `InverseTransform.can_invert` says
  so. `is_invertible` stays the file's claim, and the frame graph is unchanged.
