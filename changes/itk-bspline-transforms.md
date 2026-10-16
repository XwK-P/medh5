<!-- area: converters -->
### Added

- **ITK B-spline transforms are imported and exported.** A
  `BSplineTransform` of order 1 or 3, alone or applied first in a composite,
  is stored as 1.0 §10.5's `bspline` on a grid of its lattice, and maps every
  point of its domain as SimpleITK does; `to_itk_transform` writes a `bspline`
  back as one. Other orders and ITKv3's `BSplineDeformableTransform` are
  refused, each with its reason. B-splines were refused before.
