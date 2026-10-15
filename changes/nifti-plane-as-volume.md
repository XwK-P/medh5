<!-- area: converters -->
### Added

- **`from_nifti(plane_as_volume=True)` imports a 2-D plane where it lies.**
  A 2-D NIfTI off z = 0, or not parallel to x-y, is refused, since a 2-D grid
  has no z; the refusals now name the option (`--plane-as-volume`). With it,
  every 2-D image and mask is read as `(x, y, 1)` with the file's whole
  affine — the import of the plane re-saved that way — and a
  `plane_as_volume` decision records its z and normal. A third affine column
  that is zero or not orthogonal to the plane is refused. `read_nifti` and
  `import_seg_nifti` take the same option.
