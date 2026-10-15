<!-- area: torch -->
### Added

- **`MedH5Reader` lets MONAI's `LoadImage` and `LoadImaged` open `.medh5`
  files.** It is a MONAI `ImageReader`: it reads one image — or, with
  `annotation=`, one voxel annotation as an `int64` label volume, `65535` in
  its ignore region — with the affine `to_metatensor` gives, takes a
  collection member as `path::key`, and stacks several files as channels only
  when they share one grid. `LoadMedH5d` is `to_dict` as a dictionary
  transform. Both read through the per-process handle cache, and
  `import medh5.monai` still imports no MONAI.
