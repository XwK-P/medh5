<!-- area: torch -->
### Added

- **Patch predictions stitch back into volumes.** `GridStitcher` takes a
  `GridPatchDataset`'s predictions batch by batch, in any order, and returns
  each file's volume as soon as its last patch arrives; `finish()` refuses,
  naming each file still missing patches, rather than return a volume with
  holes. Overlaps are blended with Gaussian or constant weights.
  `medh5.sampling.PatchAggregator` and `blend_weights` do the arithmetic
  without torch, and each `GridPatchDataset` item's `meta["cover"]` says which
  of its file's patches it is.
