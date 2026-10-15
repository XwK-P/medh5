<!-- area: documentation -->
### Documentation

- **How-to guides for collections, review records, the conformance suite
  and DICOM SEG and RTSTRUCT imports.** `docs/guides/collections.md`
  packs samples into a `.medh5c`, reads and verifies its members and
  splits a cohort over them;
  `docs/guides/review-and-provenance.md` records who drew and reviewed a
  mask, its quality, and two raters' agreement, through `medh5.amend`;
  `docs/guides/conformance.md` publishes, validates and scores the suite,
  bounded for an earlier implementation, and takes over the how-to the
  conformance page held; `docs/guides/import-seg-rtstruct.md` imports a
  SEG and a structure set onto a CT series, keeps same-named ROIs apart
  with `roi_classes=`, and shows what the importers refuse.
