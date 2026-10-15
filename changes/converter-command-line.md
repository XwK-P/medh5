<!-- area: converters -->
### Added

- **Every documented converter option has a flag on `medh5 convert`.**
  `--annotated-classes all_given|all|K1,K2` (from-nifti, from-dicom-seg,
  from-rtstruct, import-seg-nifti), `--codec PROFILE` (from-nifti, from-dicom,
  from-nnunet and `medh5 migrate`, which wrote `balanced` whatever it was
  asked), `--class` and `--series-description` (to-dicom-seg), `--label`
  (to-rtstruct), `--mask-id` (from-rtstruct, with `--rasterize`),
  `--coord-system` (from-nnunet), `--file-ending` (to-nnunet), `--value-units`
  (from-nifti), and `--report`/`--json` on to-nifti, which still prints the
  file it wrote unless `--json` asks for the report.
- **`medh5 convert import-seg-nifti` adds NIfTI masks to a sample.**
  `import_seg_nifti` had no command; it takes `--mask NAME=PATH` (repeatable),
  `--id`, `--grid`, `--assume-aligned`, `--assume-geometry`,
  `--plane-as-volume` and `--annotated-classes`, and amends the sample in
  place as `from-dicom-seg` does.
- **`from_nifti(label_set=)` fixes a cohort's class ids** (`--label-set
  FILE`). Masks are matched by key to a `LabelSet`, or a JSON file
  `medh5.labels.load_file` reads, instead of minted in sorted order, so a
  case without one organ no longer renumbers the others; a mask the set has
  no class for is refused (E402).

### Behaviour changes

- **`to_rtstruct` refuses a `label` longer than 16 characters.** The
  `StructureSetLabel` is a DICOM SH: a longer label was written, and the file
  was not valid DICOM. Shorten it (`label=`, `--label`).

### Fixed

- **A class named in `annotated_classes` that the input does not supply is
  recorded as examined and absent.** The NIfTI, DICOM SEG and RTSTRUCT
  importers raised a bare `KeyError` ("label set 'converted' has no class
  'tumor'"). An import that mints its classes now mints it after the
  supplied ones, with a `label_set` decision; one into a given label set
  that lacks it is refused as E402, naming the class.
- **`to-dicom-seg --series-description` longer than 64 characters is refused
  by name.** highdicom's `ValueError` for the DICOM LO reached the command
  line as a traceback.
