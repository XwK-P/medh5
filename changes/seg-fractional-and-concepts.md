<!-- area: converters -->
### Added

- **`to_dicom_seg` writes a `probmap` as a `FRACTIONAL` SEG.** Pass
  `segmentation_type="fractional"` (`--segmentation-type fractional`), and
  `fractional_type="PROBABILITY"` or `"OCCUPANCY"` (`--fractional-type`).
  Values are stored out of 255, within 1/510; the report gives the largest
  error and how many voxels moved across the stored threshold, and says that
  `threshold` and `normalized` are not carried. Any other kind is refused.
- **A SEG export carries each class's codes, category and laterality.** The
  property type is the class's first code whose system has a DICOM
  designator (one table, `SNOMED-CT` ↔ `SCT` and six more, read both ways);
  the category comes from `properties["dicom_category"]` or `category`; a
  `laterality` is the type modifier. Where the sample states none, the
  generic SCT 91723000 is written as before, now as a guess naming each class.

### Behaviour changes

- **A SEG export's category follows the class.** A `lesion` class is SCT
  49755003 "Morphologically Abnormal Structure", a `device` SCT 260787004
  "Physical object" and a `region` SCT 309825002 "Spatial and Relational
  Concept", where every segment was "Anatomical Structure".
- **A predicted annotation exports as `AUTOMATIC`.** A `predict` activity by a
  software agent of stated version makes its segments `AUTOMATIC`, identified
  by that software (family DCM 123110 "Artificial Intelligence"); every
  segment was `MANUAL`. Without such provenance the export still writes
  `MANUAL`, and records a guess. A binary export of a `probmap` records the
  threshold it cut at.
- **A SEG import records category and laterality, and spells systems as §5.2
  does.** A minted class keeps the property category as
  `properties["dicom_category"]`, takes `category` where the concept names one
  (`lesion`, `device`, `region`) and `laterality` from the type modifier; a
  binding's `system` is `SNOMED-CT`, not `SCT`, for a designator the table
  knows. Segments of one label with different modifiers are refused, where
  Left and Right Kidney became one class.
