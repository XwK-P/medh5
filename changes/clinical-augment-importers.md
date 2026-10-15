<!-- area: cli -->
### Added

- **`medh5 clinical augment --from-meds DIR --meds-subject N` and
  `--from-fhir FILE --fhir-subject REF`** import one subject's history in
  place of a logical-record bundle, onto the clock `--clock-id` and `--clock`
  name or the sample's own, with the conversion report printed and kept by
  `--report`. The standalone binary hands them to the Python package, as it
  hands over `medh5 convert`.
