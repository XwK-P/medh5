<!-- area: annotations -->
### Added

- **`SampleWriter.add_polylines` writes open curves, and
  `PolylinesAnnotation` reads them.** Each `Polyline` is ordered vertices
  with a class, an optional grid plane (`plane=(axis, index)`) and, through
  `instance_ids=`, an instance; `polyline(i)`, `polylines()`, `offsets` and
  `planes` read them back, in index or world space. A polyline of fewer than
  two vertices is `E404`, offsets that do not delimit the vertices `E408`, and
  a polylines annotation satisfies neither `seg` nor `det`, whatever its
  `task`. In Rust, `annotations::Polyline`, `encode_polylines` and
  `SampleWriter::add_polylines`; `version::required_for_content` and
  `written_version_for_content` give the version a content needs.
