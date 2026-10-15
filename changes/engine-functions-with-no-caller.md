<!-- area: release -->
### Deprecated

- **Six engine functions and one type that nothing called are deprecated
  in the Rust crate:** `clinical::columns::columns_of` (an unknown table
  name answered with the link columns), `annotations::encode::array_data`
  (use `PayloadData::Array`), `annotations::select::bbox_volume`,
  `bench::targeted_metrics`, `storage::codecs::profiles_repr`,
  `geometry::linalg::matmul` (use `a.dot(b)`) and `sampling::PairReport`
  (Python's `medh5.sampling.PairReport` is the live one). Each note names
  its replacement; 3.0 removes them.
