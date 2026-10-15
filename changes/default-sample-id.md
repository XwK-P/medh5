<!-- area: writer -->
### Deprecated

- **A default `sample_id` or pack key that is not the file stem warns.**
  Without `sample_id`, a sample is named for its file name up to the first
  dot, and `pack` keys a member the same way, so `case.v2.medh5` and
  `case.v3.medh5` both became `case`. medh5 3.0 takes the stem (`case.v2`).
  Changing it now would silently rename samples, and split `P001.ct` and
  `P001.mr` into two subjects. So `create` and `pack` raise a
  `FutureWarning` wherever the two differ, and `medh5 pack` prints a note.
  Pass `sample_id` (and `subject_id`), or `keys`, to keep the name you want.

### Fixed

- **A file name with nothing before its first dot is refused by name.**
  `.hidden.medh5` failed "[E005] identity requires both sample_id and
  subject_id", naming neither the file nor the cause. It now names the file
  and says to pass `sample_id`.
