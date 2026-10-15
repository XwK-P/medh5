<!-- area: clinical -->
### Fixed

- **An amendment holds what a clinical event version attests, wherever it
  writes.** Writing to another path read the output for what the source held,
  and passed; leaving the clinical group behind with `amend_into` held no
  version, so one added again could carry a new payload; and an attested
  payload was compared by its datasets' digests alone, so an edit to an
  attested image's `rescale_slope` passed. Each is now `E809` (1.1 §7.3).
