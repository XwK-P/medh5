<!-- area: deidentification -->
### Added

- **`medh5 scrub --apply --redact-text` rewrites the names, dates, UIDs and
  sample ids it finds in clinical text.** Each becomes `[NAME]`, `[DATE]` or
  `[ID]` in a new document owned by a new version of its event, with every
  span into it moved through the edit; a span that overlaps a redacted range
  refuses the run (1.1 §6, §10). Python's `scrub.apply` takes
  `redact_text=True`. The redaction is recorded with the digests it wrote,
  and the new text still needs `--attest-text`.
