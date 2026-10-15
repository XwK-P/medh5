<!-- area: deidentification -->
### Added

- **`scrub --apply` de-identifies a clinical sample, as a new sample (D15).**
  The scan reads the clock, the ids and each document's text in turn
  (`clinical_clock`, `clinical_id`, `clinical_text`); `--apply` moves a `utc`
  clock's times by the one shift, makes it `shifted_utc`, and re-mints every
  id from the salt, links following, so a cutoff moved by the shift admits
  the same versions. Text is never rewritten: `--attest-text TOOL`
  (`attest_text=`) records who de-identified it, by the digests of its bytes,
  and lapses when they change. Unattested text, or a name, UID or record
  number in it, still refuses the apply.

### Documentation

- **1.1 §10 describes de-identification as a derivation,** with its row in
  Appendix A.
