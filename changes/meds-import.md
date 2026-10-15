<!-- area: clinical -->
### Added

- **A MEDS importer.** `medh5.io.meds.read_meds(source, subject_id=...,
  clock=...)` reads one subject's rows of a MEDS 0.3 dataset as clinical
  records, and `augment_from_meds` adds them to a sample. It invents no zone,
  origin, unit, code or availability: a time without a zone on a UTC clock,
  a relative clock without `origin` and a row holding both values are
  refused, availability stays null unless a column holds it, and a unit comes
  only from a declared column. Ids are hashes of the rows, so a
  re-extraction keeps them and a second import is refused. `MEDS_BIRTH` and
  free text are left out unless asked. Needs `pip install "medh5[meds]"`
  (pyarrow 14 or later).
