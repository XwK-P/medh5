<!-- area: specification -->
### Fixed

- **1.1 §10 names MEDS's columns as MEDS does.** It gave this profile's
  column names (`value_num`, `value_text`) for MEDS's, so an importer written
  from it looked for columns no MEDS dataset has; it now maps `time`,
  `numeric_value` and `text_value` onto them (Appendix A).
