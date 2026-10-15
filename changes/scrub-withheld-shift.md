<!-- area: deidentification -->
### Security

- **The amount of a date shift can be withheld (D14).** A record stating
  `date_shift_days` lets one known date restore every other.
  `--withhold-shift` (`withhold_shift=True`) leaves it out and records in the
  de-identification activity that the dates were shifted, with a salted
  commitment a re-run checks. `--profile strict` withholds by default; `basic`
  still records it and warns (`FutureWarning`) that medh5 3.0 will withhold
  it.

### Added

- **`--date-shift-range MIN:MAX` gives each subject its own shift,** taken
  from the salt and the subject id, the same for every file of one subject,
  and always withheld (`date_shift_range=(min, max)`).
- **`--out FILE` and `--out-dir DIR` scrub to a copy** and leave the source
  as it is (`out=`, `out_dir=`); a copy never replaces an existing file.

### Behaviour changes

- **A scrub keeps a `burned_in_annotation_checked` someone recorded as
  true.** It wrote `false` over it, though it does not look at pixels.

### Documentation

- **1.0 §11.4 advises against recording the shift's amount** in a file shared
  beyond whoever holds the key, and 1.1 §3 says `/meta` may withhold it; a
  conformance case holds a shifted clock whose shift is withheld.
