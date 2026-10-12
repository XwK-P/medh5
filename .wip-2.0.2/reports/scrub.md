# scrub report (w202/scrub d17b873 + 93af8cd) — all 14 items DONE
Files: curation/scrub.rs, medh5-cli/src/scrub.rs, medh5/curation/scrub.py, docs/guides/deidentify.md, tests/tools/test_scrub.py. Bindings/_core.pyi unchanged.
Separate repack dropped: apply sets writer.compact = true; freed-space tests pass.
Tests: 14 Rust (7 new); 365 pytest (133 scrub, 46 new, 36 fail on main); docs deidentify 5; other scrub tests 123; targeted after compaction 282; fmt/clippy/ruff/mypy clean; strict scan of 151 corpus files: no finding added/lost, one not_checked note added (E704 alias case).
## CHANGELOG drafts
Security:
- scrub reads every name and every copy of the document: second hard link's name, soft link and its target, committed datatype's name and attributes, strings nested in compound/array/vlen (reported unreadable). /meta under a second name or a copy is reported and removed by --apply; before, all scanned clean.
- --pseudonymise-ids replaces the old ids wherever a string holds them (a record number in a label, note or attribute used to outlive its pseudonym).
- Identifiers the tables could never match are now found: nhs_number, birth_date, date_of_birth, *_uid; top-level extra keys such as PatientID; whole-date values under any key; UIDs inside text.
- Strict name pseudonyms are salted and recorded in uid_map.
Fixed:
- A timepoint date added after a recorded shift is reported.
- The de-identification record and agents' organization are scanned.
- A pseudonym under a UID key is no longer reported, so a scrubbed file can scan clean.
Behaviour changes:
- A date_shift_days of 0, or one that differs from the recorded shift, is refused; a recorded 0 is read as no shift.
- Apply-only options (--date-shift-days, --salt, --by) are refused without --apply (or apply_changes=True in scrub_paths); scrub_paths takes keyword-only arguments, so apply=True raises TypeError; scrub_paths processes every path then raises the first refusal with .reports (None per refused path).
- More findings than before: a whole date under any key is actionable; byte buffers or undecodable types outside the imaging groups are unreadable.
Performance:
- A scrubbed file is compacted at its commit instead of being rewritten a second time.
Documentation:
- The de-identification guide states its coverage exactly, and its gate blocks on open identity findings, clinical samples and anything not examined.
## Handoffs
- docs: cli.md medh5 scrub: "every object name" -> "every link name, attribute (committed types included) and string dataset"; beside "Running it twice…" add "a different or zero shift is refused; --date-shift-days, --salt and --by need --apply".
- cli: app.rs help text: say "with --apply" on those three flags.
- spec §11.4 optional: "A recorded date_shift_days of 0 means no shift."
## Decisions
- Same-object alias of /meta reported actionable and removed; acquisition top-level keys: key rules apply, image id reviewed not renamed; software version reading as a date reported not acted on; ids < 4 chars not swept; date-after-shift flagged when > 1.5 days from lowest-index dated timepoint; not_checked names first 10 datasets; strict without salt allowed (unsalted, recorded); birth_date/date_of_birth shifted, PatientBirthDate removed; Rust engine scrub_paths unchanged (public, no callers).
