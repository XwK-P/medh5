<!-- area: task-cache -->
### Fixed

- **A cache entry with no sources is T401.** The writer refused it as stale
  (T403), but the schema requires a source, and a writer refuses an entry
  that would fail its schema (T401).
- **`CacheEntry.from_json` reads an entry as the engine does**, refusing a
  member of the wrong type (T401) rather than reading it leniently; and
  `CacheReport.ok`, `stale` and `corrupt` are the engine's verdict
  (`CacheReport.to_json`).
- **`Selection.admits_any(kind, id)`** (Python and Rust) looks in every
  fragment: `admits` answers for one, fragment 0 unless told, so a payload
  of a subject's second fragment read as not admitted.
- **A negative cutoff with an exponent** is written
  `--cutoff-hours=-5.5e-07`: after a space it reads as a flag. The reference
  says so.

### Performance

- **Validating an event-level cache opens each source once**, not once for
  its pin and again for what its versions own; `TaskManifest::rows_in`
  looks each subject up once rather than once per row.
