<!-- area: deidentification -->
### Added

- **`medh5 scrub` scans and scrubs a collection, member by member.** A
  `.medh5c` is scanned into one report whose findings name their member
  (`KEY::where`); `--apply` writes a new collection at `--out`, or under
  its own name in `--out-dir`, one of which it requires, and only reads the
  source. Each member is scrubbed with the one salt, so
  pseudonyms, re-minted ids and a range's per-subject shift agree across
  members; under `--pseudonymise-ids` a member is keyed by its pseudonym. A
  member that refuses leaves nothing written, or with `--skip-refused`
  (`skip_refused=True`) is left out and listed.
