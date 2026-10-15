<!-- area: validation -->
### Added

- **The diagnostic code table is revisioned, and a validator can pin a
  revision.** Each code names the format version whose rules it reports
  (`since`) and the revision of the table that allocated it (`revision`);
  `Code` gains both, and `medh5.errors.CODES_REVISION` is the table's own.
  `medh5 validate --codes-revision N`, `validate_file(..., codes_revision=N)`
  and the engine's `validate_file_with` report only the codes revision `N`
  allocated, so a `--level strict` gate does not move when a later revision
  adds a warning (1.0 §15.1, §15.2, §16). The conformance suite's
  `codes.json` carries both members and the table's revision.
