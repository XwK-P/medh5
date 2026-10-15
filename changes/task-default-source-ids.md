<!-- area: task-cache -->
### Fixed

- **Default source ids no longer collide, and are the same on every
  platform.** A reference given no id took the file's stem after its
  directory, so `a/site-a/P-01.medh5` and `b/site-a/P-01.medh5`, or
  `P-01.medh5` and `P-01.h5`, were one id (`T204`), and Python split the URI
  with the platform's path rules, so one script's ids, and its fingerprints,
  differed on Windows. The engine derives the id now
  (`SourceRef::default_id`), splitting on `/` only, and
  `TaskManifest.add_subject` widens a derived id another source holds — by
  the directories above it, then `@2`, `@3` — leaving the first holder's as
  it was. An id given explicitly is never changed. Manifests that validated
  before keep their ids.
