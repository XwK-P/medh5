<!-- area: validation -->
### Fixed

- **A sampling index is read in the algorithm its digest names.** §13.3
  prefixed an index's `source_digest` with `<algo>:` and never said what
  hash it was; medh5 always recomputed in sha256, so a sha512 index another
  writer put in a sha512 file was stale (W905) and the file failed `strict`.
  The prefix now decides, as it does for a dataset's digest: `fresh_indices`,
  `verify` and the validator recompute under it, and an entry naming an
  algorithm outside §2.1, or spelled otherwise than §13.1 spells a digest, is
  stale. medh5 still pins its own indexes in sha256, so every index it wrote
  stays current.
