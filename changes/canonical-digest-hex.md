<!-- area: validation -->
### Fixed

- **A digest spelled in uppercase is malformed (E703), not a mismatch (E701).**
  A digest was read as Python's `int(hex, 16)` reads a number, so a sign,
  `0x`, `_`, whitespace, uppercase or short hex parsed, and an uppercase
  digest of unchanged bytes was reported as a mismatch and its root as
  tampered with. §13.1 now says `<hex>` is lowercase hex of the algorithm's
  length; anything else is E703, in `verify`'s and `diagnose`'s `malformed`,
  and `fix --rewrite-digests` restamps it. A `content_id` spelled otherwise is
  E703 rather than E702, and an amendment refuses it. `parse_digest` stays as
  lenient as it was; `medh5::digest::is_canonical_digest` decides.
