<!-- area: writer -->
### Behaviour changes

- **An amendment refuses a file whose `content_id` no longer matches it, or
  whose digests it cannot read.** A file edited outside medh5 (`E702`), one
  holding a digest that does not parse, or one declaring a `digest_algo` §2.1
  does not permit (`E703`) is refused before anything is written, by every
  command that amends: `medh5.amend`, `fix` without `--rewrite-digests`,
  `scrub --apply`, split claims, `clinical augment` and `strip`, and `seg`.
  Previously the next scrub, split claim or annotation re-attested the edit,
  and an unknown algorithm was silently replaced by `sha256`. Repair the file
  with `medh5 fix --rewrite-digests --reason`, which restamps an unknown
  algorithm in `sha256` and records the change in the activity's parameters.
  `verify` and `diagnose` name that repair when they meet an unknown
  algorithm.
