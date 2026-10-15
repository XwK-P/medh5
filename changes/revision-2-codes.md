<!-- area: validation -->
### Behaviour changes

- **`--level strict` fails a file that draws a revision-2 warning; pass
  `--codes-revision 1` to keep 2.0's verdicts.** Revision 2 of the code table
  allocates W915: a dataset without a `digest`, inside an object a declared
  `content_id` covers, is reported at the integrity level (§13.1). Until now
  `validate --level strict` passed a file that `verify` failed as unattested,
  so a CI gating on the validator missed an edit nothing could detect. `verify`
  is unchanged; the two now fail the same files.
