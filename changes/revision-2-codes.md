<!-- area: validation -->
### Behaviour changes

- **`--level strict` fails a file that draws a revision-2 warning; pass
  `--codes-revision 1` to keep 2.0's verdicts.** Revision 2 of the code table
  allocates W915: a dataset without a `digest`, inside an object a declared
  `content_id` covers, is reported at the integrity level (§13.1). Until now
  `validate --level strict` passed a file that `verify` failed as unattested,
  so a CI gating on the validator missed an edit nothing could detect. `verify`
  is unchanged; the two now fail the same files.
  At the semantic level it allocates W916, a classification under
  `closure="implicit"` asserting a class 0.0 beside a positive `is_a`
  descendant, which `state()` answers positive (§5.4, §9); W917, two grids
  naming one `frame_uid` that §3.3 still keeps apart, such as a 2-D key image
  tagged with a CT's frame (§3.3); and W918, a voxel annotation declaring
  `closure="implicit"`, whose ancestor reads include every descendant's
  voxels (§5.4).
