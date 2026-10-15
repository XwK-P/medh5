<!-- area: cohorts -->
### Added

- **`medh5.split/2` deals a split so that adding a group moves few others.**
  `make_splits(..., algorithm="medh5.split/2")` (`--algorithm 2`) fixes each
  partition's total by largest remainder and each stratum's counts by
  controlled rounding, then fills them in hash-rank order: one group added to
  115 moves a median of 1 other where `medh5.split/1` moved 19, and strata of
  3, 5 and 7 groups at 70/15/15 get a test group. The split JSON records
  `algorithm`; one without it reads as `medh5.split/1`.
- **`make_splits(..., previous=…)` continues a split without moving any
  group.** New groups fill each stratum's shortfall by `medh5.split/2`, the
  JSON records the lineage under `previous`, and `Split.moved_from(other)`
  lists what two splits place differently (`--previous` on the command line).

### Deprecated

- **`make_splits` without `algorithm` deals by `medh5.split/1` until 3.0.** It
  raises a `FutureWarning` (the native `dataset split` prints `warning: …`)
  only where `medh5.split/2` would place some group differently; pass
  `algorithm="medh5.split/1"` (`--algorithm 1`) to keep such a split.
