<!-- area: task-cache -->
### Deprecated

- **A target with negatives states its follow-up.** `min_follow_up_us`
  defaults to the horizon, which leaves a negative only an observation at
  exactly the cutoff plus the horizon, so nearly every row without a
  positive is censored. The default cannot change under `medh5.task/1`: it
  is in every task fingerprint. `Target` with `negative` set and no
  `min_follow_up_us` now raises a `DeprecationWarning` (3.0 refuses it); a
  manifest read from a file gets a note instead
  (`TaskManifest.notes`, `medh5::companion::task::notes_of`), which
  `medh5 task validate` and `task preflight` print after the findings and
  list under `notes` with `--json`. A note fails nothing.
