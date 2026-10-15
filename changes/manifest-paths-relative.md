<!-- area: cohorts -->
### Added

- **`medh5 dataset upgrade` rewrites a 2.0 manifest as `medh5.dataset/1`.**
  `--base` names the directory its paths are relative to; it refuses, writing
  nothing, when an entry's file is missing there. `Manifest.save` takes
  `paths="keep" | "relative" | "absolute"`, `Manifest.load` takes `base=`, and
  `medh5 dataset index` takes `--absolute`.

### Deprecated

- **A manifest written before 2.1 is read against the working directory until
  3.0.** Where that reading and its own directory's differ, loading it raises
  a `FutureWarning` (the native command line prints `warning: …`); medh5 3.0
  refuses it unless `base=` is given. Upgrade it with `medh5 dataset upgrade`.

### Behaviour changes

- **A cohort manifest's paths are relative to the manifest.** Every save names
  `"schema": "medh5.dataset/1"` and writes a path relative to the manifest's
  directory when it lies beneath it, absolute otherwise; loading joins it back
  onto that directory. The saved text changes only where it was already
  unreadable from any directory but the scan's: a manifest loaded elsewhere
  listed every file as stale, `dataset check` reported `C402` for files that
  exist, and the statistics opened nothing. `sha256` is unchanged.
