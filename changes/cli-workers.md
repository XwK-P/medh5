<!-- area: cli -->
### Added

- **`--workers N` for `validate`, `verify`, `fix`, `index build` and
  `recompress`.** Up to *N* processes run the files, and the output — the
  text, the `--json` document, the exit code — is a one-process run's, byte
  for byte, in input order. A file named twice, or by a hard link, goes to one
  process; a process that dies fails only the file it held. The default, 1,
  runs in one process as before.

### Performance

- **Batch commands scale with processes.** One lock covers the HDF5 library
  in each process, so files checked in one process take turns; `--workers`
  runs them side by side: over 32 files on four cores, `verify` went from
  10.1 s to 2.7 s, `index build` from 17.4 s to 4.6 s and `recompress` from
  128 s to 45 s.
