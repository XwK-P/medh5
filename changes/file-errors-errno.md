<!-- area: python-api -->
### Added

- **A missing, unreadable or directory path raises the matching `OSError`
  subclass.** `MEDH5FileNotFoundError`, `MEDH5PermissionError` and
  `MEDH5IsADirectoryError` are each a `MEDH5FileError` and the built-in of
  their name, and every `MEDH5FileError` carries `errno` and `filename`, so
  `except FileNotFoundError` catches a missing sample. `str()` is unchanged.
- **An HDF5 read failure raises `MEDH5IOError`.** It is an `OSError`, as the
  plain `OSError` it replaces was, and not yet a `MEDH5Error`. An engine
  `OSError` that names an errno now carries it, so it is the subclass of that
  errno.

### Deprecated

- **`MEDH5IOError` derives from `MEDH5FileError` in medh5 3.0.** An HDF5 read
  failure will then be a `MEDH5Error` too; catch `OSError` to catch it in both.
