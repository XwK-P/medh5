<!-- area: release -->
### Fixed

- **CI and the advisory audit run on the patch line.** CI ran on pushes and
  pull requests to `main` only, so a fix on `release/2.0` merged untested, and
  the weekly audit, which runs on the default branch, saw only `main`'s lock.
  Both now run on `release/**`, and the weekly audit checks each line.
