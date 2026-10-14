<!-- area: release -->
### Added

- **Each pull request records its change in `changes/`.** A fragment per pull
  request, written while the change is fresh, replaces the hand-edited
  [Unreleased] section; `.github/scripts/changelog.py` checks them, previews
  them on the documentation site and assembles them at release, and CI fails a
  pull request that adds none (`changes/README.md`).
