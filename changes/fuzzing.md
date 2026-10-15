<!-- area: release -->
### Added

- **The engine's readers are fuzzed twice a week.** Four targets run under
  libFuzzer (`fuzz/`): any file validated at the strictest level, a clinical
  documents table's offsets and bytes read deferred and scanned, and the
  Zstandard and Blosc2 filters decoding chunks they did not write.
