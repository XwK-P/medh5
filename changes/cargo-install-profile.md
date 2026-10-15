<!-- area: cli -->
### Documentation

- **`cargo install medh5-cli` can build the binary the releases ship.** A
  crate on crates.io carries no build profile, so the install built without
  link-time optimisation and kept the symbol table; the install page and the
  crate's README give the release profile as environment variables, in
  PowerShell and as `--config` options, and still prefer the release binaries.
