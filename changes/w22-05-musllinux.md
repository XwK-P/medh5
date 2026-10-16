<!-- area: platform -->
### Added

- **Alpine and other musl Linux install a wheel.** x86_64 and aarch64 each get
  a `musllinux_1_2` abi3 wheel and a free-threaded 3.14 one, so medh5 no
  longer builds from source there. Each release also has a `medh5` binary
  for `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl` that links
  musl statically and so runs on any Linux of its architecture.
