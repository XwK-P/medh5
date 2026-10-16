<!-- area: platform -->
### Behaviour changes

- **macOS on Intel (x86_64) is no longer supported.** 2.2.0 publishes no
  macOS x86_64 wheel or binary, and the Homebrew formula refuses to install on
  an Intel Mac. On one, pip builds medh5 from its source package (Rust, a C
  compiler and CMake), which is no longer tested there; or stay on 2.1. Apple
  silicon is unchanged.
