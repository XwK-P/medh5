<!-- area: release -->
### Security

- **The crates are published through crates.io's trusted publishing.** The
  release held a long-lived `CARGO_REGISTRY_TOKEN` and verified one crate at a
  time, so a failure verifying `medh5-cli` left `medh5-sys` and `medh5`
  published. Every crate is now verified first, with no token in existence,
  and the upload uses a short-lived token issued to the release workflow.
