<!-- area: release -->
### Security

- **Every release artifact carries a build attestation.** The binaries'
  archives on the release page had only `SHA256SUMS`, which says what was
  uploaded, not who built it; each archive, wheel and sdist now has a signed
  statement that the release workflow built it from the tagged commit,
  checked with `gh attestation verify FILE --repo medh5-project/medh5`.
