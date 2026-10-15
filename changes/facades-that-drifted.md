<!-- area: python-api -->
### Fixed

- **A Python converter refuses a collection in the native commands' words.**
  It said "works on sample files" where every native command says "works on
  one sample file"; both now come from one function.
- **`medh5.document.schema()` returns a fresh dict on every call.** It was
  cached, so every caller shared one mutable dict, and one caller's edit was
  every later caller's schema.
