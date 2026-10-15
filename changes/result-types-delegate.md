<!-- area: python-api -->
### Fixed

- **`Measurement.to_json()` writes `None` for a NaN, as `medh5 bench --json`
  does, and `Measurement.from_json()` reads it back as NaN.** It wrote
  `NaN`, which JSON cannot carry. The result types —
  validation `Report` and `merge`, `conformance.summarize`, `Case`,
  `CaseResult`, `RecompressResult`, `Diagnosis`, `Repair`, `VerifyResult` and
  `Measurement` — now ask the engine for their verdicts, counts, text and
  JSON, so they say what the command line prints.
