<!-- area: clinical -->
### Added

- **A FHIR R4 importer.** `medh5.io.fhir.read_fhir(source, subject=...,
  clock=...)` reads one patient's Observations and DiagnosticReports from a
  Bundle as clinical records, and `augment_from_fhir` adds them to a sample.
  A date's precision becomes its bounds, `issued` the availability,
  `meta.versionId` a chain of revisions linked by `supersedes`, a comparator
  `lt`, `le`, `gt` or `ge`, and `dataAbsentReason` the `missing_reason`; each
  component is an event of its own. A resource with no coding is left out
  rather than coded from its text, and versions that cannot be ordered are
  refused. Report texts and imaging-study links are imported only when asked.
  Standard library only.
