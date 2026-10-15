<!-- area: deidentification -->
### Security

- **A tool's timestamps no longer date the visit.** Provenance `started` and
  `ended`, and the file's `created`, were passed over, though a scan run the
  day of the visit gives its date. `--profile strict` cuts them to their year
  (`tool_timestamp`), and `basic` reports them once the dates are shifted.
