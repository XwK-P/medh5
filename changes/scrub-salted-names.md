<!-- area: deidentification -->
### Security

- **A name is no longer pseudonymised without a salt.** `--profile strict`
  replaced a person or organization agent's name with an unsalted hash, which
  anyone reverses by hashing candidate names; the run is now refused without
  `--salt`. An unsalted run that pseudonymised UIDs reports
  `unsalted_pseudonym`; medh5 3.0 requires a salt.
