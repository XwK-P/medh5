<!-- area: clinical -->
### Behaviour changes

- **A time given in hours or days is rounded half to even everywhere.**
  `medh5 clinical select --cutoff-hours` and the imaging events
  `clinical augment` makes from `days_from_baseline` rounded half away from
  zero, while `medh5.clinical.hours` and `days` round half to even, so at an
  exact tie a cutoff from the command line was one microsecond later than the
  same cutoff from a script. The engine now rounds once
  (`medh5::clinical::round_micros`), half to even, as Python always has.
