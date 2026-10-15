<!-- area: geometry -->
### Performance

- **A composite prefilters a cubic component once.** A composite opened its
  components afresh at every evaluation, so a `cubic` displacement field
  inside one was read whole and prefiltered on every `transform_points` call
  — once per item of a paired loader. Components and named inverses now
  share the coefficients of the transform of that id, within the same
  process-wide budget.
