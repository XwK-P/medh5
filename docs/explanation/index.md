# Explanation

Why the format is shaped the way it is. Nothing here is needed to use the
package — it is here because the decisions below are the ones most likely to be
mistaken for arbitrary.

- **[The data model](data-model.md)** — a sample is a *subject*, not a scan, and
  most of the rest follows from that: grids, timepoints, coverage, integrity.
- **[What the converters refuse, and why](refusals.md)** — every import and
  export that fails on purpose, and the reasoning. A refusal you do not
  understand looks like a bug; a refusal you do understand is usually telling
  you something true about your data.
- **[Design rationale](design-rationale.md)** — what 0.x could not express,
  the alternatives weighed for each load-bearing decision, the costs accepted,
  and the non-goals.

The design rationale's [1.1 section](design-rationale.md#clinical-context-and-prediction-tasks-11)
covers the clinical profile and the task and cache contract.

For the normative version, see [the specification](../spec/medh5-1.0.md) and
[1.1](../spec/medh5-1.1.md).
