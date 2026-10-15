<!-- area: task-cache -->
### Added

- **The conformance suite scores feature caches and task results.**
  `companion/caches/` publishes nine feature caches over the suite's clinical
  samples: a valid cache of each level, and one for each code from T401 to
  T407. Each comes with the findings validating it must report, and with its
  task where the finding is about the task. Until now no fixture exercised
  any cache code. `medh5 conformance score SUITE RESULTS --companion` scores
  what another implementation reported on every task and cache fixture, as
  do `medh5.conformance.score_companion` and Rust's
  `conformance::companion::score`. A report's findings must equal the
  fixture's, and its `ok` must agree with them. For a fixture that states its
  rows, both a row added and a row left out fail it, and each row's reasons
  and inputs are compared as well as its status, selection, slots and label.
  Before, the fixtures were checked only from inside this engine's test
  suite.
