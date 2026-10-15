<!-- area: task-cache -->
### Added

- **Clocks and days for clinical tasks.** `Clock.utc(id)` and
  `Clock.shifted_utc(id)` build the two common clocks; `utc_us(datetime)`
  turns an aware datetime into the microseconds a task states, and
  `day_bounds(day, tz=None)` and `clock_day(t_us)` give the inclusive bounds
  of a calendar day, the span every time zone's day can cover when no zone
  is given (`Bounds::day_containing` in Rust).
- **`medh5 cache info` and `medh5 cache ls`** show what a cache holds and
  each entry it records, by source version or row, without decoding
  anything.
- **A moved cache validates through its task.** With `--task`, an entry
  source the task pins is opened where the task's reference says, and
  `cache validate --task-base DIR` resolves the task's URIs (contract §7.1).
- **Repin a task** after its samples were changed on purpose:
  `TaskManifest.repin()` and `medh5 task repin` pin each source to the
  version its file holds now, refusing one that is not whole or no longer
  its subject's, and report the rows whose selection may have moved. T302
  names the way out.
