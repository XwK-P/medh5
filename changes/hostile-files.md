<!-- area: documentation -->
### Documentation

- **`SECURITY.md` says what opening a file you did not write guarantees.** No
  filter plugin is loaded unless asked for, nesting and decompression are
  bounded, a malformed file is refused rather than crashing the process, and
  a rewrite refuses a link in another user's sticky directory. It also says
  what is not guaranteed: a file within those bounds can still cost time and
  memory in proportion to what it states.
