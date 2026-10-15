<!-- area: documentation -->
### Documentation

- **A tutorial from clinical history to a training batch.**
  `docs/tutorials/clinical-task.md` writes two subjects on a `utc` clock, each
  event with when it happened and when it became known, asks a task of them at
  a cutoff, shows a preflight excluding a row and why, and loads a batch. Its
  blocks build on each other, and the documentation tests now run such a page
  (marked `<!-- sequential -->`) top to bottom in one session.
