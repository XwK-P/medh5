<!-- area: platform -->
### Fixed

- **A thread waiting for HDF5 on free-threaded CPython lets go of the
  interpreter.** Waiting attached for another thread's HDF5 call held up every
  stop-the-world pause, a garbage collection among them, until that call ended.
  A read or write through medh5 now runs detached throughout. Closing a sample
  no longer holds up a thread asking for it, and a lock left by a panic is taken
  as it stands instead of raising again. A payload's dicts, edited by another
  thread while it is written or described, no longer panic the read.
