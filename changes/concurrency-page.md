<!-- area: documentation -->
### Documentation

- **A page on threads, processes and open files.**
  `docs/explanation/concurrency.md` gathers what was spread over five pages:
  one HDF5 lock per process and the Blosc2 window reads that decompress
  outside it, when threads help and when processes do, `fork` and `exec`,
  one thread per writer, `amend` replacing the file, and where free-threaded
  Python stands.
