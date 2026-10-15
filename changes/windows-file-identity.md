<!-- area: files -->
### Performance

- **Windows remembers a file it has checked.** Opening a sample checks that it
  reads no bytes from outside itself, and POSIX remembered an unchanged file
  by its identity, so a training loop re-opening the same files paid for the
  check once. Windows had no identity to remember a file by and walked every
  file on every open. A file there is now known by its volume, its 128-bit
  file id, its size, and its last-write and change times, read from the handle
  that opened it; a file changed since it was checked is checked again.
