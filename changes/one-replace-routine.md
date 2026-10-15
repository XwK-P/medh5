<!-- area: files -->
### Behaviour changes

- **Files medh5 writes keep a symbolic link at their path, and the file's
  mode.** A conversion report (`convert … --report`) and a NIfTI sidecar
  replaced a link at their path with a new file, wrote a `0o600` file back as
  `0o644`, failed on a read-only file on Windows, and wrote `\r\n` for every
  `\n` there. `clinical export -o`, `migrate --report`, `dataset split -o` and
  `dataset stats -o` wrote in place: through a link, and not atomically, so an
  interrupted run left half a file. They all now go through the engine's
  replace, as every amendment already did (§14.4): the link is kept and the file
  it names replaced, its mode is kept, a read-only file stays read-only, the
  bytes are written as given, and an error names the file rather than the
  temporary beside it.
