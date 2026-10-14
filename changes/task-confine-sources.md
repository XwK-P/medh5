<!-- area: task-cache -->
### Security

- **A task's sources can be confined to one directory.** Whoever wrote a
  manifest chose every path its runner opened, and `T301` telling an absent
  path from an unreadable one answered whether a path exists. `--source-root
  DIR` on `task preflight`, `task reconcile` and `cache validate`, and
  `confine=True` (the manifest's directory) or `confine=DIR` in Python, refuse
  a URI outside the directory, its `.` and `..` resolved as text, without
  examining the path (`T301`, naming the URI only). Lexically, so a symbolic
  link inside the directory still works; in Rust, `SourceAccess` and the
  `*_in` functions taking it.

### Deprecated

- **A manifest's sources outside its directory are refused by default in
  3.0.** Until then they open as before; Python warns (`FutureWarning`) when a
  manifest loaded from a file names one, and the command line says so on
  stderr. Pass `confine=False` or `--any-source-path` to keep reading them, or
  name the directory to confine to.
