<!-- area: cohorts -->
### Fixed

- **`Entry.from_json` reads an entry as the engine does.** It coerced a
  document its own way, so `"deidentified": "false"` made a de-identified
  member, a string size or mtime was kept, and `save` wrote what it coerced;
  `Manifest.load` was not affected.
