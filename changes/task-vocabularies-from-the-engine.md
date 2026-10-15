<!-- area: task-cache -->
### Deprecated

- **`medh5.task.STATUSES` and `LABELS`** are `ROW_STATUSES` and
  `TARGET_STATUSES` by other names, and `task.STATUSES` shared its name with
  `clinical.STATUSES`, a different vocabulary: reaching either now warns,
  and 3.0 removes both. Every vocabulary of the task and cache modules is
  now the engine's table (`ROI`, `CENSORING`, `ROIS`, `cache.LEVELS`,
  `cache.SUFFIX`, `clinical.ASSESSMENT_SYSTEM`, `LESION_PRESENCE`), and a
  status the engine wrote outside its table is refused rather than read as
  the table's last word. `build_document_cache` files documents by the
  engine's ownership rule, the one `validate_cache` holds entries to.
