<!-- area: curation -->
### Added

- **Agreement compares two vocabularies.** `compare`, `compare_voxel` and
  `compare_instances` take `map_classes`: `"id"` (the default), `"key"` to
  score classes of one key across two inline label sets (a label set by
  reference is refused, and one key with two codes of one ontology is
  `C102`), or an explicit one-to-one map of `a`'s classes to `b`'s. A class
  `b` lacks is listed in `skipped` with its reason, never scored 0, and each
  result's `b_class_ids` names the class of `b` every class was scored
  against. `medh5 agree` takes `--map-classes id|key` and `--map-class A=B`.
  In Rust, `curation::agreement::compare_with` takes `CompareOptions` and
  returns a `Compared`.
- **Objects pair the way you ask.** `compare` and `compare_instances` take
  `match_by`: `"auto"` (the default), `"iou"` to pair by where the objects
  are even where both carry ids, or `"instance_id"`, which refuses two
  samples because an id names an object only within its own sample (§7.4).
  `medh5 agree` takes `--match-by auto|iou|instance-id`.

### Fixed

- **A comparison across two samples no longer records the other file's path
  as its own.** Its `against` was `annotations/<the other annotation>`, which
  resolves inside the recording sample. It is now `None`, and `to_record()`
  refuses it unless `against=` names an annotation of the recording sample,
  `annotations/<id>`; in Rust, `to_record_against`. §11.2 now says whose
  annotation `against` names.
- **`skipped` names each class once.** A class each side calls by the same
  key was listed twice.
