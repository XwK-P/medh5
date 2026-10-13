## comp workstream: final report (branch `w202/comp`, base `a7351b6`; commits a502f7e, ffbf3d4, 49a9258, 0517f52, b6c3bd7)

### Items (all 13 DONE)
1. W0: `FeatureCache::get` checks the stored kind and shape against the declared dtype and shape before `data::read`; mismatch is T402, nothing read (`companion/cache.rs`).
2. With a task, the entry-URI base defaults to the task manifest's directory (validator callers, binding, `medh5/cache.py`, CLI).
3. A row whose preflight is `error`: one T404 per entry; T406, cutoff, fingerprint and pin checks skipped (`view.rs` `Admitted::in_error`, `cache.rs`).
4. `CacheWriter::add` schema-checks entries (T401), refuses duplicate ids/lookup keys; opening a cache refuses duplicate ids and lookups (T401); `CacheEntry::from_json` strict.
5. `build_document_cache` raises T302 when `content_id` differs from the pin; writer aborted, no file.
6. Missing cache path → `MEDH5FileError` (CLI exit 1); RFC 8089 `file:` URIs (`localhost`, percent-escapes, drives, UNC; `source.rs`); `TaskManifest::save` and `task reconcile` atomic, Python `TaskManifest.save` too; non-integer `cutoff_us` refused (T401).
7. Docs only: `min_follow_up_us` default in task-cache-1 §5, `TargetSpec` and Python `Target` docstrings.
8. `CacheWriter.__exit__` aborts on exception, else commits only if still open.
9. `SourceRef` default id `parent/stem:key` (sanitised, ≤128 chars); `Packed` negative indices and `IndexError`.
10. Identical results: `add_row` indexes event ids by position; `TaskManifest` caches fingerprints until changed, row fingerprints in one pass (`_core.task_row_fingerprints`); `len(FeatureCache)` via `entry_count()`.
11. Preflight `label_annotations` keeps annotations that label or examined one of the slot's classes (matches the torch loader).
12. Lease sentence in task-cache-1 §6.
13. task-cache-1 errata (§5 below); no changes section exists.
Localized edits outside own files: `h5/file.rs` new `pub fn write_atomic` (27 lines); `medh5-cli/src/clinical.rs` cache validate base default; `medh5-cli/src/app.rs` one `--base` help string; `docs/reference/cli.md` one sentence; `hooks/mkdocs_hooks.py` one marker line removed; `_core.pyi` stubs.

### Tests
Rust (`crates/medh5/tests/companion.rs`): `s8_a_payload_is_judged_by_its_declaration_before_it_is_read`, `s7_1_the_writer_refuses_what_its_reader_rejects`, `s7_1_a_manifest_with_two_entries_for_one_lookup_is_corrupt`, `s8_an_entry_of_a_row_in_error_is_stale_not_inadmissible`, `s8_a_path_naming_no_cache_is_refused_not_corrupt`, `s3_1_a_saved_manifest_replaces_the_file_whole`; `s8_caches_are_rejected_when_stale_corrupt_or_inadmissible` restructured; unit `s2_a_file_uri_names_the_path_it_was_made_from`.
Python (`tests/tools/test_task_cache.py`): 15 new tests incl. `test_S9_the_code_table_is_the_engines` (inline §9 table == `CODES`).
Run: new Rust tests fail on the old library (3); companion 16 passed; clippy -p medh5 -p medh5-cli -p medh5-python clean; test_task_cache 100, test_clinical_training 43, test_typing 34, docs_examples 232, docs_python task-cache subset 16 passed; ruff, mypy, mkdocs --strict clean.

### CHANGELOG drafts
**Security**
- **A hostile feature cache can no longer make `FeatureCache.get` allocate up to physical memory.** An entry's stored dtype and shape are checked against the cache's declaration before its payload is read; a mismatch is T402.

**Fixed**
- **A cache stored away from its task now validates against it.** With a task, entries' relative source URIs resolve against the task manifest's directory by default (`validate_cache`, `medh5 cache validate --base`); before, every entry was reported stale.
- **An entry of a row whose preflight is in error is reported once, as T404**, not T406 as if what the row admits were known.
- **`file:` URIs resolve as RFC 8089 defines them:** `localhost`, percent-escapes and Windows drive letters; before, the prefix was only stripped.
- **Saving a task manifest replaces the file whole** (`TaskManifest.save`, `medh5 task reconcile`).
- **`with CacheWriter(...)` allows `commit()` or `abort()` inside the block;** it commits only a writer still open, and aborts on an exception.
- **`Packed` columns index like Python sequences** (negative indices, `IndexError`); `Packed[-1]` returned `""`.
- **A `SourceRef` built without an id gets a valid one that tells files apart** (it includes the parent directory).

**Behaviour changes**
- **A cache manifest that lists an entry twice, or two entries for one event version or row, is refused (T401);** duplicates could hide a wrong checksum. `CacheWriter` refuses them too, and entries that fail the cache schema.
- **A non-integer `cutoff_us` in a cache entry is refused (T401)** instead of read as absent.
- **Validating a cache path that does not exist raises `MEDH5FileError`** instead of reporting T401; an existing non-cache file is still T401.
- **`build_document_cache` refuses a source whose content changed since its pin (T302)** instead of encoding the new content under the old pin.
- **Preflight `label_annotations` lists only annotations that label or examined one of the slot's classes**, as the loader does.
- **`TaskManifest.validate()` returns a schema failure as one T101 finding** instead of raising, as `medh5 task validate` reports it.

**Performance**
- **`CacheWriter.add_row` no longer builds an `Event` per selected event.**
- **`TaskManifest` fingerprints are computed once per change**, row fingerprints in one pass.
- **`len(FeatureCache)` counts entries without building them.**

**Documentation**
- **task-cache-1 errata**; §9 carries the code table inline.
- **The default `min_follow_up_us` (equal to the horizon) is documented:** only an observation at exactly `c + h` can give a negative label; the default changes in 2.1.

### Handoffs
- docs: §9 has the table inline, so the `<!--@companion-codes-->` marker was removed from `_MARKERS`; the companion renderer (`_companion_codes_markdown`, `COMPANION_SOURCE`, regexes, docstring mention) is now unused: retire it or render the table on a reference page.
- sec: review `medh5::h5::file::write_atomic` (same steps as `atomic_rewrite`).
- cli: `cache validate --base` help string in `app.rs` and the default base in `clinical.rs` changed.
- dataset, legacy: `dataset/manifest.rs:396` and `convert/legacy.rs:921` still write text non-atomically; could use `write_atomic`.
- spec: task-cache-1 §6 cites 1.1 §9.1 for "tie group = connected component"; 1.1 §9.1 needs the matching erratum (spec did it).

### Spec wording (made in task-cache-1)
§2 RFC 8089; §3.3 "MUST be identical"; §3.5 patch placement (start `c − ⌊p/2⌋` shifted to 0 or `n − p`; an axis no longer than the patch read whole and padded `⌊(p − n)/2⌋` before, rest after; padding never valid); §5 bound tests (positive `lo > c` and `hi ≤ c + h`; negative `lo ≥ c + min_follow_up`) and the default note; §6 "that label or examined one of them", connected components, the lease sentence (T302); §7.1 entry URI base and uniqueness (T401); §8 missing path refused, row in error is one T404; §9 inline 20-row table with anchors.

### Decisions
- Missing cache path: file error (no code means "no cache").
- Writer's duplicate refusal has no code; the reader reports T401.
- New API: public Rust `CacheEntry::from_json` (binding needs it) and `h5::file::write_atomic`; private `_core.source_resolve`, `_core.task_row_fingerprints`, `FeatureCacheHandle.entry_count`.
- Default source ids readable `parent/stem:key`, not hashes (same parent name + stem under different grandparents can still collide).
