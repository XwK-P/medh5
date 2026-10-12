# Workstream `comp`: task-and-cache companion (W4; W0 FeatureCache.get; F12 preflight; F21 doc; task-cache-1 errata)

Worktree: `SCR/wt/comp` (branch `w202/comp`). It is created from the integration branch, so it already contains the merged wave-1 work, including the torch agent's F12, F13 and F21 loader changes.

## Files you own
- `crates/medh5/src/companion/**`: `cache.rs`, `view.rs`, `task.rs`, `source.rs`, `mod.rs` (including its `CODES` table).
- `medh5/{cache,task}.py`.
- `crates/medh5-python/src/preflight.rs`, and the cache and task bindings in `crates/medh5-python/src/clinical.rs`. The clin agent owns the clinical-selection parts of that file and runs at the same time, so keep your hunks to the cache and task functions.
- Localized: `crates/medh5-cli/src/clinical.rs:~435-447`, the cache-URI base. The clin agent owns the rest of that file.
- `docs/spec/task-cache-1.md`.
- Tests: `tests/tools/test_task_cache.py`, `crates/medh5/tests/companion.rs`.

## Items (quoted from the audit; verify each against the code first)
1. **W0:** `FeatureCache.get` reads the whole entry before comparing its shape and dtype with the cache header (`companion/cache.rs:470-505`), so a hostile cache can make every `get()` allocate up to physical memory. Check first, then read.
2. **W4:** a cache stored away from the task's directory validates as entirely stale. Resolve cache entry URIs against the task's base when a task is given, and state the base in task-cache-1 §7.1 (`companion/cache.rs:648`; `medh5/cache.py:384-407`; the CLI's `clinical.rs:435-447`).
3. **W4:** don't add T406 for rows whose preflight is `error`; report one T404 (`view.rs:1238-1241`; `cache.rs:770-791`).
4. **W4:** `CacheWriter` commits manifests its own reader rejects (T401).
   - Schema-check entries in `CacheWriter::add`/`commit` (`cache.rs:272-365`).
   - Refuse duplicate `entry_id`s, event keys and row ids at open (`cache.rs:427-442`); today duplicates hide a wrong checksum.
   - Add the rule to §7.1.
5. **W4:** `build_document_cache` encodes a changed source under the old pin. Check `content_id` against the pin before encoding (`medh5/cache.py:477-500`).
6. **W4, smaller:**
   - a missing cache file is reported as "corrupt" T401; report it as missing, with the right existing code;
   - `file://` URIs are prefix-stripped only. Handle `localhost`, percent-escapes and Windows drives;
   - `TaskManifest::save` and `task reconcile` write non-atomically over the input. Write a temp file, then replace it;
   - the binding reads a non-integer `cutoff_us` as absent. Refuse it.
7. **W4:** document now that the default `min_follow_up_us = horizon_us` makes negative labels practically unreachable (`task.rs:166-169`; `view.rs:861, 892-900`). This is docs only in 2.0.2; the default changes in 2.1. Say it in task-cache-1 and the docstring.
8. **W2:** `CacheWriter.__exit__` commits unconditionally, so `commit()`/`abort()` inside `with` raises (`medh5/cache.py:240-249`). Commit only if still open and no exception occurred; abort on an exception.
9. **W2, small:**
   - `SourceRef()` defaults give an invalid `""` or colliding stem ids (`medh5/task.py:106`; `companion/source.rs:22`);
   - `Packed[-1]` returns `""` (`crates/medh5-python/src/preflight.rs:47`). Support negative indices, or raise `IndexError`, matching Python sequence semantics.
10. **W6:**
    - `CacheWriter.add_row` builds an `Event` per selected event;
    - `TaskManifest` fingerprints re-parse on every access;
    - `len(FeatureCache)` builds every entry.
    - Fix each with identical results.
11. **F12, preflight side** (optional, recommended): filter `label_annotations` by class in preflight (`companion/view.rs:662-664`), so the `RowView` agrees with the loader. The loader, in `medh5/torch/clinical.py` in your worktree, now skips the ignore regions of annotations that neither label nor examined a requested class. Keep the preflight report consistent with that.
12. **F21, doc:** in task-cache-1 §6, state that a lease in progress finishes on the bytes it opened, and that every new lease re-checks the source (refusal T302 once the source no longer carries the pin). Read `medh5/torch/handles.py` in your worktree for what the torch agent implemented.
13. **W8, task-cache-1 errata:**
    - tie groups are connected components of the overlap relation;
    - §3.5 patch placement around the centre: start = c − ⌊p/2⌋, shifted inside the extent, padded only when an axis is shorter. Verify against the code;
    - §5: the negative-label bound test;
    - §7.1: the entry URI base and uniqueness (items 2 and 4);
    - §9 needs a real code table, not just the placeholder. The docs hook renders the companion `CODES` table on the site; check `hooks/mkdocs_hooks.py`. If raw readers (GitHub, the sdist) see only a placeholder, decide whether to inline the table, and keep the hook's test passing.
    - Record the changes the way task-cache-1 records its revisions, if it has a changes section.

## Notes
- Caches pin `content_id` (CLAUDE.md): a changed source makes an entry *stale* (T403), told apart from a cache whose own bytes are wrong (T401/T402).
- Rows index, never copy. Report a whole cohort a subject at a time.
- Tests to run:
  - `wcargo comp test -p medh5 --lib companion::`, plus `--test companion`;
  - `wext comp`, then `wpy comp -m pytest tests/tools/test_task_cache.py tests/integrations/test_clinical_training.py -q`.
