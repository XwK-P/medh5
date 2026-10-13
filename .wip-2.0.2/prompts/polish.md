# Workstream `polish`: the small handoffs of merged workstreams, and the last GIL bindings

Worktree: `SCR/wt/polish` (branch `w202/polish`), created from the integration branch at `353e900`, which holds every workstream except `geom` and `ci`. Read `SCR/AGENT-BRIEF.md` first; its rules apply, with this addition: **the 2.0.2 release keeps every public Rust signature of 2.0.1** (the user decided). Never change an existing `pub` item's signature in `crates/medh5`; a frontend-only helper is `#[doc(hidden)] pub` (see `crates/medh5/src/curation/agreement.rs` `box_iou_f64_checked`), a crate-only one `pub(crate)`.

The merged reports are in `.wip-2.0.2/reports/*.md` in your worktree; each item below names its source. Do not edit `.wip-2.0.2/`, `CHANGELOG.md`, `CLAUDE.md`, `CONTRIBUTING.md` or `.github/` (ci owns those), nor the files `geom` owns (`crates/medh5/src/{geometry,transforms}/**`, `crates/medh5/src/storage/{chunking,codecs,recompress,mod}.rs`, `crates/medh5-python/src/{geometry,transforms,storage}.rs`, `medh5/{geometry,transforms,storage}.py`, `docs/reference/storage.md`, `docs/guides/performance.md`) until the coordinator tells you geom has merged.

## Items (each small; verify against the code first; a regression test for every code fix)
### Code
1. `medh5 task reconcile -o` writes the manifest over whatever `-o` names, including a source sample (`crates/medh5-cli/src/clinical.rs` `task_reconcile` → `TaskManifest::save`). Refuse an `-o` that is the same file as a resolved source, by resolved path and same-file identity, before writing (the F11 pattern: see `clinical export` in the same file, and `reports/clin.md`, `reports/conv2.md`).
2. `medh5/io/nifti.py` (about line 1230) writes input *paths* into the import activity's `inputs`; name only the file, as `migrate` now does (W1; `reports/legacy.md`). Check the other converters (`medh5/io/*.py`) for the same leak and fix any.
3. The engine's `compare`/`compare_instances` (agreement) accept an out-of-range IoU threshold from the Python API; only the CLI refuses it (`reports/cli.md`). Refuse NaN, ≤ 0 and > 1 in the Python API too (in the binding or the Python wrapper if the Rust signature can't change).
4. `medh5.bench` with a negative `repeats` or `patch` raises PyO3's `OverflowError`; raise `MEDH5ValidationError`, as `0` now does (`reports/cli.md`).
5. `crates/medh5/src/dataset/manifest.rs` (about line 396) and `crates/medh5/src/convert/legacy.rs` (about line 921) write text over their target non-atomically; use `crate::h5::file::write_atomic` (`reports/comp.md`).
6. `medh5/io/_common.py` `sanitize_ascii_stem` docstring says `sanitize_stem` keeps non-ASCII letters; it now folds to ASCII (`reports/legacy.md`).
7. Optional: `build_label_set` skips unreadable inputs silently under `migrate --write-labels` (`reports/legacy.md`); report them if the fix is small.
8. **GIL (W6)** in `crates/medh5-python/src/clinical.rs`: `ClinicalHandle.records`, `CacheWriterHandle.add`, `commit` (`reports/pyapi.md` §4). Follow `reports/followups.md`'s approach (`py.detach`, no Python object inside the closure, identical results).
9. **Later, when told:** the same GIL pass over `storage.rs` (`build_index`, `occupancy`), `transforms.rs` (`inside_extent`, `linear_sample`, `cubic_sample`, `sample_field`, `encode_displacement`, `to_world_vectors`, `jacobian_determinant`, `folding_fraction`) and `geometry.rs` (`index_to_world`, `world_to_index`, the `Grid` methods for large point sets). Merge the integration branch into your branch first (`git merge claude/relaxed-volta-ua1vih`).

### Docs (`docs/**`, READMEs, `hooks/mkdocs_hooks.py`)
10. `docs/guides/export.md`: conv1's wording on the examined-only SEG export, and on RTSTRUCT names, colours and the crossing warning (`reports/conv1.md`, `reports/docs.md` handoffs).
11. `docs/guides/deidentify.md` (`scrub --apply`) and `docs/reference/curation.md` (amend): sec's wording on rewrites through links and the refusal of a link in a sticky, world-writable directory another user owns (`reports/sec.md`; `docs/reference/python-api.md` already has it — match it).
12. `docs/reference/python-api.md` at `c.select(...)`: `MEDH5VersionError` for a later minor's projection (`reports/clin.md`).
13. `README.md`, `docs/install.md` and the `medh5/cli.py` docstring say `python3` only; Windows uses `python` (check `crates/medh5-cli` `Host`) (`reports/cli.md`).
14. `tests/project/test_docs_examples.py`: widen the stale-claim pattern `medh5 validate case(_0001)?\.medh5 --level strict` so it also covers `new/*.medh5` (`reports/docs.md`, `reports/legacy.md`).
15. `hooks/mkdocs_hooks.py`: the companion-codes renderer (`_companion_codes_markdown`, `COMPANION_SOURCE`, its regexes, the docstring mention) is unused now that task-cache-1 §9 carries the table inline; retire it, keeping the hook's tests and `mkdocs build --strict` passing (`reports/comp.md`).

### Spec (normative: `docs/spec/medh5-1.0.md`, `medh5-1.1.md`)
Record each normative change as the spec merge did: an Appendix C row in 1.0 (each "Found by the audit of 2.0.1", plus "; enforced from 2.0.2" where the code now enforces it), or a 1.1 Appendix A row, and update any count sentence. `tests/format/test_conformance.py` (`-k "appendix or Stated or SpecSync"`) must pass.
16. 1.0 Appendix B (`reports/legacy.md`): the root `shape` of 0.x up to 0.5; `label` without `label_name` is its own class; a review status `pending`/`flagged` becomes `submitted` (a flag kept as an issue); the fallback identity is `study:<stem>`.
17. 1.0 §2.1, the `medh5_version` row (`reports/followups.md`): "Its major is the text before the first `.`, compared exactly, and its minor a run of decimal digits; a minor that is not one is read as a higher minor (W913)."
18. 1.1 §9.1 step 5 (`reports/clin.md` "Spec and doc wording").
19. **Later, with item 9:** 1.0 §4.3 and §14.1 wording from the geom report (the coordinator will paste it).

## Tests
- `SCR/bin/wext polish` after Rust changes; targeted pytest for each touched module (`test_cli.py -k reconcile`, `tests/integrations/test_nifti*.py`/converters, agreement tests, `test_bench.py`, `test_dataset.py`, `test_migrate.py`, `test_clinical*.py`, `test_task_cache.py`), `tests/project/test_docs_examples.py`, `tests/project/test_docs_python.py`, `tests/format/test_conformance.py`, `mkdocs build --strict` (site to `SCR/tmp/polish/`).
- `wcargo polish fmt --all`; clippy on the crates you touched, once at the end; `ruff`, `mypy medh5`.

Commit after each item (or a few small ones together). Final report in the brief's format, with CHANGELOG drafts.
