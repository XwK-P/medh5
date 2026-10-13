## docs workstream: final report (branch `w202/docs`, 14 commits `3eb210c..f291a6c` on `a7351b6`)

### Items (all DONE)
1. "Check it" step: new samples checked at `--level integrity`, strict kept for a dataset being published (it promotes W903/W912). README.md, crates/medh5-cli/README.md, docs/index.md, profiles-and-levels.md, segmentation.md, detection.md, validate.md (a promoted warning still prints as `WARNING`).
2. Tracking legend (F15): longitudinal.md, python-api.md (coverage includes the annotation's ignore region).
3. `transform_between(tp, tp)`: longitudinal.md, registration.md, python-api.md describe it as it behaves (first frame pair in grid-id order); advise resolving between frames.
4. instances→dense needs `--drop-identity`/`drop_identity=True` (segmentation.md).
5. partial-labels: read-back matches the write; manifest coverage keyed as `split.paths()` names a sample (`path::key` for a member).
6. performance.md: `used_index` is a list after collate; the budget is the whole shared L3; a patch larger than the budget makes a larger chunk (8 MiB for 96³ f32); the CHANGELOG 2.0.0 figures came from an earlier 2.0 build (CHANGELOG not edited).
7. Clinical guides: setup command; amend copies (in-place amend broke the training guide's pin); SelectionPolicy with `context_us` and plans; the encoder contract (`encode`, `dim`, `header()`) with a runnable encoder; documented signatures match.
8. design-rationale: no C ABI promise, no DICOM round trip promise (line 243 left to clin).
9. Reference: kinds count; commit = fsync, rename, directory fsync, and link behaviour; after close, views and file reads raise while cached grids/headers/frames answer; a damaged chunk raises plain `OSError`; writer refusals; rust.md and crates/medh5/README.md list every public module (annotations §6–§9).
10. Site: docs/requirements.txt pins `pymdown-extensions==12.2`, `Markdown==3.11`; mkdocs.yml drops snippets/inlinehilite, excludes `examples/*.medh5`, `site_url: !ENV [READTHEDOCS_CANONICAL_URL, https://medh5.readthedocs.io/]`.
11. Front door: README.md and docs/index.md link 1.0, 1.1 and task-cache-1; the hero snippet opens `subject.medh5` (fixture added); first-sample.md uses `labelmap(priority=["lesion"])`.
12. W9 SystemExit: `_execute` passes only `SystemExit` with code None/0.
13. STALE_CLAIMS: 7 new entries; new `STALE_PHRASES` (8 entries) because STALE_CLAIMS exempts paragraphs containing "not"/"rather than".

Coordinator handoffs applied: refusals.md (conv1 + conv2 rules, the F11 section, E414 in the table); python-api.md (Windows amend, symlinks, hard links; sec's sticky-directory refusal); install.md troubleshooting for "filter plugins disabled" (`HDF5_PLUGIN_PATH`, `MEDH5_ALLOW_PLUGINS`), checked against the build.

### Tests
Added in test_docs_examples.py: the STALE_CLAIMS entries; `STALE_PHRASES` + `test_corrected_phrases_stay_corrected`; `test_transform_between_over_visits_is_described_as_it_behaves`; `test_documented_signatures_are_the_real_ones`; `test_the_front_pages_link_every_normative_document`. In test_docs_python.py: `test_a_gate_that_closes_fails`, the `subject.medh5` fixture, "patches" removed from `TORCH_NAMES`.
Run: docs tests 415 passed; packaging/conformance/bytes/transforms/task_cache 325 passed; `mkdocs build --strict` clean; ruff clean. Not run: cargo, full suite, mypy (two pre-existing mypy errors in test_docs_python.py ~122/691, outside the gate).

### CHANGELOG drafts (Documentation)
- **New samples are checked at `--level integrity`.** `strict` promotes W903 and W912, so it failed every fresh sample; the guides now keep it for a dataset being published.
- **Tracking's states include the ignore region.** `resolved` means examined at the visit, outside any ignore region, and not found.
- **`transform_between` between visits is described as it behaves.** It answers for the first frame pair in grid-id order, so on visits sharing a frame, resolve between frames.
- **The segmentation guide names both lossy transcodes.** Instances to a dense encoding drops `instance_id` and needs `--drop-identity`.
- **Partial labels agrees with its own example**, and keys samples as `split.paths()` names them.
- **The performance guide reads `used_index` as a batch holds it**, states the chunk budget the optimiser actually uses, and which build the 2.0.0 changelog figures come from.
- **The clinical guides run as written:** the setup command, copies to amend, selection policies and the encoder contract, with documented signatures tested against the code.
- **The design rationale no longer promises a C interface or DICOM round trips.**
- **Reference corrections:** the annotation kinds count, how commit replaces a file and follows links, the Rust modules at §6–§9, what keeps answering after close, and which errors are `MEDH5Error`.
- **The refusals page covers the converters' 2.0.2 rules**, and the install page explains "filter plugins disabled".
- **The front pages link 1.1 and the task-and-cache contract.**
- **The docs build pins pymdown-extensions and Markdown, and its canonical URL carries the version.**
- **A documented gate that exits non-zero now fails the docs tests.**

### Handoffs
- geom: storage.md:101-102 says L3 "per-core" and chunks "held between 512 KiB and 4 MiB"; chunking.rs reads the whole shared L3 and does not clamp. storage.md:171 says "`os.replace` it into position". If geom lands per-sharing-CPU sizing or a clamp, performance.md:83-89 needs: "80 % of one core's share of the L3 (its size over the CPUs sharing it)", and with a clamp drop "8 MiB for a 96³ `float32` patch".
- legacy: migrate-0x.md:92 still `--level strict`; then widen the STALE pattern `medh5 validate case(_0001)?\.medh5 --level strict`.
- cli: cli.md items from dataset (stats exit, class keys; index symlinks, case, `._*`), scrub (link names, attributes, string datasets; different/zero shift refused; flags need `--apply`), curation (track legend), conv2 (F11 sentence), sec (fix/amend through links).
- conv2 (export.md): conv1's examined-only SEG export, RTSTRUCT names/colours and the crossing warning.
- scrub, curation, geom: sec's link and shared-directory wording for deidentify.md (scrub --apply), curation.md (amend), storage.md (recompress).
- ci/coordinator: CLAUDE.md:85 says `annotations` (§5–§9); should be §6–§9.
- CHANGELOG discrepancy: 2.0.0's figures (3.5 ms labels, 3–3.5 ms to first patch, 0.19 ms metadata, 600 patches/s with 4 workers) came from an earlier 2.0 build; performance.md now says so.
