# Workstream `docs`: documentation corrections (W7) and the docs tests (part of W9)

Worktree: `SCR/wt/docs` (branch `w202/docs`). This workstream is **Python and docs only**. Run `SCR/bin/wext-prebuilt docs` once.

The docs toolchain: `wsh docs 'pip install -r docs/requirements.txt'` may already be satisfied; check with `wsh docs 'mkdocs --version'`. If the network refuses, report it and verify with what is installed.

## Files you own
All of `docs/**`, **except** these, which belong to other agents:
- `docs/spec/**`: spec, valid and comp;
- `docs/guides/deidentify.md`: scrub;
- `docs/guides/import-dicom.md`: conv1;
- `docs/guides/import-nifti.md` and `docs/guides/export.md`: conv2;
- `docs/guides/migrate-0x.md`: legacy;
- `docs/reference/annotations.md`: annot;
- `docs/reference/curation.md`: curation;
- `docs/reference/torch.md`: torch;
- `docs/reference/cli.md`: cli;
- `docs/reference/converters.md`: conv1 and conv2;
- `docs/reference/storage.md`: geom;
- `docs/guides/cohorts.md`: dataset.

`docs/guides/longitudinal.md` is yours, except its `PairedPatchDataset` example at lines 163-176, which the torch agent fixes.

You also own:
- `README.md`, `crates/medh5/README.md` and the other crate READMEs;
- `mkdocs.yml`, `docs/requirements.txt` and `hooks/mkdocs_hooks.py`;
- `tests/project/test_docs_python.py` and `tests/project/test_docs_examples.py` (`STALE_CLAIMS`).

## Items (quoted from the audit; verify each against the docs and code first)
1. **The "check it" step:** `--level strict` fails on every freshly converted file, because W903 and W912 are promoted. Recommend `--level integrity`, or explain. Your pages: `segmentation.md:92`, `detection.md:75`, `README.md:188`, and any other page you own that says the same (grep `level strict`). The converter pages are other agents'.
2. **`longitudinal.md:124-125`:** the tracking legend. The curation agent changes tracking so that `resolved` means "examined at this visit, outside any ignore region, and not found". An ignore region over the object's location makes it `unexamined`, and the same rule applies to `new` at the baseline. Update the text to that meaning.
3. **`transform_between(tp, tp)` is described wrongly:** it returns `None` at the first shared-frame pair, depending on grid order (`longitudinal.md:221-223`; `registration.md:255-257`; code `crates/medh5/src/sample/reader.rs:756-768`). Read the code and describe what it actually does.
4. **`segmentation.md:76-87`:** `instances` → dense is refused without `--drop-identity`.
5. **`partial-labels.md`** contradicts its own example; fix it.
6. **`performance.md:35-52`:** `used_index` is a list after collate, and "L3 per core" is wrong. The chunk optimiser reads L3 per sharing CPU; check `crates/medh5/src/storage/chunking.rs`, but leave the code to the geom agent. Also reconcile the CHANGELOG 2.0.0 performance numbers with `performance.md`. **Don't edit CHANGELOG.md:** fix `performance.md` if it is wrong, or report the discrepancy.
7. **Clinical guides** (`clinical.md`, `clinical-training.md`):
   - state the setup command (`python docs/examples/clinical_longitudinal.py cohort`);
   - have `clinical.md` amend a copy, since today it damages the cohort the next guide uses;
   - show a `SelectionPolicy` with `context_us`/`plans`;
   - document the `TextEncoder` contract (`encode`, `dim`, `header`);
   - correct `Link.between` (keyword-only, `target_annotation_id`).
   - Verify every claim against `medh5/clinical.py`, `medh5/task.py` and the bindings.
8. **`design-rationale.md`:**
   - `:213` promises a "C-compatible native library", but there is no C ABI;
   - `:258` promises "lossless round trips to DICOM", but there is no DICOM image exporter;
   - fix both. Leave `:243` (asserting events) to the clin agent.
9. **Reference:**
   - `reference/index.md:20` miscounts the annotation kinds;
   - commit is described as `os.replace`; it is fsync, rename, then fsync of the directory. Find where it is described;
   - `rust.md` and the crate README omit public modules, and put annotations at §5–§9 instead of §6–§9. A test compares the Rust block in `rust.md` with the crate README's doctest, so keep them identical; the crate README's Rust example is a doctest;
   - the "views raise after close" claim is only partly true: check what raises and say exactly;
   - `MEDH5Error` is called "base of every error", but HDF5 I/O errors are plain `OSError`; say so.
10. **Site:**
    - pin `pymdown-extensions` and `Markdown` in `docs/requirements.txt`; they supply the slugs the `--strict` anchor check relies on, so pin them to the versions installed here or the latest known-good;
    - drop the unused `pymdownx.snippets`/`inlinehilite`;
    - exclude `examples/*.medh5` from the docs build;
    - set `site_url` from `READTHEDOCS_CANONICAL_URL`, with the env syntax mkdocs supports, e.g. `!ENV [READTHEDOCS_CANONICAL_URL, <current url>]`.
11. **Front door:**
    - link the 1.1 spec and task-cache-1 beside 1.0, in `README.md` and `docs/index.md`;
    - the tutorial's `labelmap()` warning is unmentioned;
    - the hero snippet's names don't match the page's sample.
12. **W9:** `test_docs_python.py` passes any `SystemExit`, including a failing gate. Make it pass only `SystemExit(0)`/`None`, or whatever the documented examples legitimately raise.
13. **`STALE_CLAIMS`:** add each wrong statement you remove to `STALE_CLAIMS` in `tests/project/test_docs_examples.py`, as the file's convention does, so it cannot come back.

## Notes
- Never write `---` as a dash; write `—`. Python blocks in pages are executed: run `wpy docs -m pytest tests/project/test_docs_python.py tests/project/test_docs_examples.py -q` and keep them green. Note that some blocks depend on code other agents are changing; keep your examples to current, stable APIs.
- Build: `wsh docs 'mkdocs build --strict -d SCR/tmp/docs/site'`. It must pass. The `SCR/tmp/docs/site` output directory keeps the build out of the worktree.
- **Don't invent features.** Each correction must match the code as it is, or as the audit says the fix makes it.
