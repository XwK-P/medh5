# W21-16 Documentation

| | |
|---|---|
| Branch | `w21/documentation-a` (PRs 1 and 3, lane A), `w21/documentation-f` (PRs 2, 4 and 5, lane F) |
| Size | M overall; each pull request S–M |
| Lands after | W21-00 PR 4, PR 5; PR 2 after W21-09 PR 2 and W21-13 PR 3; PR 4 after W21-11 PR 4; PR 5 after W21-15 PR 2 |
| Blocks | nothing |
| Owns | `hooks/mkdocs_hooks.py` (schemas), `docs/reference/{api-index.md,*-schema.md}`, `docs/guides/{collections,review-and-provenance,label-sets,import-seg-rtstruct,conformance}.md`, `docs/tutorials/clinical-task.md`, `docs/explanation/concurrency.md`, `.github/scripts/api_index.py`, `tests/project/{test_api_documented.py,undocumented_api.txt}`, the sequential mode of `tests/project/test_docs_python.py`, `SECURITY.md`'s section on hostile files |
| Decisions | none beyond the rules |
| Backport | none |

## Session prompt

> Implement workstream W21-16 (documentation) of medh5 2.1.0 in this repository.
> First read `plans/README.md` and `plans/2.1.0/README.md`, then
> `plans/2.1.0/w21-16-documentation.md` in full. Two sessions share this
> workstream, because its pull requests are independent: lane A takes PRs 1
> and 3 on `w21/documentation-a`, lane F PRs 2, 4 and 5 on
> `w21/documentation-f` (`plans/2.1.0/README.md`, Lanes); you were told your
> lane. Work from the latest `main`; open each pull request when its "Lands
> after" has merged. The docs are tested, not proofread: every `python` block runs
> against a real sample (fewer than half may be illustrative), every documented
> flag is checked against the parser, stated corpus counts are compared with the
> corpus, `---` is refused as a dash (write `—`), and `mkdocs build --strict` must
> pass. Write only what the code does today on `main`; where a page describes
> another workstream's feature, wait for it to merge. Run the local gate from
> `CLAUDE.md`, add fragments, open `W21-16: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 16.1 All four schemas published; 16.2 opening files you did not write | S |
| 2 | 16.3 Five guides | M |
| 3 | 16.4 A reference entry for every public name | M |
| 4 | 16.5 A tutorial from clinical history to a batch, and pages that run in order | M |
| 5 | 16.6 A concurrency page | S |

## Items

### 16.1 All four schemas published (PR 1)

- **State at 2.0.2.** The hook publishes only
  `crates/medh5/data/medh5-sample-1.0.schema.json`, at
  `reference/medh5-sample-1.0.schema.json`, and renders `reference/schema.md`
  through `<!--@schema-->` (`_schema_markdown`). The other three (`medh5-clinical-1`:
  no top-level properties, 11 definitions; `medh5-task-1`: 10 properties, 8
  definitions; `medh5-cache-1`: 9 properties, 6 definitions) are embedded by the
  engine and shipped in the corpus but not on the site. Every `$id` is
  `https://github.com/XwK-P/medh5/schemas/<name>`, which resolves nowhere.
- **Change.** A `SCHEMAS` tuple of `(source, json_target, page)` in the hook:
  publish at `reference/schemas/<name>.schema.json`, keep the existing sample path
  too (a JSON file cannot redirect), render pages with `<!--@schema NAME-->`
  (`_schema_markdown(root, name)` taught to render definitions alone). New pages
  `docs/reference/{clinical,task,cache}-schema.md`, in the nav, linked from
  `task-cache-1.md`. Keep the `$id`s unchanged (an `$id` identifies; third parties
  may register the shipped bytes by it); each page states the published URL.
- **Tests.** `test_every_engine_schema_is_published` (every
  `crates/medh5/data/*.schema.json` is in `SCHEMAS`);
  `test_a_published_schema_is_the_embedded_bytes` (against `schema_text()` from
  `medh5.document`, `medh5.clinical`, `medh5.task`, `medh5.cache`).
- **Changelog.** Documentation.

### 16.2 Opening files you did not write (PR 1)

- **State at 2.0.2.** `SECURITY.md` covers reporting, what is checked, the native
  inventory and advisories, but no threat model for third-party files. 2.0.2's
  hardening (plugins opt-in, the depth limit, the filters' output caps, the
  sticky-directory link refusal, scrub's findings) is in the changelog only.
- **Change.** A `SECURITY.md` section: what `open` and `validate` guarantee on a
  hostile file (no plugin loaded unless asked, bounded depth and decompressed
  size, a refusal rather than a crash, what is not guaranteed: resource use within
  those bounds), and a pointer to W21-11's section on task manifests. Keep it to
  what the code enforces; cite the knobs by their documented names.
- **Changelog.** Documentation.

### 16.3 Five guides (PR 2)

- **State at 2.0.2.** None of these has a guide; what exists is reference
  material: collections (`python-api.md`, `storage.md`, `curation.md`,
  `clinical-training.md`); provenance, quality and agreement
  (`docs/reference/curation.md`); label sets (`docs/reference/annotations.md`; the
  embedded vocabularies `amos22-organs`, `binary-foreground`, `brats-subregions`);
  SEG and RTSTRUCT import (`docs/reference/converters.md`; `docs/guides/export.md`
  covers export only); conformance (a how-to inside the normative
  `docs/spec/conformance.md`, linked only from `task-cache-1.md`).
- **Change.** Five how-to pages under `docs/guides/`, each linked from its
  reference page, every block runnable:

  | Page | Blocks | Needs from the docs workspace fixture |
  |---|---|---|
  | `collections.md` | `pack`, members of `shard.medh5c`, `path::key` in splits, verifying members, shard size | nothing new (`shard.medh5c` holds `case_0001`, `case_0002`) |
  | `review-and-provenance.md` | `medh5.amend` of `case.medh5`: record an activity, record quality, compare two raters, store the agreement | nothing new |
  | `label-sets.md` | `LabelSet`, `LabelClass`: choosing or mapping a vocabulary, resolving C101/C102, `map_classes="key"` (W21-09) | nothing new |
  | `import-seg-rtstruct.md` | `from_dicom_seg(path, sample)`, `from_rtstruct`: frame matching, refusals, same-named ROIs (W21-13) | `seg.dcm` and `plan.dcm`, written by `to_dicom_seg`/`to_rtstruct` from `case.medh5` (the `dicomseg` and `dicom` extras); skipped where highdicom is absent |
  | `conformance.md` | `medh5.conformance.build_corpus`, `load_manifest`, `score` (with W21-03's bounds) | the corpus, built once in the fixture |

  Move the conformance how-to out of the spec page, leave a link behind, and add
  any moved anchor to `REDIRECTS`. Update `docs/guides/index.md` and the nav.
  This pull request lands after W21-09 PR 2 (`map_classes`) and W21-13 PR 3
  (same-named ROIs), which `label-sets.md` and `import-seg-rtstruct.md`
  describe; draft the other three pages meanwhile.
- **Tests.** The existing harness; `TestStatedCounts` for any count; shell blocks
  flag-checked.
- **Changelog.** Documentation.

### 16.4 A reference entry for every public name (PR 3)

- **State at 2.0.2.** 778 names in `__all__`; 470 are named on no page outside the
  spec, 509 not under `docs/reference/`. `docs/reference/python-api.md` is written
  by hand (`test_docs_examples.py` checks the signatures it documents). The docs
  job deliberately installs only `docs/requirements.txt`, so a hook cannot import
  the package.
- **Change.** A generated, committed `docs/reference/api-index.md`, kept current by
  `.github/scripts/api_index.py [--check]` from the installed package (as
  `THIRD_PARTY_NOTICES` is): module → name → kind → the first docstring line → a
  link to the hand-written section where one exists. A ratchet:
  `tests/project/undocumented_api.txt` lists names without a hand-written section,
  and may only shrink. The page in the nav under Reference; `api_index.py
  --check` joins the gate chains in `CLAUDE.md` and `CONTRIBUTING.md` (a test
  compares them). From this pull request, a pull request adding a public name
  documents it (it does not add it to the list) and regenerates the index.
- **Tests.** `tests/project/test_api_documented.py`: `test_the_api_index_is_current`
  (runs `--check`); `test_every_public_name_is_documented_or_listed`;
  `test_the_undocumented_list_only_shrinks` (no listed name now documented or
  gone); `test_every_public_name_has_a_docstring`.
- **Changelog.** Documentation.

### 16.5 A tutorial from clinical history to a batch, and pages that run in order (PR 4, after W21-11 PR 4)

- **State at 2.0.2.** Two tutorials (`first-sample.md`, `first-training-run.md`).
  `docs/guides/clinical-training.md` covers the ground as a how-to from a prebuilt
  cohort. The harness copies the workspace and builds a namespace per block
  (`tests/project/test_docs_python.py`), so blocks cannot build on each other.
- **Change.** A page whose first line after the title is `<!-- sequential -->`
  runs all its blocks in order, in one scratch copy and one namespace, as one test
  per page, the failure naming the block (`SEQUENTIAL = "<!-- sequential -->"`;
  `_runnable()` leaves those pages out; `test_documented_pages_run_in_order` runs
  them; torch skipping through `TORCH_NAMES` applies). `docs/tutorials/clinical-task.md`,
  all blocks runnable: (1) two subjects with `medh5.create`, `set_clock` (with
  `Clock.utc`), `add_event(Event(..., effective_start_us=…, available_us=…))`,
  `add_link`; (2) `TaskManifest.new(...)` (or `from_cohort`) and `.save(...)`;
  (3) `task.preflight()`, `report.counts`, `report.row(...).reasons`, one row
  excluded and why; (4) `ClinicalTaskDataset` with
  `DataLoader(collate_fn=collate_clinical)`, `next(iter(...))`. It links to the
  guide for slots, documents and caches. Nav (`mkdocs.yml`) and
  `tutorials/index.md`.
- **Tests.** `test_documented_pages_run_in_order` (main job, which has torch);
  `test_a_sequential_page_fails_at_the_block_that_fails`.
- **Changelog.** Documentation.

### 16.6 A concurrency page (PR 5, after W21-15 PR 2)

- **State at 2.0.2.** The facts are scattered: `docs/reference/torch.md` (workers,
  the handle cache, fork), `docs/guides/performance.md`,
  `docs/explanation/design-rationale.md`, `docs/examples/index.md`,
  `docs/install.md`. The existing claims are accurate.
- **Change.** `docs/explanation/concurrency.md`, read before adding threads or
  processes: the HDF5 lock (one per process, filters included; only Blosc2
  windows decode outside it; what releases the GIL, from W21-15 PR 1); threads or
  processes, with measured figures given as approximate; fork and exec
  (`DataLoader` workers and the PID-keyed handle cache that abandons handles after
  fork; close-on-exec, on every platform since W21-00 PR 6); writers (one thread
  each; `amend` replaces the file, so a handle held across it reads the old inode;
  leases pin `content_id`, T302; lost updates refused, W21-04); parallel commands
  (`--workers` runs processes, `stats --workers` threads); free-threading (where
  it stands: cp314t wheels with the GIL re-enabled, `gil_used = false` in 2.2.0).
  Two runnable blocks: `DataLoader(..., num_workers=2,
  worker_init_fn=medh5.torch.worker_init_fn)` over `case.medh5` (needs torch), and
  a `ThreadPoolExecutor` reading `case.medh5` and `case2.medh5`. The scattered
  pages keep a one-line pointer. A `STALE_CLAIMS` entry: "Blosc2 decoding releases
  HDF5's lock" (without its qualifier).
- **Changelog.** Documentation.
