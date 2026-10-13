# 2.0.2 work in progress: delete this directory before the release PR merges

This directory keeps the 2.0.2 implementation work alive across sessions; the cloud VM that ran it may be reclaimed. It is **not** part of the release. Remove it, `git rm -r .wip-2.0.2`, before the PR is marked ready.

## The request
> "Implement all recommended scope (patch: bug fixes, docs, performance; no API or format change) for 2.0.2. Prepare the 2.0.2 release PR (all in one PR)."

The scope is §3 (W0–W10) of `audit-2.0.2-2.1.0.md`, including the revision-2 items F01–F22. It is split into 19 workstreams. Each one has a prompt in `prompts/<name>.md`, and `AGENT-BRIEF.md` holds the shared rules: patch-only, no new codes or dependencies, a regression test per fix, test-naming conventions, and docs tests.

## State (2026-10-12, branch `claude/relaxed-volta-ua1vih`)
**Merged here** (squash commits; each workstream's report, with CHANGELOG drafts, is in `reports/<name>.md`):

| Workstream | Commit |
|---|---|
| writer | `17bd3e9` |
| conv1 | `b157a90` |
| sec | `69e81e0` |
| curation | `f6e04ca` |
| scrub | `b193fbe` |
| annot | `6cf90c9` |
| valid | `82d1631` |
| dataset | `6f68e5c` |
| conv2 | `280f014`, plus `c3b9f0e` (`dicom.py` `_safe` now uses conv2's `sanitize_ascii_stem`) |
| torch | `a7351b6` |
| spec | `dfcbc8b` |
| (coordinator) 2.0.1 Rust signatures kept | `357a195` |
| pyapi | `56460dd` |
| docs | `fbe9dae` |
| legacy | `bea9f87` |
| comp | `2cbf709` (+ `4a5fb5f` API surface) |
| clin | `982a561` (+ `32931ac` API surface) |
| cli | `7a4f3c2` |
| followups | `353e900` |
| ci | `7c5fdf6` (pins Rust 1.97.0: locally, link `~/.rustup/toolchains/1.97.0-x86_64-unknown-linux-gnu` to the stable install) |

At `82d1631` (all of the above except dataset), the full test run passed: pytest 2997 passed, 2 skipped, 95.03% coverage; `cargo test --workspace` green; the corpus 163/163 through both command lines; fmt, ruff and mypy clean. Neither clippy nor mkdocs had been run on the merged tree yet. Dataset's item 4 (manifest paths relative to `root`) is deferred: it needs a manifest-format decision.

**Test run at `a7351b6`** (all merges above): pytest 3081 passed, 2 skipped, 95.11% coverage; `cargo test --workspace` green; the corpus 163/163 through both command lines; fmt, ruff and mypy clean.

**In progress:** `geom` (bounding its cubic coefficient cache with a process-wide memory budget before merge; take its side of `docs/guides/performance.md`) and `polish` (`prompts/polish.md`: the small handoffs of merged workstreams; its items 9 and 19 wait for geom).

**Coordinator, after the last merge:** a polish pass for the small handoffs of merged workstreams (listed in the scratchpad's `agent-ids.md`), the GIL pass over the storage, transforms, geometry and clinical bindings, the CHANGELOG (the draft is in the scratchpad), the version bump, the full gate, deleting this directory, and the PR.

**Decided by the user:** keep the Rust struct fields `Moments.nonfinite`, `ClassStats.key` and `Diagnosis.malformed`, and the corrected scrub key arrays; the CHANGELOG lists them as Rust-only changes.

The plans are in `prompts/`. Handoffs for docs, spec, cli and others are listed in each `reports/<name>.md`, under "Handoffs" and "Spec wording".

## Resuming
1. Resume each agent in its worktree (`SCR/wt/<name>`, branch `w202/<name>`); if the worktrees were lost, redo the workstream from its prompt on a branch from this one. Finish the prompt's remaining items, test, and squash-merge here.
2. Handoffs to route:
   - **docs:** take `refusals.md` wording from `reports/conv1.md`, and the `python-api.md:419-420` Windows amend wording plus the writer reference notes from `reports/writer.md`.
   - **legacy:** `sanitize_stem` in `convert/grouping.rs` keeps Unicode letters; fold to ASCII, as the Python converters now do.
   - **valid:** the writer changed `integrity/repair.rs:177` (stamping in `writer.digest_algo()`). Open question: is the index `source_digest` always sha256 (§13.3)?
   - **scrub:** after a remap, the writer's commit now compacts, so scrub's separate repack is redundant. Scan the source, not the writer's temp file.
   - **sec:** `.github/scripts/damaged.py` was **not** extended with structural hostile inputs; a safety classifier stopped that attempt, so the user decides. The Rust tests cover deep nesting, the unknown filter and the Zstd bomb.
   - **spec:** the §14.4 symlink wording is in `reports/sec.md`.
3. **Then:**
   - the GIL-release pass over the remaining binding files (W6; the list is in `prompts/pyapi.md`);
   - set the version to 2.0.2: `[workspace.package]` and the three `=2.0.2` pins in `Cargo.toml`, then `cargo update -w`;
   - write `## [2.0.2] — <date>` in the CHANGELOG from the `reports/` drafts, with Security, Fixed, Behaviour changes, Performance and Documentation sections;
   - run the full CLAUDE.md gate;
   - delete this directory;
   - open the PR.
