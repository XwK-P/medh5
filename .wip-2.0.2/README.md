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

After scrub was merged, the combined tree passed `cargo check --workspace --all-targets`. A full test run of the merged tree was in progress when this was written.

**In progress:**
- `dataset` (`patches/dataset.patch`, refreshed by snapshots). It was resumed with priorities: stats NaN, F10, F16 stats, CLI exit code, then the rest; plus C202 folds (handed off from curation).
- `conv2` had just started (`patches/conv2.patch`, an ASCII-fold helper in `_common.py`).

**Not started:**
- wave 1: torch, docs, spec, ci;
- wave 2: geom, legacy, clin, comp, cli, pyapi. These branch from this branch.

The plans are in `prompts/`. Handoffs for docs, spec, cli and others are listed in each `reports/<name>.md`, under "Handoffs" and "Spec wording".

## Resuming
1. Branch a worktree from `1959938` and `git apply --3way patches/<name>.patch`, or apply onto this branch directly. Finish the prompt's remaining items, test, and squash-merge here.
2. Handoffs to route:
   - **docs:** take `refusals.md` wording from `reports/conv1.md`, and the `python-api.md:419-420` Windows amend wording plus the writer reference notes from `reports/writer.md`.
   - **legacy:** `sanitize_stem` in `convert/grouping.rs` keeps Unicode letters; fold to ASCII, as the Python converters now do.
   - **On merging conv2:** point `medh5/io/dicom.py` `_safe` at conv2's helper in `medh5/io/_common.py`.
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
