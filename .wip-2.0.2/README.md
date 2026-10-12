# 2.0.2 work in progress: delete this directory before the release PR merges

This directory keeps the 2.0.2 implementation work alive across sessions; the cloud VM that ran it may be reclaimed. It is **not** part of the release. Remove it, `git rm -r .wip-2.0.2`, before the PR is marked ready.

## The request
> "Implement all recommended scope (patch: bug fixes, docs, performance; no API or format change) for 2.0.2. Prepare the 2.0.2 release PR (all in one PR)."

The scope is §3 (W0–W10) of `audit-2.0.2-2.1.0.md`, including the revision-2 items F01–F22. It is split into 19 workstreams. Each one has a prompt in `prompts/<name>.md`, and `AGENT-BRIEF.md` holds the shared rules: patch-only, no new codes or dependencies, a regression test per fix, test-naming conventions, and docs tests.

## State (2026-10-12, branch `claude/relaxed-volta-ua1vih`)
**Merged on this branch** (squash commits; each agent's report, with CHANGELOG drafts, is in `reports/`):

| Workstream | Commit | What it fixes |
|---|---|---|
| writer | `17bd3e9` | Writer states, `encoding="mask"` refused, `/meta` alias re-link, digest algorithm kept on amend, … |
| conv1 | `b157a90` | DICOM SEG same-label union; RTSTRUCT contour types (F02) and per-ROI frames (F06); DICOM stacking (F07); … |
| sec | `69e81e0` | S1 depth-capped walk, S2 plugins off, S3 decompression caps, symlink replace, … |

**Partly done**: unverified patches against `1959938`, taken mid-implementation, in `patches/<name>.patch`. They may not compile, and their tests may be incomplete. Compare each with its prompt's item list before continuing.

| Patch | How far it got |
|---|---|
| `valid.patch` | 24 files. F01, F09, W3 and the scorer were mostly in; it was adding tests to `test_validate.py`. |
| `annot.patch` | 19 files. It was at the last item (13, the brats vocabulary); then tests, checks and the report. |
| `scrub.patch` | 5 files, `scrub.rs` large. It was writing tests for item 12 (`scrub_paths`); items 13–14 follow. |
| `curation.patch` | 16 files. It was at `splits.rs` (fold leaks, `anatomy_units`); then item 8 (`Tracking` `KeyError`) and the tests. |
| `dataset.patch` | 1 file; it had just started. |
| `conv2.patch` | 1 file (`medh5/io/_common.py`, an ASCII-fold helper); it had just started. |

**Not started:**
- wave 1: torch, docs, spec, ci;
- wave 2: geom, legacy, clin, comp, cli, pyapi. These should branch from this branch, after the patches above are merged.

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
