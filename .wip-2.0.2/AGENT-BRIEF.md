# medh5 2.0.2: implementation brief (read this first; it applies to every workstream)

`SCR` below means `/tmp/claude-0/-home-user-medh5/86c872f8-365f-5ea6-a3e4-b98526568914/scratchpad`.

## What we are doing
We are implementing the 2.0.2 scope of an audit of medh5 2.0.1: the tables in §2 and the checklists W0–W10 in §3 of `SCR/medh5-audit-2.0.2-2.1.0.md`. Your prompt lists the items that are yours. Everything ends up in one release PR, so quality matters more than speed.

**Patch-release rules (2.0.2 = bug fixes, docs, performance):**
- No public API removal or rename. Add public API only where an item explicitly calls for it (e.g. an additive result field).
- No file-format change: a valid input must still produce the same bytes, unless an item fixes a wrong byte.
- **No new diagnostic codes.** `crates/medh5/data/codes.json` and the §15.2 table are normative, so reuse an existing code. If an item truly needs a new code, report it and don't add it.
- **No new dependencies**, Cargo or Python.
- Turning a silent wrong result into a refusal (an error) counts as a bug fix, but list it under "Behaviour changes" in your report.
- Read `/home/user/medh5/CLAUDE.md` and keep its invariants: geometry is never invented; `class_ids` vs `annotated_class_ids` coverage; boxes at voxel edges; canonical JSON; digests cover decompressed content; copy-on-write amend; handles never cross fork; and the others listed there.
- `docs/spec/medh5-1.0.md` and `medh5-1.1.md` are normative; `docs/spec/task-cache-1.md` is versioned separately. Code implements the spec. If your fix changes what the spec says, report the wording for the spec agent, unless your prompt gives you that spec section.

## Where the details are
- The item text, quoted in your prompt, which cites the audit's line numbers.
- `SCR/medh5-audit-2.0.2-2.1.0.md` (about 94 KB): §2 has the top-issue tables, §3 the checklists, §7 the cross-check notes.
- `SCR/medh5-audit-appendix-area-reports.md` (about 525 KB; **grep it, don't read it whole**) has the area reports with line references, reproductions and fix suggestions. Part II, sections 31–40, holds the verifications of F01–F22.
- `SCR/reports/indep-*.md` are the verifier reports for F01–F22. `SCR/agents/indep/<area>/` holds their reproduction scripts, which you can adapt into regression tests. Areas: integrity, rtstruct, nifti-dicom, scrub, dataloss, labels-tracking, agreement, transforms-nodes, torch.
- **Verify each claim against the code before fixing it**, because line numbers may have drifted. If an item is wrong, already fixed, or would need an API or format change, say so in your report and don't force a change.

## Your environment
- **Your worktree** is `SCR/wt/<name>`, on branch `w202/<name>`, created from `main`. Edit files **only inside your worktree**, using absolute paths. **Never edit `/home/user/medh5`** (the coordinator's tree) or another agent's worktree. Read-only looks at other places are fine.
- **Helpers** are in `SCR/bin`. Use them for every build and test command; each one takes your workstream name first:
  - `SCR/bin/wcargo <name> <cargo args>` runs cargo in your worktree with its own target directory (a lean dev profile, no incremental, 2 jobs). Examples:
    - `SCR/bin/wcargo <name> test -p medh5 --lib annotations::read`
    - `SCR/bin/wcargo <name> test -p medh5 --test companion`
    - `SCR/bin/wcargo <name> test -p medh5-cli <filter>`
  - `SCR/bin/wext <name>` builds the Python extension from your worktree's Rust and installs it as `<wt>/medh5/_core.abi3.so`. Run it after Rust changes and before Python tests.
  - `SCR/bin/wext-prebuilt <name>`: Python-only workstreams run this once. It installs the coordinator's prebuilt extension, which comes from the same Rust as your base.
  - `SCR/bin/wpy <name> -m pytest tests/... -q` runs Python with your worktree first on `sys.path`. The `medh5` console script resolves to your worktree too.
  - `SCR/bin/wsh <name> '<command>'` runs a shell command in your worktree with the venv on PATH. Use it for `ruff`, `mypy`, the `medh5` CLI and `mkdocs`.
- **The first build in a fresh worktree** takes several minutes. Registry dependencies are pre-seeded, but your workspace crates compile from scratch. A rebuild of the engine takes about 1–4 minutes under load, so batch your edits rather than rebuilding after every line. If no Rust test needs your change yet, `wcargo <name> check -p medh5` is faster.
- **The machine is shared:** 4 CPUs and 15 GB RAM for about a dozen agents. Be frugal:
  - Never run the full Python suite or `cargo test --workspace`. Run your area's test modules in full, plus targeted tests elsewhere.
  - Run clippy only on what you touched, once near the end: `wcargo <name> clippy -p <crate> --all-targets -- -D warnings`. It is slow.
  - Before committing, run `wcargo <name> fmt --all`. If you touched Python, run `wsh <name> 'ruff check . && ruff format --check . && mypy medh5'`.
  - If you changed a binding signature, keep `medh5/_core.pyi` in step: `tests/project/test_typing.py` runs stubtest.
  - Put scratch files under `SCR/tmp/<name>/`, and leave no background processes running.
- **Python environment:** a shared venv with Python 3.13, numpy, h5py 3.16, hdf5plugin, nibabel, pydicom, highdicom, torch 2.14.1 (CPU) and MONAI 1.6.1. The network may block some hosts; don't install anything.

## Ownership (this is what keeps the merges safe)
- **Files you own** are listed in your prompt. Edit them freely.
- **Small, localized edits in files you don't own** are allowed only when an assigned item strictly needs them: one binding signature, a test in another module, one doc sentence. Keep such hunks minimal and list each one in your report.
- **Never edit** `CHANGELOG.md`, `Cargo.lock`, the versions or dependencies in any `Cargo.toml` or `pyproject.toml`, or `crates/medh5/data/codes.json`, unless your prompt says so. The coordinator writes the CHANGELOG from your drafts.
- **A substantial change in a file someone else owns** goes under "Handoffs" in your report: the file, the function, what and why. Don't make it yourself.
- **Who owns what** (for orientation; your prompt is authoritative for you):

  | Agent | Owns |
  |---|---|
  | sec | `h5/`, `medh5-sys` |
  | valid | `validate/`, `integrity/`, `conformance/` |
  | scrub | `curation/scrub.rs` |
  | annot | `annotations/`, `labels/`, `storage/index.rs`, `sample/reader.rs` |
  | writer | `sample/writer*.rs`, `sample/image.rs`, the writer and reader bindings |
  | curation | tracking, agreement, splits |
  | dataset | `dataset/`, `sampling.rs`, `collection.rs` |
  | geom | `geometry/`, `transforms/`, the storage chunking, codecs and recompress |
  | legacy | `convert/` |
  | clin | `clinical/` |
  | comp | `companion/`, `cache.py`, `task.py` |
  | cli | `crates/medh5-cli` |
  | pyapi | the nodes, document and misc bindings, `_live.py` |
  | conv1 | DICOM, DICOM SEG, RTSTRUCT |
  | conv2 | NIfTI, nnU-Net, `medh5/cli.py` |
  | torch | `medh5/torch`, `medh5/monai.py` |
  | docs | guides, reference, READMEs, mkdocs |
  | spec | `docs/spec/medh5-1.0.md`, `medh5-1.1.md` |
  | ci | `.github`, `CLAUDE.md`, `CONTRIBUTING.md` |

## Conventions (from CLAUDE.md; the release gate enforces them)
- **Python tests live with what they test:** `tests/format/` (the spec through the public API), `tests/tools/`, `tests/integrations/`, `tests/project/`. Add yours to the existing module for your area, and put new test classes next to related ones rather than all at the end of the file, to ease merges.
- **Test names cite the clause they hold:** `test_S8_1_boxes_shift_by_half_a_voxel` in Python, `s14_4_...` in Rust.
- **Fixtures are built by the public writer;** plant defects with h5py.
- **Shared builders** are in `tests/helpers.py` and `tests/kits.py`. Add to them; don't rewrite them. Guard optional dependencies with `pytest.importorskip`.
- **Rust tests** go beside the code (`#[cfg(test)]`) or in `crates/medh5/tests/`.
- **Every fixed bug gets a regression test** that fails before the fix and passes after it. Prefer tests of relationships to tests of single calls. For example:
  - validation agrees with `verify`;
  - a second export has exactly the second plan's inventory;
  - no output path aliases an input.
- **Docs are tested:**
  - `tests/project/test_docs_python.py` executes every python block. A block that can't run gets `<!-- illustrative -->` on the line above its fence; fewer than half may.
  - Every documented `medh5 …` flag is checked against the parser.
  - Never write `---` as a dash in pages, READMEs, the CHANGELOG or CONTRIBUTING; write `—`.
  - The code tables are generated from their sources (`codes.json`, `CHECK_CODES`, the companion `CODES`), so never edit a generated table.
  - `mkdocs build --strict` must pass, and it checks anchors.
- **Style:**
  - Match the surrounding code's comment density, naming and idiom.
  - Comments say why, briefly.
  - Error messages follow the surrounding style, and a refusal names the offending object (path, id, class).

## Commit
- Commit on your branch in your worktree: `git -C SCR/wt/<name> add -A && git -C SCR/wt/<name> commit -m "..."`. Use one or a few commits, each with an imperative summary line and a short body. No trailers: the coordinator squash-merges.
- Leave `git status` clean at the end, and commit no build products. `target/` and `medh5/_core*.so` are gitignored.

## Final report (your last message; the coordinator relies on it)
1. **Each item:** DONE, PARTIAL or SKIPPED (with why), and the files changed.
2. **Tests:** the names of the tests you added, and every command you ran with its result (pass and fail counts). Be explicit about anything you didn't run or that fails.
3. **CHANGELOG drafts** in the repo's style: a bold, one-sentence lead stating the user-visible effect, then 1–3 sentences of explanation. Group them under Security, Fixed, Behaviour changes, Performance and Documentation. A behaviour change is anything that now refuses, or answers differently, for input that used to be accepted or answered.
4. **Handoffs:** changes needed outside what you own (file, what, why).
5. **Spec and doc wording** that the spec or docs agents need, if your fix changes documented behaviour.
6. **Decisions** you made that the coordinator should review, and any open questions.

Keep the report under about 1,500 words.
