# Implementation agents (wave 1), for SendMessage resume
sec       a69b0a82899538474   DONE, merged (keep SCR/wt/sec/target: seed for HDF5 libs)
valid     ab394315858e1a28c   DONE, merged
scrub     afbb92090aea5e631   DONE, merged
annot     a1d21d0284ee5c8fd   DONE, merged
writer    aa50a4abbd79b841b   DONE, merged 17bd3e9
conv1     aca1fb979e5bcb653   DONE, merged
curation  a11f5353ba5bcf353   DONE, merged
dataset   acd0918378998d714   DONE, merged (item 4 deferred)
conv2     ab8c7681ec090a0e9   DONE, merged 280f014 (+ _safe unification)
torch     ac1233a378a7fb192   DONE, merged
docs      ae6b833fd52c21e9e   DONE, merged fbe9dae (handoffs routed to geom, legacy, cli, ci)
spec      a80e1258f72423b46   DONE, merged dfcbc8b
ci        a08e55dd70afd108c   DONE, merged 7c5fdf6 (rustup 1.97.0 linked locally)
# Wave 2 (prompts written, worktrees not yet created): geom legacy clin comp cli pyapi
# Integration: squash-merge w202/<name> into claude/relaxed-volta-ua1vih in /home/user/medh5
# Pending handoffs to route
- docs: refusals.md wording from conv1 (see reports-impl/conv1.md); python-api.md:419-420 amend Windows wording + writer reference notes (reports-impl/writer.md)
- legacy (wave 2): sanitize_stem in convert/grouping.rs keeps Unicode letters -> fold to ASCII like Python
- me at merge: point medh5/io/dicom.py _safe at conv2's _common.py helper
- valid: writer touched integrity/repair.rs:177 (stamp in writer.digest_algo()); index source_digest always sha256 question for §13.3
- scrub: writer says separate repack after remap is redundant; scan source not writer temp
- spec: optional "amendment keeps the file's digest_algo"; union/nesting-parity rule for RTSTRUCT(2.1.0?)
legacy    afc7e7543bc54640e   DONE, merged bea9f87
# Wave 2 launched 07:55Z from a7351b6 (worktrees seeded)
comp      a75fcd5a2754b1435   DONE, merged 2cbf709
clin      aa4b7305e08b4f87c   DONE, merged 982a561
pyapi     ae64337137c69e276   DONE, merged 56460dd
geom      ac6ff29f9428aae33   running
cli       a6a0c691b4ff36a45   DONE, merged 7a4f3c2
docs/ci worktrees fast-forwarded to a7351b6; resume after main ext rebuilt (wgate running 07:52Z)
# 2026-10-13 ~16:55Z: weekly limit reset, VM rebooted (fs intact). Gate at a7351b6 was GREEN (3081 passed, 95.11%, corpora 163/163, cargo/fmt/ruff/mypy ok).
# Resumed via SendMessage: legacy spec comp clin pyapi geom docs ci. wsnap loop restarted.
changelog a9182142054634ae9   drafting 2.0.2 CHANGELOG (wave-1 reports) -> SCR/tmp/changelog/2.0.2-draft.md
# Coordinator follow-ups (do after merges)
- valid: validator must check obb `centers` finite (E406), not only `sizes` (spec §8.3 now says so; writer already refuses)
- open: index source_digest always sha256 vs §13.3 (left open by spec; mention in PR)
- DECIDED (user, 2026-10-13): keep Rust struct fields Moments.nonfinite, ClassStats.key, Diagnosis.malformed and the corrected scrub key arrays (76/14); list them in CHANGELOG as Rust-only changes. Signatures restored in 357a195 (box_iou NaN; *_checked doc(hidden)).
- CHANGELOG draft: SCR/tmp/changelog/2.0.2-draft.md (agent a9182142054634ae9; resume it with wave-2 reports). Lead must not claim "no API change" for Rust: say Python API unchanged (additive attrs), Rust keeps 2.0.1 signatures, 3 result structs gain a field.
followups a29cfaa711f83e991   running (GIL pass in annotations/writer/reader/integrity/dataset bindings; obb centers E406; KeyError/TypeError -> format errors; version trim; optional E703/undigested). Later: resume for storage/transforms/geometry/clinical GIL after geom/comp/clin merge.
# sec worktree removed (seed now from followups: WSEED_FROM env)
- docs leftovers for coordinator (owners merged): export.md (conv1's examined-only SEG export, RTSTRUCT names/colours, crossing warning); deidentify.md (sec link/shared-dir wording for scrub --apply); curation.md (same for amend); STALE pattern widen for migrate-0x new/*.medh5 if legacy didn't
- CHANGELOG: note that 2.0.0's perf figures came from an earlier 2.0 build? (docs handoff; decide when finalizing)
- polish pass (after all merges; one agent): export.md (conv1 SEG/RTSTRUCT wording); deidentify.md + curation.md (sec link/shared-dir wording); spec 1.0 Appendix B (legacy: root shape ≤0.5, label w/o label_name own class, pending/flagged->submitted, fallback study:<stem>); medh5/io/_common.py sanitize_ascii_stem docstring (sanitize_stem now folds to ASCII); STALE pattern widen `medh5 validate new/*.medh5 --level strict`; nifti.py:~1230 import activity `inputs` names paths (W1 leak) -> file names only; legacy: build_label_set skips unreadable inputs silently (optional)
- NOTE: seeded worktrees' build outputs pointed into deleted wt/sec; repointed (sed in target/debug/build/*/output). Keep followups' target if seeding more.
- polish (+comp): retire unused companion-codes renderer in hooks/mkdocs_hooks.py (_companion_codes_markdown, COMPANION_SOURCE, regexes, docstring); dataset/manifest.rs:~396 and convert/legacy.rs:~921 write text non-atomically -> use crate::h5::file::write_atomic (now pub(crate))
- API check after each merge: diff pub items vs 1959938 (new frontend-only items -> #[doc(hidden)]; crate-only -> pub(crate)). Done through comp (4a5fb5f).
- polish (+clin): task reconcile -o over a source sample (refuse same file as a resolved source; medh5-cli clinical.rs task_reconcile); spec 1.1 §9.1 step 5 wording (clin report); python-api.md c.select MEDH5VersionError
# 2026-10-13 21:50Z session limit; VM rebooted again. Resumed geom, cli, ci, followups ~22:00Z.
- polish (+cli): engine compare/compare_instances refuse out-of-range threshold (Python API; CLI already does); medh5.bench negative repeats/patch -> MEDH5ValidationError not OverflowError; README + install.md + medh5/cli.py docstring say python3 only (Windows: python); same_file unify: SKIP (refactor)
- geom: asked to bound cubic coefficient cache with 256 MiB process-wide budget before merge (performance.md conflict: take geom side)
- spec (+geom): 1.0 §4.3 origin tolerance wording; §14.1 recommended sizing (one CPU's share of L3; start chunk rule; 4 MiB ceiling)
polish    a7a130e213cc6a36d   running (prompts/polish.md); items 9+19 wait for geom merge
