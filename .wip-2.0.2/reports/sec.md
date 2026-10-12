# sec report (w202/sec abf271e)
Items: S1 engine fix DONE (own walk, H5Literate2, parent-relative lookup, dedupe (fileno, token), >64 deep refused E001); damaged.py NOT extended (agent's attempt to craft hostile inputs in Python was stopped by a safety classifier; Rust tests cover deep nesting, unknown filter, zstd bomb). S2 DONE (plugins off by default; default search path cleared incl. build prefix; opt-in HDF5_PLUGIN_PATH non-empty -> HDF5's own handling; MEDH5_ALLOW_PLUGINS not empty/0 -> standard dir only; read once per process). S3 DONE (zstd cap declared + streaming; blosc2 checks blosc2_cbuffer_sizes, caps both branches; cap = 2x chunk buffer + 64 KiB because [Fletcher32, compressor] decompresses slightly more than the chunk). blosc2_destroy removed (patch 5) + uninitialised free in plain branch with no chunks fixed (patch 6). Symlink AtomicFile DONE (resolve up to 40 links; dangling link names file to create; sticky world-writable dir link owned by another user refused like protected_symlinks). Stat-first missing file/dir DONE. close_everything failure propagated DONE. nbytes checked DONE. recreate_fixed_strings swap safe DONE. Windows RO order DONE (not run on Windows). attrs::has H5Aexists DONE. window file size once per read DONE. medh5-sys docs "three things" DONE.
Tests: medh5-sys 15, medh5 lib h5:: 41 (lib 188), Python 445 passed; corpus 157/157 both CLIs; damaged.py existing 300 files 0 unreported; clippy/fmt/ruff/mypy clean.
## CHANGELOG drafts
Security:
- Opening a file no longer recurses without limit. The self-containment check used HDF5's recursive H5Lvisit, so 30,000 nested groups crashed the opening process (SIGSEGV) and 10,000 took a minute. A file nested more than 64 groups deep is now refused at open.
- HDF5 filter plugins no longer load unless asked for. A file naming an unknown filter made HDF5 load shared libraries from its built-in search path, which on Windows includes a directory any local user can create. Blosc2 and Zstandard are compiled in.
- A chunk can no longer decompress far past its dataset's chunk size. The Zstandard and Blosc2 filters allocated whatever size the file stated: a 32 KB chunk reached 1 GB. Output is now capped at twice HDF5's chunk buffer plus 64 KiB.
- A damaged Blosc2 chunk no longer tears down Blosc2's global state, nor frees an uninitialised pointer (vendored medh5: patches).
Fixed:
- Amend, scrub --apply, recompress and fix through a symbolic link now rewrite the file the link names, and keep the link. Before, the link was replaced by an independent copy and the real file was left untouched, which for scrub meant the identifying original.
- Read-only files can be amended on Windows, and they stay read-only.
- A failed final close now fails the commit instead of renaming the file into place.
- Widening a fixed-length string dataset no longer risks losing the original if a step fails.
- A dataset whose size overflows a machine word is now a clean error, where it used to wrap.
Behaviour changes:
- Plugins are opt-in: HDF5_PLUGIN_PATH (non-empty) as before, or MEDH5_ALLOW_PLUGINS (not empty/0) loads from HDF5's standard directory only (/usr/local/hdf5/lib/plugin, or %ALLUSERSPROFILE%\hdf5\lib\plugin on Windows); read once per process. Otherwise a dataset whose filter isn't built in fails to read with "filter plugins disabled".
- Files nested more than 64 groups deep are refused at open (E001).
- A rewrite through a link in a sticky, world-writable directory that another user owns is refused.
- A missing file or a directory now gets a fixed message, so validate --json is deterministic.
Performance:
- Attribute existence checks no longer list every attribute name.
- Window reads measure the file size once per read instead of once per chunk.
Documentation:
- The medh5-sys crate docs no longer say "three things" before a list of five.
## Handoffs
- damaged.py structural cases NOT added (safety classifier stopped the agent) — surface to user; Rust tests cover.
- Docs: amend/scrub/recompress/fix guides (links followed, shared-dir refusal); install/threat-model pages (plugin opt-in); optional: unknown-filter error hint (error.rs).
## Spec wording §14.4
"When the target path is a symbolic link, the replacement is written beside the file the link names (following a chain of links) and renamed over that file, so the link is kept. A link another user owns in a sticky, world-writable directory is not followed. Other hard links to the replaced file keep its previous content."
Amend/scrub docs: "Through a symbolic link, the linked file is rewritten in place and the link is kept. A file reached through several hard links is replaced only at the path given; for scrub --apply, remove the other names."
## Decisions
- new walk mirrors H5Lvisit failures: hard-linked object that can't be looked up fails the walk ("could not be walked"); skips already-seen objects.
- Windows RO/rename changes rest on reasoning, not a Windows run.
