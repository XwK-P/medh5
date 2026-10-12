# Workstream `sec`: untrusted-file security and the HDF5 layer (W0, part of W1/W3/W6/W9)

Worktree: `SCR/wt/sec` (branch `w202/sec`). This worktree already has a complete debug build, which is the seed the other worktrees are copied from. Do not delete `SCR/wt/sec/target`; other worktrees link against its HDF5 build products.

## Files you own
- `crates/medh5/src/h5/**`: `file.rs`, `ops.rs`, `mod.rs`, `data.rs`, `attrs.rs`, `window.rs`, `graph.rs`.
- `crates/medh5-sys/**`: `build.rs`, `src/lib.rs`, `src/zstd_filter.rs`, `src/ndebug.rs`, `vendor/hdf5-blosc2/*` including `PATCHES.md`. Mark every vendored change `medh5:` and record it in `PATCHES.md`, as the existing patches are.
- `.github/scripts/damaged.py`, the damaged-file CI harness.
- Tests for the above:
  - Rust, beside the code;
  - Python: hostile-file tests in `tests/format/test_validate.py` or `tests/format/test_storage.py`, which are localized additions;
  - amend-through-a-symlink tests in `tests/format/test_writer.py`, also localized.

## Items (quoted from the audit; verify each against the code first)
1. **S1** (§2, W0): a crafted file crashes any process that opens it.
   - `open_read` always runs the self-containment check through HDF5's recursive `H5Lvisit`, which has no depth limit, and each callback resolves the full path from the root.
   - 30,000 nested groups make `medh5.open()` die with SIGSEGV. 10,000 take 62 s and 326 MB.
   - Where: `crates/medh5/src/h5/file.rs:24-30`; `h5/ops.rs:533-553` and around `:330-370`.
   - **Fix:** replace `H5Lvisit` in the self-containment check with medh5's own iterative, depth-capped walk. `MAX_DEPTH = 64` already exists at `h5/ops.rs:177`. Use `H5Literate` per group, and resolve each link relative to its parent, not by full path from the root.
   - Add a deep-chain regression test: deeper than `MAX_DEPTH` must be a clean error, never a crash, and must stay fast.
   - Extend `.github/scripts/damaged.py` with structural hostile inputs: deep nesting, an unknown filter id, a Zstd bomb chunk.
   - The appendix's security report has the reproduction (grep "SIGSEGV" / "H5Lvisit" in `SCR/medh5-audit-appendix-area-reports.md`).
2. **S2** (W0): a file naming an unknown filter id makes HDF5 load and run shared libraries from its default plugin search path.
   - That path is baked in at build time, and on Windows it includes `%ALLUSERSPROFILE%\hdf5\lib\plugin`, which is a DLL-planting route.
   - Blosc2 and Zstd are registered statically, so medh5 never needs dynamic plugins.
   - **Fix:** turn off dynamic filter plugins at HDF5 initialisation with `H5PLset_loading_state(0)`. Re-enable them only on explicit opt-in: `HDF5_PLUGIN_PATH` set, or `MEDH5_ALLOW_PLUGINS=1`. At minimum, `H5PLreplace` the default search path so no build-prefix or `%ALLUSERSPROFILE%` entry ships.
   - Find where HDF5 is initialised and the filters registered (`h5/mod.rs`, `crates/medh5-sys/src/lib.rs`).
   - Test that the loading state is off by default and that an unknown filter is a clean read error.
   - Report the exact opt-in semantics for the docs. They are a behaviour change for anyone relying on dynamic plugins.
3. **S3** (W0): Zstd decompression bomb.
   - The filter allocates the frame's declared content size, or doubles its buffer without bound. A 32 KB chunk drove 1 GB RSS.
   - Where: `crates/medh5-sys/src/zstd_filter.rs:41-117`.
   - **Fix:** cap filter output at the chunk size HDF5 passes. In the reverse path, refuse a declared or streamed size larger than `*buf_size`.
   - Do the same in the vendored Blosc2 filter's non-B2ND branch (`vendor/hdf5-blosc2/blosc2_filter.c:614-633`), which trusts the header's `nbytes` and ignores `blosc2_cbuffer_sizes`'s return; check that return.
   - Add tests: a crafted chunk whose declared size exceeds the chunk must fail cleanly without a large allocation.
4. Remove `blosc2_destroy()` from the Blosc2 filter's failure path (`vendor/hdf5-blosc2/blosc2_filter.c:~658`), as another `medh5:` patch.
   - It tears down global Blosc2 state, contexts and mutexes, while window reads decompress outside the HDF5 lock with the GIL released.
   - This is a possible use-after-free (unverified).
5. **Top issue 1** (W1): scrub, amend, recompress and fix on a symlinked path replace the link with an independent copy and leave the real file untouched.
   - Where: `h5/file.rs` `AtomicFile`. The temp file is derived from the unresolved target, then `fs::rename(tmp, target)` at about `:232-247`.
   - **Fix:** resolve symlinks before the atomic replace. Put the temp file beside the *real* file and rename over the real path, so the link survives and points at the new content.
   - Test through the public API: amend, scrub `--apply`, recompress and fix through a symlink. Each must update the real file and keep the link. A relative link and a chain of links count too.
   - Report wording for spec §14.4 and the amend and scrub docs.
6. **W3:** a missing file or directory must be stat'ed first.
   - Today the raw HDF5 error stack, with a timestamp and a pointer, makes `validate --json` output nondeterministic (`h5/file.rs:25-28`).
   - Give a deterministic "no such file" error in the house style, and test it.
7. **W3:** `close_everything` ignores a failed final `H5Fclose`, so a commit could rename a file whose close failed (`h5/file.rs:219-233`). Propagate the failure before the rename.
8. **W3:** `data::nbytes` multiplies file-derived shapes without overflow checks (`h5/data.rs:784-794`). Use checked arithmetic and return a clean error.
9. **W3:** `recreate_fixed_strings` can lose the original on a mid-swap failure. Make the swap safe (`h5/data.rs`).
10. **W9:** add a Windows read-only amend test.
    - The engine sets the temporary file read-only and *then* opens it for write to fsync, and `MoveFileEx` refuses a read-only target (`h5/file.rs:232-247`, `fsync_path`, `set_mode`). The Python test is skipped on win32.
    - **Fix the order:** fsync before `set_mode`, and clear the target's read-only bit before the rename on Windows.
    - Add a test that runs on every platform: amend a read-only file, then check the result is still read-only and the content new. We can't run Windows here; reason carefully and keep the change minimal.
11. **W6:** `attrs::has` lists every attribute name. Use `H5Aexists` (`h5/attrs.rs`).
12. **W6:** the window path re-opens the file id and queries HDF5's file size (`H5Fget_filesize`, HDF5's cached end-of-file) for every chunk, under the lock (`h5/window.rs:108`). Do it once per `read_window` call.
    - Keep the invariant from CLAUDE.md: "A window read is HDF5's bytes or HDF5's read."
    - The existing tests against `read_hyperslab` must stay green.
13. **W7, code docs:** the medh5-sys documentation says "three things" and lists five. Fix the wording in the medh5-sys crate docs (`src/lib.rs` or its README).

## Notes
- Don't change behaviour for valid files. Window reads, digests and the bytes written must stay identical.
- Run in full: `wcargo sec test -p medh5 --lib h5::`, the medh5-sys tests, and `wcargo sec test -p medh5 --lib` filtered to anything else you touch.
- For the Python checks: `wext sec`, then `wpy sec -m pytest tests/format/test_writer.py tests/format/test_storage.py tests/format/test_validate.py tests/format/test_bytes.py tests/tools/test_scrub.py tests/tools/test_repair.py -q`.
- `damaged.py`: read how CI runs it (`.github/workflows/ci.yml`), and run it locally the same way.
