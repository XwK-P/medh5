# Workstream `followups`: the GIL pass (W6) and the handoffs nobody owns

Worktree: `SCR/wt/followups` (branch `w202/followups`), created from the integration branch after the wave-1, spec and pyapi merges. Read `SCR/AGENT-BRIEF.md` first; its rules apply. The reports of merged workstreams are in `.wip-2.0.2/reports/*.md` in your worktree (read `pyapi.md` §4 and `valid.md` first).

## Files you own
- `crates/medh5-python/src/{annotations,writer,reader,integrity,dataset}.rs`. Do NOT touch `storage.rs`, `transforms.rs` or `geometry.rs` (geom is working there) or `clinical.rs` (comp/clin). You may be asked later to do those.
- For items B–E, localized edits in `crates/medh5/src/{validate/**,sample/reader.rs,sample/image.rs,version.rs,integrity/**}` and their tests.

## Items
A. **W6, GIL release.** Release the GIL (`py.detach(move || ...)`, as `crates/medh5-python/src/misc.rs` does) around the long native calls listed in `.wip-2.0.2/reports/pyapi.md` §4 for the five files you own. Rules:
   - Only calls that do real work: file I/O, a whole-volume or whole-payload pass, encoders and transcoders. Leave cheap accessors alone.
   - Nothing that touches a Python object may run inside the closure; the compiler enforces most of this. Extract Rust data (owned arrays, `Vec`s, paths) before the closure, convert results back after it.
   - Where an input is a borrowed NumPy array, you may read it through an `ArrayView` inside the closure, as NumPy itself does when it releases the GIL. If borrowing it into the closure doesn't compile cleanly, copy only when the copy is cheap relative to the work; otherwise leave that call holding the GIL and list it.
   - Results must be identical. Run each binding's Python test modules after the change.
   - Add one test that shows the GIL is released for a representative call: for example, another Python thread makes progress during a long `transcode`/`build`. Keep it fast and deterministic; if that can't be made robust, say so and skip the test rather than adding a flaky one.
B. **Validator: obb `centers` must be finite (E406).** Spec §8.3 now requires it (see the spec merge); the writer already refuses non-finite centers, but the validator checks only `sizes`. Add the check, plus a regression test that plants a NaN center with h5py.
C. **Format errors, not Python built-ins, for bad files:**
   - a dangling image reference raises `KeyError` from `sample/reader.rs` `image_ids()`;
   - an image of strings raises `TypeError` from `sample/image.rs`.
   Both should raise the package's format error with an existing diagnostic code: E001 for a non-numeric image (F09, §4.2), and the code the validator uses for the dangling reference. Don't remap `KeyError` globally in `errors.rs`.
D. **Version strings:** `version::support` accepts `" 1.0"` but `sample/reader.rs` `require_major` refuses it. Make them agree, per 1.0 §2.1 and 1.1 §2: the version is `"MAJOR.MINOR"`; decide whether whitespace is refused (probably) and apply the decision in both places, with a test.
E. **Optional, only if clear-cut per the spec:**
   - an unknown `digest_algo` (E703) falls back to sha256 when amending/stamping (see `reports/writer.md`);
   - an undigested dataset under a `content_id` fails `verify` but no validator code reports it (see `reports/valid.md`).
   Fix with existing codes if the spec is clear; otherwise report.

## Tests
- `SCR/bin/wext followups`, then targeted pytest modules for each binding you touched (`tests/format/test_annotations*.py`, `test_writer.py`, `test_readers.py`, `test_integrity*.py`, `tests/tools/test_dataset.py` — find the right ones), plus `tests/project/test_typing.py` if any signature changed.
- `SCR/bin/wcargo followups test -p medh5 --lib validate::` and the modules you touched; clippy on `medh5` and `medh5-python` once at the end.

Priority: B, C, D first (small, correctness), then A, then E. Commit after each item. Final report in the brief's format, with CHANGELOG drafts.
