# Workstream `writer`: sample writer, amend, images (W2 writer items, F03 writer side, top issues 3 and 11, parts of W6/W7/W9)

Worktree: `SCR/wt/writer` (branch `w202/writer`).

## Files you own
- `crates/medh5/src/sample/{writer,writer_annotations,writer_clinical,image,projection,mod}.rs`.
- `crates/medh5-python/src/{writer,reader}.rs`. The annot agent may make small localized edits in `reader.rs`.
- `medh5/sample.py`, `medh5/image.py`.
- Tests: `tests/format/{test_writer,test_sample}.py`.

## Not yours
- `select_encoding`'s `prefer` check and the annotation encoders: the annot agent.
- Symlink resolution in `h5/file.rs` `AtomicFile`: the sec agent.
- Scrub's scan of `/meta` aliases: the scrub agent.

## Items (quoted from the audit; verify each against the code first)
1. **Top issue 3:** `add_segmentation(encoding="mask")` ORs every class into one mask and writes `class_ids=[]`, `annotated_class_ids=[]`, and the file validates clean.
   - **Fix:** `add_segmentation(encoding=…)` accepts only `auto` plus the transcodable kinds, and calls `check_target` (`sample/writer_annotations.rs:361-363`).
   - Add a regression test through the public API.
2. **Top issue 11:** a refused commit (e.g. E201) aborts the writer, and a second `commit()` returns `None` as if it had succeeded, with no file. Two reviewers found this (`sample/writer.rs:1318-1327, 372-377`; `crates/medh5-python/src/writer.rs:1254-1257`).
   - **Fix:** give the writer explicit states: Open, Committed, Aborted.
   - `commit()` after an abort or a failed commit raises.
   - The checks that run before anything is changed (E201, references) don't kill the writer, so the user can fix the problem and commit again.
3. **F03, writer side:** a hard-link alias of `/meta` (e.g. `x_acme/meta`) keeps the MRN after `scrub --pseudonymise-ids`. Amend copies `/meta` by path, not by object, so the alias carries the old document across (`sample/writer.rs:460-462, 1345-1346`).
   - **Fix:** record every hard path to the source `/meta` object, and re-link each one to the new `/meta` after commit. A soft-link alias is already safe.
   - Reproductions: `SCR/reports/indep-scrub.md` and `SCR/agents/indep/scrub/`.
   - Test: amend a file with a hard-link alias of `/meta`; after amend, the alias resolves to the new document.
4. `remap_frame_uids` should set `compact`, so the commit drops the old UID bytes from freed space (`sample/writer.rs:789-821`). Test that the old UID bytes are absent from the file afterwards.
5. Amend switches `digest_algo` to sha256, and with `digests=False` leaves mixed algorithms that still validate (`sample/writer.rs:1355-1358`). Keep the file's algorithm on amend; never leave mixed algorithms.
6. **W6:** the writer runs `analyse` twice for `layers`. Pass `colouring` through (one line, `writer_annotations.rs:361-373`).
7. **W2, small:** `add_mask` silently coerces a float mask (0.2 → True). Refuse a float mask whose values aren't exactly 0/1, or refuse float masks outright if that matches the spec and docs better. Decide and report.
8. Stop W907 recommending int16 for an image containing NaN (`sample/image.rs:360-369`).
9. `Image.read(physical=True)` returns the stored dtype when there is no rescale, but the docstring says float32 (`sample/image.rs:295-306` and the Python docstring).
   - Prefer fixing the docstrings and docs to match the behaviour, because changing the returned dtype would break users.
   - If the spec or reference docs clearly promise float32, report that rather than changing it.
10. **W6:** redundant copies on image reads. Find and remove avoidable copies in the image read path, with value-for-value identical results.
11. **W2, small:** `repr()` of a closed `Sample` or `Image` raises. Make it return a "closed" repr.
12. **W7, code docs:**
    - `written_version` omits the `drop_clinical` exception;
    - the amend docstrings assume POSIX rename semantics. On Windows, readers must close the file first.
13. **W9:** the test names `test_S16_amend_keeps_a_later_minor` and `test_S16_the_scrub_and_fix_doors_are_the_same_gate` (`tests/format/test_writer.py:405, 417`) no longer match what they check. Rename them, or fix them to check what they say.

## Notes
- **Bytes for valid input must not change**, apart from item 4's compaction and item 5's keeping of the algorithm. Compare `content_id`s before and after where you touch the commit path.
- Tests to run:
  - `wcargo writer test -p medh5 --lib sample::`;
  - `wext writer`, then `wpy writer -m pytest tests/format/test_writer.py tests/format/test_sample.py tests/format/test_annotations.py tests/format/test_encodings.py tests/tools/test_scrub.py -q`.
