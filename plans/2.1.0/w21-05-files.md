# W21-05 Files: one replace routine, identities, groups

| | |
|---|---|
| Branch | `w21/files` |
| Size | M overall: PR 1 M, the others S–M |
| Lands after | W21-00 PR 4, PR 5; PR 1 after W21-04 PR 1 (both change `AtomicFile`); PR 3 after W21-00 PR 6 (the handle helper) |
| Blocks | each owner moving its write sites (W21-07 PR 4, W21-11 PR 3, W21-13, W21-14) |
| Owns | `crates/medh5/src/h5/file.rs`, `crates/medh5/src/h5/ops.rs` (file identity), `crates/medh5-python/src/files.rs` (new), `medh5/_files.py`, `tests/project/test_layers.py` (new) |
| Decisions | D23 |
| Backport | none |

## Session prompt

> Implement workstream W21-05 (files: one replace routine, identities, groups)
> of medh5 2.1.0 in this repository. First read `plans/README.md`,
> `plans/2.1.0/README.md` and `plans/2.1.0/decisions.md` (D23), then
> `plans/2.1.0/w21-05-files.md` in full. Work from the latest `main` on
> `w21/files`; land the four pull requests in order. Before PR 1, check that
> W21-04 PR 1 has merged: it adds a snapshot check to `AtomicFile::commit`,
> which your refactor must keep. PR 1 changes how Python writes files (links are
> kept, modes kept, LF on Windows): rewrite the 2.0.2 test that pins the old
> link rule rather than deleting it. You move only the write sites this file
> lists; each other owner moves theirs, and your guard test's allow-list holds
> theirs until they do. Write tests first, run the local gate from `CLAUDE.md`,
> add fragments, open `W21-05: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 05.1 One replace routine behind `_core`; 05.2 the guard tests | M |
| 2 | 05.3 The command line's and `_files.py`'s write sites | S |
| 3 | 05.4 Windows remembers an unchanged file | S–M |
| 4 | 05.5 An amendment keeps the file's group | S–M |

## Items

### 05.1 One replace routine behind `_core` (PR 1, D23)

- **State at 2.0.2.** The engine replaces files one way, in
  `crates/medh5/src/h5/file.rs`: `resolve_links`, `may_follow`, `temporary_name`,
  `precreate`, `set_mode`, `fsync_path`, `replace`, `fsync_dir`, exposed as
  `AtomicFile` (HDF5 files) and `write_atomic` (`pub(crate)`, bytes). Python has
  two copies: `medh5/_files.py` (`temporary_beside`, `write_text`) and the one in
  `TaskManifest.save` (`medh5/task.py`).

  | | Engine | `_files.write_text` | `TaskManifest.save` |
  |---|---|---|---|
  | link at the target | follows the chain, keeps the link; refuses another user's link in a sticky directory | replaces the link (pinned by `test_a_report_replaces_a_link_rather_than_writing_through_it`, `tests/tools/test_cli.py`) | follows it, no sticky check |
  | temporary's mode | 0o600 when replacing | 0o666 under the umask | 0o666 under the umask |
  | target's mode kept | yes | no (0o600 becomes 0o644, verified) | `copymode`, after the write |
  | fsync file / directory | yes / yes | no / no | yes / no |
  | read-only target on Windows | handled | `PermissionError` | `PermissionError` |
  | newlines | as given | CRLF on Windows | CRLF on Windows |
  | error names | the target | the temporary | the temporary |

  Python does two things the engine does not: it retries when a temporary name is
  taken, and it keeps the target's suffix (nibabel picks a format by suffix).
- **Change.**
  - One private lifecycle in `h5/file.rs`, `Staged { given, target, tmp, mode }`
    (create, retrying on `AlreadyExists`; commit; discard), shared by
    `AtomicFile`, `write_atomic` and a new `StagedFile`. HDF5 temporaries keep
    today's name (the cohort walk skips only `._*`, and would scan a temporary
    ending in `.medh5` as a sample). Keep W21-04's snapshot check in
    `AtomicFile::commit`.

    ```rust
    #[doc(hidden)] pub struct StagedFile { /* Staged */ }
    impl StagedFile {
        pub fn create(target: &Path) -> Result<StagedFile>; // temporary `.{pid}-{hex}-{name}`: suffix kept
        pub fn path(&self) -> &Path;   // the temporary, which a foreign writer writes
        pub fn target(&self) -> &Path; // the file the commit replaces, links followed
        pub fn commit(self) -> Result<()>;
        pub fn abort(self);
    }
    impl Drop for StagedFile { fn drop(&mut self) }  // aborts
    #[doc(hidden)] pub fn write_atomic(target: &Path, bytes: &[u8]) -> Result<()>; // was pub(crate)
    ```

  - Bindings, in a new `crates/medh5-python/src/files.rs`: `write_atomic(path,
    data)` (a `str` as UTF-8, `bytes` as given, under `py.detach`);
    `stage_file(path) -> StagedFileHandle`, a `#[pyclass(module = "medh5._core",
    name = "StagedFileHandle")]` modelled on `CacheWriterHandle`, whose
    `__exit__` commits without an exception and aborts otherwise; `same_file(a,
    b) -> bool`. Stubs:

    ```python
    def write_atomic(path: str | os.PathLike[str], data: str | bytes) -> None: ...
    def stage_file(path: str | os.PathLike[str]) -> StagedFileHandle: ...
    def same_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool: ...
    @final
    class StagedFileHandle:
        @property
        def path(self) -> Path: ...
        @property
        def target(self) -> Path: ...
        def commit(self) -> None: ...
        def abort(self) -> None: ...
        def __enter__(self) -> Self: ...
        def __exit__(self, exc_type: type[BaseException] | None,
                     exc: BaseException | None, tb: TracebackType | None) -> bool: ...
    ```

  - `medh5/_files.py` keeps `write_text`, now `_core.write_atomic`, for its
    existing callers. `temporary_beside` stays, unchanged, while
    `medh5/task.py` (`TaskManifest.save`) and `medh5/io/nifti.py`
    (`write_nifti`) still import it: W21-11 PR 3 and W21-13 PR 9 move those
    sites, and whichever of them merges last deletes `temporary_beside` and
    its allow-list entry (it is private).
- **Tests.**
  - Rust: a link planted at the next temporary name is passed over; the suffix is
    kept. This replaces
    `test_a_link_at_the_temporary_name_is_not_written_through`
    (`tests/tools/test_cli.py`), whose `secrets.token_hex` monkeypatch can no
    longer reach the name (temporary names are a Rust test's, per
    `CLAUDE.md`).
  - Python: one parametrised test over every site moved so far: through a link
    (the link kept, its target replaced), over a 0o600 file (mode kept), over a
    read-only target on Windows, and `\n` bytes on every platform.
  - Python `write_atomic` and the engine's `Manifest.save` write identical bytes
    and modes.
  - Rewrite `test_a_report_replaces_a_link_rather_than_writing_through_it`
    (`tests/tools/test_cli.py`) to pin the kept link, renamed
    `test_S14_4_a_report_keeps_a_link_and_replaces_its_target` (say so in the
    pull request).
- **Changelog.** Behaviour changes, area `files`: "**Files written from Python
  keep a symbolic link at their path, and the file's mode.** …as every
  amendment already did (§14.4)…"

### 05.2 The guard tests (PR 1)

- **Change.** `tests/project/test_layers.py::test_only_the_engine_replaces_files`:
  an AST walk of `medh5/` refusing `os.replace`, `os.rename`, `shutil.move`,
  `tempfile.*`, `Path.write_text`/`write_bytes`, `open(..., "w" | "x" | "a")`, and
  any `nib.save`/`save_as` outside a `with _core.stage_file(...)` block. Its
  explicit allow-list starts as the sites other workstreams still own (below)
  and must be empty at release. A Rust test in `crates/medh5-cli` that
  `std::fs::write` appears in `src/` only inside `#[cfg(test)]` code (its
  allow-list empties in PR 2). Seed each with the 2.0.2 violation to show it
  fails.

  | Site | Owner who moves it |
  |---|---|
  | `medh5/task.py` `TaskManifest.save` (and its `temporary_beside` import) | W21-11 PR 3 (through the engine, D24) |
  | `medh5/io/nifti.py` `write_nifti` and `_replace_text` (and its `temporary_beside` import) | W21-13 PR 9 |
  | `medh5/_files.py` `temporary_beside` itself | the later of W21-11 PR 3 and W21-13 PR 9 deletes it |
  | `medh5/io/nnunetv2.py` `dataset.json` | W21-13 PR 9 |
  | `medh5/io/dicom_seg.py`, `medh5/io/rtstruct.py` `save_as` | W21-13 PR 9 |
  | `medh5/torch/clinical.py` `ConceptVocabulary.save` | W21-14 PR 4 |

### 05.3 The command line's and `_files.py`'s write sites (PR 2)

- **State at 2.0.2.** `crates/medh5-cli/src/clinical.rs` (`clinical export -o`),
  `convert.rs` (`migrate --report`), `dataset.rs` (`dataset split -o`, `dataset
  stats -o`) use `std::fs::write`: in place, through links, non-atomically; so
  `migrate --report r.json` writes through a link while `convert from-nnunet
  --report r.json` replaced it. `medh5/cli.py` `_write_report` uses
  `_files.write_text`; `_same_file` re-implements identity.
- **Change.** The four CLI sites use `medh5::h5::file::write_atomic`;
  `_write_report` uses `_core.write_atomic`; `_same_file` keeps its
  resolve-equality pre-check (for outputs not yet written) and uses
  `_core.same_file` for the identity check. The corpus build's `fs::write` into
  the directory it creates stays.
- **Tests.** The parametrised test of 05.1 over these sites; the CLI guard test's
  allow-list is empty.
- **Changelog.** Part of 05.1's entry (one entry, two pull requests).

### 05.4 Windows remembers an unchanged file (PR 3, after W21-00 PR 6)

- **State at 2.0.2.** The self-contained check is remembered per file identity
  `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` (`h5/ops.rs`), up to
  `CHECKED_LIMIT` = 65,536 entries, read from the open descriptor. Off Unix,
  `opened_identity` returns `None`, so Windows walks every file's graph on every
  open (`s2_without_an_identity_every_open_is_checked` holds that).
- **Change.** Widen the private type to `type FileIdentity = (u64, u128, u64,
  i128, i128)`. `#[cfg(windows)] opened_identity` goes from `H5Fget_vfd_handle`
  to the descriptor, to the OS handle through W21-00 PR 6's helper, then
  `GetFileInformationByHandleEx` for `FileIdInfo` (64-bit volume serial, 128-bit
  file id), `FileBasicInfo` (LastWriteTime, ChangeTime) and `FileStandardInfo`
  (EndOfFile); where `FileIdInfo` is refused (FAT, some SMB shares), fall back to
  `GetFileInformationByHandle`. Write the residual risk in the comment: a writer
  can set ChangeTime on Windows, so within one process a deliberate in-place
  edit restoring both times and the size would skip the re-check (still stronger
  than 1.x's file index alone; POSIX ctime cannot be set). W21-04's Windows
  snapshot may use the same identity.
- **Tests.** `s2_an_unchanged_file_is_remembered` runs on every platform (drop its
  `#[cfg(unix)]`); new `s2_a_file_changed_in_place_is_checked_again` (same length,
  old modification time restored with `File::set_modified`: ctime or ChangeTime
  still changes); delete `s2_without_an_identity_every_open_is_checked`.
- **Changelog.** Performance, area `files`.

### 05.5 An amendment keeps the file's group (PR 4)

- **State at 2.0.2.** Amend keeps a file's mode (`existing_mode`, `precreate`,
  `set_mode`, applied in `commit`) but not its group: the replacement belongs to
  the amending user's primary group, so a 0640 or 0660 file in a shared project
  group becomes readable by the wrong group and unreadable by the right one.
  `write_atomic` and Python's `TaskManifest.save` (`shutil.copymode`) behave the
  same.
- **Change.** In `precreate`/`commit`: `std::os::unix::fs::fchown(&tmp, uid,
  Some(target_gid))`, passing `uid = Some(target_uid)` only when `geteuid() ==
  0`, then `set_mode` (chown clears set-id bits). On `EPERM`, narrow the group
  bits to the other bits (`mode & !0o070 | (mode & 0o007) << 3`), which never
  widens access. The same in `write_atomic` (so the Python sites inherit it).
  ACLs and extended attributes are not carried; `docs/reference/python-api.md`
  and the §14.4 implementation note say so.
- **Tests.** `s14_4_an_amend_keeps_the_files_group` (uses a supplementary group
  from `getgroups`, which GitHub's runner user has; returns early when there is
  none); a unit test of the narrowing; a Python test that a saved manifest keeps
  its group.
- **Changelog.** Fixed, area `files`.
