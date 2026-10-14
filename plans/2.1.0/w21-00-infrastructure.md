# W21-00 Infrastructure, phase 0

| | |
|---|---|
| Branch | `w21/infrastructure` (one branch per pull request is fine: `w21/infra-branches`, …) |
| Size | M overall; each pull request S, PR 6 S–M, PR 8 M |
| Lands after | nothing; PR 8 after v2.0.2 is on PyPI and crates.io |
| Blocks | every feature pull request (PR 4, PR 5); see the dependency table in [README](README.md#dependencies) |
| Owns | `.github/workflows/*.yml`, `.github/scripts/{release.py,release_notes.py,changelog.py,public_api.py}`, `changes/`, `rust-toolchain.toml`, `medh5/_deprecation.py`, `crates/medh5-python/src/deprecation.rs`, `clippy.toml`, `tests/project/{test_packaging.py,test_changelog.py,test_public_api.py,test_deprecations.py}` |
| Decisions | D35, D38; the rules in [plans/README.md](../README.md) |
| Backport | PR 2 and PR 6 to `release/2.0` if a 2.0.3 is cut |

## Session prompt

> Implement workstream W21-00 (infrastructure, phase 0) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D35, D38), then `plans/2.1.0/w21-00-infrastructure.md`
> in full. Land its pull requests in order, one at a time, each from the latest
> `main`: re-check each item's "State at 2.0.2" on `main`, write the tests first
> and see them fail, implement, run the local gate in `CLAUDE.md`, push, and
> open a pull request to `main` titled `W21-00: <summary>`. Do not merge: the
> coordinator merges. Every CI change also changes `tests/project/test_packaging.py`,
> which pins the workflows' shape, so these pull requests cannot be combined
> or reordered freely. PR 8 waits until v2.0.2 is on both PyPI and crates.io;
> check before starting it. Things only the maintainer can do (rulesets,
> trusted publishers, an upstream HDF5 issue) go in the pull request
> description as a checklist for them.

## Pull requests

| PR | Item | Size |
|---|---|---|
| 1 | 00.1 Branches: CI on `release/**` | S |
| 2 | 00.2 The `pydicom` floor (backport) | S |
| 3 | 00.3 Rust 1.99 | S |
| 4 | 00.4 Changelog fragments | M |
| 5 | 00.5 The deprecation mechanism | S |
| 6 | 00.6 Inherited HDF5 handles on Windows, and the race (backport) | S–M |
| 7 | 00.7 Each platform waits for its own build | M |
| 8 | 00.8 API gates | M |

## Items

### 00.1 Branches: CI on `release/**` (PR 1)

- **State at 2.0.2.** CI runs on pushes and pull requests to `main` only
  (`.github/workflows/ci.yml`, `on:`); `advisories.yml`'s pull-request trigger is
  `main` only, and scheduled workflows run on the default branch only, so the
  weekly audit sees only `main`'s lock. Release is tag-driven: `release.py
  newest` decides which release is Latest and updates the tap; older lines are
  published `--not-latest`.
- **Change.**
  - `ci.yml` `on.push.branches` and `on.pull_request.branches` become `[main,
    "release/**"]`; the same for `advisories.yml`'s `pull_request`.
  - `advisories.yml`'s schedule audits both lines: a matrix `ref: [main,
    release/2.0]`, checked out with `ref: ${{ matrix.ref }}`, and a *step-level*
    check that ends the job successfully when the ref does not exist yet (never
    a job-level `if:`: GitHub counts a skipped required check as passing).
  - `main` keeps the 2.0.2 version until the 2.1.0 release pull request bumps
    it.
  - Maintainer checklist in the pull request: extend the ruleset that requires
    `ci-ok` and up-to-date branches to `release/**`.
- **Tests.** `test_packaging`: `ci.yml`'s branches include `release/**` (beside
  the existing "every main commit tested" test); advisories' schedule matrix
  names both refs; no gating job has a job-level `if:` that could skip it.
- **Changelog.** Fragment under Fixed, area `release`: CI and the advisory
  audit run on patch lines.
- **Done when** a pull request to a `release/test` branch (create and delete it)
  runs the full CI.

### 00.2 The `pydicom` floor is 3.0.1 (PR 2, backport)

- **State at 2.0.2.** `dicom = ["pydicom>=3.0"]` and `dicomseg = ["highdicom>=0.23",
  "pydicom>=3.0"]` (`pyproject.toml`); `test-min-deps` installs `"pydicom==3.0.*"
  "highdicom==0.23.*"`. highdicom 0.23.0 and 0.23.1 require `pydicom>=3.0.1`, so
  the job resolves 3.0.1 or later and the 3.0.0 floor has never been installed.
- **Change.** Both extras say `pydicom>=3.0.1`; `test-min-deps` installs
  `"pydicom==3.0.1" "highdicom==0.23.*"`, with a comment that highdicom sets the
  floor.
- **Tests.** `test_packaging`'s extras test expects `>=3.0.1`; the
  "every declared floor is installed" test now holds for real.
- **Changelog.** Fixed, area `release`.

### 00.3 Rust 1.99 (PR 3)

- **State at 2.0.2.** `rust-toolchain.toml` pins 1.97.0, while 2.0.1's artifacts
  were built by the runners' then-current stable, 1.99.0: 2.0.2 ships from an
  older compiler, which nobody chose.
- **Change.** `channel = "1.99.0"` (or the newest stable when you do this, if
  `cargo-semver-checks` supports its rustdoc JSON format: 0.51.0 reads formats
  v57, v60 and v61, i.e. Rust 1.97–1.99), plus fixes for any new clippy lint.
  `rust-version` stays 1.85.1, held by the `msrv` job. `release/2.0` keeps 1.97.0.
  From here on, a toolchain bump also checks semver-checks' supported formats.
- **Tests.** The toolchain test in `test_packaging` reads the pin; the gate
  passes with the new clippy.
- **Changelog.** `<!-- no entry: toolchain only -->`, or a Fixed/release line if a
  lint fix changes behaviour.

### 00.4 Changelog fragments (PR 4)

- **State at 2.0.2.** One hand-edited `CHANGELOG.md` with `[Unreleased]` at the
  top; 2.0.2's section was drafted at the end, reconciling about twenty claims.
  `release_notes.py` refuses a release while `[Unreleased]` has entries
  (`check_unreleased`) and cuts the notes (`notes`, run by `release.yml`);
  `hooks/mkdocs_hooks.py` puts the changelog on the site; the dash rule covers it
  (`tests/project/test_docs_examples.py`).
- **Change.**
  - `changes/README.md` states the format. Each pull request adds
    `changes/<slug>.md`:

    ```markdown
    <!-- area: validation -->
    ### Fixed

    - **`W907` no longer advises `int16` for a float image holding NaN.** Only the
      finite values were tested, so …
    ```

    One or more `### Section` blocks. SECTIONS, in order: Security, Added,
    Deprecated, Behaviour changes, Removed (majors only), Fixed, Performance,
    Documentation. AREAS: files, writer, python-api, annotations, geometry,
    validation, specification, curation, deidentification, cohorts, clinical,
    converters, cli, torch, task-cache, platform, release, documentation. A
    pull request with
    nothing to record adds a fragment holding only `<!-- no entry: <reason> -->`
    (a file, not a label: a re-run sees the original event, and label events
    would re-run all of CI). `changes/lead.md` holds the release's bold summary,
    written in the release pull request.
  - `.github/scripts/changelog.py` (standard library only):
    - `check`: every fragment parses; areas and sections are known; entries open
      with `- **`; lines are at most 80 columns; no `---`; `[Unreleased]` in
      `CHANGELOG.md` is empty.
    - `preview`: prints the assembled entries.
    - `build VERSION --date D`: writes `## [VERSION] — D`, the lead, then the
      sections in SECTIONS order, entries sorted by AREAS order and then file
      name, and deletes the fragments.
  - `release_notes.py main()` also refuses while `changes/` holds a fragment, so
    a tag cannot ship unassembled entries.
  - `hooks/mkdocs_hooks.py` splices `preview` under `[Unreleased]` on the site,
    so `mkdocs build --strict` resolves every fragment's links on every pull
    request.
  - `ci.yml`'s `lint` job gets `fetch-depth: 2` and a step (the `if:` on the step,
    so the job never skips):

    ```yaml
    - name: Every pull request records its change
      if: github.event_name == 'pull_request' && github.actor != 'dependabot[bot]' && github.head_ref != 'deps/cargo-lock'
      run: |
        git diff --name-only --diff-filter=AM HEAD^1 HEAD -- 'changes/*.md' \
          | grep -v -e '/README.md$' -e '/lead.md$' | grep -q . \
          || { echo "::error::add changes/<name>.md (see changes/README.md)"; exit 1; }
        python .github/scripts/changelog.py check
    ```

  - The gate chains in `CLAUDE.md` and `CONTRIBUTING.md` gain `python
    .github/scripts/changelog.py check` (a test keeps the two equal);
    `CONTRIBUTING.md`'s "Changelog and releases" section describes fragments.
  - The sdist include list gains `changes/*.md`; the dash test scans
    `changes/*.md`.
  - Move anything under `[Unreleased]` into fragments.
- **Tests.** New `tests/project/test_changelog.py`: every fragment passes
  `check`; `build` on a fixture directory gives a golden section (order, dash,
  wrapping); `preview` equals the hook's splice; `release_notes.py` refuses
  leftover fragments; `check` refuses an unknown area or section, a long line,
  `---`, and an entry not opening with `- **`. The gate-chain test is updated.
- **Changelog.** This pull request's own fragment, under Added, area
  `release`, is the first one.
- **Done when** a pull request without a fragment fails `lint`, and the site
  shows fragments under `[Unreleased]`.

### 00.5 The deprecation mechanism (PR 5)

- **State at 2.0.2.** No `#[deprecated]`, no `DeprecationWarning`, no
  `PyErr::warn`, no Deprecated changelog section, no `filterwarnings`.
  `medh5/sample.py` binds `SampleWriter`, `create` and `amend` straight from
  `_core`, so warnings for those must be raised in Rust. Tested with rustc 1.97:
  `#[deprecated]` on a `pub use` is ignored; on a `pub mod` it warns at every
  use of every item inside, so `clippy -D warnings` fails; on an item it warns
  only where that item is used.
- **Change.**
  - `medh5/_deprecation.py`:

    ```python
    PREFIX = "medh5 3.0 "

    def warn(message: str, *, category: type[Warning] = DeprecationWarning,
             stacklevel: int = 3) -> None:
        """Warn about something 3.0 removes or changes; the message starts with PREFIX."""

    def module_getattr(module: str, deprecated: Mapping[str, tuple[object, str]]):
        """A module __getattr__ serving deprecated names, each with its message."""
    ```

    `warn` asserts the prefix. `FutureWarning` for a default whose result
    changes, `DeprecationWarning` for a removal or rename (see
    plans/README.md).
  - `crates/medh5-python/src/deprecation.rs`: `pub(crate) fn warn(py: Python<'_>,
    message: &str, future: bool) -> PyResult<()>`, raising the same categories
    through `PyErr::warn` with the same prefix.
  - `pyproject.toml` `[tool.pytest.ini_options]` gains `filterwarnings =
    ["error:medh5 3\\.0 "]`.
  - `CONTRIBUTING.md` gets a short "Deprecating something" section pointing at
    the rules in plans/README.md, written as the project's own rule.
- **Tests.** `tests/project/test_deprecations.py`: `warn` refuses a message
  without the prefix; a deprecated module attribute warns, resolves, and is not
  in `__all__`; every `#[deprecated]` in `crates/` names `since = "2.1.0"` (or
  later) and "3.0" in its note; every Python deprecation message in `medh5/`
  starts with the prefix (scan `warn(` call sites and `_DEPRECATED` tables).
- **Changelog.** `<!-- no entry: mechanism only -->` (the first deprecation
  carries the first entry).

### 00.6 Inherited HDF5 handles on Windows, and the race (PR 6, backport)

- **State at 2.0.2.**
  - POSIX: HDF5's sec2 driver opens without `O_CLOEXEC` and locks with
    `flock(fd, LOCK_SH | LOCK_EX | LOCK_NB)`, released only when the descriptor
    closes. 2.0.2 marks each descriptor close-on-exec after HDF5 opens it
    (`close_on_exec` in `crates/medh5/src/h5/file.rs`, called from `open_read`,
    `create_truncate`, `convert/legacy.rs` `open_legacy` and
    `conformance/build.rs` `mutate`), held by
    `s14_4_a_child_process_inherits_no_open_file` on Unix only.
  - Windows: `HDopen` is `Wopen`, which adds only `_O_BINARY` before `_wopen`
    (`H5system.c`), so the CRT handle is inheritable, and it shares read and
    write but not delete. Rust's `Command` starts children with
    `bInheritHandles = TRUE`. A child therefore keeps the file open and locked
    (`Wflock` is `LockFileEx` over the whole file), and while it runs, the
    rename `AtomicFile::commit` replaces a file with fails "Access is denied":
    **every amend fails while any child medh5 started is alive.** The
    `rust-platforms` job runs `cargo test -p medh5 --lib` on Windows, whose tests
    start children while parallel tests hold files open: the 2.0.2 Linux flake,
    waiting to happen.
  - On every platform a child started by medh5 on another thread between
    `H5Fopen` and the mark inherits the descriptor.
- **Change.**
  - `close_on_exec` gains a `#[cfg(windows)]` branch: the same driver check and
    `H5Fget_vfd_handle`, then

    ```rust
    extern "C" { fn _get_osfhandle(fd: std::os::raw::c_int) -> isize; }
    let os = unsafe { _get_osfhandle(fd) };
    if os != -1 && os != -2 {
        unsafe { SetHandleInformation(os as HANDLE, HANDLE_FLAG_INHERIT, 0) };
    }
    ```

    (`windows_sys::Win32::Foundation`; check the feature list in the root
    `Cargo.toml`). The descriptor is valid in medh5's CRT because HDF5 and medh5
    are one module on one CRT (hdf5-metno-src builds HDF5 with CMake's default
    dynamic runtime, and Rust links the same UCRT); a test holds this. The doc
    comment drops "and so is a Windows handle". Factor the
    descriptor-to-handle step as a `pub(crate)` helper: W21-05 PR 3 reuses it.
  - **One hold.** `open_read`, `create_truncate`, `open_legacy` and a new
    `pub(crate) fn h5::file::open_rw` (which `conformance::build::mutate` then
    uses) each open and mark inside one hold of HDF5's reentrant global lock:
    `super::locked(|| { let f = hdf5::File::open(path)?; close_on_exec(&f); Ok(f) })`.
  - **Children.** `#[doc(hidden)] pub fn medh5::h5::spawn(command: &mut Command)
    -> io::Result<Child>` is `locked(|| command.spawn())`. Callers: the CLI's
    `python_running` (`crates/medh5-cli/src/lib.rs`: `.status()` becomes
    `spawn()?.wait()`, `.output()` becomes a spawn with piped stdout and stderr
    and `wait_with_output()`), the interpreter probe beside it, and
    `storage/chunking.rs`'s macOS `sysctl`. A root `clippy.toml` sets
    `disallowed-methods` for `std::process::Command::{spawn, output, status}`,
    allowed inside `spawn` itself and in tests that mean "any spawner".
  - Keep HDF5's locking on (1.0 §14.4 forbids turning it off implicitly; the lock
    refuses a second writer on one inode).
  - Document what stays open, in `close_on_exec`'s comment and
    `docs/guides/performance.md`: a child started by *other* code on another
    thread during the microseconds between `H5Fopen` and the mark; `fork`
    without `exec` (1.0 §14.4's "nor inherited across fork").
  - **Upstream** (maintainer checklist): an issue or pull request at
    HDFGroup/hdf5 adding `O_CLOEXEC` to sec2's open flags (and the core, log
    and direct drivers') and `_O_NOINHERIT` to `Wopen`. Draft its text in the
    pull request description. 2.2.0 adopts it when hdf5-metno-src ships it.
- **Tests.**
  - `s14_4_a_child_process_inherits_no_open_file` runs on every platform, with
    a `long_lived_child()` helper (`sleep 30`, or `ping -n 30 127.0.0.1` with
    stdout null on Windows) started by a plain `Command::spawn`; after
    `drop(file)`, both `create_truncate(&path)` and `fs::rename(spare, &path)`
    succeed while the child runs. Run it red on `windows-latest` without the
    fix first, to confirm both expected failures, and say so in the pull
    request.
  - `s14_4_the_descriptor_is_the_files_own`: on Windows `_get_osfhandle` of
    HDF5's descriptor names the same volume and file index as `identity(path)`;
    on POSIX `fstat` gives the same device and inode.
  - `s14_4_a_child_starts_only_between_opens`: while one thread holds `locked`,
    `h5::spawn` on another does not return (checked after 200 ms); once the lock
    is released, it does.
  - `s14_4_one_writer_at_a_time`: a second `create_truncate` of an open file fails
    "unable to lock file" (locking stays on).
- **Changelog.** Fixed, area `files`: "**On Windows, a program started while a
  file is open no longer keeps it open and locked.** …every amendment failed
  while such a child ran…"
- **Done when** `rust-platforms` (windows-latest) and `test-windows` pass with
  the test running on Windows.

### 00.7 Each platform waits for its own build (PR 7)

- **State at 2.0.2.** `build` is one five-target matrix in `ci.yml`, and every
  test job `needs: build`, so no Python test starts before the slowest of the
  five builds ends (about 20 minutes per push). Only `test-macos`,
  `test-windows` and `test-platforms` read another platform's artifacts; the
  other seven (`test`, `test-python-next`, `test-min-deps`, `test-monai`,
  `prototype`, `conformance`, `hdf5-readers`) read
  `ci-dist-x86_64-unknown-linux-gnu` (`prototype` also reads
  `ci-cli-x86_64-unknown-linux-gnu`). `ci-ok` fails on any result but success,
  `cancelled` included, and needs every job but `test-python-next`, the one
  informational job (`continue-on-error: true`), as
  `test_the_required_check_needs_every_job` (`tests/project/test_packaging.py`)
  holds.
- **Change.**
  - A reusable `.github/workflows/build.yml` (`on: workflow_call`, inputs
    `target`, `runner`, `container`; `permissions: contents: read`;
    `timeout-minutes: 60` on its job), holding today's build steps with
    `matrix.*` read as `inputs.*`.
  - Five callers in `ci.yml`: `build-linux-x86_64` and `build-linux-aarch64`
    (each with its container digest), `build-macos-arm64` (macos-14),
    `build-macos-x86_64` (macos-15-intel), `build-windows-x86_64`
    (windows-latest). A caller cannot set `timeout-minutes`; the called job does.
  - Needs: `test`, `test-python-next`, `test-min-deps`, `test-monai`,
    `prototype`, `conformance` (and PR 8's `python-api`) need
    `build-linux-x86_64`; `hdf5-readers` needs `[build-linux-x86_64,
    conformance]`; `test-macos` needs `build-macos-arm64`; `test-windows` needs
    `build-windows-x86_64`; `test-platforms` splits into `test-linux-aarch64` and
    `test-macos-x86_64`, each needing its own build.
  - `ci-ok`'s `needs` lists every job but `test-python-next` (the gate test
    still requires exactly that set), and it keeps failing on `cancelled`: with
    `!cancelled()` instead, a run cancelled by hand would skip `ci-ok`, the
    skipped check would count as passing, and the pull request would become
    mergeable untested. Its step summary prints cancelled jobs separately
    ("cancelled or superseded: re-run").
  - `release.yml` is unchanged (it downloads `ci-dist-*` and `ci-cli-*` by
    pattern).
  - `on:` gains `merge_group:`, so the merge queue (plans/README.md, Branches
    and pull requests) gets `ci-ok` on the tree it would merge. `changes`
    already tests everything on any event but `pull_request`; a merge group's
    ref is its own, so the concurrency group needs no change.
  - The macOS x86_64 pair, `build-macos-x86_64` and `test-macos-x86_64`, is
    skipped on `pull_request` (`if: github.event_name != 'pull_request'`): its
    build took 14 to 23 minutes in October 2026's runs and set every pull
    request's wall clock. It still runs on every push, merge group and the
    release's call, and macOS arm64 runs on every pull request. `ci-ok`
    accepts `skipped` for exactly those two jobs, and only on `pull_request`.
  - Optionally, `build.yml` restores and saves a Rust build cache, never on a
    tag (`!startsWith(github.ref, 'refs/tags/')`): the release calls this
    workflow on its `v*` tag and publishes what it builds, so a release build
    starts clean.
  - Maintainer checklist in the pull request: once it merges, turn on "Require
    merge queue" in the ruleset for `main` (squash; the queue replaces "require
    branches to be up to date"), with `ci-ok` still the required check. The
    queue is available only to a repository owned by an organization.
- **Tests.** In `test_packaging`: `_jobs()` also reads `build.yml`; the list of
  Rust-building jobs names the five callers (containers still 2); the 60-minute
  rule skips `uses:` jobs and checks `build.yml`; the upload-artifact count is
  four across `ci.yml` and `build.yml`; the Homebrew target regex reads the
  callers' `target:` inputs. New `test_each_platform_waits_only_for_its_own_build`:
  a job that downloads `ci-dist-<t>` or `ci-cli-<t>` needs the build job for `<t>`
  and no other build job. `ci.yml` runs on `merge_group`; the macOS x86_64
  pair, and no other job, skips on `pull_request`, and `ci-ok` excuses that
  skip on `pull_request` only; no cache step restores on a tag.
- **Changelog.** `<!-- no entry: CI only -->`.
- **Done when** the Linux test jobs start while the macOS and Windows builds are
  still running, a pull request's run ends without the macOS x86_64 pair, and a
  push to `main` still runs it.

### 00.8 API gates (PR 8, after v2.0.2 is published)

- **State at 2.0.2.** No job compares the API with a release; 2.0.2's "no public
  API change" was checked by hand and the patch still changed Rust structs.
  `crates/medh5/src/lib.rs` also exposes `pub use hdf5;` and `pub use ndarray;`,
  so an hdf5-metno or ndarray major is a medh5 major that semver-checks cannot
  see behind the re-export. Python's only API test is `stubtest` of the stub
  against the build. The Python API: 50 public modules, all with `__all__`, 778
  names, 155 classes, 627 member callables; `inspect.signature` reads all but 5
  PyO3 exception types; every module imports without the optional
  dependencies.
- **Change.**
  - `.github/scripts/release.py api-baseline [--registry crates|pypi]`: the
    newest final release at or below `workspace_version()`, from crates.io or
    PyPI's JSON; prints `version=` and `release_type=` for `$GITHUB_OUTPUT`.
    `release_type` is major, minor or patch from the version difference; when
    the two versions are equal, `patch` on `release/**` (read from `BASE_REF`)
    and `minor` elsewhere; the repository variable `API_RELEASE_TYPE` overrides
    (3.0 work only, D35).
  - A `semver` job in `ci.yml` (ubuntu-latest, timeout 45, the pinned toolchain,
    rust-cache):

    ```yaml
    - run: cargo install cargo-semver-checks --version 0.51.0 --locked
    - id: base
      run: python .github/scripts/release.py api-baseline --registry crates >> "$GITHUB_OUTPUT"
      env:
        BASE_REF: ${{ github.base_ref || github.ref_name }}
        API_RELEASE_TYPE: ${{ vars.API_RELEASE_TYPE }}
    - run: >
        cargo semver-checks -p medh5-sys -p medh5 -p medh5-cli
        --baseline-version "${{ steps.base.outputs.version }}"
        --release-type "${{ steps.base.outputs.release_type }}"
      env:
        CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS: fallback
    ```

    (medh5-sys has no `DOCS_RS` short-circuit, so HDF5 compiles for rustdoc
    twice per run; the job runs beside the builds, off the critical path.)
  - `.github/scripts/public_api.py` (standard library only):
    - `dump` walks `medh5` and every module path without a `_`-prefixed part
      (`medh5._core` stays with stubtest); one line per public name with its
      kind, bases (builtins and medh5 only), own members, and signatures with
      parameter names, kinds and literal defaults (any other default prints
      `<value>`, a dataclass factory `<factory>`); no annotations; a leading
      `self`/`cls` dropped (PyO3 renders `(self, /, …)`); addresses stripped;
      constants typed only. Names in a module's `_DEPRECATED` table print with
      kind `deprecated`. Identical output on Python 3.10–3.14 (normalise enum
      and default reprs).
    - `write` regenerates `tests/project/python_api.txt` (already in the sdist
      through `tests/**/*.txt`).
    - `compare OLD NEW --release-type T`: breaks are a removed name or member, a
      change of kind (except to `deprecated`), a reordered or renamed positional
      parameter, a removed default, a new required parameter, `deprecated` to
      absent before 3.0. Compatible: a new name, a new keyword with a default.
      Under `patch`, additions are listed in the step summary.
  - A `python-api` job (needs `build-linux-x86_64`): install the baseline
    (`pip install medh5==$BASE`, from `api-baseline --registry pypi`) in one
    venv and the CI wheel in another; dump both with `python -I` from
    `$RUNNER_TEMP`, never from the checkout; run `compare`.
  - Both jobs join `ci-ok`'s `needs`.
- **Tests.** `tests/project/test_public_api.py`: the installed package's dump
  equals the snapshot (the failure names `public_api.py write`); `compare` on
  synthetic dumps (removal, reorder, a new keyword with a default, a patch-level
  addition, a deprecated name); normalisation of addresses and factories.
  `release.py api-baseline` against fake version lists (pre-releases ignored;
  the equal-version rule chosen by branch). `test_packaging`'s job lists and
  `ci-ok`'s `needs` gain `semver` and `python-api`.
- **Changelog.** `<!-- no entry: CI only -->`.
- **Done when** a scratch pull request adding a field to a public struct fails
  `semver`, and one removing a Python keyword fails `python-api` (close both).
