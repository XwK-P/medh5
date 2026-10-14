# W21-01 Platforms, fuzzing and the supply chain

| | |
|---|---|
| Branch | `w21/platform` |
| Size | L overall; each pull request S–M |
| Lands after | W21-00 PR 4, PR 5; PR 3 after W21-00 PR 7 |
| Blocks | nothing |
| Owns | `fuzz/`, `.github/workflows/{fuzz.yml,lock-refresh.yml}`, `deny.toml`, `crates/medh5/src/conformance/robustness.rs`, `.github/scripts/damaged.py`, the release workflow's attest and crates jobs |
| Decisions | D36, D37 |
| Backport | PR 8's LTO paragraph (documentation) |

## Session prompt

> Implement workstream W21-01 (platforms, fuzzing and the supply chain) of medh5
> 2.1.0 in this repository. First read `plans/README.md`, `plans/2.1.0/README.md`
> and `plans/2.1.0/decisions.md` (D36, D37), then
> `plans/2.1.0/w21-01-platform.md` in full. The pull requests are independent
> except where noted; land them one at a time from the latest `main`, each with
> its `tests/project/test_packaging.py` change, its fragment, and the local gate
> from `CLAUDE.md` passing. Several need the maintainer (trusted publishers,
> secrets, the `windows-11-arm` runner): put what they must do in the pull
> request description as a checklist, and make the workflow fail with a clear
> message, never silently skip, when a secret is missing in this repository.

## Pull requests

| PR | Item | Size |
|---|---|---|
| 1 | 01.1 Fuzzing the Rust readers | M |
| 2 | 01.2 Structural hostile inputs for `damaged.py` | S |
| 3 | 01.3 cp314t wheels, and Windows arm64 if runners exist | M |
| 4 | 01.4 Attestations for every release artifact | S |
| 5 | 01.5 Trusted publishing to crates.io | S |
| 6 | 01.6 Licences and sources (`cargo deny`) | S |
| 7 | 01.7 A weekly lock refresh; `pip-audit` that blocks only on what ships | S |
| 8 | 01.8 `cargo install` gets the release profile; Python 3.10's last line | S |

## Items

### 01.1 Fuzzing the Rust readers (PR 1)

- **State at 2.0.2.** No fuzzing. Robustness rests on unit tests of the
  denial-of-service bugs 2.0.2 fixed (the depth limit `MAX_DEPTH` in
  `h5/ops.rs`; an unregistered filter in `h5/mod.rs`; a zstd frame larger than
  its chunk in `medh5-sys/src/zstd_filter.rs`; the Blosc2 header checks) and on
  `damaged.py`, which flips 1–6 random bytes of corpus files and rarely
  produces the structural shapes where 2.0.2's crashes were.
- **Change.**
  - `fuzz/Cargo.toml`: its own `[workspace]` (so the root workspace, the notices
    and `cargo package` never see it), `cargo-fuzz = true`, dependencies
    `libfuzzer-sys = "0.4.13"`, `arbitrary = { version = "1.5.0", features =
    ["derive"] }`, `medh5 = { path = "../crates/medh5" }` (HDF5 through
    `medh5::hdf5`), `[profile.release] debug = 1`. `.gitignore` gets
    `fuzz/{target,corpus,artifacts,coverage,Cargo.lock}`.
  - Targets:
    - `open_validate`: write the input to `/dev/shm/medh5-fuzz-<pid>.medh5`,
      call `medh5::validate::validate_file(path, "strict", None)`; any `Ok` or
      `Err` is fine, only a crash, a sanitizer report, a timeout or running out
      of memory fails.
    - `clinical_columns`: `#[derive(Arbitrary)]` offsets, lengths and bytes,
      written as a clinical table, then `read_table_deferring`,
      `TextColumn::scan` and cell reads.
    - `zstd_filter`, `blosc2_filter`: through `#[cfg(fuzzing)] #[doc(hidden)]`
      decode helpers in medh5-sys making the same calls as the existing test
      helpers.
  - Seeds: `fuzz/src/bin/seed.rs` calls `medh5::conformance::build_corpus(dir,
    None)` and keeps the files of 256 KiB or less.
  - `.github/workflows/fuzz.yml`: `schedule: cron "23 3 * * 2,5"`,
    `workflow_dispatch`, and `pull_request` on `fuzz/**` and the workflow (a
    5-minute smoke run); a matrix over the four targets, `timeout-minutes: 60`;
    `FUZZ_TOOLCHAIN: nightly-YYYY-MM-DD` (a dated nightly from just after the
    stable pin; it moves with the pin); `HDF5_PLUGIN_PRELOAD: "::"`; `cargo
    install cargo-fuzz --version 0.13.2 --locked`; the corpus in
    `actions/cache` (SHA-pinned like every action), keyed by run id with a
    per-target restore key; `cargo +$FUZZ_TOOLCHAIN fuzz run <target> --
    -max_total_time=2400 -max_len=262144 -timeout=30 -rss_limit_mb=4096
    -malloc_limit_mb=2048` (add `-detect_leaks=0` if HDF5's free lists report
    leaks); a weekly `cargo fuzz cmin`; on failure upload `fuzz/artifacts/<target>`
    for 7 days. A crash becomes a minimised regression test in the crate it hit,
    never a committed input.
- **Tests.** `test_fuzzing_runs_on_a_dated_nightly` (in `test_packaging`):
  `fuzz.yml` names a `nightly-YYYY-MM-DD`, its jobs are capped at 60 minutes,
  every action is SHA-pinned, and it uploads artifacts only on failure.
- **Changelog.** Added, area `release`: the readers are fuzzed twice a week.

### 01.2 Structural hostile inputs for `damaged.py` (PR 2)

- **State at 2.0.2.** 2.0.2 deferred this: the structural cases exist only as
  Rust unit tests.
- **Change.** These are regression fixtures for denial-of-service bugs 2.0.2
  fixed, written by the engine under test from its existing test builders;
  `damaged.py` only checks that each is reported within its bounds, and Python
  constructs no malformed file itself.
  - `#[doc(hidden)] pub fn medh5::conformance::robustness::write(dir: &Path) ->
    Result<Vec<PathBuf>>`: a chain of groups deeper than `MAX_DEPTH`; a dataset
    with filter id 40000; a zstd frame larger than its chunk; a Blosc2 chunk
    whose header claims more than it holds.
  - A hidden `medh5 conformance build --robustness DIR`.
  - `damaged.py --structural DIR`: each file must be reported (exit 1, JSON, at
    least one error) within a time limit and, on POSIX, an RSS bound
    (`resource.getrusage(RUSAGE_CHILDREN)`). It runs beside the existing calls
    (the build job and `msvc-from-crates-io`).
  - The fixtures are not in the corpus: the depth limit is this
    implementation's, not the specification's.
- **Tests.** A Rust test that every `robustness::write` file validates to an
  error, not a panic; the damaged-file test in `test_packaging` also requires
  `--structural` in both jobs.
- **Changelog.** `<!-- no entry: CI only -->`.

### 01.3 cp314t wheels, and Windows arm64 if runners exist (PR 3, after W21-00 PR 7)

- **State at 2.0.2.** One `cp310-abi3` wheel each for manylinux_2_28
  x86_64/aarch64, macOS arm64/x86_64 and win_amd64; `release.yml` checks exactly
  these and the five CLI archives. `#[pymodule(gil_used = true)]`, held by a
  `test_packaging` test. PyPI today: h5py 3.16 and numpy 2.5.4 ship cp314t
  wheels everywhere; torch 2.14.1 ships cp314t for macOS arm64, manylinux
  x86_64/aarch64 and win_amd64; hdf5plugin 7.1.0 has no win_arm64 wheel;
  pyjpegls 1.5.1 (a highdicom dependency) has no wheel after cp313.
- **Change (D37).**
  - **cp314t.** `build.yml` runs a second `maturin build --release --locked --out
    dist --interpreter "$MEDH5_FT_PYTHON"` after the abi3 build
    (`/opt/python/cp314-cp314t/bin/python` in manylinux; setup-python's `3.14t`
    on macOS and Windows). Only pyo3 and the extension rebuild; the engine and
    HDF5 are reused. The first run must show `-cp314-cp314t-<platform>.whl`;
    `release.yml` then requires one per platform. `gil_used = true` stays.
    A gating `test-free-threaded` job (ubuntu-latest, 3.14t, the 3.14 job's
    extras; if pyjpegls does not build on 3.14t, drop `dicomseg` there and say
    why in a comment); an informational `test-no-gil` job with `PYTHON_GIL=0`
    and `continue-on-error: true`.
  - **Windows arm64**, only if a test workflow shows `windows-11-arm` runs for
    this repository (check first; otherwise leave it for 2.2.0 and say so in the
    pull request): a caller `build-windows-arm64` (`runner: windows-11-arm`,
    `target: aarch64-pc-windows-msvc`; if the image has no rustup, install
    `rustup-init.exe` verified by checksum, as `build-dist.sh` does); build with
    Python 3.12 arm64; `test-windows-arm64` runs 3.12 and 3.14 with the dev extra
    minus hdf5plugin, plus nifti and dicom, no torch; the target joins
    `release.yml`'s two lists, `third_party_notices.py`'s `TARGETS`,
    `docs/install.md` and the CLI zip. `.cargo/config.toml` already sets NDEBUG
    for `aarch64-pc-windows-msvc`.
- **Tests.** The informational set in `test_packaging` becomes {test-python-next,
  test-no-gil}; a new test that `release.yml`'s platform and target tuples, the
  build callers and `TARGETS` agree.
- **Changelog.** Added, area `platform`.
- **Docs.** `docs/install.md`: free-threaded CPython installs a wheel and runs
  with the GIL re-enabled (until 2.2.0).

### 01.4 Attestations for every release artifact (PR 4)

- **State at 2.0.2.** PyPI's trusted publishing attaches PEP 740 attestations to
  the wheels and the sdist; the CLI archives on the GitHub Release (with
  `SHA256SUMS`) carry none.
- **Change.** A separate `attest` job in `release.yml`: needs `build`;
  `permissions: {contents: read, id-token: write, attestations: write,
  artifact-metadata: write}`; `actions/attest` (SHA-pinned, v4) with
  `subject-path` listing `cli/*.tar.gz`, `cli/*.zip`, `dist/*.whl` and
  `dist/*.tar.gz`; `github-release` needs `attest`. `docs/install.md` documents
  `gh attestation verify medh5-<v>-<target>.tar.gz --repo XwK-P/medh5`.
- **Tests.** `attest` precedes `github-release` and is the only job with
  `attestations: write`.
- **Changelog.** Security, area `release`.

### 01.5 Trusted publishing to crates.io (PR 5, maintainer)

- **State at 2.0.2.** A long-lived `CARGO_REGISTRY_TOKEN`; `release.py
  publish_crates` verifies and publishes one crate at a time, so a failure
  verifying `medh5-cli` leaves `medh5-sys` and `medh5` published.
- **Change.** `release.py crates` in three stages: verify everything first,
  with `cargo package --locked -p medh5-sys -p medh5 -p medh5-cli` exactly as
  the rust job runs it and no token; authenticate with
  `rust-lang/crates-io-auth-action` (SHA-pinned, `id-token: write`); for each
  crate not yet on the index, `cargo publish --locked --no-verify -p <crate>`,
  then `wait_for_index`. A failed verification publishes nothing. The
  canonical-repository and re-run handling stays. Maintainer checklist: a
  trusted publisher per crate (repository XwK-P/medh5, workflow `release.yml`,
  the crates job's environment), one release, then delete
  `CARGO_REGISTRY_TOKEN`.
- **Tests.** The crate-sequence test in `test_packaging` becomes: verify all,
  authenticate, publish.
- **Changelog.** Security, area `release`.

### 01.6 Licences and sources: `cargo deny` (PR 6)

- **State at 2.0.2.** No licence or source policy; `cargo audit --deny warnings`
  covers advisories only.
- **Change.** In the rust job, cached: `cargo install cargo-deny --version 0.20.2
  --locked`, `cargo deny --locked check licenses bans sources`. `deny.toml`:
  `[licenses] allow` is exactly the six licences of `pyproject.toml`'s
  expression (MIT, BSD-3-Clause, BSD-2-Clause, Zlib, Apache-2.0, MIT-0);
  exceptions for build-time-only crates (unicode-ident: Unicode-3.0;
  target-lexicon: Apache-2.0 WITH LLVM-exception; libloading: ISC); a
  `[[licenses.clarify]]` for hdf5-metno-src naming `ext/hdf5/LICENSE` as
  BSD-3-Clause with the hash cargo-deny prints; `[sources]`
  `unknown-registry = "deny"`, `unknown-git = "deny"`; `[bans]`
  `multiple-versions = "warn"`. cargo-audit stays the one advisory gate. The
  sdist include list gains `deny.toml`.
- **Tests.** `deny.toml`'s allow list equals the licence set in `pyproject.toml`
  (read by regex: 3.10 has no `tomllib`).
- **Changelog.** `<!-- no entry: CI only -->`.

### 01.7 A weekly lock refresh; `pip-audit` that blocks only on what ships (PR 7)

- **State at 2.0.2.** `Cargo.lock` is kept out of dependabot because
  `THIRD_PARTY_NOTICES` must be regenerated with it, so the lock moves only by
  hand. `pip-audit --strict` audits numpy together with every extra but dev, and
  the release needs it: a torch advisory with no fix would hold a medh5 release.
- **Change.**
  - `.github/workflows/lock-refresh.yml`: `cron "17 6 * * 1"` and dispatch, on
    `main` only. `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo
    update` (the msrv job stays green); every crate in `SECURITY.md`'s native
    inventory is put back with `cargo update -p <name>@<new> --precise <old>`,
    and the available bumps are listed in the pull request body (they need a
    person to update `SECURITY.md` and the VEX statements); regenerate the
    notices; open the pull request with `peter-evans/create-pull-request`
    (SHA-pinned) on branch `deps/cargo-lock`, using a GitHub App token or a
    fine-grained PAT `LOCK_REFRESH_TOKEN` (a pull request opened with
    `GITHUB_TOKEN` runs no workflows); without either, still open it and say
    "close and reopen to run CI". `dependabot.yml`'s comment points here.
  - `advisories.yml` gains `workflow_call.inputs.release` (boolean);
    `release.yml` passes `release: true`. Job `pypi` audits
    `project.dependencies` and blocks; job `pypi-extras` audits the extras with
    `continue-on-error: ${{ inputs.release == true }}` and writes its findings to
    the step summary. `SECURITY.md` and `CONTRIBUTING.md` say an extra's
    advisory is reported and dispositioned, not a release blocker.
- **Tests.** lock-refresh reverts every crate the native-inventory test names;
  advisories' `release` input exists and `release.yml` passes it.
- **Changelog.** `<!-- no entry: CI only -->`, unless `SECURITY.md`'s policy
  wording counts as Documentation.

### 01.8 `cargo install` gets the release profile; Python 3.10's last line (PR 8)

- **State at 2.0.2.** The workspace's `[profile.release]` (`lto = "thin"`,
  `codegen-units = 4`, `strip = "symbols"`) is not published: medh5-cli's manifest
  on crates.io has no profile, so `cargo install --locked medh5-cli` builds
  without LTO and with symbols. Python 3.10 reaches end of life this month.
- **Change.**
  - `crates/medh5-cli/README.md` and `docs/install.md` give
    `CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4
    CARGO_PROFILE_RELEASE_STRIP=symbols cargo install --locked medh5-cli`, the
    PowerShell (`$env:`) form and the `--config profile.release.lto="thin"` form,
    keeping the advice to prefer the release binaries.
  - D36: `docs/install.md`, `README.md` and a Deprecated fragment say 2.1.x is
    the last line to support Python 3.10; 2.2.0 requires 3.11. No runtime
    warning.
- **Tests.** In `test_docs_examples.py`: the documented `CARGO_PROFILE_RELEASE_*`
  values in both files equal `[profile.release]` in `Cargo.toml`.
- **Changelog.** Documentation (the profile); Deprecated, area `platform` (3.10).
