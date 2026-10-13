## ci workstream: final report (branch `w202/ci`, 15 commits `b0d649a..96bbfb0` on `a7351b6`)

All items DONE except 11 (pyapi's). Coordinator handoffs applied: `fd45971` (annotations §6–§9), `fe1bca8` (CLAUDE.md attestation wording; not in CONTRIBUTING.md). `damaged.py` untouched; ruff stays v0.16.3 (the dev pin).

### Items
1. Pinned toolchains: `rust-toolchain.toml` pins 1.97.0 (rustfmt, clippy); every Rust job runs `rustup toolchain install --no-self-update`; the sdist excludes the toolchain file, so the `sdist` job exports `RUSTUP_TOOLCHAIN` from it; `build-dist.sh` pins maturin 1.15.0 and cmake 4.4.4 and fetches `rustup-init` 1.28.2 checked against a per-arch SHA-256 (no `curl | sh`); manylinux images pinned by digest.
2. Advisories: `.github/workflows/advisories.yml` (`cargo audit --deny warnings` 0.22.2, `pip-audit --strict` 2.10.1 over the runtime dependency and every extra but dev), weekly, on PRs touching `Cargo.lock`/`pyproject.toml`, and from `release.yml` (`build` needs `[ci, advisories]`). New `SECURITY.md` and `vex/medh5.openvex.json` (OpenVEX "not affected" for CVE-2026-85091; verified with `nm`: `compress2`, `uncompress`, `deflate*`, `inflate*`, `adler32`, `crc32`, no `gz*`). Dependabot stays on Actions only.
3. CI efficiency: `Swatinem/rust-cache` (save on main only) in `rust`, `rust-platforms`; every job ≤ 60 min timeout; `fail-fast: false` on the Python matrix; PR artifacts 3 days, others 90.
4. Python 3.15 (adapted): a separate `test-python-next` job, `continue-on-error`, not in `ci-ok` (h5py, PyTorch, pyjpegls publish no cp315 wheels); builds h5py against Ubuntu 24.04's libhdf5, runs `[dev,nifti,dicom]` with `--no-cov`; no classifier (a test ties classifiers to the matrix).
5. Backports: `release.py newest TAG` compares against every final `v*` tag; not newest → `--latest=false`, Homebrew job skipped.
6. Smaller: concurrency group per main/tag commit (`ci-<ref>-<sha>`), PRs still cancel; `crates_io_status` returns absent/published/yanked, publish stops on yanked with instructions, `verify` reports "is yanked"; lint job runs `pre-commit run --all-files` (pre-commit 4.6.2); `pre-commit-hooks` v4.6.0 → v6.0.0; vendored C and `*.patch`/`*.diff` excluded from the whitespace hook.
7. `strip = "symbols"` in `[profile.release]`; no debug-symbol uploads; `PyInit_` stays dynamic.
8. CLAUDE.md and CONTRIBUTING.md chains identical (a test compares them), both with the notices `--check` and the native corpus run.
9. Extras floors: `test-min-deps` installs every declared floor in one resolve on Python 3.10 (numpy 1.24, h5py 3.13, hdf5plugin 4.1, torch 2.0, monai 1.3, nibabel 5.0, pydicom 3.0, highdicom 0.23), checks imports, runs the suite. Finding: highdicom 0.23 requires pydicom ≥ 3.0.1, so `dicomseg`'s `pydicom>=3.0` can never resolve to 3.0.0 (not edited; 2.0.3).
10. `hdf5-readers` job (gated; `.github/scripts/hdf5_readers.py`): HDF5 1.10 (h5py 3.13 on Ubuntu's 1.10.10), 1.12 (h5py 3.9.0), 1.14 (h5py 3.13.0) read every valid corpus case plus samples re-encoded under training/balanced/archive Blosc2 profiles; must match the newest h5py value for value. Verified locally for 1.12.2 and 1.14.6: 171/171 files, 2,510 datasets, 8,671 attributes identical; no upper-bound cap needed. `test_the_release_notes_fit_a_github_release` 9.08 s → 0.69 s, now with boundary checks. Rust time bounds: `transforms/model.rs` two bounds 10 s → 60 s; `curation/splits.rs` 10 s → 30 s.
Other non-owned edit: `pyproject.toml` sdist `include` gains `SECURITY.md`, `vex/*.json`, `.pre-commit-config.yaml`.

### Tests (`tests/project/test_packaging.py`)
16 added (toolchain pinned and installed first, advisories weekly and before release, native inventory = lock, VEX statement, bounded/cached jobs, one local gate chain, release strip, backport not latest, older HDF5 reads, older reader fails on wrong series/value, declared floors installed, hooks run at gate versions, every main commit tested, next Python tried not claimed, yanked crate handling); updated: actions pinned (all workflows), tagged-commit CI needs advisories, required checks, sdist list, release-notes limit, crates mocks.
Run: test_packaging 50 passed; test_docs_examples 232; test_conformance -k StatedCounts 5; ruff clean; YAML loads; `cargo fmt --check` under pinned 1.97.0; pre-commit-hooks v6 emulation 0 failures. Not run: Rust build/clippy/tests, full suite, mkdocs, any workflow.

### CHANGELOG drafts
**Security**
- **Advisories are checked weekly and before every release.** `cargo audit` over `Cargo.lock` and `pip-audit` over the dependency and extras run weekly, on changes to either, and gate the release workflow. SECURITY.md inventories the C libraries compiled in (HDF5 2.2.0, zlib 1.3.2, C-Blosc2 3.3.5, LZ4 1.10.0, Zstandard 1.5.7), which neither tool can see, with a disposition per advisory.
- **CVE-2026-85091 (zlib `gz_vacate`) does not affect medh5.** The bundled zlib 1.3.2 is in range, but nothing links its `gz*` functions, so the overflow is unreachable; `vex/medh5.openvex.json` states this for scanners, and libz-sys will be bumped once a fixed zlib ships.
- **Release builds use pinned inputs:** Rust in `rust-toolchain.toml`; maturin, cmake and a checksummed rustup-init in the build script; manylinux images by digest.

**Fixed**
- **A backport no longer becomes GitHub's "Latest" or moves the Homebrew tap back.**
- **A yanked crate version stops the publish step with what to do;** it read as unpublished, so a re-run hit crates.io's refusal and `verify` called the crate missing.
- **Every commit merged to `main` is tested;** a per-branch concurrency group cancelled pending runs.
- **The spec's "HDF5 ≥ 1.10" promise and every declared dependency floor are tested in CI.**

**Behaviour changes**
- **Release builds strip the symbol table** (about 5 MB of the 30 MB extension); module loading and panic messages are unchanged; a Rust backtrace loses frame names.

**Documentation**
- **SECURITY.md** says how to report a vulnerability and what is checked; CONTRIBUTING's check chain matches CLAUDE.md's and runs everything CI gates; its runbook covers a yanked crate.

### Handoffs
- Coordinator: local `~/.rustup` has only `stable` (1.97.0); link it as `1.97.0` before building a merged checkout.
- Maintainer: enable private vulnerability reporting (Settings → Code security); SECURITY.md links to it.
- docs: `design-rationale.md:243` may need the §7.3 wording (clin changed it).

### Decisions
- Pinned 1.97.0, not 1.99.0 (which built 2.0.1): what this environment has and 2.0.2 was checked on. Bump on purpose.
- `pip-audit` will hold a release on any new advisory against an optional dependency (e.g. torch); make it informational if unwanted.
- Gated but unverified here (first CI run is the check): `hdf5-readers` 1.10 (source build; hdf5plugin 4.1.3 on 1.10.10) and `test-min-deps`. Stopgap if either fails: `continue-on-error: true` on that job.
- SHAs: `Swatinem/rust-cache` v2.9.2 and `pre-commit-hooks` v6.0.0 verified with `git ls-remote`; other action SHAs reused; manylinux digests from the 2.0.1 release logs (quay.io unreachable).
- pre-commit pinned to 4.6.2 (4.7.0 is one day old); not in the dev extra; the lint job runs it.
