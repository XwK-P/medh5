# W22-05 Platforms: musllinux, macOS x86_64, `abi3t`

| | |
|---|---|
| Branch | `w22/platforms-x86` (item 2, the enabler), then `w22/platforms` (items 1, 3, 4) |
| Size | L overall; each item S–M |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `build.yml` and the per-platform callers (W21-00 PR 7); the wheel checks in `release.yml` (W21-01 PR 3) |
| Owns | the pyo3 feature line in `Cargo.toml`, `.github/workflows/{ci.yml,build.yml,release.yml}` platform lists, `.github/scripts/build-dist.sh`, `tests/project/test_packaging.py` |
| Decisions | E5 (the floor stays), E6, E10, E12 (macOS on Intel dropped) |

## Session prompt

> Implement workstream W22-05 (platforms) of medh5 2.2.0 in this repository.
> First read `plans/README.md`, `plans/2.2.0/README.md` and
> `plans/2.2.0/decisions.md` (E5, E6, E10), then
> `plans/2.2.0/w22-05-platforms.md` in full. Item 2 (the macOS x86_64
> cross-build) lands first and alone, from `w22/platforms-x86`, as lane A's
> enabler; items 1, 3 and 4 follow as one pull request from `w22/platforms`, cut
> from the latest `main`. Python 3.10 stays supported (E5): no pull request here
> changes `requires-python` or the test matrix's interpreters. Items 3 and 4 are
> conditional, and you check their condition first and say what you found. Every
> CI change carries its `test_packaging.py` change. Things only the maintainer
> can do go in the pull request description. Run the local gate from
> `CLAUDE.md`, add fragments, open `W22-05: …` pull requests, and arm auto-merge
> only as plans/README.md's "Review before the queue" says.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 05.1 musllinux wheels and a static musl CLI | M |
| 2 | 05.2 macOS x86_64 cross-built on arm64 | S–M |
| 3 | 05.3 `abi3t`, if its conditions hold | S |
| 4 | 05.4 HDF5's own close-on-exec, if released | S |
| 5 | 05.5 Drop macOS on Intel (E12), from `w22/platforms-x86` again | S |

Item 2 lands first and alone, as lane A's enabler: every queue run of 2.2.0
waits on the Intel Mac chain until it does. Items 1, 3 and 4 then land as one
pull request. The Python 3.11 floor this workstream first planned as its PR 1
is withdrawn (E5): 3.10 stays supported in 2.2.0.

**State at 2.1.0** (checked at v2.1.0), for every item:

- Five build callers in `ci.yml`. `build-macos-x86_64` and
  `test-macos-x86_64` run on `macos-15-intel` and are skipped on pull requests
  (W21-00 PR 7); `ci-ok` accepts that skip on a pull request only, and
  `test_packaging.py`'s `test_the_intel_mac_skips_pull_requests_and_nothing_else`
  pins it.
- `build-dist.sh` builds for the host only: it never passes `--target` to
  maturin or cargo, takes the binary from `target/release/medh5`, and fetches
  rustup-init for `<arch>-unknown-linux-gnu` only. Cross-building and musl
  both need that plumbing first.
- `release.yml` builds nothing: it downloads the `ci-dist-*` artifacts and
  checks them against a `platforms` dict. `TARGETS` is in
  `.github/scripts/third_party_notices.py`; `homebrew_formula.py` has its own
  table of targets. A new target goes in all three, and the packaging test
  that compares them.

## Items

### 05.1 musllinux wheels and a static musl CLI (PR 1)

- **Change.** Callers in `ci.yml` for musllinux_1_2 x86_64 and aarch64, with
  `quay.io/pypa/musllinux_1_2_<arch>@sha256:<digest>` (take the digests from a CI
  `docker pull`; they could not be fetched when this was planned);
  `build-dist.sh`'s container branch learns musl (`--compatibility
  musllinux_1_2`, rustup-init for the musl targets, verified by checksum as
  `build-dist.sh` does today); the static musl CLI archive; tests without torch or
  hdf5plugin (no musllinux wheels); `release.yml`'s lists, `TARGETS`,
  `docs/install.md`. Every run gains these jobs, and in 2.1.0 the runner
  pool, not any one job, made queue runs slow: measure the pull request's
  run, and if the musl chain is its slowest, skip it on pull requests the way
  W21-00 PR 7 skipped Intel, with `ci-ok`'s exception and its test.
- **Tests.** The release-tuple agreement test; the musl test jobs.
- **Changelog.** Added, area `platform`.

### 05.2 macOS x86_64 cross-built on arm64 (PR 2)

- **Change.** Build `x86_64-apple-darwin` on macos-14 (`--target`, which
  `build-dist.sh` learns, with the binary taken from
  `target/<target>/release/`), test under Rosetta (setup-python `architecture:
  x64`), and drop the macos-15-intel jobs: the slowest build, on an image
  GitHub is retiring (check the announced date when you start, and say it in
  the pull request). Then put the Intel jobs back on pull requests, removing
  `ci-ok`'s exception, if the pull request's own run shows the cross-built
  chain is no longer the slowest; otherwise keep the skip and say why with
  the timings. On a pull request, an Intel-only failure costs one push; in the
  queue (#51 in 2.1.0) it cost a whole re-queue.
- **Tests.** `test_the_intel_mac_skips_pull_requests_and_nothing_else` is
  rewritten to pin whichever was chosen; the release tuples, `TARGETS` and the
  Homebrew table still agree.
- **Changelog.** `<!-- no entry: CI only -->`.
- **Landed** as #70; superseded by 05.5, which removes the target.

### 05.3 `abi3t`, if its conditions hold (PR 3, E6)

- **Change.** If CPython 3.15 is final and NumPy and h5py ship cp315t wheels on
  every target: PyO3's `abi3t-py315` gives one `cp315-abi3t` wheel for every later
  free-threaded version, beside the cp314t wheel. Otherwise, record what is
  missing in the pull request and change nothing.

### 05.4 HDF5's own close-on-exec, if released (PR 4, E10)

- **Change.** If an hdf5-metno-src release ships an HDF5 whose sec2 driver opens
  with `O_CLOEXEC` and whose `Wopen` adds `_O_NOINHERIT` (the upstream request
  2.1.0 filed), bump it: `SECURITY.md`'s native inventory row, every advisory
  re-read, the VEX statement if needed, `THIRD_PARTY_NOTICES`. The engine's mark
  stays, as a no-op for safety. Otherwise, nothing.

### 05.5 Drop macOS on Intel (PR 5, E12)

- **Change.** Remove `build-macos-x86_64` and `test-macos-x86_64` from
  `ci.yml` and `ci-ok`, `build.yml`'s `rosetta` input and its two re-executions
  under Rosetta, `x86_64-apple-darwin` from `release.yml`'s platforms and
  `third_party_notices.py`'s `TARGETS`, and macOS Intel from the Homebrew
  formula, which refuses there instead. `docs/install.md`, `README.md` and
  `CONTRIBUTING.md` stop listing it. Apple silicon stays on `macos-15`.
- **Tests.** `test_macos_x86_64_is_cross_built_and_tested_on_every_run`
  becomes `test_macos_on_intel_is_not_built`; the Homebrew test checks the
  refusal; the target lists still agree.
- **Changelog.** Behaviour changes, area `platform`.
