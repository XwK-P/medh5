# W22-05 Platforms: musllinux, macOS x86_64, `abi3t`

| | |
|---|---|
| Branch | `w22/platforms` |
| Size | L overall; each pull request S–M |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `build.yml` and the per-platform callers (W21-00 PR 7); the wheel checks in `release.yml` (W21-01 PR 3) |
| Owns | the pyo3 feature line in `Cargo.toml`, `.github/workflows/{ci.yml,build.yml,release.yml}` platform lists, `.github/scripts/build-dist.sh`, `tests/project/test_packaging.py` |
| Decisions | E5 (the floor stays), E6, E10 |

## Session prompt

> Implement workstream W22-05 (platforms) of medh5 2.2.0 in this repository. First
> read `plans/README.md`, `plans/2.2.0/README.md` and `plans/2.2.0/decisions.md`
> (E5, E6, E10), then `plans/2.2.0/w22-05-platforms.md` in full. Work from the
> latest `main` on `w22/platforms`. Python 3.10 stays supported (E5): no pull
> request here changes `requires-python` or the test matrix's interpreters. The
> pull requests are independent; PR 3 and PR 4 are conditional, and you check
> their condition first and say what you found. Every CI change carries its
> `test_packaging.py` change. Things only the maintainer can do go in the pull
> request description. Run the local gate from `CLAUDE.md`, add fragments, open
> `W22-05: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 05.1 musllinux wheels and a static musl CLI | M |
| 2 | 05.2 macOS x86_64 cross-built on arm64 | S–M |
| 3 | 05.3 `abi3t`, if its conditions hold | S |
| 4 | 05.4 HDF5's own close-on-exec, if released | S |

The Python 3.11 floor this workstream first planned as its PR 1 is withdrawn
(E5): 3.10 stays supported in 2.2.0.

## Items

### 05.1 musllinux wheels and a static musl CLI (PR 1)

- **Change.** Callers in `ci.yml` for musllinux_1_2 x86_64 and aarch64, with
  `quay.io/pypa/musllinux_1_2_<arch>@sha256:<digest>` (take the digests from a CI
  `docker pull`; they could not be fetched when this was planned);
  `build-dist.sh`'s container branch learns musl (`--compatibility
  musllinux_1_2`, rustup-init for the musl targets, verified by checksum as
  `build-dist.sh` does today); the static musl CLI archive; tests without torch or
  hdf5plugin (no musllinux wheels); `release.yml`'s lists, `TARGETS`,
  `docs/install.md`.
- **Tests.** The release-tuple agreement test; the musl test jobs.
- **Changelog.** Added, area `platform`.

### 05.2 macOS x86_64 cross-built on arm64 (PR 2)

- **Change.** Build `x86_64-apple-darwin` on macos-14 (`--target`), test under
  Rosetta (setup-python `architecture: x64`), and drop the macos-15-intel build:
  the slowest build, on an image GitHub is retiring (check the announced date
  when you start, and say it in the pull request).
- **Tests.** The Homebrew target regex and the release tuples still agree.
- **Changelog.** `<!-- no entry: CI only -->`.

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
