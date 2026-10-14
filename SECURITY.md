# Security

## Reporting a vulnerability

Report a suspected vulnerability privately, through the repository's
[private vulnerability reporting](https://github.com/medh5-project/medh5/security/advisories/new)
(**Security**, then **Report a vulnerability**), not in a public issue. If that
form is unavailable, open an issue that asks for a private contact and leaves
out the details.

medh5 reads files that other people wrote, so a file that makes it crash, hang,
exhaust memory or run code is a vulnerability, as is anything that leaks what
`medh5 scrub` was asked to remove.

## What is checked, and when

`.github/workflows/advisories.yml` runs every week, on a pull request that
changes `Cargo.lock` or `pyproject.toml`, and on every release tag before
anything is published:

- **The crates** every wheel and binary compiles in, the ones `Cargo.lock`
  names, against the [RustSec](https://rustsec.org/) database, with
  `cargo audit --deny warnings`. An unmaintained, unsound or yanked crate fails
  as a vulnerable one does.
- **The Python packages** that the runtime dependency and the extras install
  today, against PyPI's advisories, with `pip-audit`.

Neither can see the C libraries the crates build from source and link
statically: a C library is not a crate, and an advisory names the library's
version, not the crate's. They are inventoried below, with a disposition for
every advisory against a version the project ships.

A finding that does not apply is passed over in the workflow by its id, and
recorded under [Advisories](#advisories) with the reason. A release ships no
finding that has neither a fix nor a disposition here.

## Running a task manifest you did not write

A task manifest names its sources by path, so whoever wrote it chooses what
`medh5 task preflight`, `task reconcile` and `cache validate` open: any path
the running user can read, relative to the manifest's directory or absolute.
What medh5 guarantees is what the pins cover. A source that is not a regular
file — a named pipe, a device — is refused before it is opened, and a source
that is not the version its reference pins is read no further: its finding
(`T302`) names the pin and nothing the file holds, and the file adds no
identity, clock, event or document to the report. A file that matches its pin
is reported in full, and whether a path exists can still be told from its
finding (`T301`), so run a manifest from someone else only over files you are
willing to have it name.

## Native inventory

What every wheel and every `medh5` binary links statically. The crate column is
what `Cargo.lock` locks, and a test holds the two together; the version column
is the library's own, which is what advisories and scanners name.

| Library | Version | Built from | Used for |
|---|---|---|---|
| HDF5 | 2.2.0 | `hdf5-metno-src` 0.10.4 | the container |
| zlib | 1.3.2 | `libz-sys` 1.1.29 | HDF5's deflate filter, C-Blosc2's zlib codec |
| C-Blosc2 | 3.3.5 | vendored: `crates/medh5-sys/vendor/c-blosc2` | the Blosc2 codecs |
| LZ4 | 1.10.0 | `lz4-sys` 1.11.1+lz4-1.10.0 | C-Blosc2's LZ4 codecs |
| Zstandard | 1.5.7 | `zstd-sys` 2.1.0+zstd.1.5.7 | C-Blosc2's Zstd codec, the Zstandard filter (32015) |
| HDF5-Blosc2 filter | 3.0.1 | vendored: `crates/medh5-sys/vendor/hdf5-blosc2` | the Blosc2 filter (32026) |

The HDF5-Blosc2 filter is the source `hdf5plugin` 7.1.0 ships, with the local
changes `crates/medh5-sys/vendor/hdf5-blosc2/PATCHES.md` lists.

When a library changes (a `-sys` crate is bumped, or a vendored source
updated), update its row, then read every advisory against the new version.

## Advisories

| Advisory | Library | Status | Reason |
|---|---|---|---|
| [CVE-2026-85091](#cve-2026-85091) | zlib 1.3.2 | not affected | the vulnerable code is not linked |

### CVE-2026-85091

A heap overflow in zlib's gzip file interface (CWE-787). After a non-blocking
`gzwrite()` stalls, `gz_vacate()` leaves stale buffer pointers, and a later
`gzprintf()` or `gzvprintf()` writes past the input buffer. Published on
2026-09-03 against zlib 1.3.1.2 through 1.3.2, so the bundled 1.3.2 is in
range. No zlib release fixes it yet.

**Not affected:** the vulnerable code is not present (VEX justification
`vulnerable_code_not_present`). Nothing medh5 compiles in uses the gzip file
interface. HDF5's deflate filter and C-Blosc2's zlib codec call
`compress2`, `uncompress`, `deflate` and `inflate` on memory buffers.
`libz-sys` compiles `gzwrite.c` into its static archive, but a linker takes an
archive member only to resolve a reference, and nothing refers to one. `nm` on
2.0.1's published Linux wheels, x86_64 and aarch64, finds zlib's `adler32`,
`compress2`, `crc32`, `deflate`, `deflateInit_`, `inflate`, `inflateInit_` and
`uncompress`, and no `gz*` symbol; so does a build of this `Cargo.lock`. From
2.0.2 the release profile strips symbols, so repeat the check on a build with
`CARGO_PROFILE_RELEASE_STRIP=none`.

A scanner that finds zlib 1.3.2 in the binary (its version string is there,
for `deflateInit_` to check) or in a bill of materials built from `Cargo.lock`
still reports it. [`vex/medh5.openvex.json`](vex/medh5.openvex.json) states
this disposition in OpenVEX, which Trivy and Grype read with `--vex`.

Once `libz-sys` ships a zlib that fixes it, bump the crate, and record the bump
in the CHANGELOG.
