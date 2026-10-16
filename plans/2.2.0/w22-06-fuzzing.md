# W22-06 Fuzzing the C libraries

| | |
|---|---|
| Branch | `w22/fuzzing` |
| Size | M–L |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `fuzz/` and `fuzz.yml`, Rust instrumented with AddressSanitizer (W21-01 PR 1) |
| Owns | `fuzz/`, `.github/workflows/fuzz.yml`; for 06.1, the smallest engine change each bound needs (in `h5/`, said in the pull request) |

## Session prompt

> Implement workstream W22-06 (fuzzing the C libraries) of medh5 2.2.0 in this
> repository. First read `plans/README.md` and `plans/2.2.0/README.md`, then
> `plans/2.2.0/w22-06-fuzzing.md` in full. Check that 2.1.0's `fuzz/` crate and
> `fuzz.yml` are on `main`. Work from the latest `main` on `w22/fuzzing`; land
> one pull request. The fuzz crate stays outside the root workspace, the notices
> and `cargo package`. A crash becomes a minimised regression test in the crate
> it hit, never a committed input, and a crash in vendored C is reported
> upstream as well as worked around here. Run the gate from `CLAUDE.md`, add a
> fragment, open a `W22-06: …` pull request, and arm auto-merge only as
> plans/README.md's "Review before the queue" says.

## Items

### 06.1 The known HDF5 failures: bounded, so the job finds new ones (PR 1)

- **State at 2.1.0** (checked at v2.1.0). `fuzz (open_validate)` has failed on
  every run since W21-01, on cases inside HDF5's C code that 2.1.0's fuzzing
  found and that are not yet reported upstream (the maintainer holds the
  inputs and decides when to report; their details stay out of this public
  plan and of the pull request). A job that always fails reports nothing
  new.
- **Change.** For each known case, find the bound in the engine that stops it
  before HDF5 does the costly or endless work, without refusing any file the
  conformance corpus or the test suite holds valid (a check of a stored size
  before HDF5 converts it, a limit on what one open may allocate). Where no
  such bound exists, keep the target off that path with a documented,
  minimal exclusion. Then the job is green on the corpus and on the known
  inputs, and a new failure is news. Bounds that change what `validate`
  reports are a Security entry; ask before adding a diagnostic code.
- **Tests.** A minimised regression test per bounded case, in the crate it
  hit, built from a description rather than a committed input; `open_validate`
  passes a run seeded with the known inputs (the maintainer supplies them to
  the session, not to the repository).
- **Changelog.** Security, area `validation`, for each bound; otherwise `<!--
  no entry: CI only -->`.

### 06.2 HDF5 and Blosc2 under AddressSanitizer (PR 1)

- **State at 2.1.0** (checked at v2.1.0). The fuzz targets (`open_validate`,
  `clinical_columns`, `zstd_filter`, `blosc2_filter`) instrument Rust only; the C
  code (HDF5, c-blosc2, zstd) runs uninstrumented, so a memory error there is
  found only if it crashes. The three are built three ways: HDF5 with CMake
  through hdf5-metno-src, the vendored C-Blosc2 by `medh5-sys`'s `build.rs`,
  and zstd by zstd-sys; each needs the flags its own way.
- **Change.** Build HDF5 and Blosc2 for the fuzz job with `CC=clang` and
  `CFLAGS=-fsanitize=address,fuzzer-no-link`, using a clang whose LLVM matches the
  dated nightly's; set `HDF5_ENABLE_USING_MEMCHECKER=ON` if hdf5-metno-src forwards
  it, so HDF5's free lists do not hide errors from ASan (if it does not, say so and
  propose the upstream change). Keep the 2.1.0 schedule and limits.
- **Tests.** `test_fuzzing_runs_on_a_dated_nightly` extended: the C build flags are
  present in the fuzz job only.
- **Changelog.** `<!-- no entry: CI only -->`.
