# W22-06 Fuzzing the C libraries

| | |
|---|---|
| Branch | `w22/fuzzing` |
| Size | M |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `fuzz/` and `fuzz.yml`, Rust instrumented with AddressSanitizer (W21-01 PR 1) |
| Owns | `fuzz/`, `.github/workflows/fuzz.yml` |

## Session prompt

> Implement workstream W22-06 (fuzzing the C libraries) of medh5 2.2.0 in this
> repository. First read `plans/README.md` and `plans/2.2.0/README.md`, then
> `plans/2.2.0/w22-06-fuzzing.md` in full. Check that 2.1.0's `fuzz/` crate and
> `fuzz.yml` are on `main`. Work from the latest `main` on `w22/fuzzing`; land one
> pull request. The fuzz crate stays outside the root workspace, the notices and
> `cargo package`. A crash becomes a minimised regression test in the crate it hit,
> never a committed input, and a crash in vendored C is reported upstream as well
> as worked around here. Run the gate from `CLAUDE.md`, add a fragment, open a
> `W22-06: …` pull request, and do not merge.

## Items

### 06.1 HDF5 and Blosc2 under AddressSanitizer (PR 1)

- **State at 2.1.0 (planned).** The fuzz targets (`open_validate`,
  `clinical_columns`, `zstd_filter`, `blosc2_filter`) instrument Rust only; the C
  code (HDF5, c-blosc2, zstd) runs uninstrumented, so a memory error there is
  found only if it crashes.
- **Change.** Build HDF5 and Blosc2 for the fuzz job with `CC=clang` and
  `CFLAGS=-fsanitize=address,fuzzer-no-link`, using a clang whose LLVM matches the
  dated nightly's; set `HDF5_ENABLE_USING_MEMCHECKER=ON` if hdf5-metno-src forwards
  it, so HDF5's free lists do not hide errors from ASan (if it does not, say so and
  propose the upstream change). Keep the 2.1.0 schedule and limits.
- **Tests.** `test_fuzzing_runs_on_a_dated_nightly` extended: the C build flags are
  present in the fuzz job only.
- **Changelog.** `<!-- no entry: CI only -->`.
