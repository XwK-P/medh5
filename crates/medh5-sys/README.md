# medh5-sys

Native dependencies of the [`medh5`](https://crates.io/crates/medh5) format
engine, built from source so that every frontend --- the Rust crate, the
Python wheel and the `medh5` command line --- carries the same libraries and
needs nothing installed:

- **HDF5**, statically linked through `hdf5-metno-sys` (`static`, `zlib`);
- **C-Blosc2** (vendored, BSD-3-Clause) with LZ4, Zstandard and zlib codecs;
- the **HDF5-Blosc2 filter** (filter id 32026, vendored from the `hdf5plugin`
  distribution, MIT/BSD), byte-compatible with what `h5py` + `hdf5plugin`
  write and read;
- the **HDF5 Zstandard filter** (filter id 32015), in `hdf5plugin`'s chunk
  format, so files compressed by that plugin read here too;
- `medh5_b2nd_read_slice`, which decompresses only the part of a stored
  Blosc2 chunk a window covers, for reads that skip HDF5's filter pipeline.

The engine calls `register_blosc2_filter()` and `register_zstd_filter()` once
before touching a file.

**Building for MSVC targets:** cmake-rs, which builds HDF5, replaces CMake's
release flags under the Visual Studio generator, `/DNDEBUG` included, so HDF5
keeps its assertions and a damaged file aborts the process instead of
returning an error. Set `CFLAGS_x86_64_pc_windows_msvc=/DNDEBUG` (or the
variable for your target) when building; this repository's
`.cargo/config.toml` does, and a crates.io build never reads that file.  So
the build script reads the flags HDF5 was compiled with (`libhdf5.settings`)
and stops an optimised build whose HDF5 lacks `NDEBUG`, naming the variable.
Should HDF5 not be rebuilt once it is set, `cargo clean --release -p
hdf5-metno-src` makes it so (a `cargo install` starts clean).
`MEDH5_SYS_SKIP_NDEBUG_CHECK=1` builds anyway.

Licences: this crate's own code is MIT (`LICENSE`); the vendored code's are in
`vendor/c-blosc2/LICENSE.txt`, `vendor/c-blosc2/LICENSES/` and
`vendor/hdf5-blosc2/LICENSE.txt`.
