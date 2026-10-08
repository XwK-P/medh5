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
before touching a file.  Licences of the vendored code are in
`vendor/*/LICENSES`.
