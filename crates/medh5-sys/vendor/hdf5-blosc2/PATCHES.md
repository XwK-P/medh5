# Local changes to the HDF5-Blosc2 filter

`blosc2_filter.c` is upstream's filter (the source `hdf5plugin` ships) with the
changes below, each marked `medh5:` in the source. They turn assertions on
values read from a file into the filter's ordinary error return, so a damaged or
hostile file is an HDF5 read error rather than an aborted process. They are
offered upstream; until they land there, keep them when the filter is updated.

Defining `NDEBUG` is not a substitute. It removes the assertion and leaves the
condition unchecked, so a file whose stored chunk size is smaller than its
chunk would go on to be decompressed into the wrong buffer.

1. **Decompression (B2ND).** `assert(outbuf_size >= size)` aborted the process
   (SIGABRT) when the chunk-size filter value (`cd_values[3]`) was smaller than
   the chunk it describes (audit F07). The size is now computed with overflow
   checks from the stored array's shape, the array's type size must be the
   filter's, and the buffer must hold the array --- exactly, when the filter
   values name the chunk shape, since HDF5 expects the whole chunk back.
   Otherwise the filter pushes an HDF5 error and fails.
2. **Filter values.** A type size of 0 and a chunk dimension below 1, above
   `INT32_MAX`, or whose product overflows `size_t` are refused before either
   direction uses them.
3. **Compression.** `compute_b2nd_block_shape` returned through assertions on
   its arguments (a type size of 0 divided by zero); it now returns `-1`, and
   the caller fails the filter.

The window reads that skip HDF5's pipeline (`src/b2nd_slice.c`) are this
crate's own code, not upstream's, and check the same conditions.
