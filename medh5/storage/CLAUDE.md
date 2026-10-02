## Codec profiles

`training` (lz4:1), `balanced` (zstd:3, default), `archive` (zstd:9),
`portable` (gzip:4, readable without hdf5plugin). Under `balanced`, labels get
`bitshuffle` where images get byte `shuffle`; `training` byte-shuffles both,
`archive` bit-shuffles both, and `portable` runs HDF5's own shuffle before gzip
for both. Chunks are sized by `optimize_chunks()` from the patch hint toward an
L3-cache budget; stacked encodings chunk per plane so one layer reads without
the others. `storage/chunking.py::grid_chunks` is that rule, shared by the
writer and `recompress --rechunk`.
