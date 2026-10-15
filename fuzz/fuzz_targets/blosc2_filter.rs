//! The Blosc2 filter (32026) decoding a chunk it did not write, with filter
//! values the input chose (plain Blosc2, or B2ND with a rank and a shape).
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Chunk {
    /// The dataset's chunk, in bytes: what HDF5 sizes the buffer by.
    buf_size: u16,
    /// Revision, block size, type size, chunk bytes, level, shuffle and
    /// codec; then, for B2ND, the rank and the chunk's shape.
    values: Vec<u32>,
    data: Vec<u8>,
}

fuzz_target!(|chunk: Chunk| {
    let values: Vec<u32> = chunk.values.into_iter().take(16).collect();
    let _ = medh5_sys::fuzzing::blosc2_decode(&values, &chunk.data, usize::from(chunk.buf_size));
});
