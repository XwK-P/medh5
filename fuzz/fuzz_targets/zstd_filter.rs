//! The Zstandard filter (32015) decoding a chunk it did not write, for a
//! dataset chunk of a size the input chose.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Chunk {
    /// The dataset's chunk, in bytes: what HDF5 sizes the buffer by.
    buf_size: u16,
    frame: Vec<u8>,
}

fuzz_target!(|chunk: Chunk| {
    if let Some(out) = medh5_sys::fuzzing::zstd_decode(&chunk.frame, usize::from(chunk.buf_size)) {
        // What the filter hands back is never more than its ceiling.
        assert!(out.len() <= 2 * usize::from(chunk.buf_size).max(chunk.frame.len()).max(1) + 64 * 1024);
    }
});
