//! Native dependencies of the medh5 format engine.
//!
//! This crate links three things into whatever depends on it, statically:
//!
//! * **HDF5**, built from source by `hdf5-metno-sys` (re-exported as
//!   [`hdf5_sys`]);
//! * **C-Blosc2** 3.3.5, vendored, with LZ4, Zstd and zlib from their -sys
//!   crates;
//! * the **HDF5-Blosc2 filter** (HDF Group filter id 32026), upstream's own C
//!   source, so chunks are written and read exactly as `hdf5plugin` writes and
//!   reads them;
//! * the **HDF5 Zstandard filter** (id 32015) in the chunk format `hdf5plugin`
//!   uses, for datasets other tools wrote into a file's extension groups;
//! * [`medh5_b2nd_read_slice`], which decompresses only the part of a stored
//!   chunk a window covers.
//!
//! Nothing here interprets a MEDH5 file; the `medh5` crate does.  The entry
//! points beyond the raw bindings are [`register_blosc2_filter`] and
//! [`register_zstd_filter`], which must run before a dataset using either
//! filter is created or read.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

pub use hdf5_metno_sys as hdf5_sys;

mod zstd_filter;
pub use zstd_filter::ZSTD_FILTER_ID;

// The build script's MSVC check, included here for its tests.
#[cfg(test)]
mod ndebug;

// The codec libraries are linked for their symbols alone; naming the crates
// keeps them in the link even though no Rust code here calls them.
use libz_sys as _;
use lz4_sys as _;
use zstd_sys as _;

/// The HDF Group's registered id for the Blosc2 filter.
pub const BLOSC2_FILTER_ID: u32 = 32026;

/// The vendored C-Blosc2 version: upstream's release, byte for byte.
///
/// 3.3.5 carries upstream's fixes to BloscLZ's bounds checks: in 3.3.2 a
/// match length accumulated past `i32::MAX` before it was checked, so a chunk
/// of a few megabytes could wrap it and write past the block (N19 of the 2.0
/// re-audit; c-blosc2 3.3.3 and 3.3.5).  Any Blosc2 chunk a file carries
/// reaches that decoder, whatever codec this crate writes with.
pub const BLOSC2_VERSION: &str = "3.3.5";

unsafe extern "C" {
    /// Registers the Blosc2 filter with the HDF5 library (upstream's
    /// `register_blosc2`).  Returns 1 when the filter is available.
    ///
    /// `version` and `date`, when both are non-null, receive `strdup`ed
    /// strings the caller must `free`.
    pub fn register_blosc2(version: *mut *mut c_char, date: *mut *mut c_char) -> c_int;

    fn blosc2_init();

    /// Decompress the slice `[start, stop)` of one stored chunk --- a B2ND
    /// frame of `frame_len` bytes --- into `out`, in C order, touching only
    /// the blocks the slice covers.
    ///
    /// Returns 0 on success; 1 when the frame is not a B2ND frame of rank
    /// `ndim`, shape `chunk_shape` and item size `typesize`, or the slice does
    /// not fit in it or in `out_len` bytes (read the chunk through HDF5
    /// instead); a negative Blosc2 error code when the blocks do not
    /// decompress.  `frame` is not modified.
    ///
    /// It touches no HDF5 state, so it needs no HDF5 lock, and each call
    /// builds its own Blosc2 contexts.
    pub fn medh5_b2nd_read_slice(
        frame: *const u8,
        frame_len: i64,
        ndim: i8,
        chunk_shape: *const i64,
        typesize: i32,
        start: *const i64,
        stop: *const i64,
        out: *mut c_void,
        out_len: i64,
    ) -> c_int;
}

/// Register the HDF5-Blosc2 filter with the process's HDF5 library.
///
/// Idempotent and cheap after the first call.  **The caller must hold HDF5's
/// global lock** (`hdf5::sync::sync`) --- the `medh5` crate does --- because the
/// HDF5 C library is not re-entrant.  Returns `false` when HDF5 refused the
/// registration.
///
/// # Safety
///
/// Calls into the HDF5 C library; see above for the locking requirement.
pub unsafe fn register_blosc2_filter() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = not yet, 1 = registered, 2 = refused.
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Acquire) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    unsafe { blosc2_init() };
    let ok = unsafe { register_blosc2(std::ptr::null_mut(), std::ptr::null_mut()) } >= 0
        && unsafe { hdf5_sys::h5z::H5Zfilter_avail(BLOSC2_FILTER_ID as _) } > 0;
    STATE.store(if ok { 1 } else { 2 }, Ordering::Release);
    ok
}

/// Register the HDF5 Zstandard filter (id 32015) with the process's HDF5
/// library.
///
/// Idempotent; the same locking requirement as [`register_blosc2_filter`].
/// Returns `false` when HDF5 refused the registration.
///
/// # Safety
///
/// Calls into the HDF5 C library; the caller must hold HDF5's global lock.
pub unsafe fn register_zstd_filter() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Acquire) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let ok = unsafe { zstd_filter::register() } && unsafe { hdf5_sys::h5z::H5Zfilter_avail(ZSTD_FILTER_ID as _) } > 0;
    STATE.store(if ok { 1 } else { 2 }, Ordering::Release);
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    unsafe extern "C" {
        fn blosc2_get_version_string() -> *const c_char;
        fn blosc1_set_compressor(compname: *const c_char) -> c_int;
        fn blosc2_compress(
            clevel: c_int,
            doshuffle: c_int,
            typesize: i32,
            src: *const c_void,
            srcsize: i32,
            dest: *mut c_void,
            destsize: i32,
        ) -> c_int;
        fn blosc2_decompress(src: *const c_void, srcsize: i32, dest: *mut c_void, destsize: i32) -> c_int;
        fn blosclz_decompress(input: *const c_void, length: c_int, output: *mut c_void, maxout: c_int) -> c_int;
    }

    #[test]
    fn the_stated_version_is_the_vendored_one() {
        // SAFETY: a static string the library owns.
        let found = unsafe { CStr::from_ptr(blosc2_get_version_string()) };
        assert_eq!(found.to_str().unwrap(), BLOSC2_VERSION);
    }

    /// Two literal bytes, then a long match whose length continues in
    /// `0xFF` bytes, `continued` of them --- enough to carry a 32-bit
    /// accumulator past `i32::MAX` --- then its last length byte and its
    /// distance.
    fn overlong_match(continued: usize) -> Vec<u8> {
        let mut stream = vec![1, b'A', b'B', 0xE0];
        stream.resize(stream.len() + continued, 0xFF);
        stream.extend([0x00, 0x01]);
        stream
    }

    #[test]
    fn n19_a_blosclz_match_length_past_i32_max_is_refused() {
        // 255 x 8,421,506 > i32::MAX: 3.3.2 added each byte before checking
        // and wrapped (undefined behaviour; in practice a negative length,
        // which then passed the bounds checks).  Refused before the addition.
        let stream = overlong_match(8_421_506);
        let mut out = [0u8; 64];
        // SAFETY: both buffers are valid for the lengths given.
        let n =
            unsafe { blosclz_decompress(stream.as_ptr().cast(), stream.len() as c_int, out.as_mut_ptr().cast(), 64) };
        assert_eq!(n, 0, "the decoder must refuse the stream");
    }

    #[test]
    fn n19_a_blosc2_chunk_with_an_overlong_match_is_refused() {
        // Upstream's own reproduction (tests/test_blosclz_bounds.c): a Blosc2
        // chunk claiming one 64 KiB BloscLZ block whose stream is a match
        // longer than the block.
        let (blocksize, cbytes) = (65_536_i32, 100_000_usize);
        let mut chunk = vec![0u8; cbytes];
        chunk[0] = 2; // format version
        chunk[1] = 4; // Blosc2 format version
        chunk[3] = 1; // typesize
        chunk[4..8].copy_from_slice(&blocksize.to_le_bytes()); // nbytes
        chunk[8..12].copy_from_slice(&blocksize.to_le_bytes()); // blocksize
        chunk[12..16].copy_from_slice(&(cbytes as i32).to_le_bytes()); // cbytes
        let header = 32; // BLOSC_EXTENDED_HEADER_LENGTH; codec 0 is BloscLZ
        chunk[header..header + 3].copy_from_slice(&[0, b'A', 0xE0]);
        chunk[header + 3..cbytes - 2].fill(0xFF);
        chunk[cbytes - 2..].copy_from_slice(&[0x01, 0x00]);
        let mut out = vec![0u8; blocksize as usize];
        // SAFETY: both buffers are valid for the lengths given.
        let n = unsafe {
            blosc2_init();
            blosc2_decompress(chunk.as_ptr().cast(), cbytes as i32, out.as_mut_ptr().cast(), blocksize)
        };
        assert!(n < 0, "the chunk must be refused, not decompressed to {n} bytes");
    }

    #[test]
    fn n19_valid_blosclz_chunks_still_round_trip() {
        let values: Vec<u8> = (0..1_048_576_u32).map(|i| ((i / 7) % 251) as u8).collect();
        let mut chunk = vec![0u8; values.len() + 32];
        let mut back = vec![0u8; values.len()];
        // SAFETY: every buffer is valid for the length given; the global
        // compressor is set to the one this test is about.
        let (written, read) = unsafe {
            blosc2_init();
            assert!(blosc1_set_compressor(c"blosclz".as_ptr()) >= 0);
            let written = blosc2_compress(
                5,
                1,
                1,
                values.as_ptr().cast(),
                values.len() as i32,
                chunk.as_mut_ptr().cast(),
                chunk.len() as i32,
            );
            assert!(written > 0, "compression failed: {written}");
            let read = blosc2_decompress(chunk.as_ptr().cast(), written, back.as_mut_ptr().cast(), back.len() as i32);
            (written, read)
        };
        assert!((written as usize) < values.len(), "BloscLZ compressed nothing");
        assert_eq!(read as usize, values.len());
        assert_eq!(back, values);
    }
}
