//! Native dependencies of the medh5 format engine.
//!
//! This crate links three things into whatever depends on it, statically:
//!
//! * **HDF5**, built from source by `hdf5-metno-sys` (re-exported as
//!   [`hdf5_sys`]);
//! * **C-Blosc2** 3.3.2, vendored, with LZ4, Zstd and zlib from their -sys
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

// The codec libraries are linked for their symbols alone; naming the crates
// keeps them in the link even though no Rust code here calls them.
use libz_sys as _;
use lz4_sys as _;
use zstd_sys as _;

/// The HDF Group's registered id for the Blosc2 filter.
pub const BLOSC2_FILTER_ID: u32 = 32026;

/// The vendored C-Blosc2 version.
pub const BLOSC2_VERSION: &str = "3.3.2";

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
