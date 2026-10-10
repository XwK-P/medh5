//! The HDF5 Zstandard filter (HDF Group filter id 32015), as `hdf5plugin`
//! writes it.
//!
//! MEDH5 writers never choose it --- every codec profile is Blosc2 or gzip
//! (§14.2) --- but a file may carry datasets another tool wrote, in an
//! extension group that `amend` carries through and `scrub` must read.  Each
//! chunk is one Zstandard frame; `cd_values[0]`, when present, is the
//! compression level, clamped to `1..=ZSTD_maxCLevel()` as `hdf5plugin` clamps
//! it.

use std::os::raw::{c_int, c_uint, c_void};

use hdf5_metno_sys::h5::{H5allocate_memory, H5free_memory};
use hdf5_metno_sys::h5z::{H5Z_class2_t, H5Zregister, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE};
use zstd_sys as zstd;

/// The HDF Group's registered id for the Zstandard filter.
pub const ZSTD_FILTER_ID: u32 = 32015;

const CONTENTSIZE_UNKNOWN: u64 = u64::MAX;
const CONTENTSIZE_ERROR: u64 = u64::MAX - 1;

/// Hand `out` (of `capacity` bytes, `used` of them filled) to HDF5 in place of
/// the chunk buffer.
unsafe fn replace(
    buf: *mut *mut c_void,
    buf_size: *mut usize,
    out: *mut c_void,
    capacity: usize,
    used: usize,
) -> usize {
    unsafe {
        H5free_memory(*buf);
        *buf = out;
        *buf_size = capacity;
    }
    used
}

/// Decompress a frame whose header does not record its size.
unsafe fn decompress_streaming(
    input: *const c_void,
    nbytes: usize,
    buf: *mut *mut c_void,
    buf_size: *mut usize,
) -> usize {
    unsafe {
        let stream = zstd::ZSTD_createDStream();
        if stream.is_null() {
            return 0;
        }
        let mut capacity = nbytes.saturating_mul(4).max(zstd::ZSTD_DStreamOutSize());
        let mut out = H5allocate_memory(capacity, 0);
        let mut used = 0usize;
        let mut source = zstd::ZSTD_inBuffer { src: input, size: nbytes, pos: 0 };
        let result = loop {
            if out.is_null() {
                break 0;
            }
            let mut sink =
                zstd::ZSTD_outBuffer { dst: out.cast::<u8>().add(used).cast(), size: capacity - used, pos: 0 };
            let status = zstd::ZSTD_decompressStream(stream, &mut sink, &mut source);
            if zstd::ZSTD_isError(status) != 0 {
                H5free_memory(out);
                break 0;
            }
            used += sink.pos;
            if status == 0 {
                break replace(buf, buf_size, out, capacity, used);
            }
            if source.pos == source.size && sink.pos < sink.size {
                // The input ended inside a frame.
                H5free_memory(out);
                break 0;
            }
            if used == capacity {
                let grown = H5allocate_memory(capacity * 2, 0);
                if !grown.is_null() {
                    std::ptr::copy_nonoverlapping(out.cast::<u8>(), grown.cast::<u8>(), used);
                }
                H5free_memory(out);
                out = grown;
                capacity *= 2;
            }
        };
        zstd::ZSTD_freeDStream(stream);
        result
    }
}

/// The filter callback: `H5Z_FLAG_REVERSE` decompresses, otherwise compress.
/// Returns the number of valid bytes in `*buf`, or 0 on failure.
unsafe extern "C" fn zstd_filter(
    flags: c_uint,
    cd_nelmts: usize,
    cd_values: *const c_uint,
    nbytes: usize,
    buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    unsafe {
        let input = (*buf).cast_const();
        if flags & H5Z_FLAG_REVERSE != 0 {
            let size = zstd::ZSTD_getFrameContentSize(input, nbytes);
            if size == CONTENTSIZE_ERROR {
                return 0;
            }
            if size == CONTENTSIZE_UNKNOWN {
                return decompress_streaming(input, nbytes, buf, buf_size);
            }
            let Ok(size) = usize::try_from(size) else { return 0 };
            let capacity = size.max(1);
            let out = H5allocate_memory(capacity, 0);
            if out.is_null() {
                return 0;
            }
            let written = zstd::ZSTD_decompress(out, size, input, nbytes);
            if zstd::ZSTD_isError(written) != 0 {
                H5free_memory(out);
                return 0;
            }
            replace(buf, buf_size, out, capacity, written)
        } else {
            let asked = if cd_nelmts > 0 { *cd_values as c_int } else { zstd::ZSTD_CLEVEL_DEFAULT as c_int };
            let level = asked.clamp(1, zstd::ZSTD_maxCLevel());
            let capacity = zstd::ZSTD_compressBound(nbytes).max(1);
            let out = H5allocate_memory(capacity, 0);
            if out.is_null() {
                return 0;
            }
            let written = zstd::ZSTD_compress(out, capacity, input, nbytes, level);
            if zstd::ZSTD_isError(written) != 0 {
                H5free_memory(out);
                return 0;
            }
            replace(buf, buf_size, out, capacity, written)
        }
    }
}

/// Register the filter; see [`crate::register_blosc2_filter`] for the locking
/// requirement.  Returns `false` when HDF5 refused it.
///
/// # Safety
///
/// Calls into the HDF5 C library with its global lock held by the caller.
pub(crate) unsafe fn register() -> bool {
    let class = H5Z_class2_t {
        version: H5Z_CLASS_T_VERS as c_int,
        id: ZSTD_FILTER_ID as _,
        encoder_present: 1,
        decoder_present: 1,
        name: c"Zstandard (hdf5plugin-compatible)".as_ptr(),
        can_apply: None,
        set_local: None,
        filter: Some(zstd_filter),
    };
    unsafe { H5Zregister((&raw const class).cast()) >= 0 }
}
