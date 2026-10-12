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

/// The most a chunk is decompressed to: twice the buffer HDF5 hands the
/// filter, and 64 KiB more.
///
/// That buffer is at least the chunk HDF5 expects back --- it allocates the
/// larger of the stored and the unfiltered chunk --- and HDF5 checks the size
/// once its pipeline has run.  A filter applied before this one when the chunk
/// was written (a Fletcher32 checksum, say) can have added a little to what
/// this one compressed, never this much; a frame that states or decodes to more
/// is no chunk of the dataset.  Uncapped, a 32 KiB frame of run-length blocks
/// asked for a gigabyte, the buffer doubling as it went, and a frame header can
/// state any size up to 2**64.
pub(crate) fn output_ceiling(buf_size: usize) -> usize {
    buf_size.saturating_mul(2).saturating_add(64 * 1024)
}

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

/// Decompress a frame whose header does not record its size, to at most
/// `limit` bytes.
unsafe fn decompress_streaming(
    input: *const c_void,
    nbytes: usize,
    limit: usize,
    buf: *mut *mut c_void,
    buf_size: *mut usize,
) -> usize {
    unsafe {
        let stream = zstd::ZSTD_createDStream();
        if stream.is_null() {
            return 0;
        }
        let mut capacity = nbytes.saturating_mul(4).max(zstd::ZSTD_DStreamOutSize()).min(limit).max(1);
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
                if capacity >= limit {
                    // The frame decodes to more than any chunk of the
                    // dataset holds.
                    H5free_memory(out);
                    break 0;
                }
                let larger = capacity.saturating_mul(2).min(limit);
                let grown = H5allocate_memory(larger, 0);
                if !grown.is_null() {
                    std::ptr::copy_nonoverlapping(out.cast::<u8>(), grown.cast::<u8>(), used);
                }
                H5free_memory(out);
                out = grown;
                capacity = larger;
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
            let limit = output_ceiling(*buf_size);
            let size = zstd::ZSTD_getFrameContentSize(input, nbytes);
            if size == CONTENTSIZE_ERROR {
                return 0;
            }
            if size == CONTENTSIZE_UNKNOWN {
                return decompress_streaming(input, nbytes, limit, buf, buf_size);
            }
            // Refused before anything is allocated for it.
            let Some(size) = usize::try_from(size).ok().filter(|size| *size <= limit) else { return 0 };
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

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 128 * 1024;

    /// A frame of `blocks` run-length blocks of 128 KiB of one byte each ---
    /// four bytes a block, 32,768 to 1 --- whose header states its content
    /// size when `declared`.
    fn rle_frame(blocks: usize, declared: bool) -> Vec<u8> {
        let mut frame = 0xFD2F_B528u32.to_le_bytes().to_vec();
        if declared {
            // An eight-byte content size, one segment.
            frame.push(0xE0);
            frame.extend((blocks as u64 * BLOCK as u64).to_le_bytes());
        } else {
            // No content size; a 128 KiB window.
            frame.extend([0x00, 0x38]);
        }
        for i in 0..blocks {
            let header = u32::from(i + 1 == blocks) | (1 << 1) | ((BLOCK as u32) << 3);
            frame.extend(&header.to_le_bytes()[..3]);
            frame.push(0x41);
        }
        frame
    }

    /// The filter run backwards on `frame` as HDF5 runs it: the frame in a
    /// buffer of `buf_size` bytes, or of the frame's own size if larger.
    fn decompress(frame: &[u8], buf_size: usize) -> Option<Vec<u8>> {
        let _held = crate::HDF5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut size = buf_size.max(frame.len());
        // SAFETY: the buffer is HDF5's, as the pipeline's is, and holds the
        // frame; the filter frees or keeps it and the result is freed here.
        unsafe {
            let mut buf = H5allocate_memory(size, 0);
            assert!(!buf.is_null());
            std::ptr::copy_nonoverlapping(frame.as_ptr(), buf.cast::<u8>(), frame.len());
            let n = zstd_filter(H5Z_FLAG_REVERSE, 0, std::ptr::null(), frame.len(), &mut size, &mut buf);
            let out = (n > 0).then(|| std::slice::from_raw_parts(buf.cast::<u8>(), n).to_vec());
            H5free_memory(buf);
            out
        }
    }

    /// A frame of 64 MiB from 2 KiB, for a 64 KiB chunk, is refused whether
    /// its header states the size (before anything is allocated) or not
    /// (once it has decoded past the ceiling): it was decoded whole.
    #[test]
    fn s14_2_a_frame_larger_than_its_chunk_is_refused() {
        let chunk = 64 * 1024;
        for declared in [false, true] {
            let bomb = rle_frame(512, declared);
            assert!(bomb.len() < 4096);
            assert_eq!(decompress(&bomb, chunk), None, "declared: {declared}");
        }
        assert_eq!(output_ceiling(chunk), 3 * chunk);
        assert_eq!(output_ceiling(usize::MAX), usize::MAX);
    }

    /// What a chunk holds still decodes, whether the frame states its size
    /// (as `hdf5plugin` writes it) or not, and a little more than the chunk
    /// --- what a filter applied before this one may add --- is no bomb.
    #[test]
    fn s14_2_a_frame_of_its_chunk_still_decodes() {
        let chunk: Vec<u8> = (0..BLOCK as u32).map(|i| (i % 251) as u8).collect();
        let mut compressed = vec![0u8; unsafe { zstd::ZSTD_compressBound(chunk.len()) }];
        // SAFETY: both buffers are valid for the lengths given.
        let written = unsafe {
            zstd::ZSTD_compress(compressed.as_mut_ptr().cast(), compressed.len(), chunk.as_ptr().cast(), chunk.len(), 3)
        };
        assert_eq!(unsafe { zstd::ZSTD_isError(written) }, 0);
        compressed.truncate(written);
        assert_eq!(decompress(&compressed, chunk.len()).as_deref(), Some(&chunk[..]));
        for declared in [false, true] {
            assert_eq!(decompress(&rle_frame(1, declared), BLOCK), Some(vec![0x41; BLOCK]));
            assert_eq!(decompress(&rle_frame(2, declared), BLOCK).map(|v| v.len()), Some(2 * BLOCK));
        }
    }
}
