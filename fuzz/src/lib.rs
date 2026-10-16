//! What the fuzz targets keep off a path HDF5 cannot leave.
//!
//! A finding the engine can bound is bounded in the engine, with a
//! regression test there.  What is here has no such bound: the work is
//! inside one HDF5 call, before anything the engine could check.  Each
//! exclusion is the narrowest description of its inputs, and goes when
//! HDF5 fixes the defect.  The module needs nothing but `std`, so its tests
//! run without the fuzz build (`rustc --test`, as `fuzz.yml` does).

/// Whether `file` holds a global heap collection that HDF5 2.2.0 (through
/// hdf5-metno-src 0.10.4) steps through without end: a free-space object,
/// index 0, stating a size of 0, with room for an object header after it.
/// Its decoder then moves on by that size, which leaves it where it was.
///
/// Every collection signature anywhere in `file` is tried, with every
/// length size HDF5 accepts, and bytes past the end of `file` read as zeros
/// (as HDF5's default driver reads them).  So this answers yes for more
/// inputs than HDF5 would hang on, never for fewer.
pub fn endless_global_heap(file: &[u8]) -> bool {
    file.windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"GCOL")
        .any(|(at, _)| [2, 4, 8].into_iter().any(|length| walks_forever(file, at as u64, length)))
}

fn byte(file: &[u8], at: u64) -> u8 {
    usize::try_from(at).ok().and_then(|at| file.get(at)).copied().unwrap_or(0)
}

/// A little-endian length of `length` bytes.
fn decode(file: &[u8], at: u64, length: u64) -> u64 {
    (0..length).rev().fold(0, |n, i| n << 8 | u64::from(byte(file, at.saturating_add(i))))
}

/// HDF5's padding: every header and object starts on 8 bytes.
fn align(n: u64) -> Option<u64> {
    Some(n.checked_add(7)? & !7)
}

/// The walk `H5HG__cache_heap_deserialize` takes over the collection at `at`.
fn walks_forever(file: &[u8], at: u64, length: u64) -> bool {
    // The collection header: signature, version 1, three reserved bytes,
    // then the collection's size, at least 4096 bytes.
    let size = decode(file, at + 8, length);
    if byte(file, at + 4) != 1 || size < 4096 {
        return false;
    }
    // Object headers: index, reference count, four reserved bytes, size.
    let header = (8 + length + 7) & !7;
    let Some(end) = at.checked_add(size) else { return false };
    let mut p = at + header;
    while p < end {
        if p + header > end {
            // Too small for a header: the rest is free space, and the walk ends.
            return false;
        }
        let index = u16::from(byte(file, p)) | u16::from(byte(file, p + 1)) << 8;
        let stated = decode(file, p + 8, length);
        // An object's header, then its data padded; free space states its
        // whole size, header included.
        let need = if index > 0 { align(stated).and_then(|s| s.checked_add(header)) } else { Some(stated) };
        match need.and_then(|need| Some((need, p.checked_add(need)?))) {
            Some((0, _)) => return true,
            Some((_, next)) if next <= end => p = next,
            _ => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::endless_global_heap;

    /// A collection of `size` bytes, with 8-byte lengths, whose objects are
    /// `(index, stated size, data length)`.
    fn collection(size: u64, objects: &[(u16, u64, usize)]) -> Vec<u8> {
        let mut bytes = b"GCOL\x01\0\0\0".to_vec();
        bytes.extend(size.to_le_bytes());
        for &(index, stated, data) in objects {
            bytes.extend(index.to_le_bytes());
            bytes.extend([1, 0, 0, 0, 0, 0]);
            bytes.extend(stated.to_le_bytes());
            bytes.resize(bytes.len() + data, 0);
        }
        bytes.resize(size as usize, 0);
        bytes
    }

    #[test]
    fn a_zero_sized_free_space_object_is_endless() {
        let mut file = b"some bytes before it".to_vec();
        file.extend(collection(4096, &[(1, 5, 8), (0, 0, 0)]));
        assert!(endless_global_heap(&file));
    }

    #[test]
    fn a_heap_as_hdf5_writes_it_is_not() {
        // One string of 5 bytes, padded to 8, then the free space: the rest
        // of the collection, its header included.
        let rest = 4096 - 16 - (16 + 8);
        assert!(!endless_global_heap(&collection(4096, &[(1, 5, 8), (0, rest, 0)])));
        // A collection whose objects fill it, and the tail too small for a
        // header.
        assert!(!endless_global_heap(&collection(4096, &[(1, 4096 - 16 - 16 - 8, 4096 - 16 - 16 - 8)])));
    }

    #[test]
    fn a_collection_hdf5_refuses_is_not() {
        let mut small = collection(4096, &[(0, 0, 0)]);
        small[8..16].copy_from_slice(&4095u64.to_le_bytes());
        assert!(!endless_global_heap(&small));
        let mut version = collection(4096, &[(0, 0, 0)]);
        version[4] = 2;
        assert!(!endless_global_heap(&version));
    }

    #[test]
    fn a_collection_past_the_end_of_the_file_reads_zeros() {
        // The collection states 4096 bytes and the file stops after its
        // header: the first object reads as free space of size 0.
        assert!(endless_global_heap(&collection(4096, &[])[..16]));
    }

    #[test]
    fn four_byte_lengths_are_aligned_as_hdf5_aligns_them() {
        // Header and object header both 16 bytes (12, padded); the zero
        // free-space object sits at 16, not at 12.
        let mut file = b"GCOL\x01\0\0\0".to_vec();
        file.extend(4096u32.to_le_bytes());
        file.resize(16, 0);
        file.extend(1u16.to_le_bytes());
        file.extend([1, 0, 0, 0, 0, 0]);
        file.extend(3u32.to_le_bytes());
        file.resize(16 + 16 + 8, 0);
        file.extend([0, 0, 1, 0, 0, 0, 0, 0]);
        file.extend(0u32.to_le_bytes());
        file.resize(4096, 0xFF);
        assert!(endless_global_heap(&file));
    }
}
