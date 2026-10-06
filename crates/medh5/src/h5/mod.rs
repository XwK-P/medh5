//! HDF5 plumbing: attribute codecs, dynamic-dtype datasets, atomic create,
//! copy-on-write and the low-level operations the high-level API lacks.
//!
//! Everything here is about *how* values reach HDF5, never about what they
//! mean.  Spec §2.5 fixes the attribute encoding; spec §14.4 fixes the write
//! model (atomic create, copy-on-write amend, unknown-object preservation).

pub mod attrs;
pub mod data;
pub mod file;
pub mod ops;

pub use attrs::AttrValue;
pub use hdf5::{Dataset, File, Group, Location};

/// Prepare the HDF5 library for MEDH5: register the Blosc2 filter.
///
/// Idempotent and cheap after the first call; every entry point that opens or
/// creates a file calls it, so a caller never has to.
pub fn init() {
    hdf5::sync::sync(|| {
        // SAFETY: the HDF5 global lock is held for the duration.
        let ok = unsafe { medh5_sys::register_blosc2_filter() };
        if !ok {
            // Reading a Blosc2 dataset will then fail with HDF5's own error,
            // which names the missing filter; nothing is gained by panicking.
        }
    });
}

/// Run raw HDF5 C calls under the library's global lock.
pub(crate) fn locked<T>(f: impl FnOnce() -> T) -> T {
    hdf5::sync::sync(f)
}

/// An object's path relative to a sample root (`/samples/x/images/CT` with
/// root `/samples/x` gives `images/CT`).
pub fn relative_name(name: &str, root: &str) -> String {
    let base = if root == "/" { "/".to_string() } else { format!("{}/", root.trim_end_matches('/')) };
    match name.strip_prefix(&base) {
        Some(rest) => rest.to_string(),
        None => name.trim_start_matches('/').to_string(),
    }
}

/// The last component of an HDF5 path.
pub fn basename(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}
