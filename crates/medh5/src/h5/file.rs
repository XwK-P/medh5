//! Opening files, and the atomic write model of spec §14.4.
//!
//! **Create** is atomic: write to a sibling temporary file, fsync it,
//! rename it over the target, fsync the directory.  A reader never observes a
//! partially written file and a crash leaves the previous file intact.  Every
//! copy-on-write command --- amend, `scrub --apply`, `fix`, `recompress` ---
//! goes through [`AtomicFile`], and when the target exists its permission bits
//! are carried onto the replacement: these are the commands most likely to be
//! pointed at data whose mode *is* its access control.

use std::fs;
use std::path::{Path, PathBuf};

use hdf5::file::LibraryVersion;

use super::ops::check_self_contained;
use crate::json::repr_str;
use crate::{Error, Result};

/// Open an HDF5 file read-only, mapping failures onto `MEDH5FileError`.
///
/// Every file is also checked to be self-contained before it is handed back
/// (see [`check_self_contained`]), because this is the one door that readers,
/// validators and every copy-on-write path open files through.
pub fn open_read(path: &Path) -> Result<hdf5::File> {
    super::init();
    let handle = hdf5::File::open(path)
        .map_err(|e| Error::File(format!("failed to open {}: {}", repr_str(&path.to_string_lossy()), e)))?;
    check_self_contained(&handle, Some(path))?;
    Ok(handle)
}

/// Create (truncate) an HDF5 file with the format's library-version bounds.
///
/// The lower bound is HDF5 1.10, the oldest container the specification
/// promises to be readable by; the upper bound is the library's latest, so
/// features newer than 1.10 are used only where an object needs them.
pub fn create_truncate(path: &Path) -> Result<hdf5::File> {
    super::init();
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .map_err(|e| Error::File(format!("failed to create {}: {}", repr_str(&path.to_string_lossy()), e)))
}

/// A sibling name no other writer in any process is using.
fn temporary_name(target: &Path) -> PathBuf {
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let unique: u32 = {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut h = RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
        );
        h.finish() as u32
    };
    target.with_file_name(format!(".{name}.tmp-{}-{unique:08x}", std::process::id()))
}

/// The permission bits of `target`, or `None` when it does not exist.
fn existing_mode(target: &Path) -> Option<u32> {
    let meta = fs::metadata(target).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Some(meta.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        Some(if meta.permissions().readonly() { 0o444 } else { 0o666 })
    }
}

/// Create the temporary file before any data, readable by its owner only when
/// it will replace an existing file.
fn precreate(tmp: &Path, mode: Option<u32>) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if mode.is_none() { 0o666 } else { 0o600 });
    }
    let _ = mode;
    options.open(tmp)?;
    Ok(())
}

fn set_mode(path: &Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    {
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_readonly(mode & 0o222 == 0);
            let _ = fs::set_permissions(path, perms);
        }
    }
}

/// Flush a closed file to stable storage before it is renamed into place.
fn fsync_path(path: &Path) -> Result<()> {
    let file = fs::OpenOptions::new().read(true).write(cfg!(windows)).open(path)?;
    file.sync_all()?;
    Ok(())
}

fn fsync_dir(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(handle) = fs::File::open(dir) {
            let _ = handle.sync_all();
        }
    }
    let _ = dir;
}

/// Close `file` and every object opened through it --- what h5py's
/// `File.close()` does.
///
/// HDF5 keeps a file open while any object in it is, so a dataset or group a
/// caller still holds would otherwise keep a written file open and unflushed
/// under a name that has already been renamed away, and keep a read file
/// locked against the next writer.  Those objects become invalid instead:
/// using one fails, and dropping one does nothing.  `flush` first, so a
/// writer learns of a failed write rather than losing it in the close.
pub fn close_everything(file: hdf5::File, flush: bool) -> Result<()> {
    use crate::h5sys::h5f::{self, H5F_scope_t};
    use crate::h5sys::h5i;
    let fid = file.id();
    // Released below, so the handle's own drop must not release it again.
    std::mem::forget(file);
    super::locked(|| unsafe {
        let flushed = !flush || h5f::H5Fflush(fid, H5F_scope_t::H5F_SCOPE_LOCAL) >= 0;
        let release = |types: std::os::raw::c_uint| {
            let count = h5f::H5Fget_obj_count(fid, types);
            if count <= 0 {
                return;
            }
            let mut ids = vec![0 as h5i::hid_t; count as usize];
            let found = h5f::H5Fget_obj_ids(fid, types, ids.len(), ids.as_mut_ptr());
            for &id in ids.iter().take(found.max(0) as usize) {
                while h5i::H5Iis_valid(id) > 0 {
                    if h5i::H5Idec_ref(id) <= 0 {
                        break;
                    }
                }
            }
        };
        release(h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_DATASET | h5f::H5F_OBJ_GROUP | h5f::H5F_OBJ_DATATYPE | h5f::H5F_OBJ_ATTR);
        // The file identifiers last: this one, and any handed out for it.
        release(h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_FILE);
        if h5i::H5Iis_valid(fid) > 0 {
            while h5i::H5Idec_ref(fid) > 0 {}
        }
        if flushed {
            Ok(())
        } else {
            Err(Error::Io(format!("could not flush the file: {}", super::ops::hdf5_error_text())))
        }
    })
}

/// An HDF5 file being written to a temporary sibling of its target.
///
/// [`commit`](AtomicFile::commit) closes it and atomically replaces the
/// target; dropping it without committing removes the temporary file and leaves
/// the target untouched.
pub struct AtomicFile {
    target: PathBuf,
    tmp: PathBuf,
    mode: Option<u32>,
    handle: Option<hdf5::File>,
}

impl AtomicFile {
    /// Start writing `target`.
    pub fn create(target: &Path) -> Result<AtomicFile> {
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let tmp = temporary_name(target);
        let mode = existing_mode(target);
        precreate(&tmp, mode)?;
        match create_truncate(&tmp) {
            Ok(handle) => Ok(AtomicFile { target: target.to_path_buf(), tmp, mode, handle: Some(handle) }),
            Err(err) => {
                let _ = fs::remove_file(&tmp);
                Err(err)
            }
        }
    }

    /// The file being written.
    pub fn handle(&self) -> &hdf5::File {
        self.handle.as_ref().expect("an AtomicFile is open until committed")
    }

    /// The path that will be replaced.
    pub fn target(&self) -> &Path {
        &self.target
    }

    /// The temporary path being written.
    pub fn temporary(&self) -> &Path {
        &self.tmp
    }

    /// Close and atomically move the file into place.
    ///
    /// Everything opened through the file is closed with it, so nothing a
    /// caller kept can hold the written file open across the rename.
    pub fn commit(mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            close_everything(handle, true)?;
        }
        let result = (|| -> Result<()> {
            if let Some(mode) = self.mode {
                set_mode(&self.tmp, mode);
            }
            fsync_path(&self.tmp)?;
            fs::rename(&self.tmp, &self.target)?;
            if let Some(parent) = self.target.parent() {
                fsync_dir(if parent.as_os_str().is_empty() { Path::new(".") } else { parent });
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&self.tmp);
        }
        // Nothing is left for Drop to clean up.
        self.tmp = PathBuf::new();
        result
    }

    /// Discard the in-progress file, leaving any existing one untouched.
    pub fn abort(mut self) {
        self.discard();
    }

    fn discard(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = close_everything(handle, false);
        }
        if !self.tmp.as_os_str().is_empty() && self.tmp.exists() {
            let _ = fs::remove_file(&self.tmp);
        }
        self.tmp = PathBuf::new();
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        self.discard();
    }
}

/// Rebuild a file from an existing one, atomically.
///
/// The source is opened read-only and handed to `build` with the new file; the
/// source handle is closed **before** the replace, which matters on Windows,
/// where a file that is still open cannot be replaced.  `target` defaults to
/// `source` --- the rewrite-in-place case.
pub fn atomic_rewrite<T>(
    source: &Path,
    target: Option<&Path>,
    build: impl FnOnce(&hdf5::File, &hdf5::File) -> Result<T>,
) -> Result<T> {
    let dst_path = target.unwrap_or(source);
    let src = open_read(source)?;
    let out = AtomicFile::create(dst_path)?;
    let value = build(&src, out.handle())?;
    // Everything the build opened in the source, closed with it: a source
    // still open cannot be replaced on Windows.
    close_everything(src, false)?;
    out.commit()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a caller kept from a closed file is invalid, not a reason the file
    /// stays open: the file reopens for writing at once.
    #[test]
    fn s14_4_close_everything_releases_what_was_opened_through_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.h5");
        {
            let file = create_truncate(&path).unwrap();
            file.new_dataset::<i32>().shape([3]).create("d").unwrap().write(&[1, 2, 3]).unwrap();
        }
        let file = open_read(&path).unwrap();
        let kept = file.dataset("d").unwrap();
        let group = file.as_group().unwrap();
        close_everything(file, false).unwrap();
        assert!(!kept.is_valid() && !group.is_valid());
        assert!(super::super::alive(&kept).is_err());
        let again = hdf5::File::open_rw(&path).unwrap();
        assert_eq!(again.dataset("d").unwrap().read_raw::<i32>().unwrap(), vec![1, 2, 3]);
        drop((kept, group));
    }
}
