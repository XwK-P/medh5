//! Opening files, and the atomic write model of spec §14.4.
//!
//! **Create** is atomic: write to a sibling temporary file, fsync it,
//! rename it over the target, fsync the directory.  A reader never observes a
//! partially written file and a crash leaves the previous file intact.  Every
//! copy-on-write command --- amend, `scrub --apply`, `fix`, `recompress` ---
//! goes through [`AtomicFile`], and when the target exists its permission bits
//! are carried onto the replacement: these are the commands most likely to be
//! pointed at data whose mode *is* its access control.  A target that is a
//! symbolic link is followed: the file it names is replaced, beside itself,
//! and the link kept.

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
    let failed = |why: &dyn std::fmt::Display| {
        Error::File(format!("failed to open {}: {why}", repr_str(&path.to_string_lossy())))
    };
    // Asked of the system first: HDF5's report of a missing file is its error
    // stack, which carries the time of the call, and a report that differs
    // from run to run is no report to compare (`validate --json`).
    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => return Err(failed(&"it is a directory, not a file")),
        Ok(_) => {}
        Err(err) => return Err(failed(&os_reason(&err))),
    }
    let handle = hdf5::File::open(path).map_err(|e| failed(&e))?;
    check_self_contained(&handle, Some(path))?;
    Ok(handle)
}

/// What the system said went wrong, without the error number: "no such file
/// or directory" alike on every platform, and the system's own words
/// otherwise.
fn os_reason(err: &std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::NotFound {
        return "no such file or directory".into();
    }
    let full = err.to_string();
    match err.raw_os_error() {
        Some(code) => full.strip_suffix(&format!(" (os error {code})")).unwrap_or(&full).to_string(),
        None => full,
    }
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

/// The most links a chain is followed through, as Linux bounds one.
const MAX_LINKS: usize = 40;

/// The file a replace of `target` must replace: where its symbolic links
/// lead, so the link stays a link and the file it names is the one rewritten.
///
/// The temporary file was made beside the link and renamed over it, which
/// turned the link into an independent copy and left the file it named
/// untouched --- for `scrub --apply`, the identifying original, while the user
/// believed it scrubbed.  Only the last component is followed: a directory
/// reached through a link is the directory the rename happens in either way.
/// A link that leads nowhere names the file to create, as `open` takes it.
fn resolve_links(target: &Path) -> Result<PathBuf> {
    let mut path = target.to_path_buf();
    for _ in 0..MAX_LINKS {
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                if !may_follow(&path, &meta) {
                    return Err(Error::File(format!(
                        "{} is a symbolic link another user owns, in a directory anyone can write to; it is not \
                         followed, as the system would not follow it to open the file",
                        repr_str(&path.to_string_lossy())
                    )));
                }
                let leads_to = fs::read_link(&path).map_err(|e| crate::error::os_error(&e, &path))?;
                // `join` keeps an absolute target as it is.
                path = match path.parent() {
                    Some(dir) => dir.join(leads_to),
                    None => leads_to,
                };
            }
            _ => return Ok(path),
        }
    }
    Err(Error::File(format!(
        "{} leads through more than {MAX_LINKS} symbolic links",
        repr_str(&target.to_string_lossy())
    )))
}

/// Whether a write may follow the link `link`: not one in a sticky directory
/// anyone can write to (`/tmp`) that neither the user nor the directory's owner
/// owns --- the rule the system applies to `open` (Linux's
/// `fs.protected_symlinks`), which a replace that resolves the link itself
/// would otherwise step around.
#[cfg(unix)]
fn may_follow(link: &Path, meta: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    let dir = link.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let Ok(dir_meta) = fs::metadata(dir) else { return true };
    // SAFETY: `geteuid` has no preconditions.
    let user = unsafe { libc::geteuid() };
    dir_meta.mode() & 0o1002 != 0o1002 || meta.uid() == user || meta.uid() == dir_meta.uid()
}

#[cfg(not(unix))]
fn may_follow(_link: &Path, _meta: &fs::Metadata) -> bool {
    true
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
///
/// Windows flushes only through a handle that may write, so this runs before
/// the file is made read-only.
fn fsync_path(path: &Path) -> Result<()> {
    let file = fs::OpenOptions::new().read(true).write(cfg!(windows)).open(path)?;
    file.sync_all()?;
    Ok(())
}

/// Rename `from` over `to`.
///
/// Windows refuses to replace a read-only file, so a read-only target is made
/// writable for the rename --- the replacement carries the mode back --- and
/// read-only again if the rename fails.
fn replace(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        if let Ok(meta) = fs::metadata(to) {
            let kept = meta.permissions();
            if kept.readonly() {
                let mut writable = kept.clone();
                // Windows only, where this clears the read-only attribute.
                #[allow(clippy::permissions_set_readonly_false)]
                writable.set_readonly(false);
                fs::set_permissions(to, writable)?;
                let renamed = fs::rename(from, to);
                if renamed.is_err() {
                    let _ = fs::set_permissions(to, kept);
                }
                return renamed;
            }
        }
    }
    fs::rename(from, to)
}

/// Remove a temporary file, which may already be read-only: Windows does not
/// delete a read-only file.
fn remove_temporary(path: &Path) {
    #[cfg(windows)]
    {
        if let Ok(meta) = fs::metadata(path) {
            let mut writable = meta.permissions();
            if writable.readonly() {
                // Windows only, where this clears the read-only attribute.
                #[allow(clippy::permissions_set_readonly_false)]
                writable.set_readonly(false);
                let _ = fs::set_permissions(path, writable);
            }
        }
    }
    let _ = fs::remove_file(path);
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
///
/// A close that fails is an error too --- an object whose cached data could
/// not be written, the file's own last write --- so a commit never renames a
/// file whose close failed.
pub fn close_everything(file: hdf5::File, flush: bool) -> Result<()> {
    use crate::h5sys::h5f::{self, H5F_scope_t};
    let fid = file.id();
    // Released below, so the handle's own drop must not release it again.
    std::mem::forget(file);
    super::locked(|| unsafe {
        let flushed = !flush || h5f::H5Fflush(fid, H5F_scope_t::H5F_SCOPE_LOCAL) >= 0;
        // Read now: the calls below clear HDF5's record of the failure.
        if !flushed {
            let why = super::ops::hdf5_error_text();
            release_all(fid);
            return Err(Error::Io(format!("could not flush the file: {why}")));
        }
        match release_all(fid) {
            None => Ok(()),
            Some(why) => Err(Error::Io(format!("could not close the file: {why}"))),
        }
    })
}

/// Release every identifier open through the file `fid`, and `fid` last;
/// what HDF5 said about the first release that failed, if one did.  The
/// caller holds the HDF5 lock.
///
/// # Safety
///
/// `fid` is a file identifier the caller owns.
unsafe fn release_all(fid: crate::h5sys::h5i::hid_t) -> Option<String> {
    use crate::h5sys::h5f;
    use crate::h5sys::h5i;
    let mut failed = None;
    // A failed release leaves the identifier open, so each is released until
    // it is gone or HDF5 refuses.
    let mut release = |id: h5i::hid_t| unsafe {
        while h5i::H5Iis_valid(id) > 0 {
            let left = h5i::H5Idec_ref(id);
            if left < 0 && failed.is_none() {
                failed = Some(super::ops::hdf5_error_text());
            }
            if left <= 0 {
                break;
            }
        }
    };
    let opened = |types: std::os::raw::c_uint| unsafe {
        let count = h5f::H5Fget_obj_count(fid, types);
        if count <= 0 {
            return Vec::new();
        }
        let mut ids = vec![0 as h5i::hid_t; count as usize];
        let found = h5f::H5Fget_obj_ids(fid, types, ids.len(), ids.as_mut_ptr());
        ids.truncate(found.max(0) as usize);
        ids
    };
    for id in opened(
        h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_DATASET | h5f::H5F_OBJ_GROUP | h5f::H5F_OBJ_DATATYPE | h5f::H5F_OBJ_ATTR,
    ) {
        release(id);
    }
    // The file identifiers last: this one, and any handed out for it.
    for id in opened(h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_FILE) {
        release(id);
    }
    release(fid);
    #[cfg(test)]
    if CLOSE_FAILS.with(|fails| fails.replace(false)) {
        failed.get_or_insert_with(|| "H5Fclose(): the close failed (a test made it)".into());
    }
    failed
}

/// What a test runs on the target just before a commit's rename.
#[cfg(test)]
type RenameHook = Box<dyn FnMut(&Path)>;

#[cfg(test)]
thread_local! {
    static BEFORE_RENAME: std::cell::RefCell<Option<RenameHook>> = const { std::cell::RefCell::new(None) };
    /// Set by a test to have the next close report the failure HDF5 reports
    /// of a close it could not finish.
    static CLOSE_FAILS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
    /// Start writing `target`: through its symbolic links, if it is one
    /// (`resolve_links`), so the file it names is the one replaced.
    pub fn create(target: &Path) -> Result<AtomicFile> {
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let target = resolve_links(target)?;
        let tmp = temporary_name(&target);
        let mode = existing_mode(&target);
        precreate(&tmp, mode)?;
        match create_truncate(&tmp) {
            Ok(handle) => Ok(AtomicFile { target, tmp, mode, handle: Some(handle) }),
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

    /// The path that will be replaced: the file the target given named,
    /// its symbolic links followed.
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
            // Flushed before the mode is set: a read-only file refuses the
            // writable handle Windows flushes through.
            fsync_path(&self.tmp)?;
            if let Some(mode) = self.mode {
                set_mode(&self.tmp, mode);
            }
            #[cfg(test)]
            BEFORE_RENAME.with(|hook| {
                if let Some(hook) = hook.borrow_mut().as_mut() {
                    hook(&self.target)
                }
            });
            replace(&self.tmp, &self.target)?;
            if let Some(parent) = self.target.parent() {
                fsync_dir(if parent.as_os_str().is_empty() { Path::new(".") } else { parent });
            }
            Ok(())
        })();
        if result.is_err() {
            remove_temporary(&self.tmp);
        }
        // Nothing is left for Drop to clean up.
        self.tmp = PathBuf::new();
        result
    }

    /// Rewrite the file being built into a fresh temporary file, `copy`
    /// taking the old root to the new one, and carry on with that: what the
    /// copy does not reach is gone from the bytes.  HDF5 never reclaims an
    /// unlinked object's space, so a group removed from a file stays in it
    /// until the file is rewritten.
    pub fn rebuild(mut self, copy: impl FnOnce(&hdf5::Group, &hdf5::Group) -> Result<()>) -> Result<AtomicFile> {
        if let Some(handle) = self.handle.take() {
            close_everything(handle, true)?;
        }
        let fresh = temporary_name(&self.target);
        precreate(&fresh, self.mode)?;
        let built = (|| -> Result<hdf5::File> {
            let src = open_read(&self.tmp)?;
            let dst = create_truncate(&fresh)?;
            copy(&src.as_group()?, &dst.as_group()?)?;
            close_everything(src, false)?;
            Ok(dst)
        })();
        match built {
            Ok(handle) => {
                let _ = fs::remove_file(&self.tmp);
                self.tmp = fresh;
                self.handle = Some(handle);
                Ok(self)
            }
            Err(err) => {
                let _ = fs::remove_file(&fresh);
                Err(err)
            }
        }
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

    /// A small file holding `value`, written through [`AtomicFile`].
    fn write_value(target: &Path, value: i32) -> Result<()> {
        let out = AtomicFile::create(target)?;
        out.handle().new_dataset::<i32>().shape([1]).create("v")?.write(&[value])?;
        out.commit()
    }

    fn read_value(path: &Path) -> i32 {
        open_read(path).unwrap().dataset("v").unwrap().read_raw::<i32>().unwrap()[0]
    }

    /// Nothing temporary is left in `dir`.
    fn no_temporaries(dir: &Path) -> bool {
        fs::read_dir(dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains(".tmp-"))
    }

    /// A missing file is reported in the system's words, the same on every
    /// run: HDF5's error stack carries the time of the call, which made
    /// `validate --json` differ from run to run.
    #[test]
    fn s14_4_a_missing_file_is_reported_the_same_on_every_run() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.medh5");
        let first = open_read(&missing).unwrap_err().to_string();
        let second = open_read(&missing).unwrap_err().to_string();
        assert_eq!(first, second);
        assert_eq!(
            first,
            format!("failed to open {}: no such file or directory", repr_str(&missing.to_string_lossy()))
        );
        let err = open_read(dir.path()).unwrap_err();
        assert!(matches!(err, Error::File(_)));
        assert!(err.to_string().ends_with(": it is a directory, not a file"), "{err}");
    }

    /// A replace through a symbolic link replaces the file the link names
    /// and keeps the link, whether the link is absolute or relative, or one
    /// of a chain: the link was replaced by an independent copy, and the file
    /// it named left as it was --- for `scrub --apply`, the identifying
    /// original.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_replace_through_a_symlink_keeps_the_link() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let farm = dir.path().join("farm");
        fs::create_dir_all(&data).unwrap();
        fs::create_dir_all(&farm).unwrap();
        let real = data.join("real.medh5");
        write_value(&real, 1).unwrap();
        let absolute = farm.join("absolute.medh5");
        symlink(&real, &absolute).unwrap();
        let relative = farm.join("relative.medh5");
        symlink("../data/real.medh5", &relative).unwrap();
        let chained = farm.join("chained.medh5");
        symlink("relative.medh5", &chained).unwrap();
        for (value, link) in [(2, &absolute), (3, &relative), (4, &chained)] {
            let inode = fs::metadata(&real).unwrap();
            write_value(link, value).unwrap();
            assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink(), "{link:?} is no longer a link");
            assert_eq!(read_value(&real), value, "{link:?}");
            assert_eq!(read_value(link), value, "{link:?}");
            use std::os::unix::fs::MetadataExt;
            assert_ne!(fs::metadata(&real).unwrap().ino(), inode.ino(), "replaced, not written in place");
            assert!(no_temporaries(&data) && no_temporaries(&farm));
        }
        // The replacement takes the real file's mode, as it always did.
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&real, fs::Permissions::from_mode(0o640)).unwrap();
        write_value(&chained, 5).unwrap();
        assert_eq!(fs::metadata(&real).unwrap().permissions().mode() & 0o777, 0o640);
    }

    /// A link that leads nowhere names the file to create, as `open` takes
    /// it; a loop of links is refused, not followed for ever.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_dangling_link_is_created_through_and_a_loop_refused() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("link.medh5");
        symlink("made.medh5", &link).unwrap();
        write_value(&link, 7).unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(read_value(&dir.path().join("made.medh5")), 7);
        let (a, b) = (dir.path().join("a.medh5"), dir.path().join("b.medh5"));
        symlink("b.medh5", &a).unwrap();
        symlink("a.medh5", &b).unwrap();
        let err = write_value(&a, 8).unwrap_err();
        assert!(err.to_string().contains("more than 40 symbolic links"), "{err}");
        assert!(no_temporaries(dir.path()));
    }

    /// A link in a sticky directory anyone can write to is followed only
    /// when the user or the directory's owner owns it, as the system follows
    /// it to open a file there; otherwise a link planted in `/tmp` could
    /// point a write at any file of the user's.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_link_another_user_planted_in_a_shared_directory_is_not_followed() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o1777)).unwrap();
        let mine = dir.path().join("mine.medh5");
        write_value(&mine, 1).unwrap();
        let link = shared.join("out.medh5");
        symlink(&mine, &link).unwrap();
        write_value(&link, 2).unwrap();
        assert_eq!(read_value(&mine), 2, "the user's own link is followed");
        // Only root can give a link to another user.
        if unsafe { libc::geteuid() } == 0 {
            let path = std::ffi::CString::new(link.as_os_str().as_encoded_bytes()).unwrap();
            assert_eq!(unsafe { libc::lchown(path.as_ptr(), 4242, 4242) }, 0);
            let err = write_value(&link, 3).unwrap_err();
            assert!(err.to_string().contains("another user owns"), "{err}");
            assert_eq!(read_value(&mine), 2);
            assert!(no_temporaries(&shared) && no_temporaries(dir.path()));
        }
    }

    /// A read-only file is replaced, and stays read-only.  The temporary
    /// file was made read-only and *then* opened for writing to flush it,
    /// which Windows refuses, and Windows does not rename over a read-only
    /// file.
    #[test]
    fn s14_4_a_read_only_file_is_replaced_and_stays_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ro.medh5");
        write_value(&path, 1).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&path, perms).unwrap();
        write_value(&path, 2).unwrap();
        assert_eq!(read_value(&path), 2);
        assert!(fs::metadata(&path).unwrap().permissions().readonly());
        assert!(no_temporaries(dir.path()));
        set_mode(&path, 0o644);
    }

    /// A close that fails stops the commit before the rename: the target is
    /// left as it was and the temporary file removed.  The final close's
    /// result was ignored, so a file whose close failed could be renamed
    /// into place.
    #[test]
    fn s14_4_a_failed_close_is_not_renamed_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.medh5");
        write_value(&path, 1).unwrap();
        CLOSE_FAILS.with(|fails| fails.set(true));
        let err = write_value(&path, 2).unwrap_err();
        assert!(err.to_string().starts_with("could not close the file: "), "{err}");
        assert_eq!(read_value(&path), 1);
        assert!(no_temporaries(dir.path()));
        CLOSE_FAILS.with(|fails| fails.set(true));
        let file = create_truncate(&dir.path().join("d.h5")).unwrap();
        assert!(close_everything(file, false).is_err());
    }

    #[test]
    fn s14_4_temporary_names_are_unique_within_a_process() {
        let names: std::collections::HashSet<PathBuf> = (0..50).map(|_| temporary_name(Path::new("x.medh5"))).collect();
        assert_eq!(names.len(), 50);
        assert!(names.iter().all(|n| n.to_string_lossy().starts_with(".x.medh5.tmp-")));
    }

    /// `fsync_path` opens a writable descriptor on Windows, where a C runtime
    /// in text mode treats a trailing 0x1A as an end-of-file mark and strips
    /// it: the 1.x Windows job found four samples in eleven hundred one byte
    /// short.
    #[test]
    fn fsync_leaves_a_trailing_control_z_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ctrlz.bin");
        let payload: Vec<u8> = (0..=255u8).chain([0x1a]).collect();
        fs::write(&path, &payload).unwrap();
        fsync_path(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), payload);
    }

    /// The HDF5 file identifiers open on `path` in this process.
    fn open_on(path: &Path) -> usize {
        use crate::h5sys::h5f;
        use crate::h5sys::h5i::hid_t;
        let want = path.to_string_lossy().into_owned();
        super::super::locked(|| unsafe {
            let all = h5f::H5F_OBJ_ALL as hid_t;
            let count = h5f::H5Fget_obj_count(all, h5f::H5F_OBJ_FILE);
            let mut ids = vec![0 as hid_t; count.max(0) as usize];
            let found = h5f::H5Fget_obj_ids(all, h5f::H5F_OBJ_FILE, ids.len(), ids.as_mut_ptr());
            ids.iter()
                .take(found.max(0) as usize)
                .filter(|&&id| {
                    let mut buf = vec![0u8; 4096];
                    let n = h5f::H5Fget_name(id, buf.as_mut_ptr().cast(), buf.len());
                    n > 0 && buf[..n as usize] == *want.as_bytes()
                })
                .count()
        })
    }

    /// The source of a rewrite in place is closed before the rename.
    ///
    /// `with open_h5(...) as src, atomic_h5(...) as dst:` exits right to left,
    /// so 1.x once replaced the file while its source was still open.  POSIX
    /// allows that, so it passed on Linux and macOS; Windows refuses to replace
    /// an open file, so every rewrite in place --- a repack, `recompress`
    /// without `out` --- failed there.  This asserts the ordering directly
    /// rather than relying on a platform to notice.
    #[test]
    fn s14_4_the_source_is_closed_before_the_replace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rw.h5");
        {
            let file = create_truncate(&path).unwrap();
            file.new_dataset::<i32>().shape([3]).create("d").unwrap().write(&[1, 2, 3]).unwrap();
        }
        let seen = std::rc::Rc::new(std::cell::Cell::new(None));
        let record = seen.clone();
        BEFORE_RENAME.with(|hook| *hook.borrow_mut() = Some(Box::new(move |target| record.set(Some(open_on(target))))));
        let kept = std::cell::RefCell::new(None);
        atomic_rewrite(&path, None, |src, dst| {
            assert_eq!(open_on(&path), 1, "the source is open while the build runs");
            let data = src.dataset("d")?;
            dst.new_dataset::<i32>().shape([3]).create("d")?.write(&data.read_raw::<i32>()?)?;
            // A view the build kept must not hold the source open either.
            *kept.borrow_mut() = Some(data);
            Ok(())
        })
        .unwrap();
        BEFORE_RENAME.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(seen.get(), Some(0), "the source was still open at the rename");
        assert!(!kept.into_inner().unwrap().is_valid());
        let again = open_read(&path).unwrap();
        assert_eq!(again.dataset("d").unwrap().read_raw::<i32>().unwrap(), vec![1, 2, 3]);
    }
}
