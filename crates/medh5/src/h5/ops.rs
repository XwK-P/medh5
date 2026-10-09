//! Low-level operations: object copy, traversal, and the self-containment
//! check.
//!
//! A MEDH5 file holds its own bytes (§2).  Three HDF5 features read bytes from
//! elsewhere --- a dataset whose raw data lives in *external storage*, a
//! *virtual dataset* mapping other files, and an *external link* --- and a tool
//! that follows one copies bytes it was never given: `recompress` of a crafted
//! file once wrote the contents of a local private key into its output.  Every
//! file is checked on open and refused if it carries any of them.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Mutex;

use crate::h5sys::{h5, h5d, h5i, h5l, h5o, h5p, h5t};
use crate::json::repr_str;
use crate::{Error, Result};

/// What kind of object a name in a group refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Group,
    Dataset,
    Other,
}

/// What a link is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    Hard,
    Soft,
    External,
    Other,
}

fn cstring(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| Error::Value(format!("name {} contains a NUL byte", repr_str(name))))
}

/// The kind of link `name` is in `group`, or `None` when it does not exist.
pub fn link_kind(group: &hdf5::Group, name: &str) -> Option<LinkKind> {
    let cname = cstring(name).ok()?;
    super::locked(|| unsafe {
        if h5l::H5Lexists(group.id(), cname.as_ptr(), h5p::H5P_DEFAULT) <= 0 {
            return None;
        }
        let mut info: h5l::H5L_info2_t = std::mem::zeroed();
        if h5l::H5Lget_info2(group.id(), cname.as_ptr(), &mut info, h5p::H5P_DEFAULT) < 0 {
            return None;
        }
        Some(match info.type_ {
            h5l::H5L_type_t::H5L_TYPE_HARD => LinkKind::Hard,
            h5l::H5L_type_t::H5L_TYPE_SOFT => LinkKind::Soft,
            h5l::H5L_type_t::H5L_TYPE_EXTERNAL => LinkKind::External,
            _ => LinkKind::Other,
        })
    })
}

/// What `name` in `group` is, following soft links; `None` when it cannot be
/// resolved.
pub fn node_kind(group: &hdf5::Group, name: &str) -> Option<NodeKind> {
    match link_kind(group, name)? {
        LinkKind::External => return Some(NodeKind::Other),
        LinkKind::Other => return Some(NodeKind::Other),
        _ => {}
    }
    match group.loc_type_by_name(name) {
        Ok(hdf5::LocationType::Group) => Some(NodeKind::Group),
        Ok(hdf5::LocationType::Dataset) => Some(NodeKind::Dataset),
        Ok(_) => Some(NodeKind::Other),
        Err(_) => None,
    }
}

/// Whether `name` exists in `group` as a link.
pub fn exists(group: &hdf5::Group, name: &str) -> bool {
    link_kind(group, name).is_some()
}

/// Whether `name` in `group` is a group.
pub fn is_group(group: &hdf5::Group, name: &str) -> bool {
    node_kind(group, name) == Some(NodeKind::Group)
}

/// Whether `name` in `group` is a dataset.
pub fn is_dataset(group: &hdf5::Group, name: &str) -> bool {
    node_kind(group, name) == Some(NodeKind::Dataset)
}

/// The names in a group, sorted (HDF5's name order).
pub fn members(group: &hdf5::Group) -> Result<Vec<String>> {
    super::alive(group)?;
    let mut names = group.member_names()?;
    names.sort();
    Ok(names)
}

/// A child group, if `name` is one.
pub fn child_group(group: &hdf5::Group, name: &str) -> Option<hdf5::Group> {
    if is_group(group, name) {
        group.group(name).ok()
    } else {
        None
    }
}

/// A child dataset, if `name` is one.
pub fn child_dataset(group: &hdf5::Group, name: &str) -> Option<hdf5::Dataset> {
    if is_dataset(group, name) {
        group.dataset(name).ok()
    } else {
        None
    }
}

/// One object reached by [`visit`].
pub enum Node {
    Group(hdf5::Group),
    Dataset(hdf5::Dataset),
}

/// An object's identity within its file: two paths to one object share it.
pub type ObjectId = (std::os::raw::c_ulong, [u8; 16]);

// `H5O_token_t` is `repr(C)` around one `[u8; H5O_MAX_TOKEN_SIZE]`.
const _: () = assert!(std::mem::size_of::<h5o::H5O_token_t>() == 16);

fn object_id(info: &h5o::H5O_info2_t) -> ObjectId {
    // SAFETY: a plain-bytes struct of exactly 16 bytes (asserted above).
    let token = unsafe { std::mem::transmute::<h5o::H5O_token_t, [u8; 16]>(info.token) };
    (info.fileno, token)
}

/// The identity of the object `name` in `group` names, without opening it.
pub fn object_id_by_name(group: &hdf5::Group, name: &str) -> Option<ObjectId> {
    let cname = cstring(name).ok()?;
    super::locked(|| unsafe {
        let mut info: h5o::H5O_info2_t = std::mem::zeroed();
        if h5o::H5Oget_info_by_name3(group.id(), cname.as_ptr(), &mut info, h5o::H5O_INFO_BASIC, h5p::H5P_DEFAULT) < 0 {
            return None;
        }
        Some(object_id(&info))
    })
}

/// The identity of an open object.
pub fn object_id_of(obj: &hdf5::Location) -> Option<ObjectId> {
    super::locked(|| unsafe {
        let mut info: h5o::H5O_info2_t = std::mem::zeroed();
        if h5o::H5Oget_info3(obj.id(), &mut info, h5o::H5O_INFO_BASIC) < 0 {
            return None;
        }
        Some(object_id(&info))
    })
}

/// How deep [`visit`] goes.  A sample is a handful of levels deep; a group
/// nested past this is refused rather than followed, so no file can exhaust
/// the stack, whatever its links.
pub const MAX_DEPTH: usize = 64;

/// Recursively visit every object under `group` through hard links, depth
/// first in name order --- h5py's `visititems`.  Paths are relative to
/// `group`.  `f` returns `false` to stop.
///
/// Each object is visited **once**, at the first path that reaches it, as
/// `H5Ovisit` does: a hard link back to an ancestor is a cycle, not an
/// infinite tree, and a second link to an object is an alias of it, not a
/// second object (1.x digested an alias once, under its first path).
pub fn visit(group: &hdf5::Group, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    fn walk(
        group: &hdf5::Group,
        prefix: &str,
        depth: usize,
        seen: &mut HashSet<ObjectId>,
        f: &mut dyn FnMut(&str, &Node) -> Result<bool>,
    ) -> Result<bool> {
        if depth > MAX_DEPTH {
            return Err(Error::File(format!(
                "{} is nested more than {MAX_DEPTH} groups deep; it is refused rather than followed",
                repr_str(&group.name())
            )));
        }
        for name in members(group)? {
            if link_kind(group, &name) != Some(LinkKind::Hard) {
                continue;
            }
            if let Some(id) = object_id_by_name(group, &name) {
                if !seen.insert(id) {
                    continue;
                }
            }
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            match node_kind(group, &name) {
                Some(NodeKind::Group) => {
                    let child = group.group(&name)?;
                    if !f(&path, &Node::Group(child.clone()))? {
                        return Ok(false);
                    }
                    if !walk(&child, &path, depth + 1, seen, f)? {
                        return Ok(false);
                    }
                }
                Some(NodeKind::Dataset) => {
                    let ds = group.dataset(&name)?;
                    if !f(&path, &Node::Dataset(ds))? {
                        return Ok(false);
                    }
                }
                _ => {}
            }
        }
        Ok(true)
    }
    // The starting group itself: a link back to it is a cycle too.
    let mut seen: HashSet<ObjectId> = object_id_of(group).into_iter().collect();
    walk(group, "", 0, &mut seen, f)?;
    Ok(())
}

/// Every dataset under `group`, with its path relative to `group`.
pub fn datasets(group: &hdf5::Group) -> Result<Vec<(String, hdf5::Dataset)>> {
    let mut out = Vec::new();
    visit(group, &mut |path, node| {
        if let Node::Dataset(ds) = node {
            out.push((path.to_string(), ds.clone()));
        }
        Ok(true)
    })?;
    Ok(out)
}

/// Copy one object (group or dataset), attributes included, into `dst`.
///
/// Soft links are expanded and external links are **not**: expanding one
/// copies another file's contents into this one.  Chunks move as stored bytes.
pub fn copy_object(src: &hdf5::Group, name: &str, dst: &hdf5::Group, dst_name: &str) -> Result<()> {
    let cname = cstring(name)?;
    let cdst = cstring(dst_name)?;
    super::locked(|| unsafe {
        let ocpypl = h5p::H5Pcreate(*crate::h5sys::h5p::H5P_CLS_OBJECT_COPY);
        h5p::H5Pset_copy_object(ocpypl, h5o::H5O_COPY_EXPAND_SOFT_LINK_FLAG);
        let lcpl = h5p::H5Pcreate(*crate::h5sys::h5p::H5P_CLS_LINK_CREATE);
        h5p::H5Pset_create_intermediate_group(lcpl, 1);
        let status = h5o::H5Ocopy(src.id(), cname.as_ptr(), dst.id(), cdst.as_ptr(), ocpypl, lcpl);
        h5p::H5Pclose(lcpl);
        h5p::H5Pclose(ocpypl);
        if status < 0 {
            return Err(Error::Io(format!(
                "could not copy {} into {}: {}",
                repr_str(name),
                dst.name(),
                hdf5_error_text()
            )));
        }
        Ok(())
    })
}

/// Copy every child of `src` not in `known` into `dst` (spec §14.4).
pub fn copy_unknown(src: &hdf5::Group, dst: &hdf5::Group, known: &[&str]) -> Result<Vec<String>> {
    let mut kept = Vec::new();
    for name in members(src)? {
        if !known.contains(&name.as_str()) {
            copy_object(src, &name, dst, &name)?;
            kept.push(name);
        }
    }
    Ok(kept)
}

/// The path a soft link holds, as stored --- not what it resolves to.
pub fn soft_link_target(group: &hdf5::Group, name: &str) -> Result<String> {
    let cname = cstring(name)?;
    super::locked(|| unsafe {
        let mut info: h5l::H5L_info2_t = std::mem::zeroed();
        if h5l::H5Lget_info2(group.id(), cname.as_ptr(), &mut info, h5p::H5P_DEFAULT) < 0
            || !matches!(info.type_, h5l::H5L_type_t::H5L_TYPE_SOFT)
        {
            return Err(Error::Value(format!("{} is not a soft link", repr_str(name))));
        }
        let size = *info.u.val_size();
        let mut buf = vec![0u8; size.max(1)];
        if h5l::H5Lget_val(group.id(), cname.as_ptr(), buf.as_mut_ptr().cast(), buf.len(), h5p::H5P_DEFAULT) < 0 {
            return Err(Error::Io(format!("could not read the soft link {}", repr_str(name))));
        }
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    })
}

/// Delete a link if present.
pub fn unlink(group: &hdf5::Group, name: &str) -> Result<()> {
    if exists(group, name) {
        group.unlink(name)?;
    }
    Ok(())
}

/// The text of the most recent HDF5 error on this thread (best effort).
pub fn hdf5_error_text() -> String {
    "HDF5 error".to_string()
}

// -- self-containment ---------------------------------------------------------

struct VisitState {
    file: h5i::hid_t,
    found: Vec<(String, String)>,
    failed: Option<String>,
}

unsafe extern "C" fn visit_link(
    group: h5i::hid_t,
    name: *const c_char,
    info: *const h5l::H5L_info2_t,
    data: *mut c_void,
) -> h5::herr_t {
    // SAFETY: HDF5 passes valid pointers for the duration of the callback.
    unsafe {
        let state = &mut *(data as *mut VisitState);
        let text = CStr::from_ptr(name).to_string_lossy().into_owned();
        match (*info).type_ {
            h5l::H5L_type_t::H5L_TYPE_EXTERNAL => {
                state.found.push((text, "an external link".into()));
                return 0;
            }
            h5l::H5L_type_t::H5L_TYPE_HARD => {}
            _ => return 0,
        }
        let mut oinfo: h5o::H5O_info2_t = std::mem::zeroed();
        if h5o::H5Oget_info_by_name3(group, name, &mut oinfo, h5o::H5O_INFO_BASIC, h5p::H5P_DEFAULT) < 0 {
            return 0;
        }
        if oinfo.type_ != h5o::H5O_type_t::H5O_TYPE_DATASET {
            return 0;
        }
        let ds = h5d::H5Dopen2(group, name, h5p::H5P_DEFAULT);
        if ds < 0 {
            // An object whose header cannot be read cannot have its data
            // read either, so it cannot leak anything.
            return 0;
        }
        let plist = h5d::H5Dget_create_plist(ds);
        if plist >= 0 {
            let layout = h5p::H5Pget_layout(plist);
            let external = h5p::H5Pget_external_count(plist);
            if layout == h5d::H5D_layout_t::H5D_VIRTUAL {
                state.found.push((text, "a virtual dataset".into()));
            } else if external > 0 {
                state.found.push((text, "external raw-data storage".into()));
            }
            h5p::H5Pclose(plist);
        }
        h5d::H5Dclose(ds);
        let _ = state.file;
        0
    }
}

/// Whether a datatype is, or holds, an HDF5 reference: an object or region
/// reference, alone or as a member of a compound, array or variable-length
/// type.  The caller holds the HDF5 lock.
fn holds_reference(tid: h5i::hid_t) -> bool {
    use h5t::H5T_class_t as Class;
    unsafe {
        match h5t::H5Tget_class(tid) {
            Class::H5T_REFERENCE => true,
            Class::H5T_COMPOUND => {
                let members = h5t::H5Tget_nmembers(tid).max(0) as u32;
                (0..members).any(|i| {
                    let member = h5t::H5Tget_member_type(tid, i);
                    if member < 0 {
                        return false;
                    }
                    let found = holds_reference(member);
                    h5t::H5Tclose(member);
                    found
                })
            }
            Class::H5T_ARRAY | Class::H5T_VLEN => {
                let base = h5t::H5Tget_super(tid);
                if base < 0 {
                    return false;
                }
                let found = holds_reference(base);
                h5t::H5Tclose(base);
                found
            }
            _ => false,
        }
    }
}

/// Every dataset and attribute under `root` whose type holds an HDF5
/// reference: `/path` for a dataset, `/path@name` for an attribute.
///
/// Attributes are read on every kind of object that carries them: groups,
/// datasets and committed (named) datatypes.  [`visit`] walks groups and
/// datasets, so a reference in an attribute of a committed datatype was
/// missed, and a rewrite that copies the type nulled it (C10 of the 2.0
/// re-audit).
pub fn reference_carriers(root: &hdf5::Group) -> Result<Vec<String>> {
    fn attributes(obj: &hdf5::Location, path: &str, found: &mut Vec<String>) -> Result<()> {
        for name in obj.attr_names()? {
            let dtype = obj.attr(&name)?.dtype()?;
            if super::locked(|| holds_reference(dtype.id())) {
                found.push(format!("{path}@{name}"));
            }
        }
        Ok(())
    }
    // The committed datatypes a group links to, each once.
    fn named_types(
        group: &hdf5::Group,
        prefix: &str,
        seen: &mut HashSet<ObjectId>,
        found: &mut Vec<String>,
    ) -> Result<()> {
        for name in members(group)? {
            if link_kind(group, &name) != Some(LinkKind::Hard)
                || group.loc_type_by_name(&name).ok() != Some(hdf5::LocationType::NamedDatatype)
            {
                continue;
            }
            if object_id_by_name(group, &name).is_some_and(|id| !seen.insert(id)) {
                continue;
            }
            let path = if prefix.is_empty() { format!("/{name}") } else { format!("/{prefix}/{name}") };
            let named = group.committed_datatype(&name)?;
            attributes(&named, &path, found)?;
        }
        Ok(())
    }
    let mut found = Vec::new();
    let mut types = HashSet::new();
    attributes(root, "/", &mut found)?;
    named_types(root, "", &mut types, &mut found)?;
    visit(root, &mut |name, node| {
        let path = format!("/{name}");
        match node {
            Node::Dataset(ds) => {
                let dtype = ds.dtype()?;
                if super::locked(|| holds_reference(dtype.id())) {
                    found.push(path.clone());
                }
                attributes(ds, &path, &mut found)?;
            }
            Node::Group(g) => {
                attributes(g, &path, &mut found)?;
                named_types(g, name, &mut types, &mut found)?;
            }
        }
        Ok(true)
    })?;
    Ok(found)
}

/// Refuse to copy a file that holds HDF5 references into a new one.
///
/// A reference is an address in the file that holds it.  MEDH5 stores none,
/// so any is another tool's content, which amend, recompress, pack and repack
/// carry over without understanding: copied, an object reference pointed
/// wherever its address happened to land in the new file, and a no-op amend
/// nulled the ones in an extension group --- while every digest still
/// verified (C10 of the 2.0 audit).
pub fn refuse_references(root: &hdf5::Group, action: &str) -> Result<()> {
    let found = reference_carriers(root)?;
    if found.is_empty() {
        return Ok(());
    }
    let named = found.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let more = if found.len() > 5 { format!(" (and {} more)", found.len() - 5) } else { String::new() };
    Err(Error::File(format!(
        "{} holds HDF5 references ({named}{more}). A reference is an address in the file that holds it, and \
         {action} writes a new file, where it would point at whatever lands at that address; MEDH5 stores no \
         references and does not rewrite another tool's, so the file is refused --- remove them, or rewrite it \
         with the tool that wrote them",
        repr_str(&root.filename())
    )))
}

/// Objects that read bytes from outside the file: `(path, what)` pairs.
pub fn outside_references(handle: &hdf5::File) -> Result<Vec<(String, String)>> {
    let mut state = VisitState { file: handle.id(), found: Vec::new(), failed: None };
    let root = cstring("/")?;
    let status = super::locked(|| unsafe {
        h5l::H5Lvisit_by_name2(
            handle.id(),
            root.as_ptr(),
            crate::h5sys::h5::H5_index_t::H5_INDEX_NAME,
            crate::h5sys::h5::H5_iter_order_t::H5_ITER_INC,
            Some(visit_link),
            (&mut state as *mut VisitState).cast(),
            h5p::H5P_DEFAULT,
        )
    });
    if status < 0 {
        return Err(Error::File(format!(
            "{} could not be walked to check that it is self-contained",
            repr_str(&handle.filename())
        )));
    }
    if let Some(msg) = state.failed {
        return Err(Error::File(msg));
    }
    Ok(state.found)
}

/// The identity of a file a handle had open:
/// `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`.
type FileIdentity = (u64, u64, u64, i128, i128);

fn checked_files() -> &'static Mutex<HashMap<FileIdentity, ()>> {
    static CHECKED: std::sync::OnceLock<Mutex<HashMap<FileIdentity, ()>>> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| Mutex::new(HashMap::new()))
}

const CHECKED_LIMIT: usize = 65_536;

#[cfg(unix)]
fn identity(meta: &std::fs::Metadata) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    Some((
        meta.dev(),
        meta.ino(),
        meta.size(),
        meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
        meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128,
    ))
}

/// The descriptor HDF5 reads `handle` through, when its driver has one.
#[cfg(unix)]
fn descriptor(handle: &hdf5::File) -> Option<std::os::raw::c_int> {
    use hdf5::file::FileDriver;
    // Only the default (sec2) driver's handle is a POSIX descriptor.
    if !matches!(handle.access_plist().ok()?.get_driver().ok()?, FileDriver::Sec2) {
        return None;
    }
    super::locked(|| unsafe {
        let mut found: *mut c_void = std::ptr::null_mut();
        if crate::h5sys::h5f::H5Fget_vfd_handle(handle.id(), h5p::H5P_DEFAULT, &mut found) < 0 || found.is_null() {
            return None;
        }
        Some(*found.cast::<std::os::raw::c_int>())
    })
}

/// The identity of the file `handle` holds open --- not whatever `path` names
/// now.
///
/// Between an open and a stat by name, another process can atomically replace
/// the path; recording the replacement as checked would let it skip the check
/// on its next open, and the next `recompress` would copy whatever it points
/// at.  So on POSIX the identity comes from the descriptor HDF5 read through:
/// device, inode, size, and modification and change times.
///
/// Elsewhere there is none.  Size and modification time alone do not tell two
/// files apart --- an archive extracts many with one mtime, and the time is
/// the author's to set --- so keying on them let a crafted file of a checked
/// file's length and time skip the check (W01 of the 2.0 audit; 1.x keyed
/// Windows on the volume serial and file index, which `std` cannot read).  No
/// identity means no memo: the file is checked on every open rather than
/// trusted on a guess.
fn opened_identity(handle: &hdf5::File, path: Option<&Path>) -> Option<FileIdentity> {
    #[cfg(unix)]
    {
        use std::os::unix::io::FromRawFd;
        let _ = path;
        let fd = descriptor(handle)?;
        // Borrowed for one `fstat`: HDF5 owns the descriptor and closes it.
        let file = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
        identity(&file.metadata().ok()?)
    }
    #[cfg(not(unix))]
    {
        let _ = (handle, path);
        None
    }
}

/// Refuse a file that reads bytes from outside itself (`MEDH5FileError`).
///
/// Memoised per file identity, so re-opening an unchanged file costs one
/// `stat` --- a training loop re-opens the same few thousand files for a run.
pub fn check_self_contained(handle: &hdf5::File, path: Option<&Path>) -> Result<()> {
    let key = opened_identity(handle, path);
    if let Some(k) = key {
        if checked_files().lock().unwrap().contains_key(&k) {
            return Ok(());
        }
    }
    let found = outside_references(handle)?;
    if !found.is_empty() {
        let named: Vec<String> = found.iter().take(5).map(|(n, w)| format!("/{n} is {w}")).collect();
        let more = if found.len() > 5 { format!(" (and {} more)", found.len() - 5) } else { String::new() };
        let where_ = path.map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| handle.filename());
        return Err(Error::File(format!(
            "{} is not self-contained: {}{more}. A MEDH5 file holds its own bytes (§2); following \
             these would read files on this machine that the file's author chose, so the file is refused",
            repr_str(&where_),
            named.join("; ")
        )));
    }
    if let Some(k) = key {
        let mut map = checked_files().lock().unwrap();
        if map.len() >= CHECKED_LIMIT {
            map.clear();
        }
        map.insert(k, ());
    }
    Ok(())
}

/// Read one stored (still compressed) chunk by its logical offset.
///
/// Returns the filter mask and the bytes exactly as stored.  `offset` has one
/// coordinate per axis: HDF5 reads that many from it, whatever its length.
pub fn read_raw_chunk(ds: &hdf5::Dataset, offset: &[u64]) -> Result<(u32, Vec<u8>)> {
    super::alive(ds)?;
    if offset.len() != ds.ndim() {
        return Err(Error::Value(format!(
            "a chunk offset names one coordinate per axis: {} has {} axes, the offset {offset:?} has {}",
            ds.name(),
            ds.ndim(),
            offset.len()
        )));
    }
    if ds.chunk().is_none() {
        return Err(Error::Value(format!("{} is not chunked, so it stores no chunks", ds.name())));
    }
    super::locked(|| unsafe {
        let mut mask: u32 = 0;
        let mut addr: u64 = 0;
        let mut nbytes: u64 = 0;
        if h5d::H5Dget_chunk_info_by_coord(ds.id(), offset.as_ptr(), &mut mask, &mut addr, &mut nbytes) < 0 {
            return Err(Error::Io(format!("no stored chunk at {offset:?} in {}", ds.name())));
        }
        let mut buf = vec![0u8; nbytes as usize];
        let mut filter_mask: u32 = 0;
        let mut size = nbytes as usize;
        if h5d::H5Dread_chunk(
            ds.id(),
            h5p::H5P_DEFAULT,
            offset.as_ptr(),
            &mut filter_mask,
            buf.as_mut_ptr().cast(),
            &mut size,
        ) < 0
        {
            return Err(Error::Io(format!("could not read the chunk at {offset:?} in {}", ds.name())));
        }
        buf.truncate(size);
        Ok((filter_mask, buf))
    })
}

/// The number of chunks actually stored for a chunked dataset.
pub fn stored_chunks(ds: &hdf5::Dataset) -> Result<Vec<Vec<u64>>> {
    let ndim = ds.ndim();
    super::locked(|| unsafe {
        let space = h5d::H5Dget_space(ds.id());
        let mut n: u64 = 0;
        let status = h5d::H5Dget_num_chunks(ds.id(), space, &mut n);
        let mut out = Vec::new();
        if status >= 0 {
            for i in 0..n {
                let mut offset = vec![0u64; ndim];
                let mut mask: u32 = 0;
                let mut addr: u64 = 0;
                let mut size: u64 = 0;
                if h5d::H5Dget_chunk_info(ds.id(), space, i, offset.as_mut_ptr(), &mut mask, &mut addr, &mut size) >= 0
                {
                    out.push(offset);
                }
            }
        }
        crate::h5sys::h5s::H5Sclose(space);
        if status < 0 {
            return Err(Error::Io(format!("could not count the chunks of {}", ds.name())));
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(path: &Path) {
        let file = crate::h5::file::create_truncate(path).unwrap();
        file.new_dataset::<u8>().shape([4]).create("data").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
    }

    /// A file carrying an external link: one of the three ways a file reads
    /// bytes from outside itself.
    fn linked(path: &Path, other: &Path) {
        plain(path);
        let file = hdf5::File::open_rw(path).unwrap();
        file.link_external(&other.to_string_lossy(), "/data", "x_link").unwrap();
    }

    /// A path replaced between the open and the check must not inherit it.
    ///
    /// The memo was keyed by a stat of the *path*, taken after the open: a
    /// replacement landing in between was recorded as checked while the handle
    /// scanned the old file, and its next open skipped the check.
    #[cfg(unix)]
    #[test]
    fn f22_the_check_is_remembered_for_the_file_it_read() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.medh5");
        let bad = dir.path().join("bad.medh5");
        plain(&target);
        linked(&bad, &target);
        let handle = hdf5::File::open(&target).unwrap();
        std::fs::rename(&bad, &target).unwrap(); // the path now names another file
        check_self_contained(&handle, Some(&target)).unwrap(); // the file read is clean
        drop(handle);
        let err = crate::h5::file::open_read(&target).unwrap_err();
        assert!(err.to_string().contains("not self-contained"), "{err}");
        assert!(err.to_string().contains("external link"), "{err}");
    }

    /// B01: HDF5 reads one coordinate per axis from the offset, so an offset
    /// of the wrong length read past it (an empty one was a segfault).
    #[test]
    fn b01_a_chunk_offset_has_one_coordinate_per_axis() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("c.h5")).unwrap();
        let ds = file.new_dataset::<u8>().chunk((2, 2)).shape((4, 4)).create("c").unwrap();
        ds.write(&ndarray::Array2::<u8>::ones((4, 4))).unwrap();
        for wrong in [&[][..], &[0][..], &[0, 0, 0][..]] {
            assert!(read_raw_chunk(&ds, wrong).is_err(), "{wrong:?}");
        }
        assert_eq!(read_raw_chunk(&ds, &[0, 2]).unwrap().1, vec![1u8; 4]);
        let flat = file.new_dataset::<u8>().shape([4]).create("flat").unwrap();
        assert!(read_raw_chunk(&flat, &[0]).is_err(), "a contiguous dataset stores no chunks");
    }

    /// An unchanged file is checked once; the memo is what keeps re-opens cheap.
    #[cfg(unix)]
    #[test]
    fn f22_an_unchanged_file_is_remembered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.medh5");
        plain(&path);
        let handle = crate::h5::file::open_read(&path).unwrap();
        let key = opened_identity(&handle, Some(&path)).expect("a sec2 file has an identity");
        assert!(checked_files().lock().unwrap().contains_key(&key));
    }

    /// W01: without the opened file's identity nothing is remembered, so
    /// every open is checked --- a file of a checked file's length and
    /// modification time is not taken for it.
    #[cfg(not(unix))]
    #[test]
    fn w01_without_an_identity_every_open_is_checked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.medh5");
        plain(&path);
        let handle = crate::h5::file::open_read(&path).unwrap();
        assert!(opened_identity(&handle, Some(&path)).is_none());
    }
}
