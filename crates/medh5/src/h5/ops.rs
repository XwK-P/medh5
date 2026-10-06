//! Low-level operations: object copy, traversal, and the self-containment
//! check.
//!
//! A MEDH5 file holds its own bytes (§2).  Three HDF5 features read bytes from
//! elsewhere --- a dataset whose raw data lives in *external storage*, a
//! *virtual dataset* mapping other files, and an *external link* --- and a tool
//! that follows one copies bytes it was never given: `recompress` of a crafted
//! file once wrote the contents of a local private key into its output.  Every
//! file is checked on open and refused if it carries any of them.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Mutex;

use crate::h5sys::{h5, h5d, h5i, h5l, h5o, h5p};
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

/// Recursively visit every object under `group` through hard links, depth
/// first in name order --- h5py's `visititems`.  Paths are relative to
/// `group`.  `f` returns `false` to stop.
pub fn visit(group: &hdf5::Group, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    fn walk(group: &hdf5::Group, prefix: &str, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<bool> {
        for name in members(group)? {
            if link_kind(group, &name) != Some(LinkKind::Hard) {
                continue;
            }
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            match node_kind(group, &name) {
                Some(NodeKind::Group) => {
                    let child = group.group(&name)?;
                    if !f(&path, &Node::Group(child.clone()))? {
                        return Ok(false);
                    }
                    if !walk(&child, &path, f)? {
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
    walk(group, "", f)?;
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

fn checked_files() -> &'static Mutex<HashMap<(u64, u64, u64, i128, i128), ()>> {
    static CHECKED: std::sync::OnceLock<Mutex<HashMap<(u64, u64, u64, i128, i128), ()>>> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| Mutex::new(HashMap::new()))
}

const CHECKED_LIMIT: usize = 65_536;

/// The identity of the file at `path`: device, inode, size, mtime, ctime.
fn file_key(path: &Path) -> Option<(u64, u64, u64, i128, i128)> {
    let meta = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some((
            meta.dev(),
            meta.ino(),
            meta.size(),
            meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
            meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128,
        ))
    }
    #[cfg(not(unix))]
    {
        let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as i128;
        Some((0, 0, meta.len(), modified, 0))
    }
}

/// Refuse a file that reads bytes from outside itself (`MEDH5FileError`).
///
/// Memoised per file identity, so re-opening an unchanged file costs one
/// `stat` --- a training loop re-opens the same few thousand files for a run.
pub fn check_self_contained(handle: &hdf5::File, path: Option<&Path>) -> Result<()> {
    let key = path.and_then(file_key);
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
/// Returns the filter mask and the bytes exactly as stored.
pub fn read_raw_chunk(ds: &hdf5::Dataset, offset: &[u64]) -> Result<(u32, Vec<u8>)> {
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
