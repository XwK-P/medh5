//! `medh5.integrity`: digests, `content_id`, verification and repair (§13).
//!
//! The file-based functions take a path and the object's path inside the
//! file: the engine opens the file itself rather than reading through another
//! library's handle.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use medh5::h5::file::open_read;
use medh5::h5::ops;
use medh5::integrity::{digest, repair, verify};

use crate::convert::{json_to_py, py_to_nd};
use crate::errors::R;

/// A verification result as the keyword arguments of `VerifyResult`.
pub fn verify_result_to_py<'py>(py: Python<'py>, r: &verify::VerifyResult) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("checked", PyTuple::new(py, &r.checked)?)?;
    out.set_item("mismatched", PyTuple::new(py, &r.mismatched)?)?;
    out.set_item("undigested", PyTuple::new(py, &r.undigested)?)?;
    out.set_item("malformed", PyTuple::new(py, &r.malformed)?)?;
    out.set_item("content_id_declared", &r.content_id_declared)?;
    out.set_item("content_id_computed", &r.content_id_computed)?;
    out.set_item("stale_index", PyTuple::new(py, &r.stale_index)?)?;
    out.set_item("unattested", PyTuple::new(py, &r.unattested)?)?;
    Ok(out.into_any())
}

fn root_of(path: &std::path::Path) -> R<medh5::hdf5::Group> {
    let file = open_read(path)?;
    Ok(file.as_group()?)
}

fn group_at(path: &std::path::Path, inner: &str) -> R<medh5::hdf5::Group> {
    let root = root_of(path)?;
    if inner.is_empty() || inner == "/" {
        return Ok(root);
    }
    Ok(root.group(inner)?)
}

#[pyfunction]
#[pyo3(signature = (path, array, algo="sha256"))]
fn array_digest(path: &str, array: &Bound<'_, PyAny>, algo: &str) -> R<String> {
    Ok(digest::array_digest(path, &py_to_nd(array)?, algo)?)
}

#[pyfunction]
#[pyo3(signature = (payload, algo="sha256"))]
fn digest_bytes(payload: &[u8], algo: &str) -> R<String> {
    Ok(medh5::digest::digest_bytes(payload, algo)?)
}

#[pyfunction]
fn parse_digest(value: &str) -> R<(String, String)> {
    Ok(medh5::digest::parse_digest(value)?)
}

#[pyfunction]
#[pyo3(signature = (file, dataset, algo="sha256"))]
fn dataset_digest_at(file: PathBuf, dataset: &str, algo: &str) -> R<String> {
    let root = root_of(&file)?;
    let ds = root.dataset(dataset)?;
    Ok(digest::dataset_digest(&ds, dataset.trim_start_matches('/'), algo)?)
}

#[pyfunction]
fn canonical_attrs_at(file: PathBuf, object: &str, names: Vec<String>) -> R<String> {
    let root = root_of(&file)?;
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let text = match ops::node_kind(&root, object) {
        Some(ops::NodeKind::Dataset) => {
            let ds = root.dataset(object)?;
            digest::canonical_attrs(&ds, &refs)?
        }
        _ => {
            let g = if object.is_empty() { root.clone() } else { root.group(object)? };
            digest::canonical_attrs(&g, &refs)?
        }
    };
    Ok(text)
}

#[pyfunction]
#[pyo3(signature = (file, group, root="", algo="sha256"))]
fn group_digest_at(file: PathBuf, group: &str, root: &str, algo: &str) -> R<String> {
    let sample_root = group_at(&file, root)?;
    let g = sample_root.group(group)?;
    Ok(digest::group_digest(&g, &sample_root, algo)?)
}

#[pyfunction]
#[pyo3(signature = (file, object, root=""))]
fn verify_object_at(file: PathBuf, object: &str, root: &str) -> R<bool> {
    Ok(verify::verify_object(&group_at(&file, root)?, object)?)
}

#[pyfunction]
#[pyo3(signature = (file, root=""))]
fn stale_index_entries_at<'py>(py: Python<'py>, file: PathBuf, root: &str) -> R<Bound<'py, PyTuple>> {
    Ok(PyTuple::new(py, verify::stale_index_entries(&group_at(&file, root)?)?)?)
}

#[pyfunction]
#[pyo3(signature = (file, *, root="", partial=None, check_content_id=true))]
fn verify_file<'py>(
    py: Python<'py>,
    file: PathBuf,
    root: &str,
    partial: Option<Vec<String>>,
    check_content_id: bool,
) -> R<Bound<'py, PyAny>> {
    let group = group_at(&file, root)?;
    let names = medh5::sample::attr_name_map_of(&group)?;
    let result = verify::verify_root(&group, Some(&names), partial.as_deref(), check_content_id)?;
    Ok(verify_result_to_py(py, &result)?)
}

#[pyfunction]
fn raw_chunks_at<'py>(py: Python<'py>, file: PathBuf, dataset: &str) -> R<Bound<'py, PyList>> {
    let root = root_of(&file)?;
    let chunks = verify::raw_chunks(&root.dataset(dataset)?)?;
    Ok(PyList::new(py, chunks.iter().map(|c| PyBytes::new(py, c)))?)
}

#[pyfunction]
fn subtrees_identical_at<'py>(
    py: Python<'py>,
    file_a: PathBuf,
    group_a: &str,
    file_b: PathBuf,
    group_b: &str,
) -> R<Bound<'py, PyTuple>> {
    let a = group_at(&file_a, group_a)?;
    let b = group_at(&file_b, group_b)?;
    Ok(PyTuple::new(py, verify::subtrees_identical(&a, &b)?)?)
}

#[pyfunction]
fn diagnose<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    let found = py.detach(move || repair::diagnose(&path))?;
    Ok(json_to_py(py, &found.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (path, *, rebuild_index=false, rewrite_digests=false, reason=None, performed_by=None, max_coords=None))]
fn fix<'py>(
    py: Python<'py>,
    path: PathBuf,
    rebuild_index: bool,
    rewrite_digests: bool,
    reason: Option<String>,
    performed_by: Option<String>,
    max_coords: Option<usize>,
) -> R<Bound<'py, PyAny>> {
    let options = repair::FixOptions { rebuild_index, rewrite_digests, reason, performed_by, max_coords };
    let done = py.detach(move || repair::fix(&path, &options))?;
    Ok(json_to_py(py, &done.to_json())?)
}

// -- functions over stored objects (`Dataset`/`Group` proxies) ---------------------------

use crate::nodes::{Dataset, Group};

/// A dataset or group proxy as an HDF5 location.
enum Loc {
    Dataset(medh5::hdf5::Dataset),
    Group(medh5::hdf5::Group),
}

fn loc_of(obj: &Bound<'_, PyAny>) -> PyResult<Loc> {
    if let Ok(d) = obj.cast::<Dataset>() {
        return Ok(Loc::Dataset(d.get().ds.clone()));
    }
    if let Ok(g) = obj.cast::<Group>() {
        return Ok(Loc::Group(g.get().group.clone()));
    }
    // A Sample (or anything else carrying a root).
    let root = obj.getattr("root")?;
    Ok(Loc::Group(root.cast::<Group>()?.get().group.clone()))
}

/// The group a `Group` proxy or a `Sample` (its root) stands for.
pub(crate) fn group_of(obj: &Bound<'_, PyAny>) -> PyResult<medh5::hdf5::Group> {
    match loc_of(obj)? {
        Loc::Group(g) => Ok(g),
        Loc::Dataset(_) => Err(pyo3::exceptions::PyTypeError::new_err("expected a group, not a dataset")),
    }
}

fn file_root(g: &medh5::hdf5::Group) -> R<medh5::hdf5::Group> {
    Ok(g.file()?.as_group()?)
}

/// Spec attribute names as the engine's `'static` table entries (names a
/// caller supplies outside the tables are interned once).
fn static_names(names: Vec<String>) -> Vec<&'static str> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static INTERNED: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);
    let mut guard = INTERNED.lock().unwrap();
    let table = guard.get_or_insert_with(HashMap::new);
    names.into_iter().map(|n| *table.entry(n.clone()).or_insert_with(|| Box::leak(n.into_boxed_str()))).collect()
}

fn attr_names_arg(
    root: &medh5::hdf5::Group,
    attr_names: Option<&Bound<'_, PyAny>>,
) -> R<medh5::integrity::AttrNameMap> {
    match attr_names.filter(|a| !a.is_none()) {
        None => Ok(medh5::sample::attr_name_map_of(root)?),
        Some(map) => {
            let mut out = medh5::integrity::AttrNameMap::new();
            for item in map.call_method0("items")?.try_iter()? {
                let (k, v): (String, Vec<String>) = item?.extract()?;
                out.insert(k, static_names(v));
            }
            Ok(out)
        }
    }
}

/// The digest of a stored dataset over its decompressed content (§13.1).
#[pyfunction]
#[pyo3(signature = (dataset, path=None, algo="sha256"))]
fn dataset_digest(dataset: &Bound<'_, PyAny>, path: Option<String>, algo: &str) -> R<String> {
    let dataset = crate::nodes::dataset_arg(dataset)?;
    let ds = &dataset.get().ds;
    let path = path.unwrap_or_else(|| ds.name().trim_start_matches('/').to_string());
    Ok(digest::dataset_digest(ds, &path, algo)?)
}

/// Canonical JSON over an object's named attributes (§13.2).
#[pyfunction]
fn canonical_attrs(obj: &Bound<'_, PyAny>, names: Vec<String>) -> R<String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    Ok(match loc_of(obj)? {
        Loc::Dataset(d) => digest::canonical_attrs(&d, &refs)?,
        Loc::Group(g) => digest::canonical_attrs(&g, &refs)?,
    })
}

#[pyfunction]
#[pyo3(signature = (obj, names, algo="sha256"))]
fn attrs_digest(obj: &Bound<'_, PyAny>, names: Vec<String>, algo: &str) -> R<String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    Ok(match loc_of(obj)? {
        Loc::Dataset(d) => digest::attrs_digest(&d, &refs, algo)?,
        Loc::Group(g) => digest::attrs_digest(&g, &refs, algo)?,
    })
}

/// An object's path relative to a sample root (the file root by default).
#[pyfunction]
#[pyo3(signature = (node, root=None))]
fn relative_path(node: &Bound<'_, PyAny>, root: Option<&Bound<'_, PyAny>>) -> R<String> {
    let (name, own_root) = match loc_of(node)? {
        Loc::Dataset(d) => (d.name(), file_root(&d.file()?.as_group()?)?),
        Loc::Group(g) => (g.name(), file_root(&g)?),
    };
    let root = match root.filter(|r| !r.is_none()) {
        Some(r) => group_of(r)?,
        None => own_root,
    };
    Ok(digest::relative_path(&name, &root))
}

/// The digest of an object group: its datasets and canonical attributes.
#[pyfunction]
#[pyo3(signature = (group, algo="sha256", *, root=None))]
fn group_digest(group: &Bound<'_, PyAny>, algo: &str, root: Option<&Bound<'_, PyAny>>) -> R<String> {
    let g = group_of(group)?;
    let root = match root.filter(|r| !r.is_none()) {
        Some(r) => group_of(r)?,
        None => file_root(&g)?,
    };
    Ok(digest::group_digest(&g, &root, algo)?)
}

/// The Merkle root over stored digests, `meta` and canonical attributes.
#[pyfunction]
#[pyo3(signature = (root, attr_names=None, *, algo=None))]
fn compute_content_id(
    root: &Bound<'_, PyAny>,
    attr_names: Option<&Bound<'_, PyAny>>,
    algo: Option<String>,
) -> R<String> {
    let g = group_of(root)?;
    let names = attr_names_arg(&g, attr_names)?;
    let algo = match algo {
        Some(a) => a,
        None => digest::root_algo(&g)?,
    };
    Ok(digest::compute_content_id(&g, &names, &algo, None)?)
}

#[pyfunction]
fn verify_object(root: &Bound<'_, PyAny>, path: &str) -> R<bool> {
    Ok(verify::verify_object(&group_of(root)?, path)?)
}

#[pyfunction]
fn stale_index_entries<'py>(py: Python<'py>, root: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    Ok(PyTuple::new(py, verify::stale_index_entries(&group_of(root)?)?)?)
}

#[pyfunction]
#[pyo3(signature = (root, attr_names=None, *, partial=None, check_content_id=true))]
fn verify_root<'py>(
    py: Python<'py>,
    root: &Bound<'py, PyAny>,
    attr_names: Option<&Bound<'py, PyAny>>,
    partial: Option<Vec<String>>,
    check_content_id: bool,
) -> R<Bound<'py, PyAny>> {
    let g = group_of(root)?;
    let names = attr_names_arg(&g, attr_names)?;
    let result = verify::verify_root(&g, Some(&names), partial.as_deref(), check_content_id)?;
    Ok(verify_result_to_py(py, &result)?)
}

#[pyfunction]
fn raw_chunks<'py>(py: Python<'py>, dataset: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    let chunks = verify::raw_chunks(&crate::nodes::dataset_arg(dataset)?.get().ds)?;
    Ok(PyList::new(py, chunks.iter().map(|c| PyBytes::new(py, c)))?)
}

#[pyfunction]
fn subtrees_identical<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    Ok(PyTuple::new(py, verify::subtrees_identical(&group_of(a)?, &group_of(b)?)?)?)
}

#[pyfunction]
#[pyo3(signature = (paths, *, rebuild_index=false, rewrite_digests=false, reason=None, performed_by=None, max_coords=None))]
fn fix_paths<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    rebuild_index: bool,
    rewrite_digests: bool,
    reason: Option<String>,
    performed_by: Option<String>,
    max_coords: Option<usize>,
) -> R<Bound<'py, PyList>> {
    let options = repair::FixOptions { rebuild_index, rewrite_digests, reason, performed_by, max_coords };
    let done = py.detach(move || {
        let refs: Vec<&std::path::Path> = paths.iter().map(PathBuf::as_path).collect();
        repair::fix_paths(&refs, &options)
    })?;
    let out = PyList::empty(py);
    for r in done {
        out.append(json_to_py(py, &r.to_json())?)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(array_digest, m)?,
        wrap_pyfunction!(digest_bytes, m)?,
        wrap_pyfunction!(parse_digest, m)?,
        wrap_pyfunction!(dataset_digest_at, m)?,
        wrap_pyfunction!(canonical_attrs_at, m)?,
        wrap_pyfunction!(group_digest_at, m)?,
        wrap_pyfunction!(verify_object_at, m)?,
        wrap_pyfunction!(stale_index_entries_at, m)?,
        wrap_pyfunction!(verify_file, m)?,
        wrap_pyfunction!(raw_chunks_at, m)?,
        wrap_pyfunction!(subtrees_identical_at, m)?,
        wrap_pyfunction!(diagnose, m)?,
        wrap_pyfunction!(fix, m)?,
        wrap_pyfunction!(fix_paths, m)?,
        wrap_pyfunction!(dataset_digest, m)?,
        wrap_pyfunction!(canonical_attrs, m)?,
        wrap_pyfunction!(attrs_digest, m)?,
        wrap_pyfunction!(relative_path, m)?,
        wrap_pyfunction!(group_digest, m)?,
        wrap_pyfunction!(compute_content_id, m)?,
        wrap_pyfunction!(verify_object, m)?,
        wrap_pyfunction!(stale_index_entries, m)?,
        wrap_pyfunction!(verify_root, m)?,
        wrap_pyfunction!(raw_chunks, m)?,
        wrap_pyfunction!(subtrees_identical, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("ATTESTED_GROUPS", PyTuple::new(m.py(), verify::ATTESTED_GROUPS)?)?;
    m.add("DEFAULT_ALGO", medh5::digest::DEFAULT_ALGO)?;
    m.add("DIGEST_ALGOS", PyTuple::new(m.py(), medh5::digest::DIGEST_ALGOS)?)?;
    m.add("STREAM_BYTES", digest::STREAM_BYTES)?;
    Ok(())
}
