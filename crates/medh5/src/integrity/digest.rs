//! Per-object digests and the Merkle `content_id` (spec §13).
//!
//! A digest covers one object's **decompressed** content, so recompressing a
//! file does not invalidate it.  The root `content_id` is a Merkle root over
//! the sorted per-object digests: incremental, partially verifiable, a content
//! address, and local --- a mismatch names the object that changed.
//!
//! Digest input for a dataset (§13.1):
//!
//! ```text
//! H( object_path || 0x00 || dtype_str || 0x00 || shape_csv
//!    || 0x00 || C-order little-endian bytes )
//! ```

use indexmap::IndexMap;

use crate::array::{Index, NdArray, Slice};
use crate::digest::{Hasher, DEFAULT_ALGO};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::{Error, Result};

/// Slab size for streaming a large dataset through the hash.
pub const STREAM_BYTES: usize = 32 * 1024 * 1024;

fn feed_header(hasher: &mut Hasher, path: &str, dtype_str: &str, shape: &[usize]) {
    hasher.update(path.as_bytes());
    hasher.update(b"\x00");
    hasher.update(dtype_str.as_bytes());
    hasher.update(b"\x00");
    let csv: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    hasher.update(csv.join(",").as_bytes());
    hasher.update(b"\x00");
}

/// Digest of an in-memory array under the §13.1 canonical byte stream.
pub fn array_digest(path: &str, array: &NdArray, algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    feed_header(&mut hasher, path, array.dtype().numpy_str(), &array.shape());
    hasher.update(&array.le_bytes());
    Ok(hasher.finish())
}

/// Digest of in-memory strings, as a variable-length string dataset hashes.
pub fn strings_digest(path: &str, values: &[String], shape: &[usize], algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    feed_header(&mut hasher, path, "|O", shape);
    for value in values {
        hasher.update(value.as_bytes());
        hasher.update(b"\x00");
    }
    Ok(hasher.finish())
}

/// Digest an HDF5 dataset, streaming so a large volume never lands in RAM.
///
/// String datasets hash their UTF-8 payloads separated by `0x00`, because a
/// variable-length dataset has no meaningful raw byte stream.
pub fn dataset_digest(ds: &hdf5::Dataset, path: &str, algo: &str) -> Result<String> {
    dataset_digest_streamed(ds, path, algo, STREAM_BYTES)
}

/// [`dataset_digest`] reading about `stream_bytes` at a time: the slab size
/// changes the reads, never the digest.
pub fn dataset_digest_streamed(ds: &hdf5::Dataset, path: &str, algo: &str, stream_bytes: usize) -> Result<String> {
    dataset_digest_inspected(ds, path, algo, stream_bytes, None)
}

/// What [`dataset_digest_inspected`] hands each block it reads to.
pub type Inspect<'a> = &'a mut dyn FnMut(&NdArray) -> Result<()>;

/// [`dataset_digest_streamed`], handing every block of a numeric dataset it
/// reads to `inspect` --- in order, leading axis first --- so a check of the
/// stored values costs no second read.
pub fn dataset_digest_inspected(
    ds: &hdf5::Dataset,
    path: &str,
    algo: &str,
    stream_bytes: usize,
    mut inspect: Option<Inspect<'_>>,
) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    let shape = ds.shape();
    match data::kind(ds)? {
        Kind::Strings => {
            feed_header(&mut hasher, path, "|O", &shape);
            for value in data::read_strings(ds)? {
                hasher.update(value.as_bytes());
                hasher.update(b"\x00");
            }
            Ok(hasher.finish())
        }
        Kind::Other(t) => Err(Error::Type(format!("{} has an unsupported datatype {t}", ds.name()))),
        Kind::Numeric(dtype) => {
            feed_header(&mut hasher, path, dtype.numpy_str(), &shape);
            let size: usize = shape.iter().product();
            if !shape.is_empty() && size == 0 {
                return Ok(hasher.finish());
            }
            if shape.is_empty() || data::is_enum(ds) {
                let all = data::read(ds)?;
                if let Some(f) = inspect.as_mut() {
                    f(&all)?;
                }
                hasher.update(&all.le_bytes());
                return Ok(hasher.finish());
            }
            let row_bytes = (shape[1..].iter().product::<usize>() * dtype.itemsize()).max(1);
            let step = (stream_bytes / row_bytes).max(1);
            let mut start = 0;
            while start < shape[0] {
                let block = data::read_region(ds, &[Index::Slice(Slice::new(start as i64, (start + step) as i64))])?;
                if let Some(f) = inspect.as_mut() {
                    f(&block)?;
                }
                hasher.update(&block.le_bytes());
                start += step;
            }
            Ok(hasher.finish())
        }
    }
}

/// An object's path relative to a sample root.
///
/// Digests hash the object path, so it must be *sample-root relative*: a
/// sample extracted from a collection digests identically standalone (§2.2).
pub fn relative_path(name: &str, root: &hdf5::Group) -> String {
    let base = format!("{}/", root.name().trim_end_matches('/'));
    match name.strip_prefix(&base) {
        Some(rest) => rest.to_string(),
        None => name.trim_start_matches('/').to_string(),
    }
}

/// Canonical JSON over an object's spec-defined attributes (§13.2): sorted
/// keys, arrays as nested lists, floats in shortest round-trip form.
pub fn canonical_attrs(obj: &hdf5::Location, names: &[&str]) -> Result<String> {
    let present = attrs::names(obj)?;
    let mut wanted: Vec<&str> = names.iter().copied().filter(|n| present.iter().any(|p| p == n)).collect();
    wanted.sort_unstable();
    wanted.dedup();
    // Canonical JSON (§5.1) written from the values themselves, so a NaN or
    // infinity is spelled as 1.x hashed it rather than lost in a JSON value.
    let mut out = String::from("{");
    for (i, name) in wanted.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        crate::json::write_string(&mut out, name, false);
        out.push(':');
        attrs::read(obj, name)?.unwrap_or(AttrValue::Unsupported(String::new())).write_canonical(&mut out);
    }
    out.push('}');
    Ok(out)
}

/// Digest of an object's canonical attributes.
pub fn attrs_digest(obj: &hdf5::Location, names: &[&str], algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    hasher.update(canonical_attrs(obj, names)?.as_bytes());
    Ok(hasher.finish())
}

fn top_level(path: &str) -> &str {
    path.split('/').next().unwrap_or(path)
}

/// Write a `digest` attribute on every dataset under `root`.
///
/// `index/` is skipped (it carries `source_digest` instead, §13.3).
/// `only_missing` digests just the datasets that carry none and returns the
/// stored value for the rest --- what an amend wants.
pub fn stamp_digests(
    root: &hdf5::Group,
    algo: &str,
    skip: &[&str],
    only_missing: bool,
) -> Result<IndexMap<String, String>> {
    let mut digests = IndexMap::new();
    for (name, ds) in ops::datasets(root)? {
        if name == "meta" || skip.contains(&top_level(&name)) {
            continue;
        }
        if only_missing {
            if let Some(stored) = attrs::get_str(&ds, "digest")? {
                digests.insert(name, stored);
                continue;
            }
        }
        let value = dataset_digest(&ds, &name, algo)?;
        attrs::write(&ds, "digest", &AttrValue::Str(value.clone()))?;
        digests.insert(name, value);
    }
    Ok(digests)
}

/// The `digest` attribute of every dataset that carries one.
pub fn collect_digests(root: &hdf5::Group, skip: &[&str]) -> Result<IndexMap<String, String>> {
    Ok(collect_digests_with_gaps(root, skip)?.0)
}

/// [`collect_digests`], and in the same walk the datasets that carry no
/// `digest` --- the root `meta` aside, which §13.2 hashes as its own line.
pub fn collect_digests_with_gaps(root: &hdf5::Group, skip: &[&str]) -> Result<(IndexMap<String, String>, Vec<String>)> {
    let mut out = IndexMap::new();
    let mut gaps = Vec::new();
    for (name, ds) in ops::datasets(root)? {
        if skip.contains(&top_level(&name)) {
            continue;
        }
        match attrs::get_str(&ds, "digest")? {
            Some(value) => {
                out.insert(name, value);
            }
            None if name != "meta" => gaps.push(name),
            None => {}
        }
    }
    gaps.sort();
    Ok((out, gaps))
}

/// A digest over every dataset in a group --- an annotation's identity.
///
/// A mini-Merkle over the group's own datasets, sorted by name; datasets not
/// stamped yet are digested on the fly with the path [`stamp_digests`] would
/// use, so an index built before the commit is not stale right after it.
pub fn group_digest(group: &hdf5::Group, root: &hdf5::Group, algo: &str) -> Result<String> {
    let mut lines = String::new();
    for name in ops::members(group)? {
        let Some(ds) = ops::child_dataset(group, &name) else { continue };
        let value = match attrs::get_str(&ds, "digest")? {
            Some(v) => v,
            None => dataset_digest(&ds, &relative_path(&ds.name(), root), algo)?,
        };
        lines.push_str(&format!("{name}\t{value}\n"));
    }
    let mut hasher = Hasher::new(algo)?;
    hasher.update(lines.as_bytes());
    Ok(hasher.finish())
}

/// Object path -> the spec-defined attribute names `content_id` covers.
pub type AttrNameMap = IndexMap<String, Vec<&'static str>>;

/// The Merkle root over dataset digests, `meta` and canonical attributes.
pub fn compute_content_id(
    root: &hdf5::Group,
    attr_names: &AttrNameMap,
    algo: &str,
    digests: Option<&IndexMap<String, String>>,
) -> Result<String> {
    Hasher::new(algo)?;
    let owned;
    let digests = match digests {
        Some(d) => d,
        None => {
            owned = collect_digests(root, &["index"])?;
            &owned
        }
    };
    let mut dataset_lines: Vec<String> = digests.iter().map(|(p, v)| format!("{p}\t{v}\n")).collect();
    dataset_lines.sort();
    let meta = ops::child_dataset(root, "meta")
        .ok_or_else(|| Error::Key("\"Unable to synchronously open object (object 'meta' doesn't exist)\"".into()))?;
    let meta_text = data::read_scalar_string(&meta)?;
    let mut meta_hasher = Hasher::new(algo)?;
    meta_hasher.update(meta_text.as_bytes());
    let meta_hash = meta_hasher.hexdigest();
    let mut attr_lines = Vec::new();
    for (path, names) in attr_names {
        if path.is_empty() {
            attr_lines.push(format!("@\t{}\n", attrs_digest(root, names, algo)?));
            continue;
        }
        let digest = match ops::node_kind(root, path) {
            Some(ops::NodeKind::Group) => {
                let g = root.group(path)?;
                attrs_digest(&g, names, algo)?
            }
            Some(ops::NodeKind::Dataset) => {
                let d = root.dataset(path)?;
                attrs_digest(&d, names, algo)?
            }
            _ => continue,
        };
        attr_lines.push(format!("@{path}\t{digest}\n"));
    }
    attr_lines.sort();
    let payload = format!("{}meta\t{meta_hash}\n{}", dataset_lines.concat(), attr_lines.concat());
    let mut root_hasher = Hasher::new(algo)?;
    root_hasher.update(payload.as_bytes());
    Ok(root_hasher.finish())
}

/// The digest algorithm a root declares, or the default.
pub fn root_algo(root: &hdf5::Group) -> Result<String> {
    Ok(attrs::get_str(root, "digest_algo")?.unwrap_or_else(|| DEFAULT_ALGO.to_string()))
}

/// Visit helper re-exported for callers that walk a sample.
pub fn walk(root: &hdf5::Group, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    ops::visit(root, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s13_1_streaming_matches_the_whole_array() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
        let data = ndarray::Array2::<i16>::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as i16);
        let ds = file.new_dataset_builder().with_data(&data).create("d").unwrap();
        let whole = array_digest("d", &NdArray::from(data.into_dyn()), "sha256").unwrap();
        // 128 bytes is one row of 64 int16: one row per read.
        assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", 128).unwrap(), whole);
        assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", 1).unwrap(), whole);
        assert_eq!(dataset_digest(&ds, "d", "sha256").unwrap(), whole);
    }
}
