//! Verification of digests, `content_id` and derived-index currency (§13).

use serde_json::{json, Value};

use super::digest::{collect_digests, compute_content_id, dataset_digest, group_digest, root_algo, AttrNameMap};
use crate::digest::{parse_digest, DEFAULT_ALGO};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops::{self, Node, NodeKind};
use crate::Result;

/// Where every dataset is part of an object a `content_id` speaks for.
pub const ATTESTED_GROUPS: [&str; 4] = ["grids", "images", "annotations", "transforms"];

/// Outcome of a verification pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VerifyResult {
    pub checked: Vec<String>,
    pub mismatched: Vec<String>,
    pub undigested: Vec<String>,
    pub malformed: Vec<String>,
    pub content_id_declared: Option<String>,
    pub content_id_computed: Option<String>,
    pub stale_index: Vec<String>,
    /// Undigested datasets inside an object a declared `content_id` covers.
    pub unattested: Vec<String>,
}

impl VerifyResult {
    /// Whether the root `content_id` matches, or `None` when unjudged.
    pub fn content_id_ok(&self) -> Option<bool> {
        match (&self.content_id_declared, &self.content_id_computed) {
            (Some(a), Some(b)) => Some(a == b),
            _ => None,
        }
    }

    pub fn ok(&self) -> bool {
        self.mismatched.is_empty()
            && self.malformed.is_empty()
            && self.unattested.is_empty()
            && self.content_id_ok() != Some(false)
    }

    pub fn summary(&self) -> Value {
        json!({
            "ok": self.ok(),
            "checked": self.checked.len(),
            "mismatched": self.mismatched,
            "undigested": self.undigested,
            "unattested": self.unattested,
            "malformed": self.malformed,
            "content_id_ok": self.content_id_ok(),
            "stale_index": self.stale_index,
        })
    }
}

/// Recompute one dataset's digest and compare it with the stored value.
pub fn verify_object(root: &hdf5::Group, path: &str) -> Result<bool> {
    let ds = root.dataset(path)?;
    let Some(stored) = attrs::get_str(&ds, "digest")? else {
        return Ok(false);
    };
    let (algo, _) = parse_digest(&stored)?;
    Ok(dataset_digest(&ds, path, &algo)? == stored)
}

/// Index entries whose `source_digest` no longer matches their source (§13.3).
///
/// A stale entry is **not** a file error: readers ignore it and rebuild.
pub fn stale_index_entries(root: &hdf5::Group) -> Result<Vec<String>> {
    let Some(index) = ops::child_group(root, "index") else {
        return Ok(Vec::new());
    };
    let mut stale = Vec::new();
    for name in ops::members(&index)? {
        let declared = match ops::node_kind(&index, &name) {
            Some(NodeKind::Group) => {
                let g = index.group(&name)?;
                attrs::get_str(&g, "source_digest")?
            }
            Some(NodeKind::Dataset) => {
                let d = index.dataset(&name)?;
                attrs::get_str(&d, "source_digest")?
            }
            _ => None,
        };
        let Some(declared) = declared else {
            stale.push(name);
            continue;
        };
        let source_path = format!("annotations/{name}");
        let current = match ops::node_kind(root, &source_path) {
            None => {
                stale.push(name);
                continue;
            }
            Some(NodeKind::Group) => Some(group_digest(&root.group(&source_path)?, root, DEFAULT_ALGO)?),
            Some(NodeKind::Dataset) => {
                let d = root.dataset(&source_path)?;
                attrs::get_str(&d, "digest")?.or(Some(String::new()))
            }
            Some(_) => None,
        };
        if current.as_deref() != Some(declared.as_str()) {
            stale.push(name);
        }
    }
    Ok(stale)
}

/// Verify a sample root; `partial` restricts the pass to named objects.
pub fn verify_root(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    partial: Option<&[String]>,
    check_content_id: bool,
) -> Result<VerifyResult> {
    let stored = collect_digests(root, &["index"])?;
    let targets: Vec<String> = match partial {
        Some(p) => p.to_vec(),
        None => {
            let mut keys: Vec<String> = stored.keys().cloned().collect();
            keys.sort();
            keys
        }
    };
    let mut result = VerifyResult::default();
    for path in targets {
        let Some(value) = stored.get(&path) else {
            result.malformed.push(path);
            continue;
        };
        let Ok((algo, _)) = parse_digest(value) else {
            result.malformed.push(path);
            continue;
        };
        result.checked.push(path.clone());
        let ds = root.dataset(&path)?;
        if &dataset_digest(&ds, &path, &algo)? != value {
            result.mismatched.push(path);
        }
    }
    let mut undigested = Vec::new();
    ops::visit(root, &mut |name, node| {
        if let Node::Dataset(ds) = node {
            if name != "meta" && !name.starts_with("index/") && !attrs::has(ds, "digest") {
                undigested.push(name.to_string());
            }
        }
        Ok(true)
    })?;
    undigested.sort();
    let declared = match attrs::read(root, "content_id")? {
        Some(AttrValue::Str(s)) => Some(s),
        Some(other) => Some(attrs::stringify_value(&other)),
        None => None,
    };
    if check_content_id && partial.is_none() {
        if let Some(names) = attr_names {
            result.content_id_computed = Some(compute_content_id(root, names, &root_algo(root)?, Some(&stored))?);
        }
    }
    if declared.is_some() && partial.is_none() {
        result.unattested = undigested
            .iter()
            .filter(|n| n.contains('/') && ATTESTED_GROUPS.contains(&n.split('/').next().unwrap_or("")))
            .cloned()
            .collect();
    }
    result.undigested = undigested;
    result.content_id_declared = declared;
    result.stale_index = stale_index_entries(root)?;
    Ok(result)
}

/// Paths at which two object trees differ, byte for byte: structure,
/// attributes and *stored* chunk bytes.  Empty means a pure copy.
pub fn subtrees_identical(a: &hdf5::Group, b: &hdf5::Group) -> Result<Vec<String>> {
    let mut out = Vec::new();
    compare_groups(a, b, "", &mut out)?;
    out.sort();
    Ok(out)
}

fn compare_attrs(a: &hdf5::Location, b: &hdf5::Location, prefix: &str, out: &mut Vec<String>) -> Result<()> {
    let label = if prefix.is_empty() { "/" } else { prefix };
    let names_a = attrs::names(a)?;
    let names_b = attrs::names(b)?;
    let mut only: Vec<&String> = names_a
        .iter()
        .filter(|n| !names_b.contains(n))
        .chain(names_b.iter().filter(|n| !names_a.contains(n)))
        .collect();
    only.sort();
    for key in only {
        out.push(format!("{label}@{key}: present in only one tree"));
    }
    let mut both: Vec<&String> = names_a.iter().filter(|n| names_b.contains(n)).collect();
    both.sort();
    for key in both {
        if attrs::read(a, key)? != attrs::read(b, key)? {
            out.push(format!("{label}@{key}: differs"));
        }
    }
    Ok(())
}

fn compare_groups(a: &hdf5::Group, b: &hdf5::Group, prefix: &str, out: &mut Vec<String>) -> Result<()> {
    compare_attrs(a, b, prefix, out)?;
    let names_a = ops::members(a)?;
    let names_b = ops::members(b)?;
    for name in names_a.iter().filter(|n| !names_b.contains(n)).chain(names_b.iter().filter(|n| !names_a.contains(n))) {
        out.push(format!("{prefix}/{name}: present in only one tree"));
    }
    for name in names_a.iter().filter(|n| names_b.contains(n)) {
        let path = format!("{prefix}/{name}");
        match (ops::node_kind(a, name), ops::node_kind(b, name)) {
            (Some(NodeKind::Group), Some(NodeKind::Group)) => {
                compare_groups(&a.group(name)?, &b.group(name)?, &path, out)?
            }
            (Some(NodeKind::Dataset), Some(NodeKind::Dataset)) => {
                let (da, db) = (a.dataset(name)?, b.dataset(name)?);
                compare_attrs(&da, &db, &path, out)?;
                if da.shape() != db.shape() || crate::h5::data::kind(&da)? != crate::h5::data::kind(&db)? {
                    out.push(format!("{path}: shape/dtype differ"));
                } else if !data_equal(&da, &db)? {
                    out.push(format!("{path}: stored bytes differ"));
                }
            }
            (ka, kb) => out.push(format!("{path}: {} vs {}", kind_name(ka), kind_name(kb))),
        }
    }
    Ok(())
}

fn kind_name(kind: Option<NodeKind>) -> &'static str {
    match kind {
        Some(NodeKind::Group) => "Group",
        Some(NodeKind::Dataset) => "Dataset",
        _ => "Other",
    }
}

/// Every stored chunk of a dataset, exactly as it sits on disk.
pub fn raw_chunks(ds: &hdf5::Dataset) -> Result<Vec<Vec<u8>>> {
    if crate::h5::data::chunks(ds).is_none() {
        return Ok(Vec::new());
    }
    ops::stored_chunks(ds)?.iter().map(|offset| Ok(ops::read_raw_chunk(ds, offset)?.1)).collect()
}

fn data_equal(a: &hdf5::Dataset, b: &hdf5::Dataset) -> Result<bool> {
    let chunks_a = raw_chunks(a)?;
    let chunks_b = raw_chunks(b)?;
    if !chunks_a.is_empty() || !chunks_b.is_empty() {
        return Ok(chunks_a == chunks_b);
    }
    if crate::h5::data::chunks(a) != crate::h5::data::chunks(b) {
        return Ok(false);
    }
    match crate::h5::data::kind(a)? {
        crate::h5::data::Kind::Strings => Ok(crate::h5::data::read_strings(a)? == crate::h5::data::read_strings(b)?),
        _ => Ok(crate::h5::data::read(a)? == crate::h5::data::read(b)?),
    }
}
