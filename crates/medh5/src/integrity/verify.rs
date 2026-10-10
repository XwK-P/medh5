//! Verification of digests, `content_id` and derived-index currency (§13).

use serde_json::{json, Value};

use super::digest::{
    collect_digests, compute_content_id, dataset_digest, dataset_digest_inspected, digested_objects, group_digest,
    root_algo, AttrNameMap, STREAM_BYTES,
};
use crate::array::NdArray;
use crate::digest::{parse_digest, DEFAULT_ALGO};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops::{self, LinkKind, Node, NodeKind};
use crate::json::repr_str;
use crate::{Error, Result};

/// Where every dataset is part of an object a `content_id` speaks for.
pub const ATTESTED_GROUPS: [&str; 4] = ["grids", "images", "annotations", "transforms"];

/// Every dataset of an object a `content_id` speaks for, at its path through
/// the attested group that holds it.  `clinical/` is attested where the
/// profile is declared (1.1 §8, E818); in a 1.0 file a group of that name is
/// somebody's extension.
///
/// Each group is walked on its own.  A walk of the whole root visits an
/// object once, at the first path that reaches it, so a hard link elsewhere
/// --- a root alias sorting before `clinical` --- reached an undigested column
/// first and took it out of its group: a pin, `verify` and a deep preflight
/// passed over it while the clinical reader read it (B03 of the 2.0
/// re-audit).  Soft links are followed, because readers follow them: a column
/// linked softly to storage under `index/` was read by every selection and
/// walked by nothing (B03 of the round-3 audit).
pub fn attested_datasets(root: &hdf5::Group) -> Result<Vec<(String, hdf5::Dataset)>> {
    let clinical =
        attrs::get_strs(root, "medh5_profiles")?.unwrap_or_default().iter().any(|p| p == crate::clinical::PROFILE);
    let mut out = Vec::new();
    for name in ATTESTED_GROUPS.into_iter().chain(clinical.then_some(crate::clinical::GROUP)) {
        let Some(group) = ops::child_group(root, name) else { continue };
        for (path, ds) in ops::datasets_resolving(&group)? {
            out.push((format!("{name}/{path}"), ds));
        }
    }
    Ok(out)
}

/// Of [`attested_datasets`], those whose object no dataset line of the root
/// covers.  The root is a Merkle root over *stored* digests (§13.2), so such a
/// dataset is content no address covers: a pin, a cache entry and `verify`
/// would all pass over it.
///
/// Covered is by identity.  A dataset without a digest is not covered; nor is
/// one whose object is listed under no path --- reached first through
/// `index/`, which the root excludes, say, or only through a soft link to
/// storage the root does not list.  A transform aliased under `index/` before
/// its sample was pinned changed by 100 mm under an unchanged pin (B03 of the
/// round-3 audit).
pub fn unattested(root: &hdf5::Group) -> Result<Vec<String>> {
    let covered = digested_objects(root, &["index"])?;
    let mut out: Vec<String> = attested_datasets(root)?
        .into_iter()
        .filter(|(_, ds)| ops::object_id_of(ds).is_none_or(|id| !covered.contains(&id)))
        .map(|(path, _)| path)
        .collect();
    out.sort();
    Ok(out)
}

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
    verify_root_inspecting(root, attr_names, partial, check_content_id, &mut |_, _| Ok(()))
}

/// [`verify_root`], handing every block of every digested dataset it reads to
/// `inspect` with the dataset's path (relative to `root`): what a check of
/// stored values needs, at the cost of no second read.
pub fn verify_root_inspecting(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    partial: Option<&[String]>,
    check_content_id: bool,
    inspect: &mut dyn FnMut(&str, &NdArray) -> Result<()>,
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
        let mut each = |block: &NdArray| inspect(&path, block);
        if &dataset_digest_inspected(&ds, &path, &algo, STREAM_BYTES, Some(&mut each))? != value {
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
        result.unattested = unattested(root)?;
    }
    result.undigested = undigested;
    result.content_id_declared = declared;
    result.stale_index = stale_index_entries(root)?;
    Ok(result)
}

/// Paths at which two object trees differ, byte for byte: structure,
/// attributes and *stored* chunk bytes.  Empty means a pure copy.
///
/// The trees are compared as graphs, the way a copy keeps them.  Each pair of
/// objects is compared once, so a hard link back to an ancestor is a cycle to
/// stop at rather than a recursion without end (a self-cycle overflowed the
/// stack, F13 of the round-4 audit); a second link to an object must pair with
/// a second link to its counterpart, so an alias and a duplicate of the same
/// bytes still differ; and a soft link is compared by the path it holds,
/// relative to its tree's root, not followed.
pub fn subtrees_identical(a: &hdf5::Group, b: &hdf5::Group) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut pairs = Pairs { roots: (a.name(), b.name()), ..Pairs::default() };
    pairs.pair(ops::object_id_of(a), ops::object_id_of(b));
    compare_groups(a, b, "", 0, &mut pairs, &mut out)?;
    out.sort();
    Ok(out)
}

/// The objects of two trees paired so far, both ways.
#[derive(Default)]
struct Pairs {
    a_to_b: std::collections::HashMap<ops::ObjectId, ops::ObjectId>,
    b_to_a: std::collections::HashMap<ops::ObjectId, ops::ObjectId>,
    /// The names of the two roots, which soft-link targets are relative to.
    roots: (String, String),
}

enum Paired {
    /// Not met before: compare them.
    New,
    /// Met before, as each other's counterparts: compared already.
    Again,
    /// One was met before with another counterpart: the link structure differs.
    Mismatched,
}

impl Pairs {
    fn pair(&mut self, a: Option<ops::ObjectId>, b: Option<ops::ObjectId>) -> Paired {
        let (Some(a), Some(b)) = (a, b) else { return Paired::New };
        match (self.a_to_b.get(&a), self.b_to_a.get(&b)) {
            (None, None) => {
                self.a_to_b.insert(a, b);
                self.b_to_a.insert(b, a);
                Paired::New
            }
            (Some(x), Some(y)) if *x == b && *y == a => Paired::Again,
            _ => Paired::Mismatched,
        }
    }
}

/// A soft link's target as a path inside the tree rooted at `root`: an
/// absolute target under the root loses the root's prefix, so a sample's
/// links compare equal in a file of their own and in a collection.
fn relative_target(root: &str, target: &str) -> String {
    let root = root.trim_end_matches('/');
    if root.is_empty() || !target.starts_with('/') {
        return target.to_string();
    }
    if target == root {
        return "/".to_string();
    }
    match target.strip_prefix(root) {
        Some(rest) if rest.starts_with('/') => rest.to_string(),
        _ => target.to_string(),
    }
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

fn compare_groups(
    a: &hdf5::Group,
    b: &hdf5::Group,
    prefix: &str,
    depth: usize,
    pairs: &mut Pairs,
    out: &mut Vec<String>,
) -> Result<()> {
    if depth > ops::MAX_DEPTH {
        return Err(Error::File(format!(
            "{} is nested more than {} groups deep; it is refused rather than followed",
            repr_str(&a.name()),
            ops::MAX_DEPTH
        )));
    }
    compare_attrs(a, b, prefix, out)?;
    let names_a = ops::members(a)?;
    let names_b = ops::members(b)?;
    for name in names_a.iter().filter(|n| !names_b.contains(n)).chain(names_b.iter().filter(|n| !names_a.contains(n))) {
        out.push(format!("{prefix}/{name}: present in only one tree"));
    }
    for name in names_a.iter().filter(|n| names_b.contains(n)) {
        let path = format!("{prefix}/{name}");
        let (link_a, link_b) = (ops::link_kind(a, name), ops::link_kind(b, name));
        if link_a == Some(LinkKind::Soft) || link_b == Some(LinkKind::Soft) {
            if link_a != link_b {
                out.push(format!("{path}: {} vs {}", link_name(link_a), link_name(link_b)));
            } else if relative_target(&pairs.roots.0, &ops::soft_link_target(a, name)?)
                != relative_target(&pairs.roots.1, &ops::soft_link_target(b, name)?)
            {
                out.push(format!("{path}: soft link targets differ"));
            }
            continue;
        }
        match pairs.pair(ops::object_id_by_name(a, name), ops::object_id_by_name(b, name)) {
            Paired::Again => continue,
            Paired::Mismatched => {
                out.push(format!("{path}: links to objects the other tree links to differently"));
                continue;
            }
            Paired::New => {}
        }
        match (ops::node_kind(a, name), ops::node_kind(b, name)) {
            (Some(NodeKind::Group), Some(NodeKind::Group)) => {
                compare_groups(&a.group(name)?, &b.group(name)?, &path, depth + 1, pairs, out)?
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

fn link_name(kind: Option<LinkKind>) -> &'static str {
    match kind {
        Some(LinkKind::Hard) => "hard link",
        Some(LinkKind::Soft) => "soft link",
        Some(LinkKind::External) => "external link",
        _ => "other link",
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A file whose root holds `images/CT` and the given extra links.
    fn tree(path: &std::path::Path, extra: impl FnOnce(&hdf5::File)) -> hdf5::File {
        let file = crate::h5::file::create_truncate(path).unwrap();
        let images = file.create_group("images").unwrap();
        images.new_dataset::<u8>().shape([4]).create("CT").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
        extra(&file);
        file
    }

    /// F13: a hard link back to an ancestor is a cycle, compared once ---
    /// the comparison of a self-cycle with itself recursed until the stack
    /// overflowed.
    #[test]
    fn f13_s14_4_a_cycle_is_compared_once() {
        let dir = tempfile::tempdir().unwrap();
        let cyclic = |file: &hdf5::File| {
            file.link_hard("/", "loop").unwrap();
            file.group("images").unwrap().link_hard("/images", "self").unwrap();
        };
        let a = tree(&dir.path().join("a.h5"), cyclic);
        let b = tree(&dir.path().join("b.h5"), cyclic);
        assert!(subtrees_identical(&a.as_group().unwrap(), &a.as_group().unwrap()).unwrap().is_empty());
        assert!(subtrees_identical(&a.as_group().unwrap(), &b.as_group().unwrap()).unwrap().is_empty());
        // A cycle against a tree without one differs, and says where.
        let c = tree(&dir.path().join("c.h5"), |file| file.link_hard("/", "loop").unwrap());
        let found = subtrees_identical(&a.as_group().unwrap(), &c.as_group().unwrap()).unwrap();
        assert!(found.iter().any(|d| d.starts_with("/images/self")), "{found:?}");
    }

    /// F13: an alias and a duplicate of the same bytes are different graphs.
    #[test]
    fn f13_s13_2_an_alias_is_not_a_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let alias = tree(&dir.path().join("alias.h5"), |file| file.link_hard("/images/CT", "zzz").unwrap());
        let copy = tree(&dir.path().join("copy.h5"), |file| {
            file.new_dataset::<u8>().shape([4]).create("zzz").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
        });
        let found = subtrees_identical(&alias.as_group().unwrap(), &copy.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/zzz: links to objects the other tree links to differently".to_string()]);
        let again = tree(&dir.path().join("again.h5"), |file| file.link_hard("/images/CT", "zzz").unwrap());
        assert!(subtrees_identical(&alias.as_group().unwrap(), &again.as_group().unwrap()).unwrap().is_empty());
    }

    /// F13: a soft link is compared by the path it holds, whether it closes a
    /// cycle, dangles or names a dataset, and never followed.
    #[test]
    fn f13_soft_links_compare_by_target() {
        let dir = tempfile::tempdir().unwrap();
        let soft = |target: &'static str| {
            move |file: &hdf5::File| {
                file.link_soft("/", "up").unwrap();
                file.link_soft("/nowhere", "dangling").unwrap();
                file.link_soft(target, "ct").unwrap();
            }
        };
        let a = tree(&dir.path().join("a.h5"), soft("/images/CT"));
        let b = tree(&dir.path().join("b.h5"), soft("/images/CT"));
        assert!(subtrees_identical(&a.as_group().unwrap(), &b.as_group().unwrap()).unwrap().is_empty());
        let c = tree(&dir.path().join("c.h5"), soft("images/CT"));
        let found = subtrees_identical(&a.as_group().unwrap(), &c.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/ct: soft link targets differ".to_string()]);
        // A soft link against a hard link to the same object is a difference.
        let d = tree(&dir.path().join("d.h5"), |file| {
            file.link_soft("/", "up").unwrap();
            file.link_soft("/nowhere", "dangling").unwrap();
            file.link_hard("/images/CT", "ct").unwrap();
        });
        let found = subtrees_identical(&a.as_group().unwrap(), &d.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/ct: soft link vs hard link".to_string()]);
        // Under another root, a target under that root is the same path.
        assert_eq!(relative_target("/samples/k", "/samples/k/images/CT"), "/images/CT");
        assert_eq!(relative_target("/samples/k", "/samples/k"), "/");
        assert_eq!(relative_target("/samples/k", "/samples/kk/x"), "/samples/kk/x");
        assert_eq!(relative_target("/", "/images/CT"), "/images/CT");
    }
}
