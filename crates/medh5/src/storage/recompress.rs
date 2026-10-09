//! Re-encoding a file's bulk data under a different codec profile (§14.2).
//!
//! Safe at any point because **digests cover decompressed content** (§13.1):
//! recompression changes every stored byte and no digest, so `content_id`
//! before and after are identical.  The output is read back and verified
//! against the digests it carries.  Chunking is preserved unless `rechunk`
//! asks for the writer's own derivation.

use std::path::Path;

use indexmap::IndexMap;
use serde_json::{json, Value};

use super::chunking::{field_chunks, fit_chunks, grid_chunks};
use super::codecs::{dataset_layout, describe_filters, profile_names, resolve_profile, CodecProfile, Role};
use crate::array::{Index, Slice};
use crate::geometry::grid::{read_grids, Grid};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::file::{atomic_rewrite, open_read};
use crate::h5::ops::{self, NodeKind};
use crate::integrity::verify_root;
use crate::json::repr_str;
use crate::sample::reader::{attr_name_map_of, require_major};
use crate::validate::SAMPLES_GROUP;
use crate::{Error, Result};

/// What one file's re-encoding did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecompressResult {
    pub path: String,
    pub profile: String,
    pub datasets: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub content_id: Option<String>,
    pub content_id_preserved: bool,
    /// Whether the *output* verifies: every object digest, and the root.
    pub verified: bool,
    pub mismatched: Vec<String>,
    pub unattested: Vec<String>,
    /// `(path, codec before, codec after)` for each dataset re-encoded.
    pub changed: Vec<(String, String, String)>,
}

impl RecompressResult {
    pub fn ratio(&self) -> f64 {
        if self.bytes_before == 0 {
            1.0
        } else {
            self.bytes_after as f64 / self.bytes_before as f64
        }
    }

    pub fn ok(&self) -> bool {
        self.verified && self.content_id_preserved
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "profile": self.profile,
            "datasets": self.datasets,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "ratio": crate::json::num(self.ratio()),
            "content_id": self.content_id,
            "content_id_preserved": self.content_id_preserved,
            "verified": self.verified,
            "mismatched": self.mismatched,
            "unattested": self.unattested,
            "ok": self.ok(),
            "changed": self.changed.iter().map(|(a, b, c)| json!([a, b, c])).collect::<Vec<_>>(),
        })
    }

    /// Python's `str()`.
    pub fn line(&self) -> String {
        format!(
            "{}: {}, {} datasets, {} -> {} bytes ({:.2}\u{d7})",
            self.path,
            self.profile,
            self.datasets,
            self.bytes_before,
            self.bytes_after,
            self.ratio()
        )
    }
}

/// Rewrite `path` with every bulk dataset re-encoded under `profile`.
///
/// Copy-on-write like every other mutation (§14.4).
pub fn recompress(path: &Path, profile: &str, out: Option<&Path>, rechunk: bool) -> Result<RecompressResult> {
    if !profile_names().contains(&profile) {
        let mut names = profile_names();
        names.sort_unstable();
        return Err(Error::invalid(format!(
            "unknown codec profile {}; expected one of {}",
            repr_str(profile),
            crate::json::repr_list(&names)
        )));
    }
    let codec = resolve_profile(Some(profile))?;
    refuse_projection(path)?;
    let target = out.unwrap_or(path);
    let mut result =
        RecompressResult { path: target.to_string_lossy().into_owned(), profile: profile.into(), ..Default::default() };
    result.bytes_before = std::fs::metadata(path)?.len();
    let before = atomic_rewrite(path, Some(target), |src, dst| {
        require_major(src, path)?;
        let before = attrs::get_str(src, "content_id")?;
        let src_root = src.as_group()?;
        let dst_root = dst.as_group()?;
        let mut copied = Copied::default();
        copied.note(&src_root, &dst_root);
        copy_group(&src_root, &dst_root, &codec, rechunk, &mut result, None, &mut copied, 0)?;
        Ok(before)
    })?;
    result.bytes_after = std::fs::metadata(target)?.len();
    let check = open_read(target)?;
    let after = attrs::get_str(&check, "content_id")?;
    let mut all_ok = true;
    let mut ids_ok = true;
    for (prefix, root) in sample_roots(&check.as_group()?)? {
        let verification = verify_root(&root, Some(&attr_name_map_of(&root)?), None, true)?;
        for name in verification.mismatched.iter().chain(&verification.malformed) {
            result.mismatched.push(format!("{prefix}{name}"));
        }
        for name in &verification.unattested {
            result.unattested.push(format!("{prefix}{name}"));
        }
        all_ok &= verification.ok();
        ids_ok &= verification.content_id_ok() != Some(false);
    }
    result.content_id = after.clone();
    result.verified = all_ok;
    result.content_id_preserved = before == after && ids_ok;
    Ok(result)
}

/// Refuse a file holding a version this engine reads only as a projection:
/// recompression replaces the file and then verifies it under the integrity
/// contract this engine implements, which a later minor may have extended
/// (1.1 §2.1) --- so the check could not vouch for what it rewrote.
fn refuse_projection(path: &Path) -> Result<()> {
    let source = open_read(path)?;
    require_major(&source, path)?;
    for (prefix, root) in sample_roots(&source.as_group()?)? {
        let version = attrs::get_str(&root, "medh5_version")?.unwrap_or_default();
        if crate::version::is_projection(&version) {
            return Err(Error::Version(format!(
                "{}{} is MEDH5 {version}; this engine implements up to {} and cannot verify a recompression of it \
                 (1.1 §2.1)",
                repr_str(&path.to_string_lossy()),
                if prefix.is_empty() {
                    String::new()
                } else {
                    format!(" member {}", repr_str(prefix.trim_end_matches('/')))
                },
                crate::FORMAT_VERSION
            )));
        }
    }
    Ok(())
}

/// `(path prefix, sample root)` for a sample, or each collection member.
pub fn sample_roots(root: &hdf5::Group) -> Result<Vec<(String, hdf5::Group)>> {
    let kind = attrs::get_str(root, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "collection" {
        return Ok(vec![(String::new(), root.clone())]);
    }
    let members = root.group(SAMPLES_GROUP)?;
    let mut out = Vec::new();
    for key in ops::members(&members)? {
        if let Some(g) = ops::child_group(&members, &key) {
            out.push((format!("{SAMPLES_GROUP}/{key}/"), g));
        }
    }
    Ok(out)
}

/// One sample root's grids, and the chunks the writer derives from them.
struct Layout {
    root: hdf5::Group,
    name: String,
    grids: Option<IndexMap<String, Grid>>,
}

impl Layout {
    fn new(root: &hdf5::Group) -> Layout {
        Layout { root: root.clone(), name: format!("{}/", root.name().trim_end_matches('/')), grids: None }
    }

    fn grid(&mut self, grid_id: Option<String>) -> Option<Grid> {
        if self.grids.is_none() {
            self.grids = Some(read_grids(&self.root).map(|g| g.into_iter().collect()).unwrap_or_default());
        }
        grid_id.and_then(|g| self.grids.as_ref().and_then(|m| m.get(&g).cloned()))
    }

    fn chunks_for(&mut self, ds: &hdf5::Dataset, parent: &hdf5::Group) -> Result<Option<Vec<usize>>> {
        let name = ds.name();
        let rest = name.strip_prefix(&self.name).unwrap_or(&name).to_string();
        let parts: Vec<&str> = rest.split('/').collect();
        let shape = ds.shape();
        let itemsize = match data::kind(ds)? {
            Kind::Numeric(d) => d.itemsize(),
            _ => return Ok(None),
        };
        let section = parts[0];
        if section == "images" && parts.len() == 2 {
            return Ok(match self.grid(attrs::get_str(ds, "grid")?) {
                None => None,
                Some(g) => fit_chunks(&grid_chunks(&g, itemsize, 0)?, &shape),
            });
        }
        if section == "images" && parts.len() == 3 && parts[2].chars().all(|c| c.is_ascii_digit()) {
            let levels = attrs::get_strs(parent, "grid_levels")?;
            let level: usize = parts[2].parse().unwrap_or(usize::MAX);
            let Some(levels) = levels.filter(|l| level < l.len()) else { return Ok(None) };
            return Ok(match self.grid(Some(levels[level].clone())) {
                None => None,
                Some(g) => fit_chunks(&grid_chunks(&g, itemsize, 0)?, &shape),
            });
        }
        if section == "annotations" && parts.len() == 3 && parts[2] == "data" {
            let Some(g) = self.grid(attrs::get_str(parent, "grid")?) else { return Ok(None) };
            if shape.len() < g.n_spatial() {
                return Ok(None);
            }
            return Ok(fit_chunks(&grid_chunks(&g, itemsize, shape.len() - g.n_spatial())?, &shape));
        }
        if section == "transforms" && parts.len() == 3 && parts[2] == "field" {
            return match self.grid(attrs::get_str(parent, "field_grid")?) {
                None => Ok(None),
                Some(g) => field_chunks(&g, &shape, itemsize),
            };
        }
        if section == "index" && parts.last() == Some(&"occupancy") && shape.len() > 1 {
            let mut c = vec![1];
            c.extend(&shape[1..]);
            return Ok(Some(c));
        }
        Ok(None)
    }
}

/// Where each source object went in the copy, by its identity: the copy keeps
/// the source's graph, so a second link to one object stays a link to one.
#[derive(Default)]
struct Copied(std::collections::HashMap<ops::ObjectId, String>);

impl Copied {
    fn note(&mut self, src: &hdf5::Group, dst: &hdf5::Group) {
        if let Some(id) = ops::object_id_of(src) {
            self.0.insert(id, dst.name());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_group(
    src: &hdf5::Group,
    dst: &hdf5::Group,
    codec: &CodecProfile,
    rechunk: bool,
    result: &mut RecompressResult,
    layout: Option<&mut Layout>,
    copied: &mut Copied,
    depth: usize,
) -> Result<()> {
    if depth > ops::MAX_DEPTH {
        return Err(Error::File(format!(
            "{} is nested more than {} groups deep; it is refused rather than followed",
            repr_str(&src.name()),
            ops::MAX_DEPTH
        )));
    }
    for key in attrs::names(src)? {
        attrs::copy_raw(src, dst, &key)?;
    }
    let mut own = if rechunk && ops::child_group(src, "grids").is_some() { Some(Layout::new(src)) } else { None };
    let mut layout = match own.as_mut() {
        Some(l) => Some(l),
        None => layout,
    };
    for name in ops::members(src)? {
        match ops::link_kind(src, &name) {
            // A soft link stays a soft link: it holds a path, and the copy has
            // the same paths.  Following it copied its target a second time
            // under another name --- and one pointing at an ancestor never
            // stopped.
            Some(ops::LinkKind::Soft) => {
                dst.link_soft(&ops::soft_link_target(src, &name)?, &name)?;
                continue;
            }
            // A second hard link to an object already copied is a link to the
            // copy, not a second copy; a link back to an ancestor is how a
            // cycle is stored, and stays one.
            Some(ops::LinkKind::Hard) => {
                if let Some(id) = ops::object_id_by_name(src, &name) {
                    if let Some(first) = copied.0.get(&id) {
                        dst.link_hard(first, &name)?;
                        continue;
                    }
                }
            }
            _ => {}
        }
        match ops::node_kind(src, &name) {
            Some(NodeKind::Group) => {
                let child = dst.create_group(&name)?;
                let source = src.group(&name)?;
                copied.note(&source, &child);
                copy_group(&source, &child, codec, rechunk, result, layout.as_deref_mut(), copied, depth + 1)?;
            }
            Some(NodeKind::Dataset) => {
                let id = ops::object_id_by_name(src, &name);
                copy_dataset(src, &name, dst, codec, rechunk, result, layout.as_deref_mut())?;
                if let Some(id) = id {
                    copied.0.insert(id, format!("{}/{name}", dst.name().trim_end_matches('/')));
                }
            }
            _ => ops::copy_object(src, &name, dst, &name)?,
        }
    }
    Ok(())
}

fn role(ds: &hdf5::Dataset) -> Role {
    let path = ds.name();
    if path.contains("/annotations/") || path.contains("/transforms/") || path.contains("/index/") {
        Role::Label
    } else {
        Role::Image
    }
}

fn copy_dataset(
    parent: &hdf5::Group,
    name: &str,
    dst: &hdf5::Group,
    codec: &CodecProfile,
    rechunk: bool,
    result: &mut RecompressResult,
    layout: Option<&mut Layout>,
) -> Result<()> {
    let ds = parent.dataset(name)?;
    let dtype = match data::kind(&ds)? {
        Kind::Numeric(d) => d,
        _ => return ops::copy_object(parent, name, dst, name),
    };
    let chunks = if rechunk {
        match layout {
            Some(l) => l.chunks_for(&ds, parent)?,
            None => None,
        }
    } else {
        data::chunks(&ds)
    };
    let shape = ds.shape();
    let new_layout = dataset_layout(&shape, dtype.itemsize(), codec, role(&ds), chunks);
    if new_layout.chunks.is_none() {
        // Below the compression threshold: copied through as stored, exactly
        // what a fresh write would leave contiguous.
        return ops::copy_object(parent, name, dst, name);
    }
    let before = describe_filters(&ds)?;
    let out = data::create_empty(dst, name, dtype, &shape, &new_layout)?;
    for (start, stop) in slabs(&shape, dtype.itemsize(), 64 << 20) {
        let block = data::read_region(&ds, &[Index::Slice(Slice::new(start as i64, stop as i64))])?;
        let mut starts = vec![0usize; shape.len()];
        starts[0] = start;
        data::write_region(&out, &block, &starts)?;
    }
    for key in attrs::names(&ds)? {
        attrs::copy_raw(&ds, &out, &key)?;
    }
    let after = describe_filters(&out)?;
    result.datasets += 1;
    if before != after {
        result.changed.push((ds.name(), before, after));
    }
    Ok(())
}

/// Row windows covering a dataset, each under `budget` bytes.
fn slabs(shape: &[usize], itemsize: usize, budget: usize) -> Vec<(usize, usize)> {
    if shape.is_empty() || shape[0] == 0 {
        return vec![(0, shape.first().copied().unwrap_or(0))];
    }
    let trailing: usize = shape[1..].iter().product::<usize>().max(1);
    let per_plane = (trailing * itemsize).max(1);
    let step = (budget / per_plane).clamp(1, shape[0]);
    let mut out = Vec::new();
    let mut start = 0;
    while start < shape[0] {
        let stop = (start + step).min(shape[0]);
        out.push((start, stop));
        start = stop;
    }
    out
}

/// Recompress many files.
pub fn recompress_paths(paths: &[&Path], profile: &str, rechunk: bool) -> Result<Vec<RecompressResult>> {
    paths.iter().map(|p| recompress(p, profile, None, rechunk)).collect()
}
