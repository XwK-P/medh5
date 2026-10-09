//! [`Sample`]: a read-only, lazy, timepoint-aware view of one sample root.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use super::image::{Image, SPEC_IMAGE_ATTRS};
use crate::annotations::header::SPEC_ANNOTATION_ATTRS;
use crate::annotations::{Annotation, Grids};
use crate::array::Slice;
use crate::clinical::{Clinical, Documents};
use crate::curation::identity::{Cohort, Identity};
use crate::curation::timeline::Timeline;
use crate::document::{SampleDocument, META_DATASET};
use crate::geometry::grid::{read_grids, Grid, SPEC_GRID_ATTRS};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, file, ops};
use crate::integrity::{compute_content_id, stale_index_entries, verify_root, AttrNameMap, VerifyResult};
use crate::json::{repr_list, repr_str};
use crate::labels::LabelSet;
use crate::storage::index::{read_indices, SamplingIndex};
use crate::transforms::model::{read_transforms, Transform, SPEC_TRANSFORM_ATTRS};
use crate::transforms::resolve::{frames_of_timepoint, resolve_between};
use crate::{Error, Result, FORMAT_VERSION};

/// The conformance profiles a sample may declare (1.0 §1.3; `clinical`, 1.1 §1).
pub const PROFILES: [&str; 10] =
    ["core", "seg", "det", "cls", "reg", "curation", "multiscale", "training", "longitudinal", "clinical"];

/// Root attributes covered by `content_id`; `created` and `generator` are
/// excluded so byte-identical samples written apart share an address (§13.2).
pub const ROOT_DIGEST_ATTRS: [&str; 3] = ["medh5_version", "medh5_kind", "medh5_profiles"];

/// Where frame-of-reference UIDs are named.
pub const FRAME_ATTRS: [(&str, &[&str]); 3] =
    [("grids", &["frame_uid"]), ("annotations", &["frame_uid"]), ("transforms", &["from_frame", "to_frame"])];

/// The `KeyError` a collection raises for a missing name.
pub fn missing_key(what: &str, key: &str, available: impl IntoIterator<Item = String>) -> Error {
    let mut names: Vec<String> = available.into_iter().collect();
    names.sort();
    Error::Key(format!("no {what} {}; available: {}", repr_str(key), repr_list(&names)))
}

fn cached<T>(cell: &OnceLock<Result<T>>, init: impl FnOnce() -> Result<T>) -> Result<&T> {
    match cell.get_or_init(init) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.clone()),
    }
}

/// A read-only view of one sample root.
///
/// Every collection is read once per handle: a `Sample` is read-only, and
/// `amend` is copy-on-write --- it replaces the inode, so an open handle never
/// sees an edit.
#[derive(Debug)]
pub struct Sample {
    pub path: Option<PathBuf>,
    pub root: hdf5::Group,
    handle: Option<hdf5::File>,
    document: OnceLock<Result<SampleDocument>>,
    grids: OnceLock<Result<Arc<Grids>>>,
    images: OnceLock<Result<IndexMap<String, Image>>>,
    annotations: OnceLock<Result<IndexMap<String, Arc<Annotation>>>>,
    transforms: OnceLock<Result<Arc<IndexMap<String, Transform>>>>,
    index: OnceLock<Result<IndexMap<String, SamplingIndex>>>,
    fresh: OnceLock<Result<BTreeSet<String>>>,
    clinical: OnceLock<Result<Option<Arc<Clinical>>>>,
    documents: OnceLock<Result<Option<Arc<Documents>>>>,
    resolved: Mutex<HashMap<(String, String), Option<Transform>>>,
}

impl Sample {
    /// A view of a root inside an open file (a standalone sample, or a
    /// collection member).
    pub fn from_root(root: hdf5::Group, handle: Option<hdf5::File>, path: Option<PathBuf>) -> Sample {
        Sample {
            path,
            root,
            handle,
            document: OnceLock::new(),
            grids: OnceLock::new(),
            images: OnceLock::new(),
            annotations: OnceLock::new(),
            transforms: OnceLock::new(),
            index: OnceLock::new(),
            fresh: OnceLock::new(),
            clinical: OnceLock::new(),
            documents: OnceLock::new(),
            resolved: Mutex::new(HashMap::new()),
        }
    }

    /// The open file, when this view owns one.
    pub fn handle(&self) -> Option<&hdf5::File> {
        self.handle.as_ref()
    }

    /// Release the file.  Further reads through this view fail.
    pub fn close(&mut self) {
        if let Some(file) = self.handle.take() {
            let _ = crate::h5::file::close_everything(file, false);
        }
    }

    /// Close the file and everything opened through it, when this view owns
    /// the file --- a collection member does not; its collection does.  An
    /// object read from the sample and still held becomes invalid rather than
    /// keeping the file open and locked.
    pub fn close_file(&self) -> Result<()> {
        match &self.handle {
            Some(file) => crate::h5::file::close_everything(file.clone(), false),
            None => Ok(()),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> Result<String> {
        Ok(format!(
            "Sample({}, {} timepoints, {} images, {} annotations)",
            repr_str(&self.identity()?.sample_id),
            self.timepoints()?.len(),
            self.images()?.len(),
            self.annotations()?.len()
        ))
    }

    // -- document -------------------------------------------------------------

    pub fn document(&self) -> Result<&SampleDocument> {
        cached(&self.document, || SampleDocument::loads(&read_document_text(&self.root)?))
    }

    pub fn identity(&self) -> Result<&Identity> {
        Ok(&self.document()?.identity)
    }

    pub fn cohort(&self) -> Result<&Cohort> {
        Ok(&self.document()?.cohort)
    }

    pub fn timepoints(&self) -> Result<&Timeline> {
        Ok(&self.document()?.timepoints)
    }

    pub fn label_set(&self) -> Result<Option<&LabelSet>> {
        Ok(self.document()?.label_set.as_ref())
    }

    pub fn version(&self) -> Result<String> {
        let v = attrs::require(&self.root, "medh5_version", "E001")?;
        Ok(v.as_str().unwrap_or_else(|| attrs::stringify_value(&v)))
    }

    pub fn kind(&self) -> Result<String> {
        Ok(attrs::get_str(&self.root, "medh5_kind")?.unwrap_or_else(|| "sample".into()))
    }

    /// The declared profiles; `{"core"}` when none are declared.
    pub fn profiles(&self) -> Result<BTreeSet<String>> {
        Ok(match attrs::get_strs(&self.root, "medh5_profiles")? {
            None => BTreeSet::from(["core".to_string()]),
            Some(v) => v.into_iter().collect(),
        })
    }

    pub fn content_id(&self) -> Result<Option<String>> {
        attrs::get_str(&self.root, "content_id")
    }

    /// What this engine can do with the sample: [`Full`] for 1.0 and 1.1;
    /// [`Projection`] for a higher minor, which reads only what this engine
    /// knows and is never amended (1.1 §2).
    ///
    /// [`Full`]: crate::version::Support::Full
    /// [`Projection`]: crate::version::Support::Projection
    pub fn support(&self) -> Result<crate::version::Support> {
        Ok(crate::version::support(&self.version()?))
    }

    /// The `clinical` profile's records (1.1 §3--§7), when the sample
    /// declares the profile; `None` otherwise.  Events and links are read
    /// now, document text when asked for.
    pub fn clinical(&self) -> Result<Option<&Arc<Clinical>>> {
        let found = cached(&self.clinical, || {
            if !self.profiles()?.contains(crate::clinical::PROFILE) {
                return Ok(None);
            }
            let projection = crate::version::is_projection(&self.version()?);
            Ok(Clinical::open(&self.root, projection)?.map(Arc::new))
        })?;
        Ok(found.as_ref())
    }

    /// The `clinical` profile's documents alone --- metadata and text
    /// offsets, not the events or the links --- when the sample declares the
    /// profile: what reading one report needs (1.1 §6).
    pub fn documents(&self) -> Result<Option<&Arc<Documents>>> {
        let found = cached(&self.documents, || {
            if !self.profiles()?.contains(crate::clinical::PROFILE) {
                return Ok(None);
            }
            let projection = crate::version::is_projection(&self.version()?);
            Ok(Documents::open(&self.root, projection)?.map(Arc::new))
        })?;
        Ok(found.as_ref())
    }

    // -- objects -----------------------------------------------------------------

    /// The sample's grids, with §3.7's implicit timepoint resolved: with
    /// exactly one declared timepoint, a grid that names none belongs to it.
    pub fn grids(&self) -> Result<&Arc<Grids>> {
        cached(&self.grids, || {
            let stored: Grids = read_grids(&self.root)?.into_iter().collect();
            let declared = match self.timepoints() {
                Ok(t) => t.ids(),
                Err(e) if e.is_medh5() => return Ok(Arc::new(stored)),
                Err(e) => return Err(e),
            };
            if declared.len() != 1 {
                return Ok(Arc::new(stored));
            }
            let only = declared[0].clone();
            Ok(Arc::new(
                stored
                    .into_iter()
                    .map(|(gid, g)| {
                        let g = if g.timepoint.is_none() { g.with_timepoint(Some(only.clone())) } else { g };
                        (gid, g)
                    })
                    .collect(),
            ))
        })
    }

    /// A grid by id (`KeyError` naming the available ones).
    pub fn grid(&self, grid_id: &str) -> Result<&Grid> {
        let grids = self.grids()?;
        grids.get(grid_id).ok_or_else(|| Error::Key(repr_str(grid_id)))
    }

    /// `grids/ref` when present, else the grid of the first image (§3.2).
    pub fn reference_grid(&self) -> Result<Grid> {
        let grids = self.grids()?;
        if let Some(g) = grids.get("ref") {
            return Ok(g.clone());
        }
        let images = self.images()?;
        if let Some((_, first)) = images.first() {
            return Ok(first.grid()?.clone());
        }
        if grids.is_empty() {
            return Err(Error::coded(
                "E111",
                format!(
                    "{} declares no grids, so it has no reference grid",
                    self.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "<memory>".into())
                ),
            ));
        }
        let mut names: Vec<&String> = grids.keys().collect();
        names.sort();
        Ok(grids[names[0].as_str()].clone())
    }

    /// The sample's images, by id (name order).
    pub fn images(&self) -> Result<&IndexMap<String, Image>> {
        cached(&self.images, || {
            let grids = self.grids()?.clone();
            let Some(node) = ops::child_group(&self.root, "images") else {
                return Ok(IndexMap::new());
            };
            ops::members(&node)?
                .into_iter()
                .map(|name| Ok((name.clone(), Image::open(&name, &node, grids.clone())?)))
                .collect()
        })
    }

    /// One image (`KeyError` naming the available ones).
    pub fn image(&self, image_id: &str) -> Result<&Image> {
        let images = self.images()?;
        images.get(image_id).ok_or_else(|| missing_key("image", image_id, images.keys().cloned()))
    }

    /// The sample's annotations, by id (name order).
    pub fn annotations(&self) -> Result<&IndexMap<String, Arc<Annotation>>> {
        cached(&self.annotations, || {
            let grids = self.grids()?.clone();
            let label_set = self.label_set()?.cloned().map(Arc::new);
            let Some(node) = ops::child_group(&self.root, "annotations") else {
                return Ok(IndexMap::new());
            };
            ops::members(&node)?
                .into_iter()
                .filter_map(|name| ops::child_group(&node, &name).map(|g| (name, g)))
                .map(|(name, group)| {
                    Ok((name.clone(), Arc::new(Annotation::open(&name, group, grids.clone(), label_set.clone())?)))
                })
                .collect()
        })
    }

    /// One annotation (`KeyError` naming the available ones).
    pub fn annotation(&self, ann_id: &str) -> Result<&Arc<Annotation>> {
        let anns = self.annotations()?;
        anns.get(ann_id).ok_or_else(|| missing_key("annotation", ann_id, anns.keys().cloned()))
    }

    /// The §7.7 ignore region of a voxel annotation, under any encoding: in
    /// band (`labelmap`, `layers`) and/or the `mask` named by
    /// `header.ignore_mask`.  All `false` where none is declared.
    pub fn ignore_region(&self, ann_id: &str, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        let annotation = self.annotation(ann_id)?;
        if !annotation.is_voxel() {
            return Err(Error::invalid(format!(
                "annotation {} is a {}, not a voxel annotation; the §7.7 ignore region is defined on voxels",
                repr_str(ann_id),
                repr_str(annotation.kind())
            )));
        }
        let window = annotation.window(roi)?;
        let shape: Vec<usize> = window.iter().map(|(a, b)| b.saturating_sub(*a)).collect();
        let mut region = ArrayD::from_elem(IxDyn(&shape), false);
        if matches!(annotation.kind(), "labelmap" | "layers") {
            let slices = window_slices(&window);
            region.zip_mut_with(&annotation.ignore_mask(Some(&slices))?, |r, v| *r |= *v);
        }
        if let Some(reference) = annotation.header.ignore_mask.clone() {
            let mask = self.referenced_mask(ann_id, "ignore_mask", &reference, &window_slices(&window))?;
            region.zip_mut_with(&mask, |r, v| *r |= *v);
        }
        Ok(region)
    }

    /// Where an image holds data (§4.4): its `valid_mask`, or everywhere.
    pub fn valid_region(&self, image_id: &str, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        let image = self.image(image_id)?;
        let shape = image.grid()?.spatial_shape();
        let window: Vec<(i64, i64)> = match roi {
            None => shape.iter().map(|n| (0, *n as i64)).collect(),
            Some(r) => r.iter().zip(&shape).map(|(s, n)| (s.start.unwrap_or(0), s.stop.unwrap_or(*n as i64))).collect(),
        };
        match image.valid_mask()? {
            None => {
                let dims: Vec<usize> = window.iter().map(|(a, b)| (b - a).max(0) as usize).collect();
                Ok(ArrayD::from_elem(IxDyn(&dims), true))
            }
            Some(reference) => {
                let slices: Vec<Slice> = window.iter().map(|(a, b)| Slice::new(*a, *b)).collect();
                self.referenced_mask(image_id, "valid_mask", &reference, &slices)
            }
        }
    }

    fn referenced_mask(&self, owner: &str, attr: &str, reference: &str, window: &[Slice]) -> Result<ArrayD<bool>> {
        let name = annotation_id(reference);
        let target = self.annotations()?.get(name).filter(|a| a.kind() == "mask");
        match target {
            None => Err(Error::coded(
                "E413",
                format!(
                    "{}: {attr} names {}, which is not a `mask` annotation in this file",
                    repr_str(owner),
                    repr_str(name)
                ),
            )),
            Some(t) => t.read_mask(Some(window)),
        }
    }

    /// The sampling index entries, by annotation id.
    pub fn index(&self) -> Result<&IndexMap<String, SamplingIndex>> {
        cached(&self.index, || read_indices(&self.root))
    }

    /// Index entries whose `source_digest` still matches their source.
    pub fn fresh_indices(&self) -> Result<&BTreeSet<String>> {
        cached(&self.fresh, || {
            let stale: BTreeSet<String> = stale_index_entries(&self.root)?.into_iter().collect();
            Ok(self.index()?.keys().filter(|n| !stale.contains(*n)).cloned().collect())
        })
    }

    /// The file's transforms, read once.
    pub fn transforms(&self) -> Result<&Arc<IndexMap<String, Transform>>> {
        cached(&self.transforms, || Ok(Arc::new(read_transforms(&self.root, self.grids()?.clone())?)))
    }

    /// One transform (`KeyError` naming the available ones).
    pub fn transform(&self, transform_id: &str) -> Result<&Transform> {
        let all = self.transforms()?;
        all.get(transform_id).ok_or_else(|| missing_key("transform", transform_id, all.keys().cloned()))
    }

    /// The transform relating two timepoints, grids or frames (§10).
    ///
    /// A key is read as a timepoint first, then as a grid, then as a frame
    /// uid.  `None` when the two already share a frame or no path exists; an
    /// unknown key is a `KeyError`.
    pub fn transform_between(&self, source: &str, target: &str) -> Result<Option<Transform>> {
        let sources = self.frames_for(source)?;
        let targets = self.frames_for(target)?;
        for a in &sources {
            for b in &targets {
                if a == b {
                    return Ok(None);
                }
                if let Some(found) = self.resolve_frames(a, b)? {
                    return Ok(Some(found));
                }
            }
        }
        Ok(None)
    }

    /// The transform relating two frame uids, resolved once per handle.
    pub fn resolve_frames(&self, from_frame: &str, to_frame: &str) -> Result<Option<Transform>> {
        if from_frame == to_frame {
            return Ok(None);
        }
        let key = (from_frame.to_string(), to_frame.to_string());
        if let Some(found) = self.resolved.lock().expect("resolution cache").get(&key) {
            return Ok(found.clone());
        }
        let found = resolve_between(self.transforms()?, from_frame, to_frame)?;
        self.resolved.lock().expect("resolution cache").insert(key, found.clone());
        Ok(found)
    }

    /// The frames a key names: a timepoint's (every grid's at that visit),
    /// else a grid's own, else the key as a frame uid --- in that order,
    /// because the namespaces are separate (§2.3) and may collide.
    pub fn frames_for(&self, key: &str) -> Result<Vec<String>> {
        let timeline = self.timepoints()?;
        if timeline.contains(key) {
            return Ok(frames_of_timepoint(self.grids()?, key));
        }
        if let Some(grid) = self.grids()?.get(key) {
            return Ok(grid.frame_uid.iter().filter(|f| !f.is_empty()).cloned().collect());
        }
        if self.known_frames()?.contains(key) {
            return Ok(vec![key.to_string()]);
        }
        Err(Error::Key(format!(
            "{} is not a timepoint, a grid or a frame of reference in this sample (timepoints: {})",
            repr_str(key),
            repr_list(&timeline.ids())
        )))
    }

    fn known_frames(&self) -> Result<BTreeSet<String>> {
        let mut frames: BTreeSet<String> =
            self.grids()?.values().filter_map(|g| g.frame_uid.clone()).filter(|f| !f.is_empty()).collect();
        for t in self.transforms()?.values() {
            frames.insert(t.from_frame().to_string());
            frames.insert(t.to_frame().to_string());
        }
        if let Some(node) = ops::child_group(&self.root, "annotations") {
            for name in ops::members(&node)? {
                if let Some(g) = ops::child_group(&node, &name) {
                    if let Some(f) = attrs::get_str(&g, "frame_uid")? {
                        frames.insert(f);
                    }
                }
            }
        }
        Ok(frames)
    }

    pub fn is_longitudinal(&self) -> Result<bool> {
        Ok(self.timepoints()?.is_longitudinal())
    }

    /// Image ids on a timepoint, in name order.
    pub fn images_at(&self, timepoint: &str) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for (name, image) in self.images()? {
            if image.timepoint()?.as_deref() == Some(timepoint) {
                out.push(name.clone());
            }
        }
        Ok(out)
    }

    /// Annotation ids pertaining to a timepoint, in name order.
    pub fn annotations_at(&self, timepoint: &str) -> Result<Vec<String>> {
        Ok(self
            .annotations()?
            .iter()
            .filter(|(_, a)| a.timepoints().iter().any(|t| t == timepoint))
            .map(|(n, _)| n.clone())
            .collect())
    }

    // -- integrity -------------------------------------------------------------------

    /// Object path -> spec-defined attribute names, for `content_id`.
    pub fn attr_name_map(&self) -> Result<AttrNameMap> {
        attr_name_map_of(&self.root)
    }

    /// Verify digests, `content_id` and index currency.
    pub fn verify(&self, partial: Option<&[String]>) -> Result<VerifyResult> {
        verify_root(&self.root, Some(&self.attr_name_map()?), partial, true)
    }

    /// Recompute the `content_id` from the stored digests.
    pub fn compute_content_id(&self) -> Result<String> {
        let algo = attrs::get_str(&self.root, "digest_algo")?.unwrap_or_else(|| "sha256".into());
        compute_content_id(&self.root, &self.attr_name_map()?, &algo, None)
    }

    // -- reporting ---------------------------------------------------------------------

    pub fn summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("path".into(), json!(self.path.as_ref().map(|p| p.to_string_lossy().into_owned())));
        out.insert("version".into(), json!(self.version()?));
        out.insert("kind".into(), json!(self.kind()?));
        out.insert("profiles".into(), json!(self.profiles()?.into_iter().collect::<Vec<_>>()));
        out.insert("content_id".into(), json!(self.content_id()?));
        if let Value::Object(doc) = self.document()?.summary() {
            out.extend(doc);
        }
        out.insert("grids".into(), Value::Array(self.grids()?.values().map(Grid::summary).collect()));
        out.insert("images".into(), Value::Array(self.images()?.values().map(Image::summary).collect::<Result<_>>()?));
        out.insert(
            "annotations".into(),
            Value::Array(self.annotations()?.values().map(|a| a.summary()).collect::<Result<_>>()?),
        );
        out.insert(
            "transforms".into(),
            Value::Array(self.transforms()?.values().map(Transform::summary).collect::<Result<_>>()?),
        );
        let mut index: Vec<String> = self.index()?.keys().cloned().collect();
        index.sort();
        out.insert("index".into(), json!(index));
        if let Some(clinical) = self.clinical()? {
            out.insert("clinical".into(), clinical.summary());
        }
        Ok(Value::Object(out))
    }
}

fn window_slices(window: &[(usize, usize)]) -> Vec<Slice> {
    window.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect()
}

/// Read the raw `/meta` payload as text.
pub fn read_document_text(root: &hdf5::Group) -> Result<String> {
    match ops::child_dataset(root, META_DATASET) {
        None => Err(Error::Schema(format!("{}: required dataset `meta` is absent", root.name()))),
        Some(ds) => data::read_scalar_string(&ds),
    }
}

/// Read and parse `<root>/meta`.
pub fn read_document(root: &hdf5::Group) -> Result<SampleDocument> {
    SampleDocument::loads(&read_document_text(root)?)
}

/// Every frame-of-reference UID in a file, and the attributes naming it.
pub fn frame_references(root: &hdf5::Group) -> Result<IndexMap<String, Vec<String>>> {
    let mut out: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (group, names) in FRAME_ATTRS {
        let Some(node) = ops::child_group(root, group) else { continue };
        for name in ops::members(&node)? {
            let loc: Option<hdf5::Location> = match ops::node_kind(&node, &name) {
                Some(ops::NodeKind::Group) => Some((*node.group(&name)?).clone()),
                Some(ops::NodeKind::Dataset) => {
                    let d = node.dataset(&name)?;
                    Some((**d).clone())
                }
                _ => None,
            };
            let Some(loc) = loc else { continue };
            for attr in names {
                let uid = attrs::get_str(&loc, attr)?.unwrap_or_default();
                if !uid.is_empty() {
                    out.entry(uid).or_default().push(format!("{group}.{name}.{attr}"));
                }
            }
        }
    }
    Ok(out.into_iter().collect())
}

/// Object path -> spec-defined attribute names, for `content_id` (§13.2).
///
/// Derived from the file, and shared by the reader and the writer, so the two
/// agree by construction.
pub fn attr_name_map_of(root: &hdf5::Group) -> Result<AttrNameMap> {
    let mut out = AttrNameMap::new();
    out.insert(String::new(), ROOT_DIGEST_ATTRS.to_vec());
    let groups: [(&str, &[&'static str]); 4] = [
        ("grids", &SPEC_GRID_ATTRS),
        ("images", &SPEC_IMAGE_ATTRS),
        ("annotations", &SPEC_ANNOTATION_ATTRS),
        ("transforms", &SPEC_TRANSFORM_ATTRS),
    ];
    for (group, names) in groups {
        let Some(node) = ops::child_group(root, group) else { continue };
        let names: Vec<&'static str> = names.iter().copied().filter(|a| *a != "digest").collect();
        for name in ops::members(&node)? {
            out.insert(format!("{group}/{name}"), names.clone());
        }
    }
    Ok(out)
}

/// An annotation id from a reference that may be written as a path.
pub fn annotation_id(reference: &str) -> &str {
    reference.strip_prefix("annotations/").unwrap_or(reference)
}

/// The file's `medh5_version`, or a refusal of a major this reader lacks.
pub fn require_major(root: &hdf5::Location, path: &Path) -> Result<String> {
    let shown = repr_str(&path.to_string_lossy());
    let Some(version) = attrs::read(root, "medh5_version")? else {
        return Err(Error::Version(format!(
            "{shown} declares no `medh5_version`; a 0.x file must be converted with `medh5 migrate`"
        )));
    };
    let text = match version {
        AttrValue::Str(s) => s,
        other => attrs::stringify_value(&other),
    };
    let major = text.split('.').next().unwrap_or("");
    if major != FORMAT_VERSION.split('.').next().unwrap_or("") {
        return Err(Error::Version(format!("{shown} is MEDH5 {text}; this reader implements {FORMAT_VERSION}")));
    }
    Ok(text)
}

/// Open a `.medh5` sample file, read-only.
pub fn open_sample(path: &Path) -> Result<Sample> {
    let handle = file::open_read(path)?;
    require_major(&handle, path)?;
    let kind = attrs::get_str(&handle, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "sample" {
        return Err(Error::File(format!(
            "{} is a {}; open it with open_collection()",
            repr_str(&path.to_string_lossy()),
            repr_str(&kind)
        )));
    }
    let root = handle.as_group()?;
    Ok(Sample::from_root(root, Some(handle), Some(path.to_path_buf())))
}

/// Rewrite `path` so freed space is not carried forward (§14.4).
///
/// HDF5 does not reclaim storage: an amend that copies an object and *then*
/// rewrites one of its attributes leaves the superseded value physically in
/// the file, where `strings` still finds it.  For de-identification that is
/// the difference between a pseudonymised file and one still carrying the
/// original UID.  Every top-level object is copied into a fresh file;
/// filters, chunking and attributes come across untouched, so every digest
/// and the `content_id` survive.
pub fn repack(path: &Path) -> Result<()> {
    crate::h5::file::atomic_rewrite(path, None, |src, dst| {
        require_major(src, path)?;
        let (src_root, dst_root) = (src.as_group()?, dst.as_group()?);
        ops::refuse_references(&src_root, "a repack")?;
        for name in ops::members(&src_root)? {
            ops::copy_object(&src_root, &name, &dst_root, &name)?;
        }
        for key in attrs::names(src)? {
            attrs::copy_raw(src, dst, &key)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::NdArray;

    /// A repack rewrites storage, not content: the digests, the `content_id`,
    /// every image's values and the file's permissions come through.
    #[test]
    fn repack_preserves_content_and_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::bench::synthetic_pair(dir.path(), &[8, 12, 10], "portable", 7).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let snapshot = |path: &Path| {
            let sample = open_sample(path).unwrap();
            let images: Vec<(String, NdArray)> = sample
                .images()
                .unwrap()
                .iter()
                .map(|(id, image)| (id.clone(), image.read(None, false, None).unwrap()))
                .collect();
            (sample.content_id().unwrap(), images)
        };
        let before = snapshot(&path);
        repack(&path).unwrap();
        assert_eq!(snapshot(&path), before);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
