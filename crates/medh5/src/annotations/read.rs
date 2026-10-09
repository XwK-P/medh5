//! The uniform read contract (spec §6, §7.6): one [`Annotation`] per stored
//! group, whose methods dispatch on its `kind`.
//!
//! Every voxel encoding answers the same predicate ---
//! `contains(class, voxel) -> bool` --- so callers ask for classes and a
//! region of interest and never for a layout.  The **coverage contract** rides
//! alongside: `class_ids` is what an annotation can express,
//! `annotated_class_ids` what the annotator committed to finding, and `0` at a
//! voxel means "verified absent" only for the second.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use super::encode::{contains_at, decode_crop, BITS_PER_PLANE, DEFAULT_THRESHOLD};
use super::header::{is_geometric_kind, is_voxel_kind, AnnotationHeader};
use super::payload::{contains_value, count_nonzero, popcounts, value_counts};
use crate::array::{DType, Element, Index, NdArray, Slice};
use crate::geometry::affine::{box_to_slices, slices_to_box};
use crate::geometry::grid::Grid;
use crate::h5::{attrs, data, ops};
use crate::json::{py_float, repr_str};
use crate::labels::{ClassKey, LabelClass, LabelSet, IGNORE_ID};
use crate::{with_array, Error, Result};

/// The grids an annotation can be read on, by id.
pub type Grids = IndexMap<String, Grid>;

/// A grid named in a call: the annotation's own, one by id, or one in hand.
#[derive(Debug, Clone, Copy)]
pub enum GridRef<'a> {
    Own,
    Id(&'a str),
    Grid(&'a Grid),
}

/// One physical object: a box, a class, an id and optionally a cropped mask.
///
/// `instance_id` is **sample-scoped**: the same lesion observed at several
/// timepoints reuses its id, so lesion tracking is a join on this field.
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub index: usize,
    pub instance_id: u64,
    pub class_id: i64,
    /// `(S, 2)` box in continuous index coordinates.
    pub bbox: Array2<f32>,
    pub mask: Option<ArrayD<bool>>,
    pub score: Option<f64>,
}

impl Instance {
    /// The box as index slices, unclipped.
    pub fn slices(&self) -> Result<Vec<(i64, i64)>> {
        let flat: Vec<f64> = self.bbox.iter().map(|v| f64::from(*v)).collect();
        box_to_slices(&flat, None)
    }

    /// Foreground voxels: the mask's, or the box's when there is no mask.
    pub fn voxel_count(&self) -> Result<u64> {
        match &self.mask {
            Some(mask) => Ok(mask.iter().filter(|v| **v).count() as u64),
            None => Ok(self.slices()?.iter().map(|(a, b)| (b - a).max(0) as u64).product()),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("Instance(id={}, class={}, box={})", self.instance_id, self.class_id, box_repr(&self.bbox))
    }
}

/// `box.tolist()` of a float32 `(S, 2)` array, as Python prints it.
pub fn box_repr(bbox: &Array2<f32>) -> String {
    let rows: Vec<String> = bbox
        .outer_iter()
        .map(|row| format!("[{}]", row.iter().map(|v| py_float(f64::from(*v))).collect::<Vec<_>>().join(", ")))
        .collect();
    format!("[{}]", rows.join(", "))
}

/// The `layers` table: `(L, K)` class ids and class id -> layer.
#[derive(Debug)]
struct LayerTable {
    table: Array2<u16>,
    layer_of: IndexMap<i64, usize>,
}

/// The `bitmask` table: bit position -> class id, and its inverse.
#[derive(Debug)]
struct BitTable {
    ids: Vec<u16>,
    position_of: HashMap<i64, usize>,
}

/// The per-object columns of an `instances` annotation, read once.
#[derive(Debug)]
struct InstanceColumns {
    boxes: ArrayD<f32>,
    class_ids: Vec<u16>,
    instance_ids: Vec<u64>,
    scores: Option<Vec<f32>>,
    mask_offsets: Option<Vec<u64>>,
    mask_shapes: Option<ArrayD<i64>>,
}

/// One annotation group, opened (spec §6).
///
/// The small per-kind tables --- `layers`' layer table, `bitmask`'s bit
/// table, `instances`' object columns --- are read once per open annotation
/// and kept: a sample is read-only and `amend` replaces the inode rather than
/// editing in place, so an open reader cannot see them change.
#[derive(Debug)]
pub struct Annotation {
    pub ann_id: String,
    pub group: hdf5::Group,
    pub header: AnnotationHeader,
    grids: Arc<Grids>,
    label_set: Option<Arc<LabelSet>>,
    layers: OnceLock<Result<LayerTable>>,
    bits: OnceLock<Result<BitTable>>,
    columns: OnceLock<Result<InstanceColumns>>,
}

fn cached<T>(cell: &OnceLock<Result<T>>, init: impl FnOnce() -> Result<T>) -> Result<&T> {
    match cell.get_or_init(init) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.clone()),
    }
}

/// A region of interest resolved against a grid: `(start, stop)` per axis.
pub type Window = Vec<(usize, usize)>;

fn window_shape(window: &Window) -> Vec<usize> {
    window.iter().map(|(a, b)| b.saturating_sub(*a)).collect()
}

fn window_index(prefix: Option<usize>, window: &Window) -> Vec<Index> {
    let mut out: Vec<Index> = prefix.map(|p| Index::At(p as i64)).into_iter().collect();
    out.extend(window.iter().map(|(a, b)| Index::Slice(Slice::new(*a as i64, *b as i64))));
    out
}

/// Elementwise `block == value`, compared in the block's own type.
pub fn equals(block: &NdArray, value: i64) -> ArrayD<bool> {
    match block {
        NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => block.to_f64().mapv(|v| v == value as f64),
        _ => with_array!(block, a => a.mapv(|v| v.to_i128() == i128::from(value))),
    }
}

impl Annotation {
    /// Open an annotation group as the reader matching its `kind`.
    pub fn open(
        ann_id: &str,
        group: hdf5::Group,
        grids: Arc<Grids>,
        label_set: Option<Arc<LabelSet>>,
    ) -> Result<Annotation> {
        let header = AnnotationHeader::read(&group)?;
        Ok(Annotation {
            ann_id: ann_id.to_string(),
            group,
            header,
            grids,
            label_set,
            layers: OnceLock::new(),
            bits: OnceLock::new(),
            columns: OnceLock::new(),
        })
    }

    // -- header passthrough -------------------------------------------------

    pub fn kind(&self) -> &str {
        &self.header.kind
    }

    pub fn task(&self) -> &str {
        &self.header.task
    }

    pub fn class_ids(&self) -> &[i64] {
        &self.header.class_ids
    }

    pub fn annotated_class_ids(&self) -> &[i64] {
        &self.header.annotated_class_ids
    }

    pub fn closure(&self) -> &str {
        &self.header.closure
    }

    pub fn ignore_id(&self) -> i64 {
        self.header.ignore_id
    }

    pub fn prov(&self) -> Option<&str> {
        self.header.prov.as_deref()
    }

    pub fn quality_key(&self) -> Option<&str> {
        self.header.quality.as_deref()
    }

    pub fn label_set(&self) -> Option<&LabelSet> {
        self.label_set.as_deref()
    }

    pub fn grid_id(&self) -> Option<&str> {
        self.header.grid.as_deref()
    }

    /// The grids this annotation can be read on.
    pub fn grids(&self) -> &Grids {
        &self.grids
    }

    /// Whether this is a voxel kind (§7).
    pub fn is_voxel(&self) -> bool {
        is_voxel_kind(self.kind())
    }

    /// Whether this is a geometric kind (§8).
    pub fn is_geometric(&self) -> bool {
        is_geometric_kind(self.kind())
    }

    /// The reader class name the Python bindings expose for this kind.
    pub fn class_name(&self) -> &'static str {
        match self.kind() {
            "labelmap" => "LabelmapAnnotation",
            "layers" => "LayersAnnotation",
            "bitmask" => "BitmaskAnnotation",
            "instances" => "InstancesAnnotation",
            "probmap" => "ProbmapAnnotation",
            "mask" => "MaskAnnotation",
            "boxes" => "BoxesAnnotation",
            "obb" => "ObbAnnotation",
            "keypoints" => "KeypointsAnnotation",
            "points" => "PointsAnnotation",
            "contours" => "ContoursAnnotation",
            "mesh" => "MeshAnnotation",
            _ => "ClassificationAnnotation",
        }
    }

    /// The annotation's grid, or an error naming why there is none.
    pub fn grid(&self) -> Result<&Grid> {
        let gid = self.header.grid.as_deref().ok_or_else(|| {
            Error::invalid(format!(
                "annotation {} of kind {} has no grid",
                repr_str(&self.ann_id),
                repr_str(self.kind())
            ))
        })?;
        self.grids.get(gid).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("annotation {} names grid {}, which does not exist", repr_str(&self.ann_id), repr_str(gid)),
            )
        })
    }

    /// Timepoints this annotation pertains to, inherited from the grid (§3.7).
    pub fn timepoints(&self) -> Vec<String> {
        if let Some(tps) = &self.header.timepoints {
            return tps.clone();
        }
        match self.header.grid.as_deref().and_then(|g| self.grids.get(g)) {
            Some(grid) => grid.timepoint.iter().filter(|t| !t.is_empty()).cloned().collect(),
            None => Vec::new(),
        }
    }

    // -- coverage -----------------------------------------------------------

    /// Whether the annotator committed to finding this class (§11.3).
    pub fn is_annotated(&self, key: &ClassKey) -> Result<bool> {
        let id = self.resolve_class(key)?;
        Ok(self.header.annotated_class_ids.contains(&id))
    }

    /// Whether every class this annotation can express was looked for.
    pub fn is_fully_covered(&self) -> bool {
        let mut a = self.header.annotated_class_ids.clone();
        let mut b = self.header.class_ids.clone();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        a == b
    }

    /// Whether an ignore region exists, named or in band (§7.7).
    pub fn has_ignore_region(&self) -> Result<bool> {
        Ok(self.header.ignore_mask.is_some() || self.encodes_ignore()?)
    }

    /// Whether the data itself carries the ignore id (`labelmap`, `layers`).
    pub fn encodes_ignore(&self) -> Result<bool> {
        match self.kind() {
            "labelmap" | "layers" => contains_value(&self.data()?, self.ignore_id()),
            _ => Ok(false),
        }
    }

    /// Resolve a class id or key to an id, using the label set for keys.
    pub fn resolve_class(&self, key: &ClassKey) -> Result<i64> {
        match key {
            ClassKey::Id(id) => Ok(*id),
            ClassKey::Key(name) => match &self.label_set {
                None => Err(Error::invalid(format!(
                    "annotation {}: cannot resolve class name {} without a label set",
                    repr_str(&self.ann_id),
                    repr_str(name)
                ))),
                Some(ls) => Ok(ls.lookup(key)?.id),
            },
        }
    }

    /// Resolve requested class keys to ids, refusing the reserved ignore id.
    ///
    /// `65535` is not a class --- §5.2 says it MUST NOT appear in `classes`
    /// --- so nothing that returns per-class planes can answer for it, and an
    /// all-zero plane would be indistinguishable from a class examined and
    /// found absent.
    pub fn resolve_classes(&self, keys: Option<&[ClassKey]>) -> Result<Vec<i64>> {
        let Some(keys) = keys else {
            return Ok(self.header.class_ids.clone());
        };
        let ids = keys.iter().map(|k| self.resolve_class(k)).collect::<Result<Vec<_>>>()?;
        if ids.contains(&IGNORE_ID) {
            return Err(Error::coded(
                "E404",
                format!(
                    "annotation {}: {IGNORE_ID} is the reserved ignore id, not a class, so no plane can be \
                     returned for it; read the ignore region with `ignore_mask()` where the encoding carries it \
                     in band, or through the `mask` annotation named by `header.ignore_mask`",
                    repr_str(&self.ann_id)
                ),
            ));
        }
        Ok(ids)
    }

    /// Resolved class entries; empty when the label set is unavailable.
    pub fn classes(&self) -> Vec<LabelClass> {
        self.resolved_entries(&self.header.class_ids)
    }

    /// Resolved entries of the classes that were looked for.
    pub fn annotated_classes(&self) -> Vec<LabelClass> {
        self.resolved_entries(&self.header.annotated_class_ids)
    }

    fn resolved_entries(&self, ids: &[i64]) -> Vec<LabelClass> {
        match &self.label_set {
            None => Vec::new(),
            Some(ls) => ids.iter().filter_map(|i| ls.by_id(*i).cloned()).collect(),
        }
    }

    /// The label-set key of a class id, or the id as text.
    pub fn class_key(&self, class_id: i64) -> String {
        self.label_set
            .as_ref()
            .and_then(|ls| ls.by_id(class_id))
            .map(|c| c.key.clone())
            .unwrap_or_else(|| class_id.to_string())
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!(
            "{}({}, kind={}, {} classes)",
            self.class_name(),
            repr_str(&self.ann_id),
            repr_str(self.kind()),
            self.class_ids().len()
        )
    }

    // -- datasets -------------------------------------------------------------

    /// A member dataset, or `None` when absent.
    pub fn optional_dataset(&self, name: &str) -> Option<hdf5::Dataset> {
        ops::child_dataset(&self.group, name)
    }

    /// A member dataset the kind requires (E410 when absent).
    pub fn dataset(&self, name: &str) -> Result<hdf5::Dataset> {
        self.optional_dataset(name).ok_or_else(|| {
            let id = repr_str(&self.ann_id);
            let message = match self.kind() {
                k if is_geometric_kind(k) => {
                    format!("annotation {id}: kind {} requires a {} dataset", repr_str(k), repr_str(name))
                }
                "labelmap" | "layers" | "bitmask" | "probmap" | "mask" if name == "data" => {
                    format!("annotation {id}: `{}` requires a `data` dataset", self.kind())
                }
                k => format!("annotation {id}: `{k}` requires a {} dataset", repr_str(name)),
            };
            Error::coded("E410", message)
        })
    }

    /// The `data` dataset of a voxel annotation.
    pub fn data(&self) -> Result<hdf5::Dataset> {
        self.dataset("data")
    }

    /// A whole numeric dataset cast to `T`.
    pub fn read_as<T: Element>(&self, name: &str) -> Result<ArrayD<T>> {
        Ok(data::read(&self.dataset(name)?)?.cast::<T>())
    }

    /// A whole numeric dataset cast to `T`, or `None` when absent.
    pub fn read_optional<T: Element>(&self, name: &str) -> Result<Option<ArrayD<T>>> {
        match self.optional_dataset(name) {
            None => Ok(None),
            Some(ds) => Ok(Some(data::read(&ds)?.cast::<T>())),
        }
    }

    /// A string dataset, or `None` when absent.
    pub fn read_strings_optional(&self, name: &str) -> Result<Option<Vec<String>>> {
        match self.optional_dataset(name) {
            None => Ok(None),
            Some(ds) => Ok(Some(data::read_strings(&ds)?)),
        }
    }

    /// The shape of a required dataset.
    pub fn dataset_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.dataset(name)?.shape())
    }

    // -- voxel kinds: the region of interest --------------------------------

    /// The spatial shape of the annotation's grid.
    pub fn spatial_shape(&self) -> Result<Vec<usize>> {
        Ok(self.grid()?.spatial_shape())
    }

    /// Resolve a region of interest against the grid; `None` is the whole grid.
    pub fn window(&self, roi: Option<&[Slice]>) -> Result<Window> {
        let shape = self.spatial_shape()?;
        let Some(roi) = roi else {
            return Ok(shape.iter().map(|n| (0, *n)).collect());
        };
        if roi.len() != shape.len() {
            return Err(Error::invalid(format!("roi has {} axes; grid has {} spatial axes", roi.len(), shape.len())));
        }
        roi.iter()
            .zip(&shape)
            .map(|(s, n)| {
                let (a, b, _) = Slice { step: None, ..*s }.resolve(*n)?;
                Ok((a, b))
            })
            .collect()
    }

    fn voxel_window(&self, voxel: &[i64]) -> Result<Window> {
        let shape = self.spatial_shape()?;
        if voxel.len() != shape.len() {
            return Err(Error::invalid(format!(
                "voxel has {} axes; grid has {} spatial axes",
                voxel.len(),
                shape.len()
            )));
        }
        voxel
            .iter()
            .zip(&shape)
            .map(|(v, n)| {
                if *v < 0 || *v >= *n as i64 {
                    Err(Error::Index(format!("voxel index {v} is out of bounds for an axis of {n} voxels")))
                } else {
                    Ok((*v as usize, *v as usize + 1))
                }
            })
            .collect()
    }

    fn require_voxel(&self, what: &str) -> Result<()> {
        if self.is_voxel() {
            Ok(())
        } else {
            Err(Error::Type(format!(
                "{what} is defined on voxel annotations; {} is a {} annotation",
                repr_str(&self.ann_id),
                repr_str(self.kind())
            )))
        }
    }

    fn require_kind(&self, kinds: &[&str], what: &str) -> Result<()> {
        if kinds.contains(&self.kind()) {
            Ok(())
        } else {
            Err(Error::Type(format!(
                "{what} is defined on {} annotations; {} is a {} annotation",
                kinds.iter().map(|k| format!("`{k}`")).collect::<Vec<_>>().join("/"),
                repr_str(&self.ann_id),
                repr_str(self.kind())
            )))
        }
    }

    // -- voxel kinds: the contract -------------------------------------------

    /// Boolean occupancy of one class over a window.
    pub fn dense_class(&self, class_id: i64, window: &Window) -> Result<ArrayD<bool>> {
        let shape = window_shape(window);
        match self.kind() {
            "labelmap" => Ok(equals(&data::read_region(&self.data()?, &window_index(None, window))?, class_id)),
            "layers" => match self.layer_table()?.layer_of.get(&class_id) {
                None => Ok(ArrayD::from_elem(IxDyn(&shape), false)),
                Some(layer) => {
                    Ok(equals(&data::read_region(&self.data()?, &window_index(Some(*layer), window))?, class_id))
                }
            },
            "bitmask" => match self.bit_table()?.position_of.get(&class_id) {
                None => Ok(ArrayD::from_elem(IxDyn(&shape), false)),
                Some(position) => {
                    let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
                    let block = data::read_region(&self.data()?, &window_index(Some(plane), window))?.cast::<u64>();
                    Ok(block.mapv(|w| (w >> bit) & 1 == 1))
                }
            },
            "probmap" => match self.class_ids().iter().position(|c| *c == class_id) {
                None => Ok(ArrayD::from_elem(IxDyn(&shape), false)),
                Some(position) => {
                    let block = data::read_region(&self.data()?, &window_index(Some(position), window))?;
                    Ok(contains_at(&block, self.threshold()?))
                }
            },
            "instances" => self.instances_dense_class(class_id, window),
            "mask" => Ok(data::read_region(&self.data()?, &window_index(None, window))?.nonzero_mask()),
            _ => {
                self.require_voxel("dense()")?;
                unreachable!()
            }
        }
    }

    /// `(C, *roi_shape)` boolean occupancy, one plane per requested class.
    ///
    /// `layers` and `bitmask` read each stored plane once rather than once per
    /// class; `mask` has no classes and answers with itself as one plane.
    pub fn dense(&self, classes: Option<&[ClassKey]>, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_voxel("dense()")?;
        if self.kind() == "mask" {
            if classes.is_some() {
                self.resolve_classes(classes)?;
            }
            return Ok(self.read_mask(roi)?.insert_axis(Axis(0)));
        }
        let ids = self.resolve_classes(classes)?;
        let window = self.window(roi)?;
        let mut shape = vec![ids.len()];
        shape.extend(window_shape(&window));
        let mut out = ArrayD::from_elem(IxDyn(&shape), false);
        match self.kind() {
            "layers" => {
                let table = self.layer_table()?;
                let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
                for (position, class_id) in ids.iter().enumerate() {
                    if let Some(layer) = table.layer_of.get(class_id) {
                        by_layer.entry(*layer).or_default().push(position);
                    }
                }
                let ds = self.data()?;
                for (layer, positions) in by_layer {
                    let block = data::read_region(&ds, &window_index(Some(layer), &window))?;
                    for position in positions {
                        out.index_axis_mut(Axis(0), position).assign(&equals(&block, ids[position]));
                    }
                }
            }
            "bitmask" => {
                let table = self.bit_table()?;
                let mut by_plane: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (position, class_id) in ids.iter().enumerate() {
                    if let Some(slot) = table.position_of.get(class_id) {
                        by_plane.entry(slot / BITS_PER_PLANE).or_default().push((position, slot % BITS_PER_PLANE));
                    }
                }
                let ds = self.data()?;
                for (plane, entries) in by_plane {
                    let block = data::read_region(&ds, &window_index(Some(plane), &window))?.cast::<u64>();
                    for (position, bit) in entries {
                        out.index_axis_mut(Axis(0), position).assign(&block.mapv(|w| (w >> bit) & 1 == 1));
                    }
                }
            }
            _ => {
                for (i, class_id) in ids.iter().enumerate() {
                    out.index_axis_mut(Axis(0), i).assign(&self.dense_class(*class_id, &window)?);
                }
            }
        }
        Ok(out)
    }

    /// The uniform predicate of §7.6: is `class` present at `voxel`?
    pub fn contains(&self, class: &ClassKey, voxel: &[i64]) -> Result<bool> {
        self.require_voxel("contains()")?;
        let class_id = self.resolve_class(class)?;
        let window = self.voxel_window(voxel)?;
        Ok(self.dense_class(class_id, &window)?.iter().next().copied().unwrap_or(false))
    }

    /// Flatten to one integer volume, breaking overlap ties explicitly.
    ///
    /// `priority` is ordered highest-precedence first; classes it omits are
    /// painted first, in `class_ids` order.  Returns the volume and, when no
    /// priority was given, how many voxels a later class overwrote --- the
    /// frontends warn on a nonzero count with [`Annotation::flatten_warning`].
    pub fn labelmap(
        &self,
        roi: Option<&[Slice]>,
        priority: Option<&[ClassKey]>,
        dtype: DType,
    ) -> Result<(NdArray, u64, Vec<i64>)> {
        self.require_voxel("labelmap()")?;
        let window = self.window(roi)?;
        if self.kind() == "labelmap" {
            let block = data::read_region(&self.data()?, &window_index(None, &window))?;
            return Ok((block.astype(dtype), 0, self.class_ids().to_vec()));
        }
        let shape = window_shape(&window);
        let mut out = ArrayD::<i64>::zeros(IxDyn(&shape));
        let mut ordered = self.class_ids().to_vec();
        if let Some(priority) = priority {
            let ranked = self.resolve_classes(Some(priority))?;
            let mut rest: Vec<i64> = ordered.iter().copied().filter(|c| !ranked.contains(c)).collect();
            rest.extend(ranked.iter().rev());
            ordered = rest;
        }
        let mut overwritten = 0u64;
        for class_id in &ordered {
            let mask = self.dense_class(*class_id, &window)?;
            ndarray::Zip::from(&mut out).and(&mask).for_each(|o, m| {
                if *m {
                    if priority.is_none() && *o != 0 {
                        overwritten += 1;
                    }
                    *o = *class_id;
                }
            });
        }
        Ok((NdArray::from(out).astype(dtype), overwritten, ordered))
    }

    /// The warning a lossy, unprioritised flatten raises.
    pub fn flatten_warning(&self, overwritten: u64, ordered: &[i64]) -> String {
        format!(
            "labelmap() flattened {overwritten} overlapping voxel(s) in {}: classes {} overlap and one integer \
             volume cannot hold both, so later classes overwrote earlier ones. Pass priority=[...] to choose \
             which class survives, or use dense()/contains() to keep the overlap.",
            repr_str(&self.ann_id),
            crate::json::repr_int_list(ordered)
        )
    }

    /// Foreground voxel count per class, over the whole grid.
    ///
    /// `labelmap` and `layers` answer every class from one pass of value
    /// counts, `bitmask` from one population count per plane; the other
    /// encodings decode per class.
    pub fn voxel_counts(&self, classes: Option<&[ClassKey]>) -> Result<IndexMap<i64, u64>> {
        self.require_voxel("voxel_counts()")?;
        let ids = self.resolve_classes(classes)?;
        if let Some(counted) = self.counts_from_planes()? {
            return Ok(ids.iter().map(|c| (*c, counted.get(c).copied().unwrap_or(0))).collect());
        }
        let window = self.window(None)?;
        ids.iter().map(|c| Ok((*c, self.dense_class(*c, &window)?.iter().filter(|v| **v).count() as u64))).collect()
    }

    fn counts_from_planes(&self) -> Result<Option<HashMap<i64, u64>>> {
        match self.kind() {
            "labelmap" | "layers" => {
                let Some(ceiling) = self.class_ids().iter().max() else {
                    return Ok(Some(HashMap::new()));
                };
                Ok(Some(value_counts(&self.data()?, *ceiling)?.into_iter().collect()))
            }
            "bitmask" => {
                let table = self.bit_table()?;
                if table.ids.is_empty() {
                    return Ok(Some(HashMap::new()));
                }
                let counts = popcounts(&self.data()?)?;
                let mut out = HashMap::new();
                for (position, class_id) in table.ids.iter().enumerate() {
                    let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
                    if plane < counts.len() {
                        out.insert(i64::from(*class_id), counts[plane][bit]);
                    }
                }
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    /// Tight `(S, 2)` index-space bounds per class; `None` when empty.
    pub fn class_bboxes(&self, classes: Option<&[ClassKey]>) -> Result<IndexMap<i64, Option<Array2<f32>>>> {
        self.require_voxel("class_bboxes()")?;
        let ids = self.resolve_classes(classes)?;
        let window = self.window(None)?;
        let mut out = IndexMap::new();
        for class_id in ids {
            let mask = self.dense_class(class_id, &window)?;
            let bounds = super::select::bbox(&mask);
            out.insert(
                class_id,
                bounds.map(|b| {
                    let slices: Vec<(i64, i64)> = b.iter().map(|(lo, hi)| (*lo as i64, *hi as i64)).collect();
                    let flat = slices_to_box(&slices);
                    Array2::from_shape_vec((slices.len(), 2), flat).expect("(S, 2)")
                }),
            );
        }
        Ok(out)
    }

    /// The in-band ignore region (§7.7) of a `labelmap` or `layers`
    /// annotation: a voxel is ignored where any layer marks it.
    pub fn ignore_mask(&self, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_kind(&["labelmap", "layers"], "ignore_mask()")?;
        let window = self.window(roi)?;
        let ignore = self.ignore_id();
        if self.kind() == "labelmap" {
            return Ok(equals(&data::read_region(&self.data()?, &window_index(None, &window))?, ignore));
        }
        let mut index = vec![Index::Slice(Slice::full())];
        index.extend(window_index(None, &window));
        let block = equals(&data::read_region(&self.data()?, &index)?, ignore);
        Ok(block.map_axis(Axis(0), |lane| lane.iter().any(|v| *v)))
    }

    /// The §7.7 ignore region over the whole grid, from the annotation alone:
    /// in band (`labelmap`, `layers`) and the sibling `mask` its
    /// `ignore_mask` names, read beside it in the same `annotations/` group.
    /// `None` where no voxel is ignored.
    pub fn ignore_region(&self) -> Result<Option<ArrayD<bool>>> {
        let mut region = None;
        if matches!(self.kind(), "labelmap" | "layers") {
            region = Some(self.ignore_mask(None)?);
        }
        if let Some(reference) = self.header.ignore_mask.clone() {
            let name = crate::sample::annotation_id(&reference);
            let path = self.group.name();
            let parent = path.rsplit_once('/').map_or("/", |(p, _)| p);
            let sibling = ops::child_group(&self.group.file()?.group(parent)?, name)
                .map(|g| Annotation::open(name, g, self.grids.clone(), self.label_set.clone()))
                .transpose()?
                .filter(|a| a.kind() == "mask")
                .ok_or_else(|| {
                    Error::coded(
                        "E413",
                        format!(
                            "{}: ignore_mask names {}, which is not a `mask` annotation in this file",
                            repr_str(&self.ann_id),
                            repr_str(name)
                        ),
                    )
                })?;
            let mask = sibling.read_mask(None)?;
            region = Some(match region {
                Some(mut inband) => {
                    inband.zip_mut_with(&mask, |r, m| *r |= *m);
                    inband
                }
                None => mask,
            });
        }
        Ok(region.filter(|r| r.iter().any(|v| *v)))
    }

    /// A `mask` annotation's volume over a window.
    pub fn read_mask(&self, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_kind(&["mask"], "read()")?;
        let window = self.window(roi)?;
        Ok(data::read_region(&self.data()?, &window_index(None, &window))?.nonzero_mask())
    }

    // -- layers -------------------------------------------------------------

    fn layer_table(&self) -> Result<&LayerTable> {
        cached(&self.layers, || {
            let ds = self.optional_dataset("layer_class_ids").ok_or_else(|| {
                Error::coded(
                    "E410",
                    format!("annotation {}: `layers` requires `layer_class_ids`", repr_str(&self.ann_id)),
                )
            })?;
            let raw = data::read(&ds)?.cast::<u16>();
            let table = if raw.ndim() == 2 {
                raw.into_dimensionality::<ndarray::Ix2>()?
            } else {
                let n = raw.len();
                raw.into_shape_with_order((1, n))?
            };
            let mut layer_of = IndexMap::new();
            for (layer, row) in table.outer_iter().enumerate() {
                for value in row {
                    let class_id = i64::from(*value);
                    if class_id == 0 {
                        continue;
                    }
                    if let Some(first) = layer_of.get(&class_id) {
                        return Err(Error::coded(
                            "E404",
                            format!(
                                "annotation {}: class {class_id} appears in layers {first} and {layer}",
                                repr_str(&self.ann_id)
                            ),
                        ));
                    }
                    layer_of.insert(class_id, layer);
                }
            }
            Ok(LayerTable { table, layer_of })
        })
    }

    /// `(L, K)` class ids per layer, zero-padded.
    pub fn layer_class_ids(&self) -> Result<Array2<u16>> {
        self.require_kind(&["layers"], "layer_class_ids")?;
        Ok(self.layer_table()?.table.clone())
    }

    /// The number of stored layers.
    pub fn n_layers(&self) -> Result<usize> {
        self.require_kind(&["layers"], "n_layers")?;
        Ok(self.data()?.shape().first().copied().unwrap_or(0))
    }

    /// Class id -> layer index.  Every class appears in exactly one layer.
    pub fn layer_of(&self) -> Result<IndexMap<i64, usize>> {
        self.require_kind(&["layers"], "layer_of")?;
        Ok(self.layer_table()?.layer_of.clone())
    }

    /// The classes of each layer, padding dropped.
    pub fn layer_classes(&self) -> Result<Vec<Vec<i64>>> {
        self.require_kind(&["layers"], "layer_classes()")?;
        Ok(self
            .layer_table()?
            .table
            .outer_iter()
            .map(|row| row.iter().filter(|v| **v != 0).map(|v| i64::from(*v)).collect())
            .collect())
    }

    /// One layer's labelmap, sliced in a single call (spec §14.5).
    pub fn read_layer(&self, layer: usize, roi: Option<&[Slice]>) -> Result<NdArray> {
        self.require_kind(&["layers"], "read_layer()")?;
        let window = self.window(roi)?;
        data::read_region(&self.data()?, &window_index(Some(layer), &window))
    }

    // -- bitmask ------------------------------------------------------------

    fn bit_table(&self) -> Result<&BitTable> {
        cached(&self.bits, || {
            let ds = self.optional_dataset("bit_class_ids").ok_or_else(|| {
                Error::coded(
                    "E410",
                    format!("annotation {}: `bitmask` requires `bit_class_ids`", repr_str(&self.ann_id)),
                )
            })?;
            let ids: Vec<u16> = data::read(&ds)?.cast::<u16>().iter().copied().collect();
            let position_of = ids.iter().enumerate().map(|(i, c)| (i64::from(*c), i)).collect();
            Ok(BitTable { ids, position_of })
        })
    }

    /// Bit position -> class id.
    pub fn bit_class_ids(&self) -> Result<Vec<u16>> {
        self.require_kind(&["bitmask"], "bit_class_ids")?;
        Ok(self.bit_table()?.ids.clone())
    }

    /// The number of `uint64` planes.
    pub fn n_planes(&self) -> Result<usize> {
        self.require_kind(&["bitmask"], "n_planes")?;
        Ok(self.data()?.shape().first().copied().unwrap_or(0))
    }

    /// Class id -> bit position.
    pub fn position_of(&self) -> Result<HashMap<i64, usize>> {
        self.require_kind(&["bitmask"], "position_of")?;
        Ok(self.bit_table()?.position_of.clone())
    }

    /// Every class present at one voxel, in O(P) reads.
    pub fn classes_at(&self, voxel: &[i64]) -> Result<Vec<i64>> {
        self.require_kind(&["bitmask"], "classes_at()")?;
        let window = self.voxel_window(voxel)?;
        let ids = &self.bit_table()?.ids;
        let ds = self.data()?;
        let mut out = Vec::new();
        for plane in 0..ds.shape().first().copied().unwrap_or(0) {
            let word = data::read_region(&ds, &window_index(Some(plane), &window))?
                .cast::<u64>()
                .iter()
                .next()
                .copied()
                .unwrap_or(0);
            if word == 0 {
                continue;
            }
            for bit in 0..BITS_PER_PLANE {
                if (word >> bit) & 1 == 1 {
                    let position = plane * BITS_PER_PLANE + bit;
                    if position < ids.len() {
                        out.push(i64::from(ids[position]));
                    }
                }
            }
        }
        Ok(out)
    }

    // -- probmap ------------------------------------------------------------

    /// Whether the probabilities sum to one per voxel (§7.5).
    pub fn normalized(&self) -> Result<bool> {
        Ok(attrs::get_bool(&self.group, "normalized")?.unwrap_or(false))
    }

    /// The declared decision threshold (§7.5), or the 0.5 default.
    pub fn threshold(&self) -> Result<f64> {
        Ok(attrs::get_f64(&self.group, "threshold")?.unwrap_or(DEFAULT_THRESHOLD))
    }

    /// `(C, *roi_shape)` float probabilities for the requested classes.
    pub fn probabilities(&self, classes: Option<&[ClassKey]>, roi: Option<&[Slice]>) -> Result<ArrayD<f32>> {
        self.require_kind(&["probmap"], "probabilities()")?;
        let ids = self.resolve_classes(classes)?;
        let window = self.window(roi)?;
        let mut shape = vec![ids.len()];
        shape.extend(window_shape(&window));
        let mut out = ArrayD::<f32>::zeros(IxDyn(&shape));
        let ds = self.data()?;
        for (i, class_id) in ids.iter().enumerate() {
            if let Some(position) = self.class_ids().iter().position(|c| c == class_id) {
                let block = data::read_region(&ds, &window_index(Some(position), &window))?.cast::<f32>();
                out.index_axis_mut(Axis(0), i).assign(&block);
            }
        }
        Ok(out)
    }

    // -- instances ----------------------------------------------------------

    fn instance_columns(&self) -> Result<&InstanceColumns> {
        cached(&self.columns, || {
            let has_masks = self.has_masks();
            Ok(InstanceColumns {
                boxes: self.read_as::<f32>("boxes")?,
                class_ids: self.read_as::<u16>("class_ids")?.iter().copied().collect(),
                instance_ids: self.read_as::<u64>("instance_ids")?.iter().copied().collect(),
                scores: self.read_optional::<f32>("scores")?.map(|a| a.iter().copied().collect()),
                mask_offsets: if has_masks {
                    Some(self.read_as::<i64>("mask_offsets")?.iter().map(|v| (*v).max(0) as u64).collect())
                } else {
                    None
                },
                mask_shapes: if has_masks { Some(self.read_as::<i64>("mask_shapes")?) } else { None },
            })
        })
    }

    /// Whether an `instances` annotation stores per-object masks.
    pub fn has_masks(&self) -> bool {
        self.optional_dataset("mask_data").is_some()
    }

    /// `(N, S, 2)` object boxes (`instances` and `boxes`).
    pub fn boxes(&self) -> Result<ArrayD<f32>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.boxes.clone()),
            "boxes" => self.read_as::<f32>("boxes"),
            _ => self.require_kind(&["instances", "boxes"], "boxes").map(|_| unreachable!()),
        }
    }

    /// Per-object class ids.
    pub fn object_class_ids(&self) -> Result<Vec<u16>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.class_ids.clone()),
            "points" => match self.read_optional::<u16>("class_ids")? {
                Some(a) => Ok(a.iter().copied().collect()),
                None => Ok(vec![0; self.dataset_shape("points")?.first().copied().unwrap_or(0)]),
            },
            "contours" => Ok(self.read_as::<u16>("contour_class_ids")?.iter().copied().collect()),
            "mesh" => match self.read_optional::<u16>("mesh_class_ids")? {
                Some(a) => Ok(a.iter().copied().collect()),
                None => Ok(self.class_ids().iter().map(|c| *c as u16).collect()),
            },
            k if is_geometric_kind(k) => Ok(self.read_as::<u16>("class_ids")?.iter().copied().collect()),
            _ => self
                .require_kind(
                    &["instances", "boxes", "obb", "keypoints", "points", "contours", "mesh"],
                    "object_class_ids",
                )
                .map(|_| unreachable!()),
        }
    }

    /// Object ids, widened to `uint64` and never narrowed (§7.4, §8.2).
    pub fn instance_ids(&self) -> Result<Option<Vec<u64>>> {
        match self.kind() {
            "instances" => Ok(Some(self.instance_columns()?.instance_ids.clone())),
            k if is_geometric_kind(k) => {
                Ok(self.read_optional::<u64>("instance_ids")?.map(|a| a.iter().copied().collect()))
            }
            _ => self.require_kind(&["instances"], "instance_ids").map(|_| unreachable!()),
        }
    }

    /// Per-object scores, when stored.
    pub fn scores(&self) -> Result<Option<Vec<f32>>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.scores.clone()),
            k if is_geometric_kind(k) => Ok(self.read_optional::<f32>("scores")?.map(|a| a.iter().copied().collect())),
            _ => self.require_kind(&["instances"], "scores").map(|_| unreachable!()),
        }
    }

    /// The number of objects of an `instances` annotation.
    pub fn n_objects(&self) -> Result<usize> {
        self.require_kind(&["instances"], "n_objects")?;
        Ok(self.instance_columns()?.boxes.shape().first().copied().unwrap_or(0))
    }

    /// Decode one object's bbox-local mask.
    pub fn crop(&self, index: usize) -> Result<Option<ArrayD<bool>>> {
        self.require_kind(&["instances"], "crop()")?;
        if !self.has_masks() {
            return Ok(None);
        }
        let columns = self.instance_columns()?;
        let offsets = columns.mask_offsets.as_ref().expect("masks present");
        let shapes = columns.mask_shapes.as_ref().expect("masks present");
        if index + 1 >= offsets.len() {
            return Err(Error::Index(format!(
                "index {index} is out of bounds for {} objects",
                offsets.len().saturating_sub(1)
            )));
        }
        let (start, stop) = (offsets[index] as i64, offsets[index + 1] as i64);
        let packed =
            data::read_region(&self.dataset("mask_data")?, &[Index::Slice(Slice::new(start, stop))])?.cast::<u8>();
        let packed: Vec<u8> = packed.iter().copied().collect();
        let local_offsets = [0u64, packed.len() as u64];
        let shape = shapes.index_axis(Axis(0), index).to_owned().insert_axis(Axis(0));
        Ok(Some(decode_crop(&packed, &local_offsets, &shape.into_dyn(), 0)?))
    }

    /// Every object, in stored order (`instances`, `boxes`).
    pub fn instances(&self) -> Result<Vec<Instance>> {
        match self.kind() {
            "instances" => {
                let columns = self.instance_columns()?;
                let n = columns.boxes.shape().first().copied().unwrap_or(0);
                (0..n)
                    .map(|i| {
                        Ok(Instance {
                            index: i,
                            instance_id: columns.instance_ids[i],
                            class_id: i64::from(columns.class_ids[i]),
                            bbox: box_at(&columns.boxes, i)?,
                            mask: self.crop(i)?,
                            score: columns
                                .scores
                                .as_ref()
                                .and_then(|s| s.get(i))
                                .filter(|v| v.is_finite())
                                .map(|v| f64::from(*v)),
                        })
                    })
                    .collect()
            }
            "boxes" => {
                let boxes = self.boxes()?;
                let classes = self.object_class_ids()?;
                let ids = self.instance_ids()?;
                let scores = self.scores()?;
                let n = boxes.shape().first().copied().unwrap_or(0);
                (0..n)
                    .map(|i| {
                        Ok(Instance {
                            index: i,
                            instance_id: ids.as_ref().map(|v| v[i]).unwrap_or(i as u64),
                            class_id: i64::from(classes[i]),
                            bbox: box_at(&boxes, i)?,
                            mask: None,
                            score: scores.as_ref().map(|s| f64::from(s[i])),
                        })
                    })
                    .collect()
            }
            k if is_voxel_kind(k) => Err(Error::invalid(format!(
                "annotation {} of kind {} does not carry instance identity, and it cannot be recovered from a \
                 dense encoding: transcoding to `instances` would merge every object of a class into one and mint \
                 an id that belongs to none of them (§7.4). Re-derive the objects from whatever source had them.",
                repr_str(&self.ann_id),
                repr_str(k)
            ))),
            _ => self.require_kind(&["instances", "boxes"], "instances()").map(|_| unreachable!()),
        }
    }

    /// One object by its id (`KeyError` when absent).
    pub fn instance(&self, instance_id: u64) -> Result<Instance> {
        self.require_kind(&["instances"], "instance()")?;
        self.instances()?
            .into_iter()
            .find(|o| o.instance_id == instance_id)
            .ok_or_else(|| Error::Key(format!("annotation {} has no instance {instance_id}", repr_str(&self.ann_id))))
    }

    /// `instance_id -> class_id`: the join key for longitudinal tracking.
    pub fn tracking(&self) -> Result<IndexMap<u64, i64>> {
        self.require_kind(&["instances"], "tracking()")?;
        let columns = self.instance_columns()?;
        Ok(columns.instance_ids.iter().zip(&columns.class_ids).map(|(i, c)| (*i, i64::from(*c))).collect())
    }

    fn instances_dense_class(&self, class_id: i64, window: &Window) -> Result<ArrayD<bool>> {
        let shape = window_shape(window);
        let mut out = ArrayD::from_elem(IxDyn(&shape), false);
        let columns = self.instance_columns()?;
        for (index, object_class) in columns.class_ids.iter().enumerate() {
            if i64::from(*object_class) != class_id {
                continue;
            }
            // Unclipped, deliberately: these slices are the frame the stored
            // crop was cut in; intersecting with the window does the clipping.
            let bbox: Vec<f64> = box_at(&columns.boxes, index)?.iter().map(|v| f64::from(*v)).collect();
            let obj = box_to_slices(&bbox, None)?;
            let mut local = Vec::with_capacity(obj.len());
            let mut target = Vec::with_capacity(obj.len());
            let mut empty = false;
            for ((want_lo, want_hi), (have_lo, have_hi)) in window.iter().zip(&obj) {
                let (want_lo, want_hi) = (*want_lo as i64, *want_hi as i64);
                let lo = want_lo.max(*have_lo);
                let hi = want_hi.min(*have_hi);
                if hi <= lo {
                    empty = true;
                    break;
                }
                local.push(((lo - have_lo) as usize, (hi - have_lo) as usize));
                target.push(((lo - want_lo) as usize, (hi - want_lo) as usize));
            }
            if empty {
                continue;
            }
            let crop = self.crop(index)?;
            let mut view = out.view_mut();
            for (axis, (a, b)) in target.iter().enumerate() {
                view.slice_axis_inplace(Axis(axis), ndarray::Slice::from(*a..*b));
            }
            match crop {
                None => view.fill(true),
                Some(crop) => {
                    let mut src = crop.view();
                    for (axis, (a, b)) in local.iter().enumerate() {
                        let n = src.shape()[axis];
                        src.slice_axis_inplace(Axis(axis), ndarray::Slice::from((*a).min(n)..(*b).min(n)));
                    }
                    if src.shape() == view.shape() {
                        ndarray::Zip::from(&mut view).and(&src).for_each(|o, s| *o |= *s);
                    } else {
                        return Err(Error::coded(
                            "E405",
                            format!(
                                "annotation {}: object {index}'s stored mask does not cover its box",
                                repr_str(&self.ann_id)
                            ),
                        ));
                    }
                }
            }
        }
        Ok(out)
    }

    // -- summaries ------------------------------------------------------------

    /// JSON-safe description for `medh5 info`.
    pub fn summary(&self) -> Result<Value> {
        if self.is_voxel() {
            return self.voxel_summary();
        }
        if self.is_geometric() {
            return self.geometric_summary();
        }
        self.classification_summary()
    }

    fn voxel_summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.ann_id));
        out.insert("kind".into(), json!(self.kind()));
        out.insert("task".into(), json!(self.task()));
        out.insert("grid".into(), json!(self.grid_id()));
        out.insert("timepoints".into(), json!(self.timepoints()));
        out.insert("classes".into(), json!(self.class_ids().len()));
        out.insert("annotated_classes".into(), json!(self.annotated_class_ids().len()));
        out.insert("fully_covered".into(), json!(self.is_fully_covered()));
        out.insert("closure".into(), json!(self.closure()));
        out.insert("quality".into(), json!(self.quality_key()));
        out.insert("prov".into(), json!(self.prov()));
        match self.kind() {
            "mask" => {
                out.insert("true_voxels".into(), json!(count_nonzero(&self.data()?)?));
            }
            "layers" => {
                out.insert("layers".into(), json!(self.n_layers()?));
            }
            "bitmask" => {
                out.insert("planes".into(), json!(self.n_planes()?));
            }
            "probmap" => {
                out.insert("normalized".into(), json!(self.normalized()?));
                out.insert("threshold".into(), crate::json::num(self.threshold()?));
            }
            "instances" => {
                out.insert("objects".into(), json!(self.n_objects()?));
                out.insert("has_masks".into(), json!(self.has_masks()));
                out.insert("instance_ids".into(), json!(self.instance_columns()?.instance_ids));
            }
            _ => {}
        }
        Ok(Value::Object(out))
    }
}

/// Row `i` of an `(N, S, 2)` box array as an `(S, 2)` array.
pub fn box_at(boxes: &ArrayD<f32>, i: usize) -> Result<Array2<f32>> {
    let row = boxes.index_axis(Axis(0), i).to_owned();
    let s = row.len() / 2;
    Ok(row.into_shape_with_order((s, 2))?)
}
