//! The sampling index: what a patch sampler would otherwise recompute (§14.3).
//!
//! A cached subsample of foreground coordinates answers "where is class c?"
//! in O(1) in volume size.  Every entry carries `source_digest`, so a stale
//! entry is detectable rather than silently wrong (§13.3).

use std::collections::HashMap;
use std::sync::OnceLock;

use indexmap::IndexMap;
use ndarray::{Array2, Array3, ArrayD, Axis, Dimension, IxDyn};
use serde_json::{json, Map, Value};

use super::codecs::{dataset_layout, CodecProfile, Role};
use crate::annotations::Annotation;
use crate::array::{Index, NdArray, Slice};
use crate::geometry::affine::slices_to_box;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::repr_str;
use crate::labels::ClassKey;
use crate::rng::{Rng, SeededRng};
use crate::{Error, Result};

/// Coordinates cached per class.
pub const DEFAULT_MAX_COORDS: usize = 4096;
/// Block edge of the occupancy map.
pub const DEFAULT_OCCUPANCY_FACTOR: usize = 8;

/// The datasets of one annotation's index entry.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexPayload {
    pub ann_id: String,
    pub class_ids: Vec<u16>,
    pub voxel_counts: Vec<i64>,
    /// `(C, S, 2)`, NaN for an empty class.
    pub class_bboxes: Array3<f32>,
    pub fg_coords: IndexMap<i64, Array2<i32>>,
    pub occupancy: Option<ArrayD<bool>>,
    pub source_digest: Option<String>,
    pub max_coords: usize,
    pub seed: u64,
    pub total_foreground: i64,
}

/// Uniformly subsample foreground voxel coordinates.
pub fn sample_foreground_coords(mask: &ArrayD<bool>, max_coords: usize, rng: &mut dyn Rng) -> Result<Array2<i32>> {
    let ndim = mask.ndim();
    let total = mask.iter().filter(|v| **v).count();
    if total == 0 {
        return Ok(Array2::zeros((0, ndim)));
    }
    let shape = mask.shape().to_vec();
    let unravel = |flat: usize| -> Vec<i32> {
        let mut rest = flat;
        let mut out = vec![0i32; ndim];
        for axis in (0..ndim).rev() {
            out[axis] = (rest % shape[axis]) as i32;
            rest /= shape[axis];
        }
        out
    };
    let picks = if total <= max_coords {
        None
    } else {
        let mut p = rng.sample_without_replacement(total, max_coords)?;
        p.sort_unstable();
        Some(p)
    };
    let mut chosen = Vec::with_capacity(total.min(max_coords));
    let mut ordinal = 0usize;
    let mut next = 0usize;
    for (flat, v) in mask.iter().enumerate() {
        if !*v {
            continue;
        }
        match &picks {
            None => chosen.push(flat),
            Some(p) => {
                if next < p.len() && p[next] == ordinal {
                    chosen.push(flat);
                    next += 1;
                }
            }
        }
        ordinal += 1;
    }
    let flat: Vec<i32> = chosen.into_iter().flat_map(unravel).collect();
    Ok(Array2::from_shape_vec((flat.len() / ndim.max(1), ndim), flat)?)
}

/// Low-resolution "is there anything in this block" map, for rejection sampling.
pub fn occupancy(mask: &ArrayD<bool>, factor: usize) -> ArrayD<bool> {
    let coarse: Vec<usize> = mask.shape().iter().map(|n| n.div_ceil(factor).max(1)).collect();
    let mut out = ArrayD::from_elem(IxDyn(&coarse), false);
    for (idx, v) in mask.indexed_iter() {
        if *v {
            let block: Vec<usize> = idx.slice().iter().map(|i| i / factor).collect();
            out[IxDyn(&block)] = true;
        }
    }
    out
}

/// Compute the sampling index for one voxel annotation.
pub fn build_index(
    annotation: &Annotation,
    classes: Option<&[ClassKey]>,
    max_coords: usize,
    occupancy_factor: Option<usize>,
    seed: u64,
    source_digest: Option<String>,
) -> Result<IndexPayload> {
    let ids = annotation.resolve_classes(classes)?;
    let mut rng = SeededRng::new(seed);
    let window = annotation.window(None)?;
    let n_spatial = annotation.spatial_shape()?.len();
    let mut counts = vec![0i64; ids.len()];
    let mut bboxes = Array3::<f32>::zeros((ids.len(), n_spatial, 2));
    let mut coords = IndexMap::new();
    let mut planes = Vec::new();
    for (i, class_id) in ids.iter().enumerate() {
        let mask = annotation.dense_class(*class_id, &window)?;
        counts[i] = mask.iter().filter(|v| **v).count() as i64;
        match crate::annotations::select::bbox(&mask) {
            Some(b) => {
                let slices: Vec<(i64, i64)> = b.iter().map(|(lo, hi)| (*lo as i64, *hi as i64)).collect();
                let flat = slices_to_box(&slices);
                for (k, v) in flat.into_iter().enumerate() {
                    bboxes[[i, k / 2, k % 2]] = v;
                }
            }
            None => bboxes.index_axis_mut(Axis(0), i).fill(f32::NAN),
        }
        coords.insert(*class_id, sample_foreground_coords(&mask, max_coords, &mut rng)?);
        if let Some(factor) = occupancy_factor.filter(|f| *f > 0) {
            planes.push(occupancy(&mask, factor));
        }
    }
    let occupancy = if planes.is_empty() {
        None
    } else {
        let views: Vec<_> = planes.iter().map(|p| p.view()).collect();
        Some(ndarray::stack(Axis(0), &views)?)
    };
    Ok(IndexPayload {
        ann_id: annotation.ann_id.clone(),
        class_ids: ids.iter().map(|c| *c as u16).collect(),
        total_foreground: counts.iter().sum(),
        voxel_counts: counts,
        class_bboxes: bboxes,
        fg_coords: coords,
        occupancy,
        source_digest,
        max_coords,
        seed,
    })
}

/// Write an index entry under `index/<ann_id>`, through the label codec.
pub fn write_index(root: &hdf5::Group, payload: &IndexPayload, codec: &CodecProfile) -> Result<hdf5::Group> {
    let index_root = match ops::child_group(root, "index") {
        Some(g) => g,
        None => root.create_group("index")?,
    };
    if ops::exists(&index_root, &payload.ann_id) {
        ops::unlink(&index_root, &payload.ann_id)?;
    }
    let group = index_root.create_group(&payload.ann_id)?;
    let store = |parent: &hdf5::Group, name: &str, array: NdArray, chunks: Option<Vec<usize>>| -> Result<()> {
        let layout = dataset_layout(&array.shape(), array.dtype().itemsize(), codec, Role::Label, chunks);
        data::create(parent, name, &array, &layout)?;
        Ok(())
    };
    store(
        &group,
        "class_ids",
        NdArray::from(ArrayD::from_shape_vec(IxDyn(&[payload.class_ids.len()]), payload.class_ids.clone())?),
        None,
    )?;
    store(
        &group,
        "voxel_counts",
        NdArray::from(ArrayD::from_shape_vec(IxDyn(&[payload.voxel_counts.len()]), payload.voxel_counts.clone())?),
        None,
    )?;
    store(&group, "class_bboxes", NdArray::from(payload.class_bboxes.clone().into_dyn()), None)?;
    let coords = group.create_group("fg_coords")?;
    for (class_id, arr) in &payload.fg_coords {
        store(&coords, &class_id.to_string(), NdArray::from(arr.clone().into_dyn()), None)?;
    }
    if let Some(occ) = &payload.occupancy {
        let mut chunks = vec![1];
        chunks.extend(&occ.shape()[1..]);
        store(&group, "occupancy", NdArray::from(occ.clone()), Some(chunks))?;
    }
    if let Some(d) = &payload.source_digest {
        attrs::write(&group, "source_digest", &AttrValue::Str(d.clone()))?;
    }
    attrs::write(&group, "max_coords", &AttrValue::Int(payload.max_coords as i64))?;
    attrs::write(&group, "seed", &AttrValue::Int(payload.seed as i64))?;
    Ok(group)
}

/// The index's class ids, and each id's row.
type ClassTable = (Vec<i64>, HashMap<i64, usize>);

/// Reader for one `index/<ann_id>` entry.  Small tables are read once.
#[derive(Debug)]
pub struct SamplingIndex {
    pub ann_id: String,
    pub group: hdf5::Group,
    class_ids: OnceLock<Result<ClassTable>>,
    counts: OnceLock<Result<IndexMap<i64, i64>>>,
}

impl SamplingIndex {
    pub fn new(ann_id: &str, group: hdf5::Group) -> SamplingIndex {
        SamplingIndex { ann_id: ann_id.into(), group, class_ids: OnceLock::new(), counts: OnceLock::new() }
    }

    fn table(&self) -> Result<&(Vec<i64>, HashMap<i64, usize>)> {
        match self.class_ids.get_or_init(|| {
            let ds = self.group.dataset("class_ids")?;
            let ids: Vec<i64> = data::read(&ds)?.cast::<i64>().iter().copied().collect();
            let positions = ids.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            Ok((ids, positions))
        }) {
            Ok(t) => Ok(t),
            Err(e) => Err(e.clone()),
        }
    }

    /// The indexed classes, in stored order.
    pub fn class_ids(&self) -> Result<Vec<i64>> {
        Ok(self.table()?.0.clone())
    }

    /// Whether a class is indexed.
    pub fn has_class(&self, class_id: i64) -> Result<bool> {
        Ok(self.table()?.1.contains_key(&class_id))
    }

    /// The digest of the annotation this entry was built from.
    pub fn source_digest(&self) -> Result<Option<String>> {
        attrs::get_str(&self.group, "source_digest")
    }

    /// Coordinates cached per class.
    pub fn max_coords(&self) -> Result<i64> {
        Ok(attrs::get_i64(&self.group, "max_coords")?.unwrap_or(DEFAULT_MAX_COORDS as i64))
    }

    /// Voxel count per class.
    pub fn voxel_counts(&self) -> Result<IndexMap<i64, i64>> {
        match self.counts.get_or_init(|| {
            let ids = self.class_ids()?;
            let counts: Vec<i64> =
                data::read(&self.group.dataset("voxel_counts")?)?.cast::<i64>().iter().copied().collect();
            Ok(ids.into_iter().zip(counts).collect())
        }) {
            Ok(c) => Ok(c.clone()),
            Err(e) => Err(e.clone()),
        }
    }

    /// A class's `(S, 2)` bounding box, `None` when empty.
    pub fn bbox(&self, class_id: i64) -> Result<Option<Array2<f32>>> {
        let Some(position) = self.table()?.1.get(&class_id).copied() else {
            return Err(Error::Value(format!("index {} has no class {class_id}", repr_str(&self.ann_id))));
        };
        let row = data::read_region(&self.group.dataset("class_bboxes")?, &[Index::At(position as i64)])?.cast::<f32>();
        if row.iter().any(|v| v.is_nan()) {
            return Ok(None);
        }
        let s = row.len() / 2;
        Ok(Some(row.into_shape_with_order((s, 2))?))
    }

    fn coord_node(&self, class_id: i64) -> Result<hdf5::Dataset> {
        let group = self.group.group("fg_coords")?;
        ops::child_dataset(&group, &class_id.to_string()).ok_or_else(|| {
            Error::Key(format!("index {} has no coordinates for class {class_id}", repr_str(&self.ann_id)))
        })
    }

    /// Every cached coordinate of a class, `(N, S)`.
    pub fn coords(&self, class_id: i64) -> Result<Array2<i32>> {
        let a = data::read(&self.coord_node(class_id)?)?.cast::<i32>();
        let n = a.shape().first().copied().unwrap_or(0);
        let s = a.shape().get(1).copied().unwrap_or(0);
        Ok(a.into_shape_with_order((n, s))?)
    }

    /// Draw `n` foreground voxel coordinates in O(1) time and memory.
    pub fn sample_foreground(&self, class_id: i64, n: usize, rng: &mut dyn Rng) -> Result<Array2<i32>> {
        let pool = self.coord_node(class_id)?;
        let size = pool.shape().first().copied().unwrap_or(0);
        if size == 0 {
            return Err(Error::invalid(format!(
                "index {}: class {class_id} has no foreground voxels",
                repr_str(&self.ann_id)
            )));
        }
        let picks = rng.integers(0, size as i64, n)?;
        if n == 1 {
            let row = data::read_region(&pool, &[Index::At(picks[0])])?.cast::<i32>();
            let s = row.len();
            return Ok(row.into_shape_with_order((1, s))?);
        }
        let all = self.coords(class_id)?;
        let s = all.ncols();
        let mut out = Array2::<i32>::zeros((n, s));
        for (i, p) in picks.iter().enumerate() {
            out.row_mut(i).assign(&all.row(*p as usize));
        }
        Ok(out)
    }

    /// Sampling weights derived from `voxel_counts`.
    pub fn class_weights(&self, mode: &str) -> Result<IndexMap<i64, f64>> {
        let counts = self.voxel_counts()?;
        match mode {
            "uniform" => Ok(counts.keys().map(|c| (*c, 1.0)).collect()),
            "inverse_frequency" => {
                let weights: IndexMap<i64, f64> =
                    counts.iter().map(|(c, n)| (*c, if *n != 0 { 1.0 / *n as f64 } else { 0.0 })).collect();
                let total: f64 = weights.values().sum();
                Ok(if total > 0.0 {
                    weights.into_iter().map(|(c, w)| (c, w / total)).collect()
                } else {
                    counts.keys().map(|c| (*c, 0.0)).collect()
                })
            }
            other => Err(Error::invalid(format!("unknown weighting mode {}", repr_str(other)))),
        }
    }

    /// Whether this entry was built from the annotation with this digest.
    pub fn is_current(&self, annotation_digest: &str) -> Result<bool> {
        Ok(self.source_digest()?.as_deref() == Some(annotation_digest))
    }

    /// Whether an occupancy map is stored.
    pub fn has_occupancy(&self) -> bool {
        ops::exists(&self.group, "occupancy")
    }

    /// Read one class's occupancy plane.
    pub fn occupancy_plane(&self, position: usize) -> Result<Option<ArrayD<bool>>> {
        match ops::child_dataset(&self.group, "occupancy") {
            None => Ok(None),
            Some(ds) => Ok(Some(
                data::read_region(&ds, &[Index::At(position as i64), Index::Slice(Slice::full())])?.nonzero_mask(),
            )),
        }
    }

    pub fn summary(&self) -> Result<Value> {
        let counts: Map<String, Value> = self.voxel_counts()?.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        Ok(json!({
            "id": self.ann_id,
            "classes": self.class_ids()?,
            "voxel_counts": counts,
            "max_coords": self.max_coords()?,
            "has_occupancy": self.has_occupancy(),
            "source_digest": self.source_digest()?,
        }))
    }
}

/// Every index entry under a sample root, by annotation id.
pub fn read_indices(root: &hdf5::Group) -> Result<IndexMap<String, SamplingIndex>> {
    let Some(node) = ops::child_group(root, "index") else {
        return Ok(IndexMap::new());
    };
    let mut out = IndexMap::new();
    for name in ops::members(&node)? {
        if let Some(g) = ops::child_group(&node, &name) {
            out.insert(name.clone(), SamplingIndex::new(&name, g));
        }
    }
    Ok(out)
}
