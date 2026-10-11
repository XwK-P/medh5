//! Overlap analysis, graph colouring and encoding selection (spec §7.6).
//!
//! A voxel may belong to several classes at once, and there may be hundreds of
//! classes.  Colouring the class **overlap graph** and storing one integer
//! volume per colour makes the number of volumes track the *overlap depth*
//! rather than the class count: on the 200-structure phantom, mean degree 3.4
//! coloured into 5 layers.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use super::payload::{normalize_masks, Masks};
use crate::Result;

/// A class is 'localized' when its bounding box covers at most this fraction.
pub const LOCALIZED_BBOX_FRACTION: f64 = 0.05;
/// Below this mean fill an `instances` encoding beats any dense one.
pub const SPARSE_FILL: f64 = 1e-3;

/// Measured properties of a set of class masks, and what they imply.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapStats {
    pub class_ids: Vec<i64>,
    pub spatial_shape: Vec<usize>,
    pub counts: BTreeMap<i64, u64>,
    pub edges: BTreeSet<(i64, i64)>,
    pub colouring: BTreeMap<i64, usize>,
    pub localized: BTreeSet<i64>,
    pub n_labelled_voxels: u64,
}

impl OverlapStats {
    pub fn n_classes(&self) -> usize {
        self.class_ids.len()
    }

    pub fn n_voxels(&self) -> u64 {
        self.spatial_shape.iter().map(|v| *v as u64).product()
    }

    pub fn total_foreground(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Mean per-class density.
    pub fn fill(&self) -> f64 {
        if self.class_ids.is_empty() || self.n_voxels() == 0 {
            return 0.0;
        }
        self.total_foreground() as f64 / (self.n_classes() as f64 * self.n_voxels() as f64)
    }

    /// Mean number of labels on a labelled voxel.
    pub fn depth(&self) -> f64 {
        if self.n_labelled_voxels == 0 {
            return 0.0;
        }
        self.total_foreground() as f64 / self.n_labelled_voxels as f64
    }

    pub fn n_layers(&self) -> usize {
        self.colouring.values().max().map(|m| m + 1).unwrap_or(0)
    }

    pub fn mean_degree(&self) -> f64 {
        if self.class_ids.is_empty() {
            return 0.0;
        }
        2.0 * self.edges.len() as f64 / self.class_ids.len() as f64
    }

    pub fn is_edgeless(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn n_planes(&self) -> usize {
        self.n_classes().div_ceil(64)
    }

    pub fn summary(&self) -> Value {
        let counts: serde_json::Map<String, Value> =
            self.counts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        json!({
            "classes": self.n_classes(),
            "voxels": self.n_voxels(),
            "fill": self.fill(),
            "depth": self.depth(),
            "layers": self.n_layers(),
            "planes": self.n_planes(),
            "edges": self.edges.len(),
            "mean_degree": self.mean_degree(),
            "localized": self.localized.iter().collect::<Vec<_>>(),
            "counts": counts,
        })
    }
}

/// Bounding-box volume of a mask (0 for an empty mask).
pub fn bbox_volume(mask: &ndarray::ArrayD<bool>) -> u64 {
    match bbox(mask) {
        Some(bounds) => bounds.iter().map(|(lo, hi)| (hi - lo) as u64).product(),
        None => 0,
    }
}

/// Tight `(start, stop)` per axis of a mask, or `None` when empty.
pub fn bbox(mask: &ndarray::ArrayD<bool>) -> Option<Vec<(usize, usize)>> {
    let ndim = mask.ndim();
    let mut lo = vec![usize::MAX; ndim];
    let mut hi = vec![0usize; ndim];
    let mut any = false;
    for (idx, v) in mask.indexed_iter() {
        if *v {
            any = true;
            for axis in 0..ndim {
                lo[axis] = lo[axis].min(idx[axis]);
                hi[axis] = hi[axis].max(idx[axis] + 1);
            }
        }
    }
    any.then(|| lo.into_iter().zip(hi).collect())
}

/// Measure counts, the overlap graph, a greedy colouring and localization.
///
/// Pairwise overlap is computed only on voxels carrying more than one label,
/// which keeps the quadratic part small.
pub fn analyse(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<OverlapStats> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let n_voxels: usize = shape.iter().product();
    let mut counts = BTreeMap::new();
    let mut depth = vec![0u16; n_voxels];
    for (cid, mask) in masks {
        let mut count = 0u64;
        for (i, v) in mask.iter().enumerate() {
            if *v {
                count += 1;
                depth[i] += 1;
            }
        }
        counts.insert(*cid, count);
    }
    let labelled = depth.iter().filter(|d| **d > 0).count() as u64;
    let multi: Vec<usize> = depth.iter().enumerate().filter(|(_, d)| **d > 1).map(|(i, _)| i).collect();
    let mut edges = BTreeSet::new();
    if !multi.is_empty() {
        let picked: Vec<(i64, Vec<bool>)> = masks
            .iter()
            .map(|(cid, mask)| {
                let flat: Vec<bool> = mask.iter().copied().collect();
                (*cid, multi.iter().map(|i| flat[*i]).collect())
            })
            .filter(|(_, p): &(i64, Vec<bool>)| p.iter().any(|v| *v))
            .collect();
        for i in 0..picked.len() {
            for j in i + 1..picked.len() {
                let (a, pa) = &picked[i];
                let (b, pb) = &picked[j];
                if pa.iter().zip(pb).any(|(x, y)| *x && *y) {
                    edges.insert(if a < b { (*a, *b) } else { (*b, *a) });
                }
            }
        }
    }
    let mut localized = BTreeSet::new();
    for (cid, mask) in masks {
        if counts[cid] == 0 {
            continue;
        }
        if bbox_volume(mask) as f64 <= LOCALIZED_BBOX_FRACTION * n_voxels as f64 {
            localized.insert(*cid);
        }
    }
    let colouring = greedy_colour(&class_ids, &edges);
    Ok(OverlapStats {
        class_ids,
        spatial_shape: shape,
        counts,
        edges,
        colouring,
        localized,
        n_labelled_voxels: labelled,
    })
}

/// Colour the overlap graph greedily, highest degree first (spec §7.6).
pub fn greedy_colour(class_ids: &[i64], edges: &BTreeSet<(i64, i64)>) -> BTreeMap<i64, usize> {
    let mut neighbours: BTreeMap<i64, BTreeSet<i64>> = class_ids.iter().map(|c| (*c, BTreeSet::new())).collect();
    for (a, b) in edges {
        if neighbours.contains_key(a) && neighbours.contains_key(b) {
            neighbours.get_mut(a).unwrap().insert(*b);
            neighbours.get_mut(b).unwrap().insert(*a);
        }
    }
    let mut order: Vec<i64> = class_ids.to_vec();
    order.sort_by_key(|c| (std::cmp::Reverse(neighbours[c].len()), *c));
    let mut colour: BTreeMap<i64, usize> = BTreeMap::new();
    for cid in order {
        let taken: BTreeSet<usize> = neighbours[&cid].iter().filter_map(|n| colour.get(n).copied()).collect();
        let mut chosen = 0;
        while taken.contains(&chosen) {
            chosen += 1;
        }
        colour.insert(cid, chosen);
    }
    colour
}

/// Group a colouring into per-layer class-id lists, ascending.
pub fn layers_from_colouring(colouring: &BTreeMap<i64, usize>) -> Vec<Vec<i64>> {
    let Some(max) = colouring.values().max() else { return Vec::new() };
    let mut buckets = vec![Vec::new(); max + 1];
    for (cid, layer) in colouring {
        buckets[*layer].push(*cid);
    }
    buckets
}

/// 1 for `uint8` when the ids fit, else 2 for `uint16` (spec §7.1).
pub fn label_dtype_size(class_ids: &[i64], ignore: bool) -> usize {
    let ceiling = class_ids.iter().copied().max().unwrap_or(0);
    if ceiling <= 254 && !ignore {
        1
    } else {
        2
    }
}

/// Raw (pre-compression) bytes per encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct CostModel {
    pub labelmap: Option<u64>,
    pub layers: u64,
    pub bitmask: u64,
    pub instances: u64,
    pub probmap: u64,
    pub detail: Value,
}

impl CostModel {
    /// The cheapest encoding (`labelmap` only when it can hold the masks).
    pub fn best(&self) -> &'static str {
        let mut options: Vec<(&'static str, u64)> =
            vec![("layers", self.layers), ("bitmask", self.bitmask), ("instances", self.instances)];
        if let Some(l) = self.labelmap {
            options.push(("labelmap", l));
        }
        let mut best = options[0];
        for o in &options[1..] {
            if o.1 < best.1 {
                best = *o;
            }
        }
        best.0
    }

    pub fn to_json(&self) -> Value {
        json!({
            "labelmap": self.labelmap,
            "layers": self.layers,
            "bitmask": self.bitmask,
            "instances": self.instances,
            "probmap": self.probmap,
            "detail": self.detail,
        })
    }
}

/// Raw bytes each encoding would consume for the analysed masks.
pub fn cost_model(stats: &OverlapStats, ignore: bool) -> CostModel {
    let n_voxels = stats.n_voxels();
    let itemsize = label_dtype_size(&stats.class_ids, ignore) as u64;
    let labelmap = stats.is_edgeless().then_some(n_voxels * itemsize);
    let layers = stats.n_layers() as u64 * n_voxels * itemsize;
    let bitmask = stats.n_planes() as u64 * n_voxels * 8;
    let instances: u64 = stats
        .class_ids
        .iter()
        .map(|cid| {
            let count = stats.counts.get(cid).copied().unwrap_or(0);
            if count == 0 {
                0
            } else {
                count / 8 + 64
            }
        })
        .sum();
    let probmap = stats.n_classes() as u64 * n_voxels * 2;
    CostModel {
        labelmap,
        layers,
        bitmask,
        instances,
        probmap,
        detail: json!({
            "itemsize": itemsize,
            "n_layers": stats.n_layers(),
            "n_planes": stats.n_planes(),
            "crossover_layers": 4 * stats.n_planes() as u64 * (2 / itemsize),
        }),
    }
}

/// Choose an encoding by measurement (spec §7.6).
///
/// `ignore` says an in-band ignore region will be written with the masks,
/// which forces `uint16` planes and moves the `layers`/`bitmask` crossover.
pub fn select_encoding(stats: &OverlapStats, soft: bool, prefer: Option<&str>, ignore: bool) -> String {
    if soft {
        return "probmap".into();
    }
    if let Some(p) = prefer {
        if p != "auto" {
            return p.into();
        }
    }
    let nonempty: Vec<i64> = stats.class_ids.iter().filter(|c| stats.counts[c] > 0).copied().collect();
    let all_localized = !nonempty.is_empty() && nonempty.iter().all(|c| stats.localized.contains(c));
    if all_localized && stats.fill() < SPARSE_FILL {
        return "instances".into();
    }
    if stats.is_edgeless() {
        return "labelmap".into();
    }
    let itemsize = label_dtype_size(&stats.class_ids, ignore);
    if stats.n_layers() < (8 / itemsize) * stats.n_planes() {
        return "layers".into();
    }
    "bitmask".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn block(shape: &[usize], origin: &[usize], size: usize) -> ArrayD<bool> {
        let mut m = ArrayD::from_elem(IxDyn(shape), false);
        for (idx, v) in m.indexed_iter_mut() {
            if (0..shape.len()).all(|a| idx[a] >= origin[a] && idx[a] < origin[a] + size) {
                *v = true;
            }
        }
        m
    }

    #[test]
    fn overlap_colouring_matches_reference() {
        let shape = [16, 24, 24];
        let mut masks = Masks::new();
        masks.insert(1, block(&shape, &[2, 2, 2], 8));
        masks.insert(2, block(&shape, &[2, 14, 2], 6));
        masks.insert(3, block(&shape, &[4, 4, 4], 3));
        let stats = analyse(&masks, None).unwrap();
        assert_eq!(stats.edges, BTreeSet::from([(1, 3)]));
        assert_eq!(stats.n_layers(), 2);
        assert_eq!(layers_from_colouring(&stats.colouring), vec![vec![1, 2], vec![3]]);
        assert_eq!(select_encoding(&stats, false, None, false), "layers");
    }
}
