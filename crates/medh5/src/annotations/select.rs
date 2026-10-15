//! Overlap analysis, graph colouring and encoding selection (spec §7.6).
//!
//! A voxel may belong to several classes at once, and there may be hundreds of
//! classes.  Colouring the class **overlap graph** and storing one integer
//! volume per colour makes the number of volumes track the *overlap depth*
//! rather than the class count: on the 200-structure phantom, mean degree 3.4
//! coloured into 5 layers.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::{json, Value};

use super::encode::check_target;
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
#[deprecated(since = "2.1.0", note = "medh5 3.0 removes bbox_volume: multiply the extents bbox(mask) returns")]
pub fn bbox_volume(mask: &ndarray::ArrayD<bool>) -> u64 {
    match bbox(mask) {
        Some(bounds) => bounds.iter().map(|(lo, hi)| (hi - lo) as u64).product(),
        None => 0,
    }
}

/// Tight `(start, stop)` per axis of a mask, or `None` when empty.
pub fn bbox(mask: &ndarray::ArrayD<bool>) -> Option<Vec<(usize, usize)>> {
    let mask = mask.as_standard_layout();
    bbox_flat(mask.as_slice().expect("standard layout"), mask.shape())
}

/// [`bbox`] of a mask given as its C-order values and its shape.
///
/// A mask is scanned a row of its last axis at a time: a row's first and
/// last `true` bound that axis, and the row's place bounds the others, so
/// no index is built per voxel.
pub(crate) fn bbox_flat(values: &[bool], shape: &[usize]) -> Option<Vec<(usize, usize)>> {
    let Some((&width, outer)) = shape.split_last() else {
        // A 0-d mask: one value, no axes.
        return values.first().copied().filter(|v| *v).map(|_| Vec::new());
    };
    if width == 0 || values.is_empty() {
        return None;
    }
    let mut lo = vec![usize::MAX; shape.len()];
    let mut hi = vec![0usize; shape.len()];
    let mut at = vec![0usize; outer.len()];
    let mut any = false;
    for row in values.chunks_exact(width) {
        if let Some(first) = row.iter().position(|v| *v) {
            let last = row.iter().rposition(|v| *v).unwrap_or(first);
            any = true;
            for (axis, index) in at.iter().enumerate() {
                lo[axis] = lo[axis].min(*index);
                hi[axis] = hi[axis].max(index + 1);
            }
            let axis = outer.len();
            lo[axis] = lo[axis].min(first);
            hi[axis] = hi[axis].max(last + 1);
        }
        // The next row's index along the outer axes, last axis fastest.
        for (index, extent) in at.iter_mut().zip(outer).rev() {
            *index += 1;
            if *index < *extent {
                break;
            }
            *index = 0;
        }
    }
    any.then(|| lo.into_iter().zip(hi).collect())
}

/// Voxels per block in [`analyse`]'s overlap pass: the per-voxel state is
/// held for one block at a time, whatever the volume.
const ANALYSE_BLOCK: usize = 1 << 16;

/// Measure counts, the overlap graph, a greedy colouring and localization.
///
/// The overlap graph comes from the set of classes each voxel carries.  The
/// sets are interned as the classes are visited in order, so a voxel's set
/// grows by one class at a time into a table of the sets that occur; two
/// classes overlap when one set holds both.  A set that later grew into a
/// larger one holds no pair the larger one lacks, so only the sets that
/// never grew are paired --- the cost follows the overlap regions, not the
/// square of the class count.
pub fn analyse(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<OverlapStats> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let n_voxels: usize = shape.iter().product();
    let planes: Vec<ndarray::CowArray<'_, bool, ndarray::IxDyn>> =
        masks.values().map(|m| m.as_standard_layout()).collect();
    let flats: Vec<&[bool]> = planes.iter().map(|p| p.as_slice().expect("standard layout")).collect();
    let mut counts = vec![0u64; flats.len()];
    // `sets[0]` is the empty set; `grown[s]` whether a class was added to
    // set `s` at some voxel, and `next` the set that made.
    let mut sets: Vec<Vec<u32>> = vec![Vec::new()];
    let mut grown: Vec<bool> = vec![false];
    let mut next: HashMap<(u32, u32), u32> = HashMap::new();
    let mut labelled = 0u64;
    let mut held = vec![0u32; ANALYSE_BLOCK.min(n_voxels)];
    let mut start = 0;
    while start < n_voxels {
        let stop = (start + ANALYSE_BLOCK).min(n_voxels);
        let held = &mut held[..stop - start];
        held.fill(0);
        for (k, flat) in flats.iter().enumerate() {
            let class = k as u32;
            // Neighbouring voxels mostly carry the same set: remember the
            // last step taken.
            let mut last: Option<(u32, u32)> = None;
            for (slot, v) in held.iter_mut().zip(&flat[start..stop]) {
                if !*v {
                    continue;
                }
                counts[k] += 1;
                let from = *slot;
                *slot = match last {
                    Some((f, to)) if f == from => to,
                    _ => {
                        let to = *next.entry((from, class)).or_insert_with(|| {
                            let mut set = sets[from as usize].clone();
                            set.push(class);
                            sets.push(set);
                            grown.push(false);
                            (sets.len() - 1) as u32
                        });
                        grown[from as usize] = true;
                        last = Some((from, to));
                        to
                    }
                };
            }
        }
        labelled += held.iter().filter(|s| **s != 0).count() as u64;
        start = stop;
    }
    let mut edges = BTreeSet::new();
    for (set, grew) in sets.iter().zip(&grown) {
        if *grew {
            continue;
        }
        for (i, a) in set.iter().enumerate() {
            for b in &set[i + 1..] {
                // Classes are visited in ascending id order.
                edges.insert((class_ids[*a as usize], class_ids[*b as usize]));
            }
        }
    }
    let counts: BTreeMap<i64, u64> = class_ids.iter().copied().zip(counts).collect();
    let mut localized = BTreeSet::new();
    for (cid, flat) in class_ids.iter().zip(&flats) {
        if counts[cid] == 0 {
            continue;
        }
        let volume: u64 = match bbox_flat(flat, &shape) {
            Some(bounds) => bounds.iter().map(|(lo, hi)| (hi - lo) as u64).product(),
            None => 0,
        };
        if volume as f64 <= LOCALIZED_BBOX_FRACTION * n_voxels as f64 {
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
///
/// A `prefer` other than `auto` is returned as given; the writer refuses one
/// the masks cannot be written in, and [`select_encoding_checked`] refuses it
/// here.
pub fn select_encoding(stats: &OverlapStats, soft: bool, prefer: Option<&str>, ignore: bool) -> String {
    if soft {
        return "probmap".into();
    }
    if let Some(p) = prefer.filter(|p| *p != "auto") {
        return p.into();
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

/// [`select_encoding`], returning a `prefer` other than `auto` only when it
/// is an encoding the masks can be written in --- one [`check_target`]
/// accepts: `mask` (E404) has no classes, so every class would merge into one
/// volume and `class_ids` would be lost, and any other name (E401) is no
/// encoding.
///
/// Public for the workspace's frontends, not as API: 2.0.x keeps the 2.0.1
/// signature of [`select_encoding`].
///
/// [`check_target`]: super::encode::check_target
#[doc(hidden)]
pub fn select_encoding_checked(stats: &OverlapStats, soft: bool, prefer: Option<&str>, ignore: bool) -> Result<String> {
    if let Some(p) = prefer.filter(|p| !soft && *p != "auto") {
        check_target(p)?;
    }
    Ok(select_encoding(stats, soft, prefer, ignore))
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
        assert_eq!(select_encoding_checked(&stats, false, None, false).unwrap(), "layers");
    }

    #[test]
    fn s7_6_a_preferred_encoding_is_one_the_masks_can_be_written_in() {
        let mut masks = Masks::new();
        masks.insert(1, block(&[4, 4, 4], &[0, 0, 0], 2));
        let stats = analyse(&masks, None).unwrap();
        for kind in ["labelmap", "layers", "bitmask", "instances", "probmap"] {
            assert_eq!(select_encoding_checked(&stats, false, Some(kind), false).unwrap(), kind);
        }
        assert_eq!(select_encoding_checked(&stats, false, Some("auto"), false).unwrap(), "labelmap");
        // `mask` keeps no class, and `rle` or a typo is no encoding at all.
        assert_eq!(select_encoding_checked(&stats, false, Some("mask"), false).unwrap_err().code(), Some("E404"));
        // The public form keeps its 2.0.1 signature: a preference is returned as given.
        assert_eq!(select_encoding(&stats, false, Some("mask"), false), "mask");
        for unknown in ["rle", "labelmaps", ""] {
            assert_eq!(select_encoding_checked(&stats, false, Some(unknown), false).unwrap_err().code(), Some("E401"));
        }
    }

    /// The per-voxel `indexed_iter` scan `bbox` was until 2.0.2.
    fn reference_bbox(mask: &ArrayD<bool>) -> Option<Vec<(usize, usize)>> {
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

    /// The pairwise scan `analyse` built the overlap graph with until 2.0.2.
    fn reference_edges(masks: &Masks) -> BTreeSet<(i64, i64)> {
        let n: usize = masks.values().next().map(|m| m.len()).unwrap_or(0);
        let mut depth = vec![0u16; n];
        for mask in masks.values() {
            for (i, v) in mask.iter().enumerate() {
                depth[i] += u16::from(*v);
            }
        }
        let multi: Vec<usize> = (0..n).filter(|i| depth[*i] > 1).collect();
        let picked: Vec<(i64, Vec<bool>)> = masks
            .iter()
            .map(|(c, m)| {
                let flat: Vec<bool> = m.iter().copied().collect();
                (*c, multi.iter().map(|i| flat[*i]).collect())
            })
            .collect();
        let mut edges = BTreeSet::new();
        for i in 0..picked.len() {
            for j in i + 1..picked.len() {
                if picked[i].1.iter().zip(&picked[j].1).any(|(x, y)| *x && *y) {
                    edges.insert((picked[i].0, picked[j].0));
                }
            }
        }
        edges
    }

    /// A deterministic mask of scattered boxes, for the equivalence tests.
    fn scattered(shape: &[usize], state: &mut u64, boxes: usize) -> ArrayD<bool> {
        let mut draw = |n: usize| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state % n.max(1) as u64) as usize
        };
        let mut m = ArrayD::from_elem(IxDyn(shape), false);
        for _ in 0..boxes {
            let origin: Vec<usize> = shape.iter().map(|n| draw(*n)).collect();
            let size: Vec<usize> = shape.iter().map(|n| 1 + draw((*n / 3).max(1))).collect();
            for (idx, v) in m.indexed_iter_mut() {
                if (0..shape.len()).all(|a| idx[a] >= origin[a] && idx[a] < origin[a] + size[a]) {
                    *v = true;
                }
            }
        }
        m
    }

    #[test]
    fn bbox_answers_as_the_per_voxel_scan_did() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for shape in [vec![7usize, 5, 9], vec![1, 1, 1], vec![3, 0, 4], vec![6], vec![4, 3, 2, 5], vec![]] {
            for boxes in [0, 1, 3] {
                let mask = scattered(&shape, &mut state, boxes);
                assert_eq!(bbox(&mask), reference_bbox(&mask), "{shape:?} with {boxes} boxes");
                // A view that is not in standard layout reads the same.
                let reversed = mask.slice_each_axis(|_| ndarray::Slice::new(0, None, -1)).to_owned();
                assert_eq!(bbox(&reversed), reference_bbox(&reversed), "{shape:?} reversed");
            }
        }
        let scalar = ArrayD::from_elem(IxDyn(&[]), true);
        assert_eq!(bbox(&scalar), Some(Vec::new()));
    }

    #[test]
    fn s7_6_analyse_finds_the_overlap_graph_the_pairwise_scan_found() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for (shape, classes) in [(vec![9usize, 11, 13], 12), (vec![40, 40], 30), (vec![5, 6, 7], 1)] {
            let mut masks = Masks::new();
            for c in 0..classes {
                masks.insert(3 * c as i64 + 1, scattered(&shape, &mut state, 2));
            }
            masks.insert(1000, ArrayD::from_elem(IxDyn(&shape), false));
            let stats = analyse(&masks, None).unwrap();
            assert_eq!(stats.edges, reference_edges(&masks), "{shape:?}");
            for (c, m) in &masks {
                assert_eq!(stats.counts[c], m.iter().filter(|v| **v).count() as u64);
            }
            let union = (0..shape.iter().product::<usize>())
                .filter(|i| masks.values().any(|m| m.as_slice().unwrap()[*i]))
                .count();
            assert_eq!(stats.n_labelled_voxels, union as u64);
        }
        // More voxels than one block holds.
        let shape = [3usize, 200, 300];
        let mut masks = Masks::new();
        for c in 1..=5 {
            masks.insert(c, scattered(&shape, &mut state, 3));
        }
        assert_eq!(analyse(&masks, None).unwrap().edges, reference_edges(&masks));
    }
}
