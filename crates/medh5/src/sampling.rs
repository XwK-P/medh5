//! Patch and timepoint-pair sampling (§14.3).
//!
//! Free of any deep-learning dependency: a sampler decides *where* to read,
//! which is a geometry question.  Foreground sampling reads the **cached
//! coordinate subsample** of §14.3 rather than scanning a mask, which is O(1)
//! in volume size; a file with no current index still works by scanning, and
//! every [`Patch`] says which happened, because a silent 20x slowdown in a
//! dataloader is indistinguishable from a slow disk.
//!
//! Every draw comes from one [`Rng`], so a seed fixes the patches whichever
//! frontend asks.

use indexmap::IndexMap;
use ndarray::Axis;
use serde_json::{json, Value};

use crate::annotations::Annotation;
use crate::json::{py_float, repr_int_tuple, repr_list, repr_str};
use crate::labels::ClassKey;
use crate::rng::Rng;
use crate::sample::Sample;
use crate::{Error, Result};

pub const STRATEGIES: [&str; 3] = ["uniform", "foreground", "balanced"];
pub const PAIR_MODES: [&str; 3] = ["consecutive", "baseline_vs_all", "all_pairs"];

/// One draw: where to read, and how the location was chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    /// `(start, stop)` per spatial axis.
    pub slices: Vec<(i64, i64)>,
    /// Per-axis `(before, after)` padding where the volume is smaller than the
    /// patch.
    pub pad: Vec<(i64, i64)>,
    pub center: Vec<i64>,
    pub strategy: String,
    pub class_id: Option<i64>,
    /// Whether the §14.3 index answered the foreground query: `None` when the
    /// question did not arise (a uniform draw consults no index).
    pub used_index: Option<bool>,
    /// The grid whose index coordinates `slices` are in.
    pub grid_id: Option<String>,
}

impl Patch {
    /// The padding, `(0, 0)` on every axis when none was recorded.
    pub fn padding(&self) -> Vec<(i64, i64)> {
        if self.pad.is_empty() {
            vec![(0, 0); self.slices.len()]
        } else {
            self.pad.clone()
        }
    }

    /// The shape a read padded with [`Patch::padding`] has.
    pub fn shape(&self) -> Vec<i64> {
        self.slices.iter().zip(self.padding()).map(|((a, b), (before, after))| (b - a) + before + after).collect()
    }

    pub fn needs_padding(&self) -> bool {
        self.padding().iter().any(|(before, after)| *before != 0 || *after != 0)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "start": self.slices.iter().map(|s| s.0).collect::<Vec<_>>(),
            "stop": self.slices.iter().map(|s| s.1).collect::<Vec<_>>(),
            "pad": self.padding().iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
            "center": self.center,
            "strategy": self.strategy,
            "class_id": self.class_id,
            "used_index": self.used_index,
            "grid_id": self.grid_id,
        })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let window: Vec<String> = self.slices.iter().map(|(a, b)| format!("{a}:{b}")).collect();
        format!("Patch([{}], {})", window.join(", "), self.strategy)
    }
}

/// A patch size: one length for every axis, or one per axis.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchSize {
    Scalar(i64),
    Axes(Vec<i64>),
}

impl PatchSize {
    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        match self {
            PatchSize::Scalar(v) => v.to_string(),
            PatchSize::Axes(v) => repr_int_tuple(v),
        }
    }
}

/// Broadcast a patch size across `ndim` spatial axes.
pub fn coerce_patch_size(patch_size: &PatchSize, ndim: usize) -> Result<Vec<i64>> {
    let size = match patch_size {
        PatchSize::Scalar(v) => vec![*v; ndim],
        PatchSize::Axes(v) => v.clone(),
    };
    if size.len() != ndim {
        return Err(Error::invalid(format!(
            "patch_size {} has {} axes; the grid has {ndim}",
            repr_int_tuple(&size),
            size.len()
        )));
    }
    if size.iter().any(|v| *v <= 0) {
        return Err(Error::invalid(format!("patch_size {} must be positive on every axis", repr_int_tuple(&size))));
    }
    Ok(size)
}

/// `(slices, padding)`: `[start, stop)` per axis, and `(before, after)` per axis.
pub type Window = (Vec<(i64, i64)>, Vec<(i64, i64)>);

/// Slices covering `patch` voxels around `center`, plus the padding needed.
///
/// A centre near the edge is shifted inwards rather than clipped, so a patch
/// keeps its requested size wherever the volume allows; padding appears only
/// on an axis genuinely shorter than the patch.
pub fn window_around(center: &[i64], patch: &[i64], shape: &[i64]) -> Window {
    let mut slices = Vec::with_capacity(patch.len());
    let mut pads = Vec::with_capacity(patch.len());
    for ((c, size), extent) in center.iter().zip(patch).zip(shape) {
        if size >= extent {
            slices.push((0, *extent));
            let spare = size - extent;
            let before = spare.div_euclid(2);
            pads.push((before, spare - before));
            continue;
        }
        let start = (c - size.div_euclid(2)).min(extent - size).max(0);
        slices.push((start, start + size));
        pads.push((0, 0));
    }
    (slices, pads)
}

/// A centre whose *window* lands uniformly over the volume.
///
/// Drawing the centre over every voxel and clamping would pile the leading and
/// trailing half-patches onto the first and last window; drawing from the
/// range that maps one-to-one onto the valid window starts makes the window
/// uniform, which is what `uniform` means.
fn uniform_center(shape: &[i64], patch: &[i64], rng: &mut Rng) -> Result<Vec<i64>> {
    let mut center = Vec::with_capacity(shape.len());
    for (extent, size) in shape.iter().zip(patch) {
        if size >= extent {
            center.push(extent.div_euclid(2));
            continue;
        }
        center.push(rng.integer(0, extent - size + 1)? + size.div_euclid(2));
    }
    Ok(center)
}

/// How foreground classes are weighted when a class is picked.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassWeights {
    /// `"uniform"`, `"inverse_frequency"` or `"frequency"`.
    Named(String),
    /// Class id -> weight; a class absent from the map weighs nothing.
    Explicit(IndexMap<i64, f64>),
}

/// Choose patch windows in a volume (§14.3).
///
/// `uniform` draws centres over the volume, `foreground` from indexed
/// foreground coordinates, and `balanced` foreground with probability
/// `foreground_prob` and uniform otherwise --- what nearly every segmentation
/// recipe uses, because pure foreground sampling never shows the model the
/// background it will be evaluated on.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchSampler {
    pub patch_size: PatchSize,
    pub strategy: String,
    pub foreground_prob: f64,
    pub foreground_classes: Option<Vec<ClassKey>>,
    pub class_weights: ClassWeights,
}

impl PatchSampler {
    pub fn new(
        patch_size: PatchSize,
        strategy: &str,
        foreground_prob: f64,
        foreground_classes: Option<Vec<ClassKey>>,
        class_weights: ClassWeights,
    ) -> Result<PatchSampler> {
        if !STRATEGIES.contains(&strategy) {
            return Err(Error::invalid(format!(
                "unknown sampling strategy {}; expected one of {}",
                repr_str(strategy),
                repr_list(&STRATEGIES)
            )));
        }
        if !(0.0..=1.0).contains(&foreground_prob) {
            return Err(Error::invalid("foreground_prob must lie in [0, 1]"));
        }
        Ok(PatchSampler { patch_size, strategy: strategy.into(), foreground_prob, foreground_classes, class_weights })
    }

    /// The defaults: `balanced`, `foreground_prob = 0.5`, uniform weights.
    pub fn with_size(patch_size: PatchSize) -> PatchSampler {
        PatchSampler {
            patch_size,
            strategy: "balanced".into(),
            foreground_prob: 0.5,
            foreground_classes: None,
            class_weights: ClassWeights::Named("uniform".into()),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("PatchSampler({}, {})", self.patch_size.repr(), self.strategy)
    }

    /// Draw one patch window from `sample`.
    ///
    /// `grid` pins the grid the window is measured in; `annotation = None`
    /// alone means "find me one", which for a longitudinal sample can find
    /// another visit's.
    pub fn draw(&self, sample: &Sample, annotation: Option<&str>, rng: &mut Rng, grid: Option<&str>) -> Result<Patch> {
        let ann = self.annotation(sample, annotation, grid)?;
        let grid_id = self.window_grid(sample, ann.as_deref(), grid)?;
        let shape: Vec<i64> = sample.grid(&grid_id)?.spatial_shape().iter().map(|v| *v as i64).collect();
        let patch = coerce_patch_size(&self.patch_size, shape.len())?;
        let want_foreground =
            self.strategy == "foreground" || (self.strategy == "balanced" && rng.random() < self.foreground_prob);
        if want_foreground {
            if let Some(name) = &ann {
                if let Some((center, class_id, used_index)) = self.foreground_center(sample, name, rng)? {
                    let (slices, pad) = window_around(&center, &patch, &shape);
                    return Ok(Patch {
                        slices,
                        pad,
                        center,
                        strategy: "foreground".into(),
                        class_id: Some(class_id),
                        used_index: Some(used_index),
                        grid_id: Some(grid_id),
                    });
                }
            }
        }
        let center = uniform_center(&shape, &patch, rng)?;
        let (slices, pad) = window_around(&center, &patch, &shape);
        Ok(Patch {
            slices,
            pad,
            center,
            strategy: "uniform".into(),
            class_id: None,
            used_index: None,
            grid_id: Some(grid_id),
        })
    }

    /// `n` draws from one generator.
    pub fn draws(&self, sample: &Sample, annotation: Option<&str>, n: usize, rng: &mut Rng) -> Result<Vec<Patch>> {
        (0..n).map(|_| self.draw(sample, annotation, rng, None)).collect()
    }

    /// The annotation to draw foreground from, auto-selected if not named.
    ///
    /// Auto-selection stays inside `grid` when the caller named one, and skips
    /// `mask` (no classes, so it can only answer "no foreground here").
    pub fn annotation(&self, sample: &Sample, annotation: Option<&str>, grid: Option<&str>) -> Result<Option<String>> {
        if let Some(name) = annotation {
            return Ok(Some(name.to_string()));
        }
        for (name, ann) in sample.annotations()? {
            if !ann.is_voxel() || ann.kind() == "mask" {
                continue;
            }
            if grid.is_some() && ann.grid_id() != grid {
                continue;
            }
            return Ok(Some(name.clone()));
        }
        Ok(None)
    }

    /// The grid a draw is measured in: the annotation's own where it has one,
    /// `grid` where the caller named one, the reference grid otherwise.
    pub fn window_grid(&self, sample: &Sample, annotation: Option<&str>, grid: Option<&str>) -> Result<String> {
        if let Some(name) = annotation {
            if let Some(ann) = sample.annotations()?.get(name) {
                if let Some(declared) = ann.grid_id() {
                    if let Some(g) = grid {
                        if declared != g {
                            return Err(Error::invalid(format!(
                                "annotation {} is on grid {}, so a window drawn from its foreground cannot be \
                                 measured in grid {}",
                                repr_str(name),
                                repr_str(declared),
                                repr_str(g)
                            )));
                        }
                    }
                    return Ok(declared.to_string());
                }
            }
        }
        if let Some(g) = grid {
            return Ok(g.to_string());
        }
        Ok(sample.reference_grid()?.grid_id)
    }

    /// A foreground voxel, from the index when there is a current one.
    ///
    /// A stale index is worse than none: its coordinates point at foreground
    /// the annotation no longer has, so it is never consulted.  Nor is one
    /// built over some of the candidate classes, which would never pick the
    /// others.  Either way the class is picked by `class_weights` from every
    /// candidate's voxel count, and `None` --- no candidate with foreground
    /// and weight --- is the answer, not a cue to pick unweighted: the
    /// index path fell back to a scan that drew classes uniformly, so a class
    /// weighted zero was drawn, and the no-index path ignored the weights
    /// altogether (F20 of the round-4 audit).
    fn foreground_center(
        &self,
        sample: &Sample,
        annotation: &str,
        rng: &mut Rng,
    ) -> Result<Option<(Vec<i64>, i64, bool)>> {
        let ann = sample.annotation(annotation)?;
        let classes = self.classes(ann)?;
        if classes.is_empty() {
            return Ok(None);
        }
        let index = if sample.fresh_indices()?.contains(annotation) { sample.index()?.get(annotation) } else { None };
        if let Some(index) = index {
            let mut covered = true;
            for c in &classes {
                covered &= index.has_class(*c)?;
            }
            if covered {
                let counts = index.voxel_counts()?;
                let counted: IndexMap<i64, i64> =
                    classes.iter().map(|c| (*c, counts.get(c).copied().unwrap_or(0))).collect();
                let Some(picked) = self.pick_class(&counted, rng)? else { return Ok(None) };
                let coords = index.sample_foreground(picked, 1, rng)?;
                let center: Vec<i64> = coords.row(0).iter().map(|v| i64::from(*v)).collect();
                return Ok(Some((center, picked, true)));
            }
        }
        self.scan_center(ann, &classes, rng)
    }

    fn classes(&self, ann: &Annotation) -> Result<Vec<i64>> {
        match &self.foreground_classes {
            None => Ok(ann.class_ids().to_vec()),
            Some(keys) => keys.iter().map(|k| ann.resolve_class(k)).collect(),
        }
    }

    /// Choose a class to sample from, weighted as configured.
    pub fn pick_class(&self, counts: &IndexMap<i64, i64>, rng: &mut Rng) -> Result<Option<i64>> {
        let present: Vec<(i64, i64)> = counts.iter().filter(|(_, n)| **n > 0).map(|(c, n)| (*c, *n)).collect();
        if present.is_empty() {
            return Ok(None);
        }
        let weights: Vec<(i64, f64)> = match &self.class_weights {
            ClassWeights::Named(name) => match name.as_str() {
                "uniform" => present.iter().map(|(c, _)| (*c, 1.0)).collect(),
                "inverse_frequency" => present.iter().map(|(c, n)| (*c, 1.0 / *n as f64)).collect(),
                "frequency" => present.iter().map(|(c, n)| (*c, *n as f64)).collect(),
                other => return Err(Error::invalid(format!("unknown class_weights {}", repr_str(other)))),
            },
            ClassWeights::Explicit(map) => {
                present.iter().map(|(c, _)| (*c, map.get(c).copied().unwrap_or(0.0))).collect()
            }
        };
        // Python's `sum()` over the dict, in insertion order.
        let total = weights.iter().fold(0.0, |acc, (_, w)| acc + w);
        if total <= 0.0 {
            return Ok(None);
        }
        let mut keys: Vec<(i64, f64)> = weights;
        keys.sort_by_key(|(c, _)| *c);
        let ids: Vec<i64> = keys.iter().map(|(c, _)| *c).collect();
        let probabilities: Vec<f64> = keys.iter().map(|(_, w)| w / total).collect();
        Ok(Some(rng.choice_weighted(&ids, &probabilities)?))
    }

    /// The O(volume) fallback for a file with no current sampling index:
    /// every candidate class counted, one dense mask at a time, a class
    /// picked by `class_weights` as the index path picks it, and a voxel of
    /// that class.
    fn scan_center(&self, ann: &Annotation, classes: &[i64], rng: &mut Rng) -> Result<Option<(Vec<i64>, i64, bool)>> {
        let mut counted: IndexMap<i64, i64> = IndexMap::new();
        for class_id in classes {
            if !counted.contains_key(class_id) {
                let mask = ann.dense(Some(&[ClassKey::Id(*class_id)]), None)?;
                counted.insert(*class_id, mask.iter().filter(|v| **v).count() as i64);
            }
        }
        let Some(class_id) = self.pick_class(&counted, rng)? else { return Ok(None) };
        let mask = ann.dense(Some(&[ClassKey::Id(class_id)]), None)?.index_axis_move(Axis(0), 0);
        let total = mask.iter().filter(|v| **v).count();
        let pick = rng.integer(0, total as i64)? as usize;
        // `np.argwhere` order: C order over the voxels.
        let (flat, _) = mask.iter().enumerate().filter(|(_, v)| **v).nth(pick).expect("pick < total");
        let mut rest = flat;
        let mut coords = vec![0i64; mask.ndim()];
        for (axis, extent) in mask.shape().iter().enumerate().rev() {
            coords[axis] = (rest % extent) as i64;
            rest /= extent;
        }
        Ok(Some((coords, class_id, false)))
    }
}

/// One ordered pair of visits, and the change label spanning it, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct TimepointPair {
    pub first: String,
    pub second: String,
    pub interval_days: Option<f64>,
    pub label: Option<String>,
}

impl TimepointPair {
    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let gap = self.interval_days.map(|d| format!(", {}d", py_float(d))).unwrap_or_default();
        format!("TimepointPair({} -> {}{gap})", self.first, self.second)
    }
}

/// Enumerate the visit pairs a longitudinal model trains on (§3.7, §9).
///
/// A cross-sectional sample yields **no** pairs, reported as a count rather
/// than absorbed silently.
#[derive(Debug, Clone, PartialEq)]
pub struct TimepointPairSampler {
    pub mode: String,
}

impl TimepointPairSampler {
    pub fn new(mode: &str) -> Result<TimepointPairSampler> {
        if !PAIR_MODES.contains(&mode) {
            return Err(Error::invalid(format!(
                "unknown pair mode {}; expected one of {}",
                repr_str(mode),
                repr_list(&PAIR_MODES)
            )));
        }
        Ok(TimepointPairSampler { mode: mode.into() })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("TimepointPairSampler({})", repr_str(&self.mode))
    }

    pub fn pairs(&self, sample: &Sample) -> Result<Vec<TimepointPair>> {
        let timeline = sample.timepoints()?;
        let ids = timeline.ids();
        if ids.len() < 2 {
            return Ok(Vec::new());
        }
        let combos: Vec<(String, String)> = match self.mode.as_str() {
            "consecutive" => ids.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect(),
            "baseline_vs_all" => ids[1..].iter().map(|later| (ids[0].clone(), later.clone())).collect(),
            _ => {
                let mut out = Vec::new();
                for (i, a) in ids.iter().enumerate() {
                    for b in &ids[i + 1..] {
                        out.push((a.clone(), b.clone()));
                    }
                }
                out
            }
        };
        combos
            .into_iter()
            .map(|(a, b)| {
                Ok(TimepointPair {
                    interval_days: timeline.interval_days(&a, &b)?,
                    label: change_label(sample, &a, &b)?,
                    first: a,
                    second: b,
                })
            })
            .collect()
    }
}

/// The classification annotation whose `timepoints` is exactly this ordered
/// pair: "grew 40 %" written `(tp1, tp0)` does not describe `(tp0, tp1)`.
fn change_label(sample: &Sample, first: &str, second: &str) -> Result<Option<String>> {
    for (name, ann) in sample.annotations()? {
        if ann.kind() != "classification" {
            continue;
        }
        let tps = ann.timepoints();
        if tps.len() == 2 && tps[0] == first && tps[1] == second {
            return Ok(Some(name.clone()));
        }
    }
    Ok(None)
}

/// What a paired dataset kept and what it skipped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairReport {
    pub files: usize,
    pub pairs: usize,
    pub skipped: Vec<String>,
}

impl PairReport {
    pub fn add_skip(&mut self, path: &str) {
        self.skipped.push(path.to_string());
    }

    pub fn summary(&self) -> Value {
        json!({
            "files": self.files,
            "pairs": self.pairs,
            "skipped_cross_sectional": self.skipped.len(),
            "examples": self.skipped.iter().take(5).collect::<Vec<_>>(),
        })
    }
}

impl std::fmt::Display for PairReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} pairs from {} files; {} cross-sectional file(s) contributed none",
            self.pairs,
            self.files,
            self.skipped.len()
        )
    }
}

/// Deterministic sliding-window cover of a volume --- the inference path.
///
/// Every voxel is covered, and the last window on each axis is shifted inwards
/// rather than padded, so predictions near the far edge come from real data.
pub fn grid_patches(shape: &[i64], patch_size: &PatchSize, overlap: i64, grid_id: Option<&str>) -> Result<Vec<Patch>> {
    let patch = coerce_patch_size(patch_size, shape.len())?;
    if overlap < 0 || patch.iter().any(|p| overlap >= *p) {
        return Err(Error::invalid("overlap must be non-negative and below patch_size"));
    }
    let mut starts_per_axis: Vec<Vec<i64>> = Vec::new();
    for (size, n) in patch.iter().zip(shape) {
        let step = size - overlap;
        if size >= n {
            starts_per_axis.push(vec![0]);
            continue;
        }
        let mut positions: Vec<i64> = (0..=(n - size)).step_by(step as usize).collect();
        if positions.last() != Some(&(n - size)) {
            positions.push(n - size);
        }
        starts_per_axis.push(positions);
    }
    let mut out = Vec::new();
    let mut odometer = vec![0usize; starts_per_axis.len()];
    loop {
        let corner: Vec<i64> = odometer.iter().zip(&starts_per_axis).map(|(i, axis)| axis[*i]).collect();
        let slices: Vec<(i64, i64)> =
            corner.iter().zip(&patch).zip(shape).map(|((start, size), n)| (*start, (start + size).min(*n))).collect();
        let pad: Vec<(i64, i64)> = slices.iter().zip(&patch).map(|((a, b), size)| (0, size - (b - a))).collect();
        let center: Vec<i64> = slices.iter().map(|(a, b)| a + (b - a).div_euclid(2)).collect();
        out.push(Patch {
            slices,
            pad,
            center,
            strategy: "grid".into(),
            class_id: None,
            used_index: None,
            grid_id: grid_id.map(str::to_string),
        });
        // Advance the last axis fastest, as the recursive odometer does.
        let mut axis = odometer.len();
        loop {
            if axis == 0 {
                return Ok(out);
            }
            axis -= 1;
            odometer[axis] += 1;
            if odometer[axis] < starts_per_axis[axis].len() {
                break;
            }
            odometer[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s14_3_windows_shift_inwards_and_pad_only_short_axes() {
        let (slices, pad) = window_around(&[1, 30], &[8, 8], &[24, 32]);
        assert_eq!(slices, vec![(0, 8), (24, 32)]);
        assert_eq!(pad, vec![(0, 0), (0, 0)]);
        let (slices, pad) = window_around(&[2], &[9], &[4]);
        assert_eq!(slices, vec![(0, 4)]);
        assert_eq!(pad, vec![(2, 3)]);
    }

    #[test]
    fn grid_patches_cover_every_voxel_without_padding_inside() {
        let patches = grid_patches(&[10, 7], &PatchSize::Scalar(4), 1, Some("ct")).unwrap();
        let starts: Vec<Vec<i64>> = patches.iter().map(|p| p.slices.iter().map(|s| s.0).collect()).collect();
        assert_eq!(starts[..4], [vec![0, 0], vec![0, 3], vec![3, 0], vec![3, 3]]);
        assert_eq!(starts.last().unwrap(), &vec![6, 3]);
        assert!(patches.iter().all(|p| !p.needs_padding()));
        assert_eq!(patches[0].repr(), "Patch([0:4, 0:4], grid)");
    }

    #[test]
    fn patch_sizes_are_checked_against_the_grid() {
        let err = coerce_patch_size(&PatchSize::Axes(vec![64]), 3).unwrap_err();
        assert_eq!(err.to_string(), "patch_size (64,) has 1 axes; the grid has 3");
        let err = coerce_patch_size(&PatchSize::Scalar(0), 2).unwrap_err();
        assert_eq!(err.to_string(), "patch_size (0, 0) must be positive on every axis");
    }
}
