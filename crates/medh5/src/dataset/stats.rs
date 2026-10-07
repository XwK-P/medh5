//! Streaming cohort statistics: intensity moments and class frequencies.
//!
//! One pass per file, constant memory per worker, and an exact merge.  Means
//! are not averaged per file (the Welford/Chan merge weights by voxel count),
//! an unexamined class is not a zero (§11.3), class statistics come from
//! voxel annotations only, and moments are over the voxels an image holds data
//! in (`valid_mask`, §4.4).  Intensities are **physical** by default --- what
//! the loaders hand a model --- and `physical = false` measures stored values.
//!
//! The arithmetic is NumPy's, operation for operation (pairwise sums, the same
//! merge formula evaluated in the same order), so a statistics file is the one
//! 1.x wrote.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use crate::array::Slice;
use crate::json::{num, repr_int_list, repr_str};
use crate::numeric::pairwise_sum;
use crate::sample::{open_sample, Sample};
use crate::{Error, Result};

/// Count, mean and M2 for one image key --- a Welford accumulator.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub minimum: f64,
    pub maximum: f64,
}

impl Default for Moments {
    fn default() -> Self {
        Moments { count: 0, mean: 0.0, m2: 0.0, minimum: f64::INFINITY, maximum: f64::NEG_INFINITY }
    }
}

/// NumPy's `min`/`max` reduction: NaN propagates.
fn reduce(values: &[f64], pick: fn(f64, f64) -> f64) -> f64 {
    let mut out = values[0];
    for v in &values[1..] {
        if out.is_nan() {
            return out;
        }
        out = if v.is_nan() { *v } else { pick(out, *v) };
    }
    out
}

impl Moments {
    /// Fold in one block of values.
    pub fn update(&mut self, block: &[f64]) {
        if block.is_empty() {
            return;
        }
        let n = block.len() as f64;
        let mean = pairwise_sum(block) / n;
        let squares: Vec<f64> = block.iter().map(|v| (v - mean) * (v - mean)).collect();
        self.merge(&Moments {
            count: block.len() as u64,
            mean,
            m2: pairwise_sum(&squares),
            minimum: reduce(block, f64::min),
            maximum: reduce(block, f64::max),
        });
    }

    /// Chan-Golub-LeVeque parallel merge --- exact, not an approximation.
    pub fn merge(&mut self, other: &Moments) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean += delta * other.count as f64 / total as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / total as f64;
        self.count = total;
        // Python's builtin `min`/`max`: the first argument unless the second
        // is strictly better.
        if other.minimum < self.minimum {
            self.minimum = other.minimum;
        }
        if other.maximum > self.maximum {
            self.maximum = other.maximum;
        }
    }

    pub fn std(&self) -> f64 {
        if self.count > 1 {
            (self.m2 / self.count as f64).sqrt()
        } else {
            0.0
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "count": self.count,
            "mean": num(self.mean),
            "std": num(self.std()),
            "min": if self.count == 0 { Value::Null } else { num(self.minimum) },
            "max": if self.count == 0 { Value::Null } else { num(self.maximum) },
        })
    }

    pub fn from_json(doc: &Value) -> Result<Moments> {
        let count = doc.get("count").and_then(Value::as_u64).ok_or_else(|| Error::Key("'count'".into()))?;
        let std = doc.get("std").and_then(Value::as_f64).unwrap_or(0.0);
        Ok(Moments {
            count,
            mean: doc.get("mean").and_then(Value::as_f64).unwrap_or(0.0),
            m2: std * std * count as f64,
            minimum: doc.get("min").and_then(Value::as_f64).unwrap_or(f64::INFINITY),
            maximum: doc.get("max").and_then(Value::as_f64).unwrap_or(f64::NEG_INFINITY),
        })
    }
}

/// How often a class occurs, over the samples that examined it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClassStats {
    pub class_id: i64,
    pub voxels: u64,
    /// Samples in which the class was found (a longitudinal sample counts once).
    pub present_in: u64,
    /// Samples that looked for the class.
    pub examined_in: u64,
}

impl ClassStats {
    pub fn new(class_id: i64) -> ClassStats {
        ClassStats { class_id, ..Default::default() }
    }

    /// Fraction of *examining* samples in which the class was found.
    pub fn prevalence(&self) -> f64 {
        if self.examined_in == 0 {
            0.0
        } else {
            self.present_in as f64 / self.examined_in as f64
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "class_id": self.class_id,
            "voxels": self.voxels,
            "present_in": self.present_in,
            "examined_in": self.examined_in,
            "prevalence": num(self.prevalence()),
        })
    }
}

/// Everything one streaming pass over a cohort learned.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    pub samples: u64,
    pub images: IndexMap<String, Moments>,
    pub classes: IndexMap<i64, ClassStats>,
    pub total_voxels: u64,
    pub failures: Vec<String>,
    /// Whether the image moments are over physical values (rescale applied).
    pub physical: bool,
}

impl Default for DatasetStats {
    fn default() -> Self {
        DatasetStats {
            samples: 0,
            images: IndexMap::new(),
            classes: IndexMap::new(),
            total_voxels: 0,
            failures: Vec::new(),
            physical: true,
        }
    }
}

impl DatasetStats {
    pub fn merge(&mut self, other: &DatasetStats) -> Result<()> {
        if other.samples > 0 {
            // One pass, one convention: a mean over physical values merged
            // with one over stored values describes no image anybody can load.
            if self.samples > 0 && other.physical != self.physical {
                return Err(Error::File(
                    "cannot merge statistics over physical values with statistics over stored values; compute both \
                     passes the same way"
                        .into(),
                ));
            }
            self.physical = other.physical;
        }
        self.samples += other.samples;
        self.total_voxels += other.total_voxels;
        self.failures.extend(other.failures.iter().cloned());
        for (key, moments) in &other.images {
            self.images.entry(key.clone()).or_default().merge(moments);
        }
        for (class_id, stats) in &other.classes {
            let mine = self.classes.entry(*class_id).or_insert_with(|| ClassStats::new(*class_id));
            mine.voxels += stats.voxels;
            mine.present_in += stats.present_in;
            mine.examined_in += stats.examined_in;
        }
        Ok(())
    }

    /// `(mean, std)` for a z-score transform, or `(0, 1)` if unseen.
    pub fn normalization(&self, image_key: &str) -> (f64, f64) {
        match self.images.get(image_key) {
            Some(m) if m.count > 0 => (m.mean, if m.std() == 0.0 { 1.0 } else { m.std() }),
            _ => (0.0, 1.0),
        }
    }

    /// Classes with no voxels in these statistics: they get no loss weight.
    pub fn unweighted_classes(&self) -> Vec<i64> {
        let mut empty: Vec<i64> = self.classes.iter().filter(|(_, s)| s.voxels == 0).map(|(c, _)| *c).collect();
        empty.sort_unstable();
        empty
    }

    /// The warning [`class_weights`](Self::class_weights) has for a caller,
    /// when some class gets no weight.
    pub fn class_weights_warning(&self) -> Option<String> {
        let empty = self.unweighted_classes();
        if empty.is_empty() {
            return None;
        }
        Some(format!(
            "class_weights: class(es) {} have no voxels in these statistics, so they get no weight; a weight for an \
             unseen class is a guess, and the inverse of zero is not one",
            repr_int_list(&empty)
        ))
    }

    /// Loss weights from measured voxel frequencies, normalised to a mean of
    /// 1 so switching schemes does not silently rescale the learning rate.
    pub fn class_weights(&self, scheme: &str) -> Result<BTreeMap<i64, f64>> {
        let counts: Vec<(i64, u64)> =
            self.classes.iter().filter(|(_, s)| s.voxels > 0).map(|(c, s)| (*c, s.voxels)).collect();
        if counts.is_empty() {
            return Ok(BTreeMap::new());
        }
        let raw: Vec<(i64, f64)> = match scheme {
            "inverse_frequency" => counts.iter().map(|(c, v)| (*c, 1.0 / *v as f64)).collect(),
            "inverse_sqrt" => counts.iter().map(|(c, v)| (*c, 1.0 / (*v as f64).sqrt())).collect(),
            "uniform" => counts.iter().map(|(c, _)| (*c, 1.0)).collect(),
            other => {
                return Err(Error::File(format!(
                    "unknown weighting scheme {}; expected inverse_frequency, inverse_sqrt or uniform",
                    repr_str(other)
                )))
            }
        };
        let total = raw.iter().fold(0.0, |acc, (_, v)| acc + v);
        let scale = raw.len() as f64 / total;
        Ok(raw.into_iter().map(|(c, v)| (c, v * scale)).collect())
    }

    pub fn to_json(&self) -> Value {
        let mut images: Vec<(&String, &Moments)> = self.images.iter().collect();
        images.sort_by(|a, b| a.0.cmp(b.0));
        let mut classes: Vec<&ClassStats> = self.classes.values().collect();
        classes.sort_by_key(|s| s.class_id);
        json!({
            "samples": self.samples,
            "physical": self.physical,
            "total_voxels": self.total_voxels,
            "images": images.into_iter().map(|(k, v)| (k.clone(), v.to_json())).collect::<Map<String, Value>>(),
            "classes": classes.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "failures": self.failures,
        })
    }

    pub fn from_json(doc: &Value) -> Result<DatasetStats> {
        let mut images = IndexMap::new();
        if let Some(Value::Object(m)) = doc.get("images") {
            for (k, v) in m {
                images.insert(k.clone(), Moments::from_json(v)?);
            }
        }
        let mut classes = IndexMap::new();
        if let Some(Value::Array(items)) = doc.get("classes") {
            for c in items {
                let id = c.get("class_id").and_then(Value::as_i64).ok_or_else(|| Error::Key("'class_id'".into()))?;
                let count = |k: &str| c.get(k).and_then(Value::as_u64).unwrap_or(0);
                classes.insert(
                    id,
                    ClassStats {
                        class_id: id,
                        voxels: count("voxels"),
                        present_in: count("present_in"),
                        examined_in: count("examined_in"),
                    },
                );
            }
        }
        Ok(DatasetStats {
            samples: doc.get("samples").and_then(Value::as_u64).unwrap_or(0),
            physical: doc.get("physical").and_then(Value::as_bool).unwrap_or(true),
            total_voxels: doc.get("total_voxels").and_then(Value::as_u64).unwrap_or(0),
            images,
            classes,
            failures: doc
                .get("failures")
                .and_then(Value::as_array)
                .map(|f| f.iter().map(crate::json::py_str).collect())
                .unwrap_or_default(),
        })
    }
}

/// What to measure in each file.
#[derive(Debug, Clone)]
pub struct StatsOptions {
    /// Image ids to include; `None` for every image.
    pub images: Option<Vec<String>>,
    /// Annotation ids to include; `None` for every annotation.
    pub annotations: Option<Vec<String>>,
    /// Read every Nth slab along the first axis (approximate, opt-in).
    pub sample_stride: usize,
    pub physical: bool,
}

impl Default for StatsOptions {
    fn default() -> Self {
        StatsOptions { images: None, annotations: None, sample_stride: 1, physical: true }
    }
}

/// One file's contribution, read slab by slab rather than whole.
pub fn stats_for(path: &Path, options: &StatsOptions) -> Result<DatasetStats> {
    let mut out = DatasetStats { samples: 1, physical: options.physical, ..Default::default() };
    let sample = open_sample(path)?;
    let mut measured: BTreeSet<String> = BTreeSet::new();
    for (key, image) in sample.images()? {
        if options.images.as_ref().is_some_and(|wanted| !wanted.contains(key)) {
            continue;
        }
        let moments = out.images.entry(key.clone()).or_default();
        for_each_block(&sample, key, options.sample_stride, options.physical, |block| moments.update(block))?;
        measured.insert(image.grid_id()?);
    }
    let fresh = sample.fresh_indices()?.clone();
    let mut examined: BTreeSet<i64> = BTreeSet::new();
    let mut present: BTreeSet<i64> = BTreeSet::new();
    for (key, annotation) in sample.annotations()? {
        if options.annotations.as_ref().is_some_and(|wanted| !wanted.contains(key)) {
            continue;
        }
        // Voxel classes only: a `mask` has none, and a classification or a
        // geometric annotation names classes it holds no voxels of.
        if !annotation.is_voxel() || annotation.kind() == "mask" {
            continue;
        }
        let counts: IndexMap<i64, u64> = if fresh.contains(key) {
            match sample.index()?.get(key) {
                Some(index) => index.voxel_counts()?.into_iter().map(|(c, n)| (c, n.max(0) as u64)).collect(),
                None => IndexMap::new(),
            }
        } else {
            annotation.voxel_counts(None)?
        };
        let looked_for: BTreeSet<i64> = annotation.annotated_class_ids().iter().copied().collect();
        examined.extend(looked_for.iter().copied());
        let mut classes: BTreeSet<i64> = annotation.class_ids().iter().copied().collect();
        classes.extend(looked_for.iter().copied());
        for class_id in classes {
            let voxels = counts.get(&class_id).copied().unwrap_or(0);
            let stats = out.classes.entry(class_id).or_insert_with(|| ClassStats::new(class_id));
            stats.voxels += voxels;
            if voxels > 0 {
                present.insert(class_id);
            }
        }
    }
    // Once per sample, not once per annotation.
    for class_id in examined {
        out.classes.entry(class_id).or_insert_with(|| ClassStats::new(class_id)).examined_in += 1;
    }
    for class_id in present {
        out.classes.entry(class_id).or_insert_with(|| ClassStats::new(class_id)).present_in += 1;
    }
    // Each measured image's own grid, once.
    for grid_id in measured {
        out.total_voxels += sample.grid(&grid_id)?.spatial_shape().iter().map(|v| *v as u64).product::<u64>();
    }
    Ok(out)
}

/// Slabs of an image along its first spatial axis, valid voxels only, one at
/// a time: memory stays at one slab however large the volume.
fn for_each_block(
    sample: &Sample,
    key: &str,
    stride: usize,
    physical: bool,
    mut visit: impl FnMut(&[f64]),
) -> Result<()> {
    let image = sample.image(key)?;
    let spatial = image.grid()?.spatial_shape();
    if spatial.is_empty() {
        return Ok(());
    }
    let masked = image.valid_mask()?.is_some();
    for start in (0..spatial[0]).step_by(stride.max(1)) {
        let mut roi = vec![Slice::new(start as i64, start as i64 + 1)];
        roi.extend(std::iter::repeat_n(Slice::full(), spatial.len() - 1));
        let block = image.read(Some(&roi), physical, None)?.to_f64();
        if masked {
            let valid = sample.valid_region(key, Some(&roi))?;
            visit(&select(&block, &valid)?);
        } else {
            visit(&block.iter().copied().collect::<Vec<_>>());
        }
    }
    Ok(())
}

/// `block[np.broadcast_to(valid, block.shape)]`.
fn select(block: &ArrayD<f64>, valid: &ArrayD<bool>) -> Result<Vec<f64>> {
    let mask = valid.broadcast(IxDyn(block.shape())).ok_or_else(|| {
        Error::Value(format!(
            "operands could not be broadcast together with remapped shapes [original->remapped]: {:?} and requested \
             shape {:?}",
            valid.shape(),
            block.shape()
        ))
    })?;
    Ok(block.iter().zip(mask.iter()).filter(|(_, keep)| **keep).map(|(v, _)| *v).collect())
}

/// One file, with failure recorded rather than raised.
fn guarded(path: &Path, options: &StatsOptions) -> Result<DatasetStats> {
    match stats_for(path, options) {
        Ok(s) => Ok(s),
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_)) => Ok(DatasetStats {
            failures: vec![format!("{}: {}", path.display(), e.python_str())],
            physical: options.physical,
            ..Default::default()
        }),
        Err(e) => Err(e),
    }
}

/// Stream a whole cohort, optionally across `workers` threads.
///
/// Workers return accumulators, not arrays, and the results are merged in
/// path order, so the statistics do not depend on the worker count.
pub fn compute_stats<P: AsRef<Path> + Sync>(
    paths: &[P],
    options: &StatsOptions,
    workers: usize,
) -> Result<DatasetStats> {
    let mut total = DatasetStats { physical: options.physical, ..Default::default() };
    if workers <= 1 || paths.len() <= 1 {
        for path in paths {
            total.merge(&guarded(path.as_ref(), options)?)?;
        }
        return Ok(total);
    }
    let owned: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<Result<DatasetStats>>> = (0..owned.len()).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<Option<Result<DatasetStats>>>> =
        (0..owned.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..workers.min(owned.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= owned.len() {
                    break;
                }
                let result = guarded(&owned[i], options);
                if let Ok(mut slot) = slots[i].lock() {
                    *slot = Some(result);
                }
            });
        }
    });
    for (i, slot) in slots.into_iter().enumerate() {
        results[i] = slot.into_inner().ok().flatten();
    }
    for result in results {
        let stats = result.ok_or_else(|| Error::Runtime("a statistics worker died".into()))??;
        total.merge(&stats)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_merge_is_exact_for_split_blocks() {
        let values: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.37 - 4.0).collect();
        let mut whole = Moments::default();
        whole.update(&values);
        let mut parts = Moments::default();
        parts.update(&values[..17]);
        parts.update(&values[17..]);
        assert_eq!(whole.count, parts.count);
        assert!((whole.mean - parts.mean).abs() < 1e-12);
        assert!((whole.std() - parts.std()).abs() < 1e-12);
        assert_eq!(parts.minimum, -4.0);
    }

    #[test]
    fn unseen_classes_get_no_weight() {
        let mut stats = DatasetStats::default();
        stats.classes.insert(1, ClassStats { class_id: 1, voxels: 10, present_in: 1, examined_in: 1 });
        stats.classes.insert(2, ClassStats { class_id: 2, voxels: 0, present_in: 0, examined_in: 1 });
        stats.classes.insert(3, ClassStats { class_id: 3, voxels: 30, present_in: 1, examined_in: 1 });
        let weights = stats.class_weights("inverse_frequency").unwrap();
        assert_eq!(weights.keys().copied().collect::<Vec<_>>(), vec![1, 3]);
        assert!((weights[&1] + weights[&3] - 2.0).abs() < 1e-12);
        assert!(stats.class_weights_warning().unwrap().contains("[2]"));
    }
}
