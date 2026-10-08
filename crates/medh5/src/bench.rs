//! Reproducing the performance targets on the reader's own hardware.
//!
//! The numbers in the performance guide and §14 were measured on one machine;
//! this module is the way to re-run them, so a claim like "18x faster
//! foreground sampling" can be checked rather than believed.  A measurement
//! below target is reported as such --- a benchmark that always passes is a
//! benchmark nobody reads.  The dataloader throughput run needs PyTorch and
//! lives in the Python package.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use crate::array::{NdArray, Slice};
use crate::labels::{ClassKey, LabelClass, LabelSet};
use crate::rng::Rng;
use crate::sample::open_sample;
use crate::sample::writer::{create, GridOptions, ImageOptions};
use crate::sample::writer_annotations::{SegmentationOptions, SegmentationSource, TransformSpec};
use crate::sampling::{PatchSampler, PatchSize};
use crate::{Error, Result};

/// Metric -> (upper bound in ms, description): the performance guide's targets.
pub const TARGETS: [(&str, f64, &str); 5] = [
    ("patch_labels_ms", 10.0, "64³ patch, multi-class labels only"),
    ("foreground_sample_ms", 1.0, "foreground centre sampling"),
    ("foreground_sample_many_ms", 1.0, "foreground centre sampling, 63 classes"),
    ("meta_read_ms", 2.0, "metadata-only read"),
    ("open_to_first_patch_ms", 15.0, "full open() → first patch"),
];

/// The class count the many-class draw is held to its O(1) claim at.
pub const MANY_CLASSES: usize = 63;

fn target(name: &str) -> (Option<f64>, String) {
    TARGETS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, t, d)| (Some(*t), d.to_string()))
        .unwrap_or((None, String::new()))
}

/// One timed metric, and whether it met its target.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub target: Option<f64>,
    pub description: String,
    pub detail: Map<String, Value>,
}

impl Measurement {
    fn targeted(name: &str, value: f64, detail: Value) -> Measurement {
        let (target, description) = target(name);
        Measurement { name: name.into(), value, unit: "ms".into(), target, description, detail: as_map(detail) }
    }

    pub fn ok(&self) -> bool {
        self.target.is_none_or(|t| self.value <= t)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "value": crate::json::num(self.value),
            "unit": self.unit,
            "target": self.target.map(crate::json::num),
            "ok": self.ok(),
            "description": self.description,
            "detail": self.detail,
        })
    }

    /// A measurement reported by a frontend (`Measurement.to_json()`).
    pub fn from_json(doc: &Value) -> Result<Measurement> {
        let field = |k: &str| doc.get(k).cloned().unwrap_or(Value::Null);
        Ok(Measurement {
            name: field("name").as_str().unwrap_or_default().to_string(),
            value: field("value").as_f64().ok_or_else(|| Error::Value("a measurement needs a value".into()))?,
            unit: field("unit").as_str().unwrap_or("ms").to_string(),
            target: field("target").as_f64(),
            description: field("description").as_str().unwrap_or_default().to_string(),
            detail: as_map(field("detail")),
        })
    }
}

impl std::fmt::Display for Measurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let goal = match self.target {
            Some(t) => format!("  (target ≤ {} {})", crate::json::format_g(t, 6), self.unit),
            None => String::new(),
        };
        let mark = if self.ok() { " " } else { "!" };
        let name = format!("{:<26}", self.name);
        write!(f, "{mark} {name} {:8.3} {}{goal}", self.value, self.unit)
    }
}

fn as_map(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// Median milliseconds per call.
///
/// The median, not the mean: one page fault in twenty runs moves a mean and
/// not a median, and the question is what a dataloader gets typically.
pub fn timed(mut f: impl FnMut() -> Result<()>, repeats: usize, warmup: usize) -> Result<f64> {
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    if n == 0 {
        return Ok(f64::NAN);
    }
    Ok(if n % 2 == 1 { samples[n / 2] } else { (samples[n / 2 - 1] + samples[n / 2]) / 2.0 })
}

/// `n` windows of side `patch` over `shape`, at seeded random places.
///
/// A dataloader reads a different window every time.  Timing one window
/// over and over times HDF5's chunk cache, which holds that window's chunks
/// after the first read, rather than a read.
fn windows(shape: &[usize], patch: usize, n: usize) -> Result<Vec<Vec<Slice>>> {
    let mut rng = Rng::new(0);
    (0..n)
        .map(|_| {
            shape
                .iter()
                .map(|extent| {
                    let side = patch.min(*extent);
                    let start = rng.integer(0, (extent - side + 1) as i64)?;
                    Ok(Slice::new(start, start + side as i64))
                })
                .collect()
        })
        .collect()
}

/// Run the §14 metrics against one existing sample.
pub fn benchmark_file(path: &Path, annotation: Option<&str>, patch: usize, repeats: usize) -> Result<Vec<Measurement>> {
    let mut out = Vec::new();
    let sample = open_sample(path)?;
    let ann_id = match annotation {
        Some(a) => Some(a.to_string()),
        None => sample.annotations()?.iter().find(|(_, a)| a.kind() != "classification").map(|(n, _)| n.clone()),
    };
    let mut image_ids: Vec<String> = sample.images()?.keys().cloned().collect();
    image_ids.sort();
    let image_id =
        image_ids.first().cloned().ok_or_else(|| Error::coded("E201", "a sample must contain at least one image"))?;
    let shape = sample.image(&image_id)?.grid()?.spatial_shape();
    // One window per call, warm-up included.
    let windows = windows(&shape, patch, repeats + 3)?;
    let next = |k: &mut usize| {
        *k += 1;
        &windows[(*k - 1) % windows.len()]
    };

    if sample.is_longitudinal()? {
        out.extend(paired_measurements(&sample, repeats)?);
    }
    if let Some(ann_id) = &ann_id {
        let ann = sample.annotation(ann_id)?.clone();
        let classes: Vec<ClassKey> = ann.class_ids().iter().map(|c| ClassKey::Id(*c)).collect();
        let mut k = 0;
        let value = timed(|| ann.dense(Some(&classes), Some(next(&mut k))).map(|_| ()), repeats, 3)?;
        out.push(Measurement::targeted(
            "patch_labels_ms",
            value,
            json!({"classes": classes.len(), "kind": ann.kind()}),
        ));
        let sampler = PatchSampler::new(
            PatchSize::Scalar(patch as i64),
            "foreground",
            0.5,
            None,
            crate::sampling::ClassWeights::Named("uniform".into()),
        )?;
        let mut rng = Rng::new(0);
        let indexed = sample.index()?.contains_key(ann_id.as_str());
        let value = timed(|| sampler.draw(&sample, Some(ann_id), &mut rng, None).map(|_| ()), repeats, 3)?;
        out.push(Measurement::targeted("foreground_sample_ms", value, json!({"used_index": indexed})));
    }
    let image = sample.image(&image_id)?;
    let mut k = 0;
    let value = timed(|| image.read(Some(next(&mut k)), false, None).map(|_| ()), repeats, 3)?;
    out.push(Measurement {
        name: "image_patch_ms".into(),
        value,
        unit: "ms".into(),
        target: None,
        description: format!("{patch}³ image patch"),
        detail: as_map(json!({"image": image_id, "shape": shape})),
    });
    drop(sample);

    let value = timed(|| open_sample(path)?.document().map(|_| ()), repeats, 3)?;
    out.push(Measurement::targeted("meta_read_ms", value, json!({})));
    let mut k = 0;
    let value =
        timed(|| open_sample(path)?.image(&image_id)?.read(Some(next(&mut k)), false, None).map(|_| ()), repeats, 3)?;
    out.push(Measurement::targeted("open_to_first_patch_ms", value, json!({})));
    Ok(out)
}

/// Moving one patch centre through the transform relating two visits: the
/// read a paired dataset does once per training item.
fn paired_measurements(sample: &crate::sample::Sample, repeats: usize) -> Result<Vec<Measurement>> {
    let ids = sample.timepoints()?.ids();
    let (first, second) = (&ids[0], &ids[1]);
    let grids = sample.grids()?;
    let mut chosen = None;
    'outer: for a in grids.values() {
        for b in grids.values() {
            if a.timepoint.as_deref() == Some(first.as_str()) && b.timepoint.as_deref() == Some(second.as_str()) {
                if let (Some(fa), Some(fb)) = (&a.frame_uid, &b.frame_uid) {
                    if sample.resolve_frames(fa, fb)?.is_some() {
                        chosen = Some((a.clone(), b.clone()));
                        break 'outer;
                    }
                }
            }
        }
    }
    let Some((source, target_grid)) = chosen else { return Ok(Vec::new()) };
    let transform = sample
        .resolve_frames(source.frame_uid.as_deref().unwrap_or(""), target_grid.frame_uid.as_deref().unwrap_or(""))?
        .ok_or_else(|| Error::Runtime("the frames resolved a moment ago".into()))?;
    let centre_index: Vec<f64> = source.spatial_shape().iter().map(|n| (n / 2) as f64).collect();
    let world = source.index_to_world(&centre_index);
    let points = ArrayD::from_shape_vec(IxDyn(&[1, world.len()]), world).map_err(Error::from)?;
    let value = timed(|| transform.transform_points(&points).map(|_| ()), repeats, 3)?;
    Ok(vec![Measurement {
        name: "paired_center_ms".into(),
        value,
        unit: "ms".into(),
        target: None,
        description: "one patch centre moved between two visits".into(),
        detail: as_map(json!({
            "transform": transform.transform_id,
            "kind": transform.kind(),
            "from": source.grid_id,
            "to": target_grid.grid_id,
        })),
    }])
}

fn int16_volume(shape: &[usize], rng: &mut Rng) -> Result<NdArray> {
    let n: usize = shape.iter().product();
    let values: Vec<i16> = rng.integers(-1000, 1500, n)?.into_iter().map(|v| v as i16).collect();
    NdArray::from_vec(shape, values)
}

/// A standard normal draw (Box-Muller); the values only have to look like a
/// displacement field.
fn normal(rng: &mut Rng, sigma: f64) -> f64 {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn object(value: Value) -> Map<String, Value> {
    as_map(value)
}

/// Two visits of one subject related by a dense displacement field.
pub fn synthetic_pair(directory: &Path, shape: &[usize], codec: &str, seed: u64) -> Result<PathBuf> {
    let mut rng = Rng::new(seed);
    std::fs::create_dir_all(directory)?;
    let path = directory.join("bench-pair.medh5");
    let mut writer = create(&path, Some("bench-pair"), None, codec, &[])?;
    writer.add_timepoint("tp0", object(json!({"days_from_baseline": 0})))?;
    writer.add_timepoint("tp1", object(json!({"days_from_baseline": 90})))?;
    let dims: Vec<i64> = shape.iter().map(|v| *v as i64).collect();
    for (tp, frame) in [("tp0", "bench:frame-0"), ("tp1", "bench:frame-1")] {
        let grid = format!("g_{tp}");
        writer.add_grid(
            &grid,
            &dims,
            &[1.0, 1.0, 1.0],
            GridOptions {
                timepoint: Some(tp.into()),
                frame_uid: Some(frame.into()),
                patch_hint: Some(vec![64, 64, 64]),
                ..Default::default()
            },
        )?;
        writer.add_image(
            &format!("CT_{tp}"),
            &int16_volume(shape, &mut rng)?,
            &grid,
            "CT",
            ImageOptions {
                value_type: Some("quantitative".into()),
                value_units: Some("HU".into()),
                ..Default::default()
            },
        )?;
    }
    let mut field_shape = vec![shape.len()];
    field_shape.extend_from_slice(shape);
    let n: usize = field_shape.iter().product();
    let values: Vec<f32> = (0..n).map(|_| normal(&mut rng, 0.5) as f32).collect();
    writer.add_transform(
        "warp",
        "displacement",
        "bench:frame-0",
        "bench:frame-1",
        TransformSpec {
            field: Some(NdArray::from_vec(&field_shape, values)?),
            field_grid: Some("g_tp0".into()),
            vector_space: Some("world".into()),
            ..Default::default()
        },
    )?;
    writer.deidentification(object(json!({"method": "synthetic"})))?;
    writer.commit(true)?;
    Ok(path)
}

/// Write a sample shaped like the one the published numbers were measured on.
pub fn synthetic_sample(
    directory: &Path,
    shape: &[usize],
    classes: usize,
    codec: &str,
    index: bool,
    seed: u64,
    name: &str,
) -> Result<PathBuf> {
    let mut rng = Rng::new(seed);
    std::fs::create_dir_all(directory)?;
    let path = directory.join(name);
    let classes_of: Vec<LabelClass> = (0..classes)
        .map(|i| LabelClass::new(i as i64 + 1, format!("c{}", i + 1), format!("Class {}", i + 1)))
        .collect::<Result<_>>()?;
    let label_set = LabelSet::new("bench", classes_of, "1.0.0", vec![], vec![], "inline", None, None)?;
    let mut masks = Vec::with_capacity(classes);
    for i in 0..classes {
        let mut mask = ArrayD::from_elem(IxDyn(shape), false);
        let corner: Vec<usize> = shape
            .iter()
            .map(|n| rng.integer(0, (n - n / 4).max(1) as i64).map(|v| v as usize))
            .collect::<Result<_>>()?;
        mask.slice_each_axis_mut(|ax| {
            let (c, n) = (corner[ax.axis.index()], shape[ax.axis.index()]);
            ndarray::Slice::from(c..(c + n / 4).min(n))
        })
        .fill(true);
        masks.push((ClassKey::Id(i as i64 + 1), mask));
    }
    let mut writer = create(&path, Some("bench"), None, codec, &[])?;
    writer.label_set(label_set);
    let dims: Vec<i64> = shape.iter().map(|v| *v as i64).collect();
    writer.add_grid(
        "g",
        &dims,
        &[1.0, 1.0, 1.0],
        GridOptions { timepoint: Some("tp0".into()), patch_hint: Some(vec![64, 64, 64]), ..Default::default() },
    )?;
    writer.add_image(
        "CT",
        &int16_volume(shape, &mut rng)?,
        "g",
        "CT",
        ImageOptions { value_type: Some("quantitative".into()), value_units: Some("HU".into()), ..Default::default() },
    )?;
    writer.add_segmentation("organs", "g", SegmentationSource::Masks(masks), SegmentationOptions::default())?;
    if index {
        writer.build_index(None, None, None, 0)?;
    }
    writer.deidentification(object(json!({"method": "synthetic"})))?;
    writer.commit(true)?;
    Ok(path)
}

/// A smaller sample with many classes, indexed: the case the draw must scale to.
pub fn synthetic_many_class_sample(directory: &Path, classes: usize) -> Result<PathBuf> {
    synthetic_sample(directory, &[64, 128, 128], classes, "training", true, 20260815, "bench-many.medh5")
}

/// Foreground centre sampling on a many-class, indexed annotation.
pub fn many_class_measurement(path: &Path, patch: usize, repeats: usize) -> Result<Measurement> {
    let sample = open_sample(path)?;
    let ann_id = sample
        .annotations()?
        .iter()
        .find(|(_, a)| a.kind() != "mask")
        .map(|(n, _)| n.clone())
        .ok_or_else(|| Error::Value("the many-class sample has no annotation".into()))?;
    let classes = sample.annotation(&ann_id)?.class_ids().len();
    let sampler = PatchSampler::new(
        PatchSize::Scalar(patch as i64),
        "foreground",
        0.5,
        None,
        crate::sampling::ClassWeights::Named("uniform".into()),
    )?;
    let mut rng = Rng::new(0);
    let value = timed(|| sampler.draw(&sample, Some(&ann_id), &mut rng, None).map(|_| ()), repeats, 3)?;
    let indexed = sample.index()?.contains_key(ann_id.as_str());
    Ok(Measurement::targeted("foreground_sample_many_ms", value, json!({"classes": classes, "used_index": indexed})))
}

/// The table `medh5 bench` prints, and the verdict under it.
pub fn report(measurements: &[Measurement]) -> String {
    let mut lines: Vec<String> = measurements.iter().map(|m| m.to_string()).collect();
    let failed: Vec<&str> = measurements.iter().filter(|m| !m.ok()).map(|m| m.name.as_str()).collect();
    lines.push(String::new());
    lines.push(if failed.is_empty() {
        "all targets met".into()
    } else {
        format!("{} metric(s) below target: {}", failed.len(), failed.join(", "))
    });
    lines.join("\n")
}

/// The metrics that have a target, for documentation.
pub fn targeted_metrics() -> BTreeSet<&'static str> {
    TARGETS.iter().map(|(n, _, _)| *n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurements_print_as_1x_did() {
        let m = Measurement::targeted("meta_read_ms", 1.23456, json!({}));
        assert_eq!(m.to_string(), "  meta_read_ms                  1.235 ms  (target ≤ 2 ms)");
        let slow = Measurement::targeted("meta_read_ms", 3.0, json!({}));
        assert!(!slow.ok());
        assert!(report(&[m, slow]).ends_with("1 metric(s) below target: meta_read_ms"));
    }

    #[test]
    fn the_median_is_the_middle_of_the_sorted_runs() {
        let mut calls = 0;
        let value = timed(
            || {
                calls += 1;
                Ok(())
            },
            4,
            2,
        )
        .unwrap();
        assert_eq!(calls, 6);
        assert!(value >= 0.0);
    }
}
