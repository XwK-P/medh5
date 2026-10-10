//! Measuring agreement between two annotations (§11.2).
//!
//! Three decisions are deliberate.  **Only shared, examined classes are
//! scored**: a class one rater never looked at (§11.3) contributes no
//! measurement and comes back under `skipped`.  **Empty-on-both is not a
//! disagreement**: Dice is undefined there, reported as `None` and excluded
//! from the mean.  **Instances match by id first**: where both sides carry
//! `instance_id` (§7.4) the correspondence is stated; greedy IoU matching is
//! the fallback, and the result says which was used.
//!
//! A record is keyed the way §11.2 keys it, `per_class` by class **id**.

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis};
use serde_json::{json, Map, Value};

use crate::annotations::{Annotation, Instance};
use crate::curation::quality::Agreement;
use crate::curation::tracking::carries_instance_ids;
use crate::json::{num, repr, repr_int_tuple, repr_str};
use crate::labels::ClassKey;
use crate::numeric::mean;
use crate::{Error, Result};

pub const DEFAULT_IOU: f64 = 0.5;

/// Kinds whose objects carry an axis-aligned box that IoU can be taken over.
pub const OBJECT_KINDS: [&str; 2] = ["instances", "boxes"];

fn count_true<'a>(it: impl Iterator<Item = &'a bool>) -> u64 {
    it.filter(|v| **v).count() as u64
}

/// Sørensen--Dice, or `None` when both masks are empty.
pub fn dice(a: &ndarray::ArrayD<bool>, b: &ndarray::ArrayD<bool>) -> Option<f64> {
    let total = count_true(a.iter()) + count_true(b.iter());
    if total == 0 {
        return None;
    }
    let both = a.iter().zip(b.iter()).filter(|(x, y)| **x && **y).count();
    Some(2.0 * both as f64 / total as f64)
}

/// Intersection over union, or `None` when both masks are empty.
pub fn iou(a: &ndarray::ArrayD<bool>, b: &ndarray::ArrayD<bool>) -> Option<f64> {
    let union = a.iter().zip(b.iter()).filter(|(x, y)| **x || **y).count();
    if union == 0 {
        return None;
    }
    let both = a.iter().zip(b.iter()).filter(|(x, y)| **x && **y).count();
    Some(both as f64 / union as f64)
}

/// `np.prod` of a float64 run: the identity times each element in turn.
fn product(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(1.0, |acc, v| acc * v)
}

/// IoU of two `(S, 2)` boxes in the same space.
pub fn box_iou(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    box_iou_f64(&a.mapv(f64::from), &b.mapv(f64::from))
}

/// [`box_iou`] of float64 boxes, as a caller gives them rather than a file
/// stores them.
pub fn box_iou_f64(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let rows = a.nrows().min(b.nrows());
    let overlap = product((0..rows).map(|i| {
        let lo = a[[i, 0]].max(b[[i, 0]]);
        let hi = a[[i, 1]].min(b[[i, 1]]);
        let d = hi - lo;
        if d < 0.0 {
            0.0
        } else {
            d
        }
    }));
    if overlap == 0.0 {
        return 0.0;
    }
    let volume = |m: &Array2<f64>| product(m.outer_iter().map(|r| r[1] - r[0]));
    let union = volume(a) + volume(b) - overlap;
    if union > 0.0 {
        overlap / union
    } else {
        0.0
    }
}

/// Per-class agreement between two voxel annotations.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelAgreement {
    pub metric: String,
    pub per_class: IndexMap<String, f64>,
    /// Classes not scored: absent from one side's coverage, or empty in both.
    pub skipped: Vec<String>,
    pub against: Option<String>,
    /// `per_class` key -> class id, which is what a record is keyed by.
    pub class_ids: IndexMap<String, i64>,
}

impl VoxelAgreement {
    /// Mean over the classes that were actually comparable; `None` when none was.
    pub fn value(&self) -> Option<f64> {
        mean(&self.per_class.values().copied().collect::<Vec<_>>())
    }

    /// The [`Agreement`] a `quality` record stores (§11.2).
    pub fn to_record(&self) -> Result<Agreement> {
        let value = measured(self.value(), &self.skipped)?;
        let mut per_class = IndexMap::new();
        for (key, score) in &self.per_class {
            per_class.insert(self.class_id(key)?.to_string(), *score);
        }
        Ok(Agreement { metric: self.metric.clone(), value, against: self.against.clone(), per_class })
    }

    fn class_id(&self, key: &str) -> Result<i64> {
        if let Some(id) = self.class_ids.get(key) {
            return Ok(*id);
        }
        if !key.is_empty() && key.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(id) = key.parse() {
                return Ok(id);
            }
        }
        Err(Error::invalid(format!(
            "per-class score {} names no class id; a `quality.agreement` record is keyed by class id (§11.2)",
            repr_str(key)
        )))
    }

    /// The report, which states an undefined comparison rather than refusing.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("metric".into(), json!(self.metric));
        out.insert("value".into(), self.value().map(num).unwrap_or(Value::Null));
        if let Some(against) = &self.against {
            out.insert("against".into(), json!(against));
        }
        out.insert(
            "per_class".into(),
            Value::Object(self.per_class.iter().map(|(k, v)| (k.clone(), num(*v))).collect()),
        );
        out.insert("skipped".into(), json!(self.skipped));
        out.insert("compared".into(), json!(self.per_class.len()));
        Value::Object(out)
    }
}

/// Object-level agreement: what matched, what did not, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceAgreement {
    /// `(index in a, index in b, IoU)` for every matched pair.
    pub matched: Vec<(usize, usize, f64)>,
    pub only_in_a: Vec<usize>,
    pub only_in_b: Vec<usize>,
    pub matched_by: String,
    pub threshold: f64,
    pub against: Option<String>,
    /// `(instance_id, class in a, class in b)` --- matched but classed apart.
    pub class_mismatches: Vec<(u64, i64, i64)>,
    /// Classes whose objects were left out: not examined by both sides.
    pub skipped: Vec<String>,
}

impl InstanceAgreement {
    /// F1 over objects; `None` when neither side has an object to compare.
    pub fn value(&self) -> Option<f64> {
        let tp = self.matched.len();
        if tp + self.only_in_a.len() + self.only_in_b.len() == 0 {
            return None;
        }
        if tp == 0 {
            return Some(0.0);
        }
        let precision = tp as f64 / (tp + self.only_in_b.len()) as f64;
        let recall = tp as f64 / (tp + self.only_in_a.len()) as f64;
        Some(2.0 * precision * recall / (precision + recall))
    }

    /// Mean IoU of the matched pairs; `None` when nothing matched.
    pub fn mean_iou(&self) -> Option<f64> {
        mean(&self.matched.iter().map(|m| m.2).collect::<Vec<_>>())
    }

    /// The [`Agreement`] a `quality` record stores (§11.2), without `per_class`.
    pub fn to_record(&self) -> Result<Agreement> {
        Ok(Agreement {
            metric: "object_f1".into(),
            value: measured(self.value(), &self.skipped)?,
            against: self.against.clone(),
            per_class: IndexMap::new(),
        })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "metric": "object_f1",
            "value": self.value().map(num),
            "mean_iou": self.mean_iou().map(num),
            "matched": self.matched.iter().map(|(i, j, v)| json!([i, j, num(*v)])).collect::<Vec<_>>(),
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "matched_by": self.matched_by,
            "threshold": num(self.threshold),
            "class_mismatches": self.class_mismatches.iter().map(|(i, a, b)| json!([i, a, b])).collect::<Vec<_>>(),
            "skipped": self.skipped,
            "against": self.against,
        })
    }
}

/// Either comparison, as [`compare`] chooses it.
#[derive(Debug, Clone, PartialEq)]
pub enum Comparison {
    Voxel(VoxelAgreement),
    Instance(InstanceAgreement),
}

impl Comparison {
    pub fn value(&self) -> Option<f64> {
        match self {
            Comparison::Voxel(v) => v.value(),
            Comparison::Instance(i) => i.value(),
        }
    }

    pub fn to_record(&self) -> Result<Agreement> {
        match self {
            Comparison::Voxel(v) => v.to_record(),
            Comparison::Instance(i) => i.to_record(),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Comparison::Voxel(v) => v.to_json(),
            Comparison::Instance(i) => i.to_json(),
        }
    }
}

/// A value worth recording, or a refusal that says why there is none.
fn measured(value: Option<f64>, skipped: &[String]) -> Result<f64> {
    value.ok_or_else(|| {
        let reason = if skipped.is_empty() { String::new() } else { format!(" (not scored: {})", skipped.join(", ")) };
        Error::invalid(format!(
            "nothing was comparable, so there is no agreement to record{reason}; agreement on an empty comparison \
             is undefined, and recording 0 would report a disagreement nobody measured"
        ))
    })
}

fn repr_opt(value: Option<&str>) -> String {
    repr(&json!(value))
}

struct Pair {
    classes: Vec<i64>,
    skipped: Vec<String>,
}

/// Classes both sides committed to finding, in a stable order (§11.3).
fn classes_to_compare(a: &Annotation, b: &Annotation, classes: Option<&[ClassKey]>) -> Result<Pair> {
    let wanted: Vec<i64> = match classes {
        Some(keys) => keys.iter().map(|k| a.resolve_class(k)).collect::<Result<_>>()?,
        None => {
            let mut all: Vec<i64> = a.class_ids().iter().chain(b.class_ids()).copied().collect();
            all.sort_unstable();
            all.dedup();
            all
        }
    };
    let mut out = Pair { classes: Vec::new(), skipped: Vec::new() };
    for class_id in wanted {
        let key = ClassKey::Id(class_id);
        if a.is_annotated(&key)? && b.is_annotated(&key)? {
            out.classes.push(class_id);
        } else {
            out.skipped.push(format!("{} (not examined by both)", a.class_key(class_id)));
        }
    }
    Ok(out)
}

/// Per-class Dice or IoU between two voxel annotations on the same grid.
pub fn compare_voxel(
    a: &Annotation,
    b: &Annotation,
    metric: &str,
    classes: Option<&[ClassKey]>,
) -> Result<VoxelAgreement> {
    if metric != "dice" && metric != "iou" {
        return Err(Error::invalid(format!("unknown agreement metric {}", repr_str(metric))));
    }
    if !on_one_grid(a, b) {
        return Err(Error::coded(
            "E101",
            format!(
                "annotations {} and {} are on different grids ({} vs {}){}; resample before comparing",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(a.grid_id()),
                repr_opt(b.grid_id()),
                if one_sample(a, b) { "" } else { ACROSS_SAMPLES }
            ),
        ));
    }
    let pair = classes_to_compare(a, b, classes)?;
    // A voxel either annotation ignores (§7.7) is evidence neither for nor
    // against agreement, so it is left out of both: counted, it scored a rater
    // against voxels the other declared unexamined (U03 of the 2.0 audit).
    let ignored = match (a.ignore_region()?, b.ignore_region()?) {
        (Some(mut left), Some(right)) => {
            same_shape(&left, &right, a, b)?;
            left.zip_mut_with(&right, |l, r| *l |= *r);
            Some(left)
        }
        (left, right) => left.or(right),
    };
    let mut per_class = IndexMap::new();
    let mut ids = IndexMap::new();
    let mut skipped = pair.skipped;
    for class_id in pair.classes {
        let wanted = [ClassKey::Id(class_id)];
        let mut left = a.dense(Some(&wanted), None)?.index_axis_move(Axis(0), 0);
        let mut right = b.dense(Some(&wanted), None)?.index_axis_move(Axis(0), 0);
        same_shape(&left, &right, a, b)?;
        if let Some(ignored) = &ignored {
            same_shape(&left, ignored, a, b)?;
            left.zip_mut_with(ignored, |v, i| *v &= !*i);
            right.zip_mut_with(ignored, |v, i| *v &= !*i);
        }
        let score = if metric == "dice" { dice(&left, &right) } else { iou(&left, &right) };
        let key = a.class_key(class_id);
        match score {
            None if ignored.is_some() => skipped.push(format!("{key} (empty in both outside the ignore region)")),
            None => skipped.push(format!("{key} (empty in both)")),
            Some(s) => {
                per_class.insert(key.clone(), s);
                ids.insert(key, class_id);
            }
        }
    }
    Ok(VoxelAgreement {
        metric: metric.into(),
        per_class,
        skipped,
        against: Some(format!("annotations/{}", b.ann_id)),
        class_ids: ids,
    })
}

/// Said of two grids in two samples that do not count one lattice.
const ACROSS_SAMPLES: &str = " --- in two samples, whose grid ids are each their own, grids are one only when they \
                              share a declared frame and are one lattice in it";

/// Whether two annotations were read from one sample, whose grid ids name
/// one set of grids.
fn one_sample(a: &Annotation, b: &Annotation) -> bool {
    std::ptr::eq(a.grids(), b.grids())
}

/// Whether two annotations count voxels of one grid: within one sample, one
/// grid id; across two, grids physically comparable (§3.3 rule 4: one
/// declared `frame_uid`, one `coord_system`) that are one lattice --- shape,
/// axes, spacing, origin and direction --- in one unit.
///
/// A grid id is a sample's own name.  Two samples each calling a grid `g`
/// were compared voxel for voxel though one was shifted 100 mm, and their
/// ignore regions merged though their shapes differed, which panicked (N10
/// of the 2.0 re-audit).
fn on_one_grid(a: &Annotation, b: &Annotation) -> bool {
    if one_sample(a, b) {
        return a.grid_id() == b.grid_id();
    }
    let (Ok(ga), Ok(gb)) = (a.grid(), b.grid()) else { return false };
    ga.comparable_with(gb) && ga.is_congruent(gb, 1e-6) && ga.units == gb.units
}

/// Refuse two voxel arrays that do not cover one grid, rather than let an
/// elementwise merge of them panic (a sibling ignore mask on another grid,
/// say).
fn same_shape(x: &ArrayD<bool>, y: &ArrayD<bool>, a: &Annotation, b: &Annotation) -> Result<()> {
    if x.shape() == y.shape() {
        return Ok(());
    }
    Err(Error::coded(
        "E405",
        format!(
            "annotations {} and {} cover voxels of shapes {} and {}; voxel agreement compares one grid's voxels",
            repr_str(&a.ann_id),
            repr_str(&b.ann_id),
            repr_int_tuple(x.shape()),
            repr_int_tuple(y.shape())
        ),
    ))
}

fn space_of(ann: &Annotation) -> String {
    ann.header.space.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "index".into())
}

fn frame_of(ann: &Annotation) -> Option<String> {
    if let Some(f) = &ann.header.frame_uid {
        return Some(f.clone());
    }
    ann.grid().ok().and_then(|g| g.frame_uid.clone())
}

/// Refuse two coordinate systems that cannot be compared number for number.
fn check_same_space(a: &Annotation, b: &Annotation) -> Result<()> {
    let (space_a, space_b) = (space_of(a), space_of(b));
    if space_a != space_b {
        return Err(Error::coded(
            "E414",
            format!(
                "annotations {} and {} store boxes in different spaces ({} vs {}); convert one before comparing",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_str(&space_a),
                repr_str(&space_b)
            ),
        ));
    }
    if space_a == "world" {
        if a.grid_id().is_some() && on_one_grid(a, b) {
            return Ok(());
        }
        let (frame_a, frame_b) = (frame_of(a), frame_of(b));
        if frame_a.is_some() && frame_a == frame_b {
            return Ok(());
        }
        return Err(Error::coded(
            "E414",
            format!(
                "annotations {} and {} are in frames {} and {}; a transform is required to relate them",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(frame_a.as_deref()),
                repr_opt(frame_b.as_deref())
            ),
        ));
    }
    if !on_one_grid(a, b) {
        return Err(Error::coded(
            "E101",
            format!(
                "annotations {} and {} are on different grids ({} vs {}){}; their index coordinates count \
                 different voxels, so resample or convert before comparing",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(a.grid_id()),
                repr_opt(b.grid_id()),
                if one_sample(a, b) { "" } else { ACROSS_SAMPLES }
            ),
        ));
    }
    Ok(())
}

/// Object-level agreement between two instance-carrying annotations.
///
/// **A miss counts only where the other side looked** (§11.3): an unmatched
/// object whose class the other annotation never examined is dropped and its
/// class listed under `skipped`.
pub fn compare_instances(
    a: &Annotation,
    b: &Annotation,
    threshold: f64,
    classes: Option<&[ClassKey]>,
) -> Result<InstanceAgreement> {
    for ann in [a, b] {
        if !OBJECT_KINDS.contains(&ann.kind()) {
            return Err(Error::invalid(format!(
                "annotation {} is {}; object agreement needs objects with boxes ({}). Compare voxel annotations \
                 with `compare_voxel`",
                repr_str(&ann.ann_id),
                repr_str(ann.kind()),
                OBJECT_KINDS.join(", ")
            )));
        }
    }
    check_same_space(a, b)?;
    let pair = classes_to_compare(a, b, classes)?;
    let mut left = a.instances()?;
    let mut right = b.instances()?;
    in_the_world_of(a, b, &mut right)?;
    if let Some(keys) = classes {
        let asked: Vec<i64> = keys.iter().map(|k| a.resolve_class(k)).collect::<Result<_>>()?;
        left.retain(|o| asked.contains(&o.class_id));
        right.retain(|o| asked.contains(&o.class_id));
    }
    let ids_b: std::collections::HashSet<u64> = right.iter().map(|o| o.instance_id).collect();
    let shared: std::collections::HashSet<u64> =
        left.iter().map(|o| o.instance_id).filter(|i| ids_b.contains(i)).collect();
    let mut result = if !shared.is_empty() && carries_instance_ids(a) && carries_instance_ids(b) {
        match_by_id(&left, &right, &shared, threshold, &b.ann_id)
    } else {
        match_by_iou(&left, &right, threshold, &b.ann_id)
    };
    let class_a: std::collections::HashMap<usize, i64> = left.iter().map(|o| (o.index, o.class_id)).collect();
    let class_b: std::collections::HashMap<usize, i64> = right.iter().map(|o| (o.index, o.class_id)).collect();
    let looked_a = a.annotated_class_ids();
    let looked_b = b.annotated_class_ids();
    result.only_in_a.retain(|i| looked_b.contains(&class_a[i]));
    result.only_in_b.retain(|j| looked_a.contains(&class_b[j]));
    result.skipped = pair.skipped;
    Ok(result)
}

/// `b`'s world boxes, in `a`'s units.
///
/// A shared frame says two annotations' world is one place, not that their
/// numbers are in one unit: the same boxes in metres and in millimetres scored
/// an F1 of 0, and boxes in LPS against RAS an IoU of 1 where they were
/// disjoint (N10 of the round-3 audit) --- which E414 now refuses, since §3.3
/// rule 4 compares world coordinates in one `coord_system` only.  An
/// annotation that names no grid states neither, and is taken as it is.
fn in_the_world_of(a: &Annotation, b: &Annotation, objects: &mut [Instance]) -> Result<()> {
    if space_of(a) != "world" || on_one_grid(a, b) {
        return Ok(());
    }
    let (Some(to), Some(from)) = (a.world_grid(), b.world_grid()) else { return Ok(()) };
    let scale = from.world_scale_into(to).map_err(|e| {
        Error::coded(
            "E414",
            format!("annotations {} and {}: {}", repr_str(&a.ann_id), repr_str(&b.ann_id), e.message()),
        )
    })?;
    if scale != 1.0 {
        for object in objects {
            object.bbox.mapv_inplace(|v| (f64::from(v) * scale) as f32);
        }
    }
    Ok(())
}

fn match_by_id(
    left: &[Instance],
    right: &[Instance],
    shared: &std::collections::HashSet<u64>,
    threshold: f64,
    against: &str,
) -> InstanceAgreement {
    let by_id_b: std::collections::HashMap<u64, &Instance> = right.iter().map(|o| (o.instance_id, o)).collect();
    let mut matched = Vec::new();
    let mut mismatches = Vec::new();
    for obj in left.iter().filter(|o| shared.contains(&o.instance_id)) {
        let other = by_id_b[&obj.instance_id];
        matched.push((obj.index, other.index, box_iou(&obj.bbox, &other.bbox)));
        if obj.class_id != other.class_id {
            mismatches.push((obj.instance_id, obj.class_id, other.class_id));
        }
    }
    InstanceAgreement {
        matched,
        only_in_a: left.iter().filter(|o| !shared.contains(&o.instance_id)).map(|o| o.index).collect(),
        only_in_b: right.iter().filter(|o| !shared.contains(&o.instance_id)).map(|o| o.index).collect(),
        matched_by: "instance_id".into(),
        threshold,
        against: Some(format!("annotations/{against}")),
        class_mismatches: mismatches,
        skipped: Vec::new(),
    }
}

/// Greedy highest-IoU-first matching, one object to at most one object.
fn match_by_iou(left: &[Instance], right: &[Instance], threshold: f64, against: &str) -> InstanceAgreement {
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, obj) in left.iter().enumerate() {
        for (j, other) in right.iter().enumerate() {
            if obj.class_id != other.class_id {
                continue;
            }
            let overlap = box_iou(&obj.bbox, &other.bbox);
            if overlap >= threshold {
                candidates.push((overlap, i, j));
            }
        }
    }
    candidates.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut used_a = std::collections::HashSet::new();
    let mut used_b = std::collections::HashSet::new();
    let mut matched = Vec::new();
    for (overlap, i, j) in candidates {
        if used_a.contains(&i) || used_b.contains(&j) {
            continue;
        }
        used_a.insert(i);
        used_b.insert(j);
        matched.push((left[i].index, right[j].index, overlap));
    }
    matched.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.total_cmp(&y.2)));
    InstanceAgreement {
        matched,
        only_in_a: left.iter().enumerate().filter(|(i, _)| !used_a.contains(i)).map(|(_, o)| o.index).collect(),
        only_in_b: right.iter().enumerate().filter(|(j, _)| !used_b.contains(j)).map(|(_, o)| o.index).collect(),
        matched_by: "iou".into(),
        threshold,
        against: Some(format!("annotations/{against}")),
        class_mismatches: Vec::new(),
        skipped: Vec::new(),
    }
}

/// Compare two annotations, choosing the comparison their kinds support.
///
/// Two object-carrying annotations are compared object by object at an IoU
/// `threshold`; two voxel annotations class by class, by `metric`.  An
/// argument the chosen comparison cannot use is refused rather than ignored.
pub fn compare(
    a: &Annotation,
    b: &Annotation,
    metric: Option<&str>,
    threshold: Option<f64>,
    classes: Option<&[ClassKey]>,
) -> Result<Comparison> {
    if OBJECT_KINDS.contains(&a.kind()) && OBJECT_KINDS.contains(&b.kind()) {
        if let Some(m) = metric {
            return Err(Error::invalid(format!(
                "metric {} scores voxels; {} and {} are compared object by object, as F1 at an IoU threshold",
                repr_str(m),
                repr_str(&a.ann_id),
                repr_str(&b.ann_id)
            )));
        }
        return Ok(Comparison::Instance(compare_instances(a, b, threshold.unwrap_or(DEFAULT_IOU), classes)?));
    }
    if a.is_voxel() && b.is_voxel() {
        if threshold.is_some() {
            return Err(Error::invalid(format!(
                "threshold matches objects; {} and {} are compared voxel by voxel",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id)
            )));
        }
        return Ok(Comparison::Voxel(compare_voxel(a, b, metric.unwrap_or("dice"), classes)?));
    }
    Err(Error::invalid(format!(
        "cannot compare {} with {}: transcode them to a common kind first",
        repr_str(a.kind()),
        repr_str(b.kind())
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn box_iou_is_zero_without_overlap_and_one_for_identity() {
        let a: Array2<f32> = array![[0.0, 2.0], [0.0, 2.0]];
        let b: Array2<f32> = array![[2.0, 4.0], [0.0, 2.0]];
        assert_eq!(box_iou(&a, &b), 0.0);
        assert_eq!(box_iou(&a, &a), 1.0);
        let c: Array2<f32> = array![[1.0, 3.0], [0.0, 2.0]];
        assert_eq!(box_iou(&a, &c), 2.0 / 6.0);
    }
}
