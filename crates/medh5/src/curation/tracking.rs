//! Longitudinal joins on `instance_id` (§7.4, §11.3).
//!
//! Tracking is a **join, not a structure**: grouping objects on
//! `instance_id` recovers the lesion a radiologist followed across visits.
//! Absence is not a measurement --- it is `resolved` only where the class was
//! looked for (`annotated_class_ids`), and `unexamined` otherwise.  A track
//! carrying several class ids is reported (W909), never resolved by a rule.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use ndarray::Array2;
use serde_json::{json, Map, Value};

use crate::annotations::{Annotation, Instance};
use crate::geometry::affine::{box_to_slices, voxel_volume};
use crate::geometry::grid::Grid;
use crate::json::{format_g, num};
use crate::labels::ClassKey;
use crate::sample::Sample;
use crate::Result;

pub const PRESENT: &str = "present";
pub const RESOLVED: &str = "resolved";
pub const UNEXAMINED: &str = "unexamined";
pub const STATES: [&str; 3] = [PRESENT, RESOLVED, UNEXAMINED];

/// One object seen once: a row of one annotation.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub timepoint: String,
    pub annotation: String,
    pub index: usize,
    pub instance_id: u64,
    pub class_id: i64,
    pub bbox: Array2<f32>,
    pub voxel_count: Option<u64>,
    /// Physical volume in the grid's `units**S`.
    pub volume: Option<f64>,
    pub units: Option<String>,
    pub score: Option<f64>,
    pub grid: Option<String>,
}

impl Observation {
    /// Box centre, in whatever space the annotation stores its boxes.
    pub fn centroid(&self) -> Vec<f64> {
        self.bbox.outer_iter().map(|r| (f64::from(r[0]) + f64::from(r[1])) / 2.0).collect()
    }

    pub fn extent(&self) -> Vec<f64> {
        self.bbox.outer_iter().map(|r| f64::from(r[1]) - f64::from(r[0])).collect()
    }

    pub fn to_json(&self) -> Value {
        let rows: Vec<Value> =
            self.bbox.outer_iter().map(|r| json!(r.iter().map(|v| num(f64::from(*v))).collect::<Vec<_>>())).collect();
        json!({
            "timepoint": self.timepoint,
            "annotation": self.annotation,
            "index": self.index,
            "class_id": self.class_id,
            "box": rows,
            "voxel_count": self.voxel_count,
            "volume": self.volume.map(num),
            "units": self.units,
            "score": self.score.map(num),
        })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let volume = self.volume.map(|v| format_g(v, 4)).unwrap_or_else(|| "-".into());
        format!(
            "Observation({}[{}] @{}, class={}, volume={volume})",
            self.annotation, self.index, self.timepoint, self.class_id
        )
    }
}

/// Every observation of one physical object, ordered by timepoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub instance_id: u64,
    pub class_ids: Vec<i64>,
    pub observations: Vec<Observation>,
    pub class_key: Option<String>,
}

impl Track {
    /// The object's class.  See [`Track::has_class_conflict`] first.
    pub fn class_id(&self) -> i64 {
        self.class_ids[0]
    }

    /// Whether this id carries more than one class id (W909).
    pub fn has_class_conflict(&self) -> bool {
        self.class_ids.len() > 1
    }

    pub fn timepoints(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for o in &self.observations {
            if !out.contains(&o.timepoint) {
                out.push(o.timepoint.clone());
            }
        }
        out
    }

    pub fn at(&self, timepoint: &str) -> Option<&Observation> {
        self.observations.iter().find(|o| o.timepoint == timepoint)
    }

    pub fn volume(&self, timepoint: &str) -> Option<f64> {
        self.at(timepoint).and_then(|o| o.volume)
    }

    pub fn volumes(&self) -> IndexMap<String, Option<f64>> {
        self.observations.iter().map(|o| (o.timepoint.clone(), o.volume)).collect()
    }

    /// `(v2 - v1) / v1` between two timepoints, or `None` if unmeasured.
    pub fn relative_change(&self, first: &str, second: &str) -> Option<f64> {
        let before = self.volume(first)?;
        let after = self.volume(second)?;
        if before <= 0.0 {
            return None;
        }
        Some((after - before) / before)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "instance_id": self.instance_id,
            "class_ids": self.class_ids,
            "class_key": self.class_key,
            "timepoints": self.timepoints(),
            "observations": self.observations.iter().map(Observation::to_json).collect::<Vec<_>>(),
        })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let class = self.class_key.clone().unwrap_or_else(|| self.class_id().to_string());
        format!("Track({}, class={class}, seen at {})", self.instance_id, crate::json::repr_list(&self.timepoints()))
    }
}

/// The result of joining objects on `instance_id` across a sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Tracking {
    pub tracks: BTreeMap<u64, Track>,
    pub timepoints: Vec<String>,
    /// `timepoint -> class ids the annotators committed to finding` (§11.3).
    pub coverage: IndexMap<String, BTreeSet<i64>>,
}

impl Tracking {
    /// `present`, `resolved` or `unexamined`.
    pub fn state_at(&self, instance_id: u64, timepoint: &str) -> Option<&'static str> {
        let track = self.tracks.get(&instance_id)?;
        if track.at(timepoint).is_some() {
            return Some(PRESENT);
        }
        let examined = self.coverage.get(timepoint);
        let looked = examined.is_some_and(|e| track.class_ids.iter().any(|c| e.contains(c)));
        Some(if looked { RESOLVED } else { UNEXAMINED })
    }

    pub fn states(&self, instance_id: u64) -> IndexMap<String, &'static str> {
        self.timepoints.iter().map(|tp| (tp.clone(), self.state_at(instance_id, tp).unwrap_or(UNEXAMINED))).collect()
    }

    /// Absent-then-present: not seen at baseline, seen later.
    pub fn is_new(&self, instance_id: u64) -> bool {
        let states = self.states(instance_id);
        match self.timepoints.first() {
            None => false,
            Some(first) => states[first] == RESOLVED && states.values().any(|s| *s == PRESENT),
        }
    }

    /// Present-then-gone, where the later visit did look for it.
    pub fn is_resolved(&self, instance_id: u64) -> bool {
        if self.timepoints.len() < 2 {
            return false;
        }
        let states = self.states(instance_id);
        states[&self.timepoints[0]] == PRESENT && states[&self.timepoints[self.timepoints.len() - 1]] == RESOLVED
    }

    pub fn is_persistent(&self, instance_id: u64) -> bool {
        self.states(instance_id).values().all(|s| *s == PRESENT)
    }

    /// Instance ids carrying more than one class id (W909).
    pub fn class_conflicts(&self) -> BTreeMap<u64, Vec<i64>> {
        self.tracks.iter().filter(|(_, t)| t.has_class_conflict()).map(|(i, t)| (*i, t.class_ids.clone())).collect()
    }

    /// `timepoint -> instance ids whose class nobody committed to finding`.
    pub fn unexamined(&self) -> IndexMap<String, Vec<u64>> {
        let mut out = IndexMap::new();
        for tp in &self.timepoints {
            let ids: Vec<u64> =
                self.tracks.keys().copied().filter(|i| self.state_at(*i, tp) == Some(UNEXAMINED)).collect();
            if !ids.is_empty() {
                out.insert(tp.clone(), ids);
            }
        }
        out
    }

    pub fn repr(&self) -> String {
        format!("Tracking({} tracks over {} timepoints)", self.tracks.len(), self.timepoints.len())
    }

    pub fn to_json(&self) -> Value {
        let mut coverage: Vec<(&String, &BTreeSet<i64>)> = self.coverage.iter().collect();
        coverage.sort_by(|a, b| a.0.cmp(b.0));
        let coverage: Map<String, Value> = coverage.into_iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let tracks: Vec<Value> = self
            .tracks
            .iter()
            .map(|(i, t)| {
                let mut v = t.to_json();
                if let Value::Object(m) = &mut v {
                    m.insert(
                        "states".into(),
                        Value::Object(self.states(*i).into_iter().map(|(k, v)| (k, json!(v))).collect()),
                    );
                }
                v
            })
            .collect();
        let conflicts: Map<String, Value> =
            self.class_conflicts().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        json!({"timepoints": self.timepoints, "coverage": coverage, "tracks": tracks, "class_conflicts": conflicts})
    }

    pub fn summary(&self) -> Value {
        let pick = |f: &dyn Fn(u64) -> bool| -> Vec<u64> { self.tracks.keys().copied().filter(|i| f(*i)).collect() };
        json!({
            "tracks": self.tracks.len(),
            "timepoints": self.timepoints,
            "new": pick(&|i| self.is_new(i)),
            "resolved": pick(&|i| self.is_resolved(i)),
            "persistent": pick(&|i| self.is_persistent(i)),
            "class_conflicts": self.class_conflicts().keys().copied().collect::<Vec<_>>(),
        })
    }
}

/// Whether an annotation declares object identity a join can trust: a
/// `boxes` annotation without `instance_ids` numbers its rows positionally.
pub fn carries_instance_ids(ann: &Annotation) -> bool {
    ann.kind() == "instances" || ann.optional_dataset("instance_ids").is_some()
}

fn measure(obj: &Instance, grid: Option<&Grid>, ann: &Annotation) -> Result<(Option<f64>, Option<u64>)> {
    let space = ann.header.space.clone().unwrap_or_else(|| "index".into());
    if let Some(mask) = &obj.mask {
        let count = mask.iter().filter(|v| **v).count() as u64;
        return Ok(match grid {
            None => (None, Some(count)),
            Some(g) => (Some(count as f64 * voxel_volume(&g.spacing)), Some(count)),
        });
    }
    if obj.bbox.ncols() != 2 {
        return Ok((None, None));
    }
    let extent: Vec<f64> = obj.bbox.outer_iter().map(|r| f64::from(r[1]) - f64::from(r[0])).collect();
    if space == "world" {
        return Ok((Some(extent.iter().product()), None));
    }
    let Some(g) = grid else { return Ok((None, None)) };
    if ann.kind() == "instances" {
        let flat: Vec<f64> = obj.bbox.iter().map(|v| f64::from(*v)).collect();
        let counted: i64 = box_to_slices(&flat, None)?.iter().map(|(a, b)| b - a).product();
        return Ok((Some(counted as f64 * voxel_volume(&g.spacing)), None));
    }
    Ok((Some(extent.iter().zip(&g.spacing).map(|(e, s)| e * s).product()), None))
}

/// Join every instance-carrying annotation in `sample` on `instance_id`.
pub fn build_tracks(sample: &Sample, class_key: Option<&ClassKey>, do_measure: bool) -> Result<Tracking> {
    let declared = sample.timepoints()?.ids();
    let mut coverage: IndexMap<String, BTreeSet<i64>> = declared.iter().map(|t| (t.clone(), BTreeSet::new())).collect();
    let mut tracks: BTreeMap<u64, Vec<Observation>> = BTreeMap::new();
    let implicit: Vec<String> = if declared.len() == 1 { declared.clone() } else { vec![String::new()] };
    for ann in sample.annotations()?.values() {
        if !carries_instance_ids(ann) {
            continue;
        }
        let wanted = match class_key {
            Some(k) => Some(ann.resolve_class(k)?),
            None => None,
        };
        let own = ann.timepoints();
        let timepoints = if own.is_empty() { implicit.clone() } else { own };
        for tp in &timepoints {
            coverage.entry(tp.clone()).or_default().extend(ann.annotated_class_ids().iter().copied());
        }
        let grid = ann.grid_id().and_then(|g| sample.grids().ok().and_then(|gs| gs.get(g).cloned()));
        for obj in ann.instances()? {
            if wanted.is_some_and(|w| obj.class_id != w) {
                continue;
            }
            let (volume, count) = if do_measure { measure(&obj, grid.as_ref(), ann)? } else { (None, None) };
            for tp in &timepoints {
                tracks.entry(obj.instance_id).or_default().push(Observation {
                    timepoint: tp.clone(),
                    annotation: ann.ann_id.clone(),
                    index: obj.index,
                    instance_id: obj.instance_id,
                    class_id: obj.class_id,
                    bbox: obj.bbox.clone(),
                    voxel_count: count,
                    volume,
                    units: grid.as_ref().map(|g| g.units.clone()),
                    score: obj.score,
                    grid: ann.grid_id().map(str::to_string),
                });
            }
        }
    }
    let order: IndexMap<&String, usize> = declared.iter().enumerate().map(|(i, t)| (t, i)).collect();
    let label_set = sample.label_set()?;
    let mut built = BTreeMap::new();
    for (instance_id, mut observations) in tracks {
        observations.sort_by(|a, b| {
            let ka = (order.get(&a.timepoint).copied().unwrap_or(1 << 30), a.annotation.clone());
            let kb = (order.get(&b.timepoint).copied().unwrap_or(1 << 30), b.annotation.clone());
            ka.cmp(&kb)
        });
        let class_ids: Vec<i64> =
            observations.iter().map(|o| o.class_id).collect::<BTreeSet<_>>().into_iter().collect();
        let class_key = label_set.and_then(|ls| ls.by_id(class_ids[0])).map(|c| c.key.clone());
        built.insert(instance_id, Track { instance_id, class_ids, observations, class_key });
    }
    Ok(Tracking { tracks: built, timepoints: declared, coverage })
}
