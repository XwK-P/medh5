//! Timepoints: the sample's observation occasions (spec §3.7).
//!
//! A sample is one subject at one **or more** timepoints.  The declaration
//! lives in `/meta -> timepoints`; every grid names one, and images,
//! annotations and transforms inherit theirs rather than repeating it.
//! `days_from_baseline` rather than `date` is what models should consume: the
//! interval survives de-identification date shifting.

use regex::Regex;
use serde_json::{json, Map, Number, Value};
use std::sync::OnceLock;

use super::provenance::check_known;
use crate::ids::is_valid_id;
use crate::json::{repr_int_list, repr_list, repr_str};
use crate::pyval::{self, get_number, get_str, require, to_int, to_str};
use crate::{Error, Result};

/// Exactly the schema's `timepoint` properties; the object is closed.
pub const TIMEPOINT_FIELDS: [&str; 9] = [
    "id",
    "index",
    "label",
    "date",
    "days_from_baseline",
    "study_uid",
    "series_uids",
    "subject_age_years",
    "description",
];

fn date_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d{4}-\d{2}-\d{2}").unwrap())
}

/// One observation occasion --- in DICOM terms, usually one study.
#[derive(Debug, Clone, PartialEq)]
pub struct Timepoint {
    pub id: String,
    pub index: i64,
    pub label: Option<String>,
    pub date: Option<String>,
    pub days_from_baseline: Option<Number>,
    pub study_uid: Option<String>,
    pub series_uids: Map<String, Value>,
    pub subject_age_years: Option<Number>,
    pub description: Option<String>,
}

impl Timepoint {
    /// A validated timepoint with only its id and index.
    pub fn new(id: impl Into<String>, index: i64) -> Result<Self> {
        let tp = Timepoint {
            id: id.into(),
            index,
            label: None,
            date: None,
            days_from_baseline: None,
            study_uid: None,
            series_uids: Map::new(),
            subject_age_years: None,
            description: None,
        };
        tp.check()?;
        Ok(tp)
    }

    pub fn check(&self) -> Result<()> {
        if !is_valid_id(&self.id) {
            return Err(Error::coded(
                "E003",
                format!("timepoint id {} must match [A-Za-z0-9_.-]{{1,128}}", repr_str(&self.id)),
            ));
        }
        if self.index < 0 {
            return Err(Error::coded("E108", format!("timepoint {}: index must be >= 0", repr_str(&self.id))));
        }
        if let Some(date) = &self.date {
            if !date_pattern().is_match(date) {
                return Err(Error::coded(
                    "E604",
                    format!("timepoint {}: date {} is not ISO 8601", repr_str(&self.id), repr_str(date)),
                ));
            }
        }
        Ok(())
    }

    /// `days_from_baseline` as a float.
    pub fn days(&self) -> Option<f64> {
        self.days_from_baseline.as_ref().and_then(Number::as_f64)
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("index".into(), json!(self.index));
        if let Some(v) = &self.label {
            out.insert("label".into(), json!(v));
        }
        if let Some(v) = &self.date {
            out.insert("date".into(), json!(v));
        }
        if let Some(v) = &self.days_from_baseline {
            out.insert("days_from_baseline".into(), Value::Number(v.clone()));
        }
        if let Some(v) = &self.study_uid {
            out.insert("study_uid".into(), json!(v));
        }
        if let Some(v) = &self.subject_age_years {
            out.insert("subject_age_years".into(), Value::Number(v.clone()));
        }
        if let Some(v) = &self.description {
            out.insert("description".into(), json!(v));
        }
        if !self.series_uids.is_empty() {
            out.insert("series_uids".into(), Value::Object(self.series_uids.clone()));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a timepoint")?;
        check_known(doc, &TIMEPOINT_FIELDS, "timepoint")?;
        let tp = Timepoint {
            id: to_str(require(doc, "id")?),
            index: to_int(require(doc, "index")?)?,
            label: get_str(doc, "label"),
            date: get_str(doc, "date"),
            days_from_baseline: get_number(doc, "days_from_baseline")?,
            study_uid: get_str(doc, "study_uid"),
            series_uids: match pyval::get(doc, "series_uids") {
                Some(Value::Object(map)) => map.clone(),
                _ => Map::new(),
            },
            subject_age_years: get_number(doc, "subject_age_years")?,
            description: get_str(doc, "description"),
        };
        tp.check()?;
        Ok(tp)
    }
}

/// The sample's timepoints, in acquisition order.
#[derive(Debug, Clone, PartialEq)]
pub struct Timeline {
    points: Vec<Timepoint>,
}

impl Timeline {
    /// A validated timeline; timepoints are ordered by `index`.
    pub fn new(mut timepoints: Vec<Timepoint>) -> Result<Self> {
        timepoints.sort_by_key(|t| t.index);
        let tl = Timeline { points: timepoints };
        tl.check()?;
        Ok(tl)
    }

    /// The one-timepoint timeline a cross-sectional sample declares.
    pub fn single(timepoint_id: &str) -> Result<Self> {
        Timeline::new(vec![Timepoint::new(timepoint_id, 0)?])
    }

    /// Validate spec §3.7 rules 1 and 2 (E108).
    pub fn check(&self) -> Result<()> {
        if self.points.is_empty() {
            return Err(Error::coded("E108", "a sample must declare at least one timepoint"));
        }
        let mut ids: Vec<&str> = self.points.iter().map(|t| t.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        if ids.len() != self.points.len() {
            return Err(Error::coded("E108", "duplicate timepoint id"));
        }
        let indices: Vec<i64> = self.points.iter().map(|t| t.index).collect();
        if indices.iter().enumerate().any(|(i, v)| *v != i as i64) {
            return Err(Error::coded(
                "E108",
                format!("timepoint indices {} must be dense and start at 0", repr_int_list(&indices)),
            ));
        }
        let known: Vec<(usize, &Number)> =
            self.points.iter().enumerate().filter_map(|(i, t)| t.days_from_baseline.as_ref().map(|d| (i, d))).collect();
        for pair in known.windows(2) {
            let (i0, d0) = pair[0];
            let (i1, d1) = pair[1];
            if d1.as_f64().unwrap_or(0.0) < d0.as_f64().unwrap_or(0.0) {
                return Err(Error::coded(
                    "E108",
                    format!(
                        "days_from_baseline decreases between index {i0} and {i1} ({d0} -> {d1}); \
                         `index` must be increasing with time",
                        d0 = number_repr(d0),
                        d1 = number_repr(d1)
                    ),
                ));
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Timepoint> {
        self.points.iter()
    }

    pub fn points(&self) -> &[Timepoint] {
        &self.points
    }

    /// By position.
    pub fn at(&self, index: usize) -> Option<&Timepoint> {
        self.points.get(index)
    }

    /// By id.
    pub fn get(&self, id: &str) -> Option<&Timepoint> {
        self.points.iter().find(|t| t.id == id)
    }

    /// By id, or a `KeyError` naming the declared ids.
    pub fn by_id(&self, id: &str) -> Result<&Timepoint> {
        self.get(id).ok_or_else(|| {
            Error::Key(format!("undeclared timepoint {}; declared: {}", repr_str(id), repr_list(&self.ids())))
        })
    }

    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    pub fn ids(&self) -> Vec<String> {
        self.points.iter().map(|t| t.id.clone()).collect()
    }

    pub fn baseline(&self) -> &Timepoint {
        &self.points[0]
    }

    pub fn is_longitudinal(&self) -> bool {
        self.points.len() > 1
    }

    /// Days between two timepoints, or `None` when either lacks an interval.
    pub fn interval_days(&self, a: &str, b: &str) -> Result<Option<f64>> {
        let (da, db) = (self.by_id(a)?.days(), self.by_id(b)?.days());
        Ok(match (da, db) {
            (Some(x), Some(y)) => Some(y - x),
            _ => None,
        })
    }

    /// Resolve an id, raising an E409-coded error with a useful message.
    pub fn require(&self, timepoint_id: &str, where_: &str) -> Result<&Timepoint> {
        self.get(timepoint_id).ok_or_else(|| {
            let prefix = if where_.is_empty() { String::new() } else { format!("{where_}: ") };
            Error::coded(
                "E409",
                format!(
                    "{prefix}timepoint {} is not declared (declared: {})",
                    repr_str(timepoint_id),
                    repr_list(&self.ids())
                ),
            )
        })
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.points.iter().map(Timepoint::to_json).collect())
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        match doc {
            Value::Array(items) => Timeline::new(items.iter().map(Timepoint::from_json).collect::<Result<Vec<_>>>()?),
            other => Err(Error::Type(format!("timepoints must be a list, not {}", pyval::type_name(other)))),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("Timeline({})", repr_list(&self.ids()))
    }
}

fn number_repr(n: &Number) -> String {
    if n.is_f64() {
        crate::json::float_repr(n.as_f64().unwrap_or(0.0))
    } else {
        n.to_string()
    }
}

impl<'a> IntoIterator for &'a Timeline {
    type Item = &'a Timepoint;
    type IntoIter = std::slice::Iter<'a, Timepoint>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}
