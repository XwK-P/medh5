//! Quality records (spec §11.2).
//!
//! Status changes are **activities**, not fields with private history: the
//! audit trail is the provenance graph, so a quality record says what is true
//! now and the graph says how it got that way.

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use super::provenance::check_known;
use crate::json::{num, repr_list, repr_str};
use crate::pyval::{self, get, get_list, get_str, require, to_float, to_int, to_str};
use crate::{Error, Result};

/// The schema's `qualityRecord` properties; the object is closed.
pub const QUALITY_FIELDS: [&str; 6] = ["status", "confidence", "reviewed_by", "agreement", "issues", "edit_effort_s"];
/// `quality.status` values (§11.2).
pub const QUALITY_STATUS: [&str; 6] = ["draft", "submitted", "reviewed", "approved", "rejected", "deprecated"];
/// `issue.severity` values.
pub const ISSUE_SEVERITY: [&str; 3] = ["info", "warning", "error"];

/// One inter-rater or against-reference agreement measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Agreement {
    pub metric: String,
    pub value: f64,
    pub against: Option<String>,
    /// Class id (as a string) -> value.
    pub per_class: IndexMap<String, f64>,
}

impl Agreement {
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("metric".into(), json!(self.metric));
        out.insert("value".into(), num(self.value));
        if let Some(against) = &self.against {
            out.insert("against".into(), json!(against));
        }
        if !self.per_class.is_empty() {
            let per: Map<String, Value> = self.per_class.iter().map(|(k, v)| (k.clone(), num(*v))).collect();
            out.insert("per_class".into(), Value::Object(per));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "an agreement")?;
        let mut per_class = IndexMap::new();
        if let Some(Value::Object(map)) = get(doc, "per_class") {
            for (k, v) in map {
                per_class.insert(k.clone(), to_float(v)?);
            }
        }
        Ok(Agreement {
            metric: to_str(require(doc, "metric")?),
            value: to_float(require(doc, "value")?)?,
            against: get_str(doc, "against"),
            per_class,
        })
    }
}

/// A known defect in an annotation, recorded rather than silently tolerated.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub code: String,
    pub severity: String,
    pub class_ids: Vec<i64>,
    pub note: Option<String>,
}

impl Issue {
    /// A validated issue.
    pub fn new(
        code: impl Into<String>,
        severity: impl Into<String>,
        class_ids: Vec<i64>,
        note: Option<String>,
    ) -> Result<Self> {
        let issue = Issue { code: code.into(), severity: severity.into(), class_ids, note };
        if !ISSUE_SEVERITY.contains(&issue.severity.as_str()) {
            return Err(Error::invalid(format!(
                "issue severity {} must be one of {}",
                repr_str(&issue.severity),
                repr_list(&ISSUE_SEVERITY)
            )));
        }
        Ok(issue)
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("code".into(), json!(self.code));
        out.insert("severity".into(), json!(self.severity));
        if !self.class_ids.is_empty() {
            out.insert("class_ids".into(), json!(self.class_ids));
        }
        if let Some(note) = &self.note {
            out.insert("note".into(), json!(note));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "an issue")?;
        Issue::new(
            to_str(require(doc, "code")?),
            doc.get("severity").map(to_str).unwrap_or_else(|| "info".into()),
            get_list(doc, "class_ids").iter().map(to_int).collect::<Result<Vec<_>>>()?,
            get_str(doc, "note"),
        )
    }
}

/// What is known about an annotation's trustworthiness.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityRecord {
    pub status: String,
    pub confidence: Option<f64>,
    pub reviewed_by: Vec<String>,
    pub agreement: Vec<Agreement>,
    pub issues: Vec<Issue>,
    pub edit_effort_s: Option<f64>,
}

impl QualityRecord {
    /// A record with only a status, validated.
    pub fn new(status: impl Into<String>) -> Result<Self> {
        let record = QualityRecord {
            status: status.into(),
            confidence: None,
            reviewed_by: Vec::new(),
            agreement: Vec::new(),
            issues: Vec::new(),
            edit_effort_s: None,
        };
        record.check()?;
        Ok(record)
    }

    /// Validate the status.
    pub fn check(&self) -> Result<()> {
        if !QUALITY_STATUS.contains(&self.status.as_str()) {
            return Err(Error::invalid(format!(
                "quality status {} must be one of {}",
                repr_str(&self.status),
                repr_list(&QUALITY_STATUS)
            )));
        }
        Ok(())
    }

    /// Whether the record marks data fit for training or evaluation.
    pub fn is_usable(&self) -> bool {
        self.status == "reviewed" || self.status == "approved"
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("status".into(), json!(self.status));
        if let Some(c) = self.confidence {
            out.insert("confidence".into(), num(c));
        }
        if !self.reviewed_by.is_empty() {
            out.insert("reviewed_by".into(), json!(self.reviewed_by));
        }
        if !self.agreement.is_empty() {
            out.insert("agreement".into(), Value::Array(self.agreement.iter().map(Agreement::to_json).collect()));
        }
        if !self.issues.is_empty() {
            out.insert("issues".into(), Value::Array(self.issues.iter().map(Issue::to_json).collect()));
        }
        if let Some(e) = self.edit_effort_s {
            out.insert("edit_effort_s".into(), num(e));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a quality record")?;
        check_known(doc, &QUALITY_FIELDS, "quality record")?;
        let record = QualityRecord {
            status: to_str(require(doc, "status")?),
            confidence: get(doc, "confidence").map(to_float).transpose()?,
            reviewed_by: get_list(doc, "reviewed_by").iter().map(to_str).collect(),
            agreement: get_list(doc, "agreement").iter().map(Agreement::from_json).collect::<Result<Vec<_>>>()?,
            issues: get_list(doc, "issues").iter().map(Issue::from_json).collect::<Result<Vec<_>>>()?,
            edit_effort_s: get(doc, "edit_effort_s").map(to_float).transpose()?,
        };
        record.check()?;
        Ok(record)
    }
}

/// Parse `/meta -> quality`.
pub fn quality_from_json(doc: Option<&Value>) -> Result<IndexMap<String, QualityRecord>> {
    let mut out = IndexMap::new();
    if let Some(Value::Object(map)) = doc {
        for (k, v) in map {
            out.insert(k.clone(), QualityRecord::from_json(v)?);
        }
    }
    Ok(out)
}

/// Serialize `/meta -> quality`.
pub fn quality_to_json(records: &IndexMap<String, QualityRecord>) -> Value {
    Value::Object(records.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
}

/// A mean-Dice [`Agreement`] from per-class values.
pub fn dice_agreement(per_class: &[(i64, f64)], against: Option<String>) -> Agreement {
    let mean =
        if per_class.is_empty() { 0.0 } else { per_class.iter().map(|(_, v)| v).sum::<f64>() / per_class.len() as f64 };
    Agreement {
        metric: "dice".into(),
        value: mean,
        against,
        per_class: per_class.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}
