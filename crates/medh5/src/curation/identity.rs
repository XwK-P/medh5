//! Identity, cohort, splits and de-identification (spec §11.4, §12).
//!
//! `subject_id` prevents the most common evaluation error in medical AI --- the
//! same patient in train and test --- and because a sample never spans
//! subjects, assigning whole files to partitions is subject-safe with no
//! further bookkeeping.  Per-occasion identifiers live on the timepoint.

use serde_json::{json, Map, Number, Value};

use super::provenance::check_timestamp;
use crate::json::{repr_list, repr_str};
use crate::pyval::{self, get, get_number, get_str, require, to_str};
use crate::{Error, Result};

pub const SEX_VALUES: [&str; 4] = ["F", "M", "O", "unknown"];
pub const LATERALITY_VALUES: [&str; 3] = ["left", "right", "bilateral"];
pub const PARTITIONS: [&str; 5] = ["train", "val", "test", "holdout", "unassigned"];
/// `identity` key recording where `sample_id` and `subject_id` came from.
pub const ID_SOURCE: &str = "id_source";
/// The [`ID_SOURCE`] value for an id `scrub` replaced with a pseudonym.
pub const PSEUDONYM_SOURCE: &str = "pseudonym";

/// Who and what the sample is about (spec §12.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Identity {
    pub sample_id: String,
    pub subject_id: String,
    pub sex: Option<String>,
    pub laterality: Option<String>,
    pub bodypart: Option<String>,
    pub extra: Map<String, Value>,
}

impl Identity {
    /// A validated identity with only the two required ids.
    pub fn new(sample_id: impl Into<String>, subject_id: impl Into<String>) -> Result<Self> {
        let id = Identity {
            sample_id: sample_id.into(),
            subject_id: subject_id.into(),
            sex: None,
            laterality: None,
            bodypart: None,
            extra: Map::new(),
        };
        id.check()?;
        Ok(id)
    }

    pub fn check(&self) -> Result<()> {
        if self.sample_id.is_empty() || self.subject_id.is_empty() {
            return Err(Error::coded("E005", "identity requires both sample_id and subject_id"));
        }
        if let Some(sex) = &self.sex {
            if !SEX_VALUES.contains(&sex.as_str()) {
                return Err(Error::coded(
                    "E005",
                    format!("sex {} must be one of {}", repr_str(sex), repr_list(&SEX_VALUES)),
                ));
            }
        }
        if let Some(lat) = &self.laterality {
            if !LATERALITY_VALUES.contains(&lat.as_str()) {
                return Err(Error::coded(
                    "E005",
                    format!("laterality {} must be one of {}", repr_str(lat), repr_list(&LATERALITY_VALUES)),
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("sample_id".into(), json!(self.sample_id));
        out.insert("subject_id".into(), json!(self.subject_id));
        for (key, value) in [("sex", &self.sex), ("laterality", &self.laterality), ("bodypart", &self.bodypart)] {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        for (k, v) in &self.extra {
            out.insert(k.clone(), v.clone());
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "identity")?;
        let known = ["sample_id", "subject_id", "sex", "laterality", "bodypart"];
        let id = Identity {
            sample_id: to_str(require(doc, "sample_id")?),
            subject_id: to_str(require(doc, "subject_id")?),
            sex: get_str(doc, "sex"),
            laterality: get_str(doc, "laterality"),
            bodypart: get_str(doc, "bodypart"),
            extra: doc
                .iter()
                .filter(|(k, _)| !known.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        id.check()?;
        Ok(id)
    }
}

/// Where the sample came from, and how to group it for splitting (§12.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cohort {
    pub dataset_id: Option<String>,
    pub site_id: Option<String>,
    pub scanner_id: Option<String>,
    pub group_id: Option<String>,
    pub acquisition_protocol: Option<String>,
    pub extra: Map<String, Value>,
}

impl Cohort {
    const KNOWN: [&'static str; 5] = ["dataset_id", "site_id", "scanner_id", "group_id", "acquisition_protocol"];

    /// `group_id` if set, else `subject_id` (§12.2).
    pub fn grouping_key<'a>(&'a self, subject_id: &'a str) -> &'a str {
        match &self.group_id {
            Some(g) if !g.is_empty() => g,
            _ => subject_id,
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        for (key, value) in [
            ("dataset_id", &self.dataset_id),
            ("site_id", &self.site_id),
            ("scanner_id", &self.scanner_id),
            ("group_id", &self.group_id),
            ("acquisition_protocol", &self.acquisition_protocol),
        ] {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        for (k, v) in &self.extra {
            out.insert(k.clone(), v.clone());
        }
        Value::Object(out)
    }

    pub fn from_json(doc: Option<&Value>) -> Result<Self> {
        let Some(doc) = doc else { return Ok(Cohort::default()) };
        if !pyval::truthy(doc) {
            return Ok(Cohort::default());
        }
        let doc = pyval::as_object(doc, "cohort")?;
        Ok(Cohort {
            dataset_id: get_str(doc, "dataset_id"),
            site_id: get_str(doc, "site_id"),
            scanner_id: get_str(doc, "scanner_id"),
            group_id: get_str(doc, "group_id"),
            acquisition_protocol: get_str(doc, "acquisition_protocol"),
            extra: doc
                .iter()
                .filter(|(k, _)| !Self::KNOWN.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        })
    }

    /// A field by name, known or extra.
    pub fn field(&self, name: &str) -> Option<Value> {
        let known = match name {
            "dataset_id" => &self.dataset_id,
            "site_id" => &self.site_id,
            "scanner_id" => &self.scanner_id,
            "group_id" => &self.group_id,
            "acquisition_protocol" => &self.acquisition_protocol,
            other => return self.extra.get(other).cloned(),
        };
        known.as_ref().map(|s| json!(s))
    }
}

/// A *claim* of split membership, not an authority (spec §12.3).
#[derive(Debug, Clone, PartialEq)]
pub struct SplitClaim {
    pub set_id: String,
    pub partition: String,
    pub fold: Option<Number>,
    pub assigned_by: Option<String>,
    pub assigned_at: Option<String>,
    pub manifest_sha256: Option<String>,
}

impl SplitClaim {
    pub fn check(&self) -> Result<()> {
        if !PARTITIONS.contains(&self.partition.as_str()) {
            return Err(Error::coded(
                "E005",
                format!("partition {} must be one of {}", repr_str(&self.partition), repr_list(&PARTITIONS)),
            ));
        }
        if let Some(at) = &self.assigned_at {
            check_timestamp(at, &format!("split {}.assigned_at", repr_str(&self.set_id)))?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("set_id".into(), json!(self.set_id));
        out.insert("partition".into(), json!(self.partition));
        if let Some(fold) = &self.fold {
            out.insert("fold".into(), Value::Number(fold.clone()));
        }
        for (key, value) in [
            ("assigned_by", &self.assigned_by),
            ("assigned_at", &self.assigned_at),
            ("manifest_sha256", &self.manifest_sha256),
        ] {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a split claim")?;
        let claim = SplitClaim {
            set_id: to_str(require(doc, "set_id")?),
            partition: to_str(require(doc, "partition")?),
            fold: get_number(doc, "fold")?,
            assigned_by: get_str(doc, "assigned_by"),
            assigned_at: get_str(doc, "assigned_at"),
            manifest_sha256: get_str(doc, "manifest_sha256"),
        };
        claim.check()?;
        Ok(claim)
    }

    /// The fold as an integer.
    pub fn fold_int(&self) -> Option<i64> {
        self.fold.as_ref().and_then(|n| n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)))
    }
}

/// Parse `/meta -> splits`.
pub fn splits_from_json(doc: Option<&Value>) -> Result<Vec<SplitClaim>> {
    match doc {
        Some(Value::Array(items)) => items.iter().map(SplitClaim::from_json).collect(),
        _ => Ok(Vec::new()),
    }
}

/// What was done to remove identifiers (spec §11.4).
#[derive(Debug, Clone, PartialEq)]
pub struct Deidentification {
    pub method: String,
    pub profile: Option<String>,
    pub date_shift_days: Option<Number>,
    pub id_mapping: Option<String>,
    pub performed_by: Option<String>,
    pub date: Option<String>,
    pub burned_in_annotation_checked: Option<bool>,
    pub extra: Map<String, Value>,
}

impl Deidentification {
    const KNOWN: [&'static str; 7] =
        ["method", "profile", "date_shift_days", "id_mapping", "performed_by", "date", "burned_in_annotation_checked"];

    pub fn check(&self) -> Result<()> {
        if let Some(date) = &self.date {
            check_timestamp(date, "deidentification.date")?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("method".into(), json!(self.method));
        if let Some(v) = &self.profile {
            out.insert("profile".into(), json!(v));
        }
        if let Some(v) = &self.date_shift_days {
            out.insert("date_shift_days".into(), Value::Number(v.clone()));
        }
        for (key, value) in
            [("id_mapping", &self.id_mapping), ("performed_by", &self.performed_by), ("date", &self.date)]
        {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        if let Some(v) = self.burned_in_annotation_checked {
            out.insert("burned_in_annotation_checked".into(), json!(v));
        }
        for (k, v) in &self.extra {
            out.insert(k.clone(), v.clone());
        }
        Value::Object(out)
    }

    /// Parse `/meta -> deidentification`; `None` for absent or empty.
    pub fn from_json(doc: Option<&Value>) -> Result<Option<Self>> {
        let Some(doc) = doc else { return Ok(None) };
        if !pyval::truthy(doc) {
            return Ok(None);
        }
        let doc = pyval::as_object(doc, "deidentification")?;
        let record = Deidentification {
            method: to_str(require(doc, "method")?),
            profile: get_str(doc, "profile"),
            date_shift_days: get_number(doc, "date_shift_days")?,
            id_mapping: get_str(doc, "id_mapping"),
            performed_by: get_str(doc, "performed_by"),
            date: get_str(doc, "date"),
            burned_in_annotation_checked: get(doc, "burned_in_annotation_checked").map(pyval::truthy),
            extra: doc
                .iter()
                .filter(|(k, _)| !Self::KNOWN.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        record.check()?;
        Ok(Some(record))
    }

    /// The date shift as an integer number of days.
    pub fn shift_days(&self) -> Option<i64> {
        self.date_shift_days.as_ref().and_then(|n| n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)))
    }
}
