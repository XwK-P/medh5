//! The sample document: everything in `/meta` (spec §2.4).
//!
//! The 1.0 rule for where a fact lives is exact, and this module is one half of
//! it: **arrays and per-object facts live in HDF5**, as attributes on the
//! object they describe; **documents live in** `/meta`.  Nothing is mirrored.

use std::sync::OnceLock;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::curation::identity::splits_from_json;
use crate::curation::quality::{quality_from_json, quality_to_json};
use crate::curation::{Cohort, Deidentification, Identity, Provenance, QualityRecord, SplitClaim, Timeline, Timepoint};
use crate::json::{dumps_meta, repr_str};
use crate::labels::LabelSet;
use crate::{Error, Result};

/// The name of the document dataset under a sample root.
pub const META_DATASET: &str = "meta";

const SCHEMA_TEXT: &str = include_str!("../data/medh5-sample-1.0.schema.json");

/// The bundled sample-document JSON Schema, as text.
pub fn schema_text() -> &'static str {
    SCHEMA_TEXT
}

/// The bundled sample-document JSON Schema.
pub fn schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| serde_json::from_str(SCHEMA_TEXT).expect("bundled schema is valid JSON"))
}

fn validator() -> &'static jsonschema::Validator {
    static VALIDATOR: OnceLock<jsonschema::Validator> = OnceLock::new();
    VALIDATOR.get_or_init(|| {
        // Formats are annotations, not assertions, under Draft 2020-12 --- the
        // reference validator never checked `date-time` here; timestamps are
        // E604's business, checked by the curation rules.
        jsonschema::draft202012::options()
            .should_validate_formats(false)
            .build(schema())
            .expect("bundled schema compiles")
    })
}

/// Validate a document, returning human-readable messages (empty when valid).
///
/// Each message is `<path>: <message>`, the path `/`-joined from the
/// document root (`<root>` for the root itself).  Messages are sorted by path.
pub fn validate_against_schema(doc: &Value) -> Vec<String> {
    let mut errors: Vec<(Vec<String>, String)> = validator()
        .iter_errors(doc)
        .map(|err| {
            let path: Vec<String> = err
                .instance_path()
                .as_str()
                .split('/')
                .filter(|s| !s.is_empty())
                .map(|s| s.replace("~1", "/").replace("~0", "~"))
                .collect();
            let message = python_message(&err);
            (path, message)
        })
        .collect();
    errors.sort_by(|a, b| a.0.cmp(&b.0));
    errors
        .into_iter()
        .map(|(path, message)| {
            let where_ = if path.is_empty() { "<root>".to_string() } else { path.join("/") };
            format!("{where_}: {message}")
        })
        .collect()
}

/// A schema error worded as Python's `jsonschema` words it, so a report reads
/// the same from every frontend: `'x' is a required property`, not `"x" ...`.
fn python_message(err: &jsonschema::ValidationError<'_>) -> String {
    use crate::json::repr;
    use jsonschema::error::{TypeKind, ValidationErrorKind as K};
    let instance = repr(err.instance());
    let plural = |n: usize| if n == 1 { "was" } else { "were" };
    match err.kind() {
        K::Required { property } => format!("{} is a required property", repr(property)),
        K::Enum { options } => format!("{instance} is not one of {}", repr(options)),
        K::Constant { expected_value } => format!("{} was expected", repr(expected_value)),
        K::Type { kind } => {
            let names: Vec<String> = match kind {
                TypeKind::Single(t) => vec![repr_str(&t.to_string())],
                TypeKind::Multiple(set) => set.iter().map(|t| repr_str(&t.to_string())).collect(),
            };
            format!("{instance} is not of type {}", names.join(", "))
        }
        K::AdditionalProperties { unexpected } => format!(
            "Additional properties are not allowed ({} {} unexpected)",
            unexpected.iter().map(|u| repr_str(u)).collect::<Vec<_>>().join(", "),
            plural(unexpected.len())
        ),
        K::UnevaluatedProperties { unexpected } => format!(
            "Unevaluated properties are not allowed ({} {} unexpected)",
            unexpected.iter().map(|u| repr_str(u)).collect::<Vec<_>>().join(", "),
            plural(unexpected.len())
        ),
        K::Minimum { limit } => format!("{instance} is less than the minimum of {}", repr(limit)),
        K::Maximum { limit } => format!("{instance} is greater than the maximum of {}", repr(limit)),
        K::ExclusiveMinimum { limit } => format!("{instance} is less than or equal to the minimum of {}", repr(limit)),
        K::ExclusiveMaximum { limit } => {
            format!("{instance} is greater than or equal to the maximum of {}", repr(limit))
        }
        K::MinItems { limit } => {
            format!("{instance} {}", if *limit == 1 { "should be non-empty" } else { "is too short" })
        }
        K::MaxItems { limit } => {
            format!("{instance} {}", if *limit == 0 { "is expected to be empty" } else { "is too long" })
        }
        K::MinLength { limit } => {
            format!("{instance} {}", if *limit == 1 { "should be non-empty" } else { "is too short" })
        }
        K::MaxLength { limit } => {
            format!("{instance} {}", if *limit == 0 { "is expected to be empty" } else { "is too long" })
        }
        K::MinProperties { limit } => {
            format!(
                "{instance} {}",
                if *limit == 1 { "should be non-empty" } else { "does not have enough properties" }
            )
        }
        K::MaxProperties { limit } => {
            format!("{instance} {}", if *limit == 0 { "is expected to be empty" } else { "has too many properties" })
        }
        K::Pattern { pattern } => format!("{instance} does not match {}", repr_str(pattern)),
        K::UniqueItems => format!("{instance} has non-unique elements"),
        K::AnyOf { .. } | K::OneOfNotValid { .. } => format!("{instance} is not valid under any of the given schemas"),
        K::OneOfMultipleValid { .. } => format!("{instance} is valid under each of the given schemas"),
        K::Not { schema } => format!("{instance} should not be valid under {}", repr(schema)),
        K::FalseSchema => format!("False schema does not allow {instance}"),
        _ => err.to_string(),
    }
}

/// Typed access to `/meta`.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleDocument {
    pub identity: Identity,
    pub timepoints: Timeline,
    pub cohort: Cohort,
    pub label_set: Option<LabelSet>,
    pub provenance: Provenance,
    pub quality: IndexMap<String, QualityRecord>,
    pub splits: Vec<SplitClaim>,
    pub acquisition: Map<String, Value>,
    pub deidentification: Option<Deidentification>,
    pub extra: Map<String, Value>,
}

impl SampleDocument {
    /// A document with the two required members filled in.
    pub fn new(identity: Identity, timepoints: Timeline) -> Self {
        SampleDocument {
            identity,
            timepoints,
            cohort: Cohort::default(),
            label_set: None,
            provenance: Provenance::default(),
            quality: IndexMap::new(),
            splits: Vec::new(),
            acquisition: Map::new(),
            deidentification: None,
            extra: Map::new(),
        }
    }

    // -- serialization ----------------------------------------------------

    pub fn to_json(&self) -> Value {
        let mut doc = Map::new();
        doc.insert("identity".into(), self.identity.to_json());
        doc.insert("timepoints".into(), self.timepoints.to_json());
        let cohort = self.cohort.to_json();
        if cohort.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            doc.insert("cohort".into(), cohort);
        }
        if let Some(ls) = &self.label_set {
            doc.insert("label_set".into(), ls.to_json(None));
        }
        if !self.provenance.is_empty() {
            doc.insert("provenance".into(), self.provenance.to_json());
        }
        if !self.quality.is_empty() {
            doc.insert("quality".into(), quality_to_json(&self.quality));
        }
        if !self.splits.is_empty() {
            doc.insert("splits".into(), Value::Array(self.splits.iter().map(SplitClaim::to_json).collect()));
        }
        if !self.acquisition.is_empty() {
            doc.insert("acquisition".into(), Value::Object(self.acquisition.clone()));
        }
        if let Some(d) = &self.deidentification {
            doc.insert("deidentification".into(), d.to_json());
        }
        if !self.extra.is_empty() {
            doc.insert("extra".into(), Value::Object(self.extra.clone()));
        }
        Value::Object(doc)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let map = doc.as_object().ok_or_else(|| Error::Schema("`meta` must hold a JSON object".into()))?;
        let identity = match map.get("identity") {
            Some(v) => Identity::from_json(v)?,
            None => return Err(missing_member("identity")),
        };
        let timepoints = match map.get("timepoints") {
            Some(v) => Timeline::from_json(v)?,
            None => return Err(missing_member("timepoints")),
        };
        let object_or_empty = |key: &str| -> Map<String, Value> {
            match map.get(key) {
                Some(Value::Object(m)) => m.clone(),
                _ => Map::new(),
            }
        };
        Ok(SampleDocument {
            identity,
            timepoints,
            cohort: Cohort::from_json(map.get("cohort"))?,
            label_set: LabelSet::from_json(map.get("label_set"))?,
            provenance: Provenance::from_json(map.get("provenance"))?,
            quality: quality_from_json(map.get("quality"))?,
            splits: splits_from_json(map.get("splits"))?,
            acquisition: object_or_empty("acquisition"),
            deidentification: Deidentification::from_json(map.get("deidentification"))?,
            extra: object_or_empty("extra"),
        })
    }

    /// Serialize to the JSON string stored in `/meta`.
    pub fn dumps(&self) -> String {
        dumps_meta(&self.to_json())
    }

    /// Serialize with indentation.
    pub fn dumps_indented(&self, indent: usize) -> String {
        crate::json::dumps(&self.to_json(), crate::json::Style::META.with_indent(Some(indent)))
    }

    /// Parse the JSON string stored in `/meta`.
    ///
    /// The `NaN` and `Infinity` 1.x wrote read as `null`; the validator
    /// reports them (E004), because JSON has neither.
    pub fn loads(text: &str) -> Result<Self> {
        let (doc, _) =
            crate::json::loads_lenient(text).map_err(|e| Error::Schema(format!("`meta` is not valid JSON: {e}")))?;
        if !doc.is_object() {
            return Err(Error::Schema("`meta` must hold a JSON object".into()));
        }
        Self::from_json(&doc)
    }

    /// Schema messages for this document (empty when valid).
    pub fn check_schema(&self) -> Vec<String> {
        validate_against_schema(&self.to_json())
    }

    // -- convenience ------------------------------------------------------

    pub fn subject_id(&self) -> &str {
        &self.identity.subject_id
    }

    pub fn group_id(&self) -> &str {
        self.cohort.grouping_key(&self.identity.subject_id)
    }

    pub fn quality_of(&self, key: Option<&str>) -> Option<&QualityRecord> {
        key.filter(|k| !k.is_empty()).and_then(|k| self.quality.get(k))
    }

    /// Compact description for `medh5 info`.
    pub fn summary(&self) -> Value {
        let timepoints: Vec<Value> = self
            .timepoints
            .iter()
            .map(|t| {
                json!({
                    "id": t.id,
                    "index": t.index,
                    "label": t.label,
                    "days_from_baseline": t.days_from_baseline.clone().map(Value::Number).unwrap_or(Value::Null),
                })
            })
            .collect();
        let label_set = match &self.label_set {
            Some(ls) => json!({"id": ls.id, "version": ls.version, "classes": ls.len(), "form": ls.form}),
            None => Value::Null,
        };
        json!({
            "sample_id": self.identity.sample_id,
            "subject_id": self.identity.subject_id,
            "timepoints": timepoints,
            "label_set": label_set,
            "agents": self.provenance.n_agents(),
            "activities": self.provenance.n_activities(),
            "quality_records": self.quality.len(),
            "deidentified": self.deidentification.is_some(),
        })
    }
}

fn missing_member(name: &str) -> Error {
    Error::Schema(format!("sample document is missing required member {}", repr_str(name)))
}

/// Start a document with the two required members filled in.
///
/// `subject_id` defaults to `sample_id`; `timepoints` defaults to one `tp0`.
pub fn new_document(
    sample_id: &str,
    subject_id: Option<&str>,
    timepoints: Option<&[String]>,
) -> Result<SampleDocument> {
    let timeline = match timepoints {
        None => Timeline::single("tp0")?,
        Some(ids) => Timeline::new(
            ids.iter().enumerate().map(|(i, id)| Timepoint::new(id.clone(), i as i64)).collect::<Result<Vec<_>>>()?,
        )?,
    };
    Ok(SampleDocument::new(Identity::new(sample_id, subject_id.unwrap_or(sample_id))?, timeline))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_schema() {
        let doc = new_document("case_1", Some("subj"), None).unwrap();
        let text = doc.dumps();
        assert_eq!(
            text,
            r#"{"identity": {"sample_id": "case_1", "subject_id": "subj"}, "timepoints": [{"id": "tp0", "index": 0}]}"#
        );
        let back = SampleDocument::loads(&text).unwrap();
        assert_eq!(back, doc);
        assert!(doc.check_schema().is_empty());
        let bad: Value = serde_json::json!({"identity": {"sample_id": "x"}, "timepoints": []});
        let errors = validate_against_schema(&bad);
        assert!(!errors.is_empty());
    }
}
