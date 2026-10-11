//! The clinical JSON Schema (`data/medh5-clinical-1.schema.json`).
//!
//! One file: its root validates the `clinical/meta` descriptor, its `$defs`
//! the logical records.  JSON Schema checks types, vocabularies and required
//! members; everything that needs the tables or the sample --- column shapes,
//! references, chains, time-bound order --- is the engine's (E8xx).

use std::sync::OnceLock;

use serde_json::Value;

use crate::document::{compile, projection_of, schema_messages};

const SCHEMA_TEXT: &str = include_str!("../../data/medh5-clinical-1.schema.json");

/// The schema's file name, as published beside the specification.
pub const SCHEMA_FILE: &str = "medh5-clinical-1.schema.json";

/// The bundled clinical schema, as text.
pub fn schema_text() -> &'static str {
    SCHEMA_TEXT
}

/// The bundled clinical schema.
pub fn schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| serde_json::from_str(SCHEMA_TEXT).expect("bundled clinical schema is valid JSON"))
}

/// The schema with its root pointed at one of its definitions.
fn rooted_at(definition: &str) -> Value {
    let mut copy = schema().clone();
    if let Value::Object(map) = &mut copy {
        map.insert("$ref".into(), Value::String(format!("#/$defs/{definition}")));
    }
    copy
}

fn descriptor_validator() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| compile(schema()))
}

fn descriptor_projection() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| compile(&projection_of(schema())))
}

fn records_validator() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| compile(&rooted_at("records")))
}

/// Messages for a descriptor (empty when valid).
pub fn validate_descriptor(doc: &Value) -> Vec<String> {
    schema_messages(descriptor_validator(), doc)
}

/// `(errors, tolerated)` for a descriptor read from a higher minor, held to
/// the projection of the schema (1.1 §2.2); see
/// [`validate_document_version`](crate::document::validate_document_version).
pub fn validate_descriptor_projection(doc: &Value) -> (Vec<String>, Vec<String>) {
    let exact = validate_descriptor(doc);
    if exact.is_empty() {
        return (exact, Vec::new());
    }
    let relaxed = schema_messages(descriptor_projection(), doc);
    let tolerated = exact.into_iter().filter(|m| !relaxed.contains(m)).collect();
    (relaxed, tolerated)
}

/// Messages for a logical-record bundle (empty when valid).
pub fn validate_records(doc: &Value) -> Vec<String> {
    schema_messages(records_validator(), doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn s3_the_descriptor_schema_requires_a_relative_origin() {
        let ok = json!({"schema": "medh5.clinical/1",
                        "clock": {"id": "c", "unit": "us", "reference": "relative", "origin_description": "baseline"}});
        assert!(validate_descriptor(&ok).is_empty());
        let no_origin =
            json!({"schema": "medh5.clinical/1", "clock": {"id": "c", "unit": "us", "reference": "relative"}});
        assert!(!validate_descriptor(&no_origin).is_empty());
        let utc = json!({"schema": "medh5.clinical/1", "clock": {"id": "c", "unit": "us", "reference": "utc"}});
        assert!(validate_descriptor(&utc).is_empty());
        let wrong_unit = json!({"schema": "medh5.clinical/1", "clock": {"id": "c", "unit": "ms", "reference": "utc"}});
        assert!(!validate_descriptor(&wrong_unit).is_empty());
    }

    #[test]
    fn s2_2_a_projection_tolerates_later_members_and_values() {
        let later = json!({"schema": "medh5.clinical/1", "future": 1,
                           "clock": {"id": "c", "unit": "us", "reference": "tai"}});
        let (errors, tolerated) = validate_descriptor_projection(&later);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tolerated.len(), 2, "{tolerated:?}");
    }

    #[test]
    fn s5_records_are_checked_by_their_definitions() {
        let records = json!({
            "clinical": {"schema": "medh5.clinical/1", "clock": {"id": "c", "unit": "us", "reference": "utc"}},
            "events": [{"event_id": "e1", "record_id": "r1", "kind": "observation", "temporal_type": "point",
                        "status": "final", "effective_start_us": [1, 1]}],
        });
        assert!(validate_records(&records).is_empty(), "{:?}", validate_records(&records));
        let bad = json!({"clinical": records["clinical"], "events": [{"event_id": "e 1"}]});
        assert!(!validate_records(&bad).is_empty());
    }
}
