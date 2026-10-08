//! The logical clinical records (1.1 §3, §5--§7): the descriptor, events,
//! documents and links, as values.
//!
//! These are what the tables *mean*.  [`table`](super::table) stores them in
//! the §4 column encoding; the JSON form here is the *logical-record*
//! interchange (`crates/medh5/data/medh5-clinical-1.schema.json`, `$defs`):
//! what `medh5 clinical augment` reads and `medh5 clinical export` writes.  In
//! JSON a pair of time bounds is one two-element array, so a half-null pair ---
//! a column-level defect (E811) --- cannot be written there at all.

use serde_json::{json, Map, Value};

use crate::json::repr_str;
use crate::{Error, Result};

/// `clinical/meta → schema`: the descriptor's own version (1.1 §3).
pub const SCHEMA: &str = "medh5.clinical/1";
/// The profile a sample declares to carry clinical context.
pub const PROFILE: &str = "clinical";
/// The group the profile's objects live in, under the sample root.
pub const GROUP: &str = "clinical";
/// The descriptor dataset, under [`GROUP`].
pub const DESCRIPTOR: &str = "meta";
/// The table of event versions.
pub const EVENTS: &str = "events";
/// The table of source documents.
pub const DOCUMENTS: &str = "documents";
/// The table of typed links.
pub const LINKS: &str = "links";
/// The member of a table holding its validity masks (§4).
pub const VALID: &str = "valid";
/// The first format version that defines the profile.
pub const MIN_VERSION: &str = "1.1";

/// `event.kind` (§5.1).
pub const EVENT_KINDS: [&str; 9] = [
    "imaging",
    "document",
    "observation",
    "diagnosis",
    "medication_order",
    "medication_administration",
    "procedure",
    "assessment",
    "other",
];
/// `event.temporal_type` (§5.2).
pub const TEMPORAL_TYPES: [&str; 4] = ["point", "interval", "static", "unknown"];
/// `event.status`: the source's status, verbatim (§5.1).
pub const STATUSES: [&str; 9] = [
    "unknown",
    "preliminary",
    "final",
    "amended",
    "entered_in_error",
    "planned",
    "in_progress",
    "completed",
    "cancelled",
];
/// `event.value_comparator` (§5.1); `eq` when a numeric value names none.
pub const COMPARATORS: [&str; 5] = ["eq", "lt", "le", "gt", "ge"];
/// `document.media_type` in this profile (§6).
pub const MEDIA_TYPES: [&str; 1] = ["text/plain"];
/// Link endpoint types (§7).
pub const ENDPOINT_TYPES: [&str; 8] =
    ["event", "document", "image", "grid", "annotation", "transform", "timepoint", "instance"];
/// Link relations (§7).
pub const RELATIONS: [&str; 6] = ["describes", "compares_with", "measures", "derived_from", "supersedes", "assesses"];
/// `clock.reference` (§3).
pub const CLOCK_REFERENCES: [&str; 3] = ["relative", "utc", "shifted_utc"];
/// The one clock unit: signed microseconds (§3).
pub const CLOCK_UNIT: &str = "us";
/// The local concept system of a lesion assessment (§7).
pub const ASSESSMENT_SYSTEM: &str = "org.medh5.assessment";
/// The lesion-presence concept (§7).
pub const LESION_PRESENCE: &str = "lesion_presence";
/// The values a lesion-presence assessment may take (§7).
pub const LESION_VALUES: [&str; 6] = ["present", "absent", "not_assessed", "outside_fov", "uncertain", "resolved"];

/// One microsecond, the clock's unit.
pub const US: i64 = 1;
/// One second, in clock units.
pub const SECOND: i64 = 1_000_000;
/// One hour, in clock units.
pub const HOUR: i64 = 3_600 * SECOND;
/// One day, in clock units.
pub const DAY: i64 = 24 * HOUR;

/// Inclusive bounds on one instant, in clock microseconds (§5.2).
///
/// An exact instant has `lo == hi`.  Day precision spans the day's instants:
/// the bounds record what the source knows, and nothing finer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bounds {
    pub lo: i64,
    pub hi: i64,
}

impl Bounds {
    pub fn new(lo: i64, hi: i64) -> Bounds {
        Bounds { lo, hi }
    }

    /// An exactly known instant.
    pub fn exact(at: i64) -> Bounds {
        Bounds { lo: at, hi: at }
    }

    /// Whether every admissible instant is at or before `t`.
    pub fn at_or_before(&self, t: i64) -> bool {
        self.hi <= t
    }

    /// Whether every admissible instant is after `t`.
    pub fn after(&self, t: i64) -> bool {
        self.lo > t
    }

    /// Whether some admissible instants are at or before `t` and some after.
    pub fn straddles(&self, t: i64) -> bool {
        self.lo <= t && self.hi > t
    }

    /// Whether the two ranges share an instant.
    pub fn overlaps(&self, other: &Bounds) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn to_json(&self) -> Value {
        json!([self.lo, self.hi])
    }

    fn from_json(value: &Value, what: &str) -> Result<Option<Bounds>> {
        match value {
            Value::Null => Ok(None),
            Value::Array(items) if items.len() == 2 => {
                let lo = items[0].as_i64();
                let hi = items[1].as_i64();
                match (lo, hi) {
                    (Some(lo), Some(hi)) => Ok(Some(Bounds { lo, hi })),
                    _ => Err(Error::coded("E811", format!("{what} must be two integers [lo, hi] in microseconds"))),
                }
            }
            _ => Err(Error::coded("E811", format!("{what} must be null or [lo, hi] in microseconds"))),
        }
    }
}

/// The subject clock every time in the profile is measured on (§3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clock {
    /// Opaque, nonempty; scoped by the dataset's subject identity namespace.
    pub id: String,
    /// Always `us`.
    pub unit: String,
    /// `relative`, `utc` or `shifted_utc`.
    pub reference: String,
    /// What a relative clock's zero is; required when `relative`.
    pub origin_description: Option<String>,
}

impl Clock {
    /// A relative clock: signed microseconds from a documented subject origin.
    pub fn relative(id: impl Into<String>, origin_description: impl Into<String>) -> Clock {
        Clock {
            id: id.into(),
            unit: CLOCK_UNIT.into(),
            reference: "relative".into(),
            origin_description: Some(origin_description.into()),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("unit".into(), json!(self.unit));
        out.insert("reference".into(), json!(self.reference));
        if let Some(o) = &self.origin_description {
            out.insert("origin_description".into(), json!(o));
        }
        Value::Object(out)
    }

    pub fn from_json(value: &Value) -> Result<Clock> {
        let map = value.as_object().ok_or_else(|| Error::coded("E802", "`clock` must be an object"))?;
        let text = |key: &str| map.get(key).and_then(Value::as_str).map(str::to_string);
        Ok(Clock {
            id: text("id").ok_or_else(|| Error::coded("E802", "`clock.id` is required"))?,
            unit: text("unit").ok_or_else(|| Error::coded("E802", "`clock.unit` is required"))?,
            reference: text("reference").ok_or_else(|| Error::coded("E802", "`clock.reference` is required"))?,
            origin_description: text("origin_description"),
        })
    }
}

/// `clinical/meta`: the profile's descriptor (§3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub schema: String,
    pub clock: Clock,
}

impl Descriptor {
    pub fn new(clock: Clock) -> Descriptor {
        Descriptor { schema: SCHEMA.into(), clock }
    }

    pub fn to_json(&self) -> Value {
        json!({"schema": self.schema, "clock": self.clock.to_json()})
    }

    /// The stored form: canonical JSON (§8, 1.0 §5.1).
    pub fn dumps(&self) -> String {
        crate::json::canonical(&self.to_json())
    }

    pub fn from_json(value: &Value) -> Result<Descriptor> {
        let map = value.as_object().ok_or_else(|| Error::coded("E802", "`clinical/meta` must hold a JSON object"))?;
        let schema = map
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::coded("E802", "`clinical/meta` names no `schema`"))?;
        let clock = map.get("clock").ok_or_else(|| Error::coded("E802", "`clinical/meta` declares no `clock`"))?;
        Ok(Descriptor { schema: schema.to_string(), clock: Clock::from_json(clock)? })
    }

    /// Parse the stored text.
    pub fn loads(text: &str) -> Result<Descriptor> {
        let value = crate::json::loads(text)
            .map_err(|e| Error::coded("E801", format!("`clinical/meta` is not valid JSON: {e}")))?;
        Descriptor::from_json(&value)
    }
}

/// One immutable version of information about the subject (§5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Event {
    pub event_id: String,
    pub record_id: String,
    pub kind: String,
    pub temporal_type: String,
    pub effective_start: Option<Bounds>,
    pub effective_end: Option<Bounds>,
    pub available: Option<Bounds>,
    pub status: String,
    pub timepoint_id: Option<String>,
    pub encounter_id: Option<String>,
    pub code_system: Option<String>,
    pub code: Option<String>,
    pub code_version: Option<String>,
    pub value_num: Option<f64>,
    pub value_comparator: Option<String>,
    pub unit: Option<String>,
    pub value_text: Option<String>,
    pub missing_reason: Option<String>,
    pub prov: Option<String>,
}

fn opt_str(map: &Map<String, Value>, key: &str, what: &str) -> Result<Option<String>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(Error::invalid(format!("{what}: `{key}` must be a string or null"))),
    }
}

fn req_str(map: &Map<String, Value>, key: &str, what: &str) -> Result<String> {
    opt_str(map, key, what)?.ok_or_else(|| Error::invalid(format!("{what}: `{key}` is required")))
}

fn closed(map: &Map<String, Value>, known: &[&str], what: &str) -> Result<()> {
    let mut unknown: Vec<&String> = map.keys().filter(|k| !known.contains(&k.as_str())).collect();
    unknown.sort();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(Error::invalid(format!("{what}: unknown field(s) {}", crate::json::repr_list(&unknown))))
    }
}

/// The JSON keys of an event record.
pub const EVENT_FIELDS: [&str; 19] = [
    "event_id",
    "record_id",
    "kind",
    "temporal_type",
    "effective_start_us",
    "effective_end_us",
    "available_us",
    "status",
    "timepoint_id",
    "encounter_id",
    "code_system",
    "code",
    "code_version",
    "value_num",
    "value_comparator",
    "unit",
    "value_text",
    "missing_reason",
    "prov",
];

impl Event {
    /// The comparator a valid `value_num` is read with: `eq` when none is named.
    pub fn comparator(&self) -> Option<&str> {
        self.value_num.map(|_| self.value_comparator.as_deref().unwrap_or("eq"))
    }

    /// Whether this is a source-backed lesion assessment (§7).
    pub fn is_lesion_assessment(&self) -> bool {
        self.code_system.as_deref() == Some(ASSESSMENT_SYSTEM) && self.code.as_deref() == Some(LESION_PRESENCE)
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("event_id".into(), json!(self.event_id));
        out.insert("record_id".into(), json!(self.record_id));
        out.insert("kind".into(), json!(self.kind));
        out.insert("temporal_type".into(), json!(self.temporal_type));
        let bounds = |b: &Option<Bounds>| b.as_ref().map(Bounds::to_json).unwrap_or(Value::Null);
        out.insert("effective_start_us".into(), bounds(&self.effective_start));
        out.insert("effective_end_us".into(), bounds(&self.effective_end));
        out.insert("available_us".into(), bounds(&self.available));
        out.insert("status".into(), json!(self.status));
        for (key, value) in [
            ("timepoint_id", &self.timepoint_id),
            ("encounter_id", &self.encounter_id),
            ("code_system", &self.code_system),
            ("code", &self.code),
            ("code_version", &self.code_version),
        ] {
            out.insert(key.into(), value.as_ref().map(|v| json!(v)).unwrap_or(Value::Null));
        }
        out.insert("value_num".into(), self.value_num.map(crate::json::num).unwrap_or(Value::Null));
        for (key, value) in [
            ("value_comparator", &self.value_comparator),
            ("unit", &self.unit),
            ("value_text", &self.value_text),
            ("missing_reason", &self.missing_reason),
            ("prov", &self.prov),
        ] {
            out.insert(key.into(), value.as_ref().map(|v| json!(v)).unwrap_or(Value::Null));
        }
        Value::Object(out)
    }

    pub fn from_json(value: &Value) -> Result<Event> {
        let map = value.as_object().ok_or_else(|| Error::invalid("an event record must be a JSON object"))?;
        let what = match map.get("event_id").and_then(Value::as_str) {
            Some(id) => format!("event {}", repr_str(id)),
            None => "event".to_string(),
        };
        closed(map, &EVENT_FIELDS, &what)?;
        let value_num = match map.get("value_num") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_f64().ok_or_else(|| Error::invalid(format!("{what}: `value_num` must be a number or null")))?,
            ),
        };
        let bounds = |key: &str| Bounds::from_json(map.get(key).unwrap_or(&Value::Null), &format!("{what}: `{key}`"));
        Ok(Event {
            event_id: req_str(map, "event_id", &what)?,
            record_id: req_str(map, "record_id", &what)?,
            kind: req_str(map, "kind", &what)?,
            temporal_type: req_str(map, "temporal_type", &what)?,
            effective_start: bounds("effective_start_us")?,
            effective_end: bounds("effective_end_us")?,
            available: bounds("available_us")?,
            status: req_str(map, "status", &what)?,
            timepoint_id: opt_str(map, "timepoint_id", &what)?,
            encounter_id: opt_str(map, "encounter_id", &what)?,
            code_system: opt_str(map, "code_system", &what)?,
            code: opt_str(map, "code", &what)?,
            code_version: opt_str(map, "code_version", &what)?,
            value_num,
            value_comparator: opt_str(map, "value_comparator", &what)?,
            unit: opt_str(map, "unit", &what)?,
            value_text: opt_str(map, "value_text", &what)?,
            missing_reason: opt_str(map, "missing_reason", &what)?,
            prov: opt_str(map, "prov", &what)?,
        })
    }
}

/// One source document: canonical, de-identified source text (§6).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Document {
    pub document_id: String,
    pub media_type: String,
    pub text: String,
    pub language: Option<String>,
    pub source_type: Option<String>,
}

/// The JSON keys of a document record.
pub const DOCUMENT_FIELDS: [&str; 5] = ["document_id", "media_type", "text", "language", "source_type"];

impl Document {
    /// A `text/plain` document.
    pub fn text(document_id: impl Into<String>, text: impl Into<String>) -> Document {
        Document {
            document_id: document_id.into(),
            media_type: "text/plain".into(),
            text: text.into(),
            language: None,
            source_type: None,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "document_id": self.document_id,
            "media_type": self.media_type,
            "text": self.text,
            "language": self.language,
            "source_type": self.source_type,
        })
    }

    pub fn from_json(value: &Value) -> Result<Document> {
        let map = value.as_object().ok_or_else(|| Error::invalid("a document record must be a JSON object"))?;
        let what = match map.get("document_id").and_then(Value::as_str) {
            Some(id) => format!("document {}", repr_str(id)),
            None => "document".to_string(),
        };
        closed(map, &DOCUMENT_FIELDS, &what)?;
        Ok(Document {
            document_id: req_str(map, "document_id", &what)?,
            media_type: opt_str(map, "media_type", &what)?.unwrap_or_else(|| "text/plain".into()),
            text: req_str(map, "text", &what)?,
            language: opt_str(map, "language", &what)?,
            source_type: opt_str(map, "source_type", &what)?,
        })
    }
}

/// One typed relationship between sample-relative objects (§7).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Link {
    pub source_type: String,
    pub source_id: String,
    pub relation: String,
    pub target_type: String,
    pub target_id: String,
    /// A half-open UTF-8 byte span `[start, end)` of a document source.
    pub source_span: Option<(u64, u64)>,
    /// The annotation an `instance` target is observed in.
    pub target_annotation_id: Option<String>,
    /// The event version supplying this relationship as clinical evidence.
    pub asserted_by_event_id: Option<String>,
}

/// The JSON keys of a link record.
pub const LINK_FIELDS: [&str; 8] = [
    "source_type",
    "source_id",
    "relation",
    "target_type",
    "target_id",
    "source_span",
    "target_annotation_id",
    "asserted_by_event_id",
];

impl Link {
    /// A link with no span, annotation or attribution.
    pub fn new(source: (&str, &str), relation: impl Into<String>, target: (&str, &str)) -> Link {
        Link {
            source_type: source.0.into(),
            source_id: source.1.into(),
            relation: relation.into(),
            target_type: target.0.into(),
            target_id: target.1.into(),
            source_span: None,
            target_annotation_id: None,
            asserted_by_event_id: None,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "source_type": self.source_type,
            "source_id": self.source_id,
            "relation": self.relation,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "source_span": self.source_span.map(|(a, b)| json!([a, b])),
            "target_annotation_id": self.target_annotation_id,
            "asserted_by_event_id": self.asserted_by_event_id,
        })
    }

    pub fn from_json(value: &Value) -> Result<Link> {
        let map = value.as_object().ok_or_else(|| Error::invalid("a link record must be a JSON object"))?;
        let what = "link";
        closed(map, &LINK_FIELDS, what)?;
        let span = match map.get("source_span") {
            None | Some(Value::Null) => None,
            Some(Value::Array(items)) if items.len() == 2 => match (items[0].as_u64(), items[1].as_u64()) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => return Err(Error::coded("E814", "link: `source_span` must be two non-negative integers")),
            },
            Some(_) => return Err(Error::coded("E814", "link: `source_span` must be null or [start, end]")),
        };
        Ok(Link {
            source_type: req_str(map, "source_type", what)?,
            source_id: req_str(map, "source_id", what)?,
            relation: req_str(map, "relation", what)?,
            target_type: req_str(map, "target_type", what)?,
            target_id: req_str(map, "target_id", what)?,
            source_span: span,
            target_annotation_id: opt_str(map, "target_annotation_id", what)?,
            asserted_by_event_id: opt_str(map, "asserted_by_event_id", what)?,
        })
    }

    /// `type:id -relation-> type:id`, for messages.
    pub fn describe(&self) -> String {
        format!("{}:{} -{}-> {}:{}", self.source_type, self.source_id, self.relation, self.target_type, self.target_id)
    }
}

/// Everything the profile holds about one sample, as logical records.
#[derive(Debug, Clone, PartialEq)]
pub struct ClinicalRecords {
    pub descriptor: Descriptor,
    pub events: Vec<Event>,
    pub documents: Vec<Document>,
    pub links: Vec<Link>,
}

impl ClinicalRecords {
    pub fn new(descriptor: Descriptor) -> ClinicalRecords {
        ClinicalRecords { descriptor, events: Vec::new(), documents: Vec::new(), links: Vec::new() }
    }

    /// The logical-record bundle: `{"clinical", "events", "documents", "links"}`.
    pub fn to_json(&self) -> Value {
        json!({
            "clinical": self.descriptor.to_json(),
            "events": self.events.iter().map(Event::to_json).collect::<Vec<_>>(),
            "documents": self.documents.iter().map(Document::to_json).collect::<Vec<_>>(),
            "links": self.links.iter().map(Link::to_json).collect::<Vec<_>>(),
        })
    }

    /// Parse a logical-record bundle (checked against its JSON Schema first).
    pub fn from_json(value: &Value) -> Result<ClinicalRecords> {
        let errors = super::schema::validate_records(value);
        if !errors.is_empty() {
            return Err(Error::coded(
                "E802",
                format!(
                    "clinical records fail their JSON Schema: {}",
                    errors.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
                ),
            ));
        }
        let map = value.as_object().ok_or_else(|| Error::invalid("clinical records must be a JSON object"))?;
        let list = |key: &str| -> Vec<Value> { map.get(key).and_then(Value::as_array).cloned().unwrap_or_default() };
        Ok(ClinicalRecords {
            descriptor: Descriptor::from_json(map.get("clinical").unwrap_or(&Value::Null))?,
            events: list("events").iter().map(Event::from_json).collect::<Result<_>>()?,
            documents: list("documents").iter().map(Document::from_json).collect::<Result<_>>()?,
            links: list("links").iter().map(Link::from_json).collect::<Result<_>>()?,
        })
    }

    pub fn event(&self, event_id: &str) -> Option<&Event> {
        self.events.iter().find(|e| e.event_id == event_id)
    }

    pub fn document(&self, document_id: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.document_id == document_id)
    }
}
