//! The task-and-cache companion contract (`docs/spec/task-cache-1.md`).
//!
//! A `.medh5` file is the patient source.  Training needs three more things
//! the format deliberately does not hold, each separately versioned and each
//! pinned to the exact source versions it read:
//!
//! - a **task manifest** (`medh5.task/1`, one JSON document): which sources,
//!   which subjects in which partition, which cutoffs, what may be read at a
//!   cutoff, and what the target is --- [`task`];
//! - the **row views** a task's preflight produces: per row, the event
//!   versions, links and payloads the strict policy admits, the image filling
//!   each modality slot, and the target label or its censoring --- [`view`];
//! - **feature caches** (`medh5.cache/1`, one HDF5 file): derived features
//!   with their dependency manifest and checksums --- [`cache`].
//!
//! Sources are named by [`source::SourceRef`]: a locator, an optional
//! collection member key, and the pinned `content_id` --- the locator finds
//! the bytes, the pin says which bytes count.
//!
//! Findings carry codes from [`CODES`] (T1xx manifest, T2xx identity and
//! splits, T3xx sources, T4xx caches), distinct from the format's E/W codes:
//! a task can be wrong about perfectly valid files.

pub mod cache;
pub mod source;
pub mod task;
pub mod view;

use serde_json::{json, Value};

pub use cache::{validate_cache, CacheReport, CacheWriter, FeatureCache};
pub use source::SourceRef;
pub use task::TaskManifest;
pub use view::{preflight, Preflight, RowView};

/// The companion contract's finding codes, and what each means.
pub const CODES: [(&str, &str); 19] = [
    ("T101", "the manifest is not JSON, or fails its JSON Schema"),
    ("T102", "the manifest's policy, slot or target definition is inconsistent"),
    ("T103", "a declared fingerprint does not match the manifest"),
    ("T201", "a subject is declared twice, or a row names an undeclared subject"),
    ("T202", "a subject's partition is not one of the split's partitions"),
    ("T203", "one source belongs to two subjects, or two subjects pin one sample"),
    ("T204", "a row id or a source id is not unique"),
    ("T301", "a source cannot be opened, or names a collection member that does not exist"),
    ("T302", "a source's content no longer matches its pinned content_id"),
    ("T303", "a source's identity.subject_id is not the subject the manifest maps it to"),
    ("T304", "a subject's fragments disagree about their clock, or it is not the subject's"),
    ("T305", "an event or document present in several fragments differs, or is not reconciled"),
    ("T306", "a source is a version or profile this engine cannot certify, or its clinical tables are invalid"),
    ("T401", "a cache's manifest is corrupt, or fails its schema"),
    ("T402", "a cache entry's payload is corrupt, or has the wrong dtype or shape"),
    ("T403", "a cache entry's source changed or cannot be reached: the entry is stale"),
    ("T404", "a cache was built for another task, cutoff or selection"),
    ("T405", "learned preprocessing was fitted on a split other than the task's training partition"),
    ("T406", "a cache entry is not admissible as input at the row's cutoff"),
];

/// The one-line meaning of a companion code, or `""`.
pub fn code_summary(code: &str) -> &'static str {
    CODES.iter().find(|(c, _)| *c == code).map(|(_, s)| *s).unwrap_or("")
}

/// One companion finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub code: String,
    /// Where: a JSON pointer into the manifest, a source id, a row id or a
    /// cache entry.
    pub location: String,
    pub message: String,
}

impl Finding {
    pub fn new(code: &str, location: impl Into<String>, message: impl Into<String>) -> Finding {
        Finding { code: code.into(), location: location.into(), message: message.into() }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "location": self.location,
            "message": self.message,
            "summary": code_summary(&self.code),
        })
    }

    /// `CODE location: message`.
    pub fn line(&self) -> String {
        format!("{} {}: {}", self.code, self.location, self.message)
    }
}

/// `"sha256:" + hex(sha256(bytes))`.
pub fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
}

/// The fingerprint of a JSON value: sha256 over its canonical form (1.0 §5.1).
pub fn fingerprint(value: &Value) -> String {
    sha256(crate::json::canonical(value).as_bytes())
}
