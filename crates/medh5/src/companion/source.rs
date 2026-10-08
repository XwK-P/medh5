//! Source references: a locator, an optional member key, and a pin.
//!
//! A URI says where bytes are; it is not what they are.  The pinned
//! `content_id` is: a reference resolves to a sample, and the sample counts
//! only while its address is the pinned one.  Since `content_id` is a Merkle
//! root over *stored* digests, the check also recomputes the root and
//! re-verifies the clinical datasets' actual bytes --- a stored root alone
//! would miss an edit under unchanged digests (1.1 §8).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::Finding;
use crate::collection::{open_any, AnyFile};
use crate::json::repr_str;
use crate::sample::Sample;
use crate::{Error, Result};

/// One sample, standalone or a collection member, at a pinned version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRef {
    /// The id rows and reconciliation records use for this source.
    pub source_id: String,
    /// A path, relative to the manifest's directory, or absolute; `file://`
    /// URIs are accepted.
    pub uri: String,
    /// The member of a `.medh5c` collection, or `None`.
    pub sample_key: Option<String>,
    /// The sample version the reference pins.
    pub content_id: String,
    /// The sample's own `identity.subject_id`, as the manifest maps it.
    pub local_subject_id: Option<String>,
}

impl SourceRef {
    pub fn new(uri: impl Into<String>, sample_key: Option<String>, content_id: impl Into<String>) -> SourceRef {
        let uri = uri.into();
        SourceRef { source_id: String::new(), uri, sample_key, content_id: content_id.into(), local_subject_id: None }
    }

    /// A reference pinned to what `sample` is now.
    pub fn pin(uri: impl Into<String>, sample_key: Option<String>, sample: &Sample) -> Result<SourceRef> {
        let content_id = sample
            .content_id()?
            .ok_or_else(|| Error::invalid("a sample without a content_id cannot be pinned; commit it first"))?;
        let mut out = SourceRef::new(uri, sample_key, content_id);
        out.local_subject_id = Some(sample.identity()?.subject_id.clone());
        Ok(out)
    }

    /// The file the locator names, relative to `base` when it is relative.
    pub fn resolve(&self, base: Option<&Path>) -> PathBuf {
        let text = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        let path = PathBuf::from(text);
        match base {
            Some(b) if path.is_relative() => b.join(path),
            _ => path,
        }
    }

    /// `uri` or `uri::key`, for messages.
    pub fn locator(&self) -> String {
        match &self.sample_key {
            Some(k) => format!("{}::{k}", self.uri),
            None => self.uri.clone(),
        }
    }

    /// Open the sample the reference names (not yet checking the pin).
    pub fn open(&self, base: Option<&Path>) -> Result<Sample> {
        let path = self.resolve(base);
        match open_any(&path, self.sample_key.as_deref())? {
            AnyFile::Sample(s) => Ok(s),
            AnyFile::Collection(_) => Err(Error::invalid(format!(
                "{} is a collection; a source names one of its members with `sample_key`",
                repr_str(&self.uri)
            ))),
        }
    }

    /// Whether `sample` is the pinned version: its stored `content_id` is the
    /// pin, the root recomputes to it, and every clinical dataset's bytes
    /// match their digests.  `deep` verifies every dataset.  Empty when it is.
    pub fn check(&self, sample: &Sample, deep: bool) -> Result<Vec<Finding>> {
        let mut out = Vec::new();
        let at = if self.source_id.is_empty() { self.locator() } else { self.source_id.clone() };
        let stored = sample.content_id()?;
        if stored.as_deref() != Some(self.content_id.as_str()) {
            out.push(Finding::new(
                "T302",
                &at,
                format!(
                    "{} is now {}; the reference pins {} --- repin it explicitly if the change is meant",
                    self.locator(),
                    stored.as_deref().unwrap_or("unaddressed"),
                    self.content_id
                ),
            ));
            return Ok(out);
        }
        let recomputed = sample.compute_content_id()?;
        if recomputed != self.content_id {
            out.push(Finding::new(
                "T302",
                &at,
                format!("{} declares the pinned content_id but its digests recompute to {recomputed}", self.locator()),
            ));
            return Ok(out);
        }
        let targets: Option<Vec<String>> = if deep {
            None
        } else {
            let all = crate::integrity::collect_digests(&sample.root, &["index"])?;
            Some(all.keys().filter(|k| k.starts_with("clinical/")).cloned().collect())
        };
        if targets.as_ref().is_none_or(|t| !t.is_empty()) {
            let verified = sample.verify(targets.as_deref())?;
            if !verified.mismatched.is_empty() || !verified.malformed.is_empty() {
                out.push(Finding::new(
                    "T302",
                    &at,
                    format!(
                        "{}: dataset bytes no longer match their digests ({}), whatever the stored root says",
                        self.locator(),
                        verified.mismatched.iter().chain(&verified.malformed).cloned().collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
        }
        Ok(out)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "source_id": self.source_id,
            "uri": self.uri,
            "sample_key": self.sample_key,
            "content_id": self.content_id,
            "local_subject_id": self.local_subject_id,
        })
    }

    /// The `{uri, sample_key, content_id}` a cache entry records.
    pub fn to_pin_json(&self) -> Value {
        json!({"uri": self.uri, "sample_key": self.sample_key, "content_id": self.content_id})
    }

    pub fn from_json(value: &Value) -> Result<SourceRef> {
        let text = |k: &str| value.get(k).and_then(Value::as_str).map(str::to_string);
        Ok(SourceRef {
            source_id: text("source_id").unwrap_or_default(),
            uri: text("uri").ok_or_else(|| Error::invalid("a source names its `uri`"))?,
            sample_key: text("sample_key"),
            content_id: text("content_id").ok_or_else(|| Error::invalid("a source pins its `content_id`"))?,
            local_subject_id: text("local_subject_id"),
        })
    }
}
