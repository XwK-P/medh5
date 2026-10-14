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

/// The local path a `file:` URI (RFC 8089) names, or `None` when `uri` is
/// not one that names a path on this machine: its host is empty or
/// `localhost`, its path is percent-decoded, and on Windows (`windows`) a
/// drive path loses its leading slash (`file:///C:/x` is `C:/x`) and another
/// host is a UNC share.  What Python's `Path.as_uri()` writes reads back as
/// the path it was made from; stripping the prefix made `file://localhost/x`
/// the relative `localhost/x`, left `%20` in a name, and joined `/C:/x` onto
/// the base.
fn file_uri_path(uri: &str, windows: bool) -> Option<String> {
    let scheme = uri.get(..5).filter(|s| s.eq_ignore_ascii_case("file:"))?;
    let rest = &uri[scheme.len()..];
    let (host, path) = match rest.strip_prefix("//") {
        Some(authority) => {
            let at = authority.find('/').unwrap_or(authority.len());
            (&authority[..at], &authority[at..])
        }
        None if rest.starts_with('/') => ("", rest),
        None => return None,
    };
    let path = percent_decode(path)?;
    let local = host.is_empty() || host.eq_ignore_ascii_case("localhost");
    if !windows {
        return local.then_some(path);
    }
    if !local {
        return Some(format!(r"\\{host}{}", path.replace('/', r"\")));
    }
    let drive = path.as_bytes();
    let drive_path = drive.len() >= 3
        && drive[0] == b'/'
        && drive[1].is_ascii_alphabetic()
        && matches!(drive[2], b':' | b'|')
        && drive.get(3).is_none_or(|c| *c == b'/');
    Some(if drive_path { format!("{}:{}", &path[1..2], &path[3..]) } else { path })
}

/// `%XX` escapes decoded, or `None` for a malformed escape or a path that
/// is not UTF-8 once decoded.
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3).filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

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
    ///
    /// Refused when the root does not bind every attested path to its bytes
    /// (1.0 §13.2): such a pin could never check clean, and a file crafted
    /// with an alias holding an attested name --- relinkable afterwards to
    /// other covered bytes under the same root --- would pin without
    /// complaint.
    pub fn pin(uri: impl Into<String>, sample_key: Option<String>, sample: &Sample) -> Result<SourceRef> {
        let content_id = sample
            .content_id()?
            .ok_or_else(|| Error::invalid("a sample without a content_id cannot be pinned; commit it first"))?;
        let mut out = SourceRef::new(uri, sample_key, content_id);
        let uncovered = crate::integrity::unattested(&sample.root)?;
        if !uncovered.is_empty() {
            return Err(Error::Integrity(format!(
                "{} cannot be pinned: no line of its content_id binds {} to {} bytes (1.0 §13.2); `medh5 validate \
                 --level integrity` says why",
                out.locator(),
                uncovered.join(", "),
                if uncovered.len() == 1 { "its" } else { "their" },
            )));
        }
        out.local_subject_id = Some(sample.identity()?.subject_id.clone());
        Ok(out)
    }

    /// The file the locator names, relative to `base` when it is relative.
    /// A `file:` URI is decoded as [`file_uri_path`] decodes it.
    pub fn resolve(&self, base: Option<&Path>) -> PathBuf {
        let path = file_uri_path(&self.uri, cfg!(windows)).map_or_else(|| PathBuf::from(&self.uri), PathBuf::from);
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
    /// pin, every dataset the root speaks for carries a digest, the root
    /// recomputes to it, and every clinical dataset's bytes match their
    /// digests.  `deep` verifies every dataset.  Empty when it is.
    ///
    /// The sample is walked once: the stored digests the root is recomputed
    /// from are the ones the bytes are checked against.  A dataset whose
    /// bytes cannot be read --- a damaged chunk --- does not match.
    pub fn check(&self, sample: &Sample, deep: bool) -> Result<Vec<Finding>> {
        let mut out = Vec::new();
        let at = if self.source_id.is_empty() { self.locator() } else { self.source_id.clone() };
        let stored = sample.content_id()?;
        if stored.as_deref() != Some(self.content_id.as_str()) {
            out.push(Finding::new(
                "T302",
                &at,
                // Only the pin: what a file off its pin stores is nothing the
                // manifest's author named (task-cache-1 Appendix A).
                format!(
                    "{} is not the version the reference pins ({}) --- repin it explicitly if the change is meant",
                    self.locator(),
                    self.content_id
                ),
            ));
            return Ok(out);
        }
        let digests = crate::integrity::collect_digests(&sample.root, &["index"])?;
        // The root covers stored digests at the paths their lines name, so a
        // dataset that carries none --- a column added later, say --- is bytes
        // no pin speaks for, and so is one reached at a path that is not its
        // own: refused before anything is recomputed (1.0 §13.2, 1.1 §8,
        // E818).  Every path through the attested groups is judged, soft links
        // followed as readers follow them: an object listed under no line ---
        // reached first through `index/`, or only through a soft link --- is
        // no more covered than one without a digest, and a name held by
        // an alias could be relinked to other covered bytes under the same
        // root.
        let uncovered = crate::integrity::unattested(&sample.root)?;
        if !uncovered.is_empty() {
            out.push(Finding::new(
                "T302",
                &at,
                format!(
                    "{}: no line of the root binds {} to {} bytes, so the pinned content_id does not cover {}",
                    self.locator(),
                    uncovered.join(", "),
                    if uncovered.len() == 1 { "its" } else { "their" },
                    if uncovered.len() == 1 { "it" } else { "them" }
                ),
            ));
            return Ok(out);
        }
        let algo = crate::integrity::digest::root_algo(&sample.root)?;
        let recomputed =
            crate::integrity::compute_content_id(&sample.root, &sample.attr_name_map()?, &algo, Some(&digests))?;
        if recomputed != self.content_id {
            out.push(Finding::new(
                "T302",
                &at,
                format!("{} declares the pinned content_id but its digests recompute to {recomputed}", self.locator()),
            ));
            return Ok(out);
        }
        // Every path through `clinical/` is its object's own, checked above,
        // so the clinical datasets are the lines under it.
        let mut failed = Vec::new();
        for (path, digest) in &digests {
            if !deep && !path.starts_with("clinical/") {
                continue;
            }
            let matches = crate::digest::parse_digest(digest).and_then(|(algo, _)| {
                let ds = sample.root.dataset(path)?;
                Ok(&crate::integrity::dataset_digest(&ds, path, &algo)? == digest)
            });
            if !matches.unwrap_or(false) {
                failed.push(path.clone());
            }
        }
        if !failed.is_empty() {
            failed.sort();
            out.push(Finding::new(
                "T302",
                &at,
                format!(
                    "{}: dataset bytes no longer match their digests ({}), whatever the stored root says",
                    self.locator(),
                    failed.join(", ")
                ),
            ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_a_file_uri_names_the_path_it_was_made_from() {
        for (uri, posix, windows) in [
            ("file:///data/a%20b.medh5", Some("/data/a b.medh5"), Some("/data/a b.medh5")),
            ("file://localhost/data/x.medh5", Some("/data/x.medh5"), Some("/data/x.medh5")),
            ("FILE:/data/%C3%A9.medh5", Some("/data/é.medh5"), Some("/data/é.medh5")),
            ("file:///C:/data/x.medh5", Some("/C:/data/x.medh5"), Some("C:/data/x.medh5")),
            ("file:///c|/x.medh5", Some("/c|/x.medh5"), Some("c:/x.medh5")),
            ("file://server/share/x.medh5", None, Some(r"\\server\share\x.medh5")),
            ("file:///bad%2", None, None),
            ("file:///bad%+1", None, None),
            ("file:relative.medh5", None, None),
            ("data/x.medh5", None, None),
        ] {
            assert_eq!(file_uri_path(uri, false).as_deref(), posix, "{uri}");
            assert_eq!(file_uri_path(uri, true).as_deref(), windows, "{uri}");
        }
        let at = |uri: &str| SourceRef::new(uri, None, "").resolve(Some(Path::new("/base")));
        if cfg!(unix) {
            assert_eq!(at("file://localhost/data/x.medh5"), PathBuf::from("/data/x.medh5"));
        }
        assert_eq!(at("x.medh5"), PathBuf::from("/base/x.medh5"));
    }
}
