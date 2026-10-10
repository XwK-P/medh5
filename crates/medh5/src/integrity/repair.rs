//! `medh5 fix` --- rebuilding what is derived, restamping what is claimed.
//!
//! **Rebuilding an index** recomputes a cache from the data it caches
//! (§14.3); a stale index is a performance bug and rebuilding it is repair.
//!
//! **Rewriting digests** is not repair: a digest that no longer matches is
//! evidence that the bytes changed, and recomputing it destroys the evidence.
//! So it needs a reason, which is recorded as a provenance activity naming
//! what was restamped and that the content was *not* verified.

use std::path::Path;

use serde_json::{json, Map, Value};

use super::{stale_index_entries, stamp_digests};
use crate::json::repr_str;
use crate::sample::{amend, open_sample};
use crate::{Error, Result, VERSION};

/// What is wrong with one file, before anything is done about it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Diagnosis {
    pub path: String,
    pub mismatched: Vec<String>,
    pub undigested: Vec<String>,
    pub unattested: Vec<String>,
    pub stale_index: Vec<String>,
    pub missing_index: Vec<String>,
    pub content_id_ok: Option<bool>,
}

impl Diagnosis {
    /// A *stale* index is a defect; an absent one is a choice.
    pub fn needs_index(&self) -> bool {
        !self.stale_index.is_empty()
    }

    pub fn needs_digests(&self) -> bool {
        !self.mismatched.is_empty() || !self.unattested.is_empty() || self.content_id_ok == Some(false)
    }

    pub fn clean(&self) -> bool {
        !self.needs_index() && !self.needs_digests()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "mismatched": self.mismatched,
            "undigested": self.undigested,
            "unattested": self.unattested,
            "stale_index": self.stale_index,
            "missing_index": self.missing_index,
            "content_id_ok": self.content_id_ok,
            "needs_index": self.needs_index(),
            "needs_digests": self.needs_digests(),
        })
    }
}

/// What was actually done to one file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Repair {
    pub path: String,
    pub diagnosis: Diagnosis,
    pub rebuilt_index: Vec<String>,
    pub rewrote_digests: bool,
    pub content_id: Option<String>,
    pub notes: Vec<String>,
}

impl Repair {
    pub fn changed(&self) -> bool {
        !self.rebuilt_index.is_empty() || self.rewrote_digests
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "diagnosis": self.diagnosis.to_json(),
            "rebuilt_index": self.rebuilt_index,
            "rewrote_digests": self.rewrote_digests,
            "content_id": self.content_id,
            "changed": self.changed(),
            "notes": self.notes,
        })
    }
}

/// What a file needs, without touching it.
pub fn diagnose(path: &Path) -> Result<Diagnosis> {
    let sample = open_sample(path)?;
    let result = sample.verify(None)?;
    let indexable: std::collections::BTreeSet<String> = sample
        .annotations()?
        .iter()
        .filter(|(_, a)| a.is_voxel() && a.kind() != "mask")
        .map(|(n, _)| n.clone())
        .collect();
    let present: std::collections::BTreeSet<String> = sample.index()?.keys().cloned().collect();
    Ok(Diagnosis {
        path: path.to_string_lossy().into_owned(),
        mismatched: result.mismatched.clone(),
        undigested: result.undigested.clone(),
        unattested: result.unattested.clone(),
        stale_index: super::stale_index_entries(&sample.root)?,
        missing_index: indexable.difference(&present).cloned().collect(),
        content_id_ok: result.content_id_ok(),
    })
}

/// Options for [`fix`].
#[derive(Debug, Clone, Default)]
pub struct FixOptions {
    pub rebuild_index: bool,
    pub rewrite_digests: bool,
    pub reason: Option<String>,
    pub performed_by: Option<String>,
    pub max_coords: Option<usize>,
}

/// Repair one file.  Neither flag means diagnose and change nothing.
///
/// Only a *stale* index is rebuilt; `missing_index` is reported and never
/// acted on, because an index is optional per annotation (§14.3).
pub fn fix(path: &Path, options: &FixOptions) -> Result<Repair> {
    let diagnosis = diagnose(path)?;
    let mut repair =
        Repair { path: path.to_string_lossy().into_owned(), diagnosis: diagnosis.clone(), ..Default::default() };
    if !options.rebuild_index && !options.rewrite_digests {
        return Ok(repair);
    }
    if options.rewrite_digests && options.reason.as_deref().unwrap_or("").is_empty() {
        return Err(Error::invalid(
            "rewriting digests re-attests content this tool did not verify; pass a reason, which is recorded in the file's provenance",
        ));
    }
    if options.rebuild_index && !options.rewrite_digests && diagnosis.needs_digests() {
        let mut message = format!(
            "{}: cannot rebuild the index without restamping digests, and this file has {} that no longer match its bytes",
            path.to_string_lossy(),
            diagnosis.mismatched.len()
        );
        if !diagnosis.unattested.is_empty() {
            message.push_str(&format!(
                " and {} path(s) inside attested objects that no line of content_id covers",
                diagnosis.unattested.len()
            ));
        }
        if diagnosis.content_id_ok == Some(false) {
            message.push_str(" (and a stale content_id)");
        }
        message.push_str(
            ". Rebuilding would recompute them and the mismatch would vanish unrecorded. Restore the content, or pass \
             rewrite_digests with a reason so the re-attestation is written into the file.",
        );
        return Err(Error::invalid(message));
    }
    if diagnosis.stale_index.is_empty() && !options.rewrite_digests {
        repair.notes.push("nothing to rebuild: no sampling index is stale or missing".into());
        return Ok(repair);
    }
    let mut writer = amend(path, None)?;
    let result = (|| -> Result<()> {
        if options.rewrite_digests {
            // Restamped before the index is planned: an entry pins its
            // annotation's digest (§13.3), so one rebuilt from the digests the
            // file had went stale when the commit restamped them --- and one
            // that matched before the restamp went stale without being rebuilt
            // (U05 of the 2.0 audit).
            stamp_digests(&writer.root()?, "sha256", &["index"], false)?;
        }
        let mut names = stale_index_entries(&writer.root()?)?;
        names.sort();
        if options.rebuild_index && !names.is_empty() {
            repair.rebuilt_index = writer.build_index(Some(&names), options.max_coords, None, 0)?;
        } else if !names.is_empty() {
            repair.notes.push(format!(
                "{} sampling index entr{} no longer match their annotations ({}); rebuild them with rebuild_index",
                names.len(),
                if names.len() == 1 { "y" } else { "ies" },
                names.join(", ")
            ));
        }
        if options.rewrite_digests {
            let agent = match &options.performed_by {
                Some(who) => writer.person(who, None, Map::new())?,
                None => writer.software("medh5", Some(VERSION), Map::new())?,
            };
            let mut restamped: Vec<String> =
                diagnosis.mismatched.iter().chain(&diagnosis.unattested).cloned().collect();
            restamped.sort();
            restamped.dedup();
            let mut fields = Map::new();
            fields.insert("tool".into(), Value::String("medh5 fix --rewrite-digests".into()));
            fields.insert(
                "params".into(),
                json!({
                    "reason": options.reason,
                    "restamped": if restamped.is_empty() { json!("all") } else { json!(restamped) },
                    "verified_content": false,
                }),
            );
            writer.activity("other", Some(&agent.id), None, fields)?;
            repair.rewrote_digests = true;
            repair.notes.push(
                "digests were recomputed from the current bytes; this asserts nothing about whether those bytes are correct"
                    .into(),
            );
        }
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    repair.content_id = open_sample(path)?.content_id()?;
    Ok(repair)
}

/// Repair many files.
pub fn fix_paths(paths: &[&Path], options: &FixOptions) -> Result<Vec<Repair>> {
    paths.iter().map(|p| fix(p, options)).collect()
}

/// `repr()` helper kept for callers formatting a path the Python way.
pub fn shown(path: &Path) -> String {
    repr_str(&path.to_string_lossy())
}
