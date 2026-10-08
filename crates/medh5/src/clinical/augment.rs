//! Adding the clinical profile to a sample, and taking it away (1.1 §10).
//!
//! **Augmentation** is an explicit amend: the sample's images, grids,
//! annotations and transforms are copied as stored (their digests do not
//! change), the records are added, and the sample becomes 1.1 with a new
//! `content_id` --- the version is part of what that address covers.  What the
//! records leave unknown stays unknown, and the report says so.
//!
//! **Stripping** is the separately requested imaging projection: the clinical
//! records are removed, the loss is reported, and the result is a different
//! sample with its own `content_id`, never a lossless equivalent.

use std::path::Path;

use serde_json::{json, Value};

use super::model::{Bounds, ClinicalRecords, Clock, Event, Link, DAY, PROFILE};
use crate::json::repr_str;
use crate::sample::{amend, open_sample};
use crate::{Error, Result};

/// What an augmentation did, and what it could not know.
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentReport {
    pub path: String,
    pub version_before: String,
    pub version_after: String,
    pub content_id_before: Option<String>,
    pub content_id_after: Option<String>,
    pub events: usize,
    pub documents: usize,
    pub links: usize,
    /// Payload datasets whose digest the augmentation left unchanged.
    pub unchanged_digests: usize,
    /// What the records leave unknown or coarse: stated, never filled in.
    pub assumptions: Vec<String>,
}

impl AugmentReport {
    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "content_id_before": self.content_id_before,
            "content_id_after": self.content_id_after,
            "events": self.events,
            "documents": self.documents,
            "links": self.links,
            "unchanged_digests": self.unchanged_digests,
            "assumptions": self.assumptions,
        })
    }
}

/// What the records leave unknown or coarse (§10: missing data stays
/// unknown, and migration reports assumptions).
pub fn assumptions(records: &ClinicalRecords) -> Vec<String> {
    let mut out = Vec::new();
    let ids = |pred: &dyn Fn(&Event) -> bool| -> Vec<String> {
        records.events.iter().filter(|e| pred(e)).map(|e| e.event_id.clone()).collect()
    };
    let show = |v: &[String]| -> String {
        let head: Vec<&str> = v.iter().take(5).map(String::as_str).collect();
        if v.len() > 5 {
            format!("{} and {} more", head.join(", "), v.len() - 5)
        } else {
            head.join(", ")
        }
    };
    let unknown_availability = ids(&|e| e.available.is_none());
    if !unknown_availability.is_empty() {
        out.push(format!(
            "{} event(s) have unknown availability ({}): kept null, so strict prospective selection never uses them",
            unknown_availability.len(),
            show(&unknown_availability)
        ));
    }
    let coarse = ids(&|e| e.effective_start.is_some_and(|b| b.hi - b.lo >= DAY - 1));
    if !coarse.is_empty() {
        out.push(format!(
            "{} event(s) are known to a day or coarser ({}): their order within that span is not invented",
            coarse.len(),
            show(&coarse)
        ));
    }
    let untimed = ids(&|e| e.temporal_type == "unknown");
    if !untimed.is_empty() {
        out.push(format!(
            "{} event(s) have an unknown occurrence time ({}): they are not treated as static",
            untimed.len(),
            show(&untimed)
        ));
    }
    let open_intervals = ids(&|e| e.temporal_type == "interval" && e.effective_end.is_none());
    if !open_intervals.is_empty() {
        out.push(format!(
            "{} interval(s) have no known end ({}): an end learned later is a new event version",
            open_intervals.len(),
            show(&open_intervals)
        ));
    }
    out
}

/// One `imaging` event per image whose timepoint declares
/// `days_from_baseline`, linked to its image by `describes`.
///
/// The interval is all §3.7 says, so each event is known to its day ---
/// `[d·DAY, (d+1)·DAY − 1]` on a relative clock whose zero is the start of the
/// baseline visit's day --- its availability is unknown, and its status is
/// `unknown`.  Images whose timepoint declares no interval get no event, and
/// the returned notes say which.
pub fn imaging_events_from_timepoints(path: &Path) -> Result<(Vec<Event>, Vec<Link>, Vec<String>)> {
    let sample = open_sample(path)?;
    let timeline = sample.timepoints()?.clone();
    let mut events = Vec::new();
    let mut links = Vec::new();
    let mut notes = Vec::new();
    for (image_id, image) in sample.images()? {
        let Some(tp) = image.timepoint()? else {
            notes.push(format!("image {} has no timepoint; no imaging event was made", repr_str(image_id)));
            continue;
        };
        let Some(days) = timeline.iter().find(|t| t.id == tp).and_then(|t| t.days()) else {
            notes.push(format!(
                "timepoint {} declares no days_from_baseline; image {} got no imaging event",
                repr_str(&tp),
                repr_str(image_id)
            ));
            continue;
        };
        let day = days.floor() as i64;
        let event_id = format!("imaging.{image_id}");
        events.push(Event {
            event_id: event_id.clone(),
            record_id: event_id.clone(),
            kind: "imaging".into(),
            temporal_type: "point".into(),
            effective_start: Some(Bounds::new(day * DAY, (day + 1) * DAY - 1)),
            status: "unknown".into(),
            timepoint_id: Some(tp.clone()),
            ..Default::default()
        });
        links.push(Link::new(("event", &event_id), "describes", ("image", image_id)));
    }
    if !events.is_empty() {
        notes.push(format!(
            "{} imaging event(s) take their time from days_from_baseline, to the day, on a clock whose zero is the \
             start of the baseline visit's day; their availability is unknown",
            events.len()
        ));
    }
    Ok((events, links, notes))
}

/// The clock [`imaging_events_from_timepoints`] measures on.
pub fn baseline_day_clock(id: &str) -> Clock {
    Clock::relative(id, "00:00 of the day of the baseline timepoint (index 0), on the subject's own timeline")
}

fn payload_digests(path: &Path) -> Result<Vec<(String, String)>> {
    let sample = open_sample(path)?;
    let mut out: Vec<(String, String)> =
        crate::integrity::collect_digests(&sample.root, &["index", super::model::GROUP])?.into_iter().collect();
    out.sort();
    Ok(out)
}

/// Add `records` to the sample at `path`, in place or into `out`.
///
/// Refuses a sample this engine cannot amend, a `clinical` group that is not
/// the profile's, a different clock, and any id already present --- before
/// anything is written.
pub fn augment(path: &Path, records: ClinicalRecords, out: Option<&Path>) -> Result<AugmentReport> {
    let target = match out {
        Some(o) if o != path => {
            if o.exists() {
                return Err(Error::invalid(format!(
                    "{} exists; augmentation writes a new file and does not overwrite one",
                    repr_str(&o.to_string_lossy())
                )));
            }
            std::fs::copy(path, o)?;
            o.to_path_buf()
        }
        _ => path.to_path_buf(),
    };
    let result = (|| -> Result<AugmentReport> {
        let (version_before, content_id_before) = {
            let s = open_sample(&target)?;
            (s.version()?, s.content_id()?)
        };
        let before = payload_digests(&target)?;
        let notes = assumptions(&records);
        let (n_events, n_documents, n_links) = (records.events.len(), records.documents.len(), records.links.len());
        let mut writer = amend(&target, None)?;
        writer.add_records(records)?;
        let content_id_after = writer.commit(true)?;
        let after = payload_digests(&target)?;
        let unchanged = before.iter().filter(|d| after.contains(d)).count();
        let version_after = open_sample(&target)?.version()?;
        Ok(AugmentReport {
            path: target.to_string_lossy().into_owned(),
            version_before,
            version_after,
            content_id_before,
            content_id_after,
            events: n_events,
            documents: n_documents,
            links: n_links,
            unchanged_digests: unchanged,
            assumptions: notes,
        })
    })();
    if result.is_err() && target != path {
        let _ = std::fs::remove_file(&target);
    }
    result
}

/// What stripping the profile removed.
#[derive(Debug, Clone, PartialEq)]
pub struct StripReport {
    pub path: String,
    pub content_id_before: Option<String>,
    pub content_id_after: Option<String>,
    pub version_after: String,
    pub events_removed: usize,
    pub documents_removed: usize,
    pub links_removed: usize,
}

impl StripReport {
    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "content_id_before": self.content_id_before,
            "content_id_after": self.content_id_after,
            "version_after": self.version_after,
            "lost": {"events": self.events_removed, "documents": self.documents_removed, "links": self.links_removed},
        })
    }
}

/// Write the imaging projection of `path` to `out`: the sample without its
/// clinical records.  Information is lost, and the report counts it.
pub fn strip(path: &Path, out: &Path) -> Result<StripReport> {
    if out == path || out.exists() {
        return Err(Error::invalid(format!(
            "the imaging projection is a different sample; write it to a new path, not {}",
            repr_str(&out.to_string_lossy())
        )));
    }
    let (content_id_before, removed) = {
        let s = open_sample(path)?;
        let removed = match s.clinical()? {
            Some(c) => (c.events.len(), c.documents().len(), c.links.len()),
            None => {
                return Err(Error::invalid(format!(
                    "{} declares no `{PROFILE}` profile; there is nothing to strip",
                    repr_str(&path.to_string_lossy())
                )))
            }
        };
        (s.content_id()?, removed)
    };
    std::fs::copy(path, out)?;
    let result = (|| -> Result<StripReport> {
        let mut writer = amend(out, None)?;
        writer.drop_clinical()?;
        let content_id_after = writer.commit(true)?;
        Ok(StripReport {
            path: out.to_string_lossy().into_owned(),
            content_id_before,
            content_id_after,
            version_after: open_sample(out)?.version()?,
            events_removed: removed.0,
            documents_removed: removed.1,
            links_removed: removed.2,
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(out);
    }
    result
}
